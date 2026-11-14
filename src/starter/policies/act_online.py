"""Wave-based residual PPO on a frozen ACT policy and the existing RemoteEnv."""
from __future__ import annotations

import csv
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from starter.config import require
from starter.envs.parallel import parallel_plan
from starter.envs.remote import RemoteEnv
from starter.interfaces import STEP_LIMIT, SUCCESS_SCORE
from starter.paths import OUTPUTS_ROOT, REPO_ROOT, run_name, run_stamp
from starter.policies.act import _training_device, _validate_task, load_offline_policy
from starter.policies.act_inference import FEATURE_DIM, FrozenACTRunner, compose_residual_action
from starter.policies.residual_ppo import PPOBatch, ResidualActorCritic, compute_gae, ppo_update


def online_output_dir(config: dict) -> Path:
    return OUTPUTS_ROOT / "act_residual" / run_name(config) / run_stamp(config)


@torch.no_grad()
def collect_wave(env, runner, actor, seeds, *, gamma, gae_lambda, residual_scale):
    """Finish every scene before resetting; never fabricate inactive transitions.

    Features and the raw sampled residual are cached before the environment
    action is clipped. Time-limit values use the final observation, not reset.
    Public scores are unchanged; training rewards are successive score deltas.
    """
    observations, _ = env.reset_batch(seeds)
    if len(observations) != len(seeds):
        raise ValueError("Environment reset width does not match num_envs.")
    runner.reset()
    active = list(range(len(seeds)))
    chunks, features = runner.predict_base(observations, active, with_features=True)
    current = {i: (chunks[pos][0], features[pos]) for pos, i in enumerate(active)}
    trajectories = {i: [] for i in active}
    previous_scores = {i: 0.0 for i in active}
    peak_scores = dict(previous_scores)
    episodes = []
    for _ in range(STEP_LIMIT):
        inputs = torch.stack([current[i][1] for i in active])
        raw_actions, log_probs, values = actor.act(inputs)
        base = np.stack([current[i][0] for i in active])
        physical = compose_residual_action(base, raw_actions, runner.stats["action_std"], residual_scale)
        results = env.step_batch(physical.tolist(), active)
        if len(results) != len(active):
            raise ValueError("Environment step results must follow active scene order.")
        # Include truncated terminal frames for bootstrapping, exclude true terminals.
        bootstrap_ids = [i for i, result in zip(active, results) if not result[2]]
        next_current = {}
        next_values = {}
        if bootstrap_ids:
            next_obs = [result[0] for result in results if not result[2]]
            next_chunks, next_features = runner.predict_base(next_obs, bootstrap_ids, with_features=True)
            _, _, bootstrap_values = actor.act(next_features, deterministic=True)
            next_current = {i: (next_chunks[pos][0], next_features[pos])
                            for pos, i in enumerate(bootstrap_ids)}
            next_values = {i: bootstrap_values[pos] for pos, i in enumerate(bootstrap_ids)}
        remaining = []
        for pos, (i, result) in enumerate(zip(active, results)):
            _, score, terminated, truncated, _ = result
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("Environment returned a non-finite score.")
            if terminated and truncated:
                raise ValueError("An episode cannot be both terminated and truncated.")
            ended = bool(terminated or truncated)
            reward = score - previous_scores[i]
            previous_scores[i] = score
            peak_scores[i] = max(peak_scores[i], score)
            trajectories[i].append((inputs[pos].cpu(), raw_actions[pos].cpu(),
                                    log_probs[pos].cpu(), values[pos].cpu(), reward,
                                    0.0 if terminated else float(next_values[i]),
                                    bool(terminated), ended))
            if ended:
                episodes.append({"seed": int(seeds[i]), "length": len(trajectories[i]),
                                 "final_score": score, "peak_score": peak_scores[i],
                                 "success": bool(terminated and score == SUCCESS_SCORE),
                                 "terminated": bool(terminated), "truncated": bool(truncated)})
            else:
                remaining.append(i)
        active = remaining
        if not active:
            break
        current = {i: next_current[i] for i in active}
    if active:
        raise RuntimeError(f"Scenes {active} exceeded the task's {STEP_LIMIT}-step limit.")
    batches = []
    for trajectory in trajectories.values():
        feature, action, log_prob, value, reward, next_value, terminated, ended = zip(*trajectory)
        values = torch.stack(value)
        advantages, returns = compute_gae(
            torch.tensor(reward, dtype=torch.float32), values,
            torch.tensor(next_value, dtype=torch.float32),
            torch.tensor(terminated, dtype=torch.bool), torch.tensor(ended, dtype=torch.bool),
            gamma, gae_lambda,
        )
        batches.append(PPOBatch(torch.stack(feature), torch.stack(action), torch.stack(log_prob),
                                values, advantages, returns))
    return concatenate_batches(batches), episodes


def concatenate_batches(batches):
    return PPOBatch(*(torch.cat([getattr(batch, field) for batch in batches])
                      for field in PPOBatch.__dataclass_fields__))


def _validate_config(config):
    _validate_task(config)
    updates, episodes, width = parallel_plan(config)
    if require(config, "rollout_task") != "stack_blocks":
        raise ValueError("ACT online training requires rollout_task='stack_blocks'.")
    if not require(config, "temporal_agg"):
        raise ValueError("Residual PPO requires temporal_agg=true for one action per observation.")
    for key in ("gamma", "gae_lambda"):
        if not 0 < float(require(config, key)) <= 1:
            raise ValueError(f"{key} must lie in (0, 1].")
    for key in ("lr", "clip_range", "value_coef", "max_grad_norm", "target_kl", "residual_scale", "initial_std"):
        value = float(require(config, key))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    if not 0 <= float(require(config, "entropy_coef")) < math.inf:
        raise ValueError("entropy_coef must be nonnegative and finite.")
    if float(config["residual_scale"]) > 0.2:
        raise ValueError("residual_scale must not exceed 0.2 of the demonstration action std.")
    if not 0.02 <= float(config["initial_std"]) <= 0.5:
        raise ValueError("initial_std must lie in [0.02, 0.5].")
    for key in ("ppo_epochs", "minibatch_size", "hidden_dim", "save_freq"):
        if int(require(config, key)) < 1:
            raise ValueError(f"{key} must be positive.")
    return updates, episodes, width


def train_online(config: dict) -> None:
    updates, episodes_per_update, width = _validate_config(config)
    checkpoint = Path(require(config, "base_checkpoint")).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    base_config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    if base_config.get("policy_type") == "act_residual_ppo":
        raise ValueError("Choose an offline ACT checkpoint as the frozen base.")
    _validate_task(base_config)
    seed = int(require(config, "seed"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _training_device(config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load_offline_policy(checkpoint, device)
    with (checkpoint.parent / "dataset_stats.pkl").open("rb") as stream:
        stats = pickle.load(stream)
    runner = FrozenACTRunner(base, stats, device, temporal_agg=True, capture_features=True)
    frozen_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
    residual_config = {"feature_dim": FEATURE_DIM, "hidden_dim": int(config["hidden_dim"]),
                       "initial_std": float(config["initial_std"])}
    residual = ResidualActorCritic(**residual_config).to(device)
    optimizer = torch.optim.Adam(residual.parameters(), lr=float(config["lr"]))
    output = online_output_dir(config)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = dict(config, base_checkpoint=str(checkpoint), policy_type="act_residual_ppo",
                    reward_type="score_delta", residual_features="normalized_qpos+base_action+frozen_visual")
    (output / "config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    # A self-contained checkpoint can be evaluated without the original run directory.
    def save(path, update, steps):
        torch.save({
            "format": "act_residual_ppo_v1", "base_config": base_config,
            "base_state_dict": frozen_state, "residual_config": residual_config,
            "residual_state_dict": residual.state_dict(),
            "stats": {key: torch.as_tensor(runner.stats[key]) for key in runner.stats},
            "residual_scale": float(config["residual_scale"]), "temporal_agg": True,
            "optimizer": optimizer.state_dict(), "update": update, "global_step": steps,
            "torch_rng_state": torch.get_rng_state(),
        }, path)

    env = RemoteEnv(str(require(config, "env_host")), int(require(config, "env_port")))
    started = time.monotonic()
    global_step = episode_count = 0
    print(f"ACT residual PPO: frozen base={checkpoint}; output={output}", flush=True)
    try:
        with ((output / "metrics.jsonl").open("x", encoding="utf-8") as metrics,
              (output / "metrics.csv").open("x", encoding="utf-8", newline="") as table):
            writer = None
            for update in range(1, updates + 1):
                waves, episodes = [], []
                residual.eval()
                for _ in range(episodes_per_update // width):
                    seeds = [seed + episode_count + i for i in range(width)]
                    batch, wave_episodes = collect_wave(
                        env, runner, residual, seeds, gamma=float(config["gamma"]),
                        gae_lambda=float(config["gae_lambda"]), residual_scale=float(config["residual_scale"]),
                    )
                    waves.append(batch)
                    episodes.extend(wave_episodes)
                    episode_count += width
                batch = concatenate_batches(waves)
                global_step += len(batch.features)
                batch = PPOBatch(*(getattr(batch, field).to(device) for field in PPOBatch.__dataclass_fields__))
                residual.train()
                losses = ppo_update(
                    residual, optimizer, batch, update_epochs=int(config["ppo_epochs"]),
                    minibatch_size=int(config["minibatch_size"]), clip_range=float(config["clip_range"]),
                    value_coef=float(config["value_coef"]), entropy_coef=float(config["entropy_coef"]),
                    max_grad_norm=float(config["max_grad_norm"]), target_kl=float(config["target_kl"]),
                )
                for key, value in base.state_dict().items():
                    if not torch.equal(value.detach().cpu(), frozen_state[key]):
                        raise RuntimeError(f"Frozen ACT state changed: {key}.")
                record = {"update": update, "global_step": global_step, "episodes": episode_count,
                          "success_rate": float(np.mean([ep["success"] for ep in episodes])),
                          "mean_final_score": float(np.mean([ep["final_score"] for ep in episodes])),
                          "mean_episode_length": float(np.mean([ep["length"] for ep in episodes])),
                          "lr": optimizer.param_groups[0]["lr"],
                          "elapsed_seconds": time.monotonic() - started, **losses}
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                if writer is None:
                    writer = csv.DictWriter(table, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                table.flush()
                print(f"update {update}/{updates}: steps={global_step} success={record['success_rate']:.1%} "
                      f"score={record['mean_final_score']:.2f} policy_loss={record['policy_loss']:.5f} "
                      f"value_loss={record['value_loss']:.5f}", flush=True)
                if update % int(config["save_freq"]) == 0:
                    save(output / f"policy_update_{update}_seed_{seed}.ckpt", update, global_step)
                if update == 1 or update % int(config["save_freq"]) == 0 or update == updates:
                    from starter.policies.act_online_metrics import plot_online_training

                    plot_online_training(output)
            save(output / "policy_last.ckpt", updates, global_step)
    finally:
        env.close()
    print(f"Saved ACT residual PPO checkpoint: {output / 'policy_last.ckpt'}", flush=True)
