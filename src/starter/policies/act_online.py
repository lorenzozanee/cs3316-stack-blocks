"""Wave-based residual PPO on a frozen ACT policy and the existing RemoteEnv.

The frozen-base residual composition and state+base-action conditioning follow
ResiP, "From Imitation to Refinement -- Residual RL for Precise Assembly":
https://arxiv.org/abs/2407.16677
https://github.com/ankile/robust-rearrangement/blob/cf80d12d4acae9e4751a49407a96f85f07b342e3/src/train/residual_ppo.py
The ACT base follows Zhao et al. (arXiv:2304.13705); PPO/GAE follow Schulman
et al. (arXiv:1707.06347, arXiv:1506.02438). RoboDojo supplies task scores.
The layout cycle and RGB+state bridge are project-specific task integration;
ResiP itself uses sparse rewards, not the potential below.
"""
from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from starter.config import require
from starter.envs.parallel import parallel_plan
from starter.envs.remote import RemoteEnv
from starter.interfaces import ACTION_DIM, BLOCK_LABELS, PARTIAL_SCORE, STEP_LIMIT, SUCCESS_SCORE
from starter.paths import OUTPUTS_ROOT, REPO_ROOT, run_name, run_stamp
from starter.policies.act import _training_device, _validate_task, _vram_stats, load_offline_policy
from starter.policies.act_inference import (
    FEATURE_DIM,
    ONLINE_FORMAT,
    FrozenACTRunner,
    compose_residual_action,
)
from starter.policies.act_model import ACTPolicy
from starter.policies.post_grasp_prefix import (
    capture_frame, check_frame, check_grasp, load_prefixes, reward_digest)
from starter.policies.residual_ppo import PPOBatch, ResidualActorCritic, compute_gae, ppo_update
from starter.policies.stacking_reward import REWARD_KEYS, StackingReward, validate_reward_config


_EPISODE_CSV_FIELDS = (
    "episode", "absolute_update", "seed", "layout_id", "length", "prefix_steps",
    "final_score", "peak_score", "episode_return", "first_partial_step",
    "arrivals", "grasps", "carries", "aligned", "opened", "settled",
)


def online_output_dir(config: dict) -> Path:
    return OUTPUTS_ROOT / "act_residual" / run_name(config) / run_stamp(config)


def _budget_exhausted(started: float, max_seconds: float) -> bool:
    return time.monotonic() - started >= max_seconds


def _remaining_seconds(elapsed: float, local_update: int, updates: int,
                       max_seconds: float) -> float:
    budget_left = max(0.0, max_seconds - elapsed)
    if updates == sys.maxsize:
        return budget_left
    return min(budget_left, elapsed / local_update * (updates - local_update))


def _numpy_rng_payload() -> dict:
    """Serialize numpy's RNG as weights_only-safe plain values and a tensor."""
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": str(name),
        "keys": torch.from_numpy(np.ascontiguousarray(keys, dtype=np.uint32)),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng(payload: dict) -> tuple:
    keys = payload["keys"]
    if isinstance(keys, torch.Tensor):
        keys = keys.detach().cpu().numpy()
    keys = np.asarray(keys, dtype=np.uint32)
    return (str(payload["name"]), keys, int(payload["position"]),
            int(payload["has_gauss"]), float(payload["cached_gaussian"]))


def _validate_online_payload(payload, device) -> None:
    if not isinstance(payload, dict) or payload.get("format") != ONLINE_FORMAT:
        raise ValueError(f"Unsupported ACT residual checkpoint format; expected {ONLINE_FORMAT!r}.")
    if payload.get("temporal_agg") is not True:
        raise ValueError("ACT residual checkpoints require temporal_agg=True.")
    required = ("base_config", "base_state_dict", "residual_config", "residual_state_dict",
                "stats", "residual_scale", "optimizer", "update", "global_step",
                "torch_rng_state", "numpy_rng_state", "layout_cursor")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"ACT residual checkpoint cannot be resumed without {missing}.")
    if device.type == "cuda" and not isinstance(payload.get("torch_cuda_rng_state"), torch.Tensor):
        raise ValueError("CUDA resume requires a saved torch_cuda_rng_state.")


def _block_positions(privileged) -> dict:
    """Read block positions in the same per-scene frame as arm proprio."""
    try:
        return {label: np.asarray(privileged["blocks"][label]["pos"], dtype=np.float64)
                for label in BLOCK_LABELS}
    except (KeyError, TypeError) as error:
        raise ValueError("Dense shaping requires privileged block poses.") from error


@torch.no_grad()
def collect_wave(env, runner, actor, seeds, *, gamma, gae_lambda, residual_scale,
                 reward_cfg, prefixes=None, capture_prefix=None):
    """Finish every scene before resetting; never fabricate inactive transitions.

    Features and the raw sampled residual are cached before the environment
    action is clipped. Time-limit values use the final observation, not reset.
    Training combines public score deltas, physical-state PBRS, ordered
    approach and carry waypoints, and dwell-gated grasp and placement bonuses.
    Milestones intentionally add a target-region preference beyond the official
    task; their hidden episode history is not part of actor/critic inputs.
    Object poses condition only the residual actor/critic, never frozen ACT.
    Replayed actions warm ACT and reward history, but never become PPO samples.
    """
    prefixes = prefixes or {}
    if any(not 0 <= i < len(seeds) or entry["scene"] != i or entry["width"] != len(seeds)
           or entry["layout_id"] != seeds[i] for i, entry in prefixes.items()):
        raise ValueError("Post-grasp replay must match layout and scene slot.")
    observations, _ = env.reset_batch(seeds)
    if len(observations) != len(seeds):
        raise ValueError("Environment reset width does not match num_envs.")
    reset_privileged = getattr(env, "_privileged", None)
    if reset_privileged is None or len(reset_privileged) != len(seeds):
        raise ValueError("Dense shaping requires privileged states in reset scene order.")
    initial_heights = {}
    initial_terms = {}
    initial_grasps = {}
    potentials = {}
    rewards = {}
    recordings = {}
    for scene, (observation, privileged) in enumerate(zip(observations, reset_privileged)):
        if scene in prefixes:
            check_frame(prefixes[scene], 0, observation, privileged, 0.0)
        elif capture_prefix:
            recordings[scene] = {"actions": [], "frames": [capture_frame(observation, privileged)]}
        positions = _block_positions(privileged)
        initial_heights[scene] = {label: float(position[2])
                                  for label, position in positions.items()}
        rewards[scene] = StackingReward(privileged, observation["proprio"], reward_cfg)
        terms = rewards[scene].initial_terms
        initial_terms[scene] = terms
        potentials[scene] = terms["phi"]
    runner.reset()
    active = list(range(len(seeds)))
    residual_obs = [dict(obs, blocks=state["blocks"])
                    for obs, state in zip(observations, reset_privileged)]
    chunks, features = runner.predict_base(residual_obs, active, with_features=True)
    current = {i: (chunks[pos][0], features[pos]) for pos, i in enumerate(active)}
    prefix_steps = {i: len(prefixes[i]["actions"]) if i in prefixes else 0 for i in active}
    trajectories = {i: [] for i in active}
    previous_scores = {i: 0.0 for i in active}
    peak_scores = dict(previous_scores)
    episode_returns = dict(previous_scores)
    shaping_returns = dict(previous_scores)
    discounted_returns = dict(previous_scores)
    nonzero_rewards = dict.fromkeys(active, 0)
    negative_rewards = dict.fromkeys(active, 0)
    first_partial_steps = dict.fromkeys(active)
    position_residual_sums = dict(previous_scores)
    phi_sums = dict(previous_scores)
    # Per-episode extrema (observability only). Final-frame fields cannot show a
    # transient lift/grasp/release, so coverage conclusions rested on a blind
    # spot. These accumulate over every step and change no action, reward,
    # potential or trajectory; they only add keys to episodes/metrics.
    extrema = {
        i: {"max_lift_m": 0.0, "max_lift_progress": 0.0, "max_pair_sum": 0.0,
            "max_manipulation_progress": initial_terms[i]["manipulation_progress"],
            "min_d_min_m": float("inf"), "max_min_gripper": 0.0,
            "max_settled_pair": 0.0, "max_lift_gate": 0.0,
            "max_lift_per_block_m": {label: 0.0 for label in BLOCK_LABELS},
            "first_lift_step": None, "first_pair_step": None,
            "first_both_open_step": None, "first_settled_step": None}
        for i in active
    }
    episodes = []
    for control_step in range(1, STEP_LIMIT + 1):
        learning = [i for i in active if control_step > prefix_steps[i]]
        sample_positions = {i: pos for pos, i in enumerate(learning)}
        physical = {}
        if learning:
            inputs = torch.stack([current[i][1] for i in learning])
            raw_actions, log_probs, values = actor.act(inputs)
            inputs, raw_actions, log_probs, values = (item.cpu() for item in (inputs, raw_actions, log_probs, values))
            base = np.stack([current[i][0] for i in learning])
            actions = compose_residual_action(base, raw_actions, runner.stats["action_std"], residual_scale)
            # Mean Euclidean target-position correction across both arms, in metres.
            position_residuals = np.linalg.norm(
                (actions - base)[:, [0, 1, 2, 8, 9, 10]].reshape(-1, 2, 3), axis=-1,
            ).mean(axis=-1)
            physical.update(zip(learning, actions.tolist()))
        for i in active:
            if i not in physical:
                physical[i] = prefixes[i]["actions"][control_step - 1]
        results = env.step_batch([physical[i] for i in active], active)
        if len(results) != len(active):
            raise ValueError("Environment step results must follow active scene order.")
        # Include truncated terminal frames for bootstrapping, exclude true terminals.
        bootstrap_ids = [i for i, result in zip(active, results) if not result[2]]
        next_current = {}
        next_values = {}
        privileged_after = env._privileged
        if len(privileged_after) != len(active):
            raise ValueError("Environment privileged states must follow active scene order.")
        if bootstrap_ids:
            next_obs = [dict(result[0], blocks=privileged_after[pos]["blocks"])
                        for pos, result in enumerate(results) if not result[2]]
            next_chunks, next_features = runner.predict_base(next_obs, bootstrap_ids, with_features=True)
            _, _, bootstrap_values = actor.act(next_features, deterministic=True)
            next_current = {i: (next_chunks[pos][0], next_features[pos])
                            for pos, i in enumerate(bootstrap_ids)}
            next_values = dict(zip(bootstrap_ids, bootstrap_values.cpu().tolist()))
        remaining = []
        for pos, (i, result) in enumerate(zip(active, results)):
            observation, score, terminated, truncated, _ = result
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("Environment returned a non-finite score.")
            if terminated and truncated:
                raise ValueError("An episode cannot be both terminated and truncated.")
            ended = bool(terminated or truncated)
            step_positions = _block_positions(privileged_after[pos])
            terms, bonus = rewards[i].step(observation["proprio"], privileged_after[pos])
            next_potential = terms["phi"]
            event_bonus = terms.get("event_bonus", bonus)
            if i not in sample_positions:
                if score != 0 or ended:
                    raise RuntimeError(f"Post-grasp prefix crossed a task score or episode end in layout {seeds[i]}.")
                check_frame(prefixes[i], control_step, observation, privileged_after[pos], event_bonus)
                potentials[i] = next_potential
                if control_step == prefix_steps[i]:
                    check_grasp(prefixes[i], rewards[i])
                    initial_terms[i] = terms
                    initial_grasps[i] = {key: dict(counts)
                                         for key, counts in rewards[i].grasp_attempts.items()}
                    extrema[i]["max_manipulation_progress"] = terms["manipulation_progress"]
                remaining.append(i)
                continue
            if i in recordings:
                if score != 0 or ended:
                    del recordings[i]
                else:
                    recording = recordings[i]
                    recording["actions"].append(physical[i])
                    recording["frames"].append(capture_frame(observation, privileged_after[pos], event_bonus))
                    for grasp in rewards[i].transport_attempts:
                        if grasp["grasp_step"] == control_step:
                            capture_prefix(dict(recordings.pop(i), layout_id=int(seeds[i]), scene=i,
                                                width=len(seeds), arm=grasp["arm"], carried=grasp["carried"],
                                                target_lower=grasp["target_lower"],
                                                reward_digest=reward_digest(reward_cfg)))
                            break
            sample = sample_positions[i]
            # A true terminal is absorbing with zero potential. Time-limit
            # truncations keep phi(next), matching their retained value bootstrap.
            shaping = gamma * (0.0 if terminated else next_potential) - potentials[i]
            potentials[i] = next_potential
            phi_sums[i] += next_potential
            reward = score - previous_scores[i] + shaping + bonus
            step = len(trajectories[i]) + 1
            episode_returns[i] += reward
            shaping_returns[i] += shaping
            discounted_returns[i] += gamma ** (step - 1) * reward
            nonzero_rewards[i] += int(reward != 0)
            negative_rewards[i] += int(reward < 0)
            position_residual_sums[i] += float(position_residuals[sample])
            if score >= PARTIAL_SCORE and first_partial_steps[i] is None:
                first_partial_steps[i] = step
            previous_scores[i] = score
            peak_scores[i] = max(peak_scores[i], score)
            ex = extrema[i]
            ex["max_lift_m"] = max(ex["max_lift_m"], terms["lift"])
            ex["max_lift_progress"] = max(ex["max_lift_progress"], terms["lift_progress"])
            ex["max_manipulation_progress"] = max(
                ex["max_manipulation_progress"], terms["manipulation_progress"])
            ex["max_pair_sum"] = max(ex["max_pair_sum"], terms["pair_sum"])
            ex["min_d_min_m"] = min(ex["min_d_min_m"], terms["d_min"])
            ex["max_min_gripper"] = max(ex["max_min_gripper"], terms["min_gripper"])
            ex["max_settled_pair"] = max(ex["max_settled_pair"], terms["settled_pair"])
            ex["max_lift_gate"] = max(ex["max_lift_gate"], terms["lift_gate"])
            for label in BLOCK_LABELS:
                block_lift = max(0.0, float(step_positions[label][2] - initial_heights[i][label]))
                if block_lift > ex["max_lift_per_block_m"][label]:
                    ex["max_lift_per_block_m"][label] = block_lift
            if ex["first_lift_step"] is None and terms["lift"] > 1e-6:
                ex["first_lift_step"] = step
            if ex["first_pair_step"] is None and terms["pair_sum"] >= 1.0:
                ex["first_pair_step"] = step
            if ex["first_both_open_step"] is None and terms["min_gripper"] >= 1.0:
                ex["first_both_open_step"] = step
            if ex["first_settled_step"] is None and terms["settled_pair"] > 0.0:
                ex["first_settled_step"] = step
            trajectories[i].append((inputs[sample], raw_actions[sample],
                                    log_probs[sample], values[sample], reward,
                                    0.0 if terminated else next_values[i],
                                    bool(terminated), ended))
            if ended:
                # Diagnostics only: measured final-state potential (terminal
                # shaping still uses zero), per-component terms, block-pair
                # geometry in block_0/1/2 label order and gripper openness.
                terms0 = initial_terms[i]
                pair_xy, pair_dz = [], []
                labels = list(BLOCK_LABELS)
                for a in range(len(labels)):
                    for b in range(a + 1, len(labels)):
                        pa, pb = step_positions[labels[a]], step_positions[labels[b]]
                        pair_xy.append(float(np.linalg.norm(pa[:2] - pb[:2])))
                        pair_dz.append(float(pb[2] - pa[2]))
                proprio_final = np.asarray(observation["proprio"], dtype=np.float64)
                episodes.append({"layout_id": int(seeds[i]), "length": len(trajectories[i]),
                                 "prefix_steps": prefix_steps[i],
                                 "final_score": score, "peak_score": peak_scores[i],
                                 "success": bool(terminated and score == SUCCESS_SCORE),
                                 "terminated": bool(terminated), "truncated": bool(truncated),
                                 "episode_return": episode_returns[i],
                                 "shaping_return": shaping_returns[i],
                                 "milestone_return": terms["milestone_return"] - terms0["milestone_return"],
                                 "milestone_steps": dict(rewards[i].milestone_steps),
                                 **({"dense_approach_return": terms["dense_approach_return"]
                                     - terms0["dense_approach_return"],
                                     "dense_transport_return": terms["dense_transport_return"]
                                     - terms0["dense_transport_return"],
                                     "release_steps": dict(rewards[i].release_steps)}
                                    if "dense_transport_return" in terms else {}),
                                 "grasp_steps_by_block": dict(rewards[i].grasp_steps_by_block),
                                 "grasp_attempts": rewards[i].grasp_attempts,
                                 "grasp_attempts_prefix": initial_grasps.get(i, {}),
                                 "transport_attempts": rewards[i].transport_attempts,
                                 "waypoint_return": terms["waypoint_return"] - terms0["waypoint_return"],
                                 "waypoint_counts": {key: count - terms0["waypoint_counts"][key]
                                                     for key, count in terms["waypoint_counts"].items()},
                                 "phi_base_final": terms["phi_base"],
                                 "base_score_final": terms["base_score"],
                                 "stable_tower_levels_final": terms["stable_tower_levels"],
                                 "target_xy": terms["target_xy"],
                                 "support_gaps_final_m": terms["support_gaps"],
                                 "discounted_return": discounted_returns[i],
                                 "nonzero_reward_steps": nonzero_rewards[i],
                                 "negative_reward_steps": negative_rewards[i],
                                 "first_partial_step": first_partial_steps[i],
                                 "position_residual_mean_m": position_residual_sums[i] / step,
                                 "phi_initial": terms0["phi"],
                                 "phi_final": terms["phi"],
                                 "phi_mean": phi_sums[i] / step,
                                 "phi_reach_final": terms["phi_reach"],
                                 "phi_manipulation_final": terms["phi_manipulation"],
                                 "max_manipulation_progress": ex["max_manipulation_progress"],
                                 "phi_pair_final": terms["phi_pair"],
                                 "phi_release_final": terms["phi_release"],
                                 "pair_sum_final": terms["pair_sum"],
                                 "d_min_initial_m": terms0["d_min"],
                                 "d_min_final_m": terms["d_min"],
                                 "lift_initial_m": terms0["lift"],
                                 "lift_final_m": terms["lift"],
                                 "lift_gate_final": terms["lift_gate"],
                                 "min_gripper_final": terms["min_gripper"],
                                 "settled_pair_final": terms["settled_pair"],
                                 "lift_progress_final": terms["lift_progress"],
                                 "final_gripper_open_measured": privileged_after[pos]["gripper_open_measured"],
                                 "final_pair_xy_m": pair_xy,
                                 "final_pair_dz_m": pair_dz,
                                 "final_gripper_openness": [float(proprio_final[7]),
                                                            float(proprio_final[15])],
                                 "max_lift_m": extrema[i]["max_lift_m"],
                                 "max_lift_progress": extrema[i]["max_lift_progress"],
                                 "max_lift_per_block_m": extrema[i]["max_lift_per_block_m"],
                                 "max_pair_sum": extrema[i]["max_pair_sum"],
                                 "min_d_min_m": extrema[i]["min_d_min_m"],
                                 "max_min_gripper": extrema[i]["max_min_gripper"],
                                 "max_settled_pair": extrema[i]["max_settled_pair"],
                                 "max_lift_gate": extrema[i]["max_lift_gate"],
                                 "first_lift_step": extrema[i]["first_lift_step"],
                                 "first_pair_step": extrema[i]["first_pair_step"],
                                 "first_both_open_step": extrema[i]["first_both_open_step"],
                                 "first_settled_step": extrema[i]["first_settled_step"]})
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


def _suffix_grasp_attempts(episode: dict) -> list[dict]:
    prefix = episode["grasp_attempts_prefix"]
    return [{name: value - prefix.get(key, {}).get(name, 0) for name, value in counts.items()}
            for key, counts in episode["grasp_attempts"].items()]


def _rollout_metrics(batch: PPOBatch, episodes: list[dict], residual_scale: float) -> dict:
    """Summarize genuine rollout samples before PPO normalizes advantages."""
    actions = batch.actions
    attempts = [counts for episode in episodes for counts in _suffix_grasp_attempts(episode)]
    arrivals = sum(counts["arrivals"] for counts in attempts)
    confirmed = sum(counts["confirmed"] for counts in attempts)
    return {
        "grasp_arrivals": arrivals,
        "grasp_confirmed": confirmed,
        "grasp_conversion_rate": confirmed / arrivals if arrivals else None,
        "mean_steps_from_arrival_to_grasp": (
            sum(counts["steps_to_grasp"] for counts in attempts) / confirmed
            if confirmed else None),
        "rollout_steps": len(batch.features),
        "replayed_prefix_steps": sum(ep.get("prefix_steps", 0) for ep in episodes),
        "episodes_this_update": len(episodes),
        "success_count": sum(ep["success"] for ep in episodes),
        "success_rate": float(np.mean([ep["success"] for ep in episodes])),
        "partial_rate": float(np.mean([ep["peak_score"] >= PARTIAL_SCORE for ep in episodes])),
        "truncated_count": sum(ep["truncated"] for ep in episodes),
        "terminated_count": sum(ep["terminated"] for ep in episodes),
        "mean_final_score": float(np.mean([ep["final_score"] for ep in episodes])),
        "mean_peak_score": float(np.mean([ep["peak_score"] for ep in episodes])),
        "mean_episode_length": float(np.mean([ep["length"] for ep in episodes])),
        "mean_episode_return": float(np.mean([ep["episode_return"] for ep in episodes])),
        "mean_shaping_return": float(np.mean([ep["shaping_return"] for ep in episodes])),
        "mean_milestone_return": float(np.mean([ep["milestone_return"] for ep in episodes])),
        "mean_phi_base": float(np.mean([ep["phi_base_final"] for ep in episodes])),
        "mean_base_score": float(np.mean([ep["base_score_final"] for ep in episodes])),
        "mean_stable_tower_levels": float(np.mean([ep["stable_tower_levels_final"] for ep in episodes])),
        **{f"episodes_with_milestone_{name}": sum((ep["milestone_steps"][name] or 0) > ep.get("prefix_steps", 0)
                                                for ep in episodes)
           for name in ("grasp", "base", "pair2", "pair3")},
        "mean_discounted_return": float(np.mean([ep["discounted_return"] for ep in episodes])),
        "mean_phi_final": float(np.mean([ep["phi_final"] for ep in episodes])),
        "mean_phi_reach": float(np.mean([ep["phi_reach_final"] for ep in episodes])),
        "mean_phi_manipulation": float(np.mean([ep["phi_manipulation_final"] for ep in episodes])),
        "mean_max_manipulation_progress": float(np.mean([ep["max_manipulation_progress"] for ep in episodes])),
        "mean_phi_pair": float(np.mean([ep["phi_pair_final"] for ep in episodes])),
        "mean_phi_release": float(np.mean([ep["phi_release_final"] for ep in episodes])),
        "mean_pair_sum": float(np.mean([ep["pair_sum_final"] for ep in episodes])),
        "mean_d_min_m": float(np.mean([ep["d_min_final_m"] for ep in episodes])),
        "fraction_episodes_with_pair": float(np.mean([ep["pair_sum_final"] >= 1.0 for ep in episodes])),
        "fraction_episodes_with_lift": float(np.mean([ep["lift_gate_final"] >= 1.0 for ep in episodes])),
        # Per-episode extrema aggregates (observability): transient lift/grasp/
        # release events the final-frame fields miss. *_over_episodes is the
        # best single episode; episodes_with_* counts any transient occurrence.
        "max_lift_m_over_episodes": float(max(ep["max_lift_m"] for ep in episodes)),
        "max_lift_progress_over_episodes": float(max(ep["max_lift_progress"] for ep in episodes)),
        "max_pair_sum_over_episodes": float(max(ep["max_pair_sum"] for ep in episodes)),
        "min_d_min_m_over_episodes": float(min(ep["min_d_min_m"] for ep in episodes)),
        "max_min_gripper_over_episodes": float(max(ep["max_min_gripper"] for ep in episodes)),
        "max_settled_pair_over_episodes": float(max(ep["max_settled_pair"] for ep in episodes)),
        "mean_max_lift_m": float(np.mean([ep["max_lift_m"] for ep in episodes])),
        "mean_max_pair_sum": float(np.mean([ep["max_pair_sum"] for ep in episodes])),
        "episodes_with_any_lift": sum(ep["first_lift_step"] is not None for ep in episodes),
        "episodes_with_any_pair": sum(ep["first_pair_step"] is not None for ep in episodes),
        "episodes_with_both_open": sum(ep["first_both_open_step"] is not None for ep in episodes),
        "episodes_with_any_settled": sum(ep["first_settled_step"] is not None for ep in episodes),
        "mean_final_min_pair_xy_m": float(np.mean([min(ep["final_pair_xy_m"]) for ep in episodes])),
        "nonzero_reward_steps": sum(ep["nonzero_reward_steps"] for ep in episodes),
        "negative_reward_steps": sum(ep["negative_reward_steps"] for ep in episodes),
        "position_residual_mean_m": sum(ep["position_residual_mean_m"] * ep["length"]
                                        for ep in episodes) / len(batch.features),
        "raw_residual_abs_mean": float(actions.abs().mean()),
        "raw_residual_rms": float(actions.square().mean().sqrt()),
        "residual_clip_fraction": float((actions.abs() > 1).float().mean()),
        "normalized_residual_abs_mean": float(actions.clamp(-1, 1).abs().mean()) * residual_scale,
        "rollout_value_mean": float(batch.values.mean()),
        "rollout_value_std": float(batch.values.std(unbiased=False)),
        "return_mean": float(batch.returns.mean()),
        "return_std": float(batch.returns.std(unbiased=False)),
        "advantage_mean": float(batch.advantages.mean()),
        "advantage_std": float(batch.advantages.std(unbiased=False)),
    }


@torch.no_grad()
def _mean_policy_metrics(actor, features, action_std, residual_scale):
    """Measure the learned mean separately from exploration on rollout states."""
    position_norm_sum = raw_abs_sum = 0.0
    device = next(actor.parameters()).device
    std = torch.as_tensor(action_std, dtype=torch.float32, device=device)
    for inputs in features.split(1024):
        mean, _, _ = actor.act(inputs.to(device), deterministic=True)
        correction = mean.clamp(-1, 1) * residual_scale * std
        position = correction[:, [0, 1, 2, 8, 9, 10]].reshape(-1, 2, 3)
        position_norm_sum += float(position.norm(dim=-1).sum())
        raw_abs_sum += float(mean.abs().sum())
    return {
        "mean_policy_position_residual_m": position_norm_sum / (2 * len(features)),
        "mean_policy_raw_abs_mean": raw_abs_sum / (ACTION_DIM * len(features)),
    }


def _reward_config(config):
    return {**{key: require(config, key) for key in REWARD_KEYS},
            "transport_to_support": config.get("transport_to_support", False),
            **({"dense_transport_reward": config["dense_transport_reward"]}
               if config.get("dense_transport_reward", False) is not False else {})}


def _validate_config(config):
    _validate_task(config)
    updates, episodes, width = parallel_plan(config)
    max_seconds = float(require(config, "max_train_seconds"))
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_train_seconds must be positive and finite.")
    if require(config, "rollout_task") != "stack_blocks":
        raise ValueError("ACT online training requires rollout_task='stack_blocks'.")
    if not require(config, "temporal_agg"):
        raise ValueError("Residual PPO requires temporal_agg=true for one action per observation.")
    for key in ("gamma", "gae_lambda"):
        if not 0 < float(require(config, key)) <= 1:
            raise ValueError(f"{key} must lie in (0, 1].")
    for key in ("lr", "critic_lr", "clip_range", "value_coef", "max_grad_norm", "target_kl", "residual_scale", "initial_std"):
        value = float(require(config, key))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    validate_reward_config(_reward_config(config))
    if "post_grasp_prefix_file" in config and (not isinstance(config["post_grasp_prefix_file"], str)
                                               or not config["post_grasp_prefix_file"].strip()):
        raise ValueError("post_grasp_prefix_file must be a path to a nonempty replay archive.")
    if not isinstance(require(config, "learn_std"), bool):
        raise ValueError("learn_std must be a boolean.")
    if not 0 <= float(require(config, "entropy_coef")) < math.inf:
        raise ValueError("entropy_coef must be nonnegative and finite.")
    if float(config["residual_scale"]) > 0.5:
        raise ValueError("residual_scale must not exceed 0.5 of the demonstration action std.")
    if not 0.02 <= float(config["initial_std"]) <= 0.5:
        raise ValueError("initial_std must lie in [0.02, 0.5].")
    for key in ("ppo_epochs", "minibatch_size", "hidden_dim", "save_freq"):
        if int(require(config, key)) < 1:
            raise ValueError(f"{key} must be positive.")
    layout_ids = require(config, "rollout_layout_ids")
    if (not isinstance(layout_ids, list) or not layout_ids
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in layout_ids)
            or len(set(layout_ids)) != len(layout_ids)):
        raise ValueError("rollout_layout_ids must be a nonempty list of unique nonnegative integers.")
    return updates, episodes, width


def train_online(config: dict) -> Path:
    updates, episodes_per_update, width = _validate_config(config)
    max_train_seconds = float(require(config, "max_train_seconds"))
    checkpoint = Path(require(config, "base_checkpoint")).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    checkpoint_config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    policy_type = checkpoint_config.get("policy_type")
    if policy_type not in (None, "act_residual_ppo"):
        raise ValueError(f"Unsupported ACT policy_type {policy_type!r}.")
    resume = policy_type == "act_residual_ppo"
    seed = int(require(config, "seed"))
    layout_ids = list(require(config, "rollout_layout_ids"))
    reward_cfg = _reward_config(config)
    inference_batch_size = config.get("act_inference_batch_size")
    replay_cycle = math.lcm(len(layout_ids), width)
    prefix_archive = {}
    if "post_grasp_prefix_file" in config:
        archive = Path(config["post_grasp_prefix_file"]).expanduser()
        if not archive.is_absolute():
            archive = REPO_ROOT / archive
        prefix_archive = load_prefixes(archive, width, layout_ids, reward_cfg)
        slots = {(layout_ids[i % len(layout_ids)], i % width) for i in range(replay_cycle)}
        if not prefix_archive.keys() <= slots:
            raise ValueError("Replay layout/scene does not match the rollout schedule.")
    learning_rate = float(require(config, "lr"))
    critic_learning_rate = float(require(config, "critic_lr"))

    def make_optimizer(model):
        # ResiP uses separate actor/critic networks and learning rates. Keep one
        # optimizer payload while preserving disjoint parameter groups.
        return torch.optim.Adam([
            {"params": model.actor_parameters(), "lr": learning_rate},
            {"params": model.critic_parameters(), "lr": critic_learning_rate},
        ])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _training_device(config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if resume:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _validate_online_payload(payload, device)
        base_config = payload["base_config"]
        _validate_task(base_config)
        base = ACTPolicy(base_config, pretrained_backbone=False)
        base.load_state_dict(payload["base_state_dict"], strict=True)
        frozen_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
        runner = FrozenACTRunner(base, payload["stats"], device, capture_features=True,
                                 inference_batch_size=inference_batch_size)
        residual_config = dict(payload["residual_config"])
        if int(residual_config.get("feature_dim", -1)) != FEATURE_DIM:
            raise ValueError(f"Residual feature_dim must be {FEATURE_DIM}.")
        residual = ResidualActorCritic(**residual_config).to(device)
        residual.load_state_dict(payload["residual_state_dict"], strict=True)
        if not bool(residual_config.get("learn_std", False)):
            # A new round may request a different fixed exploration std, like
            # the lr adaptation below. ResiP's MLP residual uses init_logstd
            # -1.5 (src/config/actor @cf80d12d); the frozen std carries no
            # optimizer moments, so learned weights and progress continue.
            # With learn_std=true the learned log_std and its moments resume.
            residual_config["initial_std"] = float(config["initial_std"])
            with torch.no_grad():
                residual.log_std.fill_(math.log(residual_config["initial_std"]))
        optimizer = make_optimizer(residual)
        optimizer.load_state_dict(payload["optimizer"])
        # A new round may request a different lr; learned state and optimizer
        # moments continue, only the param-group lr is adapted.
        for group, rate in zip(optimizer.param_groups, (learning_rate, critic_learning_rate)):
            group["lr"] = rate
        residual_scale = float(payload["residual_scale"])
        start_update = int(payload["update"])
        global_step = int(payload["global_step"])
        layout_cursor = int(payload["layout_cursor"])
        torch.set_rng_state(payload["torch_rng_state"])
        np.random.set_state(_restore_numpy_rng(payload["numpy_rng_state"]))
        if device.type == "cuda":
            torch.cuda.set_rng_state(payload["torch_cuda_rng_state"], device)
    else:
        base_config = checkpoint_config
        _validate_task(base_config)
        base = load_offline_policy(checkpoint, device)
        frozen_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
        with (checkpoint.parent / "dataset_stats.pkl").open("rb") as stream:
            stats = pickle.load(stream)
        runner = FrozenACTRunner(base, stats, device, capture_features=True,
                                 inference_batch_size=inference_batch_size)
        residual_config = {"feature_dim": FEATURE_DIM, "hidden_dim": int(config["hidden_dim"]),
                           "initial_std": float(config["initial_std"]),
                           "learn_std": bool(config["learn_std"])}
        residual = ResidualActorCritic(**residual_config).to(device)
        optimizer = make_optimizer(residual)
        residual_scale = float(config["residual_scale"])
        start_update = global_step = layout_cursor = 0

    output = online_output_dir(config)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = dict(config, base_checkpoint=str(checkpoint), policy_type="act_residual_ppo",
                    residual_scale=residual_scale, resumed_from_update=start_update,
                    reward_type="score_delta_plus_physical_pbrs_waypoints_and_milestones",
                    residual_features="normalized_qpos+base_action+frozen_visual+block_poses")
    (output / "config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "base_parameters": sum(p.numel() for p in base.parameters()),
        "residual_parameters": sum(p.numel() for p in residual.parameters()),
        "trainable_parameters": sum(p.numel() for p in residual.parameters() if p.requires_grad),
        "baseline_pins": (REPO_ROOT / "third_party/pins.lock").read_text(encoding="utf-8"),
        "metrics_scope": "stochastic training rollouts; not held-out evaluation",
        "reward": "Public-score deltas + physical-state PBRS gamma*phi(next)-phi(current) "
                  + ("+ dense approach and carry progress + grasp, release and settling events. "
                     if reward_cfg.get("dense_transport_reward") else
                     "+ ordered approach and carry waypoints + grasp and placement milestones. ") +
                  "True terminals force phi=0; time truncations retain phi and bootstrap. "
                  "Grasp and placement use measured jaw aperture, lift, support geometry, "
                  "release and low velocity as proxies. "
                  "The configured target region and trajectory events add internal preferences. "
                  "The policy does not receive route, latch or dwell history. "
                  "Official scores use the 0/15/100 task logic.",
        "reward_config": dict(reward_cfg),
        "exploration_std": "fixed-std resume applies the configured initial_std to log_std; "
                           "learn_std=true resumes keep the learned log_std and its moments",
        "episode_diagnostics": "phi_initial/final/mean are measured potentials (phi_final is the "
                               "final-state potential even when terminal shaping uses zero); "
                               "final_pair_xy_m/dz_m are block label pairs (0-1, 0-2, 1-2), dz "
                               "signed in label order; gripper openness is proprio[7] and [15]",
        "partial_rate": "fraction of episodes whose peak public score is at least 15; includes success",
        "mean_policy_position_residual_m": "post-update deterministic mean correction on rollout "
                                           "states, averaged across both arms; excludes exploration",
        "updates": "actual PPO optimizer steps in this rollout update",
        "update_scope": "'update' and 'episodes' count this run; 'absolute_update' and "
                        "'global_step' include resumed optimizer progress",
        "explained_variance": "final critic vs rollout GAE returns; null if return variance is zero",
        "vram_scope": "learner-process PyTorch allocator only; excludes Isaac and other processes",
        "vram_peak_scope": "per rollout/PPO update, reset before rollout",
        "start_update": start_update,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # A self-contained checkpoint can be evaluated without the original run directory.
    def save(path, absolute_update, steps):
        payload = {
            "format": ONLINE_FORMAT, "base_config": base_config,
            "base_state_dict": frozen_state, "residual_config": residual_config,
            "residual_state_dict": residual.state_dict(),
            "stats": {key: torch.as_tensor(runner.stats[key]) for key in runner.stats},
            "residual_scale": residual_scale, "temporal_agg": True,
            "optimizer": optimizer.state_dict(), "update": absolute_update,
            "global_step": steps, "layout_cursor": layout_cursor,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": _numpy_rng_payload(),
            "torch_cuda_rng_state": (torch.cuda.get_rng_state(device)
                                     if device.type == "cuda" else None),
        }
        # Replace atomically so an interrupted write never corrupts recovery.
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    from starter.policies.act_online_metrics import metric_row, plot_online_training

    env = RemoteEnv(str(require(config, "env_host")), int(require(config, "env_port")))
    episode_count = completed_updates = 0
    stop_reason = "completed"
    expected_waves = episodes_per_update // width
    print(f"ACT residual PPO: frozen base={checkpoint}; resumed_update={start_update}; "
          f"rollout_layout_ids={layout_ids}; budget={max_train_seconds:.0f}s; "
          f"output={output}", flush=True)
    try:
        env.connect()
        started = time.monotonic()
        print("[train] simulator connected; training budget started", flush=True)
        with ((output / "metrics.jsonl").open("x", encoding="utf-8") as metrics,
              (output / "metrics.csv").open("x", encoding="utf-8", newline="") as table,
              (output / "episodes.jsonl").open("x", encoding="utf-8") as episode_log,
              (output / "episodes.csv").open("x", encoding="utf-8", newline="") as episode_table):
            writer = None
            episode_writer = csv.DictWriter(episode_table, fieldnames=_EPISODE_CSV_FIELDS,
                                            extrasaction="ignore")
            episode_writer.writeheader()
            for local_update in range(1, updates + 1):
                absolute_update = start_update + local_update
                update_started = time.monotonic()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                waves, episodes = [], []
                residual.eval()
                for _ in range(expected_waves):
                    # Stop only at complete wave boundaries; a nonempty partial
                    # update is optimized once, an empty one is never fabricated.
                    if _budget_exhausted(started, max_train_seconds):
                        stop_reason = "budget"
                        break
                    # Cycle through the selected curriculum layouts; persist the
                    # cursor so short rounds and resumed runs keep equal coverage.
                    seeds = [layout_ids[(layout_cursor + i) % len(layout_ids)] for i in range(width)]
                    # Alternate complete slot/layout cycles so original starts stay represented.
                    prefixes = ({i: prefix_archive[layout, i] for i, layout in enumerate(seeds)
                                 if (layout, i) in prefix_archive}
                                if (layout_cursor // replay_cycle) % 2 else {})
                    wave_started = time.monotonic()
                    batch, wave_episodes = collect_wave(
                        env, runner, residual, seeds, gamma=float(config["gamma"]),
                        gae_lambda=float(config["gae_lambda"]), residual_scale=residual_scale,
                        reward_cfg=reward_cfg, prefixes=prefixes,
                    )
                    layout_cursor += width
                    waves.append(batch)
                    episodes.extend(wave_episodes)
                    for episode in wave_episodes:
                        episode_count += 1
                        episode_record = {"update": local_update,
                                          "absolute_update": absolute_update,
                                          "episode": episode_count, "seed": seed,
                                          **episode}
                        episode_log.write(json.dumps(episode_record, allow_nan=False) + "\n")
                        prefix_steps = episode["prefix_steps"]
                        attempts = episode["transport_attempts"]
                        grasp_counts = _suffix_grasp_attempts(episode)
                        episode_writer.writerow({
                            **episode_record,
                            "arrivals": sum(counts["arrivals"] for counts in grasp_counts),
                            "grasps": sum(counts["confirmed"] for counts in grasp_counts),
                            "carries": sum((row["grasp_step"] or 0) > prefix_steps for row in attempts),
                            **{name: sum((row[f"{name}_step"] or 0) > prefix_steps for row in attempts)
                               for name in ("aligned", "opened", "settled")},
                        })
                    episode_log.flush()
                    episode_table.flush()
                    print(f"[rollout] layouts={seeds} episodes={episode_count} "
                          f"scores={[ep['final_score'] for ep in wave_episodes]} "
                          f"wave_seconds={time.monotonic() - wave_started:.1f} "
                          f"position_residual_mm="
                          f"{1000 * np.mean([ep['position_residual_mean_m'] for ep in wave_episodes]):.2f}",
                          flush=True)
                if not waves:
                    break
                batch = concatenate_batches(waves)
                global_step += len(batch.features)
                rollout_stats = _rollout_metrics(batch, episodes, residual_scale)
                rollout_seconds = time.monotonic() - update_started
                optimization_started = time.monotonic()
                batch = PPOBatch(*(getattr(batch, field).to(device) for field in PPOBatch.__dataclass_fields__))
                residual.train()
                losses = ppo_update(
                    residual, optimizer, batch, update_epochs=int(config["ppo_epochs"]),
                    minibatch_size=int(config["minibatch_size"]), clip_range=float(config["clip_range"]),
                    value_coef=float(config["value_coef"]), entropy_coef=float(config["entropy_coef"]),
                    max_grad_norm=float(config["max_grad_norm"]), target_kl=float(config["target_kl"]),
                )
                losses.update(_mean_policy_metrics(residual, batch.features,
                                                   runner.stats["action_std"], residual_scale))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                optimization_seconds = time.monotonic() - optimization_started
                for key, value in base.state_dict().items():
                    if not torch.equal(value.detach().cpu(), frozen_state[key]):
                        raise RuntimeError(f"Frozen ACT state changed: {key}.")
                now = time.monotonic()
                elapsed = now - started
                update_seconds = now - update_started
                remaining = _remaining_seconds(elapsed, local_update, updates, max_train_seconds)
                total = "budget" if updates == sys.maxsize else updates
                vram = _vram_stats(device)
                budget_limited = len(waves) < expected_waves
                record = {"update": local_update, "absolute_update": absolute_update,
                          "global_step": global_step, "episodes": episode_count,
                          "waves": len(waves), "budget_limited": budget_limited,
                          **rollout_stats, **losses,
                          "lr": optimizer.param_groups[0]["lr"],
                          "critic_lr": optimizer.param_groups[1]["lr"],
                          "target_kl": float(config["target_kl"]),
                          "rollout_seconds": rollout_seconds, "optimization_seconds": optimization_seconds,
                          "update_seconds": update_seconds, "elapsed_seconds": elapsed,
                          "eta_seconds": remaining,
                          "steps_per_second": len(batch.features) / update_seconds,
                          "vram": vram}
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
                metrics.flush()
                row = metric_row(record)
                if writer is None:
                    writer = csv.DictWriter(table, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                table.flush()
                memory = "VRAM=CPU"
                if vram is not None:
                    memory = (f"VRAM allocated={vram['allocated_gib']:.2f}GiB "
                              f"reserved={vram['reserved_gib']:.2f}GiB "
                              f"peak={vram['peak_allocated_gib']:.2f}GiB "
                              f"total={vram['total_gib']:.2f}GiB (learner PyTorch)")
                print(f"update {local_update}/{total} (absolute {absolute_update}): steps={global_step} "
                      f"waves={len(waves)}/{expected_waves} "
                      f"success={record['success_count']}/{len(episodes)} ({record['success_rate']:.1%}) "
                      f"partial={record['partial_rate']:.1%} score={record['mean_final_score']:.2f} | "
                      f"policy_loss={record['policy_loss']:.5f} value_loss={record['value_loss']:.5f} "
                      f"KL={record['approx_kl']:.5f} optimizer_steps={record['updates']} "
                      f"early_stop={record['early_stopped']} | lr={record['lr']:.3e} "
                      f"mean_correction_mm={1000 * record['mean_policy_position_residual_m']:.2f} "
                      f"extrema[max_lift_mm={1000 * record['max_lift_m_over_episodes']:.2f} "
                      f"max_pair={record['max_pair_sum_over_episodes']:.2f} "
                      f"min_d_mm={1000 * record['min_d_min_m_over_episodes']:.1f} "
                      f"anyLift={record['episodes_with_any_lift']}/{len(episodes)} "
                      f"anyPair={record['episodes_with_any_pair']}/{len(episodes)} "
                      f"bothOpen={record['episodes_with_both_open']}/{len(episodes)}] "
                      f"rollout={rollout_seconds:.1f}s PPO={optimization_seconds:.1f}s "
                      f"update_time={update_seconds:.1f}s elapsed={timedelta(seconds=int(elapsed))} "
                      f"ETA={timedelta(seconds=int(remaining))} | {memory}", flush=True)
                if local_update % int(config["save_freq"]) == 0:
                    save(output / f"policy_update_{absolute_update}_seed_{seed}.ckpt",
                         absolute_update, global_step)
                save(output / "policy_last.ckpt", absolute_update, global_step)
                completed_updates = local_update
                if (local_update == 1 or local_update % int(config["save_freq"]) == 0
                        or local_update == updates or stop_reason == "budget"):
                    plot_online_training(output)
                if stop_reason == "budget":
                    break
            if completed_updates == 0:
                save(output / "policy_last.ckpt", start_update, global_step)
    finally:
        env.close()
    metadata.update(completed_updates=completed_updates, stop_reason=stop_reason,
                    final_update=start_update + completed_updates, final_global_step=global_step)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    last = output / "policy_last.ckpt"
    print(f"ACT residual PPO stopped ({stop_reason}): completed_updates={completed_updates}; "
          f"absolute_update={start_update + completed_updates}; saved {last}", flush=True)
    return last
