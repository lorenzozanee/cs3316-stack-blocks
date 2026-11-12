"""FPO++ manipulation fine-tuning with per-sample CFM ratios and PPO clipping.

Algorithm: Flow Policy Gradients for Robot Control, https://arxiv.org/abs/2602.02481
Implementation reference: https://github.com/amazon-far/fpo-control
This project implements the manipulation variant: per-sample CFM ratios with PPO
clipping, without ASPO. Visual features and rollout records stay on CPU between
updates. No demonstrations or differentiable ODE sampling are needed online.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn

from starter.config import require
from starter.data.diffusion import preprocess_image
from starter.envs.remote import RemoteEnv
from starter.interfaces import ACTION_DIM
from starter.paths import run_name, run_stamp
from starter.policies.diffusion_model import FlowMatchingPolicy


@dataclass
class FPOSettings:
    checkpoint: str = ""
    resume: str | None = None
    device: str = "cuda"
    updates: int = 100
    rollout_chunks: int = 64
    minibatch_size: int = 16
    microbatch_size: int = 1
    cfm_samples: int = 8
    update_epochs: int = 4
    value_warmup_updates: int = 1
    learning_rate: float = 1e-5
    value_learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.01
    value_clip: float = 0.2
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    checkpoint_every: int = 10
    output_dir: str = "outputs/fpo"
    preflight: bool = False

    @classmethod
    def from_config(cls, config):
        options = config.get("fpo", {})
        unknown = set(options) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown fpo settings: {', '.join(sorted(unknown))}.")
        settings = cls(**options)
        if not settings.checkpoint:
            raise ValueError("Provide the pretrained checkpoint using --checkpoint or fpo.checkpoint.")
        for key in ("updates", "rollout_chunks", "minibatch_size", "microbatch_size", "cfm_samples",
                    "update_epochs", "checkpoint_every"):
            value = getattr(settings, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"fpo.{key} must be a positive integer.")
        if not isinstance(settings.value_warmup_updates, int) or isinstance(settings.value_warmup_updates, bool) \
                or settings.value_warmup_updates < 0:
            raise ValueError("fpo.value_warmup_updates must be a nonnegative integer.")
        if settings.microbatch_size > settings.minibatch_size:
            raise ValueError("fpo.microbatch_size must not exceed minibatch_size.")
        if settings.device not in ("cuda", "cpu"):
            raise ValueError("fpo.device must be cuda or cpu.")
        for key in ("learning_rate", "value_learning_rate", "clip_epsilon", "value_clip", "max_grad_norm", "target_kl"):
            value = getattr(settings, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"fpo.{key} must be finite and positive.")
        for key in ("gamma", "gae_lambda"):
            value = getattr(settings, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"fpo.{key} must be in (0, 1].")
        if settings.clip_epsilon >= 1:
            raise ValueError("fpo.clip_epsilon must be less than 1.")
        if not isinstance(settings.preflight, bool):
            raise TypeError("fpo.preflight must be boolean.")
        return settings


def clipped_policy_loss(new_losses, old_losses, advantages, clip_epsilon):
    while advantages.ndim < new_losses.ndim:
        advantages = advantages.unsqueeze(-1)
    logratio = old_losses.float() - new_losses.float()
    if not torch.isfinite(logratio).all() or logratio.abs().max() > 80:
        raise FloatingPointError("Non-finite or excessive FPO CFM log-ratio; reduce the learning rate.")
    ratio = logratio.exp()
    loss = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon),
    ).mean()
    with torch.no_grad():
        metrics = {
            "approx_kl": ((ratio - 1) - logratio).mean(),
            "clip_fraction": ((ratio - 1).abs() > clip_epsilon).float().mean(),
        }
    return loss, metrics


def cfm_error(policy, visual, states, actions, times, noise, valid_steps):
    """Per-noise/time error, summing executed steps after averaging action dimensions."""
    condition = policy.condition_from_visual(visual, states)
    interpolated = torch.lerp(noise, actions, times[:, None, None])
    velocity = policy(interpolated, times, condition)
    error = (velocity.float() - (actions - noise).float()).square().mean(dim=-1)
    mask = torch.arange(policy.horizon, device=error.device)[None] < valid_steps[:, None]
    return (error * mask).sum(dim=-1)


def compute_gae(rewards, values, next_values, discounts, traces, dones, env_ids):
    """Variable-duration transitions; done stops traces, not truncation bootstraps."""
    advantages = torch.zeros_like(values)
    following = {}
    for index in reversed(range(len(values))):
        env_id = int(env_ids[index])
        delta = rewards[index] + discounts[index] * next_values[index] - values[index]
        continuation = 0.0 if bool(dones[index]) else following.get(env_id, 0.0)
        advantages[index] = delta + traces[index] * continuation
        following[env_id] = advantages[index]
    return advantages, advantages + values


class ValueNetwork(nn.Module):
    def __init__(self, policy):
        super().__init__()
        width = policy.n_obs_steps * (policy.vision.hidden_dim + ACTION_DIM)
        output = nn.Linear(256, 1)
        self.layers = nn.Sequential(
            nn.Linear(width, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), output,
        )
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, visual, normalized_states):
        inputs = torch.cat((visual.float().mean(dim=2).flatten(1), normalized_states.float().flatten(1)), dim=-1)
        return self.layers(inputs).squeeze(-1)


@dataclass
class Transition:
    visual: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    times: torch.Tensor
    noise: torch.Tensor
    old_losses: torch.Tensor
    value: float
    reward: float
    next_value: float
    steps: int
    done: bool
    env_id: int


class RolloutCollector:
    def __init__(self, trainer, env, num_envs, seed):
        self.trainer = trainer
        self.env = env
        self.num_envs = num_envs
        self.next_seed = seed
        self.active = []
        self.images = [deque(maxlen=trainer.policy.n_obs_steps) for _ in range(num_envs)]
        self.states = [deque(maxlen=trainer.policy.n_obs_steps) for _ in range(num_envs)]
        self.episode_seeds = []
        self.episode_steps = [0] * num_envs
        self.episode_returns = [0.0] * num_envs
        self.episode_scores = [0.0] * num_envs
        self.env_steps = 0
        self.episodes = []

    def _observe(self, index, observation):
        policy = self.trainer.policy
        image = torch.from_numpy(preprocess_image(observation["image"], policy.image_size))
        state = policy.canonicalize(torch.as_tensor(observation["proprio"]).reshape(ACTION_DIM)).cpu()
        if not self.images[index]:
            self.images[index].extend([image] * policy.n_obs_steps)
            self.states[index].extend([state] * policy.n_obs_steps)
        else:
            self.images[index].append(image)
            self.states[index].append(state)

    def _reset(self):
        seeds = list(range(self.next_seed, self.next_seed + self.num_envs))
        observations, _ = self.env.reset_batch(seeds)
        if len(observations) != self.num_envs:
            raise ValueError("RemoteEnv returned a different number of scenes than num_envs.")
        self.next_seed += self.num_envs
        self.episode_seeds = seeds
        self.active = list(range(self.num_envs))
        for index, observation in enumerate(observations):
            self.images[index].clear()
            self.states[index].clear()
            self.episode_steps[index] = 0
            self.episode_returns[index] = 0.0
            self.episode_scores[index] = 0.0
            self._observe(index, observation)

    @torch.no_grad()
    def _features(self, index):
        device = self.trainer.device
        images = torch.stack(tuple(self.images[index]))[None].to(device)
        states = torch.stack(tuple(self.states[index]))[None].to(device)
        with self.trainer.autocast():
            visual = self.trainer.policy.encode_visual(images)
        return visual.cpu(), states.cpu()

    @torch.no_grad()
    def collect(self, count):
        trainer, settings, policy = self.trainer, self.trainer.settings, self.trainer.policy
        transitions = []
        self.episodes = []
        while len(transitions) < count:
            if not self.active:
                self._reset()
            indices = list(self.active)
            plans, records = {}, {}
            for index in indices:
                visual, states = self._features(index)
                with trainer.autocast():
                    condition = policy.condition_from_visual(visual.to(trainer.device), states.to(trainer.device))
                    actions = policy.sample_normalized_actions(condition)
                    value = trainer.values(visual.to(trainer.device), states.to(trainer.device))
                if not torch.isfinite(actions).all():
                    raise FloatingPointError("Flow sampler produced non-finite actions.")
                plans[index] = policy.denormalize_actions(actions)[0].cpu().tolist()
                records[index] = {
                    "visual": visual[0], "states": states[0], "actions": actions[0].float().cpu(),
                    "value": float(value[0]), "reward": 0.0, "steps": 0, "done": False,
                    "terminated": False,
                }
            executing = list(indices)
            for offset in range(policy.action_steps):
                results = self.env.step_batch([plans[index][offset] for index in executing], executing)
                if len(results) != len(executing):
                    raise ValueError("RemoteEnv returned a different number of transitions than active scenes.")
                remaining = []
                for index, (observation, score, terminated, truncated, info) in zip(executing, results):
                    score = float(score)
                    if not math.isfinite(score):
                        raise FloatingPointError("Environment score is not finite.")
                    self._observe(index, observation)
                    record = records[index]
                    record["reward"] += settings.gamma ** offset * score
                    record["steps"] += 1
                    record["done"] = bool(terminated or truncated)
                    record["terminated"] = bool(terminated)
                    self.env_steps += 1
                    self.episode_steps[index] += 1
                    self.episode_returns[index] += score
                    self.episode_scores[index] = max(self.episode_scores[index], score)
                    if record["done"]:
                        self.episodes.append({
                            "seed": self.episode_seeds[index], "env_id": index,
                            "steps": self.episode_steps[index], "return": self.episode_returns[index],
                            "score": self.episode_scores[index],
                            "success": int(bool(terminated) and bool(info["success"])),
                            "terminated": int(bool(terminated)), "truncated": int(bool(truncated)),
                        })
                    else:
                        remaining.append(index)
                executing = remaining
                if not executing:
                    break
            self.active = executing
            for index in indices:
                record = records[index]
                if record.pop("terminated"):
                    next_value = 0.0
                else:
                    visual, states = self._features(index)
                    with trainer.autocast():
                        next_value = float(trainer.values(visual.to(trainer.device), states.to(trainer.device))[0])
                times = torch.rand(settings.cfm_samples, generator=trainer.cfm_generator)
                noise = torch.randn(settings.cfm_samples, policy.horizon, ACTION_DIM, generator=trainer.cfm_generator)
                old_losses = []
                for sample in range(settings.cfm_samples):
                    with trainer.autocast():
                        error = cfm_error(
                            policy, record["visual"][None].to(trainer.device), record["states"][None].to(trainer.device),
                            record["actions"][None].to(trainer.device), times[sample:sample + 1].to(trainer.device),
                            noise[sample:sample + 1].to(trainer.device),
                            torch.tensor([record["steps"]], device=trainer.device),
                        )
                    old_losses.append(error[0].float().cpu())
                transitions.append(Transition(
                    **record, times=times, noise=noise, old_losses=torch.stack(old_losses),
                    next_value=next_value, env_id=index,
                ))
        return transitions


class FPOTrainer:
    def __init__(self, config):
        self.config = config
        self.settings = FPOSettings.from_config(config)
        settings = self.settings
        self.device = torch.device(settings.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("fpo.device=cuda requires a CUDA-enabled PyTorch installation; choose cpu explicitly otherwise.")
        seed = int(require(config, "seed"))
        random.seed(seed)
        torch.manual_seed(seed)
        self.cfm_generator = torch.Generator().manual_seed(seed)
        self.shuffle_generator = torch.Generator().manual_seed(seed + 1)
        checkpoint_path = Path(settings.resume or settings.checkpoint).expanduser()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.policy_config = checkpoint["policy"]
        weights = checkpoint["model"]
        bounds = {f"{key}_{side}": weights[f"{key}_{side}"]
                  for key in ("state", "action") for side in ("min", "max")}
        self.policy = FlowMatchingPolicy(**self.policy_config, bounds=bounds)
        self.policy.load_state_dict(weights)
        self.policy.vision.requires_grad_(False)
        # Dropout must be disabled for paired old/new CFM comparisons.
        self.policy.to(self.device).eval()
        self.value = ValueNetwork(self.policy).to(self.device)
        self.actor_parameters = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        self.actor_optimizer = torch.optim.AdamW(
            self.actor_parameters, lr=settings.learning_rate, weight_decay=0.01, foreach=False,
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value.parameters(), lr=settings.value_learning_rate, weight_decay=0.0, foreach=False,
        )
        self.amp_dtype = None
        self.scaler = None
        if self.device.type == "cuda":
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
            if self.amp_dtype == torch.float16:
                self.scaler = torch.amp.GradScaler("cuda")
        self.iteration = 0
        self.env_steps = 0
        self.completed_episodes = 0
        self.next_seed = seed
        if settings.resume:
            if checkpoint.get("algorithm") != "fpo++" or "actor_optimizer" not in checkpoint:
                raise ValueError("--resume requires a full FPO last.ckpt, not a weights-only checkpoint.")
            self.value.load_state_dict(checkpoint["value"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
            self.iteration = int(checkpoint["iteration"])
            self.env_steps = int(checkpoint["env_steps"])
            self.completed_episodes = int(checkpoint["completed_episodes"])
            self.next_seed = int(checkpoint["next_seed"])
            self.cfm_generator.set_state(checkpoint["cfm_rng"])
            self.shuffle_generator.set_state(checkpoint["shuffle_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if self.device.type == "cuda" and "cuda_rng" in checkpoint:
                torch.cuda.set_rng_state(checkpoint["cuda_rng"], self.device)
            if self.scaler is not None and checkpoint.get("scaler") is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])
        del checkpoint, weights
        self.run_dir = Path(settings.output_dir).expanduser() / run_name(config) / run_stamp(config)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
        self.started = time.monotonic()

    def autocast(self):
        return torch.autocast("cuda", dtype=self.amp_dtype) if self.amp_dtype is not None else nullcontext()

    def values(self, visual, states):
        return self.value(visual, self.policy.normalize(states, "state"))

    def _backward(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite FPO loss.")
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()

    def update(self, rollout, *, actor_enabled=True, epochs=None):
        settings = self.settings
        def vector(key, dtype=torch.float32):
            return torch.tensor([getattr(record, key) for record in rollout], dtype=dtype)
        steps = vector("steps")
        advantages, returns = compute_gae(
            vector("reward"), vector("value"), vector("next_value"), settings.gamma ** steps,
            (settings.gamma * settings.gae_lambda) ** steps, vector("done", torch.bool), vector("env_id", torch.int64),
        )
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        old_value_vector = vector("value")
        totals = {key: 0.0 for key in ("actor_loss", "value_loss", "approx_kl", "clip_fraction", "actor_grad_norm", "value_grad_norm")}
        batches = 0
        stop = False
        for _epoch in range(settings.update_epochs if epochs is None else epochs):
            order = torch.randperm(len(rollout), generator=self.shuffle_generator).tolist()
            for start in range(0, len(order), settings.minibatch_size):
                indices = order[start:start + settings.minibatch_size]
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)
                batch_metrics = {key: 0.0 for key in totals}
                for offset in range(0, len(indices), settings.microbatch_size):
                    selected = indices[offset:offset + settings.microbatch_size]
                    records = [rollout[index] for index in selected]
                    def stack(key, batch_records=records):
                        return torch.stack([getattr(record, key) for record in batch_records]).to(self.device)
                    visual, states = stack("visual"), stack("states")
                    old_values = old_value_vector[selected].to(self.device)
                    targets = returns[selected].to(self.device)
                    with self.autocast():
                        predictions = self.values(visual, states).float()
                        clipped = old_values + (predictions - old_values).clamp(-settings.value_clip, settings.value_clip)
                        value_loss = 0.5 * torch.maximum((predictions - targets).square(), (clipped - targets).square()).sum() / len(indices)
                    self._backward(value_loss)
                    batch_metrics["value_loss"] += float(value_loss.detach())
                    if not actor_enabled:
                        continue
                    actions = stack("actions")
                    valid = torch.tensor([record.steps for record in records], device=self.device)
                    adv = advantages[selected].to(self.device)
                    old_losses = stack("old_losses")
                    times, noise = stack("times"), stack("noise")
                    weight = len(selected) / (len(indices) * settings.cfm_samples)
                    for sample in range(settings.cfm_samples):
                        with self.autocast():
                            errors = cfm_error(
                                self.policy, visual, states, actions, times[:, sample], noise[:, sample], valid,
                            )
                            loss, metrics = clipped_policy_loss(errors, old_losses[:, sample], adv, settings.clip_epsilon)
                        self._backward(loss * weight)
                        batch_metrics["actor_loss"] += float(loss.detach()) * weight
                        for key in ("approx_kl", "clip_fraction"):
                            batch_metrics[key] += float(metrics[key]) * weight
                if actor_enabled and batch_metrics["approx_kl"] > settings.target_kl:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.value_optimizer.zero_grad(set_to_none=True)
                    stop = True
                    break
                optimizers = [self.value_optimizer]
                if actor_enabled:
                    optimizers.append(self.actor_optimizer)
                if self.scaler is not None:
                    for optimizer in optimizers:
                        self.scaler.unscale_(optimizer)
                for key, parameters in (("value_grad_norm", self.value.parameters()), ("actor_grad_norm", self.actor_parameters)):
                    if key == "actor_grad_norm" and not actor_enabled:
                        continue
                    norm = nn.utils.clip_grad_norm_(parameters, settings.max_grad_norm, error_if_nonfinite=True)
                    batch_metrics[key] = float(norm)
                for optimizer in optimizers:
                    if self.scaler is None:
                        optimizer.step()
                    else:
                        self.scaler.step(optimizer)
                if self.scaler is not None:
                    self.scaler.update()
                batches += 1
                for key in totals:
                    totals[key] += batch_metrics[key]
            if stop:
                break
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        if not batches:
            raise RuntimeError("CFM ratios already exceed target_kl before the first update; check paired sampling.")
        return {key: value / batches for key, value in totals.items()}

    def save(self, *, full=True):
        directory = self.run_dir / "checkpoints"
        directory.mkdir(exist_ok=True)
        payload = {
            "algorithm": "fpo++", "model": self.policy.state_dict(), "policy": self.policy_config,
            "value": self.value.state_dict(), "iteration": self.iteration,
            "env_steps": self.env_steps, "completed_episodes": self.completed_episodes,
            "next_seed": self.next_seed, "fpo": asdict(self.settings),
        }
        if full:
            payload.update({
                "actor_optimizer": self.actor_optimizer.state_dict(), "value_optimizer": self.value_optimizer.state_dict(),
                "cfm_rng": self.cfm_generator.get_state(), "shuffle_rng": self.shuffle_generator.get_state(),
                "torch_rng": torch.get_rng_state(), "scaler": None if self.scaler is None else self.scaler.state_dict(),
            })
            if self.device.type == "cuda":
                payload["cuda_rng"] = torch.cuda.get_rng_state(self.device)
        path = directory / ("last.ckpt" if full else f"iteration_{self.iteration:04d}.ckpt")
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def preflight(self):
        settings, policy = self.settings, self.policy
        memory_before = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            memory_before = {"free_mib": free / 2 ** 20, "total_mib": total / 2 ** 20}
        with torch.no_grad(), self.autocast():
            images = torch.zeros(1, policy.n_obs_steps, 3, policy.image_size, policy.image_size,
                                 dtype=torch.uint8, device=self.device)
            midpoint = (torch.as_tensor(policy.state_min) + torch.as_tensor(policy.state_max)) / 2
            states = policy.canonicalize(midpoint, strict=False)[None, None].repeat(1, policy.n_obs_steps, 1)
            visual = policy.encode_visual(images)
            actions = policy.sample_normalized_actions(policy.condition_from_visual(visual, states))
            value = float(self.values(visual, states)[0])
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Preflight sampler produced non-finite actions.")
        rollout = []
        for index in range(max(2, settings.microbatch_size)):
            times = torch.rand(settings.cfm_samples, generator=self.cfm_generator)
            noise = torch.randn(settings.cfm_samples, policy.horizon, ACTION_DIM, generator=self.cfm_generator)
            old_losses = []
            for sample in range(settings.cfm_samples):
                with torch.no_grad(), self.autocast():
                    error = cfm_error(
                        policy, visual, states, actions, times[sample:sample + 1].to(self.device),
                        noise[sample:sample + 1].to(self.device), torch.tensor([policy.action_steps], device=self.device),
                    )
                old_losses.append(error[0].cpu())
            rollout.append(Transition(
                visual[0].cpu(), states[0].cpu(), actions[0].float().cpu(), times, noise, torch.stack(old_losses),
                value, float(index % 2), 0.0, policy.action_steps, True, 0,
            ))
        metrics = self.update(rollout, epochs=1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        report = {
            "device": str(self.device), "amp_dtype": str(self.amp_dtype),
            "parameters": sum(parameter.numel() for parameter in policy.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.actor_parameters),
            "vision_frozen": all(not parameter.requires_grad and parameter.grad is None for parameter in policy.vision.parameters()),
            "microbatch_size": settings.microbatch_size, "cfm_samples": settings.cfm_samples,
            "memory_before": memory_before,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2 ** 20 if self.device.type == "cuda" else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2 ** 20 if self.device.type == "cuda" else None,
            "elapsed_s": time.monotonic() - self.started, "synthetic_update": metrics,
        }
        path = self.run_dir / "preflight.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        print(f"preflight={path}", flush=True)
        return path

    def train(self):
        settings = self.settings
        num_envs = require(self.config, "num_envs")
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer.")
        env = RemoteEnv(str(require(self.config, "env_host")), int(require(self.config, "env_port")))
        collector = RolloutCollector(self, env, num_envs, self.next_seed)
        collector.env_steps = self.env_steps
        metrics_path = self.run_dir / "metrics.csv"
        episodes_path = self.run_dir / "episodes.csv"
        metric_fields = [
            "iteration", "env_steps", "chunks", "episodes", "completed_episodes", "success_rate", "mean_score", "mean_return",
            "actor_loss", "value_loss", "approx_kl", "clip_fraction", "actor_grad_norm", "value_grad_norm",
            "peak_gpu_mib", "elapsed_s",
        ]
        episode_fields = ["iteration", "seed", "env_id", "steps", "return", "score", "success", "terminated", "truncated"]
        checkpoint = None
        try:
            with metrics_path.open("w", newline="") as metrics_file, episodes_path.open("w", newline="") as episodes_file:
                metrics_writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
                episodes_writer = csv.DictWriter(episodes_file, fieldnames=episode_fields)
                metrics_writer.writeheader()
                episodes_writer.writeheader()
                for iteration in range(self.iteration + 1, settings.updates + 1):
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                    rollout = collector.collect(settings.rollout_chunks)
                    self.env_steps = collector.env_steps
                    self.next_seed = collector.next_seed
                    self.completed_episodes += len(collector.episodes)
                    metrics = self.update(rollout, actor_enabled=iteration > settings.value_warmup_updates)
                    self.iteration = iteration
                    episodes = collector.episodes
                    row = {
                        "iteration": iteration, "env_steps": self.env_steps, "chunks": len(rollout),
                        "episodes": len(episodes), "completed_episodes": self.completed_episodes,
                        "success_rate": sum(episode["success"] for episode in episodes) / len(episodes) if episodes else "",
                        "mean_score": sum(episode["score"] for episode in episodes) / len(episodes) if episodes else "",
                        "mean_return": sum(episode["return"] for episode in episodes) / len(episodes) if episodes else "",
                        **metrics,
                        "peak_gpu_mib": torch.cuda.max_memory_allocated(self.device) / 2 ** 20 if self.device.type == "cuda" else "",
                        "elapsed_s": time.monotonic() - self.started,
                    }
                    metrics_writer.writerow(row)
                    for episode in episodes:
                        episodes_writer.writerow({"iteration": iteration, **episode})
                    metrics_file.flush()
                    episodes_file.flush()
                    checkpoint = self.save()
                    if iteration % settings.checkpoint_every == 0 or iteration == settings.updates:
                        self.save(full=False)
                    print(
                        f"iteration={iteration}/{settings.updates} env_steps={self.env_steps} "
                        f"episodes={self.completed_episodes} actor_loss={metrics['actor_loss']:.5f} "
                        f"value_loss={metrics['value_loss']:.5f} kl={metrics['approx_kl']:.5f}", flush=True,
                    )
        finally:
            env.close()
        if checkpoint is None:
            raise ValueError("fpo.updates must exceed the iteration in the resumed checkpoint.")
        print(f"checkpoint={checkpoint}", flush=True)
        return checkpoint


def train_fpo(config):
    trainer = FPOTrainer(config)
    return trainer.preflight() if trainer.settings.preflight else trainer.train()
