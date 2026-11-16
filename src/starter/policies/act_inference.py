"""Shared frozen-ACT inference, residual composition and checkpoint loading.

Both the offline evaluator and the online residual trainer use this module so
preprocessing, temporal aggregation and the checkpoint format stay identical.
"""

from __future__ import annotations

import json
import pickle
from collections import deque
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import torch

from starter.interfaces import ACTION_DIM, BLOCK_LABELS, HEAD_CAMERA
from starter.policies.act import load_offline_policy
from starter.policies.act_model import ACTPolicy
from starter.policies.residual_ppo import ResidualActorCritic

VISUAL_FEATURE_DIM = 512 + 2 * ACTION_DIM
FEATURE_DIM = VISUAL_FEATURE_DIM + 7 * len(BLOCK_LABELS)
STAT_KEYS = ("qpos_mean", "qpos_std", "action_mean", "action_std")
ONLINE_FORMAT = "act_residual_ppo_v3"
MIN_QUATERNION_NORM = 1e-8


def _normalization_stats(stats: Mapping) -> dict[str, np.ndarray]:
    """Validate the four normalization vectors saved with an ACT checkpoint."""

    if not isinstance(stats, Mapping):
        raise ValueError("ACT normalization stats must be a mapping.")
    result = {}
    for key in STAT_KEYS:
        if key not in stats:
            raise ValueError(f"ACT normalization stats are missing {key}.")
        value = stats[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (ACTION_DIM,):
            raise ValueError(f"{key} must have shape ({ACTION_DIM},), got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains a non-finite value.")
        if key.endswith("_std") and np.any(array <= 0.0):
            raise ValueError(f"{key} must be positive.")
        result[key] = array
    return result


def block_pose_features(blocks: Mapping) -> np.ndarray:
    """Ordered object state for the residual, never the frozen ACT forward.

    ResiP (arXiv:2407.16677) conditions residual PPO on object state and the
    base action; implementation reference:
    https://github.com/ankile/robust-rearrangement/blob/cf80d12d4acae9e4751a49407a96f85f07b342e3/src/gym/env_rl_wrapper.py
    This adapter retains ACT visual features and adds all three block poses.
    Poses use RoboDojo's scene-relative metres and wxyz rotation,
    matching its Func_Parser.get_label_pose / LayoutManager.get_instance_pose.
    """
    poses = []
    for label in BLOCK_LABELS:
        try:
            position = np.asarray(blocks[label]["pos"], dtype=np.float32)
            rotation = np.asarray(blocks[label]["rot"], dtype=np.float32)
        except (KeyError, TypeError) as error:
            raise ValueError(f"Residual observation requires pos and rot for {label}.") from error
        if position.shape != (3,) or rotation.shape != (4,):
            raise ValueError(f"Residual {label} pose must have position(3) and rotation(4).")
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError(f"Residual {label} pose contains a non-finite value.")
        norm = np.linalg.norm(rotation)
        if norm < MIN_QUATERNION_NORM:
            raise ValueError(f"Residual {label} quaternion is near zero.")
        rotation = rotation / norm
        # q and -q encode the same orientation; use a consistent hemisphere.
        if rotation[0] < 0:
            rotation = -rotation
        poses.extend((position, rotation))
    return np.concatenate(poses)


class FrozenACTRunner:
    """Frozen base ACT policy with the shared observation preprocessing.

    ``temporal_agg`` keeps one weighted action per environment and step. Without
    it every call returns the full predicted chunk. Feature capture is opt-in so
    offline checkpoints whose policy has no ResNet backbone still work.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        stats: Mapping,
        device,
        temporal_agg: bool = True,
        capture_features: bool = False,
    ):
        self.device = torch.device(device)
        self.policy = policy.to(self.device).eval()
        self.policy.requires_grad_(False)
        self.stats = _normalization_stats(stats)
        self.temporal_agg = bool(temporal_agg)
        self.capture_features = bool(capture_features)
        self.chunk_size = int(policy.model.num_queries)
        if self.chunk_size < 1:
            raise ValueError("ACT policy must predict at least one action.")
        self.histories: dict[object, deque] = {}
        self._visual_features: torch.Tensor | None = None
        self._hook = None
        if self.capture_features:
            # The first backbone output map has 512 channels; global-average it
            # inside the one forward pass that also produces the base actions.
            backbone = self.policy.model.backbones[0]
            self._hook = backbone.register_forward_hook(self._capture_visual)

    def _capture_visual(self, _module, _inputs, output) -> None:
        features = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise ValueError("ACT backbone must return a [batch, channels, height, width] map.")
        if features.shape[1] != 512:
            raise ValueError(f"ACT backbone must expose 512 channels, got {features.shape[1]}.")
        self._visual_features = features.mean(dim=(2, 3))

    def reset(self) -> None:
        self.histories = {}
        self._visual_features = None

    def _preprocess(self, obs: Mapping) -> tuple[np.ndarray, np.ndarray]:
        camera = obs.get("camera", HEAD_CAMERA)
        if camera != HEAD_CAMERA:
            raise ValueError(f"ACT inference only accepts the {HEAD_CAMERA!r} camera.")
        image = np.asarray(obs["image"])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a decoded HWC RGB array.")
        if not np.isfinite(image).all():
            raise ValueError("image contains a non-finite value.")
        image = cv2.resize(image, (640, 480)).astype(np.float32) / 255.0
        image = np.moveaxis(image, -1, 0)

        proprio = np.asarray(obs["proprio"], dtype=np.float32)
        if proprio.shape != (ACTION_DIM,):
            raise ValueError(f"proprio must have shape ({ACTION_DIM},), got {proprio.shape}.")
        if not np.isfinite(proprio).all():
            raise ValueError("proprio contains a non-finite value.")
        qpos = (proprio - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos.astype(np.float32), image

    def predict_base(
        self, obs_list, env_ids, with_features: bool = False
    ) -> tuple[list[np.ndarray], torch.Tensor | None]:
        """Return action chunks and, when requested, residual-head features.

        With temporal aggregation enabled each returned chunk contains the one
        physical action selected for this step. Otherwise it contains the whole
        predicted action chunk.
        """

        if with_features and not self.temporal_agg:
            raise ValueError("Feature capture requires temporal_agg=True.")
        if with_features and not self.capture_features:
            raise ValueError("Feature capture requires capture_features=True.")
        if len(obs_list) != len(env_ids):
            raise ValueError("obs_list and env_ids must have the same length.")
        if not obs_list:
            raise ValueError("Predict at least one observation.")

        prepared = [self._preprocess(obs) for obs in obs_list]
        states, images = zip(*prepared)
        qpos = torch.from_numpy(np.stack(states)).to(self.device)
        image = torch.from_numpy(np.stack(images)[:, None]).to(self.device)

        self._visual_features = None
        # no_grad keeps captured features usable as inputs to a trainable head;
        # inference_mode tensors cannot be consumed by autograd.
        with torch.no_grad():
            predictions = self.policy(qpos, image)
        if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3:
            raise ValueError("ACT policy must return a [batch, chunk, action] tensor.")
        if predictions.shape[0] != len(obs_list) or predictions.shape[2] != ACTION_DIM:
            raise ValueError(
                f"ACT policy returned action shape {tuple(predictions.shape)} for "
                f"{len(obs_list)} observations."
            )
        predictions = predictions.detach().cpu().numpy()
        if not np.isfinite(predictions).all():
            raise ValueError("ACT produced a non-finite action.")

        chunks: list[np.ndarray] = []
        normalized_actions: list[np.ndarray] = []
        for env_id, predicted in zip(env_ids, predictions):
            if self.temporal_agg:
                history = self.histories.setdefault(env_id, deque(maxlen=self.chunk_size))
                history.append(predicted)
                count = len(history)
                # Oldest prediction first, exactly as the upstream ACT weighting.
                aligned = np.stack([chunk[count - 1 - i] for i, chunk in enumerate(history)])
                weights = np.exp(-0.01 * np.arange(count))
                selected = (aligned * (weights / weights.sum())[:, None]).sum(0, keepdims=True)
            else:
                selected = predicted
            actions = selected * self.stats["action_std"] + self.stats["action_mean"]
            if not np.isfinite(actions).all():
                raise ValueError("ACT produced a non-finite action.")
            chunks.append(actions.astype(np.float32))
            if with_features:
                normalized_actions.append(selected[0].astype(np.float32))

        features = None
        if with_features:
            visual = self._visual_features
            if visual is None or visual.shape != (len(obs_list), 512):
                raise ValueError("ACT backbone did not produce one 512-channel feature per observation.")
            if not torch.isfinite(visual).all():
                raise ValueError("ACT backbone produced a non-finite feature.")
            normalized = torch.from_numpy(np.stack(normalized_actions)).to(self.device)
            try:
                object_state = np.stack([block_pose_features(obs["blocks"]) for obs in obs_list])
            except KeyError as error:
                raise ValueError("Residual observations require block poses.") from error
            objects = torch.from_numpy(object_state).to(device=self.device, dtype=qpos.dtype)
            features = torch.cat([qpos, normalized, visual.to(qpos.dtype), objects], dim=-1)
            if features.shape != (len(obs_list), FEATURE_DIM):
                raise ValueError(f"Residual features must have shape ({len(obs_list)}, {FEATURE_DIM}).")
        return chunks, features


def compose_residual_action(
    base_actions: np.ndarray,
    raw_actions: torch.Tensor | np.ndarray,
    action_std,
    residual_scale: float,
) -> np.ndarray:
    """Add a scaled residual to base actions and restore the action contract."""

    base = np.asarray(base_actions, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != ACTION_DIM:
        raise ValueError(f"base_actions must have shape [N, {ACTION_DIM}], got {base.shape}.")
    if not np.isfinite(base).all():
        raise ValueError("base_actions contain a non-finite value.")

    if isinstance(raw_actions, torch.Tensor):
        raw_actions = raw_actions.detach().cpu().numpy()
    raw = np.asarray(raw_actions, dtype=np.float32)
    if raw.shape != base.shape:
        raise ValueError(f"raw_actions must have shape {base.shape}, got {raw.shape}.")
    if not np.isfinite(raw).all():
        raise ValueError("raw_actions contain a non-finite value.")

    scale = float(residual_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("residual_scale must be a non-negative finite number.")
    std = np.asarray(
        action_std.detach().cpu().numpy() if isinstance(action_std, torch.Tensor) else action_std,
        dtype=np.float32,
    )
    if std.ndim == 0:
        std = np.full(ACTION_DIM, float(std), dtype=np.float32)
    if std.shape != (ACTION_DIM,) or not np.isfinite(std).all():
        raise ValueError(f"action_std must be {ACTION_DIM} finite values.")

    actions = base + np.clip(raw, -1.0, 1.0) * scale * std
    actions[:, (7, 15)] = np.clip(actions[:, (7, 15)], 0.0, 1.0)
    # Pose positions stay unclipped; only scalar-first quaternions are renormalized.
    for start in (3, 11):
        quaternion = actions[:, start : start + 4]
        norms = np.linalg.norm(quaternion, axis=1)
        if not np.isfinite(quaternion).all() or not np.isfinite(norms).all():
            raise ValueError("composed quaternion contains a non-finite value.")
        if np.any(norms < MIN_QUATERNION_NORM):
            raise ValueError("composed quaternion is near zero and cannot be normalized.")
        actions[:, start : start + 4] = quaternion / norms[:, None]
    if not np.isfinite(actions).all():
        raise ValueError("composed action contains a non-finite value.")
    return actions.astype(np.float32)


def _load_online(checkpoint: Path, device, temporal_agg: bool):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != ONLINE_FORMAT:
        raise ValueError(f"Unsupported ACT residual checkpoint format; expected {ONLINE_FORMAT!r}.")
    if payload.get("temporal_agg") is not True or not temporal_agg:
        raise ValueError("ACT residual checkpoints require temporal_agg=True.")

    required = ("base_config", "base_state_dict", "residual_config", "residual_state_dict",
                "stats", "residual_scale")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"ACT residual checkpoint is missing {missing}.")
    residual_config = dict(payload["residual_config"])
    if int(residual_config.get("feature_dim", -1)) != FEATURE_DIM:
        raise ValueError(f"Residual feature_dim must be {FEATURE_DIM}.")
    residual_config.setdefault("action_dim", ACTION_DIM)

    policy = ACTPolicy(payload["base_config"], pretrained_backbone=False)
    policy.load_state_dict(payload["base_state_dict"], strict=True)
    residual = ResidualActorCritic(**residual_config)
    residual.load_state_dict(payload["residual_state_dict"], strict=True)

    runner = FrozenACTRunner(
        policy.to(device), payload["stats"], device,
        temporal_agg=True, capture_features=True,
    )
    residual = residual.to(device).eval()
    return runner, residual, float(payload["residual_scale"])


def load_act_runner(checkpoint, device, temporal_agg: bool = True):
    """Load an offline or online ACT checkpoint through the shared runner.

    Returns ``(runner, residual_model, residual_scale)``. Offline checkpoints
    return ``None`` for the last two values.
    """

    checkpoint = Path(checkpoint).expanduser().resolve()
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    policy_type = config.get("policy_type")
    if policy_type == "act_residual_ppo":
        return _load_online(checkpoint, device, temporal_agg)
    if policy_type is not None:
        raise ValueError(f"Unsupported ACT policy_type {policy_type!r}.")

    policy = load_offline_policy(checkpoint, device)
    with (checkpoint.parent / "dataset_stats.pkl").open("rb") as stream:
        stats = pickle.load(stream)
    runner = FrozenACTRunner(
        policy, stats, device, temporal_agg=temporal_agg, capture_features=False,
    )
    return runner, None, None
