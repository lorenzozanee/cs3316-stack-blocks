"""Advantage-Weighted Regression (AWR) online fine-tuning for unfrozen ACT.

Updates ACT's active inference pathway (ResNet18 visual backbone, input_proj,
Transformer, and action_head) on executed action chunks weighted by GAE
advantages, while keeping the unused CVAE encoder frozen and preserving
temporal_agg consistency with evaluation.

A lightweight training-only 2D spatial-softmax auxiliary head supervises the
visual token grid (``input_proj`` output) to predict scene-relative 3D block
centers across all training layouts without altering ``ACTPolicy``'s inference
architecture or checkpoint contract.

References:
- AWR: Peng et al., "Advantage-Weighted Regression: Simple and Scalable
  Off-Policy Reinforcement Learning" (arXiv:1910.00177, 2019).
- Spatial Softmax: Finn et al., "Deep Spatial Autoencoders for Visuomotor
  Learning" (ICRA 2016).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import normalize

from starter.config import require
from starter.interfaces import ACTION_DIM, BLOCK_LABELS

_UNUSED_CVAE_PREFIXES = (
    "model.encoder.",
    "model.cls_embed.",
    "model.encoder_action_proj.",
    "model.encoder_joint_proj.",
    "model.latent_proj.",
    "model.is_pad_head.",
)

ACT_AWR_KEYS = (
    "unfreeze_act",
    "act_lr",
    "act_backbone_lr",
    "act_weight_decay",
    "act_epochs",
    "act_batch_size",
    "act_grad_accum_steps",
    "act_max_samples",
    "act_sample_stride",
    "act_awr_beta",
    "act_awr_max_weight",
    "act_prior_coef",
    "act_max_action_shift",
    "act_max_grad_norm",
    "act_aux_pose_weight",
    "act_aux_pose_lr",
    "act_aux_warmup_epochs",
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_JPEG_QUALITY = 90
_AUX_POSE_DIM = 3 * len(BLOCK_LABELS)


def validate_act_awr_config(config: dict) -> bool:
    """Validate unfrozen-ACT hyperparameters and return ``unfreeze_act``."""
    unfreeze = require(config, "unfreeze_act")
    if not isinstance(unfreeze, bool):
        raise ValueError("unfreeze_act must be a boolean.")
    if not unfreeze:
        return False
    for key in (
        "act_lr",
        "act_backbone_lr",
        "act_awr_beta",
        "act_awr_max_weight",
        "act_max_action_shift",
        "act_max_grad_norm",
        "act_aux_pose_lr",
    ):
        value = float(require(config, key))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be positive and finite.")
    for key in ("act_weight_decay", "act_prior_coef", "act_aux_pose_weight"):
        value = float(require(config, key))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be non-negative and finite.")
    for key in (
        "act_epochs",
        "act_batch_size",
        "act_grad_accum_steps",
        "act_max_samples",
        "act_sample_stride",
    ):
        value = require(config, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    warmup = require(config, "act_aux_warmup_epochs")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("act_aux_warmup_epochs must be a non-negative integer.")
    return True


def is_unused_cvae_param(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _UNUSED_CVAE_PREFIXES)


def configure_act_unfreeze(policy: nn.Module) -> tuple[list[tuple[str, nn.Parameter]], dict[str, torch.Tensor]]:
    """Unfreeze ACT's active inference path and keep the unused CVAE encoder frozen."""
    trainable: list[tuple[str, nn.Parameter]] = []
    frozen_cvae: dict[str, torch.Tensor] = {}
    for name, param in policy.named_parameters():
        if is_unused_cvae_param(name):
            param.requires_grad_(False)
            frozen_cvae[name] = param.detach().cpu().clone()
        else:
            param.requires_grad_(True)
            trainable.append((name, param))
    if not trainable:
        raise ValueError("ACT policy exposed no trainable inference parameters.")
    return trainable, frozen_cvae


class SpatialSoftmaxBlockPoseHead(nn.Module):
    """Training-only 2D spatial-softmax head predicting 3D block centers from visual tokens."""

    def __init__(self, in_channels: int, num_keypoints: int = 32, out_dim: int = _AUX_POSE_DIM):
        super().__init__()
        if int(in_channels) < 1 or int(num_keypoints) < 1 or int(out_dim) < 1:
            raise ValueError("SpatialSoftmaxBlockPoseHead dimensions must be positive.")
        self.in_channels = int(in_channels)
        self.num_keypoints = int(num_keypoints)
        self.out_dim = int(out_dim)
        self.conv = nn.Conv2d(self.in_channels, self.num_keypoints, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_keypoints * 2, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_dim),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4 or feature_map.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B, {self.in_channels}, H, W] feature map, got {tuple(feature_map.shape)}."
            )
        logits = self.conv(feature_map)
        batch, keypoints, height, width = logits.shape
        probs = torch.softmax(logits.reshape(batch, keypoints, height * width), dim=-1).reshape(
            batch, keypoints, height, width
        )
        ys = torch.linspace(-1.0, 1.0, height, device=feature_map.device, dtype=probs.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=feature_map.device, dtype=probs.dtype)
        ey = (probs.sum(dim=-1) * ys).sum(dim=-1)
        ex = (probs.sum(dim=-2) * xs).sum(dim=-1)
        return self.mlp(torch.cat([ex, ey], dim=-1))


def build_act_optimizer(
    trainable_named: list[tuple[str, nn.Parameter]],
    aux_pose_head: SpatialSoftmaxBlockPoseHead | None,
    config: dict,
) -> torch.optim.AdamW:
    """Create AdamW with separate Transformer, visual-backbone, and aux-head LRs."""
    backbone_params = [param for name, param in trainable_named if "backbone" in name]
    other_params = [param for name, param in trainable_named if "backbone" not in name]
    groups = [
        {"params": other_params, "lr": float(config["act_lr"])},
        {"params": backbone_params, "lr": float(config["act_backbone_lr"])},
    ]
    if aux_pose_head is not None:
        groups.append({"params": list(aux_pose_head.parameters()), "lr": float(config["act_aux_pose_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(config["act_weight_decay"]))


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device | str) -> None:
    """Move AdamW moment buffers between GPU (update) and CPU (rollout/reset)."""
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if key != "step" and isinstance(value, torch.Tensor):
                state[key] = value.to(target, non_blocking=False)


def encode_rgb_anchor(image: np.ndarray) -> bytes:
    """Compress an HWC RGB frame to JPEG bytes for compact CPU rollout storage."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Anchor image must be an HWC RGB array.")
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(array[..., ::-1])
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode anchor RGB frame.")
    return encoded.tobytes()


def decode_rgb_anchor(payload: bytes) -> np.ndarray:
    """Decode JPEG anchor bytes into a CHW float32 [0, 1] 480x640 RGB image."""
    buf = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise RuntimeError("Failed to decode JPEG anchor RGB frame.")
    rgb = bgr[..., ::-1]
    if rgb.shape[0] != 480 or rgb.shape[1] != 640:
        rgb = cv2.resize(rgb, (640, 480))
    return np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)


@dataclass
class ActAnchor:
    episode_index: int
    step_index: int
    image_jpeg: bytes
    qpos_norm: np.ndarray       # float32 [ACTION_DIM]
    pred_old: np.ndarray        # float16 [chunk_size, ACTION_DIM]
    block_xyz: np.ndarray       # float32 [9]


@dataclass
class ActWaveData:
    """CPU-side anchor frames and per-episode executed action/advantage arrays."""

    anchors: list[ActAnchor]
    executed_norm: list[np.ndarray]    # per-episode float32 [T, ACTION_DIM]
    advantages: list[np.ndarray]       # per-episode float32 [T]


def concatenate_act_waves(waves: list[ActWaveData]) -> ActWaveData:
    """Merge per-wave anchor buffers across multiple waves in one update."""
    anchors: list[ActAnchor] = []
    executed_norm: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    offset = 0
    for wave in waves:
        for anchor in wave.anchors:
            anchors.append(
                ActAnchor(
                    episode_index=anchor.episode_index + offset,
                    step_index=anchor.step_index,
                    image_jpeg=anchor.image_jpeg,
                    qpos_norm=anchor.qpos_norm,
                    pred_old=anchor.pred_old,
                    block_xyz=anchor.block_xyz,
                )
            )
        executed_norm.extend(wave.executed_norm)
        advantages.extend(wave.advantages)
        offset += len(wave.executed_norm)
    return ActWaveData(anchors=anchors, executed_norm=executed_norm, advantages=advantages)


def _prepare_batch(
    anchors: list[ActAnchor],
    indices: list[int],
    executed_norm: list[np.ndarray],
    adv_norm: list[np.ndarray],
    chunk_size: int,
    beta: float,
    max_weight: float,
    weight_scale: float,
    device: torch.device,
):
    batch_size = len(indices)
    images = np.empty((batch_size, 1, 3, 480, 640), dtype=np.float32)
    qpos = np.empty((batch_size, ACTION_DIM), dtype=np.float32)
    targets = np.zeros((batch_size, chunk_size, ACTION_DIM), dtype=np.float32)
    pred_old = np.empty((batch_size, chunk_size, ACTION_DIM), dtype=np.float32)
    valid = np.zeros((batch_size, chunk_size), dtype=np.float32)
    weights = np.zeros((batch_size, chunk_size), dtype=np.float32)
    block_xyz = np.empty((batch_size, _AUX_POSE_DIM), dtype=np.float32)

    for row, idx in enumerate(indices):
        anchor = anchors[idx]
        images[row, 0] = decode_rgb_anchor(anchor.image_jpeg)
        qpos[row] = anchor.qpos_norm
        pred_old[row] = anchor.pred_old.astype(np.float32)
        block_xyz[row] = anchor.block_xyz
        ep_actions = executed_norm[anchor.episode_index]
        ep_adv = adv_norm[anchor.episode_index]
        start = anchor.step_index
        length = min(chunk_size, len(ep_actions) - start)
        if length > 0:
            targets[row, :length] = ep_actions[start : start + length]
            valid[row, :length] = 1.0
            scaled = np.clip(ep_adv[start : start + length] / beta, -20.0, 20.0)
            weights[row, :length] = np.minimum(np.exp(scaled), max_weight) * weight_scale

    return (
        torch.from_numpy(qpos).to(device),
        torch.from_numpy(images).to(device),
        torch.from_numpy(targets).to(device),
        torch.from_numpy(pred_old).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(weights).to(device),
        torch.from_numpy(block_xyz).to(device),
    )


def _warmup_aux_pose_head(
    policy: nn.Module,
    aux_pose_head: SpatialSoftmaxBlockPoseHead,
    anchors: list[ActAnchor],
    sample_indices: list[int],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    max_grad_norm: float,
    device: torch.device,
) -> None:
    """Warm up a fresh spatial-softmax head on detached backbone visual tokens."""
    if epochs < 1 or not sample_indices:
        return
    warmup_opt = torch.optim.AdamW(aux_pose_head.parameters(), lr=lr)
    amp_ctx = (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else nullcontext
    aux_pose_head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(sample_indices)).tolist()
        for start in range(0, len(perm), batch_size):
            batch_ids = [sample_indices[i] for i in perm[start : start + batch_size]]
            images = np.stack([decode_rgb_anchor(anchors[i].image_jpeg) for i in batch_ids])
            target_xyz = np.stack([anchors[i].block_xyz for i in batch_ids])
            img_t = torch.from_numpy(images).to(device)
            xyz_t = torch.from_numpy(target_xyz).to(device)
            with torch.no_grad(), amp_ctx():
                norm_img = normalize(img_t, _IMAGENET_MEAN, _IMAGENET_STD)
                feat, _ = policy.model.backbones[0](norm_img)
                visual_map = policy.model.input_proj(feat)
            pred_xyz = aux_pose_head(visual_map.detach().float())
            loss = F.smooth_l1_loss(pred_xyz, xyz_t, beta=0.05)
            if not torch.isfinite(loss):
                raise FloatingPointError("ACT aux visual pose warmup loss is non-finite.")
            warmup_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aux_pose_head.parameters(), max_grad_norm)
            warmup_opt.step()


def act_awr_update(
    policy: nn.Module,
    aux_pose_head: SpatialSoftmaxBlockPoseHead | None,
    optimizer: torch.optim.Optimizer,
    trainable_named: list[tuple[str, nn.Parameter]],
    frozen_cvae: dict[str, torch.Tensor],
    wave_data: ActWaveData,
    *,
    config: dict,
    device: torch.device,
    warmup_aux_epochs: int = 0,
) -> dict[str, float | int | bool | None]:
    """Run advantage-weighted action-chunk regression on unfrozen ACT."""
    if not wave_data.anchors:
        raise ValueError("ACT AWR update requires at least one anchor sample.")

    chunk_size = int(policy.model.num_queries)
    act_epochs = int(config["act_epochs"])
    batch_size = int(config["act_batch_size"])
    grad_accum = int(config["act_grad_accum_steps"])
    max_samples = int(config["act_max_samples"])
    beta = float(config["act_awr_beta"])
    max_weight = float(config["act_awr_max_weight"])
    prior_coef = float(config["act_prior_coef"])
    max_action_shift = float(config["act_max_action_shift"])
    max_grad_norm = float(config["act_max_grad_norm"])
    aux_pose_weight = float(config["act_aux_pose_weight"]) if aux_pose_head is not None else 0.0

    # Standardize advantages across all learning steps in the wave.
    all_adv = np.concatenate(wave_data.advantages).astype(np.float64)
    adv_mean = float(all_adv.mean())
    adv_std = float(all_adv.std()) + 1e-8
    adv_norm = [((ep - adv_mean) / adv_std).astype(np.float32) for ep in wave_data.advantages]

    total_anchors = len(wave_data.anchors)
    if total_anchors > max_samples:
        selected = torch.randperm(total_anchors)[:max_samples].tolist()
    else:
        selected = list(range(total_anchors))

    # Compute global AWR weight normalization & effective sample size on selected anchors.
    raw_weights_list = []
    for idx in selected:
        anchor = wave_data.anchors[idx]
        ep_adv = adv_norm[anchor.episode_index]
        length = min(chunk_size, len(ep_adv) - anchor.step_index)
        if length > 0:
            scaled = np.clip(ep_adv[anchor.step_index : anchor.step_index + length] / beta, -20.0, 20.0)
            raw_weights_list.append(np.minimum(np.exp(scaled), max_weight))
    if not raw_weights_list:
        raise ValueError("Selected ACT anchors contain no valid action steps.")
    raw_weights = np.concatenate(raw_weights_list).astype(np.float64)
    weight_mean = float(raw_weights.mean())
    weight_scale = 1.0 / weight_mean if weight_mean > 0.0 else 1.0
    norm_weights = raw_weights * weight_scale
    weight_max = float(norm_weights.max())
    weight_ess = float((norm_weights.sum() ** 2) / (len(norm_weights) * (norm_weights ** 2).sum()))

    before_params = [param.detach().cpu().clone() for _, param in trainable_named]
    act_params = [param for _, param in trainable_named]
    use_aux = aux_pose_head is not None and aux_pose_weight > 0.0
    aux_params = list(aux_pose_head.parameters()) if use_aux else []

    # Keep ACT in eval() so dropout stays off and FrozenBatchNorm2d stays fixed.
    policy.eval()
    if use_aux and warmup_aux_epochs > 0:
        _warmup_aux_pose_head(
            policy,
            aux_pose_head,
            wave_data.anchors,
            selected,
            epochs=warmup_aux_epochs,
            batch_size=batch_size,
            lr=float(config["act_aux_pose_lr"]),
            max_grad_norm=max_grad_norm,
            device=device,
        )

    captured_visual: dict[str, torch.Tensor | None] = {"map": None}

    def _capture_input_proj(_module, _inputs, output):
        captured_visual["map"] = output

    hook = policy.model.input_proj.register_forward_hook(_capture_input_proj) if use_aux else None
    amp_ctx = (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else nullcontext

    awr_loss_sum = prior_shift_sum = aux_loss_sum = aux_mae_sum = 0.0
    scored = 0
    updates = 0
    epochs_completed = 0
    early_stopped = False
    last_macro_shift: float | None = None
    grad_norms: list[float] = []

    # Keep optimizer state off GPU during forward/backward so activations and AdamW
    # moments never coexist in VRAM (saves ~1.3 GiB peak alongside Isaac workers).
    move_optimizer_state(optimizer, "cpu")
    try:
        if aux_pose_head is not None:
            aux_pose_head.train()
        optimizer.zero_grad(set_to_none=True)
        for _epoch in range(act_epochs):
            perm = torch.randperm(len(selected)).tolist()
            macro_batches = [
                [selected[i] for i in perm[start : start + batch_size * grad_accum]]
                for start in range(0, len(perm), batch_size * grad_accum)
            ]
            for macro in macro_batches:
                micro_batches = [
                    macro[offset : offset + batch_size]
                    for offset in range(0, len(macro), batch_size)
                ]
                macro_size = len(macro)
                macro_awr_sum = 0.0
                macro_shift_sum = 0.0
                macro_aux_loss_sum = 0.0
                macro_aux_mae_sum = 0.0
                optimizer.zero_grad(set_to_none=True)
                for micro in micro_batches:
                    qpos, images, targets, pred_old, valid, weights, block_xyz = _prepare_batch(
                        wave_data.anchors,
                        micro,
                        wave_data.executed_norm,
                        adv_norm,
                        chunk_size,
                        beta,
                        max_weight,
                        weight_scale,
                        device,
                    )
                    captured_visual["map"] = None
                    with amp_ctx():
                        pred = policy(qpos, images)
                        per_step_l1 = F.l1_loss(pred, targets, reduction="none").mean(dim=-1)
                        valid_sum = valid.sum().clamp_min(1.0)
                        awr_loss = (per_step_l1 * weights).sum() / valid_sum
                        shift_l1 = F.l1_loss(pred, pred_old, reduction="none").mean(dim=-1)
                        prior_loss = (shift_l1 * valid).sum() / valid_sum
                    loss = awr_loss + prior_coef * prior_loss
                    aux_loss_val = 0.0
                    aux_mae_val = 0.0
                    if aux_pose_head is not None and aux_pose_weight > 0.0:
                        visual_map = captured_visual["map"]
                        if visual_map is None:
                            raise RuntimeError("ACT input_proj hook did not capture visual tokens.")
                        pred_xyz = aux_pose_head(visual_map.float())
                        aux_loss = F.smooth_l1_loss(pred_xyz, block_xyz, beta=0.05)
                        loss = loss + aux_pose_weight * aux_loss
                        aux_loss_val = float(aux_loss.detach())
                        with torch.no_grad():
                            err = (pred_xyz.detach() - block_xyz).reshape(-1, len(BLOCK_LABELS), 3)
                            aux_mae_val = float(err.norm(dim=-1).mean())

                    if not torch.isfinite(loss):
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("ACT AWR loss is non-finite.")

                    micro_len = len(micro)
                    (loss * (micro_len / macro_size)).backward()
                    captured_visual["map"] = None
                    macro_awr_sum += float(awr_loss.detach()) * micro_len
                    macro_shift_sum += float(prior_loss.detach()) * micro_len
                    macro_aux_loss_sum += aux_loss_val * micro_len
                    macro_aux_mae_sum += aux_mae_val * micro_len
                    del qpos, images, targets, pred_old, valid, weights, block_xyz, pred, loss

                macro_shift = macro_shift_sum / macro_size
                last_macro_shift = macro_shift
                if macro_shift > max_action_shift:
                    optimizer.zero_grad(set_to_none=True)
                    early_stopped = True
                    break

                act_grad_norm = torch.nn.utils.clip_grad_norm_(act_params, max_grad_norm)
                if aux_params:
                    aux_grad_norm = torch.nn.utils.clip_grad_norm_(aux_params, max_grad_norm)
                else:
                    aux_grad_norm = torch.tensor(0.0, device=device)
                if not (torch.isfinite(act_grad_norm) and torch.isfinite(aux_grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("ACT AWR gradient norm is non-finite.")
                grad_norms.append(float(act_grad_norm.detach()))
                if device.type == "cuda":
                    move_optimizer_state(optimizer, device)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    move_optimizer_state(optimizer, "cpu")
                awr_loss_sum += macro_awr_sum
                prior_shift_sum += macro_shift_sum
                aux_loss_sum += macro_aux_loss_sum
                aux_mae_sum += macro_aux_mae_sum
                scored += macro_size
                updates += 1
            if early_stopped:
                break
            epochs_completed += 1
    finally:
        if hook is not None:
            hook.remove()
        captured_visual["map"] = None
        optimizer.zero_grad(set_to_none=True)
        move_optimizer_state(optimizer, "cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Verify unused CVAE params stayed frozen and updated ACT params stay finite.
    state_dict = policy.state_dict()
    for name, expected in frozen_cvae.items():
        if not torch.equal(state_dict[name].detach().cpu(), expected):
            raise RuntimeError(f"Frozen ACT CVAE parameter changed: {name}.")
    delta_sq = 0.0
    for (_, param), before in zip(trainable_named, before_params):
        after = param.detach().cpu()
        if not torch.isfinite(after).all():
            raise FloatingPointError("Unfrozen ACT parameter became non-finite.")
        diff = (after - before).to(torch.float64)
        delta_sq += float((diff * diff).sum())

    return {
        "act_samples": len(selected),
        "act_updates": updates,
        "act_epochs_completed": epochs_completed,
        "act_early_stopped": early_stopped,
        "act_awr_loss": (awr_loss_sum / scored) if scored else None,
        "act_prior_shift_l1": (prior_shift_sum / scored) if scored else last_macro_shift,
        "act_aux_pose_loss": (aux_loss_sum / scored) if (aux_pose_weight > 0.0 and scored) else None,
        "act_aux_pose_mae_m": (aux_mae_sum / scored) if (aux_pose_weight > 0.0 and scored) else None,
        "act_grad_norm": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "act_weight_max": weight_max,
        "act_weight_ess": weight_ess,
        "act_param_delta_l2": float(delta_sq ** 0.5),
    }
