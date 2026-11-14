"""Offline ACT behavior cloning and the separate online training entry."""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from datetime import timedelta
from pathlib import Path

from starter.config import require
from starter.data.demos import episode_files
from starter.paths import REPO_ROOT, act_output_dir


def train_offline(config: dict) -> None:
    """Train the baseline ACT architecture/loss on every available ee demo.

    There is no held-out split. Periodic policies support selecting a base by
    simulator scores; policy_last.ckpt always contains the final training epoch.
    """
    _validate_task(config)
    data_path = Path(require(config, "data_path")).expanduser()
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    files = episode_files(data_path)
    if not files:
        raise FileNotFoundError(f"No ACT episodes found under {data_path}; download the real demonstrations first.")
    expected = int(require(config, "num_episodes"))
    if len(files) != expected:
        raise ValueError(f"Expected all {expected} ACT episodes, found {len(files)} under {data_path}.")

    import numpy as np
    import torch
    import torchvision

    from starter.data.act import load_act_data
    from starter.policies.act_model import ACTPolicy
    from starter.policies.act_metrics import CSV_COLUMNS, metric_row, plot_training_curves

    device = _training_device(config)
    epochs = int(require(config, "num_epochs"))
    save_freq = int(require(config, "save_freq"))
    if epochs < 1 or save_freq < 1:
        raise ValueError("num_epochs and save_freq must be positive.")
    lr_min = float(require(config, "lr_min"))
    if not 0 <= lr_min <= min(float(require(config, "lr")), float(require(config, "lr_backbone"))):
        raise ValueError("lr_min must be between zero and both initial learning rates.")
    seed = int(require(config, "seed"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    third_party = str(REPO_ROOT / "third_party")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)
    train_loader, stats = load_act_data(
        data_path, int(require(config, "batch_size")), list(require(config, "camera_names")),
        num_workers=int(require(config, "num_workers")),
    )
    if train_loader.dataset.max_action_len < int(require(config, "chunk_size")):
        raise ValueError("The longest episode must contain at least chunk_size actions.")

    # Model initialization is independent of the dataset constructor's RNG use.
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Explicit FP32, with neither AMP nor TF32, for the baseline recipe.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = ACTPolicy(config, pretrained_backbone=bool(require(config, "pretrained_backbone"))).to(device)
    optimizer = policy.configure_optimizers(config)
    # Epoch 1 uses the initial rates; the final epoch uses lr_min. A single
    # epoch keeps the initial rates because there is no decay interval.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - 1, 1), eta_min=lr_min,
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())

    output = act_output_dir(config)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = dict(config, data_path=str(data_path))
    (output / "config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    with (output / "dataset_stats.pkl").open("wb") as stream:
        pickle.dump(stats, stream)
    metadata = {
        "parameters": parameter_count,
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "train_episodes": [str(p) for p in train_loader.dataset.episode_paths],
        "steps_per_epoch": len(train_loader),
        "normalization": "all episodes, last-frame padding, sample std clipped at 0.01",
        "data_usage": "all demonstrations for training; no held-out split",
        "baseline_pins": (REPO_ROOT / "third_party/pins.lock").read_text(encoding="utf-8"),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"ACT: {parameter_count:,} parameters; output={output}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with (
        (output / "metrics.jsonl").open("x", encoding="utf-8") as metrics,
        (output / "metrics.csv").open("x", newline="", encoding="utf-8") as table,
    ):
        csv_writer = csv.DictWriter(table, fieldnames=CSV_COLUMNS)
        csv_writer.writeheader()
        for epoch in range(epochs):
            epoch_started = time.monotonic()
            policy.train()
            training = _run_epoch(policy, train_loader, device, optimizer)
            elapsed = time.monotonic() - started
            epoch_seconds = time.monotonic() - epoch_started
            remaining = elapsed / (epoch + 1) * (epochs - epoch - 1)
            vram = _vram_stats(device)
            record = {"epoch": epoch + 1, "global_step": (epoch + 1) * len(train_loader),
                      "train": training, "lr": optimizer.param_groups[0]["lr"],
                      "lr_backbone": optimizer.param_groups[1]["lr"],
                      "epoch_seconds": epoch_seconds, "elapsed_seconds": elapsed,
                      "eta_seconds": remaining, "vram": vram}
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            csv_writer.writerow(metric_row(record))
            table.flush()
            memory = "VRAM=CPU"
            if vram is not None:
                memory = (f"VRAM allocated={vram['allocated_gib']:.2f}GiB "
                          f"reserved={vram['reserved_gib']:.2f}GiB "
                          f"peak={vram['peak_allocated_gib']:.2f}GiB")
            print(f"epoch {epoch + 1}/{epochs}: loss={training['loss']:.6f} "
                  f"l1={training['l1']:.6f} kl={training['kl']:.6f} | "
                  f"lr={record['lr']:.3e} lr_backbone={record['lr_backbone']:.3e} | "
                  f"epoch_time={epoch_seconds:.1f}s "
                  f"elapsed={timedelta(seconds=int(elapsed))} "
                  f"ETA={timedelta(seconds=int(remaining))} | {memory}", flush=True)
            if (epoch + 1) % save_freq == 0:
                torch.save(policy.state_dict(), output / f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            if epoch == 0 or (epoch + 1) % save_freq == 0 or epoch + 1 == epochs:
                plot_training_curves(output)
            if epoch + 1 < epochs:
                scheduler.step()
        torch.save(policy.state_dict(), output / "policy_last.ckpt")
        # Preserve optimizer, scheduler and RNG separately from policy weights.
        torch.save({
            "epoch": epochs,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, output / "training_state.pt")
    print(f"Saved ACT offline checkpoint: {output / 'policy_last.ckpt'}", flush=True)


def _validate_task(config: dict) -> None:
    for key, expected in (("data_task", "stack_blocks"), ("env_cfg", "arx_x5"), ("action_type", "ee")):
        if require(config, key) != expected:
            raise ValueError(f"ACT offline training requires {key}={expected!r}.")
    if require(config, "camera_names") != ["cam_head"]:
        raise ValueError("ACT offline training requires only camera_names: [cam_head].")


def _training_device(config: dict):
    import torch

    selected = str(require(config, "device"))
    if selected == "cuda":
        device = torch.device("cuda", int(require(config, "policy_gpu")))
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device}, but that CUDA device is unavailable.")
        torch.cuda.set_device(device)
        try:
            # Availability checks pass on GPUs the installed CUDA build has no kernels for.
            torch.arange(4, dtype=torch.float32, device=device).add_(1).sum()
            torch.cuda.synchronize(device)
        except RuntimeError as exc:
            cuda_build = torch.version.cuda or "unknown CUDA build"
            raise RuntimeError(
                f"CUDA device {device} failed a basic kernel probe with torch {torch.__version__} "
                f"(CUDA {cuda_build}). Install the torch build for this GPU as documented in README.md."
            ) from exc
        return device
    if selected == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be 'cuda' or 'cpu'; select the CUDA index using policy_gpu.")


def _run_epoch(policy, loader, device, optimizer) -> dict[str, float]:
    import torch

    totals = {"l1": 0.0, "kl": 0.0, "loss": 0.0}
    batches = 0
    optimizer.zero_grad()
    for image, qpos, actions, is_pad in loader:
        losses = policy(qpos.to(device), image.to(device), actions.to(device), is_pad.to(device))
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise FloatingPointError("ACT produced a non-finite loss.")
        losses["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        for key in totals:
            totals[key] += float(losses[key].detach())
        batches += 1
    if not batches:
        raise ValueError("ACT dataloader must contain at least one batch.")
    # Upstream averages batch means equally, including the partial last batch.
    return {key: value / batches for key, value in totals.items()}


def _vram_stats(device) -> dict[str, float] | None:
    import torch

    if device.type != "cuda":
        return None
    gib = 1024 ** 3
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
        "total_gib": torch.cuda.get_device_properties(device).total_memory / gib,
    }


def load_offline_policy(checkpoint: str | Path, device="cpu"):
    """Load a saved policy without downloading ImageNet weights or building an optimizer."""
    import torch

    from starter.policies.act_model import ACTPolicy

    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    policy = ACTPolicy(config, pretrained_backbone=False)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return policy.to(device).eval()


def train_online(config: dict) -> None:
    """Learn a residual on the frozen policy in ``stack_blocks``.

    ``scripts/train_act_residual.sh`` has already started Isaac in the simulator
    image. Connect with ``starter.envs.remote.RemoteEnv``. The action is the
    frozen action plus a residual. Keep the environment score, and do not feed
    privileged block poses to the policy.
    """
    del config
    raise NotImplementedError("实现冻结基座上的残差在线训练。")
