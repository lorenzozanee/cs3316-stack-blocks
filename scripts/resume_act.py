#!/usr/bin/env python3
"""Continue a completed ACT run in place, using linear learning-rate annealing.

Run with the ACT Python environment, for example:
    python scripts/resume_act.py --checkpoint /path/to/policy_epoch_N_seed_S.ckpt \
        --additional-epochs 2000 --lr 1e-6 --lr-min 1e-7

The checkpoint must match training_state.pt and the end of both metrics files.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import pickle
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_history(output: Path, completed: int) -> dict:
    from starter.policies.act_metrics import CSV_COLUMNS, metric_row

    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError("Unexpected metrics.csv header.")
        rows = list(reader)
    if completed < 1 or [record["epoch"] for record in records] != list(range(1, completed + 1)):
        raise ValueError("metrics.jsonl must end at the training_state.pt epoch.")
    if len(rows) != len(records):
        raise ValueError("metrics.csv and metrics.jsonl must contain the same epochs.")
    for record, row in zip(records, rows):
        expected = metric_row(record)
        if any(row[key] != ("" if expected[key] is None else str(expected[key])) for key in CSV_COLUMNS):
            raise ValueError("metrics.csv and metrics.jsonl disagree.")
    for name in ("metrics.jsonl", "metrics.csv"):
        with (output / name).open("rb") as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                raise ValueError(f"{name} must end with a newline before appending.")
    return records[-1]


def resume_training(checkpoint: str | Path, additional_epochs: int, lr: float, lr_min: float) -> None:
    """Restore weights, optimizer and RNG, then append epochs to the same run."""
    if isinstance(additional_epochs, bool) or not isinstance(additional_epochs, int) or additional_epochs < 1:
        raise ValueError("additional_epochs must be a positive integer.")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite.")
    if not math.isfinite(lr_min) or not 0 < lr_min <= lr:
        raise ValueError("lr_min must be positive, finite and no greater than lr.")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = checkpoint.parent
    # Lock an existing file so two resume processes cannot append to one run.
    with (output / "metrics.jsonl").open("r+", encoding="utf-8") as metrics:
        metrics.seek(0, 2)
        try:
            fcntl.flock(metrics, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another resume process is using {output}.") from exc
        _resume(checkpoint, additional_epochs, lr, lr_min, metrics)


def _resume(checkpoint: Path, additional_epochs: int, lr: float, lr_min: float, metrics) -> None:
    import numpy as np
    import torch

    from starter.data.act import load_act_data
    from starter.policies.act import _run_epoch, _training_device, _validate_task, _vram_stats
    from starter.policies.act_metrics import CSV_COLUMNS, metric_row, plot_training_curves
    from starter.policies.act_model import ACTPolicy

    output = checkpoint.parent
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    _validate_task(config)
    state = torch.load(output / "training_state.pt", map_location="cpu", weights_only=False)
    completed = int(state["epoch"])
    seed = int(config["seed"])
    if checkpoint.name not in (f"policy_epoch_{completed}_seed_{seed}.ckpt", "policy_last.ckpt"):
        raise ValueError("Checkpoint epoch must match training_state.pt; historical optimizer state is unavailable.")
    previous = _load_history(output, completed)
    target = completed + additional_epochs
    config.update(num_epochs=target, lr=lr, lr_backbone=lr, lr_min=lr_min)
    save_freq = int(config["save_freq"])
    if save_freq < 1:
        raise ValueError("save_freq must be positive.")
    device = _training_device(config)
    third_party = str(ROOT / "third_party")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)
    data_path = Path(config["data_path"]).expanduser()
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    loader, computed_stats = load_act_data(
        data_path, int(config["batch_size"]), list(config["camera_names"]),
        num_workers=int(config["num_workers"]),
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if ([str(path) for path in loader.dataset.episode_paths] != metadata["train_episodes"]
            or len(loader.dataset) != int(config["num_episodes"])
            or len(loader) != metadata["steps_per_epoch"]):
        raise ValueError("Resume requires the original episodes and steps_per_epoch.")
    if loader.dataset.max_action_len < int(config["chunk_size"]):
        raise ValueError("The longest episode must contain at least chunk_size actions.")
    with (output / "dataset_stats.pkl").open("rb") as stream:
        stats = pickle.load(stream)
    for name in ("action_mean", "action_std", "qpos_mean", "qpos_std"):
        if not np.array_equal(stats[name], computed_stats[name]):
            raise ValueError(f"Dataset normalization changed: {name}.")
    loader.dataset.norm_stats = stats
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = ACTPolicy(config, pretrained_backbone=False).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    optimizer = policy.configure_optimizers(config)
    optimizer.load_state_dict(state["optimizer"])
    # Keep Adam moments. First resumed epoch uses lr, final epoch uses lr_min.
    # A single additional epoch uses lr because it has no annealing interval.
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = lr
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=lr_min / lr,
        total_iters=max(additional_epochs - 1, 1),
    )
    torch.set_rng_state(state["torch_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    if device.type == "cuda":
        if state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"], device)
        torch.cuda.reset_peak_memory_stats(device)
    del state
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"ACT resume: checkpoint={checkpoint}; epochs={completed + 1}..{target}; "
          f"linear_lr={lr:.3e}->{lr_min:.3e} (main and backbone); output={output}", flush=True)
    global_step = previous["global_step"]
    elapsed_before = previous["elapsed_seconds"]
    started = time.monotonic()
    with (output / "metrics.csv").open("a", newline="", encoding="utf-8") as table:
        writer = csv.DictWriter(table, fieldnames=CSV_COLUMNS)
        for epoch in range(completed + 1, target + 1):
            epoch_started = time.monotonic()
            policy.train()
            training = _run_epoch(policy, loader, device, optimizer)
            epoch_seconds = time.monotonic() - epoch_started
            session_elapsed = time.monotonic() - started
            elapsed = elapsed_before + session_elapsed
            remaining = session_elapsed / (epoch - completed) * (target - epoch)
            global_step += len(loader)
            vram = _vram_stats(device)
            record = {
                "epoch": epoch, "global_step": global_step, "train": training,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_backbone": optimizer.param_groups[1]["lr"],
                "epoch_seconds": epoch_seconds, "elapsed_seconds": elapsed,
                "eta_seconds": remaining, "vram": vram,
            }
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            writer.writerow(metric_row(record))
            table.flush()
            print(f"epoch {epoch}/{target}: loss={training['loss']:.6f} "
                  f"l1={training['l1']:.6f} kl={training['kl']:.6f} | "
                  f"lr={record['lr']:.3e} lr_backbone={record['lr_backbone']:.3e} | "
                  f"epoch_time={epoch_seconds:.1f}s elapsed={timedelta(seconds=int(elapsed))} "
                  f"ETA={timedelta(seconds=int(remaining))}", flush=True)
            if epoch % save_freq == 0:
                torch.save(policy.state_dict(), output / f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            if epoch == completed + 1 or epoch % save_freq == 0 or epoch == target:
                plot_training_curves(output)
            if epoch < target:
                scheduler.step()
        torch.save(policy.state_dict(), output / "policy_last.ckpt")
        torch.save({
            "epoch": target,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, output / "training_state.pt")
    print(f"ACT resume complete: epoch={target}; checkpoint={output / 'policy_last.ckpt'}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--additional-epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True, help="Initial LR for both main model and backbone.")
    parser.add_argument("--lr-min", type=float, required=True, help="Final LR after linear annealing.")
    args = parser.parse_args(argv)
    resume_training(args.checkpoint, args.additional_epochs, args.lr, args.lr_min)


if __name__ == "__main__":
    main()
