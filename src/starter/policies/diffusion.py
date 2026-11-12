"""Offline flow-matching training for the image-conditioned stacking policy."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from bisect import bisect_right
from collections import OrderedDict
from itertools import accumulate
from pathlib import Path

from starter.config import require
from starter.data.demos import episode_files, pack_ee_group
from starter.interfaces import ACTION_DIM, HEAD_CAMERA
from starter.paths import dp_checkpoint_dir

CAMERA_KEY = f"vision/{HEAD_CAMERA}/colors"


class FlowMatchingWindows:
    """Episode-local observation history and action chunks from HDF5 episodes.

    State and action tables are loaded once. Camera frames stay on disk and are
    decoded per item; every DataLoader worker keeps at most eight open files.
    """

    max_open_files = 8

    def __init__(self, paths, horizon, image_size, n_obs_steps, decode_image, augment=False):
        import h5py
        import numpy as np
        import torch

        from starter.data.diffusion import preprocess_image
        from starter.policies.diffusion_model import FlowMatchingPolicy

        paths = [Path(path) for path in paths]
        if not paths:
            raise ValueError("No demonstration episodes were provided.")
        self.paths = paths
        self.horizon = horizon
        self.image_size = image_size
        self.n_obs_steps = n_obs_steps
        self.decode_image = decode_image
        self.augment = augment
        self._preprocess = preprocess_image
        self._transforms = None
        self._handles = OrderedDict()

        states, actions, lengths = [], [], []
        for path in paths:
            with h5py.File(path, "r") as handle:
                if CAMERA_KEY not in handle:
                    raise ValueError(f"{path}: missing {CAMERA_KEY}.")
                state = np.asarray(pack_ee_group(handle["state"]), dtype=np.float32)
                action = np.asarray(pack_ee_group(handle["action"]), dtype=np.float32)
                frames = len(handle[CAMERA_KEY])
            if frames == 0 or state.shape != (frames, ACTION_DIM) or action.shape != state.shape:
                raise ValueError(f"{path}: empty or mismatched image, state or action shapes.")
            if not (np.isfinite(state).all() and np.isfinite(action).all()):
                raise ValueError(f"{path}: non-finite state or action values.")
            with torch.no_grad():
                state = FlowMatchingPolicy.canonicalize(torch.from_numpy(state)).numpy()
                action = FlowMatchingPolicy.canonicalize(torch.from_numpy(action)).numpy()
            states.append(state)
            actions.append(action)
            lengths.append(frames)

        self.states = states
        self.actions = actions
        self.lengths = lengths
        self.ends = list(accumulate(lengths))

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        import numpy as np
        import torch

        episode = bisect_right(self.ends, index)
        start = self.ends[episode - 1] if episode else 0
        t = index - start
        end = self.lengths[episode]
        history = np.clip(np.arange(t - self.n_obs_steps + 1, t + 1), 0, end - 1)
        steps = np.minimum(np.arange(t, t + self.horizon), end - 1)

        handle = self._handle(self.paths[episode])
        frames = np.empty((self.n_obs_steps, 3, self.image_size, self.image_size), dtype=np.uint8)
        for slot, frame in enumerate(history):
            encoded = handle[CAMERA_KEY][int(frame)]
            frames[slot] = self._preprocess(self.decode_image(encoded), self.image_size)
        images = frames
        if self.augment:
            images = self._augment(torch.from_numpy(frames)).numpy()
        return images, self.states[episode][history], self.actions[episode][steps], min(self.horizon, end - t)

    def bounds(self):
        import numpy as np

        state = np.concatenate(self.states)
        action = np.concatenate(self.actions)
        return {
            "state_min": state.min(axis=0),
            "state_max": state.max(axis=0),
            "action_min": action.min(axis=0),
            "action_max": action.max(axis=0),
        }

    def _augment(self, images):
        if self._transforms is None:
            from torchvision.transforms import v2

            self._transforms = v2.Compose(
                (
                    v2.RandomCrop(self.image_size, padding=4, padding_mode="edge"),
                    v2.ColorJitter(0.1, 0.1, 0.1),
                )
            )
        return self._transforms(images)

    def _handle(self, path):
        import h5py

        handle = self._handles.get(path)
        if handle is not None:
            self._handles.move_to_end(path)
            return handle
        handle = h5py.File(path, "r")
        self._handles[path] = handle
        while len(self._handles) > self.max_open_files:
            _, stale = self._handles.popitem(last=False)
            stale.close()
        return handle

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handles = OrderedDict()

    def __del__(self):
        handles = getattr(self, "_handles", None)
        if handles is not None:
            for handle in handles.values():
                handle.close()
            handles.clear()


def train_offline(config: dict) -> Path:
    """Train the flow-matching policy from scratch and return ``last.ckpt``."""
    root = Path(require(config, "data_path"))
    episodes = episode_files(root)
    if not episodes:
        raise ValueError(
            f"No HDF5 episodes found under {root} (expected data/episode_*.hdf5). "
            "Place the stack_blocks demonstrations there before training."
        )

    import matplotlib
    import numpy as np
    import torch
    from torch.optim.swa_utils import AveragedModel
    from torch.utils.data import DataLoader, Subset

    from starter.policies.diffusion_model import FlowMatchingPolicy

    matplotlib.use("Agg")

    try:
        from XPolicyLab.utils.process_data import decode_image_bit
    except ModuleNotFoundError as exc:
        if exc.name != "XPolicyLab":
            raise
        raise RuntimeError("Install the pinned XPolicyLab with scripts/bootstrap_third_party.sh.") from exc

    options = config.get("diffusion", {})
    seed = int(require(config, "seed"))
    epochs = _positive_int(options, "epochs", 100)
    image_size = _positive_int(options, "image_size", 224)
    horizon = _positive_int(options, "horizon", 16)
    n_obs_steps = _positive_int(options, "n_obs_steps", 2)
    action_steps = _positive_int(options, "action_steps", 4)
    batch_size = _positive_int(options, "batch_size", 16)
    num_workers = _positive_int(options, "num_workers", 4, minimum=0)
    inference_steps = _positive_int(options, "inference_steps", 20)
    checkpoint_every = _positive_int(options, "checkpoint_every", 10)
    probe_windows = _positive_int(options, "probe_windows", 64)
    probe_every = _positive_int(options, "probe_every", 10)
    lr = _positive_float(options, "lr", 1e-4)
    vision_lr = _positive_float(options, "vision_lr", 1e-4)
    weight_decay = _finite_float(options, "weight_decay", 0.01, minimum=0.0)
    dropout = float(options.get("dropout", 0.1))
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError(f"diffusion.dropout must be in [0, 1), got {dropout}.")
    if action_steps > horizon:
        raise ValueError("diffusion.action_steps must not exceed horizon.")

    policy_config = {
        "horizon": horizon,
        "image_size": image_size,
        "inference_steps": inference_steps,
        "n_obs_steps": n_obs_steps,
        "action_steps": action_steps,
        "vision_dim": _positive_int(options, "vision_dim", 384),
        "vision_depth": _positive_int(options, "vision_depth", 12),
        "vision_heads": _positive_int(options, "vision_heads", 6),
        "model_dim": _positive_int(options, "model_dim", 512),
        "depth": _positive_int(options, "depth", 12),
        "num_heads": _positive_int(options, "num_heads", 8),
        "dropout": dropout,
    }

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_data = FlowMatchingWindows(
        episodes, horizon, image_size, n_obs_steps, decode_image_bit, augment=True,
    )
    probe_data = FlowMatchingWindows(
        episodes, horizon, image_size, n_obs_steps, decode_image_bit, augment=False,
    )
    probe_count = min(len(probe_data), probe_windows)
    probe_indices = torch.linspace(0, len(probe_data) - 1, probe_count).round().long().tolist()

    device = torch.device(
        f"cuda:{int(config.get('policy_gpu', 0))}" if torch.cuda.is_available() else "cpu"
    )
    loader_generator = torch.Generator().manual_seed(seed)
    loader_options = {"batch_size": batch_size, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
        **loader_options,
    )
    probe_loader = DataLoader(
        Subset(probe_data, probe_indices), shuffle=False, num_workers=0, **loader_options,
    )

    model = FlowMatchingPolicy(**policy_config, bounds=train_data.bounds()).to(device)
    optimizer = _make_optimizer(model, lr, vision_lr, weight_decay)
    updates_per_epoch = len(train_loader)
    total_updates = updates_per_epoch * epochs
    warmup_steps = max(1, round(0.05 * total_updates))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _learning_rate_factor(step, warmup_steps, total_updates)
    )
    ema = AveragedModel(model, multi_avg_fn=_ema_average, use_buffers=True)
    amp_dtype = None
    scaler = None
    if device.type == "cuda":
        with torch.cuda.device(device):
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
        if amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")

    train_config = {
        "data_path": str(root),
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "vision_lr": vision_lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "total_updates": total_updates,
        "amp_dtype": None if amp_dtype is None else str(amp_dtype),
        "ema_decay": 0.999,
        "ema_warmup": True,
        "checkpoint_every": checkpoint_every,
        "probe_every": probe_every,
        "from_scratch": True,
        "episodes": [str(path) for path in episodes],
        "windows": len(train_data),
        "probe_windows": probe_count,
        "probe_indices": probe_indices,
    }

    checkpoint_dir = dp_checkpoint_dir(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    run_dir = checkpoint_dir.parent
    last_checkpoint = checkpoint_dir / "last.ckpt"
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} params={total / 1e6:.2f}M trainable={trainable / 1e6:.2f}M "
        f"windows={len(train_data)} samples={len(train_data) * epochs} "
        f"updates={total_updates} amp={amp_dtype}",
        flush=True,
    )

    metrics_path = run_dir / "metrics.csv"
    steps_path = run_dir / "steps.csv"
    started = time.monotonic()
    samples_seen = 0
    global_step = 0
    with metrics_path.open("w", newline="") as metrics, steps_path.open("w", newline="") as steps:
        metrics_writer = csv.writer(metrics)
        steps_writer = csv.writer(steps)
        metrics_writer.writerow((
            "epoch", "train_loss", "train_probe_action_mae", "lr", "grad_norm", "samples", "elapsed_s",
        ))
        steps_writer.writerow(("epoch", "step", "loss", "grad_norm", "lr", "samples"))
        for epoch in range(1, epochs + 1):
            train_loss, grad_norm, records = _train_epoch(
                model, ema, train_loader, optimizer, lr_scheduler, scaler, amp_dtype, device, seed, epoch,
            )
            for loss_value, step_norm, step_lr, batch in records:
                global_step += 1
                samples_seen += batch
                steps_writer.writerow((
                    epoch, global_step, _format(loss_value), _format(step_norm),
                    _format(step_lr), samples_seen,
                ))
            steps.flush()
            probe_mae = None
            if epoch == 1 or epoch == epochs or epoch % probe_every == 0:
                probe_mae = _probe_action_mae(ema.module, probe_loader, device, seed)
            current_lr = lr_scheduler.get_last_lr()[0]
            metrics_writer.writerow((
                epoch,
                _format(train_loss),
                "" if probe_mae is None else _format(probe_mae),
                _format(current_lr),
                _format(grad_norm),
                samples_seen,
                _format(time.monotonic() - started),
            ))
            metrics.flush()
            print(
                f"epoch={epoch}/{epochs} train_loss={train_loss:.5f} "
                f"probe_action_mae={'n/a' if probe_mae is None else f'{probe_mae:.5f}'} "
                f"grad_norm={_format(grad_norm) or 'n/a'} lr={current_lr:.3g}",
                flush=True,
            )
            if epoch == 1 or epoch % checkpoint_every == 0 or epoch == epochs:
                _atomic_save(
                    _stage_checkpoint(model, ema, policy_config, train_config, epoch),
                    checkpoint_dir / f"epoch_{epoch:04d}.ckpt",
                )
                _atomic_save(
                    _last_checkpoint(
                        model, ema, optimizer, lr_scheduler, policy_config, train_config,
                        epoch, loader_generator, scaler,
                    ),
                    last_checkpoint,
                )
    plots_dir = _plot_metrics(run_dir)
    print(f"curves={plots_dir}", flush=True)
    print(f"checkpoint={last_checkpoint}", flush=True)
    return last_checkpoint


def train_online(config: dict) -> Path:
    """Fine-tune the pretrained flow policy with FPO++ and a frozen visual encoder."""
    from starter.policies.fpo import train_fpo

    return train_fpo(config)


def _positive_int(options, key, default, minimum=1):
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"diffusion.{key} must be an integer >= {minimum}, got {value!r}.")
    return value


def _finite_float(options, key, default, minimum):
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"diffusion.{key} must be a number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"diffusion.{key} must be finite and >= {minimum}, got {value!r}.")
    return value


def _positive_float(options, key, default):
    value = _finite_float(options, key, default, 0.0)
    if value == 0.0:
        raise ValueError(f"diffusion.{key} must be positive.")
    return value


def _learning_rate_factor(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _make_optimizer(model, lr, vision_lr, weight_decay):
    import torch

    vision_ids = {
        id(parameter) for name, parameter in model.named_parameters() if name.startswith("vision.")
    }
    groups = {}

    def add(group_lr, parameter):
        group_decay = weight_decay if parameter.ndim >= 2 else 0.0
        groups.setdefault((group_lr, group_decay), []).append(parameter)

    # Insert the non-vision group first so group 0 always carries ``lr``.
    for use_vision in (False, True):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or (id(parameter) in vision_ids) != use_vision:
                continue
            add(vision_lr if use_vision else lr, parameter)
    return torch.optim.AdamW(
        [
            {"params": parameters, "lr": group_lr, "weight_decay": group_decay}
            for (group_lr, group_decay), parameters in groups.items()
        ]
    )


def _seed_worker(worker_id):
    import numpy as np
    import torch

    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _ema_average(averaged, current, num_averaged):
    from torch.optim.swa_utils import get_ema_multi_avg_fn

    updates = int(num_averaged)
    decay = min(0.999, (1 + updates) / (10 + updates))
    get_ema_multi_avg_fn(decay)(averaged, current, num_averaged)


def _train_epoch(model, ema, loader, optimizer, lr_scheduler, scaler, amp_dtype, device, seed, epoch):
    import torch

    model.train()
    generator = torch.Generator(device=device).manual_seed(seed + epoch)
    total = 0.0
    count = 0
    records = []
    for images, states, actions, valid_steps in loader:
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        valid_steps = valid_steps.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = model.loss(images, states, actions, valid_steps, generator=generator)
        if not torch.isfinite(loss):
            raise FloatingPointError("Flow-matching loss is not finite.")
        if scaler is None:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True,
            )
            optimizer.step()
            lr_scheduler.step()
            ema.update_parameters(model)
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                lr_scheduler.step()
                ema.update_parameters(model)
        records.append((
            loss.detach().item(), float(grad_norm), lr_scheduler.get_last_lr()[0], len(actions),
        ))
        total += loss.detach().item() * len(actions)
        count += len(actions)
    average = total / count
    if not math.isfinite(average):
        raise FloatingPointError("Flow-matching epoch loss is not finite.")
    finite_norms = [norm for _, norm, _, _ in records if math.isfinite(norm)]
    mean_norm = sum(finite_norms) / len(finite_norms) if finite_norms else float("nan")
    return average, mean_norm, records


def _probe_action_mae(model, loader, device, seed):
    import torch

    model.eval()
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        for images, states, actions, valid_steps in loader:
            images = images.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            valid_steps = valid_steps.to(device, non_blocking=True)
            prediction = model.sample_actions(images, states, generator=generator)
            error = (
                model.normalize(prediction, "action") - model.normalize(actions, "action")
            ).abs().mean(dim=-1)
            valid = torch.arange(prediction.shape[1], device=device)[None] < valid_steps[:, None]
            total += (error * valid).sum()
            count += valid.sum()
    average = (total / count).item()
    if not math.isfinite(average):
        raise FloatingPointError("Flow-matching probe action error is not finite.")
    return average


def _format(value):
    return f"{value:.8g}" if math.isfinite(value) else ""


def _plot_metrics(run_dir):
    """Write report-ready loss, action-error, and learning-rate curves."""
    import matplotlib.pyplot as plt
    import numpy as np

    def read(path):
        return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))

    def line(x, y, color, title, xlabel, ylabel, path, smooth=0):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        if smooth and len(y) > smooth:
            ax.plot(x, y, color=color, linewidth=1, alpha=0.25, label="Per update")
            mean = np.convolve(y, np.ones(smooth) / smooth, mode="valid")
            ax.plot(x[smooth - 1:], mean, color=color, linewidth=2,
                    label=f"{smooth}-update mean")
            ax.legend(frameon=False)
        else:
            marker = {"marker": "o", "markersize": 4} if len(y) <= 20 else {}
            ax.plot(x, y, color=color, linewidth=2, **marker)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    colors = {"loss": "#2a78d6", "mae": "#eb6834", "lr": "#52514e"}
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    metrics = read(run_dir / "metrics.csv")
    steps = read(run_dir / "steps.csv")
    line(metrics["epoch"], metrics["train_loss"], colors["loss"],
         "Flow-matching training loss", "Epoch", "MSE (normalized velocity)", plots / "loss_curve.png")
    line(steps["step"], steps["loss"], colors["loss"],
         "Flow-matching training loss per update", "Update", "MSE (normalized velocity)",
         plots / "loss_steps.png", smooth=min(len(steps), max(20, len(steps) // 50)))
    measured = np.isfinite(metrics["train_probe_action_mae"])
    line(metrics["epoch"][measured], metrics["train_probe_action_mae"][measured], colors["mae"],
         "Train-probe action error", "Epoch", "Normalized MAE (train sample, not validation)",
         plots / "probe_action_mae.png")
    line(steps["step"], steps["lr"], colors["lr"],
         "Learning rate", "Update", "Learning rate", plots / "learning_rate.png")
    return plots


def _stage_checkpoint(model, ema, policy_config, train_config, epoch):
    return {
        "model": {name: value.detach().cpu() for name, value in ema.module.state_dict().items()},
        "policy": policy_config,
        "epoch": epoch,
        "train_config": train_config,
    }


def _last_checkpoint(model, ema, optimizer, lr_scheduler, policy_config, train_config, epoch, generator, scaler):
    payload = _stage_checkpoint(model, ema, policy_config, train_config, epoch)
    payload.update(
        {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "raw_model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "ema_n_averaged": int(ema.n_averaged),
            "rng": _rng_state(generator),
        }
    )
    return payload


def _rng_state(loader_generator):
    import numpy as np
    import torch

    name, keys, position, has_gauss, cached = np.random.get_state()
    state = {
        "torch": torch.get_rng_state(),
        "numpy": (name, keys.tolist(), int(position), int(has_gauss), float(cached)),
        "random": random.getstate(),
        "loader": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = list(torch.cuda.get_rng_state_all())
    return state


def _atomic_save(payload, path):
    import torch

    pending = path.with_name(path.name + ".tmp")
    torch.save(payload, pending)
    os.replace(pending, path)
