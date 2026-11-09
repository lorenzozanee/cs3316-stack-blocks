"""Offline training for the image-conditioned Diffusion Policy."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from starter.config import require
from starter.data.demos import episode_files
from starter.paths import dp_checkpoint_dir


def train_offline(config: dict) -> Path:
    """Train on stack_blocks demonstrations and return the best checkpoint."""
    root = Path(require(config, "data_path"))
    episodes = episode_files(root)
    if len(episodes) < 2:
        raise ValueError(f"Need at least two HDF5 episodes in {root}/data for an episode-level split.")

    import torch
    from diffusers.optimization import get_cosine_schedule_with_warmup
    from diffusers.training_utils import EMAModel
    from torch.utils.data import DataLoader, Subset
    from starter.data.diffusion import ActionWindows
    from starter.policies.diffusion_model import DiffusionPolicy

    try:
        from XPolicyLab.utils.process_data import decode_image_bit
    except ModuleNotFoundError as exc:
        if exc.name != "XPolicyLab":
            raise
        raise RuntimeError("Install the pinned XPolicyLab with scripts/bootstrap_third_party.sh.") from exc

    options = config.get("diffusion", {})
    seed = int(require(config, "seed"))
    horizon = int(options.get("horizon", 16))
    image_size = int(options.get("image_size", 96))
    epochs = int(options.get("epochs", 20))
    batch_size = int(options.get("batch_size", 32))
    num_workers = int(options.get("num_workers", 0))
    learning_rate = float(options.get("lr", 3e-4))
    train_timesteps = 100
    inference_steps = int(options.get("inference_steps", 20))
    if (min(horizon, image_size, epochs, batch_size) < 1 or num_workers < 0
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Diffusion training sizes and lr must be positive; num_workers must be nonnegative.")
    if not 2 <= inference_steps <= train_timesteps:
        raise ValueError(f"inference_steps must be between 2 and {train_timesteps}.")

    torch.manual_seed(seed)
    random.Random(seed).shuffle(episodes)
    n_val = max(1, len(episodes) // 10)
    print(f"Preparing RGB frames for {len(episodes)} demonstration episodes...", flush=True)
    train = ActionWindows(episodes[n_val:], horizon, image_size, decode_image_bit)
    val = ActionWindows(episodes[:n_val], horizon, image_size, decode_image_bit)
    device = torch.device(f"cuda:{require(config, 'policy_gpu')}" if torch.cuda.is_available() else "cpu")
    loader_options = dict(batch_size=batch_size, pin_memory=device.type == "cuda")
    train_loader = DataLoader(
        train, shuffle=True, num_workers=num_workers, persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed), **loader_options,
    )
    val_loader = DataLoader(val, **loader_options)
    sample_indices = torch.linspace(0, len(val) - 1, min(len(val), 256), dtype=torch.float64).long()
    sample_loader = DataLoader(Subset(val, sample_indices.tolist()), **loader_options)
    policy_config = {
        "horizon": horizon,
        "image_size": image_size,
        "inference_steps": inference_steps,
        "scheduler_config": {
            "num_train_timesteps": train_timesteps,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
            "timestep_spacing": "linspace",
        },
    }
    model = DiffusionPolicy(**policy_config, bounds=train.bounds()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps = len(train_loader) * epochs
    warmup_steps = min(500, steps // 20)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps)
    ema = EMAModel(model.parameters(), decay=0.999, use_ema_warmup=True, foreach=True)
    train_config = {
        "data_path": str(root), "seed": seed, "device": str(device), "epochs": epochs,
        "batch_size": batch_size, "num_workers": num_workers, "lr": learning_rate,
        "warmup_steps": warmup_steps, "val_sample_windows": len(sample_loader.dataset),
        "train_episodes": [str(path) for path, _, _ in train.episodes],
        "val_episodes": [str(path) for path, _, _ in val.episodes],
    }

    checkpoint_dir = dp_checkpoint_dir(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = checkpoint_dir / "best.ckpt"
    best = float("inf")
    print(f"device={device} train_windows={len(train)} val_windows={len(val)}", flush=True)
    with (checkpoint_dir.parent / "metrics.csv").open("w", newline="") as metrics:
        writer = csv.writer(metrics)
        writer.writerow(("epoch", "train_loss", "val_loss", "val_action_mae"))
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = _epoch(
                model, train_loader, device, optimizer=optimizer, lr_scheduler=lr_scheduler, ema=ema,
            )
            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            with torch.inference_mode():
                val_loss = _epoch(model, val_loader, device, generator=torch.Generator(device=device).manual_seed(seed))
                action_mae = _action_mae(
                    model, sample_loader, device, torch.Generator(device=device).manual_seed(seed),
                )
            writer.writerow((epoch, train_loss, val_loss, action_mae))
            metrics.flush()
            print(
                f"epoch={epoch}/{epochs} train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} val_action_mae={action_mae:.5f}", flush=True,
            )
            if action_mae < best:
                best = action_mae
                pending = checkpoint.with_suffix(".tmp")
                torch.save(
                    {
                        "model": model.state_dict(), "policy": policy_config, "epoch": epoch,
                        "train_config": train_config, "val_loss": val_loss, "val_action_mae": action_mae,
                    },
                    pending,
                )
                pending.replace(checkpoint)
            ema.restore(model.parameters())
    print(f"checkpoint={checkpoint}", flush=True)
    return checkpoint


def _epoch(model, loader, device, *, optimizer=None, lr_scheduler=None, ema=None, generator=None):
    import torch
    from torch.nn.functional import mse_loss

    total = torch.zeros((), device=device)
    for image, state, actions, _ in loader:
        image, state, actions = (x.to(device, non_blocking=True) for x in (image, state, actions))
        actions = model.normalize(actions, "action")
        noise = torch.randn(actions.shape, device=device, generator=generator)
        k = torch.randint(
            model.scheduler.config.num_train_timesteps, (len(actions),),
            device=device, generator=generator,
        )
        noisy_actions = model.scheduler.add_noise(actions, noise, k)
        prediction = model(noisy_actions, k, model.encode_observation(image, state))
        loss = mse_loss(prediction, noise)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step(model.parameters())
        total += loss.detach() * len(actions)
    average = total.item() / len(loader.dataset)
    if not math.isfinite(average):
        raise FloatingPointError("Diffusion loss is not finite.")
    return average


def _action_mae(model, loader, device, generator):
    import torch

    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device, dtype=torch.long)
    steps = torch.arange(model.horizon, device=device)
    for image, state, action, valid_steps in loader:
        image, state, action, valid_steps = (x.to(device, non_blocking=True) for x in (image, state, action, valid_steps))
        prediction = model.sample_actions(image, state, generator=generator)
        error = (model.normalize(prediction, "action") - model.normalize(action, "action")).abs()
        valid = steps < valid_steps[:, None]
        total += (error.mean(dim=-1) * valid).sum()
        count += valid.sum()
    average = (total / count).item()
    if not math.isfinite(average):
        raise FloatingPointError("Diffusion sampled action error is not finite.")
    return average


def train_online(config: dict) -> None:
    """Fine-tune the offline diffusion policy in ``stack_blocks``.

    ``scripts/train_dppo.sh`` has already started Isaac in the simulator image.
    Connect with ``starter.envs.remote.RemoteEnv`` using ``env_host`` and
    ``env_port``. Keep the environment score. Do not feed privileged block
    poses to the policy. The policy process and Isaac are on different GPUs.
    """

    del config
    raise NotImplementedError("实现扩散策略的在线微调。从离线 checkpoint 继续，不要随机初始化。")
