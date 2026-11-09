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
    from torch.utils.data import DataLoader
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
    if (min(horizon, image_size, epochs, batch_size) < 1 or num_workers < 0
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Diffusion training sizes and lr must be positive; num_workers must be nonnegative.")

    torch.manual_seed(seed)
    random.Random(seed).shuffle(episodes)
    n_val = max(1, len(episodes) // 10)
    train = ActionWindows(episodes[n_val:], horizon, image_size, decode_image_bit)
    val = ActionWindows(episodes[:n_val], horizon, image_size, decode_image_bit)
    device = torch.device(f"cuda:{require(config, 'policy_gpu')}" if torch.cuda.is_available() else "cpu")
    loader_options = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train, shuffle=True, **loader_options)
    val_loader = DataLoader(val, **loader_options)
    policy_config = {
        "horizon": horizon,
        "image_size": image_size,
        "scheduler_config": {
            "num_train_timesteps": 100,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
            "timestep_spacing": "linspace",
        },
    }
    model = DiffusionPolicy(**policy_config, bounds=train.bounds()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_config = {
        "data_path": str(root), "seed": seed, "device": str(device), "epochs": epochs,
        "batch_size": batch_size, "num_workers": num_workers, "lr": learning_rate,
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
        writer.writerow(("epoch", "train_loss", "val_loss"))
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = _epoch(model, train_loader, device, optimizer=optimizer)
            model.eval()
            with torch.no_grad():
                generator = torch.Generator(device=device).manual_seed(seed)
                val_loss = _epoch(model, val_loader, device, generator=generator)
            writer.writerow((epoch, train_loss, val_loss))
            metrics.flush()
            print(f"epoch={epoch}/{epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f}", flush=True)
            if val_loss < best:
                best = val_loss
                torch.save(
                    {
                        "model": model.state_dict(), "policy": policy_config, "epoch": epoch,
                        "train_config": train_config, "val_loss": val_loss,
                    },
                    checkpoint,
                )
    print(f"checkpoint={checkpoint}", flush=True)
    return checkpoint


def _epoch(model, loader, device, *, optimizer=None, generator=None):
    import torch
    from torch.nn.functional import mse_loss

    total = torch.zeros((), device=device)
    for image, state, actions in loader:
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
        total += loss.detach() * len(actions)
    average = total.item() / len(loader.dataset)
    if not math.isfinite(average):
        raise FloatingPointError("Diffusion loss is not finite.")
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
