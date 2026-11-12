"""Offline and online entry points for the diffusion pipeline."""

from __future__ import annotations

import argparse

from starter.config import load_config
from starter.policies.diffusion import train_offline, train_online


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diffusion pipeline framework.")
    parser.add_argument("stage", choices=("offline", "online"))
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--training-config", help="Online training settings overriding the environment config.")
    parser.add_argument("--checkpoint", help="Pretrained flow checkpoint for online fine-tuning.")
    parser.add_argument("--resume", help="Resume an FPO training checkpoint.")
    parser.add_argument("--preflight", action="store_true", help="Check one update without connecting to Isaac.")
    args = parser.parse_args(argv)
    if args.stage == "offline" and any((args.training_config, args.checkpoint, args.resume, args.preflight)):
        parser.error("Online training options require the online stage.")
    config = load_config(args.config)
    if args.stage == "offline":
        train_offline(config)
    else:
        if args.training_config:
            config.update(load_config(args.training_config))
        options = dict(config.get("fpo", {}))
        for name in ("checkpoint", "resume"):
            if getattr(args, name):
                options[name] = getattr(args, name)
        if args.preflight:
            options["preflight"] = True
        config["fpo"] = options
        train_online(config)


if __name__ == "__main__":
    main()
