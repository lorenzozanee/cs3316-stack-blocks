"""Offline and online entry points for the ACT pipeline."""

from __future__ import annotations

import argparse

from starter.config import load_config
from starter.policies.act import train_offline, train_online


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ACT pipeline framework.")
    parser.add_argument("stage", choices=("offline", "online"))
    parser.add_argument("--config")
    parser.add_argument("--online-config", help="Residual PPO hyperparameter configuration.")
    parser.add_argument("--checkpoint", help="Explicitly selected offline ACT checkpoint.")
    args = parser.parse_args(argv)
    if args.stage == "online" and not args.checkpoint:
        parser.error("online training requires --checkpoint pointing to an offline ACT checkpoint")
    if args.stage == "offline" and (args.checkpoint or args.online_config):
        parser.error("--checkpoint and --online-config apply only to online training")
    config_path = args.config or ("configs/act.yaml" if args.stage == "offline" else "configs/env.yaml")
    config = load_config(config_path)
    if args.stage == "offline":
        train_offline(config)
    else:
        config.update(load_config(args.online_config or "configs/act_online.yaml"))
        config["base_checkpoint"] = args.checkpoint
        train_online(config)


if __name__ == "__main__":
    main()
