"""Offline and online entry points for the ACT pipeline."""

from __future__ import annotations

import argparse

from starter.config import load_config
from starter.policies.act import train_offline, train_online


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ACT pipeline framework.")
    parser.add_argument("stage", choices=("offline", "online"))
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    config_path = args.config or ("configs/act.yaml" if args.stage == "offline" else "configs/env.yaml")
    config = load_config(config_path)
    if args.stage == "offline":
        train_offline(config)
    else:
        train_online(config)


if __name__ == "__main__":
    main()
