"""Render robot planner configuration for the simulator's mounted asset tree."""
from __future__ import annotations

from pathlib import Path

import yaml


def prepare_robot_configs(
    env_config: Path, assets: Path, output: Path, container_root: Path,
) -> list[tuple[Path, Path]]:
    """Return host/container config mounts without modifying shared assets.

    RoboDojo's ``curobo_tmp.yml`` uses ASSETS_PATH for its repository root,
    rather than its Assets directory. Render against the container namespace;
    pre-generated curobo.yml files may belong to a different host checkout.
    """
    env_config, assets, output = env_config.resolve(), assets.resolve(), output.resolve()
    selection = yaml.safe_load(env_config.read_text(encoding="utf-8"))
    robots_path = env_config.parent / "robot" / (selection["config"]["robot"] + ".yml")
    robots = yaml.safe_load(robots_path.read_text(encoding="utf-8"))["robots"]
    container_assets = container_root / "Assets"
    configs = []
    for name in sorted({robot["robot_name"] for robot in robots}):
        template = assets / "Robots" / name / "curobo_tmp.yml"
        text = template.read_text(encoding="utf-8")
        text = text.replace("${ASSETS_PATH}", str(container_root)).replace("$ASSETS_PATH", str(container_root))
        kinematics = yaml.safe_load(text)["robot_cfg"]["kinematics"]
        robot_root = Path(kinematics["asset_root_path"])
        urdf = Path(kinematics["urdf_path"])
        if robot_root != container_assets / "Robots" / name:
            raise ValueError(f"Robot asset root must resolve inside the container asset tree: {robot_root}")
        # Validate the mounted source before starting an expensive simulator.
        source_urdf = assets / urdf.relative_to(container_assets)
        if not source_urdf.is_file():
            raise FileNotFoundError(source_urdf)
        config = output / name / "curobo.yml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(text, encoding="utf-8")
        configs.append((config, robot_root / "curobo.yml"))
    return configs
