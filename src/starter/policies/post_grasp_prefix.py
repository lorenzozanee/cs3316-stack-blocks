"""Training-only replay of observed, zero-score post-grasp prefixes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from starter.interfaces import ACTION_DIM, BLOCK_LABELS, STEP_LIMIT

# Replay must agree more tightly than the block and gripper geometry gates.
_POSITION_TOLERANCE_M = 0.002
_ORIENTATION_COSINE = np.cos(np.deg2rad(1.0))  # Half-angle for 2° relative rotation.
_OPENNESS_TOLERANCE = 0.02
_LINEAR_SPEED_TOLERANCE_M_S = 0.02
_ANGULAR_SPEED_TOLERANCE_RAD_S = 0.2
# Exact discrete bonuses verify the event history. Continuous progress is
# recomputed from the replayed physical frames.
_EVENT_BONUS_TOLERANCE = 1e-9


def reward_digest(reward_cfg) -> str:
    return hashlib.sha256(
        json.dumps(["control-pose-settling-from-reset", reward_cfg], sort_keys=True, default=repr)
        .encode("utf-8")).hexdigest()


def load_prefixes(path: Path, width: int, layouts: list[int], reward_cfg) -> dict:
    digest = reward_digest(reward_cfg)
    prefixes = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            entry = json.loads(line)
            scene, layout = entry["scene"], entry["layout_id"]
            key = layout, scene
            actions = np.asarray(entry["actions"], dtype=np.float64)
            if (type(scene) is not int or type(layout) is not int
                    or entry["width"] != width or layout not in layouts or not 0 <= scene < width
                    or type(entry["arm"]) is not int or entry["arm"] not in (0, 1)
                    or entry["carried"] not in BLOCK_LABELS or key in prefixes
                    or entry["target_lower"] not in (None, *BLOCK_LABELS)
                    or actions.ndim != 2 or actions.shape[1] != ACTION_DIM
                    or not 0 < len(actions) < STEP_LIMIT or not np.isfinite(actions).all()):
                raise ValueError("invalid replay metadata or actions")
            if entry["reward_digest"] != digest:
                raise ValueError("archive was captured with a different reward configuration")
            frames = entry["frames"]
            if len(frames) != len(actions) + 1:
                raise ValueError("expected a physical snapshot at reset and after each action")
            for frame in frames:
                _snapshot(frame)
                if not np.isfinite(float(frame["bonus"])):
                    raise ValueError("invalid replay event bonus")
            prefixes[key] = {**entry, "actions": actions.tolist()}
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"Invalid post-grasp replay at {path}:{line_number}: {error}") from error
    if not prefixes:
        raise ValueError("Post-grasp replay file is empty.")
    return prefixes


def _snapshot(frame):
    blocks = frame["blocks"]
    velocities = frame["velocities"]
    proprio = np.asarray(frame["proprio"], dtype=np.float64)
    positions = np.asarray([blocks[label]["pos"] for label in BLOCK_LABELS], dtype=np.float64)
    rotations = np.asarray([blocks[label]["rot"] for label in BLOCK_LABELS], dtype=np.float64)
    measured = np.asarray(frame["measured"], dtype=np.float64)
    speeds = np.asarray([[velocities[label][key] for label in BLOCK_LABELS]
                         for key in ("linear", "angular")], dtype=np.float64)
    if (proprio.shape != (ACTION_DIM,) or positions.shape != (len(BLOCK_LABELS), 3)
            or rotations.shape != (len(BLOCK_LABELS), 4) or measured.shape != (2,)
            or speeds.shape != (2, len(BLOCK_LABELS), 3)
            or not all(np.isfinite(x).all() for x in (proprio, positions, rotations, measured, speeds))):
        raise ValueError("invalid grasp snapshot")
    wrists = proprio.reshape(2, 8)
    positions = np.concatenate((positions, wrists[:, :3]))
    rotations = np.concatenate((rotations, wrists[:, 3:7]))
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    if (norms < 1e-9).any():
        raise ValueError("zero orientation in grasp snapshot")
    return positions, rotations / norms, measured, speeds


def _physical_frame(observation, privileged) -> dict:
    return {
        "proprio": observation["proprio"],
        "blocks": privileged["blocks"],
        "measured": privileged["gripper_open_measured"],
        "velocities": privileged["block_velocities"],
    }


def capture_frame(observation, privileged, bonus=0.0) -> dict:
    frame = _physical_frame(observation, privileged)
    return {
        "bonus": bonus,
        "proprio": np.asarray(frame["proprio"]).tolist(),
        "blocks": {label: {key: np.asarray(frame["blocks"][label][key]).tolist()
                           for key in ("pos", "rot")} for label in BLOCK_LABELS},
        "measured": np.asarray(frame["measured"]).tolist(),
        "velocities": {label: {key: np.asarray(frame["velocities"][label][key]).tolist()
                               for key in ("linear", "angular")} for label in BLOCK_LABELS},
    }


def check_frame(entry, step, observation, privileged, bonus) -> None:
    actual = _snapshot(_physical_frame(observation, privileged))
    expected = _snapshot(entry["frames"][step])
    position = np.linalg.norm(actual[0] - expected[0], axis=1)
    orientation = np.abs((actual[1] * expected[1]).sum(axis=1))
    opening = np.abs(actual[2] - expected[2])
    linear, angular = np.linalg.norm(actual[3] - expected[3], axis=2)
    bonus_delta = abs(bonus - float(entry["frames"][step]["bonus"]))
    if (bonus_delta <= _EVENT_BONUS_TOLERANCE and (position <= _POSITION_TOLERANCE_M).all()
            and (orientation >= _ORIENTATION_COSINE).all() and (opening <= _OPENNESS_TOLERANCE).all()
            and (linear <= _LINEAR_SPEED_TOLERANCE_M_S).all()
            and (angular <= _ANGULAR_SPEED_TOLERANCE_RAD_S).all()):
        return

    wrists = ("arm_0", "arm_1")
    bodies = (*BLOCK_LABELS, *wrists)
    rotation_deg = np.rad2deg(2 * np.arccos(np.clip(orientation, 0, 1)))
    mismatches = [f"event_bonus delta={bonus_delta:.6g} (limit {_EVENT_BONUS_TOLERANCE:g})"] if (
        bonus_delta > _EVENT_BONUS_TOLERANCE) else []
    for metric, labels, deltas, failed, limit in (
        ("position_m", bodies, position, position > _POSITION_TOLERANCE_M, _POSITION_TOLERANCE_M),
        ("rotation_deg", bodies, rotation_deg, orientation < _ORIENTATION_COSINE, 2.0),
        ("opening", wrists, opening, opening > _OPENNESS_TOLERANCE, _OPENNESS_TOLERANCE),
        ("linear_m_s", BLOCK_LABELS, linear, linear > _LINEAR_SPEED_TOLERANCE_M_S,
         _LINEAR_SPEED_TOLERANCE_M_S),
        ("angular_rad_s", BLOCK_LABELS, angular, angular > _ANGULAR_SPEED_TOLERANCE_RAD_S,
         _ANGULAR_SPEED_TOLERANCE_RAD_S),
    ):
        mismatches.extend(f"{label}.{metric} delta={delta:.6g} (limit {limit:g})"
                          for label, delta, bad in zip(labels, deltas, failed) if bad)
    raise RuntimeError(f"Post-grasp replay diverged in layout {entry['layout_id']}, "
                       f"scene {entry['scene']}, step {step}: {'; '.join(mismatches)}")


def check_grasp(entry, reward) -> None:
    step = len(entry["actions"])
    matching = [row for row in reward.transport_attempts
                if row["arm"] == entry["arm"] and row["carried"] == entry["carried"]
                and row["grasp_step"] == step]
    if not matching:
        raise RuntimeError(f"Post-grasp replay did not confirm the recorded grasp in layout "
                           f"{entry['layout_id']}, scene {entry['scene']}, step {step}.")
    if matching[0]["target_lower"] != entry["target_lower"]:
        raise RuntimeError(f"Post-grasp replay latched a different support in layout "
                           f"{entry['layout_id']}, scene {entry['scene']}: "
                           f"{matching[0]['target_lower']!r} != {entry['target_lower']!r}.")
