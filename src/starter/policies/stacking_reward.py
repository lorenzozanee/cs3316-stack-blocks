"""Standalone numpy reward geometry for the RoboDojo stack_blocks task.

This module is deliberately independent of the training loop, the remote env
adapter and torch.  It implements the user's seven-step stacking decomposition
as a bounded, non-positive physical-state potential (PBRS; Ng, Harada &
Russell, ICML 1999) plus a small set of one-time, dwell-gated milestone
bonuses: additional event credit under sparse official scoring.

Provenance and scope
--------------------
* RoboDojo's own predicates (``func_parser.is_stacked`` and
  ``is_all_gripper_open``) use block centres, a per-size xy threshold and
  ``dz >= 0.005``; they have no velocity, dwell, contact or gripper-release
  requirement.  They are task predicates, *not* the final
  ``RewardManager`` 100 gate: the official 100 also requires ``run_reward``
  success, and a state-only score is downgraded to the 15 partial score
  without it.  They are *not* copied here as a stability claim.
* robosuite ``stack.py`` uses an eef reach kernel ``1 - tanh(dist)``, a lift
  height threshold and a stacking term conditioned on contact and on not
  grasping.  IsaacLab ``lift`` uses ``1 - tanh(dist)`` and a goal-distance
  term gated by the lifted flag.  Those are upstream *principles*; this file
  is a project-specific adaptation for three cubes, the arx_x5 dual arm and
  the geometry actually exposed by ``get_privileged_state``.
* ``RigidObject`` is built with ``track_contact_forces=False``, so no
  contact-force claim is made anywhere.  "Support" is a geometric proxy
  (face gap, lateral centre-of-mass margin, uprightness), "settled" adds an
  instantaneous velocity proxy, and "released" uses the measured gripper
  joints that the official predicate also uses.  None of these proves
  contact or static equilibrium.
* Holding maps the normalized measured opening through the real x5 joint
  range and the q=0 finger-surface gap:
  ``aperture = 2 * clip(low + (high - low) * measured, 0, high) + gap0``.
  A block is held only when this aperture matches the cube width projected
  on the gripper local y (jaw) axis within ``grasp_aperture_fraction`` of
  that width, so near-empty pinched jaws cannot match a cube.  Each arm is
  associated with at most one nearest block within
  ``grasp_centering_fraction`` of its side; release is local to that
  association and its hard rule is the official measured >= 0.8 on the
  associated arm, never a closeness product.
* Hard base/edge anchoring and the one-time milestones require the remaining
  signed support gap within 10% of the broad tolerance (0.24-0.32 mm across
  the 30/35/40 mm cubes), full central-COM lateral margin (at least one
  quarter of the lower side), upright faces, and the independent hard speed
  limits.  The broad smooth kernels still guide everywhere.  These are
  conservative internal geometric resolutions for sub-mm numerical error,
  not contact proof and not simulator-calibrated solver tolerances.

Summary of the potential (all components are <= 0)::

    phi = phi_reach + phi_manipulation + phi_base + phi_pair + phi_release

    phi_reach        reach a block that is not already placed (soft gated)
    phi_manipulation close, lift, transport and seat the next unplaced block
    phi_base         the first block resting upright and released in the table target
    phi_pair         a directed two-edge support chain anchored to that base
    phi_release      local release deficit on anchored/base blocks

The ``phi_*`` components are pure functions of the current physical
state plus reset constants.  ``terms()`` also reports ``milestone_return``,
which is episode-history diagnostics (latched bonuses already paid) and is
not part of ``phi``; the whole terms dict is therefore not history-free.

The event layer pays ordered crossings at equal fractions of each 3-D line.
Approach uses the measured grasp point and the block centre when the leg
begins. Carry starts at the centre of a confirmed held and lifted block and
ends at the table or current released support top plus the carried half-height.
Crossings use linear interpolation between adjacent control observations.
Grasp credit is once per block and requires the cube's lowest point to be at
least a quarter side above the highest surface under its footprint (the table
or any lower cube's top), so closing the jaws on a cube that still rests on
another cube -- upright, tilted or bridged -- pays nothing. Placement
milestones pay once per level after
consecutive observations of the same supported and released configuration.
Route progress and dwell are episode history and add a preference outside PBRS.

The module returns only terms and bonuses.  It never reads an action, the
official score, ``terminated`` or an RNG and it never ends an episode.
Terminal ``phi=0`` and truncation bootstrap remain the trainer's
responsibility.  Unit tests verify reward algebra and boundaries only; they
do not show that a policy learns, and no success or learning rate is claimed.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from itertools import combinations, permutations

import numpy as np

from starter.interfaces import ACTION_DIM, BLOCK_LABELS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Physical geometry is derived from the block extents and reset state.  These
# are the only task constants that are not fitted per block: the target region
# (an internal preference absent from official scoring; measured from 100
# RoboDojo demos / 300 release-command upcrossings, using the release-command
# grasp-point proxy wrist + 0.143 * R @ x_hat, not a measured released block
# position; max radial 6.706 mm, mean [-0.000349, -0.201014]) and
# dimensionless fractions of a block side.
DEFAULTS = {
    "reach_weight": 2.0,
    "reach_scale": 0.35,
    "manipulation_weight": 1.0,
    "grasp_closure_weight": 0.25,
    "potential_offset_fraction": 0.0,
    "base_weight": 1.0,
    "pair_weight": 1.5,
    "release_weight": 0.5,
    # One-time event layer.  Milestones mark the two large events the user
    # asked for (a true grasp/lift, and each completed support level); the
    # route waypoints below give ordered credit along the straight lines.
    "milestone_grasp": 1.2,
    "milestone_base": 2.0,
    "milestone_pair2": 2.5,
    "milestone_pair3": 3.5,
    "milestone_dwell_steps": 3,
    "waypoint_count": 3,
    "waypoint_bonus": 0.15,
    "waypoint_radius": 0.025,
    # Target region.  x/y are task configuration, not baked into formulas.
    "placement_x": 0.0,
    "placement_y": -0.2,
    "placement_radius": 0.01,
    "placement_soft_scale": 0.15,
    # Gripper calibration from the robot asset (not reward geometry):
    # measured=(joint-low)/(high-low); real joint limits are [0, high], so
    # measured < 0.185 is unreachable and clamps to the q=0 aperture.
    "gripper_scale_low": -0.01,
    "gripper_scale_high": 0.044,
    "gripper_aperture_zero": 0.000807165976524353,
    # Grasp proxies.  The aperture tolerance and centering radius scale with
    # the cube side; both are internal geometric tolerances, not contact
    # proof and not simulator-calibrated solver thresholds.
    "grasp_offset": 0.143,
    "grasp_aperture_fraction": 0.10,
    "grasp_centering_fraction": 0.50,
    # Support and settling tolerances.
    "support_gap_floor": 0.002,
    "support_gap_fraction": 0.08,
    "settle_linear_speed": 0.01,
    "settle_angular_speed": 0.1,
    "upright_cos_min": math.cos(math.radians(15.0)),
}

REWARD_KEYS = tuple(DEFAULTS)

# Dimensionless geometry fractions (side-length units).
_LIFT_RAMP_FRACTION = 0.25      # start lifting above 0.25 s
_LIFT_FULL_FRACTION = 0.75      # full lift credit from 0.75 s
_LIFT_PLATEAU_FRACTION = 2.0    # still full at two side lengths (third block)
_LIFT_ZERO_FRACTION = 3.0       # no credit above three side lengths
_LATERAL_FULL_FRACTION = 0.25   # lateral margin full at a quarter side
_GRIPPER_FULL_OPEN = 0.8        # official is_all_gripper_open threshold
_GRASP_HOLD_MIN = 0.5
_RELEASE_MIN = 0.8              # applied directly to a locally associated arm
_CUBE_EXTENT_REL_TOL = 0.02
# Hard support thresholds are conservative internal geometric resolutions,
# not contact proof or simulator-calibrated solver tolerances.  The broad
# smooth kernels above still guide; only the hard predicates and one-time
# milestones require the tighter values.
_HARD_GAP_FRACTION = 0.10       # |signed gap| <= 10% of the current tolerance
_HARD_LATERAL_KERNEL_MIN = 1.0  # COM margin >= one quarter of the lower side

_MILESTONES = ("grasp", "base", "pair2", "pair3")
_WAYPOINT_ROUTES = tuple(("approach", label) for label in BLOCK_LABELS) + tuple(
    ("carry", label) for label in BLOCK_LABELS)
_WEIGHT_KEYS = (
    "reach_weight", "manipulation_weight", "base_weight", "pair_weight",
    "release_weight", "milestone_grasp",
    "milestone_base", "milestone_pair2", "milestone_pair3", "waypoint_bonus",
)
_POSITIVE_KEYS = (
    "reach_scale", "placement_radius", "placement_soft_scale", "grasp_offset",
    "grasp_aperture_fraction", "grasp_centering_fraction",
    "settle_linear_speed", "settle_angular_speed",
    "support_gap_fraction", "waypoint_radius",
)


def _is_real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def validate_reward_config(cfg: Mapping) -> None:
    """Validate a reward configuration; raise ``ValueError`` on any defect.

    No defaults are filled in here: the caller must supply every key from
    :data:`REWARD_KEYS`, matching the project's "missing key is an error"
    convention.  This keeps a typo from silently disabling a reward term.
    """
    if not isinstance(cfg, Mapping):
        raise ValueError("Reward configuration must be a mapping.")
    missing = [key for key in REWARD_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"Reward configuration is missing keys: {missing}.")
    for key in REWARD_KEYS:
        value = cfg[key]
        if not _is_real(value):
            raise ValueError(f"Reward key {key!r} must be a numeric scalar.")
        if not math.isfinite(float(value)):
            raise ValueError(f"Reward key {key!r} must be finite.")
    dwell = cfg["milestone_dwell_steps"]
    if not isinstance(dwell, numbers.Integral) or isinstance(dwell, bool):
        raise ValueError("milestone_dwell_steps must be an integer, not bool.")
    if int(dwell) < 1:
        raise ValueError("milestone_dwell_steps must be at least 1.")
    points = cfg["waypoint_count"]
    if not isinstance(points, numbers.Integral) or isinstance(points, bool):
        raise ValueError("waypoint_count must be an integer, not bool.")
    if int(points) < 1:
        raise ValueError("waypoint_count must be at least 1.")
    if float(cfg["upright_cos_min"]) <= 0.0 or float(cfg["upright_cos_min"]) >= 1.0:
        raise ValueError("upright_cos_min must lie strictly between 0 and 1.")
    if float(cfg["support_gap_floor"]) < 0.0:
        raise ValueError("support_gap_floor must be non-negative.")
    if not float(cfg["gripper_scale_low"]) < float(cfg["gripper_scale_high"]):
        raise ValueError("gripper_scale_low must be smaller than gripper_scale_high.")
    if float(cfg["gripper_scale_high"]) <= 0.0:
        raise ValueError("gripper_scale_high must be positive for the joint clip [0, high].")
    if float(cfg["gripper_aperture_zero"]) < 0.0:
        raise ValueError("gripper_aperture_zero must be non-negative.")
    if not 0.0 < float(cfg["grasp_aperture_fraction"]) < 0.5:
        raise ValueError("grasp_aperture_fraction must lie strictly in (0, 0.5).")
    if not 0.0 < float(cfg["grasp_centering_fraction"]) < 1.0:
        # Radius below one side keeps a vertically adjacent cube (one side
        # away) out of the holding arm's local association.
        raise ValueError("grasp_centering_fraction must lie strictly in (0, 1).")
    for key in _WEIGHT_KEYS:
        if float(cfg[key]) < 0.0:
            raise ValueError(f"Reward weight {key!r} must be non-negative.")
    for key in _POSITIVE_KEYS:
        if float(cfg[key]) <= 0.0:
            raise ValueError(f"Reward parameter {key!r} must be positive.")
    if not 0.0 <= cfg["grasp_closure_weight"] <= 0.5:
        raise ValueError("grasp_closure_weight must lie in [0, 0.5].")
    if not 0.0 <= cfg["potential_offset_fraction"] <= 1.0:
        raise ValueError("potential_offset_fraction must lie in [0, 1].")
    if float(cfg["placement_radius"]) >= float(cfg["placement_soft_scale"]):
        raise ValueError("placement_radius must be smaller than placement_soft_scale.")
    milestone_sum = (sum(float(cfg[f"milestone_{name}"]) for name in _MILESTONES[1:])
                     + len(BLOCK_LABELS) * float(cfg["milestone_grasp"]))
    waypoint_sum = 2 * len(BLOCK_LABELS) * int(points) * float(cfg["waypoint_bonus"])
    if milestone_sum + waypoint_sum >= 15.0:
        raise ValueError("Milestone and waypoint bonuses must sum to less than 15.")


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _vector(value, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value.")
    return array


def _rotation_wxyz(quat) -> np.ndarray:
    """Rotation matrix from a wxyz quaternion with a strict norm check."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Block rotation must be a finite wxyz quaternion.")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("Block rotation quaternion is degenerate.")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _grasp_points(proprio, offset: float) -> np.ndarray:
    """Grasp centres ``wrist + offset * R(q) @ x_hat`` for both arms."""
    arms = _vector(proprio, ACTION_DIM, "proprio")
    points = []
    for start in (0, 8):
        rotation = _rotation_wxyz(arms[start + 3:start + 7])
        points.append(arms[start:start + 3] + offset * rotation[:, 0])
    return np.stack(points)


def _mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _labels_map(container, name: str) -> Mapping:
    container = _mapping(container, name)
    missing = [label for label in BLOCK_LABELS if label not in container]
    if missing:
        raise ValueError(f"{name} is missing blocks {missing}.")
    return container


def _block_poses(state: Mapping, name: str) -> dict[str, dict]:
    blocks = _labels_map(state.get("blocks"), f"{name}.blocks")
    poses = {}
    for label in BLOCK_LABELS:
        entry = _mapping(blocks[label], f"{name}.blocks[{label}]")
        if "pos" not in entry or "rot" not in entry:
            raise ValueError(f"{name}.blocks[{label}] needs pos and rot.")
        poses[label] = {
            "pos": _vector(entry["pos"], 3, f"{name}.blocks[{label}].pos"),
            "rot": _vector(entry["rot"], 4, f"{name}.blocks[{label}].rot"),
        }
    return poses


def _block_velocities(state: Mapping, name: str) -> dict[str, dict[str, np.ndarray]]:
    velocities = _labels_map(state.get("block_velocities"),
                             f"{name}.block_velocities")
    result = {}
    for label in BLOCK_LABELS:
        entry = _mapping(velocities[label], f"{name}.block_velocities[{label}]")
        if "linear" not in entry or "angular" not in entry:
            raise ValueError(f"{name}.block_velocities[{label}] needs linear/angular.")
        result[label] = {
            "linear": _vector(entry["linear"], 3,
                              f"{name}.block_velocities[{label}].linear"),
            "angular": _vector(entry["angular"], 3,
                               f"{name}.block_velocities[{label}].angular"),
        }
    return result


def _measured_grippers(state: Mapping, name: str) -> np.ndarray:
    if "gripper_open_measured" not in state:
        raise ValueError(f"{name} is missing gripper_open_measured.")
    measured = _vector(state["gripper_open_measured"], 2,
                       f"{name}.gripper_open_measured")
    if (measured < 0.0).any():
        raise ValueError("Measured gripper openings must be non-negative.")
    return measured


def _block_extents(state: Mapping) -> dict[str, np.ndarray]:
    extents = _labels_map(state.get("block_extents"), "reset_state.block_extents")
    result = {}
    for label in BLOCK_LABELS:
        value = _vector(extents[label], 3, f"block_extents[{label}]")
        if (value <= 0.0).any():
            raise ValueError(f"Block extent for {label} must be positive.")
        side = float(np.mean(value))
        if float(np.max(value) - np.min(value)) > _CUBE_EXTENT_REL_TOL * side:
            raise ValueError(f"Block {label} is not a cube within tolerance.")
        result[label] = value
    return result


def _table(state: Mapping) -> dict[str, float]:
    table = _mapping(state.get("table"), "reset_state.table")
    required = ("top_z", "x_min", "x_max", "y_min", "y_max")
    missing = [key for key in required if key not in table]
    if missing:
        raise ValueError(f"Table metadata is missing {missing}.")
    result = {}
    for key in required:
        value = table[key]
        if not _is_real(value) or not math.isfinite(float(value)):
            raise ValueError(f"Table field {key!r} must be a finite number.")
        result[key] = float(value)
    if result["x_min"] >= result["x_max"] or result["y_min"] >= result["y_max"]:
        raise ValueError("Table bounds are inverted.")
    return result


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _hard_gap_ok(gap: float, tolerance: float) -> bool:
    """Hard support gap: remaining signed numerical gap within 10% of the
    current tolerance (broad kernels still guide everywhere).  This is a
    conservative internal geometric resolution, not contact proof."""
    return abs(float(gap)) <= _HARD_GAP_FRACTION * float(tolerance) + 1e-12


class _WaypointPath:
    def __init__(self, start: np.ndarray, target: np.ndarray, count: int, radius: float):
        self.start = start.copy()
        self.target = target.copy()
        self.previous = start.copy()
        delta = target - start
        length = float(np.linalg.norm(delta))
        self.axis = delta / length if length > 1e-9 else np.zeros_like(delta)
        self.spacing = length / (count + 1)
        self.count = count
        self.radius = min(radius, 0.45 * self.spacing)
        self.passed = 0

    def advance(self, position: np.ndarray) -> int:
        """Count ordered forward crossings within a disk on each waypoint plane."""
        previous = self.previous
        self.previous = position.copy()
        if self.passed == self.count or self.spacing <= 1e-9:
            return self.passed
        before = float((previous - self.start) @ self.axis)
        after = float((position - self.start) @ self.axis)
        while self.passed < self.count:
            along = (self.passed + 1) * self.spacing
            if not before < along <= after + 1e-12 or after <= before:
                break
            at_plane = previous + (position - previous) * min(1.0, (along - before) / (after - before))
            point = self.start + along * self.axis
            if np.linalg.norm(at_plane - point) > self.radius + 1e-12:
                break
            self.passed += 1
        return self.passed


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class StackingReward:
    """Per-episode reward geometry.  Pure terms plus one-time event latches."""

    def __init__(self, reset_state: Mapping, reset_proprio, cfg: Mapping):
        validate_reward_config(cfg)
        self.cfg = {key: cfg[key] for key in REWARD_KEYS}
        self.cfg["milestone_dwell_steps"] = int(cfg["milestone_dwell_steps"])
        self._reset_proprio = _vector(reset_proprio, ACTION_DIM, "reset_proprio")
        reset_state = _mapping(reset_state, "reset_state")
        self._reset_poses = _block_poses(reset_state, "reset_state")
        self._extents = _block_extents(reset_state)
        self._table = _table(reset_state)
        # Validate the remaining reset fields now; they are part of the
        # episode contract even though only current state is scored later.
        _block_velocities(reset_state, "reset_state")
        _measured_grippers(reset_state, "reset_state")

        self._reset_heights = {
            label: float(self._reset_poses[label]["pos"][2]) for label in BLOCK_LABELS
        }
        self._sides = {
            label: float(np.mean(self._extents[label])) for label in BLOCK_LABELS
        }
        self._half_widths = {
            label: 0.5 * float(min(self._extents[label][0], self._extents[label][1]))
            for label in BLOCK_LABELS
        }
        self._target = np.array(
            [float(self.cfg["placement_x"]), float(self.cfg["placement_y"])],
            dtype=np.float64)
        self._validate_target_fits_table()

        self._step_index = 0
        self.milestone_steps = {name: None for name in _MILESTONES}
        self.milestone_return = 0.0
        self._dwell = {name: 0 for name in _MILESTONES[1:]}
        self._dwell_ids = {name: None for name in _MILESTONES[1:]}
        self.grasp_attempts = {}
        self._active_arrivals = {}
        self.grasp_steps_by_block = {label: None for label in BLOCK_LABELS}
        self._grasp_dwell = {}
        self._waypoint_count = int(self.cfg["waypoint_count"])
        self._waypoint_radius = float(self.cfg["waypoint_radius"])
        self.waypoint_return = 0.0
        self.waypoint_counts = {f"{kind}:{label}": 0 for kind, label in _WAYPOINT_ROUTES}
        reset_grasp = _grasp_points(reset_proprio, float(self.cfg["grasp_offset"]))
        self._approach = {
            label: [self._path(point, self._reset_poses[label]["pos"]) for point in reset_grasp]
            for label in BLOCK_LABELS}
        self._approach_arm = {}
        self._carry = {}
        self._route_support = ()

        self.initial_terms = self.terms(self._reset_proprio, reset_state)

    # -- reset-time validation ------------------------------------------------

    def _validate_target_fits_table(self) -> None:
        radius = float(self.cfg["placement_radius"])
        margin = radius + max(self._half_widths.values())
        x, y = float(self._target[0]), float(self._target[1])
        table = self._table
        if not (table["x_min"] + margin <= x <= table["x_max"] - margin
                and table["y_min"] + margin <= y <= table["y_max"] - margin):
            raise ValueError("Target region and tower footprint do not fit the table.")

    # -- geometry -------------------------------------------------------------

    def _gap_tolerance(self, lower: str, upper: str) -> float:
        side = min(self._sides[lower], self._sides[upper])
        return max(float(self.cfg["support_gap_floor"]),
                   float(self.cfg["support_gap_fraction"]) * side)

    def _inside_table(self, label: str, position: np.ndarray) -> bool:
        half = self._half_widths[label]
        table = self._table
        return bool(table["x_min"] + half <= position[0] <= table["x_max"] - half
                    and table["y_min"] + half <= position[1] <= table["y_max"] - half)

    def _evaluate(self, proprio, physical: Mapping):
        """Return physical terms and the geometry used to update route events."""
        physical = _mapping(physical, "physical")
        poses = _block_poses(physical, "physical")
        velocities = _block_velocities(physical, "physical")
        measured = _measured_grippers(physical, "physical")
        arms = _vector(proprio, ACTION_DIM, "proprio")
        grasp = _grasp_points(proprio, float(self.cfg["grasp_offset"]))

        limits_linear = float(self.cfg["settle_linear_speed"])
        limits_angular = float(self.cfg["settle_angular_speed"])
        cos_min = float(self.cfg["upright_cos_min"])
        radius = float(self.cfg["placement_radius"])
        soft = float(self.cfg["placement_soft_scale"])
        table_top = self._table["top_z"]

        pos, rotations, upright, half_height, bottom = {}, {}, {}, {}, {}
        half_x, half_y = {}, {}
        quiet_ok, quiet = {}, {}
        for label in BLOCK_LABELS:
            position = poses[label]["pos"]
            rotation = _rotation_wxyz(poses[label]["rot"])
            pos[label] = position
            rotations[label] = rotation
            # Any cube face may face up: the largest world-z component of a
            # body axis is 1 for a face-upright cube, unlike R[2, 2] alone.
            upright[label] = float(np.max(np.abs(rotation[2, :])))
            half_height[label] = 0.5 * float(
                np.dot(np.abs(rotation[2, :]), self._extents[label]))
            # Horizontal extents of the rotated bounding box (plan view).
            half_x[label] = 0.5 * float(np.dot(np.abs(rotation[0, :]),
                                               self._extents[label]))
            half_y[label] = 0.5 * float(np.dot(np.abs(rotation[1, :]),
                                               self._extents[label]))
            bottom[label] = float(position[2] - half_height[label])
            linear = velocities[label]["linear"]
            angular = velocities[label]["angular"]
            linear_norm = float(np.linalg.norm(linear))
            angular_norm = float(np.linalg.norm(angular))
            quiet_ok[label] = bool(
                linear_norm <= limits_linear + 1e-12
                and angular_norm <= limits_angular + 1e-12)
            quiet[label] = (
                _clip01(1.0 - linear_norm / limits_linear)
                * _clip01(1.0 - angular_norm / limits_angular)
            )

        upright_score = {
            label: _clip01((upright[label] - cos_min) / (1.0 - cos_min))
            for label in BLOCK_LABELS
        }

        # Per-arm local association, aperture-based holding and local release.
        # Each arm is associated with at most its nearest eligible block within
        # grasp_centering_fraction * that block's side.  Holding requires the
        # measured aperture to match the cube width projected on the gripper's
        # local y (jaw) axis, so near-empty pinched jaws cannot match a cube and
        # real normalized x5 grasps are reachable.  Release is local: only arms
        # associated with a block can withhold its release, and the hard rule is
        # the official measured >= 0.8 on those arms (no closeness product).
        low = float(self.cfg["gripper_scale_low"])
        high = float(self.cfg["gripper_scale_high"])
        gap0 = float(self.cfg["gripper_aperture_zero"])
        joints = np.clip(low + (high - low) * measured, 0.0, high)
        apertures = 2.0 * joints + gap0
        max_aperture = 2.0 * high + gap0
        aperture_fraction = float(self.cfg["grasp_aperture_fraction"])
        centering_radius = {
            label: float(self.cfg["grasp_centering_fraction"]) * self._sides[label]
            for label in BLOCK_LABELS}
        arm_rotations = [_rotation_wxyz(arms[start + 3:start + 7])
                         for start in (0, 8)]
        proximity, hold, grasp_closure = {}, {}, {}
        associated_arms = {label: [] for label in BLOCK_LABELS}
        for arm in (0, 1):
            start = 8 * arm
            rotation = arm_rotations[arm]
            jaw_axis = rotation[:, 1]
            distances = {}
            for label in BLOCK_LABELS:
                relative = rotation.T @ (pos[label] - arms[start:start + 3])
                grasp_relative = relative - np.array(
                    [float(self.cfg["grasp_offset"]), 0.0, 0.0])
                distances[label] = float(np.linalg.norm(grasp_relative))
            eligible = [(distances[label], label) for label in BLOCK_LABELS
                        if distances[label] <= centering_radius[label]]
            association = min(eligible)[1] if eligible else None
            if association is not None:
                associated_arms[association].append(arm)
            for label in BLOCK_LABELS:
                allowed = centering_radius[label]
                if label != association:
                    proximity[(arm, label)] = 0.0
                    hold[(arm, label)] = 0.0
                    grasp_closure[(arm, label)] = 0.0
                    continue
                proximity[(arm, label)] = _clip01(
                    (allowed - distances[label]) / allowed)
                expected_width = float(np.dot(
                    np.abs(jaw_axis @ rotations[label]), self._extents[label]))
                tolerance = aperture_fraction * expected_width
                aperture = float(apertures[arm])
                aperture_match = _clip01(
                    1.0 - abs(aperture - expected_width) / tolerance)
                hold[(arm, label)] = (
                    proximity[(arm, label)] * aperture_match)
                if aperture >= expected_width:
                    open_span = max(max_aperture - expected_width, tolerance)
                    close_score = _clip01(
                        1.0 - (aperture - expected_width) / open_span)
                else:
                    close_score = _clip01(
                        1.0 - (expected_width - aperture) / tolerance)
                grasp_closure[(arm, label)] = (
                    proximity[(arm, label)] * close_score)
        held_by_block = {
            label: max(hold[(arm, label)] for arm in (0, 1))
            for label in BLOCK_LABELS
        }
        grasp_closure_by_block = {
            label: max(grasp_closure[(arm, label)] for arm in (0, 1))
            for label in BLOCK_LABELS
        }
        release_block, released_hard = {}, {}
        for label in BLOCK_LABELS:
            local_arms = associated_arms[label]
            if local_arms:
                release_block[label] = _clip01(
                    min(float(measured[arm]) for arm in local_arms) / _RELEASE_MIN)
                released_hard[label] = bool(all(
                    float(measured[arm]) >= _RELEASE_MIN for arm in local_arms))
            else:
                # A remote arm never affects a block it is not associated with.
                release_block[label] = 1.0
                released_hard[label] = True
        min_gripper = _clip01(float(np.min(measured)) / _GRIPPER_FULL_OPEN)

        # Reset-relative lift diagnostics; reward uses support clearance below.
        lift = {}
        lift_gate = {}
        for label in BLOCK_LABELS:
            lift[label] = max(0.0, float(pos[label][2] - self._reset_heights[label]))
            ratio = lift[label] / self._sides[label]
            rising = _clip01((ratio - _LIFT_RAMP_FRACTION)
                             / (_LIFT_FULL_FRACTION - _LIFT_RAMP_FRACTION))
            falling = _clip01((_LIFT_ZERO_FRACTION - ratio)
                              / (_LIFT_ZERO_FRACTION - _LIFT_PLATEAU_FRACTION))
            lift_gate[label] = rising * falling
        lift_progress = min(sum(lift_gate.values()), 2.0)

        # First block at the configured target region: bottom face on the
        # actual table, quiet, face-upright, locally released.
        distance_xy = {
            label: float(np.linalg.norm(pos[label][:2] - self._target))
            for label in BLOCK_LABELS
        }
        target_proximity = {
            label: 1.0 - float(np.tanh(
                max(0.0, distance_xy[label] - radius) / soft))
            for label in BLOCK_LABELS
        }
        table_gap = {label: abs(bottom[label] - table_top) for label in BLOCK_LABELS}
        table_gap_tol = {
            label: max(float(self.cfg["support_gap_floor"]),
                       float(self.cfg["support_gap_fraction"]) * self._sides[label])
            for label in BLOCK_LABELS
        }
        table_contact = {
            label: _clip01((table_gap_tol[label] - table_gap[label])
                           / table_gap_tol[label])
            for label in BLOCK_LABELS
        }
        inside_table = {
            label: self._inside_table(label, pos[label]) for label in BLOCK_LABELS
        }
        table_support = {
            label: (table_contact[label] * upright_score[label] * quiet[label]
                    * float(inside_table[label]))
            for label in BLOCK_LABELS
        }
        base_geom = {label: target_proximity[label] * table_support[label]
                     for label in BLOCK_LABELS}
        base_score = {
            label: base_geom[label] * release_block[label]
            for label in BLOCK_LABELS
        }
        base_ok = {
            label: bool(
                distance_xy[label] <= radius
                and _hard_gap_ok(table_gap[label], table_gap_tol[label])
                and inside_table[label]
                and upright[label] >= cos_min
                and quiet_ok[label]
                and released_hard[label])
            for label in BLOCK_LABELS
        }
        deficit_offset = 1.0 - float(self.cfg["potential_offset_fraction"])
        phi_base = float(self.cfg["base_weight"]) * (max(base_score.values()) - deficit_offset)

        # Directed support edges.  Upper means strictly higher centre; equal
        # z is a side-by-side or tabletop co-planar pair and carries no edge.
        geometry_edge: dict[tuple[str, str], float] = {}
        quiet_edge: dict[tuple[str, str], float] = {}
        release_edge: dict[tuple[str, str], float] = {}
        edge_hard: dict[tuple[str, str], bool] = {}
        for lower, upper in permutations(BLOCK_LABELS, 2):
            if pos[upper][2] <= pos[lower][2]:
                continue
            gap = float(bottom[upper] - (pos[lower][2] + half_height[lower]))
            tolerance = self._gap_tolerance(lower, upper)
            gap_kernel = _clip01((tolerance - abs(gap)) / tolerance)
            lateral = float(np.linalg.norm(pos[upper][:2] - pos[lower][:2]))
            margin = self._half_widths[lower] - lateral
            lateral_kernel = _clip01(margin / (_LATERAL_FULL_FRACTION
                                               * self._sides[lower]))
            geometry = (gap_kernel * lateral_kernel
                        * upright_score[lower] * upright_score[upper])
            quieted = geometry * quiet[lower] * quiet[upper]
            release_pair = min(release_block[lower], release_block[upper])
            geometry_edge[(lower, upper)] = geometry
            quiet_edge[(lower, upper)] = quieted
            release_edge[(lower, upper)] = quieted * release_pair
            edge_hard[(lower, upper)] = bool(
                _hard_gap_ok(gap, tolerance)
                and lateral_kernel >= _HARD_LATERAL_KERNEL_MIN - 1e-12
                and upright[lower] >= cos_min
                and upright[upper] >= cos_min
                and quiet_ok[lower] and quiet_ok[upper]
                and released_hard[lower] and released_hard[upper])

        # pair_sum is geometry-only and is deliberately not required to be
        # table-anchored: a max-pair chain can be a hovering transport pair.
        # The stable/potential pair score below is always anchored instead.
        pair_sum = 0.0
        chain_id = None
        for lower, middle, upper in permutations(BLOCK_LABELS):
            first = geometry_edge.get((lower, middle), 0.0)
            second = geometry_edge.get((middle, upper), 0.0)
            value = first + second
            if value > pair_sum:
                pair_sum = value
                chain_id = (lower, middle, upper)
        pair_sum = min(pair_sum, 2.0)

        # Official stacking has no target XY; only the release route retains it.
        pair_level = 0.0
        support_member = dict(base_geom)
        for lower, middle, upper in permutations(BLOCK_LABELS):
            first = quiet_edge.get((lower, middle), 0.0)
            second = quiet_edge.get((middle, upper), 0.0)
            supported_first = table_support[lower] * first
            supported_second = supported_first * second
            pair_level = max(pair_level, supported_first + supported_second)
            target = target_proximity[lower]
            support_member[middle] = max(support_member[middle], target * supported_first)
            support_member[upper] = max(support_member[upper], target * supported_second)
        pair_level = min(pair_level, 2.0)

        phi_pair = float(self.cfg["pair_weight"]) * (pair_level - 2.0 * deficit_offset)

        # Table-anchored, released, quiet support (diagnostic and release
        # penalty weight); the lower block need not be at the target.
        settled_pair = 0.0
        for (lower, upper), value in release_edge.items():
            settled_pair = max(settled_pair,
                               value * table_contact[lower])
        settled_pair = min(settled_pair, 1.0)

        # Local release deficit on blocks participating in an anchored base
        # or support.  Remote grippers do not enter this value.
        release_deficit = max(
            (1.0 - release_block[label]) * support_member[label]
            for label in BLOCK_LABELS)
        phi_release = -float(self.cfg["release_weight"]) * release_deficit

        distances = {
            label: float(np.min(np.linalg.norm(grasp - pos[label], axis=1)))
            for label in BLOCK_LABELS
        }

        # Event predicates.  These use hard velocity limits, face margin,
        # contact tolerance and measured local release; no continuous product
        # is allowed to stand in for a hard threshold.
        # Clearance above the highest surface beneath the cube's footprint:
        # the table, or the top of any lower cube whose plan-view box overlaps.
        # Grasp pay needs the same 0.25-side lift ramp above that surface, so a
        # cube still on another cube (upright, tilted or bridged) cannot pay
        # while stationary, and a genuinely lifted cube still does.
        support_top = {label: table_top for label in BLOCK_LABELS}
        for first, second in combinations(BLOCK_LABELS, 2):
            if pos[first][2] == pos[second][2]:
                continue
            lower, upper = ((first, second) if pos[first][2] < pos[second][2]
                            else (second, first))
            if (abs(float(pos[upper][0] - pos[lower][0]))
                    >= half_x[lower] + half_x[upper]
                    or abs(float(pos[upper][1] - pos[lower][1]))
                    >= half_y[lower] + half_y[upper]):
                continue
            support_top[upper] = max(
                support_top[upper], float(pos[lower][2] + half_height[lower]))
        grasp_candidates = {
            label: max((0, 1), key=lambda arm: hold[arm, label])
            for label in BLOCK_LABELS
            if held_by_block[label] >= _GRASP_HOLD_MIN
            and bottom[label] - support_top[label] + 1e-12
            >= _LIFT_RAMP_FRACTION * self._sides[label]
        }
        base_candidates = [label for label in BLOCK_LABELS if base_ok[label]]
        base_id = (max(base_candidates, key=lambda label: base_score[label])
                   if base_candidates else None)

        pair2_candidates = []
        for lower, upper in permutations(BLOCK_LABELS, 2):
            if base_ok[lower] and edge_hard.get((lower, upper), False):
                pair2_candidates.append(
                    (release_edge[(lower, upper)], (lower, upper)))
        pair2_id = (max(pair2_candidates, key=lambda item: (item[0], item[1]))[1]
                    if pair2_candidates else None)

        pair3_candidates = []
        for lower, middle, upper in permutations(BLOCK_LABELS):
            if (base_ok[lower]
                    and edge_hard.get((lower, middle), False)
                    and edge_hard.get((middle, upper), False)):
                score = (release_edge[(lower, middle)]
                         * release_edge[(middle, upper)])
                pair3_candidates.append((score, (lower, middle, upper)))
        pair3_id = (max(pair3_candidates,
                        key=lambda item: (item[0], item[1]))[1]
                    if pair3_candidates else None)

        support = pair3_id or pair2_id or ((base_id,) if base_id is not None else ())
        stable_level = len(support)
        reach_scale = float(self.cfg["reach_scale"])
        stage_reach_weight = float(self.cfg["reach_weight"]) / len(BLOCK_LABELS)
        manipulation_weight = float(self.cfg["manipulation_weight"])
        closure_weight = float(self.cfg["grasp_closure_weight"])
        next_reach = next_manipulation = 0.0
        best_score = -1.0
        # robosuite Stack.staged_rewards uses lift-gated XY alignment. Track
        # this task's next support in 3-D, with a handoff from lift to seating.
        lower = support[-1] if support else None
        for label in BLOCK_LABELS:
            if label in support:
                continue
            clearance = _LIFT_RAMP_FRACTION * self._sides[label]
            raised = _clip01((bottom[label] - support_top[label]) / clearance)
            if lower is None:
                lateral = max(0.0, distance_xy[label] - radius)
                gap = bottom[label] - table_top
                tolerance = table_gap_tol[label]
                aligned = _clip01(1.0 - lateral / clearance) if inside_table[label] else 0.0
            else:
                lateral = float(np.linalg.norm(pos[label][:2] - pos[lower][:2]))
                gap = bottom[label] - (pos[lower][2] + half_height[lower])
                tolerance = self._gap_tolerance(lower, label)
                aligned = _clip01((self._half_widths[lower] - lateral)
                                  / (_LATERAL_FULL_FRACTION * self._sides[lower]))
            aligned *= upright_score[label]
            descent = aligned * _clip01(
                1.0 - max(0.0, -gap, gap - clearance) / clearance)
            tracking = 1.0 - math.tanh(math.hypot(lateral, gap) / soft)
            # Reserve transport credit for actual aperture match and clearance;
            # only aligned blocks can retain it while descending to the surface.
            grasp_lift_carry = (closure_weight * grasp_closure_by_block[label]
                               + max(raised, descent) * held_by_block[label]
                               * (0.25 + 0.25 * tracking))
            seated = aligned * _clip01(
                (tolerance - abs(gap)) / ((1.0 - _HARD_GAP_FRACTION) * tolerance))
            reach = max(seated, 1.0 - math.tanh(distances[label] / reach_scale))
            manipulation = seated + (1.0 - seated) * grasp_lift_carry
            # Both terms track the same block. Once placed, its stage credit
            # remains and the next reach starts without undoing the descent.
            score = stage_reach_weight * reach + manipulation_weight * manipulation
            if score > best_score:
                best_score = score
                next_reach, next_manipulation = reach, manipulation
        stage_deficit = len(BLOCK_LABELS) * deficit_offset - stable_level
        phi_reach = stage_reach_weight * (next_reach - stage_deficit)
        phi_manipulation = manipulation_weight * (next_manipulation - stage_deficit)
        manipulation_progress = stable_level + next_manipulation

        support_gaps = []
        for first, second in combinations(BLOCK_LABELS, 2):
            if pos[second][2] >= pos[first][2]:
                lower, upper = first, second
            else:
                lower, upper = second, first
            gap = float(bottom[upper] - (pos[lower][2] + half_height[lower]))
            support_gaps.append({"lower": lower, "upper": upper, "gap": gap})

        phi = phi_reach + phi_manipulation + phi_base + phi_pair + phi_release
        terms = {
            "phi": float(phi),
            "phi_reach": float(phi_reach),
            "phi_manipulation": float(phi_manipulation),
            "phi_base": float(phi_base),
            "phi_pair": float(phi_pair),
            "phi_release": float(phi_release),
            "manipulation_progress": float(manipulation_progress),
            "pair_sum": float(pair_sum),
            "d_min": float(min(distances.values())),
            "lift": float(max(lift.values())),
            "lift_gate": float(lift_progress / 2.0),
            "min_gripper": float(min_gripper),
            "settled_pair": float(settled_pair),
            "lift_progress": float(lift_progress),
            "base_score": float(max(base_score.values())),
            "base_ok": bool(base_id is not None),
            "pair2_ok": bool(pair2_id is not None),
            "pair3_ok": bool(pair3_id is not None),
            "grasp_ok": bool(grasp_candidates),
            "stable_tower_levels": int(stable_level),
            "target_xy": [float(self._target[0]), float(self._target[1])],
            "per_block_lift": {label: float(lift[label]) for label in BLOCK_LABELS},
            "support_gaps": support_gaps,
            "milestone_return": float(self.milestone_return),
            "waypoint_return": float(self.waypoint_return),
            "waypoint_counts": dict(self.waypoint_counts),
        }
        identities = {"base": base_id, "pair2": pair2_id, "pair3": pair3_id}
        if not math.isfinite(terms["phi"]):
            raise ValueError("Reward potential is non-finite.")
        return terms, identities, (pos, grasp, half_height, bottom, hold,
                                   associated_arms, grasp_candidates, support)

    def _path(self, start: np.ndarray, target: np.ndarray) -> _WaypointPath:
        return _WaypointPath(start, target, self._waypoint_count, self._waypoint_radius)

    def _claim_waypoints(self, kind: str, label: str, reached: int) -> int:
        key = f"{kind}:{label}"
        new = max(0, reached - self.waypoint_counts[key])
        self.waypoint_counts[key] += new
        return new

    def _route_events(self, pos, grasp, half_height, bottom, hold,
                      associated_arms, grasp_candidates, support):
        if support != self._route_support:
            self._route_support = support
            self._approach.clear()
            self._approach_arm.clear()
            self._carry.clear()

        new_points = 0
        grasp_bonus = 0.0
        placed_levels = sum(self.milestone_steps[name] is not None for name in _MILESTONES[1:])
        table_top = self._table["top_z"]
        if support:
            lower = support[-1]
            support_xy = pos[lower][:2]
            support_z = pos[lower][2] + half_height[lower]
        else:
            support_xy, support_z = self._target, table_top

        for label in BLOCK_LABELS:
            # A continuous association is one diagnostic attempt, not a bonus.
            for index in (0, 1):
                key = f"{index}:{label}"
                if label in support or index not in associated_arms[label]:
                    self._active_arrivals.pop(key, None)
                elif key not in self._active_arrivals:
                    counts = self.grasp_attempts.setdefault(
                        key, {"arrivals": 0, "confirmed": 0, "steps_to_grasp": 0})
                    counts["arrivals"] += 1
                    self._active_arrivals[key] = self._step_index
            if label in support:
                self._carry.pop(label, None)
                self._grasp_dwell.pop(label, None)
                continue
            arm = grasp_candidates.get(label)
            previous_arm, dwell = self._grasp_dwell.get(label, (None, 0))
            if arm is None:
                self._grasp_dwell.pop(label, None)
            else:
                dwell = dwell + 1 if arm == previous_arm else 1
                self._grasp_dwell[label] = (arm, dwell)
                if dwell >= self.cfg["milestone_dwell_steps"]:
                    key = f"{arm}:{label}"
                    arrival = self._active_arrivals[key]
                    if arrival is not None:
                        self.grasp_attempts[key]["confirmed"] += 1
                        self.grasp_attempts[key]["steps_to_grasp"] += self._step_index - arrival
                        self._active_arrivals[key] = None
                    if self.grasp_steps_by_block[label] is None:
                        self.grasp_steps_by_block[label] = self._step_index
                        if self.milestone_steps["grasp"] is None:
                            self.milestone_steps["grasp"] = self._step_index
                        grasp_bonus += float(self.cfg["milestone_grasp"])

            carry = self._carry.get(label)
            if carry is not None:
                carry_arm, path = carry
                if (len(support) < placed_levels or bottom[label] <= table_top + 1e-9
                        or hold[carry_arm, label] < _GRASP_HOLD_MIN):
                    self._carry.pop(label)
                    carry = None
            ready_to_carry = arm is not None and dwell >= self.cfg["milestone_dwell_steps"]
            if len(support) >= placed_levels and (ready_to_carry or carry is not None):
                target = np.array([*support_xy, support_z + half_height[label]])
                if carry is None:
                    self._carry[label] = (arm, self._path(pos[label], target))
                elif np.linalg.norm(path.target - target) > min(self._waypoint_radius,
                                                                 self._sides[label] / 4):
                    self._carry[label] = (carry_arm, self._path(pos[label], target))
                else:
                    new_points += self._claim_waypoints("carry", label, path.advance(pos[label]))

            if (arm is not None or label in self._carry
                    or self.waypoint_counts[f"approach:{label}"] == self._waypoint_count):
                self._approach.pop(label, None)
                self._approach_arm.pop(label, None)
                continue
            paths = self._approach.get(label)
            if (paths is None or np.linalg.norm(paths[0].target - pos[label])
                    > min(self._waypoint_radius, self._sides[label] / 4)):
                self._approach[label] = [self._path(point, pos[label]) for point in grasp]
                self._approach_arm.pop(label, None)
                continue
            selected = self._approach_arm.get(label)
            if selected is None:
                reached = [path.advance(grasp[index]) for index, path in enumerate(paths)]
                selected = int(np.argmax(reached))
                if reached[selected] == 0:
                    continue
                self._approach_arm[label] = selected
                progress = reached[selected]
            else:
                progress = paths[selected].advance(grasp[selected])
            new_points += self._claim_waypoints("approach", label, progress)
        return grasp_bonus, new_points * float(self.cfg["waypoint_bonus"])

    def terms(self, proprio, physical: Mapping) -> dict:
        """Evaluate physical terms and read event counters without advancing them."""
        terms, _, _ = self._evaluate(proprio, physical)
        return terms

    def step(self, proprio, physical: Mapping) -> tuple[dict, float]:
        """Advance ordered routes and dwell-gated grasp and placement events."""
        terms, identities, route_state = self._evaluate(proprio, physical)
        self._step_index += 1
        milestone_bonus, waypoint_bonus = self._route_events(*route_state)
        for previous, name in zip((None, "base", "pair2"), _MILESTONES[1:]):
            identity = identities[name]
            if identity is None or identity != self._dwell_ids[name]:
                self._dwell[name] = 0
            self._dwell_ids[name] = identity
            if identity is not None:
                self._dwell[name] += 1
            if (self.milestone_steps[name] is not None
                    or self._dwell[name] < self.cfg["milestone_dwell_steps"]
                    or (previous is not None and self.milestone_steps[previous] is None)):
                continue
            self.milestone_steps[name] = self._step_index
            milestone_bonus += float(self.cfg[f"milestone_{name}"])

        self.milestone_return += milestone_bonus
        self.waypoint_return += waypoint_bonus
        terms["milestone_return"] = self.milestone_return
        terms["waypoint_return"] = self.waypoint_return
        terms["waypoint_counts"] = dict(self.waypoint_counts)
        return terms, milestone_bonus + waypoint_bonus


__all__ = ["DEFAULTS", "REWARD_KEYS", "StackingReward", "validate_reward_config"]
