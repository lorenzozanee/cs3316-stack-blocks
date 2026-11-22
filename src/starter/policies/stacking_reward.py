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

The baseline potential depends on current physical state and reset constants.
The optional support-directed transport potential also reads the episode's
confirmed-grasp destination latch. ``terms()`` never advances latches or events.

The event layer pays ordered crossings at equal fractions of each 3-D line.
Approach uses the measured grasp point and the block centre when the leg
begins. Carry starts at the centre of a confirmed held and lifted block and
ends at the table or current released support top plus the carried half-height.
With support-directed transport on, the confirmed same-arm grasp chooses the
hard tower top, otherwise the nearest eligible tabletop block, otherwise the
fixed table goal. A support that becomes invalid is not reused until a new
confirmed grasp; the current attempt falls back to table shaping without
carry credit. Crossings use linear interpolation between adjacent observations.
The passive transport timeline locks the first geometrically aligned support
and requires opening and quiet, released dwell on that same support. It is not
an official score or a reachability claim.
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
from dataclasses import dataclass
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
    # 40 mm is the tabletop base target circle (>3x inside the 132.26 mm
    # closest initial spawn across all 55 seed-0 layouts) so first placements
    # establish the tower anchor for 2nd/3rd-layer Z targets; tower-top lateral
    # alignment remains governed by _lateral_support (<= 0.25 * lower_side).
    "placement_x": 0.0,
    "placement_y": -0.2,
    "placement_radius": 0.04,
    "placement_soft_scale": 0.15,
    # Gripper calibration from the robot asset (not reward geometry):
    # measured=(joint-low)/(high-low); real joint limits are [0, high], so
    # measured < 0.185 is unreachable and clamps to the q=0 aperture.
    "gripper_scale_low": -0.01,
    "gripper_scale_high": 0.044,
    "gripper_aperture_zero": 0.000807165976524353,
    # Grasp proxies.  The aperture tolerance and centering radius scale with
    # the cube side; both are internal geometric tolerances, not contact
    # proof and not simulator-calibrated solver thresholds. 0.18 stays strictly
    # zero for empty pinched jaws (0.8 mm) and open jaws (>= 67.2 mm) across all
    # 30-40 mm cubes at every rotation while absorbing PhysX jaw compliance.
    "grasp_offset": 0.143,
    "grasp_aperture_fraction": 0.18,
    "grasp_centering_fraction": 0.65,
    # Support and settling tolerances.
    "support_gap_floor": 0.002,
    "support_gap_fraction": 0.08,
    "settle_linear_speed": 0.01,
    "settle_angular_speed": 0.1,
    "upright_cos_min": math.cos(math.radians(15.0)),
    # Event penalties (non-negative magnitudes; subtracted when triggered).
    "empty_grasp_penalty": 0.3,
    "drop_penalty": 0.4,
    "empty_place_penalty": 0.3,
    "empty_tower_intrusion_penalty": 0.4,
    "tower_break_penalty": 1.5,
    "arm_conflict_penalty": 0.4,
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
_RELEASE_REWARD_FRACTION = 0.6
# Empty-gripper, drop, and tower-break penalty geometry and episode caps.
_EMPTY_CARRY_LIFT_FRACTION = 0.15
_EMPTY_CARRY_RADIUS_FRACTION = 1.25
_EMPTY_CARRY_GRACE_STEPS = 40
_EMPTY_CLOSED_MAX = 0.65
_EMPTY_PINCH_MAX = 0.30
_EMPTY_GRASP_REARM = 0.75
_EMPTY_ZONE_EXIT_MARGIN = 0.04
_ARM_CONFLICT_DISTANCE = 0.06
_DROP_CONFIRM_STEPS = 2
_STACK_APPROACH_CLEARANCE_FRACTION = 0.6
_TOWER_BREAK_DWELL_STEPS = 2
_EMPTY_TOWER_BREAK_WINDOW = 5
_MAX_EMPTY_GRASP_PENALTIES = 4
_MAX_DROP_PENALTIES = 4
_MAX_EMPTY_PLACE_PENALTIES = 4
_MAX_EMPTY_TOWER_INTRUSIONS = 4
_MAX_TOWER_BREAK_PENALTIES = 4
_MAX_ARM_CONFLICT_PENALTIES = 4
_PENALTY_KEYS = (
    "empty_grasp", "drop", "empty_place", "empty_tower_intrusion",
    "tower_break", "empty_tower_break", "arm_conflict",
)

_MILESTONES = ("grasp", "base", "pair2", "pair3")
_WAYPOINT_ROUTES = tuple(("approach", label) for label in BLOCK_LABELS) + tuple(
    ("carry", label) for label in BLOCK_LABELS)
_WEIGHT_KEYS = (
    "reach_weight", "manipulation_weight", "base_weight", "pair_weight",
    "release_weight", "milestone_grasp",
    "milestone_base", "milestone_pair2", "milestone_pair3", "waypoint_bonus",
    "empty_grasp_penalty", "drop_penalty", "empty_place_penalty",
    "empty_tower_intrusion_penalty", "tower_break_penalty",
    "arm_conflict_penalty",
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

    Every key from :data:`REWARD_KEYS` is required. The experiment-only
    transport modes are opt-in and mutually exclusive.
    """
    if not isinstance(cfg, Mapping):
        raise ValueError("Reward configuration must be a mapping.")
    for key in ("transport_to_support", "dense_transport_reward"):
        if not isinstance(cfg.get(key, False), bool):
            raise ValueError(f"{key} must be a bool.")
    if cfg.get("transport_to_support") and cfg.get("dense_transport_reward"):
        raise ValueError("Select one of transport_to_support and dense_transport_reward.")
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
        self.origin_target = target.copy()
        self.previous = start.copy()
        self.count = count
        self.nominal_radius = radius
        self.passed = 0
        self.retarget(target)

    def retarget(self, target: np.ndarray) -> None:
        self.target = target.copy()
        delta = target - self.start
        length = float(np.linalg.norm(delta))
        self.axis = delta / length if length > 1e-9 else np.zeros_like(delta)
        self.spacing = length / (self.count + 1)
        self.radius = min(self.nominal_radius, 0.45 * self.spacing)

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

@dataclass
class _Frame:
    pos: dict
    grasp: np.ndarray
    half_height: dict
    half_x: dict
    half_y: dict
    bottom: dict
    hold: dict
    associated_arms: dict
    grasp_candidates: dict
    support: tuple
    base_ok: dict
    base_intact: dict
    tabletop_released: dict
    edge_hard: dict
    edge_intact: dict
    measured: np.ndarray
    released: dict
    upright: dict
    upright_score: dict
    inside_table: dict
    table_gap_tol: dict
    support_top: dict
    distances: dict
    closure: dict


@dataclass
class _CarryDestination:
    arm: int
    anchor: tuple[str, ...]  # Empty is the fixed tabletop target.
    hard_tower: bool = False
    invalid: bool = False


@dataclass(kw_only=True)
class _DenseCarry(_CarryDestination):
    row: dict
    previous: np.ndarray | None = None
    drop_pending: int = 0


@dataclass
class _TransportTrace:
    row: dict
    held_opening: float
    dwell: int = 0
    interrupted: bool = False
    done: bool = False


class StackingReward:
    """Per-episode reward geometry.  Pure terms plus one-time event latches."""

    def __init__(self, reset_state: Mapping, reset_proprio, cfg: Mapping):
        validate_reward_config(cfg)
        self.cfg = {key: cfg[key] for key in REWARD_KEYS}
        self.cfg["milestone_dwell_steps"] = int(cfg["milestone_dwell_steps"])
        self._transport_to_support = cfg.get("transport_to_support", False)
        self._dense_transport_reward = cfg.get("dense_transport_reward", False)
        self._settled_anchor: tuple[str, ...] = ()
        self._best_distance: dict[tuple[str, str, tuple[str, ...]], float] = {}
        self._carry_scales: dict[tuple[str, str, tuple[str, ...]], float] = {}
        self._progress_credit: dict[tuple[str, str], float] = {}
        self.release_steps = {name: None for name in _MILESTONES[1:]}
        self._released_placements: dict[str, tuple[tuple[str, ...], dict]] = {}
        self._dense_approach_return = self._dense_transport_return = 0.0
        self.transport_attempts: list[dict] = []
        self._transport_traces: dict[str, _TransportTrace] = {}
        self._carry_destinations: dict[str, _CarryDestination] = {}
        self._reset_proprio = _vector(reset_proprio, ACTION_DIM, "reset_proprio")
        reset_state = _mapping(reset_state, "reset_state")
        self._reset_poses = _block_poses(reset_state, "reset_state")
        self._extents = _block_extents(reset_state)
        self._table = _table(reset_state)
        # Validate the remaining reset fields now; they are part of the
        # episode contract even though only current state is scored later.
        _block_velocities(reset_state, "reset_state")
        reset_measured = _measured_grippers(reset_state, "reset_state")

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
        self.penalty_return = 0.0
        self.penalty_counts = {key: 0 for key in _PENALTY_KEYS}
        self._intact_anchor: tuple[str, ...] = ()
        self._tower_break_dwell = 0
        self._arm_last_carry_step: list[int | None] = [None, None]
        self._arm_last_empty_tower_step: list[int | None] = [None, None]
        self._arm_zone_state: list[str] = ["outside", "outside"]
        self._arm_tower_intruding: list[bool] = [False, False]
        self._arm_prev_measured: list[float] = [float(reset_measured[0]), float(reset_measured[1])]
        self._arm_empty_grasp_armed: list[bool] = [
            float(reset_measured[0]) >= _EMPTY_GRASP_REARM,
            float(reset_measured[1]) >= _EMPTY_GRASP_REARM,
        ]
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
        self._arms_conflicting: bool = (
            math.dist(reset_grasp[0], reset_grasp[1]) <= _ARM_CONFLICT_DISTANCE
        )
        self._previous_grasp = reset_grasp
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
        motion = physical.get("settle_speeds")
        if motion is None:
            velocities = _block_velocities(physical, "physical")
        else:
            motion = _labels_map(motion, "physical.settle_speeds")
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
            if motion is None:
                linear_norm = float(np.linalg.norm(velocities[label]["linear"]))
                angular_norm = float(np.linalg.norm(velocities[label]["angular"]))
            else:
                linear_norm, angular_norm = _vector(motion[label], 2, f"settle_speeds.{label}")
                if min(linear_norm, angular_norm) < 0:
                    raise ValueError(f"Negative settling speed for {label}.")
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
        released_at_table_height = {
            label: (_hard_gap_ok(table_gap[label], table_gap_tol[label])
                    and upright[label] >= cos_min and quiet_ok[label]
                    and released_hard[label])
            for label in BLOCK_LABELS
        }
        base_intact = {
            label: bool(distance_xy[label] <= radius + _EMPTY_ZONE_EXIT_MARGIN
                        and inside_table[label]
                        and _hard_gap_ok(table_gap[label], table_gap_tol[label])
                        and upright[label] >= cos_min)
            for label in BLOCK_LABELS
        }
        base_ok = {
            label: bool(distance_xy[label] <= radius and inside_table[label]
                        and _hard_gap_ok(table_gap[label], table_gap_tol[label])
                        and upright[label] >= cos_min
                        and quiet_ok[label] and released_hard[label])
            for label in BLOCK_LABELS
        }
        tabletop_released = {
            label: bool(released_at_table_height[label]
                        and self._table["x_min"] + half_x[label] <= pos[label][0]
                        <= self._table["x_max"] - half_x[label]
                        and self._table["y_min"] + half_y[label] <= pos[label][1]
                        <= self._table["y_max"] - half_y[label])
            for label in BLOCK_LABELS
        }
        phi_base = float(self.cfg["base_weight"]) * (max(base_score.values()) - 1.0)

        # Directed support edges.  Upper means strictly higher centre; equal
        # z is a side-by-side or tabletop co-planar pair and carries no edge.
        geometry_edge: dict[tuple[str, str], float] = {}
        quiet_edge: dict[tuple[str, str], float] = {}
        release_edge: dict[tuple[str, str], float] = {}
        edge_hard: dict[tuple[str, str], bool] = {}
        edge_intact: dict[tuple[str, str], bool] = {}
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
            intact = bool(
                _hard_gap_ok(gap, tolerance)
                and lateral_kernel >= _HARD_LATERAL_KERNEL_MIN - 1e-12
                and upright[lower] >= cos_min
                and upright[upper] >= cos_min)
            edge_intact[(lower, upper)] = intact
            edge_hard[(lower, upper)] = bool(
                intact
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

        phi_pair = float(self.cfg["pair_weight"]) * (pair_level - 2.0)

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

        support_gaps = []
        for first, second in combinations(BLOCK_LABELS, 2):
            if pos[second][2] >= pos[first][2]:
                lower, upper = first, second
            else:
                lower, upper = second, first
            gap = float(bottom[upper] - (pos[lower][2] + half_height[lower]))
            support_gaps.append({"lower": lower, "upper": upper, "gap": gap})

        terms = {
            "phi_base": float(phi_base),
            "phi_pair": float(phi_pair),
            "phi_release": float(phi_release),
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
            "penalty_return": float(self.penalty_return),
            "penalty_counts": dict(self.penalty_counts),
            "waypoint_return": float(self.waypoint_return),
            "waypoint_counts": dict(self.waypoint_counts),
        }
        identities = {"base": base_id, "pair2": pair2_id, "pair3": pair3_id}
        frame = _Frame(pos, grasp, half_height, half_x, half_y, bottom, hold, associated_arms,
                       grasp_candidates, support, base_ok, base_intact, tabletop_released,
                       edge_hard, edge_intact, measured, released_hard, upright, upright_score,
                       inside_table, table_gap_tol, support_top, distances, grasp_closure_by_block)
        return terms, identities, frame

    def _surface(self, frame: _Frame, lower: str | None) -> tuple[np.ndarray, float]:
        if lower is None:
            return self._target, self._table["top_z"]
        return frame.pos[lower][:2], float(frame.pos[lower][2] + frame.half_height[lower])

    def _lateral_support(self, lower: str, lateral: float) -> float:
        return _clip01((self._half_widths[lower] - lateral)
                       / (_LATERAL_FULL_FRACTION * self._sides[lower]))

    def _carry_goal(self, frame: _Frame, lower: str | None, carried: str) -> np.ndarray:
        xy, z = self._surface(frame, lower)
        clearance = 0.0
        if lower is not None and self._dense_transport_reward:
            lateral = math.dist(frame.pos[carried][:2], xy)
            side = self._sides[lower]
            clearance = (_STACK_APPROACH_CLEARANCE_FRACTION * side
                         * _clip01((lateral - 0.5 * side) / side))
        return np.array([*xy, z + frame.half_height[carried] + clearance])

    def _occupied(self, frame: _Frame, lower: str | None, carried: str) -> bool:
        xy, z = self._surface(frame, lower)
        top = z + 2.0 * frame.half_height[carried]
        for other in BLOCK_LABELS:
            if other in (carried, lower):
                continue
            if frame.edge_hard.get((lower, other), False):
                return True
            # Projected AABBs: another body may block the planned slot even when
            # it does not sit on the lower block (unequal sizes).
            if (frame.bottom[other] < top - 1e-9
                    and frame.pos[other][2] + frame.half_height[other] > z + 1e-9
                    and abs(frame.pos[other][0] - xy[0])
                    < frame.half_x[carried] + frame.half_x[other] - 1e-9
                    and abs(frame.pos[other][1] - xy[1])
                    < frame.half_y[carried] + frame.half_y[other] - 1e-9):
                return True
        return False

    def _destination_valid(self, frame: _Frame, carried: str, dest: _CarryDestination) -> bool:
        if dest.invalid:
            return False
        anchor = dest.anchor
        if not anchor:
            return True
        base_ok = frame.base_ok if dest.hard_tower else frame.tabletop_released
        return (base_ok[anchor[0]] and not self._occupied(frame, anchor[-1], carried)
                and all(frame.edge_hard.get(edge, False) for edge in zip(anchor, anchor[1:])))

    def _stage_terms(self, terms: dict, frame: _Frame) -> None:
        radius = float(self.cfg["placement_radius"])
        soft = float(self.cfg["placement_soft_scale"])
        stage_reach_weight = float(self.cfg["reach_weight"]) / len(BLOCK_LABELS)
        manipulation_weight = float(self.cfg["manipulation_weight"])
        closure_weight = float(self.cfg["grasp_closure_weight"])
        reach_scale = float(self.cfg["reach_scale"])
        next_reach = next_manipulation = 0.0
        best_score = -1.0
        # Both terms follow the confirmed carried block when there is one.
        active = {label: dest for label, dest in self._carry_destinations.items()
                  if label not in frame.support and frame.hold[dest.arm, label] >= _GRASP_HOLD_MIN}
        for label in active or BLOCK_LABELS:
            if label in frame.support:
                continue
            destination = active.get(label)
            if destination is None:
                lower = frame.support[-1] if frame.support else None
            else:
                lower = (destination.anchor[-1] if destination.anchor
                         and self._destination_valid(frame, label, destination) else None)
            xy, z = self._surface(frame, lower)
            lateral = float(np.linalg.norm(frame.pos[label][:2] - xy))
            gap = frame.bottom[label] - z
            clearance = _LIFT_RAMP_FRACTION * self._sides[label]
            raised = _clip01((frame.bottom[label] - frame.support_top[label]) / clearance)
            if lower is None:
                lateral = max(0.0, lateral - radius)
                tolerance = frame.table_gap_tol[label]
                aligned = (_clip01(1.0 - lateral / clearance)
                           if frame.inside_table[label] else 0.0)
            else:
                tolerance = self._gap_tolerance(lower, label)
                aligned = self._lateral_support(lower, lateral)
            aligned *= frame.upright_score[label]
            descent = aligned * _clip01(
                1.0 - max(0.0, -gap, gap - clearance) / clearance)
            tracking = 1.0 - math.tanh(math.hypot(lateral, gap) / soft)
            held = (frame.hold[destination.arm, label] if destination is not None
                    else max(frame.hold[arm, label] for arm in (0, 1)))
            grasp_lift_carry = (closure_weight * frame.closure[label]
                               + max(raised, descent) * held * (0.25 + 0.25 * tracking))
            seated = aligned * _clip01(
                (tolerance - abs(gap)) / ((1.0 - _HARD_GAP_FRACTION) * tolerance))
            reach = max(seated, 1.0 - math.tanh(frame.distances[label] / reach_scale))
            manipulation = seated + (1.0 - seated) * grasp_lift_carry
            score = stage_reach_weight * reach + manipulation_weight * manipulation
            if score > best_score:
                best_score = score
                next_reach, next_manipulation = reach, manipulation
        stable_level = len(frame.support)
        deficit = len(BLOCK_LABELS) - stable_level
        phi_reach = stage_reach_weight * (next_reach - deficit)
        phi_manipulation = manipulation_weight * (next_manipulation - deficit)
        phi = phi_reach + phi_manipulation + terms["phi_base"] + terms["phi_pair"] + terms["phi_release"]
        if not math.isfinite(phi):
            raise ValueError("Reward potential is non-finite.")
        terms.update(phi=float(phi), phi_reach=float(phi_reach),
                     phi_manipulation=float(phi_manipulation),
                     manipulation_progress=float(stable_level + next_manipulation))

    def _path(self, start: np.ndarray, target: np.ndarray) -> _WaypointPath:
        return _WaypointPath(start, target, self._waypoint_count, self._waypoint_radius)

    def _claim_waypoints(self, kind: str, label: str, reached: int) -> int:
        key = f"{kind}:{label}"
        new = max(0, reached - self.waypoint_counts[key])
        self.waypoint_counts[key] += new
        return new

    def _grasp_events(self, frame: _Frame) -> tuple[float, dict[str, int]]:
        bonus = 0.0
        confirmed = {}
        for label in BLOCK_LABELS:
            for arm in (0, 1):
                key = f"{arm}:{label}"
                if label in frame.support or arm not in frame.associated_arms[label]:
                    self._active_arrivals.pop(key, None)
                elif key not in self._active_arrivals:
                    counts = self.grasp_attempts.setdefault(
                        key, {"arrivals": 0, "confirmed": 0, "steps_to_grasp": 0})
                    counts["arrivals"] += 1
                    self._active_arrivals[key] = self._step_index
            arm = frame.grasp_candidates.get(label)
            if arm is None or label in frame.support:
                self._grasp_dwell.pop(label, None)
                continue
            previous_arm, dwell = self._grasp_dwell.get(label, (None, 0))
            dwell = dwell + 1 if arm == previous_arm else 1
            self._grasp_dwell[label] = (arm, dwell)
            if dwell < self.cfg["milestone_dwell_steps"]:
                continue
            confirmed[label] = arm
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
                bonus += float(self.cfg["milestone_grasp"])
        return bonus, confirmed

    def _update_destinations(self, frame: _Frame, confirmed: dict[str, int]) -> None:
        for label, dest in list(self._carry_destinations.items()):
            if label in frame.support or frame.hold[dest.arm, label] < _GRASP_HOLD_MIN:
                del self._carry_destinations[label]
                self._carry.pop(label, None)
            elif not self._destination_valid(frame, label, dest):
                dest.invalid = True
                self._carry.pop(label, None)
        for label, arm in confirmed.items():
            if label in self._carry_destinations:
                continue
            anchors = [frame.support] if frame.support else self._soft_anchors(frame, label)
            if anchors:
                anchor = min(anchors, key=lambda choice: (
                    np.linalg.norm(frame.pos[label] - self._carry_goal(frame, choice[-1], label)),
                    choice))
            else:
                anchor = ()
            self._carry_destinations[label] = _CarryDestination(arm, anchor, bool(frame.support))

    def _progress_bonus(self, phase: str, label: str, anchor: tuple[str, ...],
                        previous: float, distance: float, scale: float) -> float:
        key = (phase, label, anchor)
        closest = min(previous, self._best_distance.get(key, previous))
        self._best_distance[key] = min(closest, distance)
        credit_key = (phase, label)
        paid = self._progress_credit.get(credit_key, 0.0)
        cap = self._waypoint_count * float(self.cfg["waypoint_bonus"])
        if phase == "carry":
            carry_scale = self._carry_scales.setdefault(key, max(previous, scale))
            gain = (closest - distance) / carry_scale
        else:
            gain = math.tanh(closest / scale) - math.tanh(distance / scale)
        credit = min(max(0.0, cap - paid), cap * max(0.0, gain))
        self._progress_credit[credit_key] = paid + credit
        return credit

    def _dense_approach(self, frame: _Frame) -> float:
        free_arms = [arm for arm in (0, 1) if all(
            frame.hold[arm, label] < _GRASP_HOLD_MIN for label in BLOCK_LABELS)]
        reward = 0.0
        anchor = self._available_anchor(frame)
        for label in BLOCK_LABELS:
            if not free_arms or label in anchor or label in frame.support:
                continue
            if label in self._carry_destinations or any(
                    frame.hold[arm, label] >= _GRASP_HOLD_MIN for arm in (0, 1)):
                continue
            target = frame.pos[label]
            previous = min(math.dist(self._previous_grasp[arm], target) for arm in free_arms)
            distance = min(math.dist(frame.grasp[arm], target) for arm in free_arms)
            reward += self._progress_bonus("approach", label, (), previous, distance,
                                           float(self.cfg["reach_scale"]))
        self._previous_grasp = frame.grasp
        return reward

    def _near_release_goal(self, frame: _Frame, label: str, lower: str | None) -> bool:
        xy, top = self._surface(frame, lower)
        lateral = math.dist(frame.pos[label][:2], xy)
        gap = frame.bottom[label] - top
        tolerance = frame.table_gap_tol[label] if lower is None else self._gap_tolerance(lower, label)
        centered = ((lateral <= self.cfg["placement_radius"] and frame.inside_table[label])
                    if lower is None else
                    self._lateral_support(lower, lateral) >= _HARD_LATERAL_KERNEL_MIN - 1e-12)
        return (centered and frame.upright[label] >= self.cfg["upright_cos_min"]
                and -_HARD_GAP_FRACTION * tolerance - 1e-12 <= gap
                <= _LIFT_RAMP_FRACTION * self._sides[label] + 1e-12)

    @staticmethod
    def _end_attempt(attempt: _DenseCarry, reason: str) -> None:
        attempt.invalid = True
        attempt.row["end_reason"] = reason

    def _advance_dense_carry(self, frame: _Frame, label: str,
                             attempt: _DenseCarry) -> tuple[float, float]:
        anchor, arm = attempt.anchor, attempt.arm
        lower = anchor[-1] if anchor else None
        held = frame.hold[arm, label] >= _GRASP_HOLD_MIN
        available = self._available_anchor(frame)
        placement = _MILESTONES[len(anchor) + 1]
        settled_self = (not held and available == anchor + (label,)
                        and self.release_steps[placement] is None
                        and attempt.row["aligned_step"] is not None)
        carrying = held and (attempt.previous is not None or attempt.drop_pending > 0)
        if anchor != self._settled_anchor[:len(anchor)]:
            self._end_attempt(attempt, "anchor")
            return 0.0, 0.0
        if not settled_self and len(available) > len(anchor):
            if not carrying or label in available:
                self._end_attempt(attempt, "anchor")
                return 0.0, 0.0
            anchor = attempt.anchor = available
            attempt.hard_tower = True
            lower = anchor[-1]
            placement = _MILESTONES[len(anchor) + 1]
            attempt.row.update(lower=lower, target_lower=lower, aligned_step=None)
        if ((not settled_self and available != anchor)
                or not self._destination_valid(frame, label, attempt)
                or (lower is None and self._occupied(frame, None, label))):
            if carrying:
                # Keep the identity across transient instability; skip this motion.
                attempt.drop_pending = 0
                attempt.previous = frame.pos[label].copy()
                return 0.0, 0.0
            self._end_attempt(attempt, "destination")
            return 0.0, 0.0

        near_goal = self._near_release_goal(frame, label, lower)
        if held:
            attempt.drop_pending = 0
            if near_goal and attempt.row["aligned_step"] is None:
                attempt.row["aligned_step"] = self._step_index
            if attempt.previous is None:
                attempt.previous = frame.pos[label].copy()
                return 0.0, 0.0
            # Compare both positions against the current goal so moving the
            # support alone cannot pay. Use only new closest progress.
            target = self._carry_goal(frame, lower, label)
            distance = math.dist(frame.pos[label], target)
            previous = math.dist(attempt.previous, target)
            attempt.previous = frame.pos[label].copy()
            return self._progress_bonus("carry", label, anchor, previous, distance,
                                        float(self.cfg["placement_soft_scale"])), 0.0

        # Free block: keep waiting only at the goal after this arm brought it
        # there; dropping elsewhere or moving to another block ends the attempt.
        attempt.previous = None
        if frame.hold[1 - arm, label] >= _GRASP_HOLD_MIN:
            self._end_attempt(attempt, "other_arm")
            return 0.0, 0.0
        if any(arm in frame.associated_arms[other]
               for other in BLOCK_LABELS if other != label):
            self._end_attempt(attempt, "other_block")
            return 0.0, 0.0
        if not near_goal or (attempt.row["aligned_step"] is None
                             and arm not in frame.associated_arms[label]):
            attempt.drop_pending += 1
            if (float(frame.measured[arm]) >= _RELEASE_MIN
                    or attempt.drop_pending >= _DROP_CONFIRM_STEPS):
                self._end_attempt(attempt, "dropped")
            return 0.0, 0.0
        attempt.drop_pending = 0
        if attempt.row["aligned_step"] is None:
            attempt.row["aligned_step"] = self._step_index
        if float(frame.measured[arm]) < _RELEASE_MIN or not frame.released[label]:
            return 0.0, 0.0
        self._end_attempt(attempt, "released")
        attempt.row["opened_step"] = self._step_index
        if self.milestone_steps[placement] is None or settled_self:
            if self.milestone_steps[placement] is None:
                self._dwell[placement] = 0
                self._released_placements[placement] = (anchor + (label,), attempt.row)
            else:
                attempt.row["settled_step"] = self.milestone_steps[placement]
            if self.release_steps[placement] is None:
                self.release_steps[placement] = self._step_index
                return 0.0, _RELEASE_REWARD_FRACTION * float(self.cfg[f"milestone_{placement}"])
        return 0.0, 0.0

    @staticmethod
    def _stable_prefix(frame: _Frame, anchor: tuple[str, ...]) -> tuple[str, ...]:
        if anchor and not frame.base_ok[anchor[0]]:
            return ()
        for index, edge in enumerate(zip(anchor, anchor[1:]), 1):
            if not frame.edge_hard.get(edge, False):
                return anchor[:index]
        return anchor

    @staticmethod
    def _intact_prefix(frame: _Frame, anchor: tuple[str, ...]) -> tuple[str, ...]:
        if anchor and not frame.base_intact[anchor[0]]:
            return ()
        for index, edge in enumerate(zip(anchor, anchor[1:]), 1):
            if not frame.edge_intact.get(edge, False):
                return anchor[:index]
        return anchor

    def _available_anchor(self, frame: _Frame) -> tuple[str, ...]:
        return self._stable_prefix(frame, self._settled_anchor)

    def _dense_events(
        self, frame: _Frame, confirmed: dict[str, int]
    ) -> tuple[float, float, float, set[int], set[int]]:
        carry_bonus = placement_bonus = drop_penalty = 0.0
        active_arms = {attempt.arm for attempt in self._carry_destinations.values()}
        ending_arms: set[int] = set()
        for label, attempt in list(self._carry_destinations.items()):
            carry, placement = self._advance_dense_carry(frame, label, attempt)
            carry_bonus += carry
            placement_bonus += placement
            if attempt.invalid:
                reason = attempt.row["end_reason"]
                if reason == "released":
                    ending_arms.add(attempt.arm)
                elif reason == "dropped":
                    ending_arms.add(attempt.arm)
                    if self.penalty_counts["drop"] < _MAX_DROP_PENALTIES:
                        self.penalty_counts["drop"] += 1
                        drop_penalty += float(self.cfg["drop_penalty"])
                del self._carry_destinations[label]

        anchor = self._available_anchor(frame)
        for label, arm in confirmed.items():
            if label in self._carry_destinations or label in anchor:
                continue
            lower = anchor[-1] if anchor else None
            aligned = self._step_index if self._near_release_goal(frame, label, lower) else None
            row = dict(arm=arm, carried=label, target_lower=lower, lower=lower,
                       grasp_step=self._step_index, aligned_step=aligned,
                       opened_step=None, settled_step=None, end_reason=None)
            self.transport_attempts.append(row)
            self._carry_destinations[label] = _DenseCarry(
                arm, anchor, hard_tower=bool(anchor), row=row, previous=frame.pos[label].copy())
            active_arms.add(arm)
        return carry_bonus, placement_bonus, drop_penalty, active_arms, ending_arms

    def _arm_has_block(self, frame: _Frame, arm: int, protected: set[str],
                       active_arms: set[int], ending_arms: set[int]) -> bool:
        if arm in active_arms or arm in ending_arms:
            return True
        if any(frame.hold[arm, label] >= _GRASP_HOLD_MIN for label in BLOCK_LABELS):
            return True
        return any(
            label not in protected
            and (frame.bottom[label] - frame.support_top[label] + 1e-12
                 >= _EMPTY_CARRY_LIFT_FRACTION * self._sides[label])
            and (math.dist(frame.grasp[arm], frame.pos[label])
                 <= _EMPTY_CARRY_RADIUS_FRACTION * self._sides[label])
            for label in BLOCK_LABELS
        )

    def _arm_engaging_unplaced(self, frame: _Frame, arm: int, protected: set[str]) -> bool:
        return any(
            label not in protected
            and (arm in frame.associated_arms[label]
                 or math.dist(frame.grasp[arm], frame.pos[label])
                 <= _EMPTY_CARRY_RADIUS_FRACTION * self._sides[label])
            for label in BLOCK_LABELS
        )

    def _in_tower_volume(self, frame: _Frame, arm: int, anchor: tuple[str, ...],
                         margin: float = 0.0) -> bool:
        z = float(frame.grasp[arm][2])
        for label in anchor:
            xy_radius = self._sides[label] + margin
            z_low = frame.bottom[label] - 0.01 - margin
            z_high = frame.pos[label][2] + frame.half_height[label] + 0.5 * self._sides[label] + margin
            if math.dist(frame.grasp[arm][:2], frame.pos[label][:2]) <= xy_radius and z_low <= z <= z_high:
                return True
        return False

    def _dense_penalties(self, frame: _Frame, active_arms: set[int],
                         ending_arms: set[int]) -> float:
        penalty = 0.0
        available = self._available_anchor(frame)
        anchor = available or self._intact_anchor
        protected = set(self._intact_anchor) | set(frame.support) | set(available)
        lower = anchor[-1] if anchor else None
        goal_xy, surface_z = self._surface(frame, lower)
        max_side = max(self._sides.values())
        zone_radius = float(self.cfg["placement_radius"]) + 0.5 * max_side
        zone_top = surface_z + 2.0 * max_side
        zone_low = surface_z - 0.01

        for arm in (0, 1):
            measured = float(frame.measured[arm])
            if self._arm_has_block(frame, arm, protected, active_arms, ending_arms):
                self._arm_last_carry_step[arm] = self._step_index
                self._arm_zone_state[arm] = "outside"
                self._arm_tower_intruding[arm] = False
                self._arm_empty_grasp_armed[arm] = False
                continue

            if measured >= _EMPTY_GRASP_REARM:
                self._arm_empty_grasp_armed[arm] = True

            xy_dist = math.dist(frame.grasp[arm][:2], goal_xy)
            z = float(frame.grasp[arm][2])
            in_zone = xy_dist <= zone_radius and zone_low <= z <= zone_top
            outside_zone = (
                xy_dist > zone_radius + _EMPTY_ZONE_EXIT_MARGIN
                or z > zone_top + _EMPTY_ZONE_EXIT_MARGIN
                or z < zone_low - _EMPTY_ZONE_EXIT_MARGIN
            )
            if self._arm_zone_state[arm] == "penalized" and outside_zone:
                self._arm_zone_state[arm] = "outside"
            if self._arm_tower_intruding[arm] and (
                not self._intact_anchor
                or not self._in_tower_volume(frame, arm, self._intact_anchor, _EMPTY_ZONE_EXIT_MARGIN)
            ):
                self._arm_tower_intruding[arm] = False

            in_grace = (self._arm_last_carry_step[arm] is not None
                        and self._step_index - self._arm_last_carry_step[arm] <= _EMPTY_CARRY_GRACE_STEPS)
            engaging = self._arm_engaging_unplaced(frame, arm, protected)
            in_tower = bool(self._intact_anchor
                            and self._in_tower_volume(frame, arm, self._intact_anchor))

            # Placement-zone and tower-volume events own the arm while it is
            # there; the midair empty-grasp check applies only outside both.
            if not in_grace and not engaging:
                opening_empty = (self._arm_prev_measured[arm] < _RELEASE_MIN
                                 and measured >= _RELEASE_MIN)
                empty_place_action = measured <= _EMPTY_CLOSED_MAX or opening_empty
                if in_tower:
                    self._arm_last_empty_tower_step[arm] = self._step_index
                    if not self._arm_tower_intruding[arm]:
                        self._arm_tower_intruding[arm] = True
                        if self.penalty_counts["empty_tower_intrusion"] < _MAX_EMPTY_TOWER_INTRUSIONS:
                            self.penalty_counts["empty_tower_intrusion"] += 1
                            penalty += float(self.cfg["empty_tower_intrusion_penalty"])
                elif (self._arm_zone_state[arm] != "penalized"
                      and in_zone and empty_place_action):
                    self._arm_zone_state[arm] = "penalized"
                    if self.penalty_counts["empty_place"] < _MAX_EMPTY_PLACE_PENALTIES:
                        self.penalty_counts["empty_place"] += 1
                        penalty += float(self.cfg["empty_place_penalty"])

            if (self._arm_empty_grasp_armed[arm] and not in_grace
                    and not engaging and not in_zone and not in_tower):
                if measured <= _EMPTY_PINCH_MAX:
                    self._arm_empty_grasp_armed[arm] = False
                    if self.penalty_counts["empty_grasp"] < _MAX_EMPTY_GRASP_PENALTIES:
                        self.penalty_counts["empty_grasp"] += 1
                        penalty += float(self.cfg["empty_grasp_penalty"])

        self._arm_prev_measured = [float(frame.measured[0]), float(frame.measured[1])]

        ee_dist = math.dist(frame.grasp[0], frame.grasp[1])
        same_block_hold = any(
            frame.hold[0, label] >= _GRASP_HOLD_MIN
            and frame.hold[1, label] >= _GRASP_HOLD_MIN
            for label in BLOCK_LABELS
        )
        conflict_now = ee_dist <= _ARM_CONFLICT_DISTANCE or same_block_hold
        if (self._arms_conflicting
                and ee_dist > _ARM_CONFLICT_DISTANCE + _EMPTY_ZONE_EXIT_MARGIN
                and not same_block_hold):
            self._arms_conflicting = False
        if conflict_now and not self._arms_conflicting:
            self._arms_conflicting = True
            if self.penalty_counts["arm_conflict"] < _MAX_ARM_CONFLICT_PENALTIES:
                self.penalty_counts["arm_conflict"] += 1
                penalty += float(self.cfg["arm_conflict_penalty"])

        if len(available) > len(self._intact_anchor) and self._intact_prefix(frame, available) == available:
            self._intact_anchor = available
            self._tower_break_dwell = 0
        elif self._intact_anchor:
            intact = self._intact_prefix(frame, self._intact_anchor)
            if len(intact) == len(self._intact_anchor):
                self._tower_break_dwell = 0
            else:
                self._tower_break_dwell += 1
                if self._tower_break_dwell >= _TOWER_BREAK_DWELL_STEPS:
                    self._intact_anchor = intact
                    self._tower_break_dwell = 0
                    if self.penalty_counts["tower_break"] < _MAX_TOWER_BREAK_PENALTIES:
                        self.penalty_counts["tower_break"] += 1
                        penalty += float(self.cfg["tower_break_penalty"])
                        if any(step is not None
                               and self._step_index - step <= _EMPTY_TOWER_BREAK_WINDOW
                               for step in self._arm_last_empty_tower_step):
                            self.penalty_counts["empty_tower_break"] += 1
        return penalty

    def _soft_anchors(self, frame: _Frame, carried: str) -> list[tuple[str, ...]]:
        """Released tabletop blocks and their released, edge-stable top blocks."""
        anchors = []
        for lower in BLOCK_LABELS:
            if lower == carried or self._occupied(frame, lower, carried):
                continue
            if frame.tabletop_released[lower]:
                anchors.append((lower,))
                continue
            anchors.extend((base, lower) for base in BLOCK_LABELS
                           if base != carried and frame.tabletop_released[base]
                           and frame.edge_hard.get((base, lower), False))
        return anchors

    def _diagnostic_support(self, frame: _Frame, lower: str, carried: str) -> bool:
        return frame.tabletop_released[lower] or any(
            base != carried and frame.tabletop_released[base]
            and frame.edge_hard.get((base, lower), False) for base in BLOCK_LABELS)

    def _transport_aligned(self, frame: _Frame, lower: str, carried: str) -> bool:
        if not self._diagnostic_support(frame, lower, carried):
            return False
        xy, z = self._surface(frame, lower)
        lateral = float(np.linalg.norm(frame.pos[carried][:2] - xy))
        gap = frame.bottom[carried] - z
        return bool(
            frame.upright[carried] >= self.cfg["upright_cos_min"]
            and self._lateral_support(lower, lateral) >= _HARD_LATERAL_KERNEL_MIN - 1e-12
            and (-_HARD_GAP_FRACTION * self._gap_tolerance(lower, carried) - 1e-12
                 <= gap <= _LIFT_RAMP_FRACTION * self._sides[carried] + 1e-12))

    def _advance_trace(self, frame: _Frame, label: str, trace: _TransportTrace) -> None:
        row = trace.row
        arm, lower = row["arm"], row["lower"]
        held = frame.hold[arm, label] >= _GRASP_HOLD_MIN
        if row["opened_step"] is None:
            if lower is not None and not self._diagnostic_support(frame, lower, label):
                trace.done = True
                return
            if held and lower is None:
                for candidate in BLOCK_LABELS:
                    if candidate != label and self._transport_aligned(frame, candidate, label):
                        row["lower"] = candidate
                        row["aligned_step"] = self._step_index
                        break
            if held:
                if trace.interrupted:
                    # Re-closing after a lost hold needs a fresh confirmed grasp;
                    # otherwise a noisy aperture could credit the old attempt.
                    trace.done = True
                    return
                trace.held_opening = float(frame.measured[arm])
                return
            trace.interrupted = True
            opening = frame.measured[arm]
            if (lower is None or opening <= trace.held_opening
                    or frame.hold[1 - arm, label] >= _GRASP_HOLD_MIN
                    or arm not in frame.associated_arms[label]
                    or not self._transport_aligned(frame, lower, label)):
                trace.done = True
                return
            # The measured jaw can lose the aperture match before reaching open.
            if opening < _RELEASE_MIN:
                return
            if not frame.released[label]:
                trace.done = True
                return
            row["opened_step"] = self._step_index
        if not frame.released[label] or not self._transport_aligned(frame, lower, label):
            trace.done = True
            return
        if frame.edge_hard.get((lower, label), False):
            trace.dwell += 1
            if trace.dwell >= self.cfg["milestone_dwell_steps"]:
                row["settled_step"] = self._step_index
                trace.done = True
        else:
            trace.dwell = 0

    def _transport_events(self, frame: _Frame, confirmed: dict[str, int]) -> None:
        for label, trace in list(self._transport_traces.items()):
            if (trace.interrupted and label in confirmed
                    and self._grasp_dwell[label][1] == self.cfg["milestone_dwell_steps"]):
                del self._transport_traces[label]
                continue
            if not trace.done:
                self._advance_trace(frame, label, trace)
            if trace.done and (trace.row["opened_step"] is not None
                               or frame.hold[trace.row["arm"], label] < _GRASP_HOLD_MIN):
                del self._transport_traces[label]
        for label, arm in confirmed.items():
            if label in self._transport_traces:
                continue
            destination = self._carry_destinations.get(label)
            anchor = destination.anchor if destination is not None else frame.support
            row = dict(arm=arm, carried=label, target_lower=anchor[-1] if anchor else None,
                       lower=None, grasp_step=self._step_index,
                       aligned_step=None, opened_step=None, settled_step=None)
            trace = _TransportTrace(row, float(frame.measured[arm]))
            self.transport_attempts.append(row)
            self._transport_traces[label] = trace
            self._advance_trace(frame, label, trace)

    def _route_events(self, frame: _Frame, confirmed: dict[str, int]) -> float:
        pos, grasp, support = frame.pos, frame.grasp, frame.support
        if support != self._route_support:
            self._route_support = support
            self._approach.clear()
            self._approach_arm.clear()
            if not self._transport_to_support:
                self._carry.clear()

        new_points = 0
        placed_levels = sum(self.milestone_steps[name] is not None for name in _MILESTONES[1:])
        carry_allowed = self._transport_to_support or len(support) >= placed_levels
        table_top = self._table["top_z"]
        for label in BLOCK_LABELS:
            if label in support:
                self._carry.pop(label, None)
                continue
            carry = self._carry.get(label)
            arm = frame.grasp_candidates.get(label)
            ready = confirmed.get(label)
            lower = support[-1] if support else None
            destination = self._carry_destinations.get(label)
            if destination is not None:
                ready = destination.arm if ready == destination.arm and not destination.invalid else None
                lower = destination.anchor[-1] if destination.anchor else None
            if carry is not None:
                carry_arm, path = carry
                if (not carry_allowed or frame.bottom[label] <= table_top + 1e-9
                        or frame.hold[carry_arm, label] < _GRASP_HOLD_MIN):
                    self._carry.pop(label)
                    carry = None
            if carry_allowed and (ready is not None or carry is not None):
                target = self._carry_goal(frame, lower, label)
                if carry is None:
                    self._carry[label] = (ready, self._path(pos[label], target))
                elif np.linalg.norm(path.origin_target - target) > min(self._waypoint_radius,
                                                                        self._sides[label] / 4):
                    self._carry[label] = (carry_arm, self._path(pos[label], target))
                else:
                    if destination is not None and not np.array_equal(path.target, target):
                        # Compare both positions on the same updated plane; target motion alone cannot cross it.
                        path.retarget(target)
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
        return new_points * float(self.cfg["waypoint_bonus"])

    def terms(self, proprio, physical: Mapping) -> dict:
        """Evaluate physical terms and read event counters without advancing them."""
        terms, _, frame = self._evaluate(proprio, physical)
        self._stage_terms(terms, frame)
        if self._dense_transport_reward:
            terms.update(dense_approach_return=self._dense_approach_return,
                         dense_transport_return=self._dense_transport_return)
        return terms

    def step(self, proprio, physical: Mapping) -> tuple[dict, float]:
        """Advance ordered routes and dwell-gated grasp and placement events."""
        terms, identities, frame = self._evaluate(proprio, physical)
        self._step_index += 1
        milestone_bonus, confirmed = self._grasp_events(frame)
        approach_bonus = carry_bonus = penalty_step = 0.0
        active_arms: set[int] = set()
        released_arms: set[int] = set()
        if self._dense_transport_reward:
            carry_bonus, placement_bonus, drop_penalty, active_arms, released_arms = (
                self._dense_events(frame, confirmed))
            approach_bonus = self._dense_approach(frame)
            self._dense_approach_return += approach_bonus
            self._dense_transport_return += carry_bonus
            milestone_bonus += placement_bonus
            penalty_step += drop_penalty
            terms.update(dense_approach_return=self._dense_approach_return,
                         dense_transport_return=self._dense_transport_return)
        elif self._transport_to_support:
            self._update_destinations(frame, confirmed)
        self._stage_terms(terms, frame)
        if not self._dense_transport_reward:
            self._transport_events(frame, confirmed)
        waypoint_bonus = 0.0 if self._dense_transport_reward else self._route_events(frame, confirmed)
        for previous, name in zip((None, "base", "pair2"), _MILESTONES[1:]):
            identity = identities[name]
            release = self._released_placements.get(name)
            if release is not None:
                released_chain, trace = release
                if self._stable_prefix(frame, released_chain) == released_chain:
                    identity = released_chain[0] if previous is None else released_chain
            if identity is None or identity != self._dwell_ids[name]:
                self._dwell[name] = 0
            self._dwell_ids[name] = identity
            if identity is not None:
                self._dwell[name] += 1
            if (self._dwell[name] < self.cfg["milestone_dwell_steps"]
                    or (previous is not None and self.milestone_steps[previous] is None)):
                continue
            if self._dense_transport_reward:
                chain = (identity,) if isinstance(identity, str) else identity
                if len(chain) > len(self._available_anchor(frame)):
                    self._settled_anchor = chain
                if release is not None:
                    if chain != released_chain:
                        continue
                    if trace["settled_step"] is None:
                        trace["settled_step"] = self._step_index
            if self.milestone_steps[name] is not None:
                continue
            self.milestone_steps[name] = self._step_index
            scale = (1.0 - _RELEASE_REWARD_FRACTION) if self._dense_transport_reward else 1.0
            milestone_bonus += scale * float(self.cfg[f"milestone_{name}"])
        if self._dense_transport_reward:
            penalty_step += self._dense_penalties(frame, active_arms, released_arms)
        self.milestone_return += milestone_bonus
        self.penalty_return -= penalty_step
        self.waypoint_return += waypoint_bonus
        terms["milestone_return"] = self.milestone_return
        terms["penalty_return"] = self.penalty_return
        terms["penalty_counts"] = dict(self.penalty_counts)
        terms["waypoint_return"] = self.waypoint_return
        terms["waypoint_counts"] = dict(self.waypoint_counts)
        event_bonus = milestone_bonus + waypoint_bonus - penalty_step
        if self._dense_transport_reward:
            terms["event_bonus"] = event_bonus
        return terms, event_bonus + approach_bonus + carry_bonus


__all__ = ["DEFAULTS", "REWARD_KEYS", "StackingReward", "validate_reward_config"]
