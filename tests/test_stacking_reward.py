"""CPU-only tests for the standalone stacking reward geometry.

Run with::

    PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
    /home/cc/miniconda3/envs/act/bin/python -m unittest tests.test_stacking_reward
"""
from __future__ import annotations

import copy
import math
import unittest
from itertools import permutations

import numpy as np

from starter.interfaces import ACTION_DIM, BLOCK_LABELS
from starter.policies.stacking_reward import (
    DEFAULTS,
    REWARD_KEYS,
    StackingReward,
    validate_reward_config,
)

TOP = 0.765
TABLE = {"top_z": TOP, "x_min": -0.7, "x_max": 0.7,
         "y_min": -0.6, "y_max": 0.5}
SIZE = 0.030
HALF = SIZE / 2.0


def _grip_measured(size: float) -> float:
    """Measured opening for a cube of side size (official x5 normalization)."""
    low = DEFAULTS["gripper_scale_low"]
    high = DEFAULTS["gripper_scale_high"]
    joint = (size - DEFAULTS["gripper_aperture_zero"]) / 2.0
    return (joint - low) / (high - low)


REAL_GRIP_30 = _grip_measured(SIZE)
REAL_GRIP_35 = _grip_measured(0.035)
REAL_GRIP_40 = _grip_measured(0.040)
M_GRASP = DEFAULTS["milestone_grasp"]
M_BASE = DEFAULTS["milestone_base"]
M_PAIR2 = DEFAULTS["milestone_pair2"]
M_PAIR3 = DEFAULTS["milestone_pair3"]
M_LEVEL = M_BASE + M_PAIR2 + M_PAIR3
# Measured value at the real joint limit q=0: jaws closed with no cube.
EMPTY_CLOSED = (0.0 - DEFAULTS["gripper_scale_low"]) / (
    DEFAULTS["gripper_scale_high"] - DEFAULTS["gripper_scale_low"])


def _cfg(**overrides) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(overrides)
    return cfg


def _extents(sizes=None) -> dict:
    sizes = sizes or {label: [SIZE, SIZE, SIZE] for label in BLOCK_LABELS}
    return {label: list(value) for label, value in sizes.items()}


def _proprio(left=(0.6, 0.6, 0.9), right=(-0.6, 0.6, 0.9),
             left_quat=(1.0, 0.0, 0.0, 0.0),
             right_quat=(1.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    value = np.zeros(ACTION_DIM, dtype=np.float64)
    value[0:3] = left
    value[3:7] = left_quat
    value[8:11] = right
    value[11:15] = right_quat
    return value


def _state(positions=None, rotations=None, linear=None, angular=None,
           measured=(1.0, 1.0), extents=None) -> dict:
    extents = extents or _extents()
    if positions is None:
        positions = {
            label: [-0.3 + 0.3 * index, 0.0, TOP + extents[label][2] / 2.0]
            for index, label in enumerate(BLOCK_LABELS)
        }
    if rotations is None:
        rotations = {label: [1.0, 0.0, 0.0, 0.0] for label in BLOCK_LABELS}
    if linear is None:
        linear = {label: [0.0, 0.0, 0.0] for label in BLOCK_LABELS}
    if angular is None:
        angular = {label: [0.0, 0.0, 0.0] for label in BLOCK_LABELS}
    return {
        "blocks": {
            label: {"pos": [float(value) for value in positions[label]],
                    "rot": [float(value) for value in rotations[label]]}
            for label in BLOCK_LABELS
        },
        "block_extents": {label: [float(v) for v in extents[label]]
                          for label in BLOCK_LABELS},
        "table": dict(TABLE),
        "block_velocities": {
            label: {"linear": [float(v) for v in linear[label]],
                    "angular": [float(v) for v in angular[label]]}
            for label in BLOCK_LABELS
        },
        "gripper_open_measured": [float(value) for value in measured],
    }


def _reset(extents=None) -> dict:
    extents = extents or _extents()
    positions = {
        label: [-0.35 + 0.3 * index, -0.4, TOP + extents[label][2] / 2.0]
        for index, label in enumerate(BLOCK_LABELS)
    }
    return _state(positions=positions, extents=extents)


def _reward(extents=None, **overrides) -> StackingReward:
    return StackingReward(_reset(extents), _proprio(), _cfg(**overrides))


def _tower_positions(size: float = SIZE, xy=(0.0, -0.2)) -> dict:
    """block_0 at the target, block_1 above it, block_2 on top."""
    x, y = xy
    return {
        "block_0": [x, y, TOP + size / 2.0],
        "block_1": [x, y, TOP + 1.5 * size],
        "block_2": [x, y, TOP + 2.5 * size],
    }


class ConfigValidationTests(unittest.TestCase):
    def test_defaults_are_complete_and_numeric(self):
        self.assertEqual(REWARD_KEYS, tuple(DEFAULTS))
        validate_reward_config(_cfg())
        for value in DEFAULTS.values():
            self.assertIsInstance(value, (int, float))
            self.assertFalse(isinstance(value, bool))

    def test_default_milestone_sum_below_partial_score(self):
        total = len(BLOCK_LABELS) * M_GRASP + M_LEVEL
        self.assertAlmostEqual(total, 11.6)
        waypoints = 2 * len(BLOCK_LABELS) * DEFAULTS["waypoint_count"] * DEFAULTS["waypoint_bonus"]
        self.assertAlmostEqual(waypoints, 2.7)
        self.assertLess(total + waypoints, 15.0)

    def test_missing_key_rejected(self):
        cfg = _cfg()
        del cfg["reach_weight"]
        with self.assertRaises(ValueError):
            validate_reward_config(cfg)

    def test_nonfinite_and_wrong_types_rejected(self):
        for key, value in (("reach_weight", float("nan")),
                           ("reach_weight", "1.0"),
                           ("manipulation_weight", True),
                           ("placement_x", float("inf"))):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    validate_reward_config(_cfg(**{key: value}))

    def test_negative_weights_and_positive_scales(self):
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(pair_weight=-0.1))
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(reach_scale=0.0))
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(grasp_centering_fraction=0.0))

    def test_gripper_calibration_bounds(self):
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(gripper_scale_low=0.05))
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(gripper_scale_high=0.0))
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(gripper_scale_high=-0.01))
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(gripper_aperture_zero=-0.001))
        for value in (0.0, 0.5, -0.1):
            with self.subTest(key="grasp_aperture_fraction", value=value):
                with self.assertRaises(ValueError):
                    validate_reward_config(_cfg(grasp_aperture_fraction=value))
        for value in (0.0, 1.0, -0.1):
            with self.subTest(key="grasp_centering_fraction", value=value):
                with self.assertRaises(ValueError):
                    validate_reward_config(_cfg(grasp_centering_fraction=value))

    def test_target_radius_must_be_below_soft_scale(self):
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(placement_radius=0.15,
                                        placement_soft_scale=0.15))

    def test_upright_threshold_bounds(self):
        for value in (0.0, 1.0, 1.2):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_reward_config(_cfg(upright_cos_min=value))

    def test_dwell_must_be_positive_int_not_bool(self):
        for value in (0, 1.5, True, False):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_reward_config(_cfg(milestone_dwell_steps=value))

    def test_waypoint_count_must_be_positive_int_not_bool(self):
        for value in (0, 1.5, True, False):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_reward_config(_cfg(waypoint_count=value))

    def test_milestone_sum_must_stay_below_15(self):
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(milestone_pair3=15.0))
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(milestone_pair3=4.0, waypoint_bonus=0.5))

    def test_support_gap_floor_nonnegative(self):
        with self.assertRaises(ValueError):
            validate_reward_config(_cfg(support_gap_floor=-0.001))


class ResetValidationTests(unittest.TestCase):
    def test_initial_terms_and_public_state(self):
        reward = _reward()
        self.assertIsInstance(reward.initial_terms, dict)
        self.assertLessEqual(reward.initial_terms["phi"], 0.0)
        self.assertEqual(reward.milestone_steps,
                         {"grasp": None, "base": None, "pair2": None, "pair3": None})
        self.assertEqual(reward.milestone_return, 0.0)

    def test_missing_geometry_rejected(self):
        reset = _reset()
        for key in ("table", "block_extents", "block_velocities",
                    "gripper_open_measured"):
            broken = copy.deepcopy(reset)
            del broken[key]
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    StackingReward(broken, _proprio(), _cfg())

    def test_nonfinite_and_bad_shapes_rejected(self):
        reset = _reset()
        bad = copy.deepcopy(reset)
        bad["blocks"]["block_0"]["pos"][0] = float("nan")
        with self.assertRaises(ValueError):
            StackingReward(bad, _proprio(), _cfg())
        bad = copy.deepcopy(reset)
        bad["gripper_open_measured"] = [1.0]
        with self.assertRaises(ValueError):
            StackingReward(bad, _proprio(), _cfg())
        bad = copy.deepcopy(reset)
        bad["block_extents"]["block_1"] = [0.03, 0.03, float("inf")]
        with self.assertRaises(ValueError):
            StackingReward(bad, _proprio(), _cfg())
        with self.assertRaises(ValueError):
            StackingReward(reset, np.zeros(ACTION_DIM - 1), _cfg())
        with self.assertRaises(ValueError):
            StackingReward(reset, _proprio(), _cfg(placement_x=1.0))

    def test_non_cube_extents_rejected(self):
        extents = _extents()
        extents["block_1"] = [0.030, 0.030, 0.036]
        with self.assertRaises(ValueError):
            StackingReward(_reset(extents), _proprio(), _cfg())

    def test_target_outside_table_rejected(self):
        for placement in ({"placement_x": 0.69}, {"placement_y": -0.59},
                          {"placement_x": -0.69}):
            with self.subTest(placement=placement):
                with self.assertRaises(ValueError):
                    _reward(**placement)

    def test_degenerate_quaternion_rejected_by_terms(self):
        reward = _reward()
        state = _reset()
        state["blocks"]["block_0"]["rot"] = [0.0, 0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            reward.terms(_proprio(), state)


class FullTowerTests(unittest.TestCase):
    def test_positive_full_sequential_trace(self):
        reward = _reward()
        state = _state(positions=_tower_positions())
        terms = reward.terms(_proprio(), state)
        self.assertAlmostEqual(terms["phi"], 0.0, places=8)
        self.assertAlmostEqual(terms["phi_reach"], 0.0, places=8)
        self.assertAlmostEqual(terms["phi_manipulation"], 0.0, places=8)
        self.assertAlmostEqual(terms["phi_base"], 0.0, places=8)
        self.assertAlmostEqual(terms["phi_pair"], 0.0, places=8)
        self.assertAlmostEqual(terms["phi_release"], 0.0, places=8)
        self.assertAlmostEqual(terms["pair_sum"], 2.0, places=8)
        self.assertAlmostEqual(terms["settled_pair"], 1.0, places=8)
        self.assertTrue(terms["base_ok"])
        self.assertTrue(terms["pair2_ok"])
        self.assertTrue(terms["pair3_ok"])
        self.assertEqual(terms["stable_tower_levels"], 3)
        self.assertAlmostEqual(terms["base_score"], 1.0, places=8)

    def test_holding_full_tower_pays_once_in_stage_order(self):
        reward = _reward()
        state = _state(positions=_tower_positions())
        bonuses = [reward.step(_proprio(), state)[1] for _ in range(5)]
        self.assertEqual(bonuses[:2], [0.0, 0.0])
        self.assertAlmostEqual(bonuses[2], M_LEVEL)  # no grasp in this fixture
        self.assertEqual(bonuses[3:], [0.0, 0.0])
        self.assertEqual(reward.milestone_steps,
                         {"grasp": None, "base": 3, "pair2": 3, "pair3": 3})
        self.assertAlmostEqual(reward.milestone_return, M_LEVEL)

    def test_sequential_stages_and_no_rebuild_payment(self):
        reward = _reward()
        base = _state(positions={
            "block_0": [0.0, -0.2, TOP + HALF],
            "block_1": [0.35, 0.2, TOP + HALF],
            "block_2": [0.45, 0.2, TOP + HALF]})
        pair2 = _state(positions={
            "block_0": [0.0, -0.2, TOP + HALF],
            "block_1": [0.0, -0.2, TOP + 3 * HALF],
            "block_2": [0.45, 0.2, TOP + HALF]})
        pair3 = _state(positions=_tower_positions())
        neutral = _reset()
        for _ in range(3):
            reward.step(_proprio(), base)
        self.assertAlmostEqual(reward.milestone_return, M_BASE)
        for _ in range(3):
            reward.step(_proprio(), pair2)
        self.assertAlmostEqual(reward.milestone_return, M_BASE + M_PAIR2)
        for _ in range(3):
            reward.step(_proprio(), pair3)
        self.assertAlmostEqual(reward.milestone_return, M_LEVEL)
        # Destroy and rebuild: nothing is paid twice.
        for _ in range(3):
            reward.step(_proprio(), neutral)
        for _ in range(3):
            reward.step(_proprio(), pair3)
        self.assertAlmostEqual(reward.milestone_return, M_LEVEL)

    def test_grasp_milestone_completes_per_episode_max(self):
        reward = _reward()
        held = {
            "block_0": [0.35, 0.2, TOP + 2 * HALF],
            "block_1": [-0.4, 0.4, TOP + HALF],
            "block_2": [0.45, 0.4, TOP + HALF],
        }
        # Left gripper closed with its grasp point at block_0's centre.
        left = np.array(held["block_0"]) - np.array([0.143, 0.0, 0.0])
        proprio = _proprio(left=left, right=(-0.6, 0.6, 0.9))
        held_state = _state(positions=held, measured=(REAL_GRIP_30, 1.0))
        self.assertTrue(reward.terms(proprio, held_state)["grasp_ok"])
        for _ in range(3):
            reward.step(proprio, held_state)
        self.assertAlmostEqual(reward.milestone_return, M_GRASP)
        tower = _state(positions=_tower_positions())
        for _ in range(3):
            reward.step(_proprio(), tower)
        self.assertAlmostEqual(reward.milestone_return, M_GRASP + M_LEVEL)
        self.assertLess(reward.milestone_return, 15.0)

    def test_per_episode_state_is_independent(self):
        first = _reward()
        second = _reward()
        tower = _state(positions=_tower_positions())
        for _ in range(3):
            first.step(_proprio(), tower)
        self.assertAlmostEqual(first.milestone_return, M_LEVEL)
        self.assertEqual(second.milestone_return, 0.0)
        for _ in range(2):
            second.step(_proprio(), tower)
        self.assertAlmostEqual(second.milestone_return, 0.0)
        second.step(_proprio(), tower)
        self.assertAlmostEqual(second.milestone_return, M_LEVEL)

    def test_terms_are_pure(self):
        reward = _reward()
        tower = _state(positions=_tower_positions())
        for _ in range(5):
            reward.terms(_proprio(), tower)
        self.assertEqual(reward.milestone_return, 0.0)
        self.assertEqual(reward.milestone_steps["base"], None)
        self.assertEqual(reward._dwell["base"], 0)


class SizeAndRotationTests(unittest.TestCase):
    def test_all_curriculum_sizes_stack(self):
        for size in (0.030, 0.035, 0.040):
            with self.subTest(size=size):
                extents = _extents({label: [size, size, size]
                                    for label in BLOCK_LABELS})
                reward = _reward(extents)
                state = _state(positions=_tower_positions(size),
                               extents=extents)
                terms = reward.terms(_proprio(), state)
                self.assertTrue(terms["pair2_ok"])
                self.assertTrue(terms["pair3_ok"])
                self.assertAlmostEqual(terms["pair_sum"], 2.0, places=8)
                gaps = sorted(entry["gap"] for entry in terms["support_gaps"])
                self.assertAlmostEqual(gaps[0], 0.0, places=8)
                self.assertAlmostEqual(gaps[1], 0.0, places=8)
                # The non-adjacent bottom/top pair has a full block between.
                self.assertAlmostEqual(gaps[2], size, places=8)

    def test_unequal_sizes_use_pairwise_half_heights(self):
        extents = _extents({
            "block_0": [0.030, 0.030, 0.030],
            "block_1": [0.040, 0.040, 0.040],
            "block_2": [0.040, 0.040, 0.040],
        })
        reward = _reward(extents)
        base_z = TOP + 0.015
        upper_z = TOP + 0.015 + (0.030 + 0.040) / 2.0
        positions = {
            "block_0": [0.0, -0.2, base_z],
            "block_1": [0.0, -0.2, upper_z],
            "block_2": [0.45, 0.2, TOP + 0.02],
        }
        terms = reward.terms(_proprio(),
                             _state(positions=positions, extents=extents))
        self.assertTrue(terms["pair2_ok"])
        for offset, expected in ((0.030, False), (0.040, False), (0.035, True)):
            with self.subTest(center_offset=offset):
                moved = dict(positions)
                moved["block_1"] = [0.0, -0.2, base_z + offset]
                result = reward.terms(
                    _proprio(), _state(positions=moved, extents=extents))
                self.assertEqual(result["pair2_ok"], expected)

    def test_any_cube_face_is_upright(self):
        quarter = math.sqrt(0.5)
        rotations = {label: [quarter, quarter, 0.0, 0.0]
                     for label in BLOCK_LABELS}
        reward = _reward()
        state = _state(positions=_tower_positions(), rotations=rotations)
        terms = reward.terms(_proprio(), state)
        self.assertTrue(terms["pair3_ok"])
        self.assertAlmostEqual(terms["phi"], 0.0, places=8)

    def test_tilted_cube_is_not_supported(self):
        angle = math.radians(20.0)
        rotation = [math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0]
        rotations = {"block_0": [1.0, 0.0, 0.0, 0.0],
                     "block_1": rotation,
                     "block_2": [1.0, 0.0, 0.0, 0.0]}
        reward = _reward()
        terms = reward.terms(
            _proprio(), _state(positions=_tower_positions(), rotations=rotations))
        self.assertFalse(terms["pair2_ok"])

    def test_labels_and_permutations_are_symmetric(self):
        positions = _tower_positions()
        reward = _reward()
        baseline = reward.terms(_proprio(), _state(positions=positions))
        for order in permutations(BLOCK_LABELS):
            with self.subTest(order=order):
                permuted = {label: positions[old]
                            for label, old in zip(BLOCK_LABELS, order)}
                terms = reward.terms(_proprio(),
                                     _state(positions=permuted))
                self.assertAlmostEqual(terms["pair_sum"],
                                       baseline["pair_sum"], places=8)
                self.assertTrue(terms["pair3_ok"])
                self.assertAlmostEqual(terms["phi"],
                                       baseline["phi"], places=8)


class SupportRejectionTests(unittest.TestCase):
    def _base_positions(self):
        return {
            "block_0": [0.0, -0.2, TOP + HALF],
            "block_1": [0.0, -0.2, TOP + 3 * HALF],
            "block_2": [0.45, 0.2, TOP + HALF],
        }

    def _terms(self, positions, **kwargs):
        reward = _reward()
        return reward.terms(_proprio(), _state(positions=positions, **kwargs))

    def test_pair_potential_accepts_table_support_away_from_target(self):
        at_target = self._terms(self._base_positions())
        off_target = self._base_positions()
        for label in ("block_0", "block_1"):
            off_target[label][0] = 0.15
        supported = self._terms(off_target)
        self.assertAlmostEqual(supported["phi_pair"], at_target["phi_pair"], places=8)
        self.assertLess(supported["phi_base"], at_target["phi_base"])
        self.assertFalse(supported["base_ok"])
        self.assertFalse(supported["pair2_ok"])

        for label in ("block_0", "block_1"):
            off_target[label][2] += 0.05
        floating = self._terms(off_target)
        self.assertAlmostEqual(floating["pair_sum"], 1.0, places=8)
        self.assertAlmostEqual(floating["phi_pair"], -2.0 * DEFAULTS["pair_weight"], places=8)

    def test_airborne_pair_rejected(self):
        positions = self._base_positions()
        positions["block_1"][2] += 0.010
        terms = self._terms(positions)
        self.assertFalse(terms["pair2_ok"])
        self.assertAlmostEqual(terms["phi_release"], 0.0, places=8)

    def test_side_by_side_rejected(self):
        positions = self._base_positions()
        positions["block_1"] = [0.03, -0.2, TOP + HALF]
        terms = self._terms(positions)
        self.assertFalse(terms["pair2_ok"])
        self.assertAlmostEqual(terms["pair_sum"], 0.0, places=8)

    def test_overlap_rejected(self):
        positions = self._base_positions()
        positions["block_1"][2] -= 0.010
        terms = self._terms(positions)
        self.assertFalse(terms["pair2_ok"])

    def test_overhang_rejected(self):
        positions = self._base_positions()
        positions["block_1"][0] = HALF + 0.002
        terms = self._terms(positions)
        self.assertFalse(terms["pair2_ok"])

    def test_tabletop_coplanar_rejected(self):
        positions = self._base_positions()
        positions["block_1"] = [0.0, -0.2, TOP + HALF]
        terms = self._terms(positions)
        self.assertFalse(terms["pair2_ok"])
        self.assertAlmostEqual(terms["pair_sum"], 0.0, places=8)

    def test_moving_and_spinning_blocks_rejected(self):
        positions = self._base_positions()
        fast_linear = {label: [0.0, 0.0, 0.0] for label in BLOCK_LABELS}
        fast_linear["block_1"] = [0.02, 0.0, 0.0]
        self.assertFalse(self._terms(positions, linear=fast_linear)["pair2_ok"])
        fast_angular = {label: [0.0, 0.0, 0.0] for label in BLOCK_LABELS}
        fast_angular["block_1"] = [0.0, 0.0, 0.2]
        self.assertFalse(self._terms(positions, angular=fast_angular)["pair2_ok"])

    def test_control_motion_drives_settling_with_raw_velocities_preserved(self):
        positions = self._base_positions()
        angular = {label: [0.0, 0.0, 0.213] for label in BLOCK_LABELS}
        state = _state(positions, angular=angular)
        state["settle_speeds"] = {label: [0.0, 0.0] for label in BLOCK_LABELS}
        action = _proprio()
        reward = StackingReward(state, action, _cfg())
        first = reward.terms(action, state)
        after, bonus = reward.step(action, state)
        self.assertTrue(first["pair2_ok"])
        self.assertEqual(first["phi"], after["phi"])
        self.assertEqual(bonus, 0.0)
        self.assertEqual(state["block_velocities"]["block_1"]["angular"], angular["block_1"])
        for speed in ([0.011, 0.0], [0.0, 0.101]):
            state["settle_speeds"]["block_1"] = speed
            self.assertFalse(reward.terms(action, state)["pair2_ok"])

    def test_hard_speed_threshold_is_not_a_soft_product(self):
        positions = self._base_positions()
        at_limit = {label: [0.0, 0.0, 0.0] for label in BLOCK_LABELS}
        at_limit["block_1"] = [0.010, 0.0, 0.0]
        self.assertTrue(self._terms(positions, linear=at_limit)["pair2_ok"])
        above = {label: [0.0, 0.0, 0.0] for label in BLOCK_LABELS}
        above["block_1"] = [0.0100001, 0.0, 0.0]
        self.assertFalse(self._terms(positions, linear=above)["pair2_ok"])


class HardAnchorBoundaryTests(unittest.TestCase):
    """Conservative internal geometric resolution for hard support anchors.

    The broad smooth kernels keep guiding outside these boundaries; only the
    hard predicates require the narrow gap and full central-COM margin.
    """

    def _pair_terms(self, size=0.03, gap=0.0, lateral=0.0):
        extents = _extents({label: [size, size, size] for label in BLOCK_LABELS})
        reward = _reward(extents)
        positions = {
            "block_0": [0.0, -0.2, TOP + size / 2.0],
            "block_1": [lateral, -0.2,
                        TOP + size + gap + size / 2.0],
            "block_2": [0.45, 0.2, TOP + size / 2.0],
        }
        return reward.terms(
            _proprio(), _state(positions=positions, extents=extents))

    def test_hard_gap_accepts_exactly_ten_percent_tolerance(self):
        expected = {0.030: 0.00024, 0.035: 0.00028, 0.040: 0.00032}
        for size, millimetres in expected.items():
            tolerance = max(DEFAULTS["support_gap_floor"],
                            DEFAULTS["support_gap_fraction"] * size)
            hard = 0.10 * tolerance
            with self.subTest(size=size):
                self.assertAlmostEqual(hard, millimetres, places=9)
                self.assertTrue(self._pair_terms(size, gap=hard)["pair2_ok"])
                self.assertTrue(self._pair_terms(size, gap=-hard)["pair2_ok"])
                self.assertFalse(
                    self._pair_terms(size, gap=hard + 1e-5)["pair2_ok"])

    def test_reported_two_point_three_millimetre_gap_rejected(self):
        for gap in (0.0023, -0.0023):
            with self.subTest(gap=gap):
                self.assertFalse(
                    self._pair_terms(0.035, gap=gap)["pair2_ok"])

    def test_central_com_margin_boundary(self):
        for size in (0.030, 0.035, 0.040):
            full = 0.25 * size
            with self.subTest(size=size):
                self.assertTrue(
                    self._pair_terms(size, lateral=full)["pair2_ok"])
                self.assertFalse(
                    self._pair_terms(size, lateral=full + 1e-4)["pair2_ok"])
                # Exact overhang edge: no central COM above the lower block.
                self.assertFalse(
                    self._pair_terms(size, lateral=0.5 * size)["pair2_ok"])

    def test_fork_bridged_over_two_blocks_rejected(self):
        # A block centred between two neighbours is not centrally supported
        # by either one, so no single directed edge may count as hard.
        extents = _extents()
        reward = _reward()
        positions = {
            "block_0": [-0.009, -0.2, TOP + HALF],
            "block_1": [0.009, -0.2, TOP + HALF],
            "block_2": [0.0, -0.2, TOP + SIZE + HALF],
        }
        terms = reward.terms(
            _proprio(), _state(positions=positions, extents=extents))
        self.assertFalse(terms["pair2_ok"])
        self.assertEqual(terms["stable_tower_levels"], 1)


class ReleaseTests(unittest.TestCase):
    def _base_state(self, measured):
        return _state(positions={
            "block_0": [0.0, -0.2, TOP + HALF],
            "block_1": [0.35, 0.2, TOP + HALF],
            "block_2": [0.45, 0.2, TOP + HALF],
        }, measured=measured)

    def test_remote_empty_closed_gripper_keeps_local_release(self):
        reward = _reward()
        terms = reward.terms(_proprio(), self._base_state((EMPTY_CLOSED, 1.0)))
        self.assertTrue(terms["base_ok"])
        self.assertAlmostEqual(terms["phi_release"], 0.0, places=8)
        self.assertAlmostEqual(terms["min_gripper"], EMPTY_CLOSED / 0.8, places=8)

    def test_empty_closed_gripper_near_base_blocks_release(self):
        reward = _reward()
        left = np.array([0.0, -0.2, TOP + HALF]) - np.array([0.143, 0.0, 0.0])
        proprio = _proprio(left=left)
        terms = reward.terms(proprio, self._base_state((EMPTY_CLOSED, 1.0)))
        self.assertFalse(terms["base_ok"])
        release = EMPTY_CLOSED / 0.8
        self.assertAlmostEqual(terms["phi_release"],
                               -DEFAULTS["release_weight"] * (1.0 - release),
                               places=8)

    def test_local_hard_release_requires_measured_eight(self):
        # An arm associated with the base with measured 0.64/0.65/0.79 is not
        # released; 0.80 is.  The threshold is the official measured value,
        # never the closeness product (which made 0.64 sufficient).
        left = np.array([0.0, -0.2, TOP + HALF]) - np.array([0.143, 0.0, 0.0])
        proprio = _proprio(left=left)
        for measured in (0.64, 0.65, 0.79):
            with self.subTest(measured=measured):
                terms = _reward().terms(
                    proprio, self._base_state((measured, 1.0)))
                self.assertFalse(terms["base_ok"])
                self.assertAlmostEqual(terms["phi_release"],
                                       -DEFAULTS["release_weight"]
                                       * (1.0 - measured / 0.8), places=8)
        terms = _reward().terms(proprio, self._base_state((0.8, 1.0)))
        self.assertTrue(terms["base_ok"])
        self.assertAlmostEqual(terms["phi_release"], 0.0, places=8)

    def test_remote_arm_never_affects_a_released_block(self):
        # Left arm is closed (empty) but far from the base: no association, so
        # the base is released and pays no penalty.
        reward = _reward()
        terms = reward.terms(
            _proprio(left=(0.6, 0.6, 0.9), right=(-0.6, 0.6, 0.9)),
            self._base_state((EMPTY_CLOSED, EMPTY_CLOSED)))
        self.assertTrue(terms["base_ok"])
        self.assertAlmostEqual(terms["phi_release"], 0.0, places=8)

    def test_upper_grasp_does_not_revoke_placed_base_release(self):
        # Holding the upper cube above the base associates only the upper arm
        # (its grasp point coincides with the upper centre; the base is one
        # side away, outside the half-side centering radius).
        reward = _reward()
        upper = np.array([0.0, -0.2, TOP + 3 * HALF])
        left = upper - np.array([0.143, 0.0, 0.0])
        state = _state(positions={
            "block_0": [0.0, -0.2, TOP + HALF],
            "block_1": [0.0, -0.2, TOP + 3 * HALF],
            "block_2": [0.45, 0.2, TOP + HALF],
        }, measured=(REAL_GRIP_30, 1.0))
        terms = reward.terms(_proprio(left=left), state)
        self.assertTrue(terms["base_ok"])
        self.assertAlmostEqual(terms["base_score"], 1.0, places=8)
        # The upper is still held, so the second edge is not released yet.
        self.assertFalse(terms["pair2_ok"])
        # Releasing both arms completes the second edge.
        released = reward.terms(_proprio(), state)
        self.assertTrue(released["pair2_ok"])

    def test_partial_openness_makes_release_intermediate(self):
        reward = _reward()
        left = np.array([0.0, -0.2, TOP + HALF]) - np.array([0.143, 0.0, 0.0])
        proprio = _proprio(left=left)
        terms = reward.terms(proprio, self._base_state((0.4, 1.0)))
        self.assertAlmostEqual(terms["min_gripper"], 0.5, places=8)
        self.assertFalse(terms["base_ok"])

    def test_fully_open_at_official_threshold(self):
        reward = _reward()
        left = np.array([0.0, -0.2, TOP + HALF]) - np.array([0.143, 0.0, 0.0])
        proprio = _proprio(left=left)
        terms = reward.terms(proprio, self._base_state((0.8, 1.0)))
        self.assertTrue(terms["base_ok"])
        self.assertAlmostEqual(terms["min_gripper"], 1.0, places=8)


class DwellTests(unittest.TestCase):
    def _state_at(self, label: str):
        positions = {
            "block_0": [0.4, 0.2, TOP + HALF],
            "block_1": [-0.4, 0.2, TOP + HALF],
            "block_2": [0.45, 0.2, TOP + HALF],
        }
        positions[label] = [0.0, -0.2, TOP + HALF]
        return _state(positions=positions)

    def test_interruption_resets_dwell(self):
        reward = _reward()
        first = self._state_at("block_0")
        neutral = _reset()
        reward.step(_proprio(), first)
        reward.step(_proprio(), first)
        reward.step(_proprio(), neutral)
        self.assertEqual(reward.milestone_return, 0.0)
        reward.step(_proprio(), first)
        reward.step(_proprio(), first)
        self.assertEqual(reward.milestone_return, 0.0)
        _, bonus = reward.step(_proprio(), first)
        self.assertAlmostEqual(bonus, M_BASE)

    def test_identity_swap_resets_dwell(self):
        reward = _reward()
        first = self._state_at("block_0")
        second = self._state_at("block_1")
        reward.step(_proprio(), first)
        reward.step(_proprio(), first)
        reward.step(_proprio(), second)  # identity changed -> count 1
        reward.step(_proprio(), first)   # identity changed -> count 1
        reward.step(_proprio(), first)
        self.assertEqual(reward.milestone_return, 0.0)
        _, bonus = reward.step(_proprio(), first)
        self.assertAlmostEqual(bonus, M_BASE)

    def test_milestone_steps_exposes_event_time(self):
        reward = _reward()
        tower = _state(positions=_tower_positions())
        for _ in range(3):
            reward.step(_proprio(), tower)
        self.assertEqual(reward.milestone_steps["base"], 3)
        self.assertEqual(reward.milestone_steps["pair2"], 3)
        self.assertEqual(reward.milestone_steps["pair3"], 3)


class DenseSignalTests(unittest.TestCase):
    @staticmethod
    def _held_at(reward, positions, label, aperture, offset=0.0):
        wrist = np.asarray(positions[label]) + [offset - DEFAULTS["grasp_offset"], 0, 0]
        state = _state(positions=positions, measured=(aperture, 1.0))
        return reward.terms(_proprio(left=wrist), state)

    def test_arrival_rewards_matching_width_then_clearing_support(self):
        reward = _reward()
        positions = {label: data["pos"] for label, data in _reset()["blocks"].items()}
        label = "block_0"

        def sample(aperture, offset=0.0):
            return self._held_at(reward, positions, label, aperture, offset)

        open_ = sample(1.0)["phi_manipulation"]
        self.assertEqual(sample(EMPTY_CLOSED)["phi_manipulation"], open_)
        self.assertEqual(sample(REAL_GRIP_30, SIZE)["phi_manipulation"], open_)
        partly_closed = sample(0.75)["phi_manipulation"]
        matched = sample(REAL_GRIP_30)
        self.assertGreater(partly_closed, open_)
        self.assertGreater(matched["phi_manipulation"], partly_closed)
        self.assertFalse(matched["grasp_ok"])
        positions[label][2] += SIZE / 8
        lifted = sample(REAL_GRIP_30)
        self.assertGreater(lifted["phi_manipulation"], matched["phi_manipulation"])
        self.assertFalse(lifted["grasp_ok"])
        self.assertEqual(sample(1.0)["phi_manipulation"], open_)

    def test_closure_weight_strengthens_local_signal_without_changing_grasp_gate(self):
        original = _reward(grasp_closure_weight=0.25)
        stronger = _reward(grasp_closure_weight=0.5)
        positions = {label: data["pos"] for label, data in _reset()["blocks"].items()}
        label = "block_0"
        for offset in (0.0, SIZE / 4):
            old = self._held_at(original, positions, label, REAL_GRIP_30, offset)
            new = self._held_at(stronger, positions, label, REAL_GRIP_30, offset)
            self.assertGreater(new["phi_manipulation"], old["phi_manipulation"])
            self.assertEqual(new["grasp_ok"], old["grasp_ok"])
        positions[label][2] += SIZE / 2
        for reward in (original, stronger):
            self.assertTrue(self._held_at(reward, positions, label, REAL_GRIP_30)["grasp_ok"])
            self.assertFalse(self._held_at(reward, positions, label, EMPTY_CLOSED)["grasp_ok"])
        far_old = self._held_at(original, positions, label, REAL_GRIP_30, SIZE)
        far_new = self._held_at(stronger, positions, label, REAL_GRIP_30, SIZE)
        self.assertEqual(far_new["phi_manipulation"], far_old["phi_manipulation"])

    def test_transport_descend_and_release_at_each_support(self):
        reward = _reward()
        positions = {label: data["pos"] for label, data in _reset()["blocks"].items()}
        for level, label in enumerate(BLOCK_LABELS):
            with self.subTest(level=level):
                top = TOP + level * SIZE
                x, y = (0.0, -0.2) if level == 0 else positions[BLOCK_LABELS[level - 1]][:2]
                # The first block is off-centre but inside the table target.
                if level == 0:
                    x += DEFAULTS["placement_radius"] / 2
                poses = [(x + 0.14, y, top + HALF + 0.06),
                         (x + 0.08, y, top + HALF + 0.06),
                         (x, y, top + HALF + 0.06),
                         (x, y, top + HALF + SIZE / 4),
                         (x, y, top + HALF + 0.0024),
                         (x, y, top + HALF + 0.002),
                         (x, y, top + HALF + 0.001),
                         (x, y, top + HALF)]
                phi = []
                for point in poses:
                    positions[label] = point
                    held = self._held_at(reward, positions, label, REAL_GRIP_30)
                    phi.append(held["phi"])
                    self.assertEqual(held["stable_tower_levels"], level)
                    if point == poses[1]:
                        opened = self._held_at(reward, positions, label, 1.0)
                        self.assertLess(opened["phi_manipulation"], held["phi_manipulation"])
                self.assertTrue(all(a < b for a, b in zip(phi, phi[1:])), phi)
                released = self._held_at(reward, positions, label, 1.0)
                self.assertEqual(released["stable_tower_levels"], level + 1)
                self.assertGreaterEqual(released["phi_manipulation"], held["phi_manipulation"])
                self.assertGreater(released["phi"], phi[-1])

    def test_hover_without_holding_does_not_earn_transport(self):
        reward = _reward()
        positions = _tower_positions()
        positions["block_2"][2] += 0.02
        airborne = self._held_at(reward, positions, "block_2", 1.0)
        self.assertEqual(airborne["manipulation_progress"], 2.0)
        positions["block_0"][0] += 0.15
        no_anchor = self._held_at(reward, positions, "block_2", 1.0)
        self.assertEqual(no_anchor["manipulation_progress"], 0.0)

    def test_stable_pair_and_next_grasp_improve(self):
        reward = _reward()
        base = {
            "block_0": [0.0, -0.2, TOP + HALF],
            "block_1": [0.0, -0.2, TOP + 3 * HALF],
            "block_2": [0.35, 0.2, TOP + HALF],
        }
        far = reward.terms(
            _proprio(left=(0.7, 0.6, 0.9), right=(-0.7, 0.6, 0.9)),
            _state(positions=base))
        # Grasp point at block_2's centre.
        left = np.array(base["block_2"]) - np.array([0.143, 0.0, 0.0])
        near = reward.terms(_proprio(left=left),
                            _state(positions=base))
        self.assertGreater(near["phi_reach"], far["phi_reach"])
        # A grasp near the already placed base is softly rejected.
        base_left = np.array(base["block_0"]) - np.array([0.143, 0.0, 0.0])
        near_base = reward.terms(_proprio(left=base_left),
                                 _state(positions=base))
        self.assertGreater(near["phi_reach"], near_base["phi_reach"])

    def test_all_terms_bounded_nonpositive(self):
        reward = _reward()
        cases = [
            _reset(),
            _state(positions=_tower_positions()),
            _state(positions={
                "block_0": [0.2, -0.2, TOP + 2 * HALF],
                "block_1": [0.1, 0.2, TOP + 2 * HALF],
                "block_2": [0.4, 0.1, TOP + 4 * HALF]}),
        ]
        for state in cases:
            terms = reward.terms(_proprio(), state)
            self.assertLessEqual(terms["phi"], 1e-9)
            for key in ("phi_reach", "phi_manipulation", "phi_base",
                        "phi_pair", "phi_release"):
                self.assertLessEqual(terms[key], 1e-9)
                self.assertTrue(math.isfinite(terms[key]))
            self.assertGreaterEqual(terms["phi_release"], -0.5 - 1e-9)

    def test_returns_required_diagnostic_fields(self):
        reward = _reward()
        terms = reward.terms(_proprio(), _reset())
        required = {
            "phi", "phi_reach", "phi_manipulation", "phi_pair", "phi_release",
            "manipulation_progress",
            "pair_sum", "d_min", "lift", "lift_gate",
            "min_gripper", "settled_pair", "lift_progress", "phi_base",
            "base_score", "base_ok", "pair2_ok", "pair3_ok", "grasp_ok",
            "stable_tower_levels", "target_xy", "per_block_lift",
            "support_gaps", "milestone_return",
        }
        self.assertTrue(required.issubset(terms))
        self.assertEqual(terms["target_xy"], [0.0, -0.2])
        self.assertEqual(set(terms["per_block_lift"]), set(BLOCK_LABELS))
        self.assertEqual(len(terms["support_gaps"]), 3)
        self.assertIn(terms["stable_tower_levels"], (0, 1, 2, 3))


class GraspGeometryTests(unittest.TestCase):
    """Holding uses measured x5 openings and the jaw-axis cube projection."""

    def _pickup_terms(self, size, measured, rotation=(1.0, 0.0, 0.0, 0.0)):
        extents = _extents({label: [size, size, size] for label in BLOCK_LABELS})
        position = [0.2, -0.2, TOP + 1.5 * size]
        left = np.array(position) - np.array([0.143, 0.0, 0.0])
        rotations = {label: [1.0, 0.0, 0.0, 0.0] for label in BLOCK_LABELS}
        rotations["block_0"] = list(rotation)
        return _reward(extents).terms(
            _proprio(left=tuple(left)),
            _state(positions={
                "block_0": position,
                "block_1": [0.5, 0.3, TOP + size / 2.0],
                "block_2": [0.55, 0.3, TOP + size / 2.0],
            }, rotations=rotations, measured=(measured, 1.0), extents=extents))

    def test_real_x5_openings_reach_hold(self):
        official = {0.030: 0.4556, 0.035: 0.5019, 0.040: 0.5481}
        for size, value in official.items():
            with self.subTest(size=size):
                measured = _grip_measured(size)
                self.assertAlmostEqual(measured, value, delta=5e-4)
                held = self._pickup_terms(size, measured)
                open_ = self._pickup_terms(size, 1.0)
                empty = self._pickup_terms(size, EMPTY_CLOSED)
                self.assertTrue(held["grasp_ok"])
                self.assertFalse(open_["grasp_ok"])
                self.assertFalse(empty["grasp_ok"])

    def test_rotated_cube_uses_projected_jaw_width(self):
        size = 0.030
        angle = math.radians(45.0)
        rotation = (math.cos(angle / 2.0), 0.0, 0.0,
                    math.sin(angle / 2.0))
        projected = size * math.sqrt(2.0)
        matched = self._pickup_terms(
            size, _grip_measured(projected), rotation=rotation)
        side_grip = self._pickup_terms(
            size, _grip_measured(size), rotation=rotation)
        open_ = self._pickup_terms(size, 1.0, rotation=rotation)
        self.assertTrue(matched["grasp_ok"])
        self.assertFalse(side_grip["grasp_ok"])
        self.assertFalse(open_["grasp_ok"])


class SupportedBlockGraspTests(unittest.TestCase):
    """A cube still resting on the tower must not pay the grasp milestone."""

    @staticmethod
    def _held(positions, label):
        centre = np.asarray(positions[label], dtype=np.float64)
        return centre, _proprio(left=centre - np.array([0.143, 0.0, 0.0]))

    def test_stationary_supported_cube_pays_nothing(self):
        reward = _reward()
        # Away from the target so the base milestone cannot pay in this fixture.
        positions = _tower_positions(xy=(0.3, 0.2))
        _centre, proprio = self._held(positions, "block_1")
        state = _state(positions=positions, measured=(REAL_GRIP_30, 1.0))
        for _ in range(4):
            terms, bonus = reward.step(proprio, state)
            self.assertFalse(terms["grasp_ok"])
            self.assertAlmostEqual(bonus, 0.0)
        self.assertIsNone(reward.grasp_steps_by_block["block_1"])
        self.assertNotIn("block_1", reward._carry)

    def test_lifting_off_the_support_pays_grasp(self):
        reward = _reward()
        positions = _tower_positions(xy=(0.3, 0.2))
        positions["block_1"] = np.array(positions["block_1"]) + [0.0, 0.0, 0.10]
        _centre, proprio = self._held(positions, "block_1")
        state = _state(positions=positions, measured=(REAL_GRIP_30, 1.0))
        for _ in range(2):
            terms, bonus = reward.step(proprio, state)
            self.assertTrue(terms["grasp_ok"])
            self.assertAlmostEqual(bonus, 0.0)
        terms, bonus = reward.step(proprio, state)
        self.assertTrue(terms["grasp_ok"])
        self.assertAlmostEqual(bonus, M_GRASP)

    def test_tilted_lower_support_still_blocks_grasp(self):
        reward = _reward()
        angle = math.radians(20.0)
        positions = _tower_positions(xy=(0.3, 0.2))
        rotations = {label: [1.0, 0.0, 0.0, 0.0] for label in BLOCK_LABELS}
        rotations["block_0"] = [math.cos(angle / 2.0), math.sin(angle / 2.0),
                                0.0, 0.0]
        _centre, proprio = self._held(positions, "block_1")
        state = _state(positions=positions, rotations=rotations,
                       measured=(REAL_GRIP_30, 1.0))
        for _ in range(3):
            terms, bonus = reward.step(proprio, state)
            self.assertFalse(terms["grasp_ok"])
            self.assertAlmostEqual(bonus, 0.0)

    def test_bridged_support_still_blocks_grasp(self):
        reward = _reward()
        positions = {
            "block_0": [-HALF, 0.2, TOP + HALF],
            "block_1": [HALF, 0.2, TOP + HALF],
            "block_2": [0.0, 0.2, TOP + SIZE + HALF],
        }
        _centre, proprio = self._held(positions, "block_2")
        state = _state(positions=positions, measured=(REAL_GRIP_30, 1.0))
        for _ in range(3):
            terms, bonus = reward.step(proprio, state)
            self.assertFalse(terms["grasp_ok"])
            self.assertAlmostEqual(bonus, 0.0)

    def test_lift_below_the_ramp_pays_nothing(self):
        reward = _reward()
        positions = _tower_positions(xy=(0.3, 0.2))
        positions["block_1"] = np.array(positions["block_1"]) + [0.0, 0.0, 0.003]
        _centre, proprio = self._held(positions, "block_1")
        state = _state(positions=positions, measured=(REAL_GRIP_30, 1.0))
        for _ in range(3):
            terms, bonus = reward.step(proprio, state)
            self.assertFalse(terms["grasp_ok"])
            self.assertAlmostEqual(bonus, 0.0)

    def test_lift_clearance_includes_exact_quarter_side(self):
        for clearance, expected in ((SIZE / 4 - 1e-6, False), (SIZE / 4, True)):
            with self.subTest(clearance=clearance):
                reward = _reward()
                positions = _tower_positions(xy=(0.3, 0.2))
                positions["block_2"] = [0.5, 0.2, TOP + HALF]
                positions["block_1"][2] += clearance
                _, proprio = self._held(positions, "block_1")
                state = _state(positions=positions, measured=(REAL_GRIP_30, 1.0))
                for expected_bonus in (0.0, 0.0, M_GRASP if expected else 0.0):
                    terms, bonus = reward.step(proprio, state)
                    self.assertEqual(terms["grasp_ok"], expected)
                    self.assertAlmostEqual(bonus, expected_bonus)


class GraspAttemptTests(unittest.TestCase):
    def test_failed_entry_and_other_arm_success_are_counted_separately(self):
        reward = _reward()
        floor = _reset()
        positions = {label: np.asarray(block["pos"])
                     for label, block in floor["blocks"].items()}
        label = "block_0"
        point = positions[label]
        wrist = point - [DEFAULTS["grasp_offset"], 0, 0]
        lifted = {**positions, label: point + [0, 0, 0.05]}
        lifted_right = _proprio(right=wrist + [0, 0, 0.05])
        physical = _state(positions=lifted, measured=(1.0, REAL_GRIP_30))
        reward.terms(_proprio(left=wrist), floor)
        self.assertEqual(reward.grasp_attempts, {})
        reward.step(_proprio(), floor)
        for _ in range(2):
            reward.step(_proprio(left=wrist), floor)
        reward.step(_proprio(), floor)
        reward.step(_proprio(right=wrist), floor)
        for _ in range(5):
            reward.step(lifted_right, physical)
        self.assertEqual(reward.grasp_attempts, {
            "0:block_0": {"arrivals": 1, "confirmed": 0, "steps_to_grasp": 0},
            "1:block_0": {"arrivals": 1, "confirmed": 1, "steps_to_grasp": 3},
        })
        self.assertEqual(reward.grasp_steps_by_block[label], 8)
        self.assertAlmostEqual(reward.milestone_return, M_GRASP)
        reward.step(_proprio(), floor)
        for _ in range(3):
            reward.step(lifted_right, physical)
        self.assertEqual(reward.grasp_attempts["1:block_0"],
                         {"arrivals": 2, "confirmed": 2, "steps_to_grasp": 5})
        self.assertAlmostEqual(reward.milestone_return, M_GRASP)


class WaypointRewardTests(unittest.TestCase):
    @staticmethod
    def _wrist(point):
        return np.array(point) - np.array([DEFAULTS["grasp_offset"], 0.0, 0.0])

    @staticmethod
    def _positions(reset):
        return {label: np.array(reset["blocks"][label]["pos"]) for label in BLOCK_LABELS}

    def _hold(self, reward, positions, label, point, extents=None, arm=0, measured=None):
        moved = {**positions, label: np.asarray(point)}
        side = SIZE if extents is None else extents[label][0]
        opening = _grip_measured(side) if measured is None else measured
        proprio = _proprio(**{("left", "right")[arm]: self._wrist(point)})
        openings = (opening, 1.0) if arm == 0 else (1.0, opening)
        return reward.step(proprio, _state(positions=moved, measured=openings, extents=extents))[0]

    def test_approach_crosses_in_order_on_one_arm(self):
        reset = _reset()
        label = "block_0"
        goal = self._positions(reset)[label]
        left = goal + [0.0, -0.24, 0.0]
        right = goal + [0.0, 0.24, 0.0]
        def proprio(a, b):
            return _proprio(left=self._wrist(a), right=self._wrist(b))
        reward = StackingReward(reset, proprio(left, right), _cfg())
        route = f"approach:{label}"
        terms, _ = reward.step(proprio(left, right), reset)
        self.assertEqual(terms["waypoint_counts"][route], 0)
        off_route = left + [0.05, 0.0, 0.0]
        reward.step(proprio(off_route, right), reset)
        terms, _ = reward.step(proprio(off_route + [0.0, 0.13, 0.0], right), reset)
        self.assertEqual(terms["waypoint_counts"][route], 0)
        reward.step(proprio(left, right), reset)
        terms, _ = reward.step(proprio(left + [0.0, 0.08, 0.0], right), reset)
        self.assertEqual(terms["waypoint_counts"][route], 1)
        terms, _ = reward.step(proprio(left + [0.0, 0.08, 0.0], goal), reset)
        self.assertEqual(terms["waypoint_counts"][route], 1)
        terms, _ = reward.step(proprio(goal, goal), reset)
        self.assertEqual(terms["waypoint_counts"][route], 3)
        reward.step(proprio(left, right), reset)
        terms, _ = reward.step(proprio(goal, right), reset)
        self.assertEqual(terms["waypoint_counts"][route], 3)

    def test_zero_length_route_and_stationary_samples_pay_nothing(self):
        reset = _reset()
        point = self._positions(reset)["block_0"]
        proprio = _proprio(left=self._wrist(point))
        reward = StackingReward(reset, proprio, _cfg())
        for _ in range(4):
            terms, bonus = reward.step(proprio, reset)
            self.assertEqual(bonus, 0.0)
            self.assertEqual(terms["waypoint_counts"]["approach:block_0"], 0)

    def test_carry_starts_in_air_and_uses_each_support_height(self):
        sizes = _extents({label: [size] * 3 for label, size in zip(BLOCK_LABELS, (0.04, 0.03, 0.035))})
        reset = _reset(sizes)
        reward = StackingReward(reset, _proprio(), _cfg())
        positions = self._positions(reset)
        support_z = TOP
        for level, label in enumerate(BLOCK_LABELS):
            side = sizes[label][2]
            destination = np.array([0.0, -0.2, support_z + side / 2.0])
            lift_start = positions[label] + [0.0, 0.0, 0.13]
            key = f"carry:{label}"
            terms = self._hold(reward, positions, label, positions[label], sizes)
            self.assertEqual(terms["waypoint_counts"][key], 0)
            for _ in range(3):
                terms = self._hold(reward, positions, label, lift_start, sizes)
                self.assertEqual(terms["waypoint_counts"][key], 0)
            self.assertIsNotNone(reward.grasp_steps_by_block[label])
            # Sweeping through 25%, 50%, and 75% must use the actual unequal support heights.
            delta = destination - lift_start
            terms = self._hold(reward, positions, label, lift_start + 0.78 * delta, sizes)
            self.assertEqual(terms["waypoint_counts"][key], 3)
            milestone = ("base", "pair2", "pair3")[level]
            self.assertIsNone(reward.milestone_steps[milestone])
            positions[label] = destination
            for _ in range(3):
                reward.step(_proprio(), _state(positions=positions, extents=sizes))
            self.assertIsNotNone(reward.milestone_steps[milestone])
            support_z += side
        self.assertAlmostEqual(reward.milestone_return, M_LEVEL + 3 * M_GRASP)
        self.assertAlmostEqual(reward.waypoint_return,
                               sum(reward.waypoint_counts.values()) * DEFAULTS["waypoint_bonus"])

    def test_drop_and_other_arm_cannot_bridge_carried_waypoints(self):
        reset = _reset()
        reward = StackingReward(reset, _proprio(), _cfg())
        positions = self._positions(reset)
        label = "block_0"
        start = positions[label] + [0.0, 0.0, 0.10]
        goal = np.array([0.0, -0.2, TOP + HALF])
        direction = goal - start
        for _ in range(3):
            self._hold(reward, positions, label, start)
        terms = self._hold(reward, positions, label, start + 0.30 * direction)
        self.assertEqual(terms["waypoint_counts"]["carry:block_0"], 1)
        terms = self._hold(reward, positions, label, start + 0.85 * direction, measured=1.0)
        self.assertEqual(terms["waypoint_counts"]["carry:block_0"], 1)
        terms = self._hold(reward, positions, label, start + 0.85 * direction, arm=1)
        self.assertEqual(terms["waypoint_counts"]["carry:block_0"], 1)
        for _ in range(3):
            terms = self._hold(reward, positions, label, start + 0.85 * direction, arm=1)
        self.assertEqual(terms["waypoint_counts"]["carry:block_0"], 1)

    def test_disturbed_base_blocks_upper_carry(self):
        reset = _reset()
        reward = StackingReward(reset, _proprio(), _cfg())
        positions = self._positions(reset)
        positions["block_0"] = np.array([0.0, -0.2, TOP + HALF])
        for _ in range(3):
            reward.step(_proprio(), _state(positions=positions))
        self.assertIsNotNone(reward.milestone_steps["base"])
        label = "block_1"
        start = positions[label] + [0.0, 0.0, 0.10]
        goal = np.array([0.0, -0.2, TOP + 3 * HALF])
        for _ in range(3):
            self._hold(reward, positions, label, start)
        self.assertIn(label, reward._carry)
        positions["block_0"][1] += 0.15
        terms = self._hold(reward, positions, label, start + 0.8 * (goal - start))
        self.assertEqual(terms["waypoint_counts"]["carry:block_1"], 0)
        self.assertIsNone(reward.milestone_steps["pair2"])


class SupportTransportTests(unittest.TestCase):
    carried, lower, spare = BLOCK_LABELS
    route = f"carry:{carried}"

    def setUp(self):
        self.positions = {
            self.carried: np.array([-0.28, 0.16, TOP + HALF + 0.12]),
            self.lower: np.array([0.04, 0.16, TOP + HALF]),
            self.spare: np.array([-0.28, -0.36, TOP + HALF]),
        }
        self.start = self.positions[self.carried].copy()
        self.goal = self.positions[self.lower] + [0, 0, SIZE]

    def _step(self, reward, point, *, opening=REAL_GRIP_30, arm=0,
              other_opening=1.0, other_point=None, **physical):
        positions = {**self.positions, self.carried: point}
        wrists = {("left", "right")[arm]: WaypointRewardTests._wrist(point)}
        if other_point is not None:
            wrists[("right", "left")[arm]] = WaypointRewardTests._wrist(other_point)
        measured = (opening, other_opening) if arm == 0 else (other_opening, opening)
        return reward.step(_proprio(**wrists), _state(positions=positions, measured=measured, **physical))

    def _confirm(self, reward, arm=0, **physical):
        for _ in range(3):
            terms, _ = self._step(reward, self.start, arm=arm, **physical)
        return terms

    def test_transport_flag_is_strictly_boolean(self):
        for value in (0, 1, None, "false"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "transport_to_support"):
                _reward(transport_to_support=value)

    def test_post_confirmation_potential_and_waypoints_use_same_off_target_top(self):
        sizes = _extents({label: [side] * 3 for label, side in zip(BLOCK_LABELS, (SIZE, 0.04, SIZE))})
        self.positions[self.lower][2] = TOP + 0.02
        self.goal = self.positions[self.lower] + [0, 0, 0.02 + HALF]
        control = _reward(extents=sizes)
        enabled = _reward(extents=sizes, transport_to_support=True)
        for _ in range(2):
            observed, _ = self._step(enabled, self.start, extents=sizes)
            baseline, _ = self._step(control, self.start, extents=sizes)
            self.assertEqual(observed, baseline)
            self.assertEqual(enabled.transport_attempts, [])
        observed, _ = self._step(enabled, self.start, extents=sizes)
        baseline, _ = self._step(control, self.start, extents=sizes)
        self.assertGreater(observed["phi_manipulation"], baseline["phi_manipulation"])
        self.assertEqual(enabled.transport_attempts[0]["target_lower"], self.lower)
        self.assertIsNone(control.transport_attempts[0]["target_lower"])
        self.assertEqual(enabled.transport_attempts[0]["grasp_step"], 3)
        np.testing.assert_allclose(enabled._carry[self.carried][1].target, self.goal)

        self.positions[self.lower][0] += 0.001
        self.goal[0] += 0.001
        observed, _ = self._step(enabled, self.start, extents=sizes)
        self.assertEqual(observed["waypoint_counts"][self.route], 0)
        np.testing.assert_allclose(enabled._carry[self.carried][1].target, self.goal)
        point = self.start + 0.80 * (self.goal - self.start)
        observed, _ = self._step(enabled, point, extents=sizes)
        baseline, _ = self._step(control, point, extents=sizes)
        self.assertEqual(observed["waypoint_counts"][self.route], 3)
        self.assertEqual(baseline["waypoint_counts"][self.route], 0)
        self.assertGreater(observed["phi_manipulation"], baseline["phi_manipulation"])
        self.assertFalse(observed["base_ok"])
        self.assertEqual(observed["stable_tower_levels"], 0)

    def test_target_motion_cannot_cross_a_waypoint_for_stationary_carried_block(self):
        reward = _reward(transport_to_support=True)
        self._confirm(reward)
        point = self.start + 0.249 * (self.goal - self.start)
        terms, _ = self._step(reward, point)
        self.assertEqual(terms["waypoint_counts"][self.route], 0)
        self.positions[self.lower][0] -= 0.004
        terms, _ = self._step(reward, point)
        path = reward._carry[self.carried][1]
        self.assertGreater((point - path.start) @ path.axis, path.spacing)
        self.assertEqual(terms["waypoint_counts"][self.route], 0)
        self._step(reward, self.start)
        terms, _ = self._step(reward, point)
        self.assertEqual(terms["waypoint_counts"][self.route], 1)

    def test_invalidation_cannot_hop_or_resume_until_fresh_same_arm_confirmation(self):
        reward = _reward(transport_to_support=True)
        self._confirm(reward)
        # Another released tabletop block becomes closer without disturbing the chosen one.
        self.positions[self.spare] = self.start * [1, 1, 0] + [0.0, -0.04, TOP + HALF]
        self._step(reward, self.start)
        np.testing.assert_allclose(reward._carry[self.carried][1].target, self.goal)

        point = self.start + 0.80 * (self.goal - self.start)
        fast = {label: [0.0, 0.0, 0.0] for label in BLOCK_LABELS}
        fast[self.lower] = [0.02, 0.0, 0.0]
        invalid, _ = self._step(reward, point, linear=fast)
        self.assertEqual(invalid["waypoint_counts"][self.route], 0)
        self.assertNotIn(self.carried, reward._carry)
        recovered, _ = self._step(reward, point)
        self.assertEqual(recovered["waypoint_counts"][self.route], 0)
        self.assertEqual(recovered["phi_manipulation"], invalid["phi_manipulation"])
        self.assertEqual(len(reward.transport_attempts), 1)
        self.assertEqual(reward.transport_attempts[0]["target_lower"], self.lower)

        # A new attempt starts only after the original hold is lost and the new arm dwells.
        self._step(reward, self.start, opening=1.0)
        for _ in range(2):
            self._step(reward, self.start, arm=1)
            self.assertEqual(len(reward.transport_attempts), 1)
        self._step(reward, self.start, arm=1)
        self.assertEqual([row["arm"] for row in reward.transport_attempts], [0, 1])
        self.assertEqual(reward.transport_attempts[-1]["target_lower"], self.spare)
        np.testing.assert_allclose(reward._carry[self.carried][1].target,
                                   self.positions[self.spare] + [0, 0, SIZE])
        self.assertAlmostEqual(reward.milestone_return, M_GRASP)

    def test_initial_fixed_table_fallback_stays_fixed_when_support_becomes_eligible(self):
        reward = _reward(transport_to_support=True)
        # One is hovering above the hard table-gap tolerance; one is closed in the other arm.
        self.positions[self.lower][2] += 0.001
        self._confirm(reward, other_opening=REAL_GRIP_30, other_point=self.positions[self.spare])
        first = reward.transport_attempts[0]
        self.assertIsNone(first["target_lower"])
        fixed = np.array([0.0, -0.2, TOP + HALF])
        np.testing.assert_allclose(reward._carry[self.carried][1].target, fixed)
        self.positions[self.lower][2] -= 0.001
        terms, _ = self._step(reward, self.start + 0.80 * (fixed - self.start))
        self.assertEqual(terms["waypoint_counts"][self.route], 3)
        self.assertIsNone(first["target_lower"])

    def test_off_target_timeline_requires_alignment_open_and_quiet_same_id_dwell(self):
        for enabled in (False, True):
            with self.subTest(transport_to_support=enabled):
                reward = _reward(transport_to_support=enabled)
                self._confirm(reward)
                hover = self.goal + [0, 0, 0.005]
                held, _ = self._step(reward, hover)
                self.assertEqual(held["settled_pair"], 0.0)
                row = reward.transport_attempts[0]
                self.assertEqual(row["aligned_step"], 4)
                self.assertEqual(row["lower"], self.lower)
                self.assertIsNone(row["opened_step"])

                # Hovering while open does not count as settled, even if it is quiet for many steps.
                for _ in range(4):
                    self._step(reward, hover, opening=1.0, other_opening=EMPTY_CLOSED)
                self.assertEqual(row["opened_step"], 5)
                self.assertIsNone(row["settled_step"])
                noisy = {label: [0.0, 0.0, 0.0] for label in BLOCK_LABELS}
                noisy[self.carried] = [0, 0, 0.02]
                self._step(reward, self.goal, opening=1.0)
                self._step(reward, self.goal, opening=1.0, linear=noisy)
                self._step(reward, self.goal, opening=1.0)
                self._step(reward, self.goal, opening=1.0)
                for _ in range(3):
                    reward.terms(_proprio(), _state(positions={**self.positions, self.carried: self.goal}))
                self.assertIsNone(row["settled_step"])
                terms, _ = self._step(reward, self.goal, opening=1.0)
                self.assertEqual(row["settled_step"], 13)
                self.assertFalse(terms["pair2_ok"])
                self.assertIsNone(reward.milestone_steps["pair2"])
                terminal_row = dict(row)
                for _ in range(3):
                    self._step(reward, hover, opening=1.0)
                self.assertEqual(reward.transport_attempts, [terminal_row])
                self.assertEqual(_reward(transport_to_support=enabled).transport_attempts, [])

    def test_timeline_preserves_alignment_through_partial_jaw_opening(self):
        for enabled in (False, True):
            with self.subTest(transport_to_support=enabled):
                reward = _reward(transport_to_support=enabled)
                self._confirm(reward)
                self._step(reward, self.goal)
                partial = self._step(reward, self.goal, opening=0.65)[0]
                self.assertFalse(partial["grasp_ok"])
                row = reward.transport_attempts[0]
                self.assertEqual(row["aligned_step"], 4)
                self.assertIsNone(row["opened_step"])
                for _ in range(3):
                    self._step(reward, self.goal, opening=0.8)
                self.assertEqual(row["opened_step"], 6)
                self.assertEqual(row["settled_step"], 8)

    def test_timeline_never_switches_aligned_lower_or_counts_wrong_arm_open(self):
        reward = _reward(transport_to_support=True)
        self._confirm(reward)
        self._step(reward, self.goal)
        row = reward.transport_attempts[0]
        self.assertEqual(row["lower"], self.lower)
        other_goal = self.positions[self.spare] + [0, 0, SIZE]
        self._step(reward, other_goal)
        for _ in range(3):
            self._step(reward, other_goal, opening=1.0)
        self.assertIsNone(row["opened_step"])
        self.assertIsNone(row["settled_step"])

        self._confirm(reward)
        self._step(reward, self.goal)
        row = reward.transport_attempts[-1]
        for _ in range(3):
            # The grasping arm closes without a cube; the other hand is open.
            self._step(reward, self.goal, opening=EMPTY_CLOSED)
        self.assertEqual(row["lower"], self.lower)
        self.assertIsNone(row["opened_step"])
        self.assertIsNone(row["settled_step"])

    def test_two_layer_tabletop_tower_is_a_carry_destination(self):
        reward = _reward(transport_to_support=True)
        self.positions[self.spare] = np.array([-0.28, -0.36, TOP + HALF])
        self.positions[self.lower] = self.positions[self.spare] + [0, 0, SIZE]
        self._confirm(reward)
        row = reward.transport_attempts[0]
        self.assertEqual(row["target_lower"], self.lower)
        self.assertEqual(reward._carry_destinations[self.carried].anchor,
                         (self.spare, self.lower))
        np.testing.assert_allclose(reward._carry[self.carried][1].target,
                                   self.positions[self.lower] + [0, 0, SIZE])

    def test_blocking_neighbor_body_removes_the_target_slot(self):
        tilt = np.pi / 4
        half = 0.5 * SIZE * (abs(np.sin(tilt)) + abs(np.cos(tilt)))
        rotations = {label: [1.0, 0.0, 0.0, 0.0] for label in BLOCK_LABELS}
        rotations[self.spare] = [float(np.cos(tilt)), 0.0, float(np.sin(tilt)), 0.0]
        reward = _reward(transport_to_support=True)
        # A tilted neighbour on the table is not itself a released destination,
        # but its projected body fills the slot above the lower block.
        self.positions[self.spare] = np.array(
            [self.positions[self.lower][0] + 0.024, self.positions[self.lower][1],
             TOP + half])
        self._confirm(reward, rotations=rotations)
        row = reward.transport_attempts[0]
        self.assertIsNone(row["target_lower"])
        self.assertEqual(reward._carry_destinations[self.carried].anchor, ())

    def test_wrong_arm_takeover_cannot_open_or_settle_the_old_attempt(self):
        for enabled in (False, True):
            with self.subTest(transport_to_support=enabled):
                reward = _reward(transport_to_support=enabled)
                self._confirm(reward)
                self._step(reward, self.goal)
                row = reward.transport_attempts[0]
                self.assertEqual(row["aligned_step"], 4)
                self.assertIsNone(row["opened_step"])
                # The grasping arm leaves for another block while the carried block
                # rests on the lower one; a vacuum release must not credit it.
                carried = {**self.positions, self.carried: self.goal}
                elsewhere = WaypointRewardTests._wrist(self.positions[self.spare] + [0, 0, 0.1])
                for opening in (0.65, 0.65, 0.8, 0.8):
                    reward.step(_proprio(left=elsewhere),
                                _state(positions=carried, measured=(opening, EMPTY_CLOSED)))
                self.assertIsNone(row["opened_step"])
                self.assertIsNone(row["settled_step"])

    def test_reclosing_after_a_lost_hold_needs_a_fresh_confirmed_grasp(self):
        for enabled in (False, True):
            with self.subTest(transport_to_support=enabled):
                reward = _reward(transport_to_support=enabled)
                self._confirm(reward)
                self._step(reward, self.goal)
                row = reward.transport_attempts[0]
                self.assertEqual(row["lower"], self.lower)
                self._step(reward, self.goal, opening=0.65)
                self._step(reward, self.goal, opening=REAL_GRIP_30)
                for _ in range(3):
                    self._step(reward, self.goal, opening=0.8)
                self.assertIsNone(row["opened_step"])
                self.assertIsNone(row["settled_step"])

    def test_hard_tower_disturbance_invalidates_then_fresh_attempt_uses_tabletop_support(self):
        reward = _reward(transport_to_support=True)
        self.positions[self.lower] = np.array([0, -0.2, TOP + HALF])
        for _ in range(3):
            reward.step(_proprio(), _state(positions=self.positions))
        self._confirm(reward)
        self.assertEqual(reward.transport_attempts[0]["target_lower"], self.lower)
        self.positions[self.lower][2] += 0.02
        terms, _ = self._step(reward, self.start + 0.8 * (np.array([0, -0.2, TOP + 3 * HALF]) - self.start))
        self.assertEqual(terms["waypoint_counts"][self.route], 0)
        self.assertNotIn(self.carried, reward._carry)
        self._step(reward, self.start, opening=1.0)
        self._confirm(reward)
        self.assertEqual(reward.transport_attempts[-1]["target_lower"], self.spare)
        goal = self.positions[self.spare] + [0, 0, SIZE]
        terms, _ = self._step(reward, self.start + 0.8 * (goal - self.start))
        self.assertEqual(terms["waypoint_counts"][self.route], 3)
        self.assertEqual(reward.milestone_steps["base"], 3)
        self.assertIsNone(reward.milestone_steps["pair2"])

    def test_established_hard_tower_top_and_hard_milestones_are_preserved(self):
        reward = _reward(transport_to_support=True)
        tower = _tower_positions()
        # The tracked carried block will be placed on the existing two-block tower.
        self.positions[self.lower] = np.array(tower[self.carried])
        self.positions[self.spare] = np.array(tower[self.lower])
        for _ in range(3):
            reward.step(_proprio(), _state(positions=self.positions))
        self.assertEqual(reward.milestone_steps, dict(grasp=None, base=3, pair2=3, pair3=None))
        self._confirm(reward)
        self.assertEqual(reward.transport_attempts[0]["target_lower"], self.spare)
        goal = np.array(tower[self.spare])
        np.testing.assert_allclose(reward._carry[self.carried][1].target, goal)
        self._step(reward, goal)
        for _ in range(3):
            terms, _ = self._step(reward, goal, opening=1.0)
        self.assertTrue(terms["pair3_ok"])
        self.assertEqual(reward.milestone_steps["pair3"], 10)
        self.assertEqual(reward.transport_attempts[0]["settled_step"], 10)
        self.assertAlmostEqual(reward.milestone_return, M_LEVEL + M_GRASP)


class DenseRouteTests(unittest.TestCase):
    def setUp(self):
        self.reset = _reset()
        self.positions = WaypointRewardTests._positions(self.reset)
        self.reward = _reward(dense_transport_reward=True)
        self.base_goal = np.array([0.0, -0.2, TOP + HALF])

    def step_at(self, label, point, opening=REAL_GRIP_30, *, arm=0, linear=None):
        self.positions[label] = np.array(point)
        wrist = WaypointRewardTests._wrist(point)
        proprio = _proprio(**{("left", "right")[arm]: wrist})
        measured = (opening, 1.0) if arm == 0 else (1.0, opening)
        return self.reward.step(proprio, _state(self.positions, measured=measured, linear=linear))

    def lift(self, label, start, *, arm=0):
        for _ in range(3):
            self.step_at(label, start, arm=arm)

    def test_approach_rewards_small_steps_without_repaying_motion_or_moving_target(self):
        label = BLOCK_LABELS[0]
        point = self.positions[label].copy()
        start = point + [0.0, -0.19, 0.0]
        wrist = lambda offset: _proprio(left=WaypointRewardTests._wrist(start + [0.0, offset, 0.0]))
        self.reward = StackingReward(self.reset, wrist(0.0), _cfg(dense_transport_reward=True))
        for offset in (0.005, 0.010, 0.015):
            terms, bonus = self.reward.step(wrist(offset), self.reset)
            self.assertGreater(bonus, 0.0)
            self.assertEqual(terms["dense_transport_return"], 0.0)
            self.assertFalse(any(terms["waypoint_counts"].values()))
        for offset in (0.005, 0.015, 0.015):
            self.assertEqual(self.reward.step(wrist(offset), self.reset)[1], 0.0)
        self.positions[label] = point + [0.0, -0.01, 0.0]
        self.assertEqual(self.reward.step(wrist(0.015), _state(self.positions))[1], 0.0)

    def test_displaced_base_can_earn_unused_approach_progress(self):
        label = BLOCK_LABELS[0]
        self.positions[label] = self.base_goal.copy()
        point = self.base_goal + [0.06, 0.0, 0.0]
        far = _proprio(left=WaypointRewardTests._wrist(point + [0.0, -0.10, 0.0]))
        near = _proprio(left=WaypointRewardTests._wrist(point + [0.0, -0.02, 0.0]))
        for _ in range(3):
            self.reward.step(far, _state(self.positions))
        self.assertIsNotNone(self.reward.milestone_steps["base"])
        key = ("approach", label)
        before = self.reward._progress_credit.get(key, 0.0)
        self.positions[label] = point
        self.reward.step(far, _state(self.positions))
        self.assertEqual(self.reward._progress_credit.get(key, 0.0), before)
        self.reward.step(near, _state(self.positions))
        paid = self.reward._progress_credit[key]
        self.assertGreater(paid, before)
        for action in (far, near):
            self.reward.step(action, _state(self.positions))
        self.assertEqual(self.reward._progress_credit[key], paid)

    def test_carried_progress_starts_after_confirmation_and_survives_retry(self):
        label = BLOCK_LABELS[0]
        start = self.positions[label] + [0, 0, 0.07]
        delta = self.base_goal - start
        for factor in (0.00, 0.01, 0.02):
            terms, _ = self.step_at(label, start + factor * delta)
            self.assertEqual(terms["dense_transport_return"], 0.0)
        last = 0.0
        for factor in (0.03, 0.04, 0.05):
            terms, _ = self.step_at(label, start + factor * delta)
            self.assertGreater(terms["dense_transport_return"], last)
            last = terms["dense_transport_return"]
        for factor in (0.04, 0.05, 0.05):
            terms, _ = self.step_at(label, start + factor * delta)
            self.assertEqual(terms["dense_transport_return"], last)
        self.step_at(label, start, opening=1.0)
        self.lift(label, start)
        terms, _ = self.step_at(label, start + 0.05 * delta)
        self.assertEqual(terms["dense_transport_return"], last)
        terms, _ = self.step_at(label, start + 0.06 * delta)
        self.assertGreater(terms["dense_transport_return"], last)

    def test_long_carry_pays_equal_credit_for_equal_new_progress(self):
        label = BLOCK_LABELS[0]
        start = self.base_goal + [0, -0.45, 0.07]
        self.lift(label, start)
        distance = np.linalg.norm(start - self.base_goal)
        for fraction in (0.01, 0.02, 0.03):
            terms, _ = self.step_at(label, start + fraction * (self.base_goal - start))
            self.assertAlmostEqual(terms["dense_transport_return"], 0.45 * fraction)
        terms, _ = self.step_at(label, self.base_goal)
        self.assertAlmostEqual(terms["dense_transport_return"], 0.45)
        self.assertGreater(distance, self.reward.cfg["placement_soft_scale"])

    def test_release_and_dwell_are_separate_and_target_tracks_the_actual_tower(self):
        before = self.reward.milestone_return
        xy = self.base_goal[:2] + [0.005, 0.0]
        for level, (label, placement, total) in enumerate(zip(
                BLOCK_LABELS, ("base", "pair2", "pair3"), (M_BASE, M_PAIR2, M_PAIR3))):
            goal = np.array([*xy, TOP + (2 * level + 1) * HALF])
            start = self.positions[label] + [0.0, 0.0, 0.07]
            self.lift(label, start)
            before += M_GRASP
            self.assertEqual(self.reward.transport_attempts[-1]["target_lower"],
                             BLOCK_LABELS[level - 1] if level else None)
            if level == 1:
                initial, _ = self.step_at(label, start)
                xy += [0.001, 0.0]
                self.positions[BLOCK_LABELS[0]][:2] = xy
                stationary, _ = self.step_at(label, start)
                self.assertEqual(stationary["dense_transport_return"], initial["dense_transport_return"])
                goal[:2] = xy
            self.step_at(label, goal)
            self.step_at(label, goal, opening=0.65)
            self.assertIsNone(self.reward.release_steps[placement])
            self.step_at(label, goal, opening=0.8)
            self.assertIsNotNone(self.reward.release_steps[placement])
            self.assertIsNone(self.reward.milestone_steps[placement])
            self.assertAlmostEqual(self.reward.milestone_return, before + 0.6 * total)
            fast = dict.fromkeys(BLOCK_LABELS, [0.0, 0.0, 0.0])
            fast[label] = [0.02, 0.0, 0.0]
            self.step_at(label, goal, opening=1.0, linear=fast)
            for _ in range(2):
                self.step_at(label, goal, opening=1.0)
            self.assertIsNone(self.reward.milestone_steps[placement])
            self.step_at(label, goal, opening=1.0)
            self.assertIsNotNone(self.reward.milestone_steps[placement])
            before += total
            for _ in range(3):
                terms, _ = self.step_at(label, goal, opening=1.0)
            self.assertAlmostEqual(self.reward.milestone_return, before)
        self.assertLessEqual(terms["dense_transport_return"], 3 * 0.45)
        self.assertLessEqual(terms["dense_approach_return"], 3 * 0.45)

    def test_temporary_support_motion_pauses_progress_without_losing_placed_identity(self):
        base, upper, _ = BLOCK_LABELS
        self.lift(base, self.positions[base] + [0, 0, 0.07])
        self.step_at(base, self.base_goal)
        for _ in range(3):
            self.step_at(base, self.base_goal, opening=1.0)
        start = self.positions[upper] + [0, 0, 0.07]
        self.lift(upper, start)
        before, _ = self.step_at(upper, start)
        fast = dict.fromkeys(BLOCK_LABELS, [0.0, 0.0, 0.0])
        fast[base] = [0.02, 0.0, 0.0]
        goal = self.base_goal + [0, 0, SIZE]
        middle = (start + goal) / 2
        moving, _ = self.step_at(upper, middle, linear=fast)
        recovered, _ = self.step_at(upper, middle)
        self.assertEqual(moving["dense_transport_return"], before["dense_transport_return"])
        self.assertEqual(recovered["dense_transport_return"], before["dense_transport_return"])
        final, _ = self.step_at(upper, goal)
        self.assertGreater(final["dense_transport_return"], before["dense_transport_return"])
        for _ in range(3):
            self.step_at(upper, goal, opening=1.0)
        self.assertIsNotNone(self.reward.milestone_steps["pair2"])
        self.assertEqual(len(self.reward.transport_attempts), 2)

    def test_wrong_arm_and_off_goal_opening_do_not_pay_release_bonus(self):
        label = BLOCK_LABELS[0]
        start = self.positions[label] + [0, 0, 0.07]
        self.lift(label, start)
        self.step_at(label, self.base_goal)
        wrist = WaypointRewardTests._wrist(self.base_goal)
        for _ in range(5):
            # Arm 1 opens at the goal while arm 0 still holds the block.
            self.reward.step(_proprio(left=wrist, right=wrist),
                             _state(self.positions, measured=(REAL_GRIP_30, 1.0)))
        self.assertIsNone(self.reward.release_steps["base"])
        self.assertIsNone(self.reward.milestone_steps["base"])
        self.lift(label, start)
        before = self.reward.milestone_return
        off_goal = self.base_goal + [0.05, 0.0, 0.04]
        self.step_at(label, off_goal, opening=1.0)
        for _ in range(5):
            self.step_at(label, off_goal, opening=1.0)
        self.assertIsNone(self.reward.release_steps["base"])
        self.assertIsNone(self.reward.milestone_steps["base"])
        self.assertAlmostEqual(self.reward.milestone_return, before)

    def test_arrive_and_open_same_step_pays_release_and_settle(self):
        label = BLOCK_LABELS[0]
        start = self.positions[label] + [0, 0, 0.07]
        self.lift(label, start)
        self.step_at(label, self.base_goal + [0, 0, 0.02])
        self.step_at(label, self.base_goal, opening=0.8)
        self.assertIsNotNone(self.reward.release_steps["base"])
        for _ in range(2):
            self.step_at(label, self.base_goal, opening=1.0)
        self.assertIsNotNone(self.reward.milestone_steps["base"])
        self.assertAlmostEqual(self.reward.milestone_return, M_GRASP + M_BASE)

    def test_release_pays_when_the_same_arm_opens_at_the_goal_after_hold_loss(self):
        for one_hold_frame in (False, True):
            with self.subTest(one_hold_frame=one_hold_frame):
                self.setUp()
                label = BLOCK_LABELS[0]
                near_goal = self.base_goal + [0, 0, 0.005]
                self.lift(label, self.positions[label] + [0, 0, 0.07])
                self.step_at(label, near_goal)
                self.step_at(label, near_goal, opening=EMPTY_CLOSED)
                if one_hold_frame:
                    self.step_at(label, near_goal)
                self.step_at(label, self.base_goal, opening=0.8)
                self.assertIsNotNone(self.reward.release_steps["base"])
                row = self.reward.transport_attempts[0]
                self.assertIsNotNone(row["opened_step"])
                self.assertEqual(row["end_reason"], "released")

    def test_dropping_off_goal_ends_attempt_restores_approach_and_starts_fresh_attempt(self):
        label = BLOCK_LABELS[0]
        drop = self.positions[label] + [0.0, 0.0, 0.05]
        self.lift(label, drop)
        self.assertEqual(len(self.reward.transport_attempts), 1)
        # Drop the block so it falls away from the gripper and off the goal.
        landed = drop + [0.10, 0.0, -0.05]
        self.positions[label] = landed.copy()
        far = _proprio(left=WaypointRewardTests._wrist(landed + [0.0, -0.15, 0.0]))
        near = _proprio(left=WaypointRewardTests._wrist(landed + [0.0, -0.02, 0.0]))
        self.reward.step(far, _state(self.positions, measured=(1.0, 1.0)))
        self.assertEqual(self.reward.transport_attempts[0]["end_reason"], "dropped")
        self.assertNotIn(label, self.reward._carry_destinations)
        before_approach = self.reward._dense_approach_return
        self.reward.step(near, _state(self.positions, measured=(1.0, 1.0)))
        self.assertGreater(self.reward._dense_approach_return, before_approach)
        # Re-grasping creates a fresh transport attempt rather than reusing the dropped one.
        self.lift(label, landed + [0.0, 0.0, 0.05])
        self.assertEqual(len(self.reward.transport_attempts), 2)
        self.assertGreater(self.reward.transport_attempts[1]["grasp_step"],
                           self.reward.transport_attempts[0]["grasp_step"])

    def test_external_move_to_goal_while_attempt_arm_is_far_does_not_pay_release(self):
        label = BLOCK_LABELS[0]
        start = self.positions[label] + [0.0, 0.0, 0.07]
        self.lift(label, start)
        self.positions[label] = self.base_goal.copy()
        self.reward.step(_proprio(), _state(self.positions, measured=(1.0, 1.0)))
        self.assertEqual(self.reward.transport_attempts[0]["end_reason"], "dropped")
        self.assertIsNone(self.reward.release_steps["base"])
        self.assertIsNone(self.reward.transport_attempts[0]["opened_step"])

    def test_slow_opening_after_settle_still_pays_release_unless_arm_moves_to_other_block(self):
        label = BLOCK_LABELS[0]
        self.lift(label, self.positions[label] + [0.0, 0.0, 0.07])
        self.step_at(label, self.base_goal)
        for opening in (0.60, 0.70, 0.78):
            self.reward.step(_proprio(), _state(self.positions, measured=(opening, 1.0)))
        self.assertIsNotNone(self.reward.milestone_steps["base"])
        self.assertIsNone(self.reward.release_steps["base"])
        self.assertAlmostEqual(self.reward.milestone_return, M_GRASP + 0.4 * M_BASE)
        self.reward.step(_proprio(), _state(self.positions, measured=(0.82, 1.0)))
        self.assertIsNotNone(self.reward.release_steps["base"])
        row = self.reward.transport_attempts[0]
        self.assertEqual(row["end_reason"], "released")
        self.assertEqual(row["settled_step"], self.reward.milestone_steps["base"])
        self.assertAlmostEqual(self.reward.milestone_return, M_GRASP + M_BASE)

        # If the arm moves to another block before opening, no release bonus is paid.
        self.setUp()
        other = BLOCK_LABELS[1]
        self.lift(label, self.positions[label] + [0.0, 0.0, 0.07])
        self.step_at(label, self.base_goal)
        at_other = _proprio(left=WaypointRewardTests._wrist(self.positions[other]))
        self.reward.step(at_other, _state(self.positions, measured=(EMPTY_CLOSED, 1.0)))
        self.assertEqual(self.reward.transport_attempts[0]["end_reason"], "other_block")
        self.reward.step(at_other, _state(self.positions, measured=(1.0, 1.0)))
        self.assertIsNone(self.reward.release_steps["base"])

    def test_opening_can_complete_while_the_same_arm_withdraws(self):
        label = BLOCK_LABELS[0]
        self.lift(label, self.positions[label] + [0, 0, 0.07])
        self.step_at(label, self.base_goal)
        self.step_at(label, self.base_goal, opening=0.65)
        self.reward.step(_proprio(), _state(self.positions, measured=(0.8, 1.0)))
        row = self.reward.transport_attempts[0]
        self.assertEqual(row["opened_step"], 6)
        self.assertIsNone(row["settled_step"])
        for _ in range(2):
            self.reward.step(_proprio(), _state(self.positions))
        self.assertEqual(row["settled_step"], 8)
        self.assertAlmostEqual(self.reward.milestone_return, M_GRASP + M_BASE)

    def test_another_block_cannot_collect_the_released_blocks_settle_bonus(self):
        carried, other, _ = BLOCK_LABELS
        self.lift(carried, self.positions[carried] + [0, 0, 0.07])
        self.step_at(carried, self.base_goal + [0, 0, 0.005])
        self.step_at(carried, self.base_goal + [0, 0, 0.005], opening=0.8)
        self.assertIsNotNone(self.reward.release_steps["base"])
        self.positions[carried] = self.base_goal + [0.05, 0, 0]
        self.positions[other] = self.base_goal.copy()
        for _ in range(4):
            self.reward.step(_proprio(), _state(self.positions))
        self.assertIsNone(self.reward.milestone_steps["base"])
        self.assertIsNone(self.reward.transport_attempts[0]["settled_step"])
        self.assertAlmostEqual(self.reward.milestone_return, M_GRASP + 0.6 * M_BASE)

    def test_geometric_tower_without_carry_attempt_pays_settle_fallback(self):
        for level, (label, name) in enumerate(zip(BLOCK_LABELS, ("base", "pair2", "pair3"))):
            self.positions[label] = self.base_goal + [0, 0, level * SIZE]
        for _ in range(3):
            terms, _ = self.step_at(BLOCK_LABELS[0], self.positions[BLOCK_LABELS[0]],
                                    opening=1.0, arm=0)
        # The full stack must end with zero potential so the true terminal
        # branch cannot pay a spurious negative shaping reward.
        self.assertAlmostEqual(terms["phi"], 0.0, places=9)
        self.assertIsNone(self.reward.release_steps["base"])
        self.assertIsNotNone(self.reward.milestone_steps["base"])
        self.assertIsNotNone(self.reward.milestone_steps["pair2"])
        self.assertIsNotNone(self.reward.milestone_steps["pair3"])
        self.assertAlmostEqual(self.reward.milestone_return, 0.4 * (M_BASE + M_PAIR2 + M_PAIR3))

    def test_brief_hold_dropout_and_concurrent_tower_growth_preserve_dense_carry(self):
        base, upper, _ = BLOCK_LABELS
        start_upper = self.positions[upper] + [0, 0, 0.07]
        self.lift(upper, start_upper, arm=1)
        # Brief 1-step hold dropout without opening (measured=0.0 < REAL_GRIP_30) pauses, does not drop attempt.
        mid = (start_upper + self.base_goal) / 2
        self.step_at(upper, mid, opening=0.0, arm=1)
        # While upper is held by arm 1, base settles on the table so anchor grows to (base,).
        self.positions[base] = self.base_goal.copy()
        for _ in range(3):
            terms, _ = self.step_at(upper, mid, opening=REAL_GRIP_30, arm=1)
        self.assertIsNotNone(self.reward.milestone_steps["base"])
        before = terms["dense_transport_return"]
        goal = self.base_goal + [0, 0, SIZE]
        terms, _ = self.step_at(upper, goal, opening=REAL_GRIP_30, arm=1)
        self.assertGreater(terms["dense_transport_return"], before)
        self.step_at(upper, goal, opening=0.8, arm=1)
        self.assertIsNotNone(self.reward.release_steps["pair2"])
        for _ in range(2):
            self.step_at(upper, goal, opening=1.0, arm=1)
        self.assertIsNotNone(self.reward.milestone_steps["pair2"])

    def test_placement_radius_04_anchors_layer2_with_vertical_clearance(self):
        base, upper, _ = BLOCK_LABELS
        # Base placed 3.5 cm from target center (outside old 1 cm radius, inside new 4 cm radius).
        off_center = self.base_goal + [0.035, 0.0, 0.0]
        self.lift(base, self.positions[base] + [0.0, 0.0, 0.07])
        self.step_at(base, off_center)
        self.step_at(base, off_center, opening=0.8)
        for _ in range(3):
            self.step_at(base, off_center, opening=1.0)
        self.assertIsNotNone(self.reward.milestone_steps["base"])
        self.assertEqual(self.reward._settled_anchor, (base,))

        # Lifting 2nd block sets target_lower to base, with higher Z + vertical clearance during approach.
        side_pos = np.array([off_center[0] + 0.12, off_center[1], TOP + 3 * HALF])
        self.lift(upper, side_pos, arm=1)
        self.assertEqual(self.reward.transport_attempts[-1]["target_lower"], base)
        low_terms, _ = self.step_at(upper, side_pos, arm=1)
        # Moving UP to clear base's top earns positive carry progress.
        high_pos = side_pos + [0.0, 0.0, 0.6 * SIZE]
        high_terms, _ = self.step_at(upper, high_pos, arm=1)
        self.assertGreater(high_terms["dense_transport_return"], low_terms["dense_transport_return"])
        # Moving to exact top of base completes carry to full 0.45 and pays pair2.
        goal2 = off_center + [0.0, 0.0, SIZE]
        final_terms, _ = self.step_at(upper, goal2, arm=1)
        self.assertGreater(final_terms["dense_transport_return"], high_terms["dense_transport_return"])
        self.step_at(upper, goal2, opening=0.8, arm=1)
        for _ in range(3):
            self.step_at(upper, goal2, opening=1.0, arm=1)
        self.assertIsNotNone(self.reward.milestone_steps["pair2"])
        self.assertEqual(self.reward.penalty_return, 0.0)

    def test_drop_penalty_triggers_with_flicker_guard_and_episode_cap(self):
        label = BLOCK_LABELS[0]
        drop_pen = DEFAULTS["drop_penalty"]
        start_home = WaypointRewardTests._positions(self.reset)[label].copy()
        # 1) Single-frame closed-jaw flicker does NOT drop or penalize.
        start = start_home + [0.0, 0.0, 0.07]
        self.lift(label, start)
        terms_flicker, _ = self.step_at(label, start, opening=0.0)
        self.assertEqual(terms_flicker["penalty_counts"]["drop"], 0)
        self.assertEqual(self.reward.penalty_return, 0.0)
        self.assertIsNone(self.reward.transport_attempts[-1]["end_reason"])
        # Recovering hold clears pending counter.
        self.step_at(label, start, opening=REAL_GRIP_30)
        self.assertEqual(self.reward.penalty_counts["drop"], 0)

        # 2) Two consecutive closed-jaw unheld frames off-goal confirm drop once.
        self.step_at(label, start, opening=0.0)
        self.assertEqual(self.reward.penalty_counts["drop"], 0)
        terms_drop, bonus = self.step_at(label, start, opening=0.0)
        self.assertEqual(self.reward.transport_attempts[-1]["end_reason"], "dropped")
        self.assertEqual(terms_drop["penalty_counts"]["drop"], 1)
        self.assertAlmostEqual(terms_drop["penalty_return"], -drop_pen)
        self.assertLess(bonus, 0.0)

        # 3) Opening off-goal drops immediately; total drop penalties cap at 4.
        far = _proprio(left=WaypointRewardTests._wrist(start_home + [0.0, -0.15, 0.0]))
        for expected in (2, 3, 4, 4):
            self.positions[label] = start_home + [0.0, 0.0, 0.07]
            self.lift(label, self.positions[label])
            self.positions[label] = start_home.copy()
            self.reward.step(far, _state(self.positions, measured=(1.0, 1.0)))
            self.assertEqual(self.reward.penalty_counts["drop"], expected)
        self.assertAlmostEqual(self.reward.penalty_return, -4 * drop_pen)

    def test_empty_place_penalty_for_closed_or_opening_jaw_with_exemptions_and_hysteresis(self):
        empty_pen = DEFAULTS["empty_place_penalty"]
        in_zone = _proprio(left=WaypointRewardTests._wrist(self.base_goal + [0.0, 0.0, 0.02]))
        outside = _proprio(left=WaypointRewardTests._wrist(self.base_goal + [0.0, -0.20, 0.15]))

        # Open empty gripper hovering in placement zone is NOT penalized.
        self.reward.step(in_zone, _state(self.positions, measured=(1.0, 1.0)))
        self.assertEqual(self.reward.penalty_counts["empty_place"], 0)

        # Closed empty gripper in placement zone fires once and latches until exit + margin.
        terms1, bonus1 = self.reward.step(in_zone, _state(self.positions, measured=(0.0, 1.0)))
        self.assertEqual(terms1["penalty_counts"]["empty_place"], 1)
        self.assertAlmostEqual(terms1["penalty_return"], -empty_pen)
        self.assertLess(bonus1, 0.0)
        # Staying inside does not re-fire.
        self.reward.step(in_zone, _state(self.positions, measured=(0.0, 1.0)))
        self.assertEqual(self.reward.penalty_counts["empty_place"], 1)

        # Exiting past margin resets latch; entering at 0.70 then opening to 0.85 also triggers!
        self.reward.step(outside, _state(self.positions, measured=(0.70, 1.0)))
        self.reward.step(in_zone, _state(self.positions, measured=(0.70, 1.0)))
        self.assertEqual(self.reward.penalty_counts["empty_place"], 1)
        self.reward.step(in_zone, _state(self.positions, measured=(0.85, 1.0)))
        self.assertEqual(self.reward.penalty_counts["empty_place"], 2)

        # Reaching an unplaced block that sits inside the placement zone is exempt.
        self.reward.step(outside, _state(self.positions, measured=(1.0, 1.0)))
        home = WaypointRewardTests._positions(self.reset)
        self.positions[BLOCK_LABELS[0]] = self.base_goal + [0.0, 0.0, 0.002]
        self.reward.step(in_zone, _state(self.positions, measured=(0.0, 1.0)))
        self.assertEqual(self.reward.penalty_counts["empty_place"], 2)

        # Restore block outside zone and verify cap at 4.
        self.positions[BLOCK_LABELS[0]] = home[BLOCK_LABELS[0]].copy()
        for expected in (3, 4, 4):
            self.reward.step(outside, _state(self.positions, measured=(1.0, 1.0)))
            self.reward.step(in_zone, _state(self.positions, measured=(0.0, 1.0)))
            self.assertEqual(self.reward.penalty_counts["empty_place"], expected)
        self.assertAlmostEqual(self.reward.penalty_return, -4 * empty_pen)

    def test_empty_tower_intrusion_and_tower_break_with_grace_and_nudge_hysteresis(self):
        base, upper, _ = BLOCK_LABELS
        intrusion_pen = DEFAULTS["empty_tower_intrusion_penalty"]
        break_pen = DEFAULTS["tower_break_penalty"]
        # Place and settle base block cleanly with arm 0.
        self.lift(base, self.positions[base] + [0.0, 0.0, 0.07], arm=0)
        self.step_at(base, self.base_goal, arm=0)
        self.step_at(base, self.base_goal, opening=0.8, arm=0)
        for _ in range(3):
            self.step_at(base, self.base_goal, opening=1.0, arm=0)
        self.assertIsNotNone(self.reward.milestone_steps["base"])

        # During 40-step post-release grace, arm 0 lingering at tower is exempt.
        at_tower = _proprio(left=WaypointRewardTests._wrist(self.base_goal + [0.0, 0.0, 0.005]))
        self.reward.step(at_tower, _state(self.positions, measured=(0.0, 1.0)))
        self.assertEqual(self.reward.penalty_counts["empty_tower_intrusion"], 0)
        self.assertEqual(self.reward.penalty_counts["empty_place"], 0)

        # Idle arm 1 (never carried) entering tower with closed jaw fires intrusion once,
        # NOT double-counted with empty_place on the same frame.
        arm1_in_tower = _proprio(right=WaypointRewardTests._wrist(self.base_goal + [0.0, 0.0, 0.005]))
        terms_intr, _ = self.reward.step(arm1_in_tower, _state(self.positions, measured=(1.0, 0.0)))
        self.assertEqual(terms_intr["penalty_counts"]["empty_tower_intrusion"], 1)
        self.assertEqual(terms_intr["penalty_counts"]["empty_place"], 0)
        self.assertAlmostEqual(terms_intr["penalty_return"], -intrusion_pen)

        # Minor 2 mm base nudge across 4 cm circle boundary (to 4.2 cm) does NOT fire tower_break.
        self.positions[base] = self.base_goal + [0.042, 0.0, 0.0]
        for _ in range(3):
            self.reward.step(_proprio(), _state(self.positions, measured=(1.0, 1.0)))
        self.assertEqual(self.reward.penalty_counts["tower_break"], 0)

        # Restore base inside circle, let arm 1 re-enter tower (2nd intrusion),
        # then knock base over for 2 frames: 1st frame alone does not trigger;
        # 2nd frame triggers tower_break + empty_tower_break attribution.
        self.positions[base] = self.base_goal.copy()
        self.reward.step(arm1_in_tower, _state(self.positions, measured=(1.0, 1.0)))
        self.assertEqual(self.reward.penalty_counts["empty_tower_intrusion"], 2)
        tilted = [math.cos(math.radians(22.5)), math.sin(math.radians(22.5)), 0.0, 0.0]
        tilted_rots = {label: [1.0, 0.0, 0.0, 0.0] for label in BLOCK_LABELS}
        tilted_rots[base] = tilted
        self.reward.step(_proprio(), _state(self.positions, rotations=tilted_rots))
        self.assertEqual(self.reward.penalty_counts["tower_break"], 0)
        terms_brk, _ = self.reward.step(_proprio(), _state(self.positions, rotations=tilted_rots))
        self.assertEqual(terms_brk["penalty_counts"]["tower_break"], 1)
        self.assertEqual(terms_brk["penalty_counts"]["empty_tower_break"], 1)
        self.assertAlmostEqual(terms_brk["penalty_return"], -(2 * intrusion_pen + break_pen))

    def test_empty_grasp_penalty_requires_rearm_exempts_engaging_and_caps(self):
        pen = DEFAULTS["empty_grasp_penalty"]
        midair = _proprio(left=WaypointRewardTests._wrist(np.array([-0.35, -0.35, TOP + 0.15])))
        self.reward.step(midair, _state(self.positions, measured=(1.0, 1.0)))
        # Midair empty pinch fires once and requires rearm >= 0.75 before firing again.
        terms1, bonus1 = self.reward.step(midair, _state(self.positions, measured=(0.15, 1.0)))
        self.assertEqual(terms1["penalty_counts"]["empty_grasp"], 1)
        self.assertAlmostEqual(terms1["penalty_return"], -pen)
        self.assertAlmostEqual(terms1["event_bonus"], -pen)
        self.assertAlmostEqual(bonus1, -pen)
        # Staying closed or opening only to 0.60 does not rearm.
        self.reward.step(midair, _state(self.positions, measured=(0.60, 1.0)))
        self.reward.step(midair, _state(self.positions, measured=(0.15, 1.0)))
        self.assertEqual(self.reward.penalty_counts["empty_grasp"], 1)
        # Rearming >= 0.75 near an unplaced block is exempt (engaging).
        near_block = _proprio(left=WaypointRewardTests._wrist(self.positions[BLOCK_LABELS[0]]))
        self.reward.step(near_block, _state(self.positions, measured=(0.80, 1.0)))
        self.reward.step(near_block, _state(self.positions, measured=(0.15, 1.0)))
        self.assertEqual(self.reward.penalty_counts["empty_grasp"], 1)
        # Episode cap at 4.
        for expected in (2, 3, 4, 4):
            self.reward.step(midair, _state(self.positions, measured=(0.80, 1.0)))
            self.reward.step(midair, _state(self.positions, measured=(0.15, 1.0)))
            self.assertEqual(self.reward.penalty_counts["empty_grasp"], expected)
        self.assertAlmostEqual(self.reward.penalty_return, -4 * pen)

    def test_arm_conflict_penalty_on_collision_or_same_block_hold_allows_parallel(self):
        pen = DEFAULTS["arm_conflict_penalty"]
        # Productive dual-arm parallel work on different blocks far apart: zero penalty.
        p0 = self.positions[BLOCK_LABELS[0]]
        p2 = self.positions[BLOCK_LABELS[2]]
        parallel = _proprio(
            left=WaypointRewardTests._wrist(p0),
            right=WaypointRewardTests._wrist(p2),
        )
        self.reward.step(parallel, _state(self.positions, measured=(REAL_GRIP_30, REAL_GRIP_30)))
        self.assertEqual(self.reward.penalty_counts["arm_conflict"], 0)
        self.assertEqual(self.reward.penalty_return, 0.0)

        # Two grippers colliding within 60 mm fire once and latch until > 100 mm apart.
        center = np.array([0.0, -0.35, TOP + 0.12])
        collide = _proprio(
            left=WaypointRewardTests._wrist(center + [-0.02, 0.0, 0.0]),
            right=WaypointRewardTests._wrist(center + [0.02, 0.0, 0.0]),
        )
        terms1, bonus1 = self.reward.step(collide, _state(self.positions, measured=(1.0, 1.0)))
        self.assertEqual(terms1["penalty_counts"]["arm_conflict"], 1)
        self.assertAlmostEqual(terms1["penalty_return"], -pen)
        self.assertAlmostEqual(terms1["event_bonus"], -pen)
        self.assertLess(bonus1, 0.0)
        # At 80 mm (still inside 60+40=100 mm hysteresis exit), stays latched.
        mid_sep = _proprio(
            left=WaypointRewardTests._wrist(center + [-0.04, 0.0, 0.0]),
            right=WaypointRewardTests._wrist(center + [0.04, 0.0, 0.0]),
        )
        self.reward.step(mid_sep, _state(self.positions, measured=(1.0, 1.0)))
        self.reward.step(collide, _state(self.positions, measured=(1.0, 1.0)))
        self.assertEqual(self.reward.penalty_counts["arm_conflict"], 1)

        # Separating past 100 mm clears latch; contesting same block triggers conflict and caps at 4.
        far_sep = _proprio(
            left=WaypointRewardTests._wrist(center + [-0.08, 0.0, 0.0]),
            right=WaypointRewardTests._wrist(center + [0.08, 0.0, 0.0]),
        )
        for expected in (2, 3, 4, 4):
            self.reward.step(far_sep, _state(self.positions, measured=(1.0, 1.0)))
            self.reward.step(collide, _state(self.positions, measured=(1.0, 1.0)))
            self.assertEqual(self.reward.penalty_counts["arm_conflict"], expected)
        self.assertAlmostEqual(self.reward.penalty_return, -4 * pen)

    def test_negative_penalty_weights_rejected(self):
        for key in ("empty_grasp_penalty", "drop_penalty", "empty_place_penalty",
                    "empty_tower_intrusion_penalty", "tower_break_penalty",
                    "arm_conflict_penalty"):
            with self.assertRaises(ValueError):
                validate_reward_config(_cfg(**{key: -0.1}))


if __name__ == "__main__":
    unittest.main()
