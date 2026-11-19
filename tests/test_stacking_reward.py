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


if __name__ == "__main__":
    unittest.main()
