"""Wave collection and residual PPO integration without Isaac or real scores."""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from starter.config import load_config
from starter.interfaces import ACTION_DIM, BLOCK_LABELS, PARTIAL_SCORE, SUCCESS_SCORE
from starter.policies import act_online
from starter.policies.act_inference import FEATURE_DIM, load_act_runner
from starter.policies.act_model import ACTPolicy
from starter.policies.act_online import collect_wave
from tests.test_act import small_config

ROOT = Path(__file__).resolve().parents[1]


def _shaping_cfg(**overrides) -> dict:
    """Default staged shaping config (configs/act_online.yaml values)."""
    cfg = {key: float(value) for key, value in load_config(
        ROOT / "configs/act_online.yaml").items() if key in act_online.SHAPING_KEYS}
    cfg.update(overrides)
    return cfg


def _zero_shaping_cfg() -> dict:
    """All stage weights zero: shaping contributes nothing to rewards."""
    return _shaping_cfg(reach_weight=0.0, lift_weight=0.0, pair_weight=0.0,
                        release_weight=0.0)


def _test_proprio() -> np.ndarray:
    """linspace fixture with valid identity wxyz quats and open grippers."""
    proprio = np.linspace(-0.2, 0.3, ACTION_DIM, dtype=np.float32)
    proprio[[3, 11]] = 1.0
    proprio[[4, 5, 6, 12, 13, 14]] = 0.0
    proprio[[7, 15]] = 1.0
    return proprio


def _observation(scene: int, step: int) -> dict:
    shade = np.array([(scene * 37 + step * 11) % 256, 64, 128], dtype=np.uint8)
    return {
        "image": np.broadcast_to(shade, (8, 8, 3)).copy(),
        "proprio": _test_proprio(),
    }


def _privileged_state(z: float = 0.1) -> dict:
    return {"blocks": {label: {"pos": [0.0, 0.0, z], "rot": [1.0, 0.0, 0.0, 0.0]}
                       for label in BLOCK_LABELS},
            "gripper_open_measured": [1.0, 1.0],
            "block_velocities": {label: {"linear": [0.0, 0.0, 0.0],
                                         "angular": [0.0, 0.0, 0.0]}
                                 for label in BLOCK_LABELS}}


class _ScriptedRunner:
    """Deterministic fake of FrozenACTRunner for collect_wave tests."""

    def __init__(self):
        self.stats = {"action_std": np.full(ACTION_DIM, 0.1, dtype=np.float32)}
        self.reset_calls = 0
        self.calls = []

    def reset(self):
        self.reset_calls += 1

    def predict_base(self, obs_list, env_ids, with_features=False):
        self.calls.append([int(obs["scene"]) for obs in obs_list])
        features = torch.tensor(
            [[float(obs["scene"]), float(obs["step"]), 1.0] for obs in obs_list],
            dtype=torch.float32,
        )
        chunks = []
        for obs in obs_list:
            chunk = np.zeros((1, ACTION_DIM), dtype=np.float32)
            chunk[0, :3] = 0.01 * float(obs["scene"])
            chunk[0, (3, 11)] = 1.0
            chunk[0, (7, 15)] = 0.5
            chunks.append(chunk)
        return chunks, features if with_features else None


class _RecordingActor:
    """Residual actor whose value is the feature sum and whose actions are raw."""

    def __init__(self):
        self.calls = []

    def act(self, features, deterministic=False):
        self.calls.append(
            {
                "features": features.detach().clone(),
                "deterministic": deterministic,
                "values": features.sum(-1).detach().clone(),
            }
        )
        raw = torch.zeros((features.shape[0], ACTION_DIM), dtype=features.dtype)
        log_probs = torch.zeros(features.shape[0], dtype=features.dtype)
        return raw, log_probs, features.sum(-1)


class _ScriptedEnv:
    """One episode per scene with a public score recorded per step."""

    def __init__(self, scripts, scores):
        self.scripts = scripts
        self.scores = scores
        self.reset_calls = []
        self.step_calls = []
        self._steps = {}
        self._privileged = []

    def reset_batch(self, seeds):
        self.reset_calls.append([int(seed) for seed in seeds])
        self._steps = {scene: 0 for scene in range(len(seeds))}
        self._privileged = [_privileged_state() for _ in seeds]
        return [self._observation(scene, 0) for scene in range(len(seeds))], [
            {} for _ in seeds
        ]

    @staticmethod
    def _observation(scene, step):
        return {"scene": int(scene), "step": int(step), "proprio": _test_proprio()}

    def step_batch(self, actions, active):
        self.step_calls.append([int(scene) for scene in active])
        if len(actions) != len(active):
            raise AssertionError("actions and active scenes must align")
        self._privileged = [_privileged_state() for _ in active]
        results = []
        for scene in active:
            self._steps[scene] += 1
            step = self._steps[scene]
            script = self.scripts[scene]
            score = float(self.scores[scene][step - 1])
            ended = step >= script["length"]
            terminated = bool(ended and script["terminated"])
            truncated = bool(ended and not script["terminated"])
            results.append(
                (self._observation(scene, step), score, terminated, truncated, {})
            )
        return results


class CollectWaveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_active_subsets_preserve_order_and_never_step_ended_scenes(self):
        scripts = {
            0: {"length": 3, "terminated": True},
            1: {"length": 1, "terminated": False},
            2: {"length": 4, "terminated": False},
            3: {"length": 2, "terminated": True},
        }
        scores = {0: [0.0, 10.0, 100.0], 1: [0.0], 2: [0.0, 5.0, 15.0, 15.0],
                  3: [0.0, 100.0]}
        env = _ScriptedEnv(scripts, scores)
        runner = _ScriptedRunner()
        actor = _RecordingActor()
        scores_before = copy.deepcopy(env.scores)

        batch, episodes = collect_wave(
            env, runner, actor, [0, 1, 2, 3], gamma=0.9, gae_lambda=0.8,
            residual_scale=0.05, shaping_cfg=_zero_shaping_cfg(),
        )

        self.assertEqual(env.reset_calls, [[0, 1, 2, 3]])
        self.assertEqual(env.step_calls, [[0, 1, 2, 3], [0, 2, 3], [0, 2], [2]])
        self.assertEqual(runner.reset_calls, 1)
        # Inactive scenes appear only in the step where they ended.
        self.assertNotIn(1, env.step_calls[1])
        self.assertNotIn(3, env.step_calls[2])
        self.assertNotIn(0, env.step_calls[3])
        # Bootstrap forwards include truncated frames but skip true terminals:
        # scene 1 bootstraps at step 1, scene 3 skips at step 2, scene 0 skips
        # at step 3, and scene 2 bootstraps once more when truncated.
        self.assertEqual(
            runner.calls, [[0, 1, 2, 3], [0, 1, 2, 3], [0, 2], [2], [2]]
        )

        self.assertEqual(len(batch.features), 3 + 1 + 4 + 2)
        scene_column = [int(value) for value in batch.features[:, 0]]
        self.assertEqual(scene_column, [0, 0, 0, 1, 2, 2, 2, 2, 3, 3])
        self.assertEqual(len(batch.actions), len(batch.features))
        for field in ("log_probs", "values", "advantages", "returns"):
            self.assertEqual(len(getattr(batch, field)), len(batch.features))

        self.assertEqual([episode["layout_id"] for episode in episodes], [1, 3, 0, 2])
        self.assertEqual([episode["length"] for episode in episodes], [1, 2, 3, 4])
        self.assertEqual([episode["final_score"] for episode in episodes],
                         [0.0, 100.0, 100.0, 15.0])
        self.assertEqual([episode["terminated"] for episode in episodes],
                         [False, True, True, False])
        self.assertEqual([episode["truncated"] for episode in episodes],
                         [True, False, False, True])
        self.assertEqual(env.scores, scores_before)

    def test_episode_diagnostics_preserve_score_drops_and_discounted_rewards(self):
        env = _ScriptedEnv({0: {"length": 4, "terminated": True}},
                           {0: [PARTIAL_SCORE, PARTIAL_SCORE, 0.0, SUCCESS_SCORE]})
        batch, episodes = collect_wave(
            env, _ScriptedRunner(), _RecordingActor(), [42], gamma=0.9,
            gae_lambda=0.8, residual_scale=0.05, shaping_cfg=_zero_shaping_cfg(),
        )
        episode = episodes[0]
        self.assertEqual(episode["episode_return"], SUCCESS_SCORE)
        self.assertAlmostEqual(episode["discounted_return"],
                               PARTIAL_SCORE - 0.9 ** 2 * PARTIAL_SCORE + 0.9 ** 3 * SUCCESS_SCORE)
        self.assertEqual(episode["nonzero_reward_steps"], 3)
        self.assertEqual(episode["negative_reward_steps"], 1)
        self.assertEqual(episode["first_partial_step"], 1)
        self.assertEqual(episode["position_residual_mean_m"], 0.0)
        # Raw clipping and normalized correction diagnostics use the collected
        # Gaussian samples, not the executed absolute targets.
        batch.actions[0, 0] = 2.0
        diagnostics = act_online._rollout_metrics(batch, episodes, 0.05)
        self.assertEqual(diagnostics["success_count"], 1)
        self.assertEqual(diagnostics["partial_rate"], 1.0)
        self.assertEqual(diagnostics["rollout_steps"], 4)
        self.assertEqual(diagnostics["residual_clip_fraction"], 1 / (4 * ACTION_DIM))
        self.assertAlmostEqual(diagnostics["normalized_residual_abs_mean"],
                               0.05 / (4 * ACTION_DIM))
        self.assertAlmostEqual(diagnostics["advantage_std"],
                               float(batch.advantages.std(unbiased=False)))

    def test_episode_diagnostics_log_potential_components_and_pair_geometry(self):
        # Static fixture: blocks at [0,0,0.1]; identity-quat proprio, grippers
        # open. Grasp point = wrist + [0.143,0,0]; d_min identical at both steps.
        cfg = _shaping_cfg()
        env = _ScriptedEnv({0: {"length": 2, "terminated": False}}, {0: [0.0, 0.0]})
        _, episodes = collect_wave(
            env, _ScriptedRunner(), _RecordingActor(), [14], gamma=0.9, gae_lambda=0.8,
            residual_scale=0.05, shaping_cfg=cfg,
        )
        ep = episodes[0]
        proprio = _test_proprio().astype(np.float64)
        grasp = proprio[8:11] + np.array([0.143, 0.0, 0.0])
        d_min = float(np.linalg.norm(grasp - np.array([0.0, 0.0, 0.1])))
        phi_static = -2.0 * math.tanh(d_min / 0.35) - 3.0 - 4.0
        for field in ("phi_initial", "phi_final", "phi_mean"):
            self.assertAlmostEqual(ep[field], phi_static, places=5)
        self.assertAlmostEqual(ep["d_min_initial_m"], d_min, places=6)
        self.assertAlmostEqual(ep["d_min_final_m"], d_min, places=6)
        self.assertEqual(ep["lift_initial_m"], 0.0)
        self.assertEqual(ep["lift_final_m"], 0.0)
        self.assertEqual(ep["pair_sum_final"], 0.0)
        self.assertEqual(ep["reach_gate_final"], 1.0)
        self.assertAlmostEqual(ep["phi_lift_final"], -3.0, places=6)
        self.assertAlmostEqual(ep["phi_pair_final"], -4.0, places=6)
        self.assertEqual(ep["phi_release_final"], 0.0)
        self.assertEqual(ep["min_gripper_final"], 1.0)
        self.assertEqual(ep["settled_pair_final"], 0.0)
        self.assertEqual(ep["lift_progress_final"], 0.0)
        self.assertEqual(ep["final_gripper_open_measured"], [1.0, 1.0])
        self.assertEqual(ep["final_pair_xy_m"], [0.0, 0.0, 0.0])
        self.assertEqual(ep["final_pair_dz_m"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(ep["final_gripper_openness"], [1.0, 1.0], atol=1e-6)

        # Lifted final frame: all three blocks rise 0.05 m (fixture lifts every
        # block); per-block gates sum to 2 (capped), phi_lift=0; pair dz stays 0.
        # Truncation keeps the final potential instead of zeroing it.
        env = _ScriptedEnv({0: {"length": 2, "terminated": False}}, {0: [0.0, 0.0]})
        import sys
        with mock.patch.object(sys.modules[__name__], "_privileged_state",
                               side_effect=[_privileged_state(0.1), _privileged_state(0.1),
                                            _privileged_state(0.15)]):
            _, episodes = collect_wave(
                env, _ScriptedRunner(), _RecordingActor(), [14], gamma=0.9, gae_lambda=0.8,
                residual_scale=0.05, shaping_cfg=cfg,
            )
        ep = episodes[0]
        self.assertAlmostEqual(ep["lift_initial_m"], 0.0, places=6)
        self.assertAlmostEqual(ep["lift_final_m"], 0.05, places=6)
        self.assertAlmostEqual(ep["lift_gate_final"], 1.0, places=6)
        self.assertAlmostEqual(ep["phi_initial"], phi_static, places=5)
        # All blocks move 0.05 up: d_min shrinks (grasp point z 0.1 vs 0.15).
        d_lift = float(np.linalg.norm(grasp - np.array([0.0, 0.0, 0.15])))
        phi_lifted = -2.0 * math.tanh(d_lift / 0.35) + 0.0 - 4.0
        self.assertAlmostEqual(ep["phi_final"], phi_lifted, places=5)
        self.assertAlmostEqual(ep["phi_mean"], (phi_static + phi_lifted) / 2, places=5)
        # Two truncation steps: (g*phi1 - phi0) + (g*phi2 - phi1).
        self.assertAlmostEqual(ep["shaping_return"],
                               0.9 * phi_static - phi_static
                               + 0.9 * phi_lifted - phi_static, places=5)

    def test_truncation_bootstraps_from_terminal_observation(self):
        scripts = {
            0: {"length": 1, "terminated": True},
            1: {"length": 1, "terminated": False},
        }
        scores = {0: [SUCCESS_SCORE], 1: [PARTIAL_SCORE]}
        env = _ScriptedEnv(scripts, scores)
        runner = _ScriptedRunner()
        actor = _RecordingActor()

        batch, episodes = collect_wave(
            env, runner, actor, [10, 11], gamma=0.9, gae_lambda=0.8,
            residual_scale=0.05, shaping_cfg=_zero_shaping_cfg(),
        )

        # Terminated scene 0: next value is zero. Truncated scene 1: the value
        # of its terminal observation (feature sum 1 + 1 + 1) is bootstrapped.
        expected_returns = torch.tensor([SUCCESS_SCORE, PARTIAL_SCORE + 0.9 * 3.0])
        expected_advantages = torch.tensor(
            [SUCCESS_SCORE - 1.0, PARTIAL_SCORE + 0.9 * 3.0 - 2.0]
        )
        torch.testing.assert_close(batch.returns, expected_returns)
        torch.testing.assert_close(batch.advantages, expected_advantages)
        self.assertEqual(len(batch.features), 2)

        # Exactly one bootstrap forward, using only the truncated terminal obs.
        self.assertEqual(runner.calls, [[0, 1], [1]])
        self.assertEqual(len(actor.calls), 2)
        self.assertFalse(actor.calls[0]["deterministic"])
        self.assertTrue(actor.calls[1]["deterministic"])
        torch.testing.assert_close(
            actor.calls[1]["features"], torch.tensor([[1.0, 1.0, 1.0]])
        )
        self.assertEqual([episode["layout_id"] for episode in episodes], [10, 11])
        self.assertEqual([episode["terminated"] for episode in episodes], [True, False])


class _ShapingEnv:
    """One two-step episode where the arm aligns horizontally and lifts block_0."""

    def __init__(self):
        self._step = 0
        self._privileged = []

    def reset_batch(self, seeds):
        self._step = 0
        self._privileged = [_privileged_state() for _ in seeds]
        return [self._observation(0.05)], [{}]

    def _observation(self, end_effector_x):
        proprio = np.zeros(ACTION_DIM, dtype=np.float32)
        proprio[[0, 8]] = end_effector_x
        proprio[[2, 10]] = 0.24
        proprio[[3, 11]] = 1.0  # identity wxyz
        proprio[[7, 15]] = 1.0  # open grippers
        return {"scene": 0, "step": self._step, "proprio": proprio}

    def step_batch(self, actions, active):
        self._step += 1
        if self._step == 1:
            return [(self._observation(0.02), 0.0, False, False, {})]
        self._privileged[0]["blocks"]["block_0"]["pos"][2] = 0.15
        return [(self._observation(0.02), 0.0, True, False, {})]


class ShapingTest(unittest.TestCase):
    """Hand-computed staged potential checks (round14 spec values)."""

    def _terms(self, positions, proprio, heights=None, physical=None, **overrides):
        if heights is None:
            heights = {label: float(pos[2]) for label, pos in positions.items()}
        if physical is None:
            physical = _privileged_state()
        return act_online._shaping_terms(positions, proprio, heights,
                                         _shaping_cfg(**overrides), physical)

    def _down_proprio(self, wrist):
        """Both arms at wrist, tool axis pointing down (R@[1,0,0]=[0,0,-1])."""
        proprio = np.zeros(ACTION_DIM, dtype=np.float64)
        for start in (0, 8):
            proprio[start:start + 3] = wrist
            # wxyz quat rotating +x to -z: 90 deg about y -> [cos45,0,sin45,0]
            proprio[start + 3:start + 7] = [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]
        proprio[[7, 15]] = 1.0
        return proprio

    def test_grasp_point_uses_quaternion_not_wrist_xy(self):
        offset = 0.143
        # Identity quat: grasp point = wrist + [offset,0,0].
        points = act_online._grasp_points(_test_proprio(), offset)
        np.testing.assert_allclose(points[0], _test_proprio()[:3] + [offset, 0, 0], atol=1e-6)
        # Demo-like downward tool axis: grasp point directly below wrist.
        proprio = self._down_proprio([0.0, 0.0, 0.243])
        points = act_online._grasp_points(proprio, offset)
        np.testing.assert_allclose(points, [[0, 0, 0.1], [0, 0, 0.1]], atol=1e-6)
        with self.assertRaises(ValueError):
            act_online._grasp_points(np.zeros(ACTION_DIM), offset)

    def test_reach_term_rewards_grasp_point_approach_in_3d(self):
        positions = {"block_0": np.array([0.0, 0.0, 0.1]),
                     "block_1": np.array([0.45, 0.0, 0.1]),
                     "block_2": np.array([0.0, 0.45, 0.1])}
        heights = {label: 0.1 for label in positions}
        far = self._down_proprio([0.0, 0.0, 0.493])  # grasp point z 0.35, d 0.25
        near = self._down_proprio([0.0, 0.0, 0.393])  # grasp point z 0.25, d 0.15
        phi_far = self._terms(positions, far, heights)["phi"]
        phi_near = self._terms(positions, near, heights)["phi"]
        self.assertGreater(phi_near, phi_far)
        # Exact reach value at d=0.15, no lift/pair/release: -2*tanh(0.15/0.35) - 1.5 - 4.
        expected = -2.0 * math.tanh(0.15 / 0.35) - 3.0 - 4.0
        self.assertAlmostEqual(phi_near, expected, places=6)
        # Grasp point on block centre: reach contributes 0.
        on = self._down_proprio([0.0, 0.0, 0.243])
        self.assertAlmostEqual(self._terms(positions, on, heights)["phi_reach"], 0.0, places=6)

    def test_lift_gate_ramp_plateau_and_decay(self):
        positions = {"block_0": np.array([0.0, 0.0, 0.1]),
                     "block_1": np.array([0.45, 0.0, 0.1]),
                     "block_2": np.array([0.0, 0.45, 0.1])}
        heights = {label: 0.1 for label in positions}
        proprio = self._down_proprio([0.0, 0.0, 0.243])
        cases = {0.0: -3.0, 0.0075: -2.25, 0.015: -1.5, 0.100: -1.5,
                 0.140: -1.5, 0.160: -2.25, 0.180: -3.0}
        for lift, expected in cases.items():
            moved = {label: pos.copy() for label, pos in positions.items()}
            moved["block_0"][2] += lift
            terms = self._terms(moved, proprio, heights)
            self.assertAlmostEqual(terms["phi_lift"], expected, places=6,
                                   msg=f"lift={lift}")
        # Anti-tip: max tip-only centre rise 8.3mm earns at most 0.55 gate.
        moved = {label: pos.copy() for label, pos in positions.items()}
        moved["block_0"][2] += 0.0083
        self.assertLessEqual(self._terms(moved, proprio, heights)["lift_gate"], 0.28)
        moved["block_2"][2] += 0.015
        terms = self._terms(moved, proprio, heights)
        self.assertAlmostEqual(terms["lift_progress"], 1.0 + 0.0083 / 0.015)
        self.assertGreater(terms["phi_lift"], -1.5)

    def test_pair_window_matches_official_predicate_and_resting_exclusion(self):
        # Resting mixed-size blocks (|dz|=0.0049) must NOT register a pair.
        resting = {"block_0": np.array([0.0, 0.0, 0.7803]),
                   "block_1": np.array([0.0, 0.0, 0.7852]),
                   "block_2": np.array([0.45, 0.0, 0.78])}
        proprio = self._down_proprio([0.0, 0.0, 0.92])
        self.assertEqual(self._terms(resting, proprio)["pair_sum"], 0.0)
        # One adjacent layer: xy=0.01 (inside official 0.015), dz=0.03 (block h).
        stacked = {"block_0": np.array([0.0, 0.0, 0.78]),
                   "block_1": np.array([0.01, 0.0, 0.81]),
                   "block_2": np.array([0.45, 0.0, 0.78])}
        terms = self._terms(stacked, proprio)
        align = min((0.4 - 0.01) / (0.4 - 0.012), 1.0)
        self.assertAlmostEqual(terms["pair_sum"], align, places=6)
        self.assertAlmostEqual(terms["phi_pair"], 2.0 * (align - 2.0), places=6)
        # Two layers up (dz=0.06 > 0.058) gets no direct pair credit.
        two_up = {"block_0": np.array([0.0, 0.0, 0.78]),
                  "block_1": np.array([0.45, 0.0, 0.78]),
                  "block_2": np.array([0.0, 0.0, 0.84])}
        self.assertEqual(self._terms(two_up, proprio)["pair_sum"], 0.0)
        # Full 3-stack at xy=0: S capped at 2, reach gate 0, phi_pair 0.
        full = {"block_0": np.array([0.0, 0.0, 0.78]),
                "block_1": np.array([0.0, 0.0, 0.81]),
                "block_2": np.array([0.0, 0.0, 0.84])}
        terms = self._terms(full, proprio)
        # (0,2) spans two layers (dz 0.06 > 0.058) -> 0; adjacent pairs sum to 2.
        self.assertAlmostEqual(terms["pair_sum"], 2.0, places=6)
        self.assertAlmostEqual(terms["phi_pair"], 0.0, places=6)
        self.assertEqual(terms["reach_gate"], 0.0)
        self.assertEqual(terms["phi_reach"], 0.0)
        # Label permutations, including negative label-order dz, preserve a tower.
        for order in ((2, 1, 0), (1, 0, 2)):
            relabelled = {f"block_{i}": full[f"block_{j}"].copy()
                          for i, j in enumerate(order)}
            self.assertAlmostEqual(self._terms(relabelled, proprio)["pair_sum"], 2.0)

    def test_pair_chain_does_not_count_shared_upper_block_twice(self):
        # Two blocks at table height with one above cannot be a three-layer tower.
        fork = {"block_0": np.array([0.0, 0.0, 0.78]),
                "block_1": np.array([0.0, 0.0, 0.81]),
                "block_2": np.array([0.03, 0.0, 0.78])}
        proprio = self._down_proprio([0.0, 0.0, 0.92])
        terms = self._terms(fork, proprio)
        self.assertAlmostEqual(terms["pair_sum"], 1.0)
        self.assertAlmostEqual(terms["phi_pair"], -2.0)
        # Also reject the inverted fork (two top blocks sharing one bottom).
        fork["block_2"][2] = 0.81
        self.assertAlmostEqual(self._terms(fork, proprio)["pair_sum"], 1.0)

    def test_release_only_after_pair_and_open_grippers_pay(self):
        stacked = {"block_0": np.array([0.0, 0.0, 0.78]),
                   "block_1": np.array([0.0, 0.0, 0.81]),
                   "block_2": np.array([0.45, 0.0, 0.78])}
        proprio = self._down_proprio([0.0, 0.0, 0.92])
        open_terms = self._terms(stacked, proprio)
        physical = _privileged_state()
        physical["gripper_open_measured"] = [0.3, 0.3]
        closed_terms = self._terms(stacked, proprio, physical=physical)
        self.assertAlmostEqual(open_terms["phi_release"], 0.0, places=6)
        self.assertAlmostEqual(closed_terms["phi_release"], -0.5 * 1.0 * (1.0 - 0.3 / 0.8), places=6)
        for openings in ([1.0, 0.2], [0.2, 1.0]):
            physical = _privileged_state()
            physical["gripper_open_measured"] = openings
            self.assertAlmostEqual(self._terms(stacked, proprio, physical=physical)["phi_release"], -0.375)
        # Measured release, not the command in proprio, determines credit.
        proprio[[7, 15]] = 0.0
        self.assertEqual(self._terms(stacked, proprio)["phi_release"], 0.0)
        physical = _privileged_state()
        physical["gripper_open_measured"] = [0.8, 0.8]
        self.assertEqual(self._terms(stacked, proprio, physical=physical)["phi_release"], 0.0)
        physical["gripper_open_measured"] = [0.0, 1.0]
        self.assertAlmostEqual(self._terms(stacked, proprio, physical=physical)["phi_release"], -0.5)
        # Moving blocks are not settled, even with the same positions.
        physical["block_velocities"]["block_0"]["linear"] = [0.01, 0.0, 0.0]
        self.assertEqual(self._terms(stacked, proprio, physical=physical)["phi_release"], 0.0)
        # Angular motion alone also kills settling.
        physical = _privileged_state()
        physical["block_velocities"]["block_1"]["angular"] = [0.1, 0.0, 0.0]
        self.assertEqual(self._terms(stacked, proprio, physical=physical)["phi_release"], 0.0)
        # No pair -> release exactly zero regardless of gripper state.
        apart = {"block_0": np.array([0.0, 0.0, 0.78]),
                 "block_1": np.array([0.45, 0.0, 0.78]),
                 "block_2": np.array([0.0, 0.45, 0.78])}
        self.assertEqual(self._terms(apart, proprio)["phi_release"], 0.0)

    def test_placed_exclusion_retargets_reach_to_next_block(self):
        stacked = {"block_0": np.array([0.0, 0.0, 0.78]),
                   "block_1": np.array([0.0, 0.0, 0.81]),
                   "block_2": np.array([0.2, 0.0, 0.78])}
        # Downward tool axis: grasp point [0,0,0.78] sits on block_0 centre.
        proprio = self._down_proprio([0.0, 0.0, 0.923])
        terms = self._terms(stacked, proprio)
        # Broad geometry alone cannot label the block 0.2 m away as placed.
        self.assertAlmostEqual(terms["d_min"], 0.2, places=6)
        settled = {"block_0": np.array([0.0, 0.0, 0.78]),
                   "block_1": np.array([0.01, 0.0, 0.81]),
                   "block_2": np.array([0.2, 0.0, 0.78])}
        terms = self._terms(settled, proprio)
        expected_align = min((0.4 - 0.01) / (0.4 - 0.012), 1.0)
        expected_tight = min((0.02 - 0.01) / (0.02 - 0.012), 1.0)
        expected = min(0.2, 0.9 * expected_align * expected_tight)
        self.assertAlmostEqual(terms["d_min"], expected, places=6)
        # Measured max tower->third distance (layout 21) must transit monotonically.
        far = {"block_0": np.array([0.0, 0.0, 0.78]),
               "block_1": np.array([0.0, 0.0, 0.81]),
               "block_2": np.array([0.8084, 0.0, 0.78])}
        values = []
        for t in np.linspace(0.0, 0.8084, 20):
            arm = self._down_proprio([t, 0.0, 0.923])
            values.append(self._terms(far, arm)["phi_reach"])
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(values, values[1:])))
        # Retarget requires measured release: closed jaws keep the tower nearest.
        physical = _privileged_state()
        physical["gripper_open_measured"] = [0.0, 0.0]
        self.assertAlmostEqual(self._terms(settled, proprio, physical=physical)["d_min"], 0.0, places=6)

    def test_potential_shaping_rewards_approach_and_lift(self):
        cfg = _shaping_cfg()
        env = _ShapingEnv()
        _, episodes = collect_wave(
            env, _ScriptedRunner(), _RecordingActor(), [0], gamma=0.9,
            gae_lambda=0.8, residual_scale=0.05, shaping_cfg=cfg,
        )
        # _ShapingEnv: identity quat, wrist [x,0,0.24] -> grasp point
        # [x+0.143,0,0.24]; blocks coincide at [0,0,0.1], block_0 rises to
        # 0.15 at the final step (creating dz=0.05 pairs with the other two).
        def phi(x, z0):
            pos = {"block_0": np.array([0.0, 0.0, z0]),
                   "block_1": np.array([0.0, 0.0, 0.1]),
                   "block_2": np.array([0.0, 0.0, 0.1])}
            proprio = np.zeros(ACTION_DIM)
            proprio[[0, 8]] = x
            proprio[[2, 10]] = 0.24
            proprio[[3, 11]] = 1.0
            proprio[[7, 15]] = 1.0
            heights = {"block_0": 0.1, "block_1": 0.1, "block_2": 0.1}
            return act_online._shaping_potential(pos, proprio, heights, cfg, _privileged_state())
        phi0, phi_mid, phi1 = phi(0.05, 0.1), phi(0.02, 0.1), phi(0.02, 0.15)
        # True termination: phi(terminal)=0 even with a lifted final frame.
        expected = (0.9 * phi_mid - phi0) + (0.0 - phi_mid)
        episode = episodes[0]
        self.assertEqual(episode["final_score"], 0.0)
        self.assertAlmostEqual(episode["shaping_return"], expected, places=5)
        self.assertAlmostEqual(episode["episode_return"], expected, places=5)

    def test_terminal_shaping_telescopes_but_truncation_keeps_potential(self):
        cfg = _shaping_cfg()
        _, terminal_episodes = collect_wave(
            _ShapingEnv(), _ScriptedRunner(), _RecordingActor(), [0], gamma=0.9,
            gae_lambda=0.8, residual_scale=0.05, shaping_cfg=cfg,
        )
        pos0 = {label: np.array([0.0, 0.0, 0.1]) for label in BLOCK_LABELS}
        proprio0 = np.zeros(ACTION_DIM)
        proprio0[[0, 8]] = 0.05
        proprio0[[2, 10]] = 0.24
        proprio0[[3, 11]] = 1.0
        proprio0[[7, 15]] = 1.0
        heights = {label: 0.1 for label in BLOCK_LABELS}
        phi0 = act_online._shaping_potential(pos0, proprio0, heights, cfg, _privileged_state())
        # Discounted shaping over a terminated trajectory = -phi(start).
        self.assertAlmostEqual(terminal_episodes[0]["discounted_return"], -phi0, places=5)

        truncated = _ShapingEnv()
        step = truncated.step_batch

        def truncate(actions, active):
            return [(obs, score, False, terminated, info)
                    for obs, score, terminated, _, info in step(actions, active)]

        truncated.step_batch = truncate
        _, episodes = collect_wave(
            truncated, _ScriptedRunner(), _RecordingActor(), [0], gamma=0.9,
            gae_lambda=0.8, residual_scale=0.05, shaping_cfg=cfg,
        )
        pos1 = {label: np.array([0.0, 0.0, 0.1]) for label in BLOCK_LABELS}
        pos1["block_0"] = np.array([0.0, 0.0, 0.15])
        proprio1 = proprio0.copy()
        proprio1[[0, 8]] = 0.02
        phi_final = act_online._shaping_potential(pos1, proprio1, heights, cfg, _privileged_state())
        self.assertAlmostEqual(episodes[0]["discounted_return"],
                               -phi0 + 0.9 ** 2 * phi_final, places=5)


class _OnlineFakeEnv:
    """Two-step episodes with public scores 0, 15 and 100 only."""

    SCRIPTS = {
        0: {"length": 2, "final": SUCCESS_SCORE, "terminated": True},
        1: {"length": 1, "final": PARTIAL_SCORE, "terminated": True},
        2: {"length": 2, "final": PARTIAL_SCORE, "terminated": False},
        3: {"length": 2, "final": 0.0, "terminated": False},
    }

    def __init__(self):
        self.reset_calls = []
        self.step_calls = []
        self.scores_seen = []
        self.closed = False
        self._state = {}
        self._privileged = []

    def reset_batch(self, seeds):
        self.reset_calls.append([int(seed) for seed in seeds])
        self._state = {}
        self._privileged = [_privileged_state() for _ in seeds]
        for scene, seed in enumerate(seeds):
            script = dict(self.SCRIPTS[int(seed) % len(self.SCRIPTS)])
            self._state[scene] = {"step": 0, "done": False, **script}
        return [_observation(scene, 0) for scene in range(len(seeds))], [
            {} for _ in seeds
        ]

    def step_batch(self, actions, active):
        self.step_calls.append([int(scene) for scene in active])
        if len(actions) != len(active):
            raise AssertionError("actions and active scenes must align")
        self._privileged = [_privileged_state() for _ in active]
        results = []
        for scene in active:
            state = self._state[scene]
            if state["done"]:
                raise AssertionError(f"inactive scene {scene} was stepped")
            state["step"] += 1
            state["done"] = state["step"] >= state["length"]
            score = float(state["final"]) if state["done"] else 0.0
            terminated = bool(state["done"] and state["terminated"])
            truncated = bool(state["done"] and not state["terminated"])
            self.scores_seen.append(score)
            results.append(
                (_observation(scene, state["step"]), score, terminated, truncated, {})
            )
        return results

    def close(self):
        self.closed = True


def _online_config(base_checkpoint: Path) -> dict:
    config = load_config(ROOT / "configs/env.yaml")
    config.update(load_config(ROOT / "configs/act_online.yaml"))
    config.update(
        base_checkpoint=str(base_checkpoint),
        device="cpu",
        finetune_updates=2,
        finetune_episodes=4,
        num_envs=2,
        hidden_dim=16,
        lr=1e-2,
        ppo_epochs=2,
        minibatch_size=2,
        target_kl=10.0,
        save_freq=1,
        max_train_seconds=3600.0,
        run_stamp="unit-test",
        seed=7,
        rollout_layout_ids=[0],
    )
    return config


class TrainOnlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _write_base(self, root: Path) -> tuple[Path, dict, dict]:
        config = small_config()
        policy = ACTPolicy(config, pretrained_backbone=False)
        base_state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        directory = root / "base"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        torch.save(policy.state_dict(), directory / "policy_last.ckpt")
        action_mean = np.zeros(ACTION_DIM, dtype=np.float32)
        action_mean[[3, 11]] = 1.0
        action_mean[[7, 15]] = 0.5
        stats = {
            "qpos_mean": np.zeros(ACTION_DIM, dtype=np.float32),
            "qpos_std": np.ones(ACTION_DIM, dtype=np.float32),
            "action_mean": action_mean,
            "action_std": np.full(ACTION_DIM, 0.1, dtype=np.float32),
        }
        with (directory / "dataset_stats.pkl").open("wb") as stream:
            pickle.dump(stats, stream)
        return directory / "policy_last.ckpt", config, base_state

    def test_rollout_layout_ids_require_unique_nonnegative_integers(self):
        for value in ([], [-1], [1.5], [True], ["0"], [0, 0], 0):
            with self.subTest(value=value):
                config = _online_config(Path("/unused/offline.ckpt"))
                config["rollout_layout_ids"] = value
                with self.assertRaises(ValueError):
                    act_online._validate_config(config)

    def test_max_train_seconds_is_required_and_positive(self):
        config = _online_config(Path("/unused/offline.ckpt"))
        del config["max_train_seconds"]
        with self.assertRaises(KeyError):
            act_online._validate_config(config)
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value):
                config = _online_config(Path("/unused/offline.ckpt"))
                config["max_train_seconds"] = value
                with self.assertRaises(ValueError):
                    act_online._validate_config(config)

    def test_shaping_weights_must_be_nonnegative_and_finite(self):
        for key in act_online.SHAPING_KEYS:
            for value in (-1, float("inf"), float("nan")):
                with self.subTest(key=key, value=value):
                    config = _online_config(Path("/unused/offline.ckpt"))
                    config[key] = value
                    with self.assertRaises(ValueError):
                        act_online._validate_config(config)

    def test_shaping_window_ordering_is_validated(self):
        for key, value in (("pair_align_saturate", 0.5), ("pair_height_low", 0.03),
                           ("pair_height_zero", 0.04), ("lift_ramp", 0.2),
                           ("lift_plateau", 0.01), ("pair_cap", 0.0), ("reach_scale", 0.0),
                           ("settle_linear_speed", 0.0), ("settle_angular_speed", 0.0),
                           ("placement_xy_tolerance", 0.012),
                           ("pair_height_full", 0.05)):
            with self.subTest(key=key, value=value):
                config = _online_config(Path("/unused/offline.ckpt"))
                config[key] = value
                with self.assertRaises(ValueError):
                    act_online._validate_config(config)

    def test_mean_policy_metrics_measure_learned_correction_without_sampling(self):
        model = act_online.ResidualActorCritic(feature_dim=3, hidden_dim=8)
        with torch.no_grad():
            # Both arms correct 3/4 cm before scaling, so the Euclidean norm is
            # 5 cm. Std and residual scale give exactly a 1 mm mean correction.
            model.mean_head.bias[[0, 8]] = 0.3
            model.mean_head.bias[[1, 9]] = 0.4
        metrics = act_online._mean_policy_metrics(
            model, torch.zeros(1030, 3), np.full(ACTION_DIM, 0.1), 0.02,
        )
        self.assertAlmostEqual(metrics["mean_policy_position_residual_m"], 0.001, places=8)

    def test_numpy_rng_payload_round_trips_plain_weights_only_data(self):
        np.random.seed(7)
        state = act_online._numpy_rng_payload()
        restored = act_online._restore_numpy_rng(state)
        np.random.set_state(restored)
        first = np.random.random(3)
        np.random.seed(7)
        second = np.random.random(3)
        np.testing.assert_array_equal(first, second)
        self.assertIsInstance(state["keys"], torch.Tensor)
        self.assertEqual(state["keys"].dtype, torch.uint32)

    def test_two_ppo_updates_freeze_base_and_save_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, base_config, base_state = self._write_base(root)
            output = root / "online"
            env = _OnlineFakeEnv()
            config = _online_config(checkpoint)
            real_residual = act_online.ResidualActorCritic
            captured = {}

            def build_residual(**kwargs):
                model = real_residual(**kwargs)
                captured["initial"] = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
                captured["model"] = model
                return model

            console = io.StringIO()
            with mock.patch.object(act_online, "ResidualActorCritic", side_effect=build_residual), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=env), \
                    mock.patch.object(act_online, "online_output_dir", return_value=output), \
                    mock.patch("sys.stdout", console):
                act_online.train_online(config)

            self.assertTrue(env.closed)
            # A singleton curriculum still selects the same layout for every scene.
            # Action sampling remains stochastic.
            self.assertEqual(env.reset_calls, [[0, 0]] * 4)
            self.assertEqual(env.step_calls, [[0, 1]] * 8)
            self.assertEqual(set(env.scores_seen), {0.0, SUCCESS_SCORE})

            records = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["update"] for record in records], [1, 2])
            self.assertEqual([record["global_step"] for record in records], [8, 16])
            self.assertEqual([record["episodes"] for record in records], [4, 8])
            self.assertAlmostEqual(records[0]["mean_final_score"], SUCCESS_SCORE)
            self.assertAlmostEqual(records[1]["mean_final_score"], SUCCESS_SCORE)
            self.assertAlmostEqual(records[0]["success_rate"], 1.0)
            self.assertAlmostEqual(records[1]["success_rate"], 1.0)
            with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                csv_records = list(csv.DictReader(stream))
            self.assertEqual(len(csv_records), 2)
            for record, row in zip(records, csv_records):
                for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                            "clip_fraction", "updates", "grad_norm", "total_loss",
                            "policy_std_mean", "advantage_std", "return_mean"):
                    self.assertTrue(np.isfinite(record[key]), (key, record))
                    self.assertAlmostEqual(float(row[key]), record[key])
                self.assertIsNone(record["vram"])
                for field in ("allocated_gib", "reserved_gib", "peak_allocated_gib",
                              "peak_reserved_gib", "total_gib"):
                    self.assertEqual(row[f"vram_{field}"], "")
                self.assertGreater(record["update_seconds"], 0)
                self.assertGreater(record["rollout_seconds"], 0)
                self.assertGreater(record["optimization_seconds"], 0)
                self.assertGreaterEqual(record["update_seconds"],
                                        record["rollout_seconds"] + record["optimization_seconds"])
                self.assertAlmostEqual(record["steps_per_second"] * record["update_seconds"],
                                       record["rollout_steps"])
                self.assertAlmostEqual(record["eta_seconds"],
                                       record["elapsed_seconds"] / record["update"] *
                                       (config["finetune_updates"] - record["update"]))
            self.assertEqual(records[-1]["eta_seconds"], 0.0)
            self.assertEqual([record["rollout_steps"] for record in records], [8, 8])
            self.assertEqual([record["success_count"] for record in records], [4, 4])
            self.assertEqual([record["partial_rate"] for record in records], [1.0, 1.0])
            self.assertEqual([record["terminated_count"] for record in records], [4, 4])
            self.assertEqual([record["truncated_count"] for record in records], [0, 0])
            self.assertIn("ETA=", console.getvalue())
            self.assertIn("rollout_layout_ids=[0]", console.getvalue())
            self.assertIn("VRAM=CPU", console.getvalue())
            self.assertIn("lr=", console.getvalue())
            self.assertEqual(console.getvalue().count("update_time="), 2)
            episode_records = [json.loads(line) for line in
                               (output / "episodes.jsonl").read_text().splitlines()]
            with (output / "episodes.csv").open(newline="", encoding="utf-8") as stream:
                episode_rows = list(csv.DictReader(stream))
            self.assertEqual(len(episode_records), 8)
            self.assertEqual([ep["episode"] for ep in episode_records], list(range(1, 9)))
            self.assertEqual([ep["update"] for ep in episode_records], [1, 1, 1, 1, 2, 2, 2, 2])
            # The config seed and the rollout layout id are distinct columns.
            self.assertEqual([ep["seed"] for ep in episode_records], [7] * 8)
            self.assertEqual([ep["layout_id"] for ep in episode_records], [0] * 8)
            self.assertEqual([row["seed"] for row in episode_rows], ["7"] * 8)
            self.assertEqual([row["layout_id"] for row in episode_rows], ["0"] * 8)
            for episode, row in zip(episode_records, episode_rows):
                self.assertAlmostEqual(episode["episode_return"],
                                       episode["final_score"] + episode["shaping_return"])
                self.assertAlmostEqual(float(row["discounted_return"]), episode["discounted_return"])
                self.assertGreaterEqual(episode["position_residual_mean_m"], 0.0)
            self.assertEqual(episode_records[-1]["first_partial_step"], 2)
            self.assertEqual(episode_rows[-1]["first_partial_step"], "2")
            snapshot = json.loads((output / "config.json").read_text())
            self.assertEqual(snapshot["rollout_layout_ids"], [0])
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertIn("excludes Isaac", metadata["vram_scope"])
            self.assertIn("not held-out", metadata["metrics_scope"])
            self.assertEqual(metadata["trainable_parameters"],
                             sum(p.numel() for p in captured["model"].parameters()
                                 if p.requires_grad))
            for name in ("training_curves.png", "training_curves.svg"):
                self.assertGreater((output / name).stat().st_size, 1000)

            payload = torch.load(output / "policy_last.ckpt", map_location="cpu",
                                 weights_only=True)
            self.assertEqual(payload["format"], "act_residual_ppo_v3")
            self.assertEqual(payload["update"], 2)
            self.assertEqual(payload["global_step"], 16)
            self.assertEqual(payload["base_config"], base_config)
            for key, value in base_state.items():
                torch.testing.assert_close(payload["base_state_dict"][key], value, msg=key)
            for key, value in payload["residual_state_dict"].items():
                if key == "log_std":
                    torch.testing.assert_close(value, captured["initial"][key])
                else:
                    self.assertFalse(torch.equal(value, captured["initial"][key]), key)
            self.assertFalse(payload["residual_config"]["learn_std"])
            self.assertEqual(len(payload["optimizer"]["param_groups"]), 2)
            self.assertAlmostEqual(payload["optimizer"]["param_groups"][1]["lr"],
                                   config["critic_lr"])

            runner, residual, residual_scale = load_act_runner(
                output / "policy_last.ckpt", "cpu"
            )
            self.assertIsNotNone(residual)
            self.assertAlmostEqual(residual_scale, config["residual_scale"])
            for key, value in runner.policy.state_dict().items():
                torch.testing.assert_close(value, base_state[key], msg=key)
            for key, value in residual.state_dict().items():
                torch.testing.assert_close(
                    value.cpu(), payload["residual_state_dict"][key], msg=key
                )

            chunks, features = runner.predict_base(
                [dict(_observation(0, 0), blocks=_privileged_state()["blocks"])],
                [0], with_features=True
            )
            self.assertEqual(chunks[0].shape, (1, ACTION_DIM))
            self.assertEqual(tuple(features.shape), (1, FEATURE_DIM))
            with torch.no_grad():
                actions, log_probs, values = residual.act(features)
            self.assertEqual(tuple(actions.shape), (1, ACTION_DIM))
            self.assertEqual(tuple(log_probs.shape), (1,))
            self.assertEqual(tuple(values.shape), (1,))

    def test_budget_stops_at_complete_wave_and_keeps_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, _, _ = self._write_base(root)
            output = root / "online"
            env = _OnlineFakeEnv()
            config = _online_config(checkpoint)
            console = io.StringIO()
            with mock.patch.object(act_online, "_budget_exhausted",
                                   side_effect=[False, True]), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=env), \
                    mock.patch.object(act_online, "online_output_dir", return_value=output), \
                    mock.patch("sys.stdout", console):
                last = act_online.train_online(config)

            self.assertEqual(last, output / "policy_last.ckpt")
            self.assertTrue(env.closed)
            # The second wave was never started; the single complete wave was
            # optimized exactly once and then the bounded loop stopped.
            self.assertEqual(env.reset_calls, [[0, 0]])
            self.assertEqual(env.step_calls, [[0, 1]] * 2)
            records = [json.loads(line) for line in
                       (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["update"], 1)
            self.assertEqual(records[0]["absolute_update"], 1)
            self.assertEqual(records[0]["waves"], 1)
            self.assertTrue(records[0]["budget_limited"])
            self.assertEqual(records[0]["rollout_steps"], 4)
            self.assertEqual(records[0]["global_step"], 4)
            self.assertEqual(records[0]["episodes"], 2)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["completed_updates"], 1)
            self.assertEqual(metadata["stop_reason"], "budget")
            self.assertEqual(metadata["final_update"], 1)
            self.assertEqual(metadata["final_global_step"], 4)
            for name in ("policy_last.ckpt",):
                payload = torch.load(output / name, map_location="cpu", weights_only=True)
                self.assertEqual(payload["update"], 1)
                self.assertEqual(payload["global_step"], 4)
                self.assertIn("numpy_rng_state", payload)
                self.assertIsNotNone(payload["torch_rng_state"])
            # Only the recoverable last checkpoint is kept between rounds.
            self.assertFalse((output / "policy_latest.ckpt").exists())
            self.assertFalse((output / "policy_last.ckpt.tmp").exists())
            load_act_runner(output / "policy_last.ckpt", "cpu")

    def test_expired_budget_saves_current_state_without_an_empty_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, _, _ = self._write_base(root)
            output = root / "online"
            env = _OnlineFakeEnv()
            config = _online_config(checkpoint)
            with mock.patch.object(act_online, "_budget_exhausted", return_value=True), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=env), \
                    mock.patch.object(act_online, "online_output_dir", return_value=output), \
                    mock.patch("sys.stdout", io.StringIO()):
                last = act_online.train_online(config)

            self.assertEqual(env.reset_calls, [])
            self.assertEqual(env.step_calls, [])
            self.assertEqual((output / "metrics.jsonl").read_text(), "")
            self.assertEqual((output / "episodes.jsonl").read_text(), "")
            payload = torch.load(last, map_location="cpu", weights_only=True)
            self.assertEqual(payload["update"], 0)
            self.assertEqual(payload["global_step"], 0)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["completed_updates"], 0)
            self.assertEqual(metadata["stop_reason"], "budget")

    def test_online_base_continues_residual_optimizer_and_global_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, base_config, base_state = self._write_base(root)
            first_output = root / "first"
            first_config = _online_config(checkpoint)
            first_config["finetune_updates"] = 1
            first_config["rollout_layout_ids"] = [14, 15, 16]
            first_env = _OnlineFakeEnv()
            with mock.patch.object(act_online, "RemoteEnv", return_value=first_env), \
                    mock.patch.object(act_online, "online_output_dir", return_value=first_output), \
                    mock.patch("sys.stdout", io.StringIO()):
                first_last = act_online.train_online(first_config)
            first = torch.load(first_last, map_location="cpu", weights_only=True)
            self.assertEqual(first["update"], 1)
            self.assertEqual(first_env.reset_calls, [[14, 15], [16, 14]])
            self.assertEqual(first["layout_cursor"], 4)
            self.assertEqual(first["global_step"], 8)

            second_output = root / "second"
            second_config = _online_config(first_last)
            second_config["finetune_updates"] = 1
            second_config["rollout_layout_ids"] = [14, 15, 16]
            second_config["lr"] = 5e-3
            second_env = _OnlineFakeEnv()
            captured = {}
            real_ppo_update = act_online.ppo_update

            def capture_ppo_update(model, optimizer, batch, **kwargs):
                captured["residual"] = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
                captured["lr"] = optimizer.param_groups[0]["lr"]
                captured["optimizer_steps"] = {
                    key: float(state["step"])
                    for key, state in optimizer.state_dict()["state"].items()
                }
                return real_ppo_update(model, optimizer, batch, **kwargs)

            with mock.patch.object(act_online, "ppo_update", side_effect=capture_ppo_update), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=second_env), \
                    mock.patch.object(act_online, "online_output_dir", return_value=second_output), \
                    mock.patch("sys.stdout", io.StringIO()):
                second_last = act_online.train_online(second_config)

            self.assertTrue(second_env.closed)
            self.assertEqual(second_env.reset_calls, [[15, 16], [14, 15]])
            # Before the first new PPO step the residual, optimizer moments and
            # requested lr come from the saved online checkpoint.
            for key, value in first["residual_state_dict"].items():
                torch.testing.assert_close(captured["residual"][key], value, msg=key)
            self.assertAlmostEqual(captured["lr"], 5e-3)
            first_steps = {key: float(state["step"])
                           for key, state in first["optimizer"]["state"].items()}
            self.assertTrue(first_steps)
            for key, steps in first_steps.items():
                self.assertEqual(captured["optimizer_steps"][key], steps, key)

            second = torch.load(second_last, map_location="cpu", weights_only=True)
            self.assertEqual(second["update"], 2)
            self.assertEqual(second["layout_cursor"], 8)
            self.assertEqual(second["global_step"], 16)
            self.assertEqual(second["base_config"], base_config)
            for key, value in base_state.items():
                torch.testing.assert_close(second["base_state_dict"][key], value, msg=key)
            for key, value in first["residual_state_dict"].items():
                if key == "log_std":
                    torch.testing.assert_close(second["residual_state_dict"][key], value)
                else:
                    self.assertFalse(torch.equal(second["residual_state_dict"][key], value), key)
            self.assertAlmostEqual(second["optimizer"]["param_groups"][0]["lr"], 5e-3)
            for key, state in first["optimizer"]["state"].items():
                self.assertGreater(float(second["optimizer"]["state"][key]["step"]),
                                   float(state["step"]), key)
            records = [json.loads(line) for line in
                       (second_output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(r["update"], r["absolute_update"], r["global_step"])
                              for r in records], [(1, 2, 16)])
            metadata = json.loads((second_output / "metadata.json").read_text())
            self.assertEqual(metadata["start_update"], 1)
            self.assertEqual(metadata["completed_updates"], 1)
            self.assertEqual(metadata["final_update"], 2)
            self.assertEqual(metadata["final_global_step"], 16)

            runner, residual, residual_scale = load_act_runner(second_last, "cpu")
            self.assertAlmostEqual(residual_scale, first_config["residual_scale"])
            for key, value in residual.state_dict().items():
                torch.testing.assert_close(value.cpu(), second["residual_state_dict"][key], msg=key)


    def test_resume_applies_requested_fixed_exploration_std(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, _, _ = self._write_base(root)
            first_output = root / "first"
            first_config = _online_config(checkpoint)
            first_config["finetune_updates"] = 1
            first_config["rollout_layout_ids"] = [14, 15, 16]
            first_config["initial_std"] = 0.1
            with mock.patch.object(act_online, "RemoteEnv", return_value=_OnlineFakeEnv()), \
                    mock.patch.object(act_online, "online_output_dir", return_value=first_output), \
                    mock.patch("sys.stdout", io.StringIO()):
                first_last = act_online.train_online(first_config)
            first = torch.load(first_last, map_location="cpu", weights_only=True)

            second_output = root / "second"
            second_config = _online_config(first_last)
            second_config["finetune_updates"] = 1
            second_config["rollout_layout_ids"] = [14, 15, 16]
            second_config["initial_std"] = 0.25
            captured = {}
            real_ppo_update = act_online.ppo_update

            def capture_ppo_update(model, optimizer, batch, **kwargs):
                captured["state"] = {key: value.detach().clone()
                                     for key, value in model.state_dict().items()}
                return real_ppo_update(model, optimizer, batch, **kwargs)

            with mock.patch.object(act_online, "ppo_update", side_effect=capture_ppo_update), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=_OnlineFakeEnv()), \
                    mock.patch.object(act_online, "online_output_dir", return_value=second_output), \
                    mock.patch("sys.stdout", io.StringIO()):
                second_last = act_online.train_online(second_config)

            # Before the first new PPO step: requested std applied, all learned
            # weights continue from the saved checkpoint.
            torch.testing.assert_close(captured["state"]["log_std"],
                                       torch.full((ACTION_DIM,), math.log(0.25)))
            for key, value in first["residual_state_dict"].items():
                if key != "log_std":
                    torch.testing.assert_close(captured["state"][key], value, msg=key)
            second = torch.load(second_last, map_location="cpu", weights_only=True)
            self.assertEqual(second["residual_config"]["initial_std"], 0.25)
            self.assertEqual(second["update"], 2)
            self.assertEqual(second["layout_cursor"], first["layout_cursor"] + 4)


if __name__ == "__main__":
    unittest.main()
