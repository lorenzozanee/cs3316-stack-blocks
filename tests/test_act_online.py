"""Wave collection and residual PPO integration without Isaac or real scores."""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import pickle
import sys
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
from starter.policies.stacking_reward import REWARD_KEYS
from tests.test_act import small_config

ROOT = Path(__file__).resolve().parents[1]

# Realistic reset scene shared by the collector fixtures: three 30 mm cubes
# resting upright on a 0.78 m table, separated so no two starting blocks
# overlap. The target region (0, -0.2) is empty at reset.
BLOCK_SIDE = 0.03
TABLE_TOP = 0.78
TABLE = {"top_z": TABLE_TOP, "x_min": -0.7, "x_max": 0.7,
         "y_min": -0.6, "y_max": 0.5}
BLOCK_EXTENTS = {label: [BLOCK_SIDE, BLOCK_SIDE, BLOCK_SIDE]
                 for label in BLOCK_LABELS}
START_POSITIONS = {
    "block_0": [-0.15, -0.10, TABLE_TOP + BLOCK_SIDE / 2.0],
    "block_1": [0.00, 0.10, TABLE_TOP + BLOCK_SIDE / 2.0],
    "block_2": [0.15, -0.10, TABLE_TOP + BLOCK_SIDE / 2.0],
}
TARGET_XY = (0.0, -0.2)


def _reward_cfg(**overrides) -> dict:
    """Reward configuration read from configs/act_online.yaml by REWARD_KEYS.

    A missing key is an error (no silent defaults); dwell stays an int."""
    config = load_config(ROOT / "configs/act_online.yaml")
    cfg = {key: config[key] for key in REWARD_KEYS}
    cfg.update(overrides)
    return cfg


def _zero_reward_cfg() -> dict:
    """All dense and milestone weights zero: reward is pure score delta."""
    return _reward_cfg(reach_weight=0.0, manipulation_weight=0.0, base_weight=0.0,
                       pair_weight=0.0, release_weight=0.0, waypoint_bonus=0.0,
                       milestone_grasp=0.0, milestone_base=0.0,
                       milestone_pair2=0.0, milestone_pair3=0.0,
                       empty_grasp_penalty=0.0, drop_penalty=0.0,
                       empty_place_penalty=0.0, empty_tower_intrusion_penalty=0.0,
                       tower_break_penalty=0.0, arm_conflict_penalty=0.0)


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


def _privileged_state(positions: dict | None = None) -> dict:
    """Full reward-state fixture, including the module's reset geometry contract."""
    positions = START_POSITIONS if positions is None else positions
    return {"blocks": {label: {"pos": [float(value) for value in positions[label]],
                               "rot": [1.0, 0.0, 0.0, 0.0]}
                       for label in BLOCK_LABELS},
            "block_extents": {label: list(BLOCK_EXTENTS[label])
                              for label in BLOCK_LABELS},
            "table": dict(TABLE),
            "gripper_open_measured": [1.0, 1.0],
            "block_velocities": {label: {"linear": [0.0, 0.0, 0.0],
                                         "angular": [0.0, 0.0, 0.0]}
                                 for label in BLOCK_LABELS}}


def _state_with_block_lift(dz: float) -> dict:
    """Fresh state with block_0 raised by dz above its reset height."""
    positions = {label: list(value) for label, value in START_POSITIONS.items()}
    positions["block_0"][2] += dz
    return _privileged_state(positions)


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
                (self._observation(scene, step), score, terminated, truncated,
                 {"success_reward": float(terminated)})
            )
        return results


class PostGraspReplayTest(unittest.TestCase):
    class Env(_ScriptedEnv):
        def __init__(self, scores=(0, 0, 0, 15, 15)):
            super().__init__({0: {"length": 5, "terminated": False},
                              1: {"length": 4, "terminated": False}}, {0: scores, 1: [0]*4})
            self.actions = []

        def step_batch(self, actions, active):
            self.actions.append(dict(zip(active, copy.deepcopy(actions))))
            results = super().step_batch(actions, active)
            state = _state_with_block_lift(0.02)
            config = _reward_cfg()
            measured = ((BLOCK_SIDE - config['gripper_aperture_zero']) / 2
                        - config['gripper_scale_low']) / (config['gripper_scale_high']
                                                          - config['gripper_scale_low'])
            if 0 in active:
                idx = active.index(0)
                state["gripper_open_measured"][0] = measured
                self._privileged[idx] = state
                wrist = results[idx][0]["proprio"]
                wrist[:3] = state["blocks"]["block_0"]["pos"]
                wrist[0] -= config['grasp_offset']
                wrist[7] = measured
            return results

    def _run(self, env, reward_cfg=None, **kwargs):
        runner, actor = _ScriptedRunner(), _RecordingActor()
        batch, episodes = collect_wave(env, runner, actor, [14, 15], gamma=0.9,
                                       gae_lambda=0.95, residual_scale=0.2,
                                       reward_cfg=reward_cfg or _reward_cfg(), **kwargs)
        return batch, {ep['layout_id']: ep for ep in episodes}, runner, actor

    def test_capture_and_replay_preserves_history_and_excludes_forced_samples(self):
        entries = []
        plain, original, _, _ = self._run(self.Env(), capture_prefix=entries.append)
        self.assertEqual(len(entries), 1)
        self.assertEqual((entries[0]['layout_id'], entries[0]['scene'], entries[0]['arm'],
                          entries[0]['carried'], len(entries[0]['actions'])),
                         (14, 0, 0, 'block_0', 3))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'prefixes.jsonl'
            path.write_text(json.dumps(entries[0]) + '\n')
            entry = act_online.load_prefixes(path, 2, [14, 15], _reward_cfg())[14, 0]
            with self.assertRaisesRegex(ValueError, 'metadata'):
                act_online.load_prefixes(path, 4, [14, 15], _reward_cfg())
            with self.assertRaisesRegex(ValueError, 'different reward configuration'):
                act_online.load_prefixes(path, 2, [14, 15],
                                         {**_reward_cfg(), 'milestone_grasp': 9.9})
        requested_action = copy.deepcopy(entry['actions'][0])
        requested_action[0] += 0.01
        entry['actions'][0] = requested_action
        env = self.Env()
        replay, records, runner, actor = self._run(env, prefixes={0: entry})
        self.assertEqual(env.actions[0][0], requested_action)
        self.assertEqual(records[14]['prefix_steps'], 3)
        self.assertEqual(records[14]['length'], 2)
        self.assertEqual(records[15]['prefix_steps'], 0)
        self.assertEqual(records[14]['final_score'], 15)
        self.assertAlmostEqual(original[14]['milestone_return'] - records[14]['milestone_return'], 1.2)
        torch.testing.assert_close(replay.returns, plain.returns[3:])
        self.assertEqual(runner.calls[:4], [[0, 1]]*4)
        sampled = [c['features'][:, :2].tolist() for c in actor.calls if not c['deterministic']]
        self.assertEqual(sampled, [[[1, 0]], [[1, 1]], [[1, 2]], [[0, 3], [1, 3]], [[0, 4]]])

    def test_dense_prefix_recomputes_progress_and_checks_discrete_events(self):
        class ApproachEnv(self.Env):
            def __init__(self, offset=0.0):
                super().__init__()
                self.offset = offset

            def step_batch(self, actions, active):
                results = super().step_batch(actions, active)
                if 0 in active and self._steps[0] == 1:
                    index = active.index(0)
                    right = results[index][0]["proprio"][8:11]
                    right[:] = START_POSITIONS["block_1"]
                    right[0] -= _reward_cfg()["grasp_offset"]
                    right[1] -= 0.1 + self.offset
                return results

        config = _reward_cfg(dense_transport_reward=True)
        entries = []
        original, episodes, _, _ = self._run(ApproachEnv(), config, capture_prefix=entries.append)
        entry = entries[0]
        self.assertGreater(episodes[14]["dense_approach_return"], 0.0)
        self.assertEqual(entry["frames"][1]["bonus"], 0.0)
        self.assertAlmostEqual(entry["frames"][3]["bonus"], 1.2)
        replay, records, _, _ = self._run(ApproachEnv(0.001), config, prefixes={0: entry})
        self.assertEqual(records[14]["prefix_steps"], 3)
        torch.testing.assert_close(replay.returns, original.returns[3:])
        wrong_event = copy.deepcopy(entry)
        wrong_event["frames"][3]["bonus"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "diverged"):
            self._run(ApproachEnv(), config, prefixes={0: wrong_event})

    def test_divergent_or_scoring_prefix_is_rejected(self):
        entries = []
        self._run(self.Env(), capture_prefix=entries.append)
        entry = entries[0]
        with self.assertRaisesRegex(RuntimeError, 'crossed a task score'):
            self._run(self.Env((0, 15, 15, 15, 15)), prefixes={0: entry})
        different_position = copy.deepcopy(entry)
        different_position['frames'][-1]['blocks']['block_0']['pos'][0] += 0.05
        different_speed = copy.deepcopy(entry)
        different_speed['frames'][-1]['velocities']['block_0']['linear'][0] += 0.05
        for altered in (different_position, different_speed):
            with self.assertRaisesRegex(RuntimeError, 'replay diverged.*layout 14'):
                self._run(self.Env(), prefixes={0: altered})

    def test_intermediate_divergence_or_event_mismatch_is_rejected(self):
        entries = []
        self._run(self.Env(), capture_prefix=entries.append)
        self.assertEqual(len(entries[0]['frames']), len(entries[0]['actions']) + 1)
        mid_position = copy.deepcopy(entries[0])
        mid_position['frames'][1]['blocks']['block_1']['pos'][1] += 0.05
        bonus = copy.deepcopy(entries[0])
        bonus['frames'][1]['bonus'] += 1.0
        for altered, metric in ((mid_position, 'block_1.position_m'), (bonus, 'event_bonus')):
            with self.assertRaisesRegex(RuntimeError, rf'replay diverged.*layout 14, scene 0, step 1: {metric} delta='):
                self._run(self.Env(), prefixes={0: altered})
        other_support = copy.deepcopy(entries[0])
        other_support['target_lower'] = 'block_2'
        if entries[0]['target_lower'] != 'block_2':
            with self.assertRaisesRegex(RuntimeError, 'latched a different support'):
                self._run(self.Env(), prefixes={0: other_support})
        missing = copy.deepcopy(entries[0])
        missing['frames'] = missing['frames'][:-1]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'prefixes.jsonl'
            path.write_text(json.dumps(missing) + '\n')
            with self.assertRaisesRegex(ValueError, 'physical snapshot'):
                act_online.load_prefixes(path, 2, [14, 15], _reward_cfg())


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
            residual_scale=0.05, reward_cfg=_zero_reward_cfg(),
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
            gae_lambda=0.8, residual_scale=0.05, reward_cfg=_zero_reward_cfg(),
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
        self.assertIn("mean_penalty_return", diagnostics)
        self.assertEqual(diagnostics["mean_penalty_return"], 0.0)
        for key in ("drop", "empty_place", "empty_tower_intrusion", "tower_break", "empty_tower_break"):
            self.assertEqual(diagnostics[f"penalty_{key}_count"], 0)
        self.assertEqual(diagnostics["residual_clip_fraction"], 1 / (4 * ACTION_DIM))
        self.assertAlmostEqual(diagnostics["normalized_residual_abs_mean"],
                               0.05 / (4 * ACTION_DIM))
        self.assertAlmostEqual(diagnostics["advantage_std"],
                               float(batch.advantages.std(unbiased=False)))

    def test_episode_diagnostics_log_potential_components_and_tower_state(self):
        # Static fixture: three resting cubes, identity-quat proprio, open
        # grippers. Every measurement is identical at both frames.
        cfg = _reward_cfg()
        env = _ScriptedEnv({0: {"length": 2, "terminated": False}}, {0: [0.0, 0.0]})
        _, episodes = collect_wave(
            env, _ScriptedRunner(), _RecordingActor(), [14], gamma=0.9, gae_lambda=0.8,
            residual_scale=0.05, reward_cfg=cfg,
        )
        ep = episodes[0]
        self.assertEqual(ep["phi_initial"], ep["phi_final"])
        self.assertEqual(ep["phi_initial"], ep["phi_mean"])
        self.assertAlmostEqual(
            ep["phi_final"],
            ep["phi_reach_final"] + ep["phi_manipulation_final"] + ep["phi_base_final"]
            + ep["phi_pair_final"] + ep["phi_release_final"], places=6)
        self.assertAlmostEqual(ep["d_min_initial_m"], ep["d_min_final_m"], places=9)
        self.assertGreater(ep["d_min_final_m"], 0.0)
        self.assertEqual(ep["lift_initial_m"], 0.0)
        self.assertEqual(ep["lift_final_m"], 0.0)
        self.assertEqual(ep["pair_sum_final"], 0.0)
        self.assertAlmostEqual(ep["phi_manipulation_final"], -3.0 * cfg["manipulation_weight"])
        self.assertEqual(ep["grasp_attempts"], {})
        self.assertAlmostEqual(ep["phi_pair_final"], -2.0 * cfg["pair_weight"])
        # Nearest starting cube (block_0) is still outside the 10 mm target
        # radius; its transport/base geometry contributes only a soft fraction.
        target_distance = math.hypot(START_POSITIONS["block_0"][0] - TARGET_XY[0],
                                     START_POSITIONS["block_0"][1] - TARGET_XY[1])
        proximity = 1.0 - math.tanh(
            max(0.0, target_distance - cfg["placement_radius"]) / cfg["placement_soft_scale"])
        self.assertAlmostEqual(ep["phi_base_final"], -cfg["base_weight"] * (1.0 - proximity))
        self.assertEqual(ep["phi_release_final"], 0.0)
        self.assertEqual(ep["min_gripper_final"], 1.0)
        self.assertEqual(ep["settled_pair_final"], 0.0)
        self.assertEqual(ep["lift_progress_final"], 0.0)
        self.assertEqual(ep["final_gripper_open_measured"], [1.0, 1.0])
        self.assertEqual(ep["final_pair_dz_m"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(ep["final_gripper_openness"], [1.0, 1.0], atol=1e-6)
        # Starting cubes are separated: every label pair is a real xy distance.
        for xy in ep["final_pair_xy_m"]:
            self.assertGreater(xy, 0.0)
        # Milestone diagnostics are present and empty for a static episode.
        self.assertEqual(ep["milestone_return"], 0.0)
        self.assertEqual(ep["milestone_steps"],
                         {"grasp": None, "base": None, "pair2": None, "pair3": None})
        self.assertEqual(ep["stable_tower_levels_final"], 0)
        # Two truncated steps keep the (constant) potential each time.
        self.assertAlmostEqual(ep["shaping_return"],
                               2.0 * (0.9 * ep["phi_final"] - ep["phi_initial"]), places=6)

        # Airborne but not held: lift is logged, not rewarded as manipulation.
        # Truncation still keeps the observed final potential.
        env = _ScriptedEnv({0: {"length": 2, "terminated": False}}, {0: [0.0, 0.0]})
        static_state = _privileged_state()
        with mock.patch.object(sys.modules[__name__], "_privileged_state",
                               side_effect=[_privileged_state(), static_state,
                                            _state_with_block_lift(0.05)]):
            _, episodes = collect_wave(
                env, _ScriptedRunner(), _RecordingActor(), [14], gamma=0.9, gae_lambda=0.8,
                residual_scale=0.05, reward_cfg=cfg,
            )
        ep = episodes[0]
        self.assertAlmostEqual(ep["lift_initial_m"], 0.0, places=6)
        self.assertAlmostEqual(ep["lift_final_m"], 0.05, places=6)
        self.assertAlmostEqual(ep["lift_gate_final"], 0.5, places=6)
        self.assertAlmostEqual(ep["lift_progress_final"], 1.0, places=6)
        self.assertAlmostEqual(ep["phi_manipulation_final"], -3.0 * cfg["manipulation_weight"])
        self.assertEqual(ep["max_manipulation_progress"], 0.0)
        self.assertAlmostEqual(ep["phi_initial"], ep["phi_mean"] * 2 - ep["phi_final"],
                               places=6)
        self.assertAlmostEqual(ep["phi_final"], ep["phi_initial"], places=6)
        self.assertEqual(ep["stable_tower_levels_final"], 0)
        self.assertEqual(ep["pair_sum_final"], 0.0)
        phi0, phi1 = ep["phi_initial"], ep["phi_final"]
        self.assertAlmostEqual(ep["shaping_return"],
                               0.9 * phi0 - phi0 + 0.9 * phi1 - phi0, places=6)

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
            residual_scale=0.05, reward_cfg=_zero_reward_cfg(),
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
        self.assertEqual([episode["final_success_reward"] for episode in episodes], [1.0, 0.0])


class PBRSCollectorTest(unittest.TestCase):
    """Collector PBRS: reward.step terms drive terminal/truncation shaping."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _collect(self, terminated: bool, after_state: dict, score: float):
        env = _ScriptedEnv({0: {"length": 1, "terminated": terminated}}, {0: [score]})
        # Milestone weights zero: isolate the potential-based shaping term.
        cfg = _reward_cfg(reach_weight=2.0, manipulation_weight=1.0, base_weight=1.0,
                          milestone_grasp=0.0, milestone_base=0.0,
                          milestone_pair2=0.0, milestone_pair3=0.0)
        with mock.patch.object(sys.modules[__name__], "_privileged_state",
                               side_effect=[_privileged_state(), after_state]):
            return collect_wave(
                env, _ScriptedRunner(), _RecordingActor(), [0], gamma=0.9,
                gae_lambda=0.8, residual_scale=0.05, reward_cfg=cfg,
            )

    def test_true_terminal_forces_zero_potential_and_telescopes(self):
        placed = {**START_POSITIONS, "block_0": [*TARGET_XY, START_POSITIONS["block_0"][2]]}
        batch, episodes = self._collect(True, _privileged_state(placed), SUCCESS_SCORE)
        ep = episodes[0]
        self.assertTrue(ep["terminated"])
        self.assertEqual(ep["final_score"], SUCCESS_SCORE)
        self.assertEqual(ep["milestone_return"], 0.0)
        # The final frame really has a different potential, so the terminal
        # zero is doing work rather than matching a constant phi.
        self.assertNotAlmostEqual(ep["phi_final"], ep["phi_initial"], places=6)
        # True terminal: gamma * phi(terminal) = 0, so shaping telescopes to
        # -phi(start) over the single step (Grzes, AAMAS 2017).
        expected_shaping = -ep["phi_initial"]
        self.assertAlmostEqual(ep["shaping_return"], expected_shaping, places=6)
        self.assertAlmostEqual(ep["episode_return"],
                               SUCCESS_SCORE + expected_shaping, places=6)
        self.assertAlmostEqual(ep["discounted_return"], ep["episode_return"], places=6)
        # No value bootstrap at a true terminal: the return is reward only.
        self.assertAlmostEqual(float(batch.returns[0]),
                               SUCCESS_SCORE + expected_shaping, delta=1e-4)

    def test_truncation_retains_final_potential_and_bootstraps(self):
        batch, episodes = self._collect(False, _state_with_block_lift(0.05), 0.0)
        ep = episodes[0]
        self.assertTrue(ep["truncated"])
        self.assertNotAlmostEqual(ep["phi_final"], 0.0, places=6)
        # Truncation keeps phi(next) instead of zeroing it (Pardo, ICML 2018):
        # reward = gamma * phi(s') - phi(s).
        expected_shaping = 0.9 * ep["phi_final"] - ep["phi_initial"]
        self.assertAlmostEqual(ep["shaping_return"], expected_shaping, places=6)
        self.assertAlmostEqual(ep["episode_return"], expected_shaping, places=6)
        self.assertAlmostEqual(ep["discounted_return"], expected_shaping, places=6)
        # The terminal observation's critic value (feature sum 0+1+1) is still
        # bootstrapped, unlike the true terminal case above.
        self.assertAlmostEqual(float(batch.returns[0]),
                               expected_shaping + 0.9 * 2.0, delta=1e-4)


class MilestoneBonusTest(unittest.TestCase):
    """One-time, dwell-gated bonuses are outside the PBRS shaping return."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def _target_state() -> dict:
        positions = {label: list(value) for label, value in START_POSITIONS.items()}
        positions["block_0"][:2] = list(TARGET_XY)
        return _privileged_state(positions)

    def _collect_static(self, cfg: dict, length: int) -> dict:
        state = self._target_state()
        env = _ScriptedEnv({0: {"length": length, "terminated": False}},
                           {0: [0.0] * length})
        with mock.patch.object(sys.modules[__name__], "_privileged_state",
                               side_effect=[state] * (length + 1)):
            _, episodes = collect_wave(
                env, _ScriptedRunner(), _RecordingActor(), [0], gamma=0.9,
                gae_lambda=0.8, residual_scale=0.05, reward_cfg=cfg,
            )
        return episodes[0]

    def test_bonus_paid_once_and_accounted_separately_from_shaping(self):
        cfg = _reward_cfg(reach_weight=2.0, base_weight=1.0, milestone_dwell_steps=2)
        ep = self._collect_static(cfg, length=4)
        self.assertEqual(ep["final_score"], 0.0)
        # The base predicate holds from step 2 onward; the bonus latches once.
        self.assertAlmostEqual(ep["milestone_return"], cfg["milestone_base"])
        self.assertEqual(ep["milestone_steps"],
                         {"grasp": None, "base": 2, "pair2": None, "pair3": None})
        self.assertNotEqual(ep["shaping_return"], 0.0)
        # Shaping and milestone accounting are disjoint but both reach the
        # episode return: shaping inside shaping_return, bonus outside it.
        self.assertAlmostEqual(ep["episode_return"],
                               ep["final_score"] + ep["shaping_return"]
                               + ep["milestone_return"], places=6)

    def test_zero_milestone_weights_suppress_the_event_reward(self):
        cfg = _zero_reward_cfg()
        cfg["milestone_dwell_steps"] = 1
        ep = self._collect_static(cfg, length=3)
        # The latch still records the event, but a zero weight pays nothing.
        self.assertEqual(ep["milestone_return"], 0.0)
        self.assertEqual(ep["milestone_steps"],
                         {"grasp": None, "base": 1, "pair2": None, "pair3": None})
        # All dense weights are zero too, so the reward is the pure score delta.
        self.assertEqual(ep["shaping_return"], 0.0)
        self.assertEqual(ep["episode_return"], ep["final_score"])


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
        self.connected = False
        self.closed = False
        self._state = {}
        self._privileged = []

    def connect(self):
        self.connected = True

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
    # The reward configuration travels in the same flat dict as the trainer
    # hyperparameters; REWARD_KEYS is the module contract.
    config.update(_reward_cfg())
    config.update(
        base_checkpoint=str(base_checkpoint),
        device="cpu",
        unfreeze_act=False,
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

    def test_residual_scale_allows_half_a_demonstration_std(self):
        config = _online_config(Path("/unused/offline.ckpt"))
        config["residual_scale"] = 0.5
        act_online._validate_config(config)
        config["residual_scale"] = 0.51
        with self.assertRaisesRegex(ValueError, "residual_scale"):
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

    def test_reward_weights_must_be_nonnegative_and_finite(self):
        for key in ("reach_weight", "manipulation_weight", "base_weight", "pair_weight",
                    "release_weight", "milestone_grasp", "milestone_base",
                    "milestone_pair2", "milestone_pair3",
                    "empty_grasp_penalty", "drop_penalty", "empty_place_penalty",
                    "empty_tower_intrusion_penalty", "tower_break_penalty",
                    "arm_conflict_penalty"):
            for value in (-1, float("inf"), float("nan")):
                with self.subTest(key=key, value=value):
                    config = _online_config(Path("/unused/offline.ckpt"))
                    config[key] = value
                    with self.assertRaises(ValueError):
                        act_online._validate_config(config)

    def test_reward_geometry_constraints_are_validated(self):
        # Positive scales and tolerances (zero is invalid for each).
        for key in ("reach_scale", "placement_radius", "placement_soft_scale",
                    "grasp_offset", "grasp_aperture_fraction", "grasp_centering_fraction",
                    "gripper_scale_high", "settle_linear_speed",
                    "settle_angular_speed", "support_gap_fraction"):
            with self.subTest(key=key):
                config = _online_config(Path("/unused/offline.ckpt"))
                config[key] = 0.0
                with self.assertRaises(ValueError):
                    act_online._validate_config(config)
        cases = {
            # placement_radius must stay strictly inside the soft scale.
            "placement_radius": 0.2,
            "support_gap_floor": -0.001,
        }
        for key, value in cases.items():
            with self.subTest(key=key, value=value):
                config = _online_config(Path("/unused/offline.ckpt"))
                config[key] = value
                with self.assertRaises(ValueError):
                    act_online._validate_config(config)
        for value in (0.0, 1.0, -0.5):
            with self.subTest(key="upright_cos_min", value=value):
                config = _online_config(Path("/unused/offline.ckpt"))
                config["upright_cos_min"] = value
                with self.assertRaises(ValueError):
                    act_online._validate_config(config)

    def test_milestone_dwell_must_be_a_true_integer(self):
        for value in (0, -1, 1.5, 2.0, True, False, "3", None):
            with self.subTest(value=value):
                config = _online_config(Path("/unused/offline.ckpt"))
                config["milestone_dwell_steps"] = value
                with self.assertRaises(ValueError):
                    act_online._validate_config(config)
        for value in (1, 3):
            with self.subTest(value=value):
                config = _online_config(Path("/unused/offline.ckpt"))
                config["milestone_dwell_steps"] = value
                act_online._validate_config(config)

    def test_milestone_total_bonus_must_stay_below_partial_score(self):
        config = _online_config(Path("/unused/offline.ckpt"))
        # 0.5 + 1.0 + 1.0 + 13.0 = 15.5 >= the 15-point partial score.
        config["milestone_pair3"] = 13.0
        with self.assertRaises(ValueError):
            act_online._validate_config(config)

    def test_negative_placement_xy_is_valid_configuration(self):
        config = _online_config(Path("/unused/offline.ckpt"))
        config["placement_x"] = -0.3
        config["placement_y"] = -0.25
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
                phases = [record[key] for key in ("reset_seconds", "act_seconds", "step_rpc_seconds")]
                self.assertTrue(all(seconds > 0 for seconds in phases))
                self.assertGreater(record["rollout_seconds"], sum(phases))
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
                                       episode["final_score"] + episode["shaping_return"]
                                       + episode["milestone_return"])
                self.assertEqual(episode["milestone_return"], 0.0)
                self.assertAlmostEqual(float(row["episode_return"]), episode["episode_return"])
                self.assertEqual(list(row), [
                    "episode", "absolute_update", "seed", "layout_id", "length", "prefix_steps",
                    "final_score", "peak_score", "episode_return", "first_partial_step",
                    "arrivals", "grasps", "carries", "aligned", "opened", "settled",
                ])
                for field, counter in (("arrivals", "arrivals"), ("grasps", "confirmed")):
                    self.assertEqual(int(row[field]), sum(
                        counts[counter] for counts in episode["grasp_attempts"].values()))
                self.assertIn("discounted_return", episode)
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

    def test_remaining_seconds_uses_the_earliest_stop(self):
        self.assertAlmostEqual(act_online._remaining_seconds(120.0, 2, 10, 3600.0), 480.0)
        self.assertAlmostEqual(act_online._remaining_seconds(120.0, 2, 10, 300.0), 180.0)
        self.assertAlmostEqual(act_online._remaining_seconds(120.0, 2, sys.maxsize, 3600.0),
                               3480.0)
        self.assertAlmostEqual(act_online._remaining_seconds(3600.0, 3, sys.maxsize, 3600.0),
                               0.0)

    def test_budget_stops_at_complete_wave_and_keeps_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, _, _ = self._write_base(root)
            output = root / "online"
            env = _OnlineFakeEnv()
            config = _online_config(checkpoint)
            console = io.StringIO()
            connected_at = []
            connect = env.connect

            def record_connection():
                connect()
                connected_at.append(act_online.time.monotonic())

            env.connect = record_connection
            with mock.patch.object(act_online, "_budget_exhausted",
                                   side_effect=[False, True]) as budget, \
                    mock.patch.object(act_online, "RemoteEnv", return_value=env), \
                    mock.patch.object(act_online, "online_output_dir", return_value=output), \
                    mock.patch("sys.stdout", console):
                last = act_online.train_online(config)

            self.assertEqual(last, output / "policy_last.ckpt")
            self.assertTrue(env.closed)
            self.assertEqual(len(connected_at), 1)
            self.assertGreaterEqual(budget.call_args_list[0].args[0], connected_at[0])
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

    def test_budget_exit_after_full_updates_renders_final_history(self):
        from starter.policies import act_online_metrics

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, _, _ = self._write_base(root)
            output = root / "online"
            env = _OnlineFakeEnv()
            config = _online_config(checkpoint)
            config.update(finetune_episodes=2, finetune_updates=100, save_freq=100)
            plotted_updates = []

            def record_plot(path):
                self.assertTrue(env.closed)
                self.assertEqual(path, output)
                records = act_online_metrics._load_records(path)
                plotted_updates.append([record["absolute_update"] for record in records])

            with mock.patch.object(act_online, "_budget_exhausted", side_effect=[False, False, True]), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=env), \
                    mock.patch.object(act_online, "online_output_dir", return_value=output), \
                    mock.patch.object(act_online_metrics, "plot_online_training", side_effect=record_plot), \
                    mock.patch("sys.stdout", io.StringIO()):
                act_online.train_online(config)

            self.assertEqual(plotted_updates, [[1, 2]])
            self.assertEqual(json.loads((output / "metadata.json").read_text())["stop_reason"], "budget")

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
            second_config["rollout_layout_ids"] = list(range(55))
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
            self.assertEqual(second_env.reset_calls, [[4, 5], [6, 7]])
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
            self.assertEqual(second["global_step"], 15)
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
                              for r in records], [(1, 2, 15)])
            metadata = json.loads((second_output / "metadata.json").read_text())
            self.assertEqual(metadata["start_update"], 1)
            self.assertEqual(metadata["completed_updates"], 1)
            self.assertEqual(metadata["final_update"], 2)
            self.assertEqual(metadata["final_global_step"], 15)

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

    def test_env_workers_validation_and_sharded_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, _, _ = self._write_base(root)
            for bad_workers in (0, -1, True, 1.5, 3):
                cfg = _online_config(checkpoint)
                cfg.update(num_envs=4, finetune_episodes=4, finetune_updates=1, env_workers=bad_workers)
                with self.assertRaises(ValueError):
                    act_online._validate_config(cfg)
            overflow = _online_config(checkpoint)
            overflow.update(num_envs=4, finetune_episodes=4, finetune_updates=1, env_workers=2, env_port=65535)
            with self.assertRaises(ValueError):
                act_online._validate_config(overflow)

            cfg = _online_config(checkpoint)
            cfg.update(
                num_envs=4, finetune_episodes=4, finetune_updates=1,
                env_workers=2, min_reset_free_vram_mib=2560,
            )
            fake_env = _OnlineFakeEnv()
            out = root / "sharded_out"
            with mock.patch.object(act_online, "ShardedRemoteEnv", return_value=fake_env) as sharded_cls, \
                    mock.patch.object(act_online, "online_output_dir", return_value=out), \
                    mock.patch("sys.stdout", io.StringIO()):
                ckpt = act_online.train_online(cfg)
            sharded_cls.assert_called_once_with(
                "127.0.0.1", [5555, 5556], num_envs=4, min_reset_free_vram_mib=2560
            )
            meta = json.loads((ckpt.parent / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["env_workers"], 2)


if __name__ == "__main__":
    unittest.main()
