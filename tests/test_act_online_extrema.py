"""Focused tests for the round15 per-episode extrema logging (observability).

The extrema accumulate over every step so a transient lift/grasp/release is
visible even when the final frame is at the floor. These tests assert the
logging is write-only (no action/reward/score/trajectory drift) and that it
captures transient events the final-frame fields miss.
"""

from __future__ import annotations

import json
import unittest

import torch

from starter.interfaces import BLOCK_LABELS
from starter.policies import act_online
from starter.policies.act_online import collect_wave
from tests.test_act_online import (
    START_POSITIONS,
    _privileged_state,
    _RecordingActor,
    _reward_cfg,
    _ScriptedEnv,
    _ScriptedRunner,
    _zero_reward_cfg,
)


def _lifted_state(z0: float) -> dict:
    """Fresh privileged state with only the first block raised to z0."""
    state = _privileged_state()
    state["blocks"][BLOCK_LABELS[0]]["pos"][2] = z0
    return state


class _TransientEnv(_ScriptedEnv):
    """Scripted env whose privileged state varies per step (transient events)."""

    def __init__(self, scripts, scores, privileged_script):
        super().__init__(scripts, scores)
        self.privileged_script = privileged_script

    def step_batch(self, actions, active):
        self.step_calls.append([int(scene) for scene in active])
        if len(actions) != len(active):
            raise AssertionError("actions and active scenes must align")
        results = []
        privileged = []
        for scene in active:
            self._steps[scene] += 1
            step = self._steps[scene]
            script = self.scripts[scene]
            score = float(self.scores[scene][step - 1])
            ended = step >= script["length"]
            terminated = bool(ended and script["terminated"])
            truncated = bool(ended and not script["terminated"])
            privileged.append(self.privileged_script[scene][step - 1])
            results.append(
                (self._observation(scene, step), score, terminated, truncated, {})
            )
        self._privileged = privileged
        return results


class ExtremaLoggingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_transient_lift_captured_while_final_frame_at_floor(self):
        scripts = {0: {"length": 3, "terminated": False}}
        scores = {0: [0.0, 0.0, 0.0]}
        privileged = {0: [_lifted_state(0.795), _lifted_state(0.855), _lifted_state(0.795)]}
        env = _TransientEnv(scripts, scores, privileged)
        _, episodes = collect_wave(
            env, _ScriptedRunner(), _RecordingActor(), [0],
            gamma=0.999, gae_lambda=0.95, residual_scale=0.2,
            reward_cfg=_reward_cfg(),
        )
        ep = episodes[0]
        # Final frame is back at the reset height: the old fields see nothing.
        self.assertEqual(ep["lift_final_m"], 0.0)
        self.assertEqual(ep["lift_progress_final"], 0.0)
        self.assertEqual(ep["peak_score"], 0.0)
        # Extrema reveal the 60 mm transient lift at step 2.
        self.assertAlmostEqual(ep["max_lift_m"], 0.06, places=9)
        self.assertAlmostEqual(ep["max_lift_per_block_m"][BLOCK_LABELS[0]], 0.06, places=9)
        self.assertEqual(ep["first_lift_step"], 2)
        self.assertGreaterEqual(ep["max_lift_progress"], 1.0)
        # lift_gate is lift_progress/2 (a tower needs two raised blocks).
        self.assertEqual(ep["max_lift_gate"], 0.5)
        # Other blocks never moved.
        for label in BLOCK_LABELS[1:]:
            self.assertEqual(ep["max_lift_per_block_m"][label], 0.0)
        # A single lifted block is not a tower and pays no event bonus.
        self.assertEqual(ep["stable_tower_levels_final"], 0)
        self.assertEqual(ep["milestone_return"], 0.0)

    def test_grasp_attempts_and_terminal_step_reach_episode_metrics(self):
        resting_z = START_POSITIONS["block_0"][2]
        held_z = resting_z + 0.05
        held = _lifted_state(held_z)
        cfg = _reward_cfg()
        joint = (0.03 - cfg["gripper_aperture_zero"]) / 2.0
        low, high = cfg["gripper_scale_low"], cfg["gripper_scale_high"]
        held["gripper_open_measured"][0] = (joint - low) / (high - low)

        class PickupEnv(_TransientEnv):
            @staticmethod
            def _observation(scene, step):
                obs = _TransientEnv._observation(scene, step)
                if step:
                    x, y, _ = START_POSITIONS["block_0"]
                    obs["proprio"][:3] = (
                        x - cfg["grasp_offset"], y, resting_z if step == 1 else held_z)
                return obs

        env = PickupEnv({0: {"length": 4, "terminated": False}}, {0: [0.0] * 4},
                        {0: [_privileged_state(), held, held, held]})
        batch, episodes = collect_wave(
            env, _ScriptedRunner(), _RecordingActor(), [14], gamma=0.999,
            gae_lambda=0.95, residual_scale=0.2, reward_cfg=cfg,
        )
        ep = episodes[0]
        self.assertEqual(ep["grasp_attempts"], {
            "0:block_0": {"arrivals": 1, "confirmed": 1, "steps_to_grasp": 3},
        })
        self.assertEqual(ep["grasp_steps_by_block"]["block_0"], 4)
        self.assertEqual(ep["final_score"], 0.0)
        metrics = act_online._rollout_metrics(batch, episodes, 0.2)
        self.assertEqual(metrics["grasp_arrivals"], 1)
        self.assertEqual(metrics["grasp_confirmed"], 1)
        self.assertEqual(metrics["grasp_conversion_rate"], 1.0)
        self.assertEqual(metrics["mean_steps_from_arrival_to_grasp"], 3.0)

    def test_logging_is_write_only_and_consistent_on_constant_state(self):
        scripts = {0: {"length": 2, "terminated": False},
                   1: {"length": 2, "terminated": True}}
        scores = {0: [0.0, 15.0], 1: [0.0, 0.0]}
        env = _ScriptedEnv(scripts, scores)
        runner = _ScriptedRunner()
        actor = _RecordingActor()
        batch, episodes = collect_wave(
            env, runner, actor, [0, 1], gamma=0.9, gae_lambda=0.8,
            residual_scale=0.05, reward_cfg=_zero_reward_cfg(),
        )
        # Raw residual is zero in the fixture: actions stay zero, values are the
        # feature sum. The extrema code must not perturb the training batch.
        self.assertEqual(int(torch.count_nonzero(batch.actions)), 0)
        self.assertEqual(len(batch.features), 4)
        ep0 = next(ep for ep in episodes if ep["layout_id"] == 0)
        ep1 = next(ep for ep in episodes if ep["layout_id"] == 1)
        # Core reward/score/trajectory fields are unchanged by logging.
        self.assertEqual(ep0["final_score"], 15.0)
        self.assertEqual(ep0["peak_score"], 15.0)
        self.assertEqual(ep0["length"], 2)
        self.assertTrue(ep0["truncated"])
        self.assertAlmostEqual(ep0["episode_return"], 15.0, places=6)
        self.assertEqual(ep1["final_score"], 0.0)
        self.assertTrue(ep1["terminated"])
        self.assertAlmostEqual(ep1["episode_return"], 0.0, places=6)
        # Constant privileged state: extrema equal the final-frame values.
        for ep in (ep0, ep1):
            self.assertEqual(ep["max_lift_m"], 0.0)
            self.assertEqual(ep["lift_final_m"], 0.0)
            self.assertEqual(ep["max_pair_sum"], 0.0)
            self.assertEqual(ep["min_d_min_m"], ep["d_min_final_m"])
            self.assertEqual(ep["max_min_gripper"], 1.0)
            self.assertEqual(ep["first_both_open_step"], 1)
            self.assertIsNone(ep["first_lift_step"])
            self.assertIsNone(ep["first_pair_step"])
            self.assertIsNone(ep["first_settled_step"])
            self.assertEqual(ep["stable_tower_levels_final"], 0)
            self.assertEqual(ep["milestone_return"], 0.0)

    def test_rollout_metrics_extrema_aggregates_and_json_safe(self):
        scripts = {0: {"length": 3, "terminated": False},
                   1: {"length": 2, "terminated": False}}
        scores = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0]}
        privileged = {
            0: [_lifted_state(0.795), _lifted_state(0.855), _lifted_state(0.795)],
            1: [_privileged_state(), _privileged_state()],
        }
        env = _TransientEnv(scripts, scores, privileged)
        batch, episodes = collect_wave(
            env, _ScriptedRunner(), _RecordingActor(), [0, 1],
            gamma=0.999, gae_lambda=0.95, residual_scale=0.2,
            reward_cfg=_reward_cfg(),
        )
        metrics = act_online._rollout_metrics(batch, episodes, 0.2)
        self.assertAlmostEqual(metrics["max_lift_m_over_episodes"], 0.06, places=9)
        self.assertAlmostEqual(metrics["mean_max_lift_m"], 0.03, places=9)
        self.assertEqual(metrics["episodes_with_any_lift"], 1)
        self.assertEqual(metrics["episodes_with_both_open"], 2)
        self.assertEqual(metrics["episodes_with_any_pair"], 0)
        self.assertEqual(metrics["episodes_with_any_settled"], 0)
        self.assertEqual(metrics["max_pair_sum_over_episodes"], 0.0)
        self.assertEqual(metrics["min_d_min_m_over_episodes"],
                         min(ep["min_d_min_m"] for ep in episodes))
        # metrics.jsonl is written with allow_nan=False; every value must be finite.
        json.dumps(metrics, allow_nan=False)
        for ep in episodes:
            json.dumps(ep, allow_nan=False)


if __name__ == "__main__":
    unittest.main()