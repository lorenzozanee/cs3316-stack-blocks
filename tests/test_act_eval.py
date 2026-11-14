"""ACT evaluation normalization, batching and temporal aggregation without Isaac."""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from policies.act import model as adapter


class RecordingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(camera_names=["cam_head"], num_queries=3)
        self.calls = 0
        self.inputs = None

    def forward(self, qpos, image):
        self.inputs = (qpos.clone(), image.clone())
        value = qpos[:, :1, None] + 10 * self.calls + torch.arange(3)[None, :, None]
        self.calls += 1
        result = value.expand(-1, -1, 16).clone()
        result[:, :, 7] = 2
        result[:, :, 15] = -2
        return result


def observation(index, position):
    return {
        "env_idx": index,
        "state": {"left_ee_pose": np.full(7, position), "left_ee_joint_state": [0.5],
                  "right_ee_pose": np.full(7, position), "right_ee_joint_state": [0.5]},
        "vision": {"cam_head": {"color": np.full((8, 8, 3), [255, 0, 64], dtype=np.uint8)},
                   "cam_left_wrist": {"color": object()}},
        "privileged": {"block_0": object()},
    }


class ACTEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.checkpoint = root / "policy_last.ckpt"
        self.checkpoint.touch()
        stats = {"qpos_mean": np.ones(16), "qpos_std": np.full(16, 2.0),
                 "action_mean": np.zeros(16), "action_std": np.full(16, 2.0)}
        with (root / "dataset_stats.pkl").open("wb") as stream:
            pickle.dump(stats, stream)
        self.network = RecordingPolicy()

    def make_model(self, temporal=True):
        cfg = {"bench_name": "RoboDojo", "task_name": "stack_blocks", "env_cfg_type": "arx_x5",
               "action_type": "ee", "checkpoint_path": str(self.checkpoint),
               "device": "cpu", "temporal_agg": temporal}
        with mock.patch.object(adapter, "load_offline_policy", return_value=self.network), \
             mock.patch.object(adapter, "get_robot_action_dim_info", return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}):
            return adapter.Model(cfg)

    def test_rgb_normalization_and_ee_action_contract(self):
        policy = self.make_model(temporal=False)
        policy.update_obs(observation(0, 3.0))
        actions = policy.get_action()
        qpos, image = self.network.inputs
        self.assertEqual(tuple(image.shape), (1, 1, 3, 480, 640))
        np.testing.assert_allclose(image[0, 0, :, 0, 0].numpy(), [1, 0, 64 / 255])
        self.assertEqual(float(qpos[0, 0]), 1.0)
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(qpos.dtype, torch.float32)
        self.assertEqual(len(actions), 3)
        self.assertEqual(set(actions[0]), {"left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state"})
        np.testing.assert_allclose(actions[0]["left_ee_pose"], np.full(7, 2.0))
        np.testing.assert_array_equal(actions[0]["left_ee_joint_state"], [1.0])
        np.testing.assert_array_equal(actions[0]["right_ee_joint_state"], [0.0])

    def test_temporal_history_keeps_environment_ids_when_batch_shrinks(self):
        policy = self.make_model()
        policy.update_obs_batch([observation(0, 3), observation(2, 5)])
        first = policy.get_action_batch([2, 0])
        self.assertEqual(float(first[0][0]["left_ee_pose"][0]), 4.0)
        self.assertEqual(float(first[1][0]["left_ee_pose"][0]), 2.0)
        policy.update_obs_batch([observation(2, 11)])
        second = policy.get_action_batch([2])
        expected = 2 * (3 + 15 * np.exp(-0.01)) / (1 + np.exp(-0.01))
        self.assertAlmostEqual(float(second[0][0]["left_ee_pose"][0]), expected, places=5)
        self.assertEqual(len(second[0]), 1)
        self.assertEqual(len(policy.histories[0]), 1)
        self.assertEqual(len(policy.histories[2]), 2)
        policy.reset()
        self.assertEqual(policy.histories, {})
        self.assertEqual(policy.observations, {})

    def test_temporal_history_is_bounded_and_aligns_current_step(self):
        policy = self.make_model()
        policy.update_obs(observation(0, 3))
        for _ in range(5):
            action = policy.get_action()
        weights = np.exp(-0.01 * np.arange(3))
        expected = 2 * np.dot([23, 32, 41], weights / weights.sum())
        self.assertAlmostEqual(float(action[0]["left_ee_pose"][0]), expected, places=5)
        self.assertEqual(len(policy.histories[0]), 3)

    def test_nonfinite_predictions_fail_before_sending_actions(self):
        policy = self.make_model()
        policy.update_obs(observation(0, 3))
        with mock.patch.object(self.network, "forward", return_value=torch.full((1, 3, 16), float("nan"))):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                policy.get_action()


if __name__ == "__main__":
    unittest.main()
