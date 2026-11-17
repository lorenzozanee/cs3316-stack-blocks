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
sys.path.insert(0, str(ROOT / "src"))

from policies.act import deploy as deploy_adapter
from policies.act import model as adapter
from starter.interfaces import BLOCK_LABELS
from starter.policies.act_inference import FrozenACTRunner


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


class FakeFuncParser:
    def __init__(self, poses):
        self.poses = poses

    def get_label_pose(self, label):
        return (
            [env_poses[label]["pos"] for env_poses in self.poses],
            [env_poses[label]["rot"] for env_poses in self.poses],
        )


def block_pose(env_index, label_index):
    return {
        "pos": np.array([0.1 * label_index, float(env_index), 2.0], dtype=np.float64),
        "rot": np.array([1.0, 0.0, 0.0, 0.01 * label_index], dtype=np.float64),
    }


class FakeRobotManager:
    """Measured gripper joints per env; the command cache is not read."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.calls = []
        self.robot_list = [SimpleNamespace(type="target", gripper_scale=(0.0, 1.0))]

    def get_end_effector_real_val(self, robot, env_idx_list=None):
        self.calls.append(list(env_idx_list))
        return {env_index: [0.1 * env_index] for env_index in env_idx_list}


class FakeLayoutManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.velocity_calls = []
        self.table_info = [
            {"height": 0.765 + 0.01 * index, "pos": [0.0, -0.05, 0.74],
             "size": [-0.7, -0.55, 0.7, 0.55]}
            for index in range(num_envs)
        ]

    def get_instance_metadata(self, env_idx, label):
        BLOCK_LABELS.index(label)
        side = 0.03 + 0.005 * env_idx
        return {"geometry": {"oriented_bbox": {"extents": [side] * 3}}}

    def get_instance_name(self, env_index, label):
        return f"{label}_env{env_index}"

    def get_scene_object(self, env_index, inst_name):
        self.velocity_calls.append((env_index, inst_name))
        index = BLOCK_LABELS.index(inst_name.split("_env")[0])
        return SimpleNamespace(
            scale=[1.0, 1.0, 1.0],
            get_linear_velocity=lambda: np.array([0.0, 0.0, float(env_index)]),
            get_angular_velocity=lambda: np.array([float(index), 0.0, 0.0]),
        )


class FakeSceneManager:
    def __init__(self, num_envs):
        self.layout_manager = FakeLayoutManager(num_envs)


class FakeEvaluationTask:
    """Minimal per-env task exposing RoboDojo's label-pose interface."""

    num_envs = 3

    def __init__(self):
        self.poses = [
            {label: block_pose(env_index, label_index)
             for label_index, label in enumerate(BLOCK_LABELS)}
            for env_index in range(self.num_envs)
        ]
        self.reward_manager = SimpleNamespace(func_parser=FakeFuncParser(self.poses))
        self.robot_manager = FakeRobotManager(self.num_envs)
        self.scene_manager = FakeSceneManager(self.num_envs)
        self.running = [0, 1, 2]
        self.returned = []

    def get_obs(self):
        return self.get_obs_batch([0])[0]

    def get_obs_batch(self, env_idx_list=None, last_frame=False):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        self.returned = [
            {"env_idx": int(env_index), "marker": f"env-{int(env_index)}"}
            for env_index in env_idx_list
        ]
        return self.returned

    def get_running_env_idx_list(self):
        return list(self.running)


class DeployProxyTest(unittest.TestCase):
    def test_privileged_poses_follow_env_index_and_leave_frames_unchanged(self):
        task = FakeEvaluationTask()
        proxy = deploy_adapter.PrivilegedObsTask(task)
        frames = proxy.get_obs_batch([2, 0])
        originals = task.returned
        self.assertEqual([frame["env_idx"] for frame in frames], [2, 0])
        for frame, original, env_index in zip(frames, originals, (2, 0)):
            self.assertIsNot(frame, original)
            self.assertNotIn("privileged", original)
            blocks = frame["privileged"]["blocks"]
            self.assertEqual(list(blocks), list(BLOCK_LABELS))
            for label in BLOCK_LABELS:
                np.testing.assert_allclose(blocks[label]["pos"], task.poses[env_index][label]["pos"])
                np.testing.assert_allclose(blocks[label]["rot"], task.poses[env_index][label]["rot"])
            # Measured openness and velocities follow the frame's env_index,
            # not its position in the returned batch.
            self.assertAlmostEqual(frame["privileged"]["gripper_open_measured"][0], 0.1 * env_index)
            self.assertAlmostEqual(frame["privileged"]["table"]["top_z"], 0.765 + 0.01 * env_index)
            for extents in frame["privileged"]["block_extents"].values():
                np.testing.assert_allclose(extents, [0.03 + 0.005 * env_index] * 3)
            velocities = frame["privileged"]["block_velocities"]
            self.assertEqual(list(velocities), list(BLOCK_LABELS))
            for label in BLOCK_LABELS:
                np.testing.assert_allclose(velocities[label]["linear"], [0.0, 0.0, float(env_index)])
                np.testing.assert_allclose(
                    velocities[label]["angular"], [float(BLOCK_LABELS.index(label)), 0.0, 0.0])
        # A shrunk batch renumbers list positions; env 0 must still use env 0.
        shrunk = proxy.get_obs_batch([0])
        np.testing.assert_allclose(
            shrunk[0]["privileged"]["blocks"]["block_0"]["pos"], task.poses[0]["block_0"]["pos"],
        )
        single = proxy.get_obs()
        np.testing.assert_allclose(
            single["privileged"]["blocks"]["block_0"]["pos"], task.poses[0]["block_0"]["pos"],
        )
        self.assertEqual(proxy.get_running_env_idx_list(), task.running)

    def test_standard_loops_receive_the_proxy_and_call_upstream(self):
        task = FakeEvaluationTask()
        client = object()
        with mock.patch.object(deploy_adapter, "_eval_one_episode") as upstream:
            deploy_adapter.eval_one_episode(task, client)
        proxy, passed_client = upstream.call_args.args
        self.assertIsInstance(proxy, deploy_adapter.PrivilegedObsTask)
        self.assertIs(proxy._task, task)
        self.assertIs(passed_client, client)

        with mock.patch.object(deploy_adapter, "_eval_one_episode_batch") as upstream_batch:
            deploy_adapter.eval_one_episode_batch(task, client)
        proxy_batch, passed_client = upstream_batch.call_args.args
        self.assertIsInstance(proxy_batch, deploy_adapter.PrivilegedObsTask)
        self.assertIs(proxy_batch._task, task)
        self.assertIs(passed_client, client)
        self.assertIn("privileged", proxy_batch.get_obs())


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
        self.stats = stats
        self.network = RecordingPolicy()

    def make_model(self, temporal=True, residual=False):
        cfg = {"bench_name": "RoboDojo", "task_name": "stack_blocks", "env_cfg_type": "arx_x5",
               "action_type": "ee", "checkpoint_path": str(self.checkpoint),
               "device": "cpu", "temporal_agg": temporal}
        runner = FrozenACTRunner(self.network, self.stats, "cpu", temporal_agg=temporal)
        loaded = (runner, object(), 0.1) if residual else (runner, None, None)
        with mock.patch.object(adapter, "load_act_runner", return_value=loaded), \
             mock.patch.object(adapter, "get_robot_action_dim_info", return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}):
            return adapter.Model(cfg)

    def test_offline_does_not_require_privileged_poses(self):
        policy = self.make_model(temporal=False)
        with_poses = observation(0, 3.0)
        policy.update_obs(with_poses)
        expected = policy.get_action()
        self.network.calls = 0
        without_poses = observation(0, 3.0)
        del without_poses["privileged"]
        policy.update_obs(without_poses)
        actual = policy.get_action()
        self.assertNotIn("blocks", policy.observations[0])
        for expected_chunk, actual_chunk in zip(expected, actual):
            for key in expected_chunk:
                np.testing.assert_allclose(actual_chunk[key], expected_chunk[key])

    def test_online_requires_and_attaches_privileged_block_poses(self):
        policy = self.make_model(residual=True)
        with self.assertRaisesRegex(ValueError, "privileged"):
            policy.update_obs(observation(0, 3.0))
        obs = observation(0, 3.0)
        obs["privileged"] = {"blocks": {
            label: {"pos": np.arange(3, dtype=np.float32) + index,
                    "rot": np.array([1.0, 0.0, 0.0, 0.1 * index], dtype=np.float32)}
            for index, label in enumerate(BLOCK_LABELS)
        }}
        policy.update_obs(obs)
        stored = policy.observations[0]["blocks"]
        self.assertEqual(list(stored), list(BLOCK_LABELS))
        np.testing.assert_allclose(stored["block_1"]["pos"], np.arange(3) + 1)
        obs["privileged"]["blocks"]["block_2"]["rot"] = [1.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "rot"):
            policy.update_obs(obs)

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
        self.assertEqual(len(policy.runner.histories[0]), 1)
        self.assertEqual(len(policy.runner.histories[2]), 2)
        policy.reset()
        self.assertEqual(policy.runner.histories, {})
        self.assertEqual(policy.observations, {})

    def test_temporal_history_is_bounded_and_aligns_current_step(self):
        policy = self.make_model()
        policy.update_obs(observation(0, 3))
        for _ in range(5):
            action = policy.get_action()
        weights = np.exp(-0.01 * np.arange(3))
        expected = 2 * np.dot([23, 32, 41], weights / weights.sum())
        self.assertAlmostEqual(float(action[0]["left_ee_pose"][0]), expected, places=5)
        self.assertEqual(len(policy.runner.histories[0]), 3)

    def test_nonfinite_predictions_fail_before_sending_actions(self):
        policy = self.make_model()
        policy.update_obs(observation(0, 3))
        with mock.patch.object(self.network, "forward", return_value=torch.full((1, 3, 16), float("nan"))):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                policy.get_action()


if __name__ == "__main__":
    unittest.main()
