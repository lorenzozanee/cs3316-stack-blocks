"""Environment and data contracts. Policy training is not implemented."""

from __future__ import annotations

import socket
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from starter.config import load_config, parse_simple_yaml
from starter.data.demos import hold_demos
from starter.envs.fake import FakeTask
from starter.envs.stack_blocks import ParallelAdapter, StackBlocksAdapter
from starter.interfaces import ACTION_DIM, BLOCK_LABELS, pack_ee_action, unpack_ee_action
from starter.policies.act import train_offline as act_offline
from starter.policies.act import train_online as act_online
from starter.policies.diffusion import train_offline as diffusion_offline
from starter.policies.diffusion import train_online as diffusion_online
from starter.rewards.intrinsic import combine_reward

ROOT = Path(__file__).resolve().parents[1]


class _FakeRobot:
    """Target robot exposing the measured-gripper API used by the official check."""

    def __init__(self, type_: str, scale: tuple, measured: list) -> None:
        self.type = type_
        self.gripper_scale = scale
        self._measured = measured

    def measured(self, env_index: int):
        return self._measured[env_index]


class _FakeRobotManager:
    """Two target arms plus one non-target entry to prove ordering/filtering."""

    def __init__(self) -> None:
        self.robot_list = [
            _FakeRobot("target", (-0.01, 0.044), [[-0.01], [0.044]]),
            _FakeRobot("follower", (-0.01, 0.044), [[0.044], [0.044]]),
            _FakeRobot("target", (0.0, 0.05), [[0.025], [0.0]]),
        ]
        self.calls = []

    def get_end_effector_real_val(self, robot, env_idx_list=None):
        self.calls.append((robot.type, list(env_idx_list)))
        return {env_idx: robot.measured(env_idx) for env_idx in env_idx_list}


class _FakeBlock:
    def __init__(self, linear, angular) -> None:
        self.scale = [1.0, 1.0, 1.0]
        self._linear = np.asarray(linear, dtype=np.float64)
        self._angular = np.asarray(angular, dtype=np.float64)

    def get_linear_velocity(self):
        return self._linear

    def get_angular_velocity(self):
        return self._angular


class _FakeLayoutManager:
    def __init__(self, num_envs: int) -> None:
        self._num_envs = num_envs
        self.table_info = [
            {"height": 0.765, "pos": [0.0, -0.05, 0.715],
             "size": [-0.7, -0.55, 0.7, 0.55]}
            for _ in range(num_envs)
        ]
        self._instances = {
            (env_index, label): f"block_{index}_env{env_index}"
            for env_index in range(num_envs)
            for index, label in enumerate(BLOCK_LABELS)
        }
        self.velocity_calls = []

    def get_instance_metadata(self, env_idx, label):
        self.get_instance_name(env_idx, label)
        return {"geometry": {"oriented_bbox": {"extents": [0.03, 0.03, 0.03]}}}

    def get_instance_name(self, env_index, label):
        return self._instances[(env_index, label)]

    def get_scene_object(self, env_index, inst_name):
        self.velocity_calls.append((env_index, inst_name))
        index = int(inst_name.split("_")[1])
        return _FakeBlock(
            linear=[float(env_index), float(index), 0.5],
            angular=[0.0, float(index), -float(env_index)],
        )


class _FakeSceneManager:
    def __init__(self, num_envs: int) -> None:
        self.layout_manager = _FakeLayoutManager(num_envs)


def _multi_env_task(num_envs: int, score: float = 0.0) -> FakeTask:
    """FakeTask whose label poses exist for every scene index."""
    task = FakeTask(score=score)
    task.num_envs = num_envs
    task.robot_manager = _FakeRobotManager()
    task.scene_manager = _FakeSceneManager(num_envs)

    def poses(label):
        del label
        return (
            [[0.0, 0.0, 0.1 * env_index] for env_index in range(num_envs)],
            [[1.0, 0.0, 0.0, 0.0] for _ in range(num_envs)],
        )

    task.reward_manager.func_parser.get_label_pose = poses
    return task


class _BatchFake(FakeTask):
    def __init__(self) -> None:
        super().__init__(step_lim=4, score=0.0)
        self.num_envs = 2
        self.batch_calls = []
        self.reward_manager.get_reward = lambda final_check=False: [0.0, 0.0]
        self.reward_manager.get_score = lambda reward_lst=None: [0.0, 15.0]

        def poses(label):
            del label
            return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]], [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]

        self.reward_manager.func_parser.get_label_pose = poses
        self.robot_manager = _FakeRobotManager()
        self.scene_manager = _FakeSceneManager(self.num_envs)
    def reset(self, seed=None, options=None) -> None:
        del options
        self.reset_seeds.append(seed)

    def get_obs_batch(self, env_idx_list=None):
        return [self.get_obs() for _ in env_idx_list]

    def take_action_batch(self, actions, env_idx_list=None) -> None:
        self.batch_calls.append((list(env_idx_list), len(actions)))


class ObservationContractTest(unittest.TestCase):
    def test_control_motion_uses_each_slots_clock_and_resets_its_history(self):
        from unittest.mock import patch

        task = _BatchFake()
        time = 1.0
        angle = 0.0
        warmup_displacement = 0.002
        pose_reads = []

        def poses(label):
            pose_reads.append(label)
            return ([[0.0, 0.0, 0.0], [angle, 0.0, 0.1]],
                    [[-1.0 if time > 1.0 else 1.0, 0.0, 0.0, 0.0],
                     [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]])

        task.reward_manager.func_parser.get_label_pose = poses
        env = ParallelAdapter(task, physics_clock=lambda: time)
        reset = task.reset

        def reset_with_warmup(**kwargs):
            nonlocal angle, time
            reset(**kwargs)
            angle = 0.0
            env.sample_reset_motion()
            time += 0.04
            angle = warmup_displacement

        task.reset = reset_with_warmup
        observations, _ = env.reset_batch([14, 15])
        scene0, scene1 = env.privileged_batch([0, 1])
        self.assertEqual(scene0["settle_speeds"]["block_0"], [0, 0])
        np.testing.assert_allclose(scene1["settle_speeds"]["block_0"], [0.05, 0.05])
        self.assertEqual(pose_reads, list(BLOCK_LABELS) * 2)
        actions = [observation["proprio"] for observation in observations]
        interval = 0.04

        def advance(actions, indices):
            nonlocal time, angle
            time += interval
            angle += 0.005

        task.take_action_batch = advance
        with patch.object(task.reward_manager, "get_reward", wraps=task.reward_manager.get_reward) as scorer:
            result = env.step_batch(actions, [0, 1])
            scorer.assert_called_once_with(final_check=False)
        self.assertEqual([row[1] for row in result], [0.0, 15.0])
        scene1, scene0 = env.privileged_batch([1, 0])
        self.assertEqual(scene0["settle_speeds"]["block_0"], [0.0, 0.0])
        np.testing.assert_allclose(scene1["settle_speeds"]["block_0"], [0.125, 0.125])
        self.assertEqual(env.privileged_batch([1])[0]["settle_speeds"], scene1["settle_speeds"])
        self.assertEqual(scene1["block_velocities"]["block_0"]["linear"][0], 1.0)

        interval = 0.02
        env.step_batch([actions[0]], [0])
        self.assertEqual(env.privileged_batch([0])[0]["settle_speeds"]["block_0"], [0, 0])
        self.assertEqual(env.privileged_batch([1])[0]["settle_speeds"], scene1["settle_speeds"])
        env.step_batch([actions[1]], [1])
        np.testing.assert_allclose(env.privileged_batch([1])[0]["settle_speeds"]["block_0"], [0.25, 0.25])
        time = 0.0
        warmup_displacement = 0.0
        env.reset_batch([16, 17])
        for state in env.privileged_batch([0, 1]):
            self.assertEqual(state["settle_speeds"]["block_0"], [0, 0])

    def test_reset_marks_plain_task_scenes_active(self):
        from starter.envs.stack_blocks import _seed_gripper_state

        class _Robot:
            arm_name = "left_arm"

        class _RobotManager:
            def __init__(self) -> None:
                self.robot_list = [_Robot()]
                self.init_calls = 0

            def set_origin_endpose(self) -> None:
                return None

            def set_robot_init_state(self) -> None:
                self.init_calls += 1

            def get_real_endpose(self, robot):
                del robot
                return [
                    [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                    [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0],
                ]

        class _Reward:
            def __init__(self) -> None:
                self.calls = 0
                self.func_parser = type("Parser", (), {"robot_origin_endpose": [{}, {}]})()

            def init_state(self) -> None:
                self.calls += 1

        class _Task:
            num_envs = 2

            def __init__(self) -> None:
                self.robot_manager = _RobotManager()
                self.reward_manager = _Reward()

        task = _Task()
        _seed_gripper_state(task)
        self.assertEqual(task.success, [True, True])
        self.assertEqual(task.reward_manager.calls, 1)
        task.success[0] = False
        _seed_gripper_state(task)
        self.assertEqual(task.success, [True, True])

        class _EvalTask(_Task):
            def __init__(self) -> None:
                super().__init__()
                self.end_flag = [False, False]
                self.success = [False, True]

        eval_task = _EvalTask()
        _seed_gripper_state(eval_task)
        self.assertEqual(eval_task.success, [False, True])

    def test_policy_view_drops_wrists_and_block_pose(self):
        task = FakeTask(score=15.0)
        task.robot_manager = _FakeRobotManager()
        task.scene_manager = _FakeSceneManager(1)
        adapter = StackBlocksAdapter(task)
        observation, _info = adapter.reset(seed=3)
        self.assertEqual(observation["camera"], "cam_head")
        self.assertEqual(len(observation["proprio"]), ACTION_DIM)
        self.assertNotIn("block_0", observation)
        self.assertNotIn("cam_left_wrist", observation)
        privileged = adapter.get_privileged_state()
        self.assertIn("block_0", privileged["blocks"])
        self.assertNotIn("blocks", observation)

    def test_privileged_geometry_roundtrips_into_reward_without_policy_leak(self):
        import json
        from starter.envs.remote import _pack
        from starter.policies.stacking_reward import REWARD_KEYS, StackingReward

        task = _multi_env_task(2)
        task.scene_manager.layout_manager.table_info[1]["height"] = 0.8
        raw_observation = task.get_obs

        def valid_observation():
            raw = raw_observation()
            for key in ("left_ee_pose", "right_ee_pose"):
                raw["state"][key] = [0.3, 0.3, 0.9, 1.0, 0.0, 0.0, 0.0]
            return raw

        task.get_obs = valid_observation
        adapter = StackBlocksAdapter(task, env_index=1)
        observation, _ = adapter.reset(seed=0)
        physical = json.loads(json.dumps(_pack(adapter.get_privileged_state()), allow_nan=False))
        self.assertEqual(physical["table"]["top_z"], 0.8)
        self.assertEqual(physical["block_extents"],
                         {label: [0.03, 0.03, 0.03] for label in BLOCK_LABELS})
        config = load_config(ROOT / "configs/act_online.yaml")
        reward = StackingReward(physical, observation["proprio"],
                                {key: config[key] for key in REWARD_KEYS})
        terms, bonus = reward.step(observation["proprio"], physical)
        self.assertTrue(np.isfinite(terms["phi"]))
        self.assertEqual(bonus, 0.0)
        for key in ("table", "block_extents", "blocks"):
            self.assertNotIn(key, observation)

    def test_measured_gripper_openness_uses_joints_not_command(self):
        # Command cache in the observation reads 0.0 for both jaws, while the
        # measured joint values encode a closed left jaw (0.0) and a half-open
        # right jaw (0.5). Only the measured path may be reported.
        task = _multi_env_task(1)
        adapter = StackBlocksAdapter(task)
        observation, _info = adapter.reset(seed=0)
        self.assertEqual(observation["proprio"][7], 0.0)
        self.assertEqual(observation["proprio"][15], 0.0)
        measured = adapter.get_privileged_state()["gripper_open_measured"]
        # Target robots in robot_list order: first maps -0.01 -> 0.0, second
        # maps 0.025 in [0.0, 0.05] -> 0.5. The non-target entry is skipped.
        self.assertEqual(len(measured), 2)
        self.assertAlmostEqual(measured[0], 0.0, places=6)
        self.assertAlmostEqual(measured[1], 0.5, places=6)
        self.assertEqual(task.robot_manager.calls, [("target", [0]), ("target", [0])])

    def test_measured_openness_follows_env_index(self):
        task = _multi_env_task(2)
        adapter = StackBlocksAdapter(task, env_index=1)
        measured = adapter.get_privileged_state()["gripper_open_measured"]
        # env 1: first target 0.044 -> 1.0, second 0.0 -> 0.0.
        self.assertAlmostEqual(measured[0], 1.0, places=6)
        self.assertAlmostEqual(measured[1], 0.0, places=6)
        self.assertEqual(task.robot_manager.calls, [("target", [1]), ("target", [1])])

    def test_block_velocities_follow_labels_and_env_index(self):
        task = _multi_env_task(2)
        adapter = StackBlocksAdapter(task, env_index=1)
        velocities = adapter.get_privileged_state()["block_velocities"]
        self.assertEqual(list(velocities), list(BLOCK_LABELS))
        for index, label in enumerate(BLOCK_LABELS):
            self.assertEqual(velocities[label]["linear"], [1.0, float(index), 0.5])
            self.assertEqual(velocities[label]["angular"], [0.0, float(index), -1.0])
        # Object reads serve measured velocities and scaled geometry, in label order.
        self.assertEqual(task.scene_manager.layout_manager.velocity_calls,
                         [(1, f"block_{index}_env1") for index in range(len(BLOCK_LABELS))] * 2)

    def test_privileged_state_requires_measured_apis(self):
        task = FakeTask(score=0.0)
        with self.assertRaises(AttributeError):
            StackBlocksAdapter(task).get_privileged_state()
        task.robot_manager = _FakeRobotManager()
        with self.assertRaises(AttributeError):
            StackBlocksAdapter(task).get_privileged_state()
        task.scene_manager = _FakeSceneManager(1)
        task.scene_manager.layout_manager.get_scene_object = lambda env_index, inst_name: object()
        with self.assertRaises(AttributeError):
            StackBlocksAdapter(task).get_privileged_state()

    def test_step_returns_external_score_unchanged(self):
        task = FakeTask(step_lim=4, score=15.0)
        adapter = StackBlocksAdapter(task)
        adapter.reset(seed=0)
        action = [0.1] * ACTION_DIM
        action[7] = 1.5
        _obs, reward, terminated, truncated, info = adapter.step(action)
        self.assertEqual(reward, 15.0)
        self.assertEqual(info["external_score"], 15.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(task.reward_manager.score, 15.0)
        self.assertEqual(combine_reward(reward, 3.0), 18.0)
        self.assertEqual(task.actions[0]["left_ee_joint_state"], [1.0])

    def test_parallel_reset_registers_global_reward_checks_once(self):
        task = _BatchFake()
        registrations = []
        task.run_reward = lambda: registrations.append("reward")
        task.get_score = lambda: registrations.append("score")
        ParallelAdapter(task).reset_batch([14, 15])
        self.assertEqual(registrations, ["reward", "score"])

    def test_parallel_step_runs_every_active_scene(self):
        task = _BatchFake()
        adapter = ParallelAdapter(task)
        observations, infos = adapter.reset_batch([3, 4])
        self.assertEqual(len(observations), 2)
        self.assertEqual([item["seed"] for item in infos], [3, 4])
        action = [0.0] * ACTION_DIM
        results = adapter.step_batch([action, action], [0, 1])
        self.assertEqual(results[1][1], 15.0)
        self.assertEqual(task.batch_calls, [([0, 1], 2)])

    def test_block_pose_rejects_zero_or_nonfinite_quaternions(self):
        from starter.envs.stack_blocks import _block_pose

        bad_zero = {label: {"pos": [0.0, 0.0, 0.8], "rot": [0.0, 0.0, 0.0, 0.0]} for label in BLOCK_LABELS}
        bad_nan = {label: {"pos": [0.0, 0.0, 0.8], "rot": [float("nan"), 0.0, 0.0, 1.0]} for label in BLOCK_LABELS}
        with self.assertRaises(ValueError):
            _block_pose(bad_zero)
        with self.assertRaises(ValueError):
            _block_pose(bad_nan)

    def test_parallel_adapter_guards_missing_clock_and_motion_baseline(self):
        task = _multi_env_task(1)
        no_clock = ParallelAdapter(task)
        no_clock.sample_reset_motion()

        clock_adapter = ParallelAdapter(task, physics_clock=lambda: 0.04)
        with self.assertRaisesRegex(RuntimeError, "Missing motion baseline"):
            clock_adapter.privileged_batch([0])

    def test_step_limit_truncates_without_success(self):
        task = FakeTask(step_lim=2, score=15.0)
        adapter = StackBlocksAdapter(task)
        adapter.reset(seed=1)
        action = [0.0] * ACTION_DIM
        _obs, _reward, terminated, truncated, _info = adapter.step(action)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        _obs, _reward, terminated, truncated, info = adapter.step(action)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "timeout")

    def test_only_run_reward_success_terminates(self):
        # EvalEnv.is_episode_end ignores the public 0/15/100 score; a partial
        # or even a 100-point score does not end the episode by itself.
        task = FakeTask(step_lim=4, score=100.0)
        adapter = StackBlocksAdapter(task)
        adapter.reset(seed=1)
        _obs, reward, terminated, _truncated, info = adapter.step([0.0] * ACTION_DIM)
        self.assertEqual(reward, 100.0)
        self.assertFalse(terminated)
        self.assertFalse(info["success"])
        self.assertEqual(info["termination_reason"], "running")

        task.reward_manager.success_reward = 1.0
        _obs, _reward, terminated, _truncated, info = adapter.step([0.0] * ACTION_DIM)
        self.assertTrue(terminated)
        self.assertTrue(info["success"])
        self.assertEqual(info["termination_reason"], "success")


class ActionPackTest(unittest.TestCase):
    def test_round_trip(self):
        state = {
            "left_ee_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            "left_ee_joint_state": [0.4],
            "right_ee_pose": [0.5, 0.6, 0.7, 1.0, 0.0, 0.0, 0.0],
            "right_ee_joint_state": [0.8],
        }
        self.assertEqual(unpack_ee_action(pack_ee_action(state)), state)


class DataAndConfigTest(unittest.TestCase):
    def test_hold_demos_use_the_ee_width(self):
        observation, action = hold_demos(count=2, seed=0)[0]
        self.assertEqual(len(action), ACTION_DIM)
        self.assertEqual(len(observation), ACTION_DIM + 1)

    def test_environment_configs_omit_policy_hyperparameters(self):
        env = load_config(ROOT / "configs/env.yaml")
        evaluation = load_config(ROOT / "configs/eval.yaml")
        self.assertEqual(env["rollout_task"], "stack_blocks")
        self.assertEqual(env["data_task"], "stack_blocks")
        self.assertEqual(env["eval_task"], "stack_blocks")
        if env["policy_gpu"] == env["env_gpu"]:
            online = load_config(ROOT / "configs/act_online.yaml")
            self.assertTrue(online["allow_shared_gpu"])
        self.assertGreaterEqual(env["policy_gpu"], 0)
        self.assertGreaterEqual(env["env_gpu"], 0)
        self.assertEqual(evaluation["task"], "stack_blocks")
        self.assertEqual(evaluation["action_type"], "ee")
        blob = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "configs/env.yaml", ROOT / "configs/eval.yaml")
        )
        for key in ("learning_rate", "clip:", "denoising", "residual_std", "ppo_epochs", "gamma:"):
            self.assertNotIn(key, blob)
        image_env = (ROOT / "configs/sim_image.env").read_text(encoding="utf-8")
        self.assertIn(
            "SIM_IMAGE_URL=https://huggingface.co/datasets/NantianHe/stack-blocks-images/resolve/main/stack-blocks-sim_isaacsim5.1.tar",
            image_env,
        )
        self.assertIn(
            "POLICY_IMAGE_URL=https://huggingface.co/datasets/NantianHe/stack-blocks-images/resolve/main/stack-blocks-policy_dp-act.tar",
            image_env,
        )

    def test_inline_comment_does_not_join_the_value(self):
        parsed = parse_simple_yaml("num_envs: 4  # at most 5\n")
        self.assertEqual(parsed["num_envs"], 4)


class PolicySlotTest(unittest.TestCase):
    def test_remaining_policy_training_is_unimplemented(self):
        for function in (diffusion_offline, diffusion_online):
            with self.assertRaises(NotImplementedError):
                function({})

    def test_act_requires_explicit_configuration(self):
        for function in (act_offline, act_online):
            with self.subTest(function=function.__name__), self.assertRaises(KeyError):
                function({})

    def test_act_rejects_missing_real_demonstrations(self):
        import tempfile

        config = load_config(ROOT / "configs/act.yaml")
        with tempfile.TemporaryDirectory() as directory:
            config["data_path"] = directory
            with self.assertRaises(FileNotFoundError):
                act_offline(config)


    def test_act_rejects_incomplete_demonstration_download(self):
        import tempfile

        config = load_config(ROOT / "configs/act.yaml")
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            data.mkdir()
            (data / "episode_0000000.hdf5").touch()
            config["data_path"] = directory
            with self.assertRaisesRegex(ValueError, "Expected all 100 ACT episodes, found 1"):
                act_offline(config)


class RemoteImageParityTest(unittest.TestCase):
    def test_camera_transport_preserves_every_rgb_pixel(self):
        import numpy as np
        from starter.envs.remote import _pack_image, _unpack_image

        # Fine colour boundaries expose JPEG chroma subsampling; frozen ACT
        # must see the same pixels as the official lossless websocket path.
        image = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        np.testing.assert_array_equal(_unpack_image(_pack_image(image)), image)


class RemoteResetTimeoutTest(unittest.TestCase):
    def test_recycle_between_waves_retries_the_same_layouts_on_a_fresh_server(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, call, patch
        from starter.envs.remote import RemoteEnv, serve

        first_server, first_client = socket.socketpair()
        second_server, second_client = socket.socketpair()
        first_listener, second_listener = Mock(), Mock()
        first_listener.accept.return_value = (first_server, None)
        second_listener.accept.return_value = (second_server, None)
        first = SimpleNamespace(
            num_envs=2,
            reset_batch=Mock(return_value=([{}, {}], [{}, {}])),
            step_batch=Mock(side_effect=lambda _actions, active: [
                ({}, 15.0, False, True, {}) for _ in active
            ]),
            privileged_batch=Mock(side_effect=lambda active: [{} for _ in active]),
        )
        second = SimpleNamespace(
            num_envs=2,
            reset_batch=Mock(return_value=([{}, {}], [{"layout": 18}, {"layout": 19}])),
            privileged_batch=Mock(return_value=[{}, {}]),
        )
        recycle = Mock(side_effect=[False, True])
        fresh_recycle = Mock(return_value=True)
        env = RemoteEnv("127.0.0.1", 9, reset_timeout=8)
        first_thread = threading.Thread(target=serve, args=(first, "127.0.0.1", 9),
                                        kwargs={"should_recycle": recycle})
        second_thread = threading.Thread(target=serve, args=(second, "127.0.0.1", 9),
                                         kwargs={"should_recycle": fresh_recycle})
        with (patch("starter.envs.remote.socket.socket", return_value=first_listener),
              patch("starter.envs.remote.socket.create_connection",
                    side_effect=[first_client, second_client]) as connect):
            first_thread.start()
            try:
                env.reset_batch([14, 15])
                recycle.assert_not_called()
                self.assertTrue(env.step_batch([[0.0]], [0])[0][3])
                self.assertTrue(env.step_batch([[0.0]], [1])[0][3])
                env.reset_batch([16, 17])
                recycle.assert_called_once_with()
                env.step_batch([[0.0], [0.0]], [0, 1])
                with patch("starter.envs.remote.socket.socket", return_value=second_listener):
                    second_thread.start()
                    observations, infos = env.reset_batch([18, 19])
                self.assertEqual(observations, [{}, {}])
                self.assertEqual(infos, [{"layout": 18}, {"layout": 19}])
                self.assertEqual(connect.call_count, 2)
                self.assertEqual(env._request({"op": "close"}), {"ok": True})
            finally:
                env.close()
                for sock in (first_client, second_client):
                    sock.close()
                for thread in (first_thread, second_thread):
                    if thread.ident is not None:
                        thread.join(timeout=2)
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(first.reset_batch.call_args_list, [call([14, 15]), call([16, 17])])
        self.assertEqual(recycle.call_count, 2)
        second.reset_batch.assert_called_once_with([18, 19])
        fresh_recycle.assert_not_called()
        first_listener.close.assert_called_once()
        second_listener.close.assert_called_once()

    def test_server_reset_error_reaches_learner_without_reconnect(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from starter.envs.remote import RemoteEnv, serve

        server, client = socket.socketpair()
        listener = Mock()
        listener.accept.return_value = (server, None)
        task = SimpleNamespace(num_envs=2, reset_batch=Mock(
            side_effect=RuntimeError("Unstable training layouts in scenes [1], layout ids [15]."),
        ))
        env = RemoteEnv("127.0.0.1", 9, reset_timeout=60)
        env._socket = client
        errors = []

        def run_server():
            try:
                serve(task, "127.0.0.1", 9)
            except RuntimeError as error:
                errors.append(error)

        with patch("starter.envs.remote.socket.socket", return_value=listener):
            thread = threading.Thread(target=run_server)
            thread.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "Unstable.*layout ids.*15"):
                    env.reset_batch([14, 15])
            finally:
                env.close()
                thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        task.reset_batch.assert_called_once_with([14, 15])
        self.assertEqual(len(errors), 1)
        listener.close.assert_called_once()

    def test_reset_waits_past_the_step_timeout(self):
        from starter.envs.remote import RemoteEnv

        server, client = socket.socketpair()
        env = RemoteEnv("127.0.0.1", 9, timeout=0.05, reset_timeout=1.0)
        env._socket = client

        def respond() -> None:
            incoming = b""
            while b"\n" not in incoming:
                incoming += server.recv(4096)
            time.sleep(0.2)
            server.sendall(b'{"observations":[{}],"infos":[{}],"privileged":[{}]}\n')

        thread = threading.Thread(target=respond)
        thread.start()
        try:
            observations, _infos = env.reset_batch([0])
        finally:
            thread.join(timeout=2)
            server.close()
            env.close()
        self.assertEqual(observations, [{}])

    def test_reset_timeout_names_the_wait(self):
        from starter.envs.remote import RemoteEnv

        _server, client = socket.socketpair()
        env = RemoteEnv("127.0.0.1", 9, timeout=0.05, reset_timeout=0.15)
        env._socket = client
        try:
            with self.assertRaises(ConnectionError) as caught:
                env.reset_batch([0])
        finally:
            _server.close()
            env.close()
        self.assertIn("did not complete reset within", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
