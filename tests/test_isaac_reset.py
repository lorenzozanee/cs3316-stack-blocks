"""Online resets restore pooled scene objects before initializing rewards."""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from starter.envs.isaac_batch import bind_eval_layouts, install_room_fixture_cleanup
from starter.envs.stack_blocks import ParallelAdapter
from tests.test_env_data import _BatchFake


class RoomFixtureCleanupTest(unittest.TestCase):
    def test_cleanup_matches_room_spawn_root_for_each_scene(self):
        first = "/World/envs/env_0/Rooms"
        second = "/World/envs/env_1/Rooms"
        existing = {first, second}
        other_deletions = []
        manager = SimpleNamespace(
            env_roots=["/World/envs/env_0", "/World/envs/env_1"],
            delete_fixture_prims=lambda index, name: other_deletions.append((index, name)),
        )
        prims = SimpleNamespace(is_prim_path_valid=existing.__contains__, delete_prim=existing.remove)
        with patch.dict(sys.modules, {"isaacsim.core.utils.prims": prims}):
            install_room_fixture_cleanup(manager)
        manager.delete_fixture_prims(0, "Room")
        manager.delete_fixture_prims(0, "Room")
        self.assertEqual(existing, {second})
        manager.delete_fixture_prims(1, "Room")
        self.assertEqual(existing, set())
        manager.delete_fixture_prims(0, "Table")
        self.assertEqual(other_deletions, [(0, "Table")])


class SavedLayoutResetTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.saved = {}
        self.positions = [None, None]
        task = self.task = _BatchFake()

        def reload(seed=None, options=None):
            task.reset_seeds.append(seed)
            self.positions[:] = [100000.0, 100000.0]
            self.events.append("reload")

        def restore(env_idx_list):
            self.events.append("restore")
            for index in env_idx_list:
                self.positions[index] = self.saved[index]["position"]

        def robot_init():
            self.assertTrue(all(position in (0.1, 0.2) for position in self.positions))
            self.events.append("robot_init")

        def reward_init():
            self.assertTrue(all(position in (0.1, 0.2) for position in self.positions))
            self.events.append("reward_init")

        task.reset = reload
        def stability(env):
            self.assertIs(env, task)
            self.assertEqual(task.success, [True, True])
            self.events.append("stability")
            return True, []

        task.scene_manager = SimpleNamespace(
            layout_manager=SimpleNamespace(
                set_saved_layout=lambda index, info: self.saved.__setitem__(index, info),
                layout_valid=[True, True],
                check_layout_stability=stability,
            ),
            apply_saved_poses=restore,
        )
        task.online_obs = SimpleNamespace(
            reset=lambda: self.events.append("obs_reset"),
            render_for_capture=lambda: self.events.append("capture"),
            get_obs=lambda: self.events.append("warmup_observation"),
        )
        task.render = lambda: self.events.append("render")
        task.sim_step = lambda: self.events.append("physics")
        task.robot_manager = SimpleNamespace(
            set_origin_endpose=lambda: self.events.append("origin"),
            set_robot_init_state=robot_init,
        )
        task.reward_manager.init_state = reward_init
        manager = SimpleNamespace(
            seed_list=[5, 9], init_eval=lambda: None,
            get_seed_scene_info=lambda layout: {"position": {5: 0.1, 9: 0.2}[layout]},
        )
        seed_module = SimpleNamespace(SeedManager=lambda config: manager)
        with patch.dict(sys.modules, {"env.seed_manager.seed_manager": seed_module}):
            self.assertEqual(bind_eval_layouts(task, "stack_blocks", "arx_x5"), 2)
        self.adapter = ParallelAdapter(task)

    def test_every_reset_restores_selected_objects_before_reward_initialization(self):
        for seeds, expected in (([0, 1], [0.1, 0.2]), ([3, 2], [0.2, 0.1])):
            observations, infos = self.adapter.reset_batch(seeds)
            self.assertEqual(self.positions, expected)
            self.assertEqual([info["seed"] for info in infos], seeds)
            self.assertEqual(len(observations), 2)
        sequence = ["reload", "obs_reset", "restore", "stability"] + ["render"] * 10
        for step in range(200):
            sequence.append("physics")
            if step % 5 == 0:
                sequence.extend(["capture", "warmup_observation"])
        sequence.extend(["origin", "robot_init", "reward_init"])
        self.assertEqual(self.events, sequence * 2)
        self.assertEqual(self.task.reset_seeds, [[0, 1], [3, 2]])

    def test_invalid_or_unstable_layout_never_reaches_policy_or_reward_init(self):
        manager = self.task.scene_manager.layout_manager
        manager.layout_valid[1] = False
        with self.assertRaisesRegex(RuntimeError, "Invalid.*1"):
            self.adapter.reset_batch([0, 1])
        self.assertNotIn("robot_init", self.events)
        self.assertNotIn("reward_init", self.events)

        manager.layout_valid[1] = True
        self.events.clear()
        manager.check_layout_stability = lambda env: (True, [1])
        with self.assertRaisesRegex(RuntimeError, "Unstable.*1"):
            self.adapter.reset_batch([0, 1])
        self.assertNotIn("robot_init", self.events)
        self.assertNotIn("reward_init", self.events)

    def test_invalid_seed_count_does_not_reset_or_restore(self):
        with self.assertRaisesRegex(ValueError, "Expected 2 seeds"):
            self.task.reset(seed=[0])
        self.assertEqual(self.events, [])

    def test_failed_restore_does_not_initialize_rewards_or_return_observations(self):
        def fail(env_idx_list):
            raise RuntimeError("scene restore failed")

        self.task.scene_manager.apply_saved_poses = fail
        with self.assertRaisesRegex(RuntimeError, "scene restore failed"):
            self.adapter.reset_batch([0, 1])
        self.assertEqual(self.events, ["reload", "obs_reset"])


if __name__ == "__main__":
    unittest.main()
