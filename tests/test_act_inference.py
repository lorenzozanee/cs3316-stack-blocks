"""Shared ACT inference: frozen features, composition and checkpoint loading."""

from __future__ import annotations

import json
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

from policies.act import model as adapter
from starter.policies import act_inference
from starter.policies.act_inference import (
    FEATURE_DIM,
    FrozenACTRunner,
    compose_residual_action,
)

STATS = {
    "qpos_mean": np.ones(16, dtype=np.float32),
    "qpos_std": np.full(16, 2.0, dtype=np.float32),
    "action_mean": np.zeros(16, dtype=np.float32),
    "action_std": np.full(16, 2.0, dtype=np.float32),
}


class CountingBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        batch = image.shape[0]
        channels = torch.arange(512, dtype=image.dtype, device=image.device)
        features = channels.view(1, 512, 1, 1).expand(batch, 512, 2, 2)
        return features, torch.zeros_like(features)


class FeaturePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            camera_names=["cam_head"],
            num_queries=3,
            backbones=torch.nn.ModuleList([CountingBackbone()]),
        )

    def forward(self, qpos, image):
        self.model.backbones[0](image[:, 0])
        value = qpos[:, :1, None] + torch.arange(3)[None, :, None]
        result = value.expand(-1, -1, 16).clone()
        result[:, :, 7] = 2
        result[:, :, 15] = -2
        return result


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image):
        batch = image.shape[0]
        features = self.weight.view(1, 1, 1, 1).expand(batch, 512, 1, 1)
        return features, features


class TinyPolicy(torch.nn.Module):
    def __init__(self, config=None, *, pretrained_backbone=True):
        super().__init__()
        self.model = SimpleNamespace(
            camera_names=["cam_head"],
            num_queries=2,
            backbones=torch.nn.ModuleList([TinyBackbone()]),
        )


class FixedResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deterministic = None
        self.feature_shape = None

    def act(self, features, deterministic=False):
        self.deterministic = deterministic
        self.feature_shape = tuple(features.shape)
        raw = torch.ones(features.shape[0], 16, dtype=features.dtype)
        zeros = raw.sum(-1) * 0.0
        return raw, zeros, zeros


class TinyResidual(torch.nn.Module):
    def __init__(self, feature_dim, action_dim=16, hidden_dim=8, initial_std=0.1):
        super().__init__()
        self.mean = torch.nn.Linear(feature_dim, action_dim)

    def act(self, features, deterministic=False):
        raw = self.mean(features)
        zeros = raw.sum(-1) * 0.0
        return raw, zeros, zeros


def block_poses(x=0.1):
    return {label: {"pos": [x, 0.2, 0.8], "rot": [1.0, 0.0, 0.0, 0.0]}
            for label in ("block_0", "block_1", "block_2")}


def flat_observation(position=3.0, image_value=255):
    return {
        "image": np.full((8, 8, 3), image_value, dtype=np.uint8),
        "proprio": np.full(16, position, dtype=np.float32),
        "camera": "cam_head",
        "blocks": block_poses(),
    }


class FrozenRunnerTest(unittest.TestCase):
    def setUp(self):
        self.policy = FeaturePolicy()
        self.runner = FrozenACTRunner(
            self.policy, STATS, "cpu", temporal_agg=True, capture_features=True,
        )

    def test_features_use_the_frozen_backbone_without_another_forward(self):
        chunks, features = self.runner.predict_base(
            [flat_observation()], [7], with_features=True,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape, (1, 16))
        self.assertEqual(tuple(features.shape), (1, FEATURE_DIM))
        self.assertFalse(features.requires_grad)
        # One image pass produced both the base action and the visual feature.
        self.assertEqual(self.policy.model.backbones[0].calls, 1)
        np.testing.assert_allclose(features[0, :16].numpy(), np.ones(16))
        expected_action = np.ones(16)
        expected_action[[7, 15]] = [2.0, -2.0]
        np.testing.assert_allclose(features[0, 16:32].numpy(), expected_action)
        np.testing.assert_allclose(features[0, 32:544].numpy(), np.arange(512))
        np.testing.assert_allclose(features[0, 544:].numpy(),
                                   np.tile([0.1, 0.2, 0.8, 1, 0, 0, 0], 3))
        # The base remains frozen.
        self.assertFalse(self.policy.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.policy.parameters()))

    def test_object_poses_change_residual_features_but_not_base_actions(self):
        first = flat_observation()
        second = flat_observation()
        second["blocks"] = block_poses(x=0.4)
        chunks, features = self.runner.predict_base([first, second], [0, 1], with_features=True)
        np.testing.assert_array_equal(chunks[0], chunks[1])
        torch.testing.assert_close(features[0, :544], features[1, :544])
        self.assertFalse(torch.equal(features[0, 544:], features[1, 544:]))
        self.assertEqual(self.policy.model.backbones[0].calls, 1)

    def test_missing_or_invalid_object_poses_fail_for_residual_only(self):
        observation = flat_observation()
        del observation["blocks"]
        self.runner.predict_base([observation], [0], with_features=False)
        with self.assertRaisesRegex(ValueError, "block poses"):
            self.runner.predict_base([observation], [0], with_features=True)
        for pose, message in (({"pos": [0, 0, 0], "rot": [0, 0, 0, 0]}, "near zero"),
                              ({"pos": [float("nan"), 0, 0], "rot": [1, 0, 0, 0]}, "non-finite")):
            observation = flat_observation()
            observation["blocks"]["block_0"] = pose
            with self.assertRaisesRegex(ValueError, message):
                self.runner.predict_base([observation], [0], with_features=True)

    def test_feature_capture_requires_temporal_aggregation(self):
        runner = FrozenACTRunner(
            self.policy, STATS, "cpu", temporal_agg=False, capture_features=True,
        )
        with self.assertRaisesRegex(ValueError, "temporal_agg"):
            runner.predict_base([flat_observation()], [0], with_features=True)

        offline = FrozenACTRunner(
            self.policy, STATS, "cpu", temporal_agg=True, capture_features=False,
        )
        with self.assertRaisesRegex(ValueError, "capture_features"):
            offline.predict_base([flat_observation()], [0], with_features=True)

    def test_nonfinite_inputs_and_actions_are_rejected(self):
        bad_proprio = flat_observation()
        bad_proprio["proprio"] = np.full(16, np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.runner.predict_base([bad_proprio], [0])

        bad_image = flat_observation()
        bad_image["image"] = np.full((8, 8, 3), np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.runner.predict_base([bad_image], [0])

        with mock.patch.object(
            self.policy, "forward", return_value=torch.full((1, 3, 16), float("nan")),
        ):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.runner.predict_base([flat_observation()], [0])


class ComposeResidualTest(unittest.TestCase):
    def test_residual_is_scaled_clipped_and_contract_is_restored(self):
        base = np.zeros((2, 16), dtype=np.float32)
        base[:, 3:7] = [1, 0, 0, 0]
        base[:, 11:15] = [1, 0, 0, 0]
        raw = torch.full((2, 16), 5.0)
        composed = compose_residual_action(base, raw, np.full(16, 0.5), 0.2)
        np.testing.assert_allclose(composed[:, :3], 0.1)
        np.testing.assert_allclose(composed[:, (7, 15)], 0.1)
        for start in (3, 11):
            quaternion = composed[:, start : start + 4]
            np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1.0)
        self.assertTrue(np.isfinite(composed).all())

    def test_pose_positions_are_not_clipped(self):
        base = np.zeros((1, 16), dtype=np.float32)
        base[:, :3] = 100.0
        base[:, 3:7] = [1, 0, 0, 0]
        base[:, 11:15] = [1, 0, 0, 0]
        composed = compose_residual_action(
            base, np.zeros((1, 16), dtype=np.float32), np.ones(16), 0.1,
        )
        np.testing.assert_allclose(composed[:, :3], 100.0)

    def test_degenerate_quaternion_and_nonfinite_values_raise(self):
        base = np.zeros((1, 16), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "quaternion"):
            compose_residual_action(base, np.zeros((1, 16)), np.ones(16), 0.1)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compose_residual_action(base, np.full((1, 16), np.nan), np.ones(16), 0.1)
        with self.assertRaisesRegex(ValueError, "shape"):
            compose_residual_action(base, np.zeros((2, 16)), np.ones(16), 0.1)


def robot_observation(index, privileged):
    state = {"left_ee_pose": np.full(7, 3.0), "left_ee_joint_state": [0.5],
             "right_ee_pose": np.full(7, 3.0), "right_ee_joint_state": [0.5]}
    return {
        "env_idx": index,
        "state": state,
        "vision": {"cam_head": {"color": np.full((8, 8, 3), [255, 0, 64], dtype=np.uint8)}},
        "privileged": privileged,
    }


class OnlineAdapterTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "policy_last.ckpt"
        self.checkpoint.touch()
        self.policy = FeaturePolicy()
        self.runner = FrozenACTRunner(
            self.policy, STATS, "cpu", temporal_agg=True, capture_features=True,
        )
        self.residual = FixedResidual()
        self.residual_scale = 0.1

    def make_model(self):
        cfg = {"bench_name": "RoboDojo", "task_name": "stack_blocks", "env_cfg_type": "arx_x5",
               "action_type": "ee", "checkpoint_path": str(self.checkpoint),
               "device": "cpu", "temporal_agg": True}
        with mock.patch.object(
            adapter, "load_act_runner",
            return_value=(self.runner, self.residual, self.residual_scale),
        ), mock.patch.object(
            adapter, "get_robot_action_dim_info",
            return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]},
        ):
            return adapter.Model(cfg)

    def test_online_action_is_deterministic_and_keeps_base_bounds(self):
        model = self.make_model()
        model.update_obs(robot_observation(0, {"blocks": block_poses()}))
        actions = model.get_action()
        self.assertTrue(self.residual.deterministic)
        self.assertEqual(self.residual.feature_shape, (1, FEATURE_DIM))
        # get_action returns the one physical action selected for this scene.
        self.assertEqual(len(actions), 1)
        action = actions[0]
        np.testing.assert_allclose(action["left_ee_pose"][:3], 2.2)
        np.testing.assert_allclose(
            np.linalg.norm(action["left_ee_pose"][3:7]), 1.0,
        )
        np.testing.assert_allclose(
            np.linalg.norm(action["right_ee_pose"][3:7]), 1.0,
        )
        self.assertGreaterEqual(float(action["left_ee_joint_state"][0]), 0.0)
        self.assertLessEqual(float(action["left_ee_joint_state"][0]), 1.0)
        self.assertEqual(float(action["right_ee_joint_state"][0]), 0.0)

    def test_privileged_state_reaches_only_residual_features(self):
        model = self.make_model()
        model.update_obs_batch([
            robot_observation(0, {"blocks": block_poses(x=0.1)}),
            robot_observation(1, {"blocks": block_poses(x=0.4)}),
        ])
        self.assertNotEqual(float(model.observations[0]["blocks"]["block_0"]["pos"][0]),
                            float(model.observations[1]["blocks"]["block_0"]["pos"][0]))
        actions = model.get_action_batch([0, 1])
        # This fixture's residual outputs a constant, so ACT actions stay equal.
        np.testing.assert_allclose(
            actions[0][0]["left_ee_pose"], actions[1][0]["left_ee_pose"],
        )


class LoadRunnerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _write_config(self, config):
        (self.root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def test_offline_checkpoint_uses_saved_stats_without_features(self):
        import pickle

        checkpoint = self.root / "policy_last.ckpt"
        checkpoint.touch()
        self._write_config({"camera_names": ["cam_head"]})
        with (self.root / "dataset_stats.pkl").open("wb") as stream:
            pickle.dump(STATS, stream)
        policy = TinyPolicy()
        with mock.patch.object(act_inference, "load_offline_policy", return_value=policy):
            runner, residual, scale = act_inference.load_act_runner(
                checkpoint, "cpu", temporal_agg=False,
            )
        self.assertIsNone(residual)
        self.assertIsNone(scale)
        self.assertFalse(runner.capture_features)
        self.assertFalse(runner.temporal_agg)

    def _write_online(self, **overrides):
        checkpoint = self.root / "policy_last.ckpt"
        self._write_config({"policy_type": "act_residual_ppo", "temporal_agg": True})
        residual_config = {"feature_dim": FEATURE_DIM, "hidden_dim": 8, "initial_std": 0.1}
        payload = {
            "format": "act_residual_ppo_v3",
            "base_config": {},
            "base_state_dict": TinyPolicy().state_dict(),
            "residual_config": residual_config,
            "residual_state_dict": TinyResidual(**residual_config).state_dict(),
            "stats": {key: torch.from_numpy(value) for key, value in STATS.items()},
            "residual_scale": 0.25,
            "temporal_agg": True,
        }
        payload.update(overrides)
        torch.save(payload, checkpoint)
        return checkpoint

    def test_online_checkpoint_loads_frozen_base_and_residual(self):
        checkpoint = self._write_online()
        with mock.patch.object(act_inference, "ACTPolicy", TinyPolicy), \
                mock.patch.object(act_inference, "ResidualActorCritic", TinyResidual):
            runner, residual, scale = act_inference.load_act_runner(checkpoint, "cpu")
        self.assertIsInstance(residual, TinyResidual)
        self.assertEqual(scale, 0.25)
        self.assertTrue(runner.capture_features)
        self.assertTrue(runner.temporal_agg)
        self.assertFalse(residual.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in runner.policy.parameters()))

    def test_checkpoint_symlink_uses_target_directory_metadata(self):
        checkpoint = self._write_online()
        links = self.root / "useful_checkpoints"
        links.mkdir()
        link = links / "selected.ckpt"
        link.symlink_to(checkpoint)
        with mock.patch.object(act_inference, "ACTPolicy", TinyPolicy), \
                mock.patch.object(act_inference, "ResidualActorCritic", TinyResidual):
            runner, residual, scale = act_inference.load_act_runner(link, "cpu")
        self.assertIsInstance(residual, TinyResidual)
        self.assertEqual(scale, 0.25)
        self.assertTrue(runner.capture_features)

    def test_online_rejects_format_and_temporal_mismatch(self):
        # The independent actor/critic v3 architecture never migrates the old
        # shared-trunk v2 checkpoint silently.
        checkpoint = self._write_online(format="act_residual_ppo_v2")
        with mock.patch.object(act_inference, "ACTPolicy", TinyPolicy), \
                mock.patch.object(act_inference, "ResidualActorCritic", TinyResidual):
            with self.assertRaisesRegex(ValueError, "format"):
                act_inference.load_act_runner(checkpoint, "cpu")

        checkpoint = self._write_online()
        with mock.patch.object(act_inference, "ACTPolicy", TinyPolicy), \
                mock.patch.object(act_inference, "ResidualActorCritic", TinyResidual):
            with self.assertRaisesRegex(ValueError, "temporal_agg"):
                act_inference.load_act_runner(checkpoint, "cpu", temporal_agg=False)

    def test_unknown_policy_type_is_rejected(self):
        checkpoint = self.root / "policy_last.ckpt"
        checkpoint.touch()
        self._write_config({"policy_type": "something_else"})
        with self.assertRaisesRegex(ValueError, "policy_type"):
            act_inference.load_act_runner(checkpoint, "cpu")


if __name__ == "__main__":
    unittest.main()
