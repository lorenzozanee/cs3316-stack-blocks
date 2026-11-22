"""Unit and integration tests for unfrozen ACT AWR online fine-tuning."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from starter.config import load_config
from starter.interfaces import ACTION_DIM, BLOCK_LABELS
from starter.policies import act_online
from starter.policies.act_awr import (
    ACT_AWR_KEYS,
    ActAnchor,
    ActWaveData,
    SpatialSoftmaxBlockPoseHead,
    act_awr_update,
    build_act_optimizer,
    configure_act_unfreeze,
    decode_rgb_anchor,
    encode_rgb_anchor,
    is_unused_cvae_param,
    move_optimizer_state,
    validate_act_awr_config,
)
from starter.policies.act_inference import load_act_runner
from starter.policies.act_model import ACTPolicy
from tests.test_act import small_config
from tests.test_act_online import (
    _OnlineFakeEnv,
    _online_config,
    _observation,
    _privileged_state,
)

ROOT = Path(__file__).resolve().parents[1]


class ActAwrUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_yaml_has_all_act_awr_keys_and_validates(self):
        cfg = load_config(ROOT / "configs/act_online.yaml")
        for key in ACT_AWR_KEYS:
            self.assertIn(key, cfg, f"Missing {key} in configs/act_online.yaml")
        self.assertTrue(validate_act_awr_config(cfg))

    def test_validate_act_awr_config_rejects_bad_values(self):
        base = load_config(ROOT / "configs/act_online.yaml")
        bad_bool = dict(base, unfreeze_act="yes")
        with self.assertRaises(ValueError):
            validate_act_awr_config(bad_bool)
        for key in ("act_lr", "act_backbone_lr", "act_awr_beta", "act_awr_max_weight",
                    "act_max_action_shift", "act_max_grad_norm", "act_aux_pose_lr"):
            bad = dict(base, **{key: 0.0})
            with self.assertRaises(ValueError, msg=key):
                validate_act_awr_config(bad)
        for key in ("act_epochs", "act_batch_size", "act_grad_accum_steps",
                    "act_max_samples", "act_sample_stride"):
            bad = dict(base, **{key: 0})
            with self.assertRaises(ValueError, msg=key):
                validate_act_awr_config(bad)
        with self.assertRaises(ValueError):
            validate_act_awr_config(dict(base, act_aux_warmup_epochs=-1))
        with self.assertRaises(ValueError):
            validate_act_awr_config(dict(base, act_prior_coef=-0.1))

    def test_configure_act_unfreeze_splits_inference_and_unused_cvae(self):
        policy = ACTPolicy(small_config(), pretrained_backbone=False)
        trainable, frozen_cvae = configure_act_unfreeze(policy)
        self.assertTrue(trainable)
        self.assertTrue(frozen_cvae)
        trainable_names = {name for name, _ in trainable}
        self.assertTrue(any("backbone" in name for name in trainable_names))
        self.assertTrue(any("transformer" in name for name in trainable_names))
        self.assertTrue(any("action_head" in name for name in trainable_names))
        for name, param in policy.named_parameters():
            if is_unused_cvae_param(name):
                self.assertFalse(param.requires_grad, name)
                self.assertIn(name, frozen_cvae)
            else:
                self.assertTrue(param.requires_grad, name)
                self.assertIn(name, trainable_names)

    def test_jpeg_anchor_roundtrip_preserves_shape_range_and_channels(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[:, :, 0] = 220  # strong red channel
        rgb[:, :, 1] = 40
        rgb[:, :, 2] = 15
        payload = encode_rgb_anchor(rgb)
        self.assertIsInstance(payload, bytes)
        self.assertGreater(len(payload), 0)
        decoded = decode_rgb_anchor(payload)
        self.assertEqual(decoded.shape, (3, 480, 640))
        self.assertEqual(decoded.dtype, np.float32)
        self.assertTrue(np.isfinite(decoded).all())
        # Channel order preserved: red remains dominant over green and blue.
        self.assertGreater(decoded[0].mean(), 0.75)
        self.assertLess(decoded[2].mean(), 0.20)

    def test_spatial_softmax_head_shape_and_gradients(self):
        head = SpatialSoftmaxBlockPoseHead(in_channels=32, num_keypoints=8, out_dim=9)
        fmap = torch.randn(3, 32, 6, 8, requires_grad=True)
        out = head(fmap)
        self.assertEqual(out.shape, (3, 9))
        out.sum().backward()
        self.assertIsNotNone(fmap.grad)
        self.assertTrue(torch.isfinite(fmap.grad).all())

    def test_act_awr_update_moves_active_weights_keeps_cvae_frozen_and_early_stops(self):
        torch.manual_seed(0)
        policy = ACTPolicy(small_config(), pretrained_backbone=False)
        trainable, frozen_cvae = configure_act_unfreeze(policy)
        aux_head = SpatialSoftmaxBlockPoseHead(policy.model.input_proj.out_channels)
        cfg = load_config(ROOT / "configs/act_online.yaml")
        cfg.update(
            act_lr=1e-3,
            act_backbone_lr=5e-4,
            act_epochs=2,
            act_batch_size=2,
            act_grad_accum_steps=1,
            act_max_samples=4,
            act_aux_pose_weight=0.25,
            act_aux_warmup_epochs=1,
            act_max_action_shift=10.0,
        )
        optimizer = build_act_optimizer(trainable, aux_head, cfg)
        chunk_size = policy.model.num_queries

        # Build two short synthetic episodes with distinct advantages.
        anchors = []
        executed = []
        advantages = []
        for ep in range(2):
            ep_len = chunk_size + 2
            exec_norm = np.full((ep_len, ACTION_DIM), 0.5 if ep == 0 else -0.5, dtype=np.float32)
            adv = np.full(ep_len, 2.0 if ep == 0 else -2.0, dtype=np.float32)
            executed.append(exec_norm)
            advantages.append(adv)
            img = np.full((32, 32, 3), 80 + ep * 90, dtype=np.uint8)
            qpos = np.zeros(ACTION_DIM, dtype=np.float32)
            with torch.no_grad():
                pred0 = policy(
                    torch.from_numpy(qpos[None]),
                    torch.from_numpy(decode_rgb_anchor(encode_rgb_anchor(img))[None, None]),
                )[0].numpy().astype(np.float16)
            anchors.append(
                ActAnchor(
                    episode_index=ep,
                    step_index=0,
                    image_jpeg=encode_rgb_anchor(img),
                    qpos_norm=qpos,
                    pred_old=pred0,
                    block_xyz=np.linspace(-0.2, 0.8, 9, dtype=np.float32),
                )
            )

        wave = ActWaveData(anchors=anchors, executed_norm=executed, advantages=advantages)
        metrics = act_awr_update(
            policy, aux_head, optimizer, trainable, frozen_cvae, wave,
            config=cfg, device=torch.device("cpu"), warmup_aux_epochs=1,
        )
        self.assertGreater(metrics["act_updates"], 0)
        self.assertGreater(metrics["act_param_delta_l2"], 0.0)
        self.assertFalse(metrics["act_early_stopped"])
        self.assertIsNotNone(metrics["act_aux_pose_mae_m"])
        self.assertGreater(metrics["act_weight_ess"], 0.0)
        self.assertLessEqual(metrics["act_weight_ess"], 1.0 + 1e-6)
        # Optimizer moment tensors stay on CPU after update.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    self.assertEqual(value.device.type, "cpu", key)

        # Tight shift threshold triggers early stop without stepping.
        strict_cfg = dict(cfg, act_max_action_shift=1e-9)
        strict_metrics = act_awr_update(
            policy, aux_head, optimizer, trainable, frozen_cvae, wave,
            config=strict_cfg, device=torch.device("cpu"), warmup_aux_epochs=0,
        )
        self.assertTrue(strict_metrics["act_early_stopped"])
        self.assertEqual(strict_metrics["act_updates"], 0)
        self.assertEqual(strict_metrics["act_param_delta_l2"], 0.0)
        self.assertIsNone(strict_metrics["act_awr_loss"])
        self.assertIsNone(strict_metrics["act_aux_pose_loss"])
        self.assertIsNone(strict_metrics["act_aux_pose_mae_m"])
        self.assertGreater(strict_metrics["act_prior_shift_l1"], 1e-9)


class TrainOnlineUnfreezeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _write_offline_base(self, root: Path) -> tuple[Path, dict, dict]:
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
            "action_std": np.full(ACTION_DIM, 0.25, dtype=np.float32),
        }
        with (directory / "dataset_stats.pkl").open("wb") as stream:
            pickle.dump(stats, stream)
        return directory / "policy_last.ckpt", config, base_state

    def test_resume_from_frozen_v3_into_unfrozen_act_and_resume_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            offline_ckpt, _base_cfg, initial_base_state = self._write_offline_base(root)

            # Round A (frozen V3 checkpoint, simulating Round 26 output).
            frozen_cfg = _online_config(offline_ckpt)
            frozen_cfg.update(
                unfreeze_act=False,
                finetune_updates=1,
                finetune_episodes=2,
                num_envs=2,
                run_stamp="round26-frozen",
                rollout_layout_ids=[0, 1, 2, 3],
            )
            env_a = _OnlineFakeEnv()
            with mock.patch.object(act_online, "OUTPUTS_ROOT", root / "outputs"), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=env_a):
                round26_ckpt = act_online.train_online(frozen_cfg)

            p26 = torch.load(round26_ckpt, map_location="cpu", weights_only=True)
            self.assertNotIn("base_optimizer", p26)

            # Round B (unfrozen ACT resuming from Round 26 frozen V3 checkpoint).
            unfrozen_cfg = _online_config(round26_ckpt)
            unfrozen_cfg.update(
                unfreeze_act=True,
                act_lr=1e-3,
                act_backbone_lr=5e-4,
                act_epochs=1,
                act_batch_size=2,
                act_grad_accum_steps=1,
                act_max_samples=8,
                act_sample_stride=1,
                act_max_action_shift=10.0,
                act_aux_pose_weight=0.25,
                act_aux_warmup_epochs=1,
                finetune_updates=1,
                finetune_episodes=2,
                num_envs=2,
                run_stamp="round27-unfrozen",
                rollout_layout_ids=[0, 1, 2, 3],
            )
            env_b = _OnlineFakeEnv()
            with mock.patch.object(act_online, "OUTPUTS_ROOT", root / "outputs"), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=env_b):
                round27_ckpt = act_online.train_online(unfrozen_cfg)

            p27 = torch.load(round27_ckpt, map_location="cpu", weights_only=True)
            self.assertEqual(p27["update"], 2)
            self.assertIn("base_optimizer", p27)
            self.assertIn("aux_pose_state_dict", p27)
            # Active ACT inference params actually moved; unused CVAE stayed frozen.
            changed = 0
            for key, value in initial_base_state.items():
                if is_unused_cvae_param(key):
                    torch.testing.assert_close(p27["base_state_dict"][key], value, msg=key)
                elif not torch.equal(p27["base_state_dict"][key], value):
                    changed += 1
            self.assertGreater(changed, 0)

            # Metrics JSONL records ACT AWR and aux visual pose metrics.
            records = [json.loads(line) for line in
                       (round27_ckpt.parent / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            for key in ("act_awr_loss", "act_prior_shift_l1", "act_aux_pose_mae_m",
                        "act_param_delta_l2", "act_optimization_seconds"):
                self.assertIn(key, records[0])
                self.assertIsNotNone(records[0][key])

            # Standard eval loader loads the unfrozen checkpoint strictly with updated ACT weights.
            runner, residual, scale = load_act_runner(round27_ckpt, "cpu")
            self.assertIsNotNone(residual)
            self.assertAlmostEqual(scale, frozen_cfg["residual_scale"])
            for key, value in p27["base_state_dict"].items():
                torch.testing.assert_close(runner.policy.state_dict()[key], value, msg=key)
            chunks, features = runner.predict_base(
                [dict(_observation(0, 0), blocks=_privileged_state()["blocks"])],
                [0], with_features=True,
            )
            self.assertEqual(len(chunks), 1)
            self.assertEqual(features.shape[0], 1)

            # Round C (resuming unfrozen ACT from Round 27 preserves ACT optimizer & aux head).
            resume_cfg = dict(unfrozen_cfg, base_checkpoint=str(round27_ckpt), run_stamp="round28-unfrozen")
            env_c = _OnlineFakeEnv()
            with mock.patch.object(act_online, "OUTPUTS_ROOT", root / "outputs"), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=env_c):
                round28_ckpt = act_online.train_online(resume_cfg)

            p28 = torch.load(round28_ckpt, map_location="cpu", weights_only=True)
            self.assertEqual(p28["update"], 3)
            # Optimizer step count advanced from Round 27 -> Round 28.
            steps27 = [float(s["step"]) for s in p27["base_optimizer"]["state"].values() if "step" in s]
            steps28 = [float(s["step"]) for s in p28["base_optimizer"]["state"].values() if "step" in s]
            self.assertTrue(steps27 and steps28)
            self.assertGreater(steps28[0], steps27[0])


if __name__ == "__main__":
    unittest.main()
