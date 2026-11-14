"""CLI configuration handoff for ACT offline and residual online training."""
from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

from starter.pipelines import act

ROOT = Path(__file__).resolve().parents[1]


class ACTPipelineTest(unittest.TestCase):
    def test_online_merges_environment_and_ppo_config_with_explicit_base(self):
        with mock.patch.object(act, "train_online") as train:
            act.main(["online", "--config", str(ROOT / "configs/env.yaml"),
                      "--online-config", str(ROOT / "configs/act_online.yaml"),
                      "--checkpoint", "/selected/offline.ckpt"])
        config = train.call_args.args[0]
        self.assertEqual(config["base_checkpoint"], "/selected/offline.ckpt")
        self.assertEqual(config["data_task"], "stack_blocks")
        self.assertEqual(config["camera_names"], ["cam_head"])
        self.assertEqual(config["finetune_episodes"] % config["num_envs"], 0)
        self.assertEqual(config["residual_scale"], 0.05)
        self.assertEqual(config["lr"], 3e-4)

    def test_online_requires_checkpoint_before_starting_training(self):
        with mock.patch.object(act, "train_online") as train, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                act.main(["online"])
        train.assert_not_called()

    def test_offline_uses_only_offline_configuration(self):
        with mock.patch.object(act, "train_offline") as train:
            act.main(["offline", "--config", str(ROOT / "configs/act.yaml")])
        self.assertEqual(train.call_args.args[0]["hidden_dim"], 512)
        self.assertNotIn("base_checkpoint", train.call_args.args[0])

    def test_offline_rejects_online_options(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            act.main(["offline", "--checkpoint", "/selected/offline.ckpt"])


if __name__ == "__main__":
    unittest.main()
