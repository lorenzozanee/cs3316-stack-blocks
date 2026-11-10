import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.interfaces import ARM_KEYS, ARM_WIDTHS
from starter.policies.diffusion import train_offline


def _write_episode(path, frames, offset, encoded):
    import h5py
    import numpy as np

    rows = np.zeros((frames, 16), dtype=np.float32)
    for index in range(frames):
        rows[index, 0] = offset + index / 4
        rows[index, 3:7] = [1, 0, 0, 0]
        rows[index, 7] = 0.5
        rows[index, 8] = offset + index / 4
        rows[index, 11:15] = [1, 0, 0, 0]
        rows[index, 15] = 0.5
    with h5py.File(path, "w") as handle:
        handle["vision/cam_head/colors"] = np.stack([encoded] * frames)
        cursor = 0
        for group in ("state", "action"):
            cursor = 0
            for key, width in zip(ARM_KEYS, ARM_WIDTHS):
                handle[f"{group}/{key}"] = rows[:, cursor:cursor + width]
                cursor += width


class DiffusionOfflineTest(unittest.TestCase):
    def test_requires_real_episodes(self):
        with tempfile.TemporaryDirectory() as temp, \
                self.assertRaisesRegex(ValueError, "No HDF5 episodes"):
            train_offline({"data_path": temp})

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in
            ("torch", "torchvision", "matplotlib", "cv2", "h5py", "XPolicyLab")),
        "requires the DP dependencies and pinned XPolicyLab on PYTHONPATH",
    )
    def test_train_records_curves_and_reloads_policy(self):
        import numpy as np
        import torch
        from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit

        from starter.data.diffusion import ActionWindows
        from starter.policies.diffusion_model import load_policy

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[..., 0] = 220
        encoded = np.frombuffer(encode_image_bit(image), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            for episode in range(2):
                _write_episode(root / "data" / f"episode_{episode}.hdf5", 5, episode * 10, encoded)
            dataset = ActionWindows(sorted((root / "data").glob("*.hdf5")), 8, 16, decode_image_bit)
            self.assertEqual(len(dataset), 10)
            _, _, tail, valid = dataset[3]
            self.assertEqual(valid, 2)
            np.testing.assert_array_equal(tail[:, 0], [0.75, 1, 1, 1, 1, 1, 1, 1])

            config = {
                "data_path": str(root), "seed": 0, "policy_gpu": 0,
                "diffusion": {
                    "epochs": 2, "horizon": 3, "action_steps": 2, "batch_size": 3,
                    "image_size": 16, "inference_steps": 2, "num_workers": 0,
                    "vision_dim": 16, "vision_depth": 1, "vision_heads": 2,
                    "model_dim": 32, "depth": 1, "num_heads": 2, "dropout": 0.0,
                    "checkpoint_every": 1, "probe_windows": 4, "probe_every": 1,
                },
            }
            run = root / "run"
            with patch("starter.policies.diffusion.dp_checkpoint_dir", return_value=run / "checkpoints"), \
                    patch("torch.cuda.is_available", return_value=False):
                checkpoint = train_offline(config)

            self.assertTrue((run / "config.json").is_file())
            self.assertTrue((run / "checkpoints" / "epoch_0001.ckpt").is_file())
            self.assertTrue((run / "checkpoints" / "epoch_0002.ckpt").is_file())
            self.assertEqual(checkpoint, run / "checkpoints" / "last.ckpt")
            with (run / "metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), [
                "epoch", "train_loss", "train_probe_action_mae", "lr", "grad_norm", "samples", "elapsed_s",
            ])
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["train_probe_action_mae"] for row in rows))
            with (run / "steps.csv").open() as handle:
                steps = list(csv.DictReader(handle))
            self.assertEqual(list(steps[0]), ["epoch", "step", "loss", "grad_norm", "lr", "samples"])
            self.assertEqual(len(steps), 8)
            self.assertTrue(all(float(step["loss"]) > 0 for step in steps))
            for name in ("loss_curve.png", "loss_steps.png", "probe_action_mae.png", "learning_rate.png"):
                self.assertGreater((run / "plots" / name).stat().st_size, 0)

            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for key in ("model", "policy", "epoch", "train_config", "optimizer",
                        "lr_scheduler", "raw_model", "ema_n_averaged", "rng"):
                self.assertIn(key, saved)
            self.assertEqual(saved["epoch"], 2)

            policy = load_policy(checkpoint)
            self.assertEqual(policy.n_obs_steps, 2)
            actions = policy.predict_action_chunk(image, np.full(16, 10.5))
            self.assertEqual(tuple(actions.shape), (3, 16))
            self.assertTrue(torch.isfinite(actions).all())
            self.assertTrue(((actions[..., [7, 15]] >= 0) & (actions[..., [7, 15]] <= 1)).all())
