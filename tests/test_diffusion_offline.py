import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.interfaces import ARM_KEYS, ARM_WIDTHS
from starter.policies.diffusion import train_offline


class DiffusionOfflineTest(unittest.TestCase):
    def test_requires_real_episodes(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "at least two HDF5 episodes"):
                train_offline({"data_path": temp})

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("torch", "diffusers", "cv2", "h5py", "XPolicyLab")),
        "requires the DP dependencies and pinned XPolicyLab on PYTHONPATH",
    )
    def test_train_and_reload_action_sampler(self):
        import h5py
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
                with h5py.File(root / "data" / f"episode_{episode}.hdf5", "w") as handle:
                    handle["vision/cam_head/colors"] = np.stack([encoded] * 5)
                    for group in ("state", "action"):
                        for key, width in zip(ARM_KEYS, ARM_WIDTHS):
                            values = episode * 10 + np.arange(5, dtype=np.float32)[:, None] / 4
                            handle[f"{group}/{key}"] = np.repeat(values, width, axis=1)
            dataset = ActionWindows(sorted((root / "data").glob("*.hdf5")), 8, 16, decode_image_bit)
            self.assertEqual(len(dataset), 10)
            rgb, _, tail, valid = dataset[3]
            self.assertEqual(valid, 2)
            np.testing.assert_array_equal(tail[:, 0], [0.75, 1, 1, 1, 1, 1, 1, 1])
            np.testing.assert_array_equal(dataset[5][2][:, 0], [10, 10.25, 10.5, 10.75, 11, 11, 11, 11])
            self.assertGreater(rgb[0].mean(), rgb[2].mean())

            config = {
                "data_path": str(root), "seed": 0, "policy_gpu": 0,
                "diffusion": {"epochs": 2, "horizon": 3, "batch_size": 3, "image_size": 16, "inference_steps": 4},
            }
            with patch("starter.policies.diffusion.dp_checkpoint_dir", return_value=root / "checkpoints"):
                with patch("torch.cuda.is_available", return_value=False):
                    checkpoint = train_offline(config)

            policy = load_policy(checkpoint)
            self.assertTrue(torch.all(policy.action_min == 10))
            self.assertTrue(torch.all(policy.action_max == 11))
            self.assertEqual(policy.inference_steps, 4)
            actions = policy.predict_action_chunk(image, np.full(16, 10.5))
            self.assertEqual(tuple(actions.shape), (3, 16))
            self.assertTrue(torch.isfinite(actions).all())
            self.assertTrue(((actions >= 10) & (actions <= 11)).all())
            with (root / "metrics.csv").open() as metrics:
                rows = list(csv.DictReader(metrics))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(math.isfinite(float(row["val_action_mae"])) for row in rows))
            best = min(rows, key=lambda row: float(row["val_action_mae"]))
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["epoch"], int(best["epoch"]))
            self.assertEqual(saved["val_action_mae"], float(best["val_action_mae"]))
            self.assertEqual(saved["train_config"]["val_sample_windows"], 5)
