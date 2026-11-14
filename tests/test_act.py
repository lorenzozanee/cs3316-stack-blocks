"""Model parity and an end-to-end ACT offline training check."""

from __future__ import annotations

import csv
import importlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torchvision.transforms.functional import normalize

from starter.config import load_config
from starter.policies.act import load_offline_policy, train_offline
from starter.policies.act_model import ACTPolicy
from tests.test_act_data import _solid, _write_episode

ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = load_config(ROOT / "configs/act.yaml")
    config.update(hidden_dim=32, dim_feedforward=64, enc_layers=1, dec_layers=2,
                  nheads=4, chunk_size=4, pretrained_backbone=False,
                  batch_size=2, num_workers=0, num_epochs=1, save_freq=1,
                  num_episodes=3, device="cpu")
    return config


def _write_training_episodes(root: Path) -> None:
    rng = np.random.RandomState(10)
    for index, length in enumerate((5, 6, 7)):
        _write_episode(root, f"episode_{index}.hdf5",
                       rng.normal(size=(length, 16)).astype(np.float32),
                       rng.normal(size=(length, 16)).astype(np.float32),
                       [_solid((40, 80, 120)) for _ in range(length)])


class ACTModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_baseline_parameter_count_and_buffers(self):
        config = load_config(ROOT / "configs/act.yaml")
        with torch.device("meta"):
            policy = ACTPolicy(config, pretrained_backbone=False)
        self.assertEqual(sum(p.numel() for p in policy.parameters()), 83_901_585)
        self.assertTrue(all(p.requires_grad for p in policy.parameters()))
        self.assertEqual(sum(t.numel() for t in policy.state_dict().values()), 83_947_409)
        self.assertNotIn("model.backbones.0.0.body.fc.weight", policy.state_dict())
        self.assertNotIn("model.backbones.0.0.body.bn1.weight", dict(policy.named_parameters()))
        self.assertIn("model.backbones.0.0.body.bn1.weight", dict(policy.named_buffers()))

    def test_loss_retains_padding_in_denominator(self):
        policy = ACTPolicy(small_config(), pretrained_backbone=False)
        prediction = torch.full((1, 4, 16), 2.0)
        latent = torch.zeros(1, 32)
        with mock.patch.object(policy.model, "forward", return_value=(prediction, None, (latent, latent))):
            losses = policy(torch.zeros(1, 16), torch.zeros(1, 1, 3, 32, 32),
                            torch.zeros(1, 4, 16), torch.tensor([[False, False, True, True]]))
        self.assertEqual(float(losses["l1"]), 1.0)
        self.assertEqual(float(losses["kl"]), 0.0)
        self.assertEqual(float(losses["loss"]), 1.0)

    def test_inference_is_deterministic_and_optimizer_groups_match(self):
        config = small_config()
        config["lr_backbone"] = 2e-6
        policy = ACTPolicy(config, pretrained_backbone=False).eval()
        image, state = torch.rand(2, 1, 3, 64, 64), torch.rand(2, 16)
        with torch.inference_mode():
            first, second = policy(state, image), policy(state, image)
        self.assertEqual(first.shape, (2, 4, 16))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        optimizer = policy.configure_optimizers(config)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 2e-6])
        backbone_ids = {id(p) for p in policy.model.backbones.parameters()}
        self.assertEqual({id(p) for p in optimizer.param_groups[1]["params"]}, backbone_ids)

    def test_model_matches_pinned_upstream_forward_gradient_and_update(self):
        # Load only upstream model code; bypass its CLI, data conversion and CUDA-only builder.
        source = ROOT / "third_party/XPolicyLab/policy/ACT/detr"
        with mock.patch.object(sys, "path", [str(source), *sys.path]):
            upstream = importlib.import_module("models.detr_vae")
            backbone = importlib.import_module("models.backbone")
        config = small_config()
        args = SimpleNamespace(**config, backbone="resnet18", dilation=False, masks=False,
                               position_embedding="sine", pre_norm=False)
        torch.manual_seed(7)
        with mock.patch.dict("os.environ", {"ACT_ACTION_DIM": "16"}), \
                mock.patch.object(backbone, "is_main_process", return_value=False):
            reference = upstream.build(args).train()
        torch.manual_seed(7)
        actual = ACTPolicy(config, pretrained_backbone=False).train()
        self.assertEqual(set(actual.model.state_dict()), set(reference.state_dict()))
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(actual.model.state_dict()[name], value, rtol=0, atol=0, msg=name)
        image, state = torch.rand(2, 1, 3, 64, 64), torch.rand(2, 16)
        actions = torch.randn(2, 6, 16)
        mask = torch.tensor([[False, False, False, True, True, True], [False] * 6])
        normalized = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        torch.manual_seed(29)
        predicted, _, (mu, logvar) = reference(state, normalized, None, actions[:, :4], mask[:, :4])
        expected_l1 = ((predicted - actions[:, :4]).abs() * ~mask[:, :4, None]).mean()
        expected_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
        expected_loss = expected_l1 + config["kl_weight"] * expected_kl
        torch.manual_seed(29)
        actual_losses = actual(state, image, actions, mask)
        for key, expected in (("l1", expected_l1), ("kl", expected_kl), ("loss", expected_loss)):
            torch.testing.assert_close(actual_losses[key], expected, rtol=0, atol=0)
        expected_loss.backward()
        actual_losses["loss"].backward()
        for name, parameter in reference.named_parameters():
            other = dict(actual.model.named_parameters())[name]
            if parameter.grad is None:
                self.assertIsNone(other.grad, name)
            else:
                torch.testing.assert_close(other.grad, parameter.grad, rtol=0, atol=0, msg=name)
        # The baseline reads the first decoder layer, but retains later layers in the graph.
        later_grad = actual.model.transformer.decoder.layers[1].linear1.weight.grad
        self.assertIsNotNone(later_grad)
        self.assertEqual(torch.count_nonzero(later_grad).item(), 0)
        ref_optimizer = torch.optim.AdamW([
            {"params": [p for n, p in reference.named_parameters() if "backbone" not in n]},
            {"params": [p for n, p in reference.named_parameters() if "backbone" in n], "lr": config["lr_backbone"]},
        ], lr=config["lr"], weight_decay=config["weight_decay"])
        ref_optimizer.step()
        actual.configure_optimizers(config).step()
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(dict(actual.model.named_parameters())[name], parameter,
                                       rtol=0, atol=0, msg=name)
        actual.eval()
        reference.eval()
        with torch.inference_mode():
            torch.testing.assert_close(actual(state, image), reference(state, normalized, None)[0], rtol=0, atol=0)


class ACTTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_raw_hdf5_training_checkpoint_and_reload(self):
        config = small_config()
        rng = np.random.RandomState(10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, length in enumerate((5, 6, 7)):
                _write_episode(root, f"episode_{index}.hdf5",
                               rng.normal(size=(length, 16)).astype(np.float32),
                               rng.normal(size=(length, 16)).astype(np.float32),
                               [_solid((40, 80, 120)) for _ in range(length)])
            config["data_path"] = str(root)
            output = root / "run"
            with mock.patch("starter.policies.act.act_output_dir", return_value=output):
                train_offline(config)
            expected_files = {"config.json", "metadata.json", "metrics.jsonl", "metrics.csv", "dataset_stats.pkl",
                              "policy_epoch_1_seed_0.ckpt", "policy_last.ckpt", "training_state.pt",
                              "training_curves.png", "training_curves.svg"}
            self.assertEqual({p.name for p in output.iterdir()}, expected_files)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(len(metadata["train_episodes"]), 3)
            self.assertNotIn("val_episodes", metadata)
            self.assertEqual(metadata["steps_per_epoch"], 2)
            records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["epoch"], 1)
            self.assertTrue(np.isfinite(records[0]["train"]["loss"]))
            self.assertGreater(records[0]["elapsed_seconds"], 0)
            self.assertGreater(records[0]["epoch_seconds"], 0)
            self.assertEqual(records[0]["eta_seconds"], 0)
            self.assertIsNone(records[0]["vram"])
            self.assertEqual(records[0]["global_step"], 2)
            with (output / "metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["global_step"]), 2)
            self.assertEqual(float(rows[0]["loss"]), records[0]["train"]["loss"])
            self.assertEqual(float(rows[0]["lr"]), config["lr"])
            self.assertEqual(float(rows[0]["lr_backbone"]), config["lr_backbone"])
            self.assertEqual(rows[0]["vram_allocated_gib"], "")
            from PIL import Image

            with Image.open(output / "training_curves.png") as image:
                self.assertGreater(image.width, 1000)
                image.verify()
            self.assertIn("<svg", (output / "training_curves.svg").read_text())
            with (output / "dataset_stats.pkl").open("rb") as stream:
                stats = pickle.load(stream)
            self.assertEqual(stats["action_mean"].shape, (16,))
            policy = load_offline_policy(output / "policy_last.ckpt")
            saved = torch.load(output / "policy_last.ckpt", weights_only=True)
            periodic = torch.load(output / "policy_epoch_1_seed_0.ckpt", weights_only=True)
            for name, value in policy.state_dict().items():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
                torch.testing.assert_close(value, periodic[name], rtol=0, atol=0)
            state = torch.load(output / "training_state.pt", weights_only=False)
            self.assertTrue(state["optimizer"]["state"])
            self.assertEqual(state["epoch"], 1)
            self.assertEqual([group["lr"] for group in state["optimizer"]["param_groups"]],
                             [config["lr"], config["lr_backbone"]])
            self.assertEqual(state["scheduler"]["last_epoch"], 0)
            self.assertIsNone(state["cuda_rng_state"])
            with torch.inference_mode():
                result = policy(torch.zeros(1, 16), torch.zeros(1, 1, 3, 64, 64))
            self.assertEqual(result.shape, (1, 4, 16))
            self.assertTrue(bool(torch.isfinite(result).all()))

    def test_cosine_annealing_uses_initial_rates_then_lr_min(self):
        from starter.policies import act as act_module

        config = small_config()
        config.update(num_epochs=3, lr=1e-5, lr_backbone=2e-6, lr_min=1e-6, save_freq=1)
        actual_rates = []
        original_run_epoch = act_module._run_epoch

        def record_rates(policy, loader, device, optimizer):
            # The optimizer still holds the rates used for the epoch about to run.
            actual_rates.append([group["lr"] for group in optimizer.param_groups])
            return original_run_epoch(policy, loader, device, optimizer)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_training_episodes(root)
            config["data_path"] = str(root)
            output = root / "run"
            with (
                mock.patch("starter.policies.act.act_output_dir", return_value=output),
                mock.patch("starter.policies.act._run_epoch", side_effect=record_rates),
                mock.patch("starter.policies.act_metrics.plot_training_curves"),
            ):
                train_offline(config)
            records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            with (output / "metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            state = torch.load(output / "training_state.pt", weights_only=False)

        expected_main = (1e-5, 5.5e-6, 1e-6)
        expected_backbone = (2e-6, 1.5e-6, 1e-6)
        self.assertEqual([record["epoch"] for record in records], [1, 2, 3])
        self.assertEqual(len(actual_rates), 3)
        for index, (rates, record, row) in enumerate(zip(actual_rates, records, rows)):
            self.assertAlmostEqual(rates[0], expected_main[index], places=12)
            self.assertAlmostEqual(rates[1], expected_backbone[index], places=12)
            self.assertEqual(record["lr"], rates[0])
            self.assertEqual(record["lr_backbone"], rates[1])
            self.assertEqual(float(row["lr"]), record["lr"])
            self.assertEqual(float(row["lr_backbone"]), record["lr_backbone"])
        # The final step lands exactly on the shared minimum and is not stepped again.
        self.assertEqual([group["lr"] for group in state["optimizer"]["param_groups"]],
                         [config["lr_min"], config["lr_min"]])
        self.assertEqual(state["scheduler"]["last_epoch"], 2)

    def test_lr_min_outside_supported_range_is_rejected(self):
        for lr_min in (-1e-7, 3e-6, float("nan")):
            with self.subTest(lr_min=lr_min):
                config = small_config()
                config.update(lr=1e-5, lr_backbone=2e-6, lr_min=lr_min)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    _write_training_episodes(root)
                    config["data_path"] = str(root)
                    output = root / "run"
                    with mock.patch("starter.policies.act.act_output_dir", return_value=output):
                        with self.assertRaises(ValueError):
                            train_offline(config)


if __name__ == "__main__":
    unittest.main()
