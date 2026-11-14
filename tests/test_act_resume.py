"""Resume ACT with linear annealing while preserving existing run history."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party"))

import torch

from starter.policies import act_metrics
from starter.policies.act import train_offline
from tests.test_act import _write_training_episodes, small_config


def _load_resume_module():
    spec = importlib.util.spec_from_file_location("resume_act", ROOT / "scripts/resume_act.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def _max_optimizer_step(state: dict) -> int:
    return max(int(entry["step"]) for entry in state["optimizer"]["state"].values())


def _train_fresh(config: dict, output: Path) -> None:
    with mock.patch("starter.policies.act.act_output_dir", return_value=output):
        train_offline(config)


class ACTResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.resume = _load_resume_module()

    def test_resume_linearly_anneals_and_appends_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_training_episodes(root)
            config = small_config()
            config["data_path"] = str(root)
            output = root / "run"
            _train_fresh(config, output)
            original_files = {path.name for path in output.iterdir()}
            original_log = (output / "metrics.jsonl").read_bytes()
            original_csv = (output / "metrics.csv").read_bytes()
            original_state = torch.load(output / "training_state.pt", weights_only=False)
            self.assertEqual(_max_optimizer_step(original_state), 2)
            rates = []
            from starter.policies import act

            original_run_epoch = act._run_epoch

            def record_rates(policy, loader, device, optimizer):
                rates.append([group["lr"] for group in optimizer.param_groups])
                return original_run_epoch(policy, loader, device, optimizer)

            with (
                mock.patch.object(act_metrics, "plot_training_curves") as plot,
                mock.patch.object(act, "_run_epoch", side_effect=record_rates),
            ):
                self.resume.resume_training(output / "policy_epoch_1_seed_0.ckpt", 3, 1e-6, 1e-7)
            plot.assert_called_with(output)
            resumed_log = (output / "metrics.jsonl").read_bytes()
            self.assertTrue(resumed_log.startswith(original_log))
            self.assertTrue((output / "metrics.csv").read_bytes().startswith(original_csv))
            records = [json.loads(line) for line in resumed_log.decode().splitlines()]
            self.assertEqual([record["epoch"] for record in records], [1, 2, 3, 4])
            self.assertEqual([record["global_step"] for record in records], [2, 4, 6, 8])
            elapsed = [record["elapsed_seconds"] for record in records]
            self.assertTrue(all(b > a for a, b in zip(elapsed, elapsed[1:])))
            with (output / "metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["epoch"]) for row in rows], [1, 2, 3, 4])
            self.assertEqual(len(rates), 3)
            for expected, actual, record, row in zip((1e-6, 5.5e-7, 1e-7), rates, records[1:], rows[1:]):
                for key, rate in zip(("lr", "lr_backbone"), actual):
                    self.assertAlmostEqual(rate, expected, places=15)
                    self.assertEqual(record[key], rate)
                    self.assertEqual(float(row[key]), rate)
            state = torch.load(output / "training_state.pt", weights_only=False)
            self.assertEqual(_max_optimizer_step(state), 8)
            self.assertEqual(state["epoch"], 4)
            self.assertEqual(state["scheduler"]["total_iters"], 2)
            self.assertEqual(state["scheduler"]["last_epoch"], 2)
            self.assertEqual(state["scheduler"]["end_factor"], 0.1)
            for group in state["optimizer"]["param_groups"]:
                self.assertAlmostEqual(group["lr"], 1e-7, places=15)
            snapshot = json.loads((output / "config.json").read_text())
            self.assertEqual(snapshot["num_epochs"], 4)
            self.assertEqual(snapshot["lr"], 1e-6)
            self.assertEqual(snapshot["lr_backbone"], 1e-6)
            self.assertEqual(snapshot["lr_min"], 1e-7)
            expected_files = original_files | {f"policy_epoch_{i}_seed_0.ckpt" for i in (2, 3, 4)}
            self.assertEqual({path.name for path in output.iterdir()}, expected_files)

    def test_resume_rejects_stale_checkpoint_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_training_episodes(root)
            config = small_config()
            config.update(data_path=str(root), num_epochs=2)
            output = root / "run"
            _train_fresh(config, output)
            before = _snapshot(output)
            with self.assertRaises(ValueError):
                self.resume.resume_training(output / "policy_epoch_1_seed_0.ckpt", 1, 1e-6, 1e-7)
            self.assertEqual(_snapshot(output), before)

    def test_resume_rejects_invalid_epochs_and_rates_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_training_episodes(root)
            config = small_config()
            config["data_path"] = str(root)
            output = root / "run"
            _train_fresh(config, output)
            checkpoint = output / "policy_last.ckpt"
            before = _snapshot(output)
            invalid = (
                (0, 1e-6, 1e-7), (-1, 1e-6, 1e-7), (1.5, 1e-6, 1e-7),
                (True, 1e-6, 1e-7), (1, 0, 1e-7), (1, math.nan, 1e-7),
                (1, math.inf, 1e-7), (1, 1e-6, 0), (1, 1e-6, -1e-7),
                (1, 1e-6, math.nan), (1, 1e-6, math.inf), (1, 1e-6, 2e-6),
            )
            for args in invalid:
                with self.subTest(args=args):
                    with self.assertRaises(ValueError):
                        self.resume.resume_training(checkpoint, *args)
                    self.assertEqual(_snapshot(output), before)


if __name__ == "__main__":
    unittest.main()
