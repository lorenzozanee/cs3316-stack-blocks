"""Check full-history loss scales and readable late-training detail plots."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from starter.policies.act_metrics import plot_training_curves


class TrainingCurvePlotTest(unittest.TestCase):
    def _render(self, values):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            records = [{
                "epoch": i, "global_step": i * 7,
                "lr": 1e-6, "lr_backbone": 1e-6,
                "epoch_seconds": 1.5, "elapsed_seconds": i * 1.5,
                "eta_seconds": (len(values) - i) * 1.5,
                "train": {"loss": value, "l1": value / 10, "kl": value / 100},
                "vram": None,
            } for i, value in enumerate(values, 1)]
            (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            original_csv = b"epoch,loss\n1,100\n"
            (run / "metrics.csv").write_bytes(original_csv)
            plot_training_curves(run)
            self.assertEqual((run / "metrics.csv").read_bytes(), original_csv)
            self.assertTrue((run / "training_curves.png").is_file())
            svg = (run / "training_curves.svg").read_text()
            self.assertIn("All epoch values retained", svg)
            return records, svg

    def test_full_history_and_late_detail_preserve_every_value(self):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        values = [40.0] * 15 + [0.04, 0.05, 0.045] * 5
        original_subplots = plt.subplots
        captured = []

        def capture(*args, **kwargs):
            result = original_subplots(*args, **kwargs)
            captured.append(result)
            return result

        with mock.patch.object(plt, "subplots", side_effect=capture):
            records, svg = self._render(values)
        self.assertIn("Latter half", svg)
        _, axes = captured[0]
        for axis, key in zip(axes[0], ("loss", "l1", "kl")):
            actual = [r["train"][key] for r in records]
            self.assertEqual(axis.get_yscale(), "log")
            self.assertEqual(list(axis.lines[0].get_ydata()), actual)
            low, high = axis.get_ylim()
            self.assertLessEqual(low, min(actual))
            self.assertGreaterEqual(high, max(actual))
            detail = axis.child_axes[0]
            self.assertEqual(detail.get_yscale(), "linear")
            self.assertEqual(list(detail.lines[0].get_ydata()), actual[len(actual) // 2:])
            low, high = detail.get_ylim()
            self.assertLessEqual(low, min(actual[len(actual) // 2:]))
            self.assertGreaterEqual(high, max(actual[len(actual) // 2:]))
        lr_axis = axes[1, 0]
        self.assertEqual(lr_axis.get_title(loc="left"), "Learning rate")
        self.assertEqual(list(lr_axis.lines[0].get_ydata()), [r["lr"] * 1e6 for r in records])
        self.assertGreater(lr_axis.get_ylim()[0], 0)
        self.assertEqual(lr_axis.get_ylabel(), "LR (×10⁻⁶)")
        self.assertEqual([axis.get_title(loc="left") for axis in (axes[1, 1], axes[1, 2])],
                         ["Estimated time remaining", "Training VRAM"])

    def test_exact_zero_losses_are_retained(self):
        self._render([1.0, 0.5, 0.1, 0.0, 0.05, 0.0])

    def test_short_flat_run_renders(self):
        self._render([0.0])


if __name__ == "__main__":
    unittest.main()
