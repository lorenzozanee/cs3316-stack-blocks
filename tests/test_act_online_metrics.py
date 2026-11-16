"""Flatten PPO records and render full-history, signed, and undefined diagnostics."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from starter.policies.act_online_metrics import metric_row, plot_online_training

ROOT = Path(__file__).resolve().parents[1]


def _vram(capacity: float) -> dict:
    return {
        "allocated_gib": 2.0,
        "reserved_gib": 2.5,
        "peak_allocated_gib": 2.75,
        "peak_reserved_gib": 3.0,
        "total_gib": capacity,
    }


def _record(update: int, **overrides) -> dict:
    """One complete flat record, matching the trainer's schema."""
    record = {
        "update": update,
        "global_step": update * 10,
        "episodes": update * 2,
        "updates": 4,
        "success_rate": 0.1 * update,
        "partial_rate": 0.2 + 0.1 * update,
        "mean_final_score": 20.0 + 5.0 * update,
        "mean_peak_score": 30.0 + 6.0 * update,
        "mean_episode_length": 100.0 + update,
        "mean_discounted_return": 50.0 + 10.0 * update,
        "policy_loss": -0.2 - 0.01 * update,
        "value_loss": 1.0 / update,
        "entropy": 2.0 - 0.1 * update,
        "approx_kl": 0.004 + 0.001 * update,
        "target_kl": 0.02,
        "clip_fraction": 0.05 * update,
        "explained_variance": 0.5,
        "policy_std_mean": 0.1,
        "policy_std_min": 0.08,
        "policy_std_max": 0.12,
        "residual_clip_fraction": 0.01 * update,
        "rollout_seconds": 10.0 + update,
        "optimization_seconds": 2.0 + 0.1 * update,
        "lr": 1e-4,
        "eta_seconds": 3600.0 - 100.0 * update,
        "vram": None,
    }
    record.update(overrides)
    return record


def _write_run(directory: Path, records: list[dict]) -> Path:
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (directory / "metrics.csv").write_bytes(b"update,success_rate\n1,0.1\n")
    return directory


def _render(records: list[dict]):
    """Render one run and return the live axes plus the SVG text."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with tempfile.TemporaryDirectory() as temporary:
        run = _write_run(Path(temporary), records)
        original_subplots = plt.subplots
        captured = []

        def capture(*args, **kwargs):
            result = original_subplots(*args, **kwargs)
            captured.append(result)
            return result

        with mock.patch.object(plt, "subplots", side_effect=capture):
            plot_online_training(run)
        assert (run / "training_curves.png").is_file()
        assert (run / "training_curves.svg").is_file()
        # Regenerating figures must not rewrite the exact-value CSV table.
        assert (run / "metrics.csv").read_bytes() == b"update,success_rate\n1,0.1\n"
        svg = (run / "training_curves.svg").read_text(encoding="utf-8")
    _, axes = captured[0]
    return axes, svg


class MetricRowTest(unittest.TestCase):
    def test_gpu_vram_is_flattened_and_every_scalar_is_preserved(self):
        record = _record(2, vram=_vram(24.0))
        row = metric_row(record)
        self.assertNotIn("vram", row)
        self.assertEqual(row["vram_allocated_gib"], 2.0)
        self.assertEqual(row["vram_reserved_gib"], 2.5)
        self.assertEqual(row["vram_peak_allocated_gib"], 2.75)
        self.assertEqual(row["vram_peak_reserved_gib"], 3.0)
        self.assertEqual(row["vram_total_gib"], 24.0)
        for key, value in record.items():
            if key != "vram":
                self.assertEqual(row[key], value, key)
        self.assertEqual(row["updates"], 4)

    def test_cpu_vram_columns_are_blank(self):
        record = _record(1, vram=None)
        row = metric_row(record)
        for field in ("allocated_gib", "reserved_gib", "peak_allocated_gib",
                      "peak_reserved_gib", "total_gib"):
            self.assertIsNone(row[f"vram_{field}"])
        self.assertEqual(row["policy_loss"], record["policy_loss"])


class OnlineCurvePlotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")

    def test_single_cpu_update_renders_all_sixteen_panels(self):
        axes, svg = _render([_record(1, explained_variance=None)])
        self.assertEqual(len(axes.flat), 16)
        self.assertIn("Training rollout metrics, not held-out evaluation", svg)
        self.assertIn("excluding Isaac Sim", svg)
        self.assertIn("Raw per-update values, no smoothing", svg)
        titles = [axis.get_title(loc="left") for axis in axes.flat]
        self.assertEqual(titles[0], "Episode outcome rates")
        self.assertEqual(titles[5], "Value loss (zeros preserved)")
        self.assertEqual(titles[9], "Explained variance")
        self.assertEqual(titles[15], "Learner VRAM (PyTorch only)")
        # CPU has no CUDA memory, so the panel states it instead of inventing zeros.
        self.assertIn("CPU run: no CUDA VRAM", [text.get_text() for text in axes[3, 3].texts])
        self.assertEqual(len(axes[3, 3].lines), 0)
        # A None explained variance is undefined, never a plotted zero.
        explained = axes[2, 1]
        self.assertTrue(all(math.isnan(value) for value in explained.lines[0].get_ydata()))
        self.assertIn("Undefined: zero return variance",
                      [text.get_text() for text in explained.texts])
        # Single-series panels need no legend box.
        self.assertIsNone(axes[0, 2].get_legend())

    def test_full_history_signs_legends_and_gpu_series(self):
        records = [
            _record(1, value_loss=1.0, explained_variance=0.5, vram=_vram(24.0)),
            _record(2, value_loss=0.5, explained_variance=0.4, vram=_vram(24.0)),
            _record(3, value_loss=0.0, explained_variance=None, vram=_vram(24.0)),
            _record(4, value_loss=0.2, explained_variance=0.3, vram=_vram(24.0)),
            _record(5, value_loss=0.1, explained_variance=0.2, vram=_vram(24.0),
                    success_rate=0.5, partial_rate=0.5),
        ]
        axes, _ = _render(records)
        updates = [record["update"] for record in records]

        # Full history is retained and the padded x-range still contains both ends.
        for axis in axes.flat:
            self.assertEqual(list(axis.lines[0].get_xdata()), updates)
            low, high = axis.get_xlim()
            self.assertLessEqual(low, updates[0])
            self.assertGreaterEqual(high, updates[-1])

        # Outcome rates: success plus a clearly named partial rate.
        rates = axes[0, 0]
        self.assertEqual([[float(value) for value in line.get_ydata()]
                          for line in (rates.lines[0], rates.lines[2])],
                         [[record["success_rate"] for record in records],
                          [record["partial_rate"] for record in records]])
        labels = [text.get_text() for text in rates.get_legend().texts]
        self.assertEqual(labels, ["Success", "Partial (peak ≥ 15, includes successes)"])
        self.assertAlmostEqual(rates.get_ylim()[1], 1.05)

        # Signed losses and entropy stay linear even when negative.
        for index in ((1, 0), (1, 2)):
            self.assertEqual(axes[index].get_yscale(), "linear", axes[index].get_title())
        self.assertLess(min(axes[1, 0].lines[0].get_ydata()), 0)
        # An exact zero value loss forces symlog rather than dropping the point.
        value_axis = axes[1, 1]
        self.assertEqual(value_axis.get_yscale(), "symlog")
        self.assertIn(0.0, [float(value) for value in value_axis.lines[0].get_ydata()])

        # KL panel keeps the fixed target as a neutral dashed threshold.
        from matplotlib.colors import to_rgba

        kl_axis = axes[1, 3]
        target = kl_axis.lines[2]
        self.assertEqual(target.get_linestyle(), "--")
        self.assertEqual(to_rgba(target.get_color()), to_rgba("#52514e"))
        self.assertEqual([text.get_text() for text in kl_axis.get_legend().texts],
                         ["Approx KL", "Target KL"])

        # Undefined explained variance leaves a gap; defined values remain.
        explained = axes[2, 1]
        explained_data = list(explained.lines[0].get_ydata())
        self.assertAlmostEqual(explained_data[0], 0.5)
        self.assertTrue(math.isnan(explained_data[2]))
        self.assertAlmostEqual(explained_data[4], 0.2)
        self.assertTrue(explained.texts)

        # Three raw-std series and two wall-time series carry legends.
        std_axis = axes[2, 2]
        self.assertEqual(len(std_axis.lines), 6)
        self.assertEqual([text.get_text() for text in std_axis.get_legend().texts],
                         ["Mean", "Min", "Max"])
        time_axis = axes[3, 0]
        self.assertEqual([text.get_text() for text in time_axis.get_legend().texts],
                         ["Rollout", "Optimization"])

        # VRAM panel uses the three requested learner series.
        memory = axes[3, 3]
        self.assertEqual([text.get_text() for text in memory.get_legend().texts],
                         ["Allocated", "Reserved", "Peak allocated"])
        self.assertEqual([float(line.get_ydata()[0]) for line in (memory.lines[0], memory.lines[2],
                                                                  memory.lines[4])],
                         [2.0, 2.5, 2.75])

        # Converging endpoint values receive vertically spaced direct labels.
        offsets = sorted(text.xyann[1] for text in rates.texts)
        self.assertGreaterEqual(len(offsets), 2)
        self.assertLess(offsets[0], offsets[-1])

    def test_undefined_explained_variance_is_not_zero(self):
        axes, _ = _render([
            _record(1, explained_variance=None),
            _record(2, explained_variance=None),
        ])
        explained = axes[2, 1]
        self.assertTrue(all(math.isnan(value) for value in explained.lines[0].get_ydata()))
        self.assertIn("Undefined: zero return variance",
                      [text.get_text() for text in explained.texts])

    def test_cli_regenerates_figures_from_metrics_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_run(Path(temporary), [_record(1), _record(2)])
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            result = subprocess.run(
                [sys.executable, "-m", "starter.policies.act_online_metrics", str(run)],
                capture_output=True, text=True, env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((run / "training_curves.png").is_file())
            self.assertIn("<svg", (run / "training_curves.svg").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
