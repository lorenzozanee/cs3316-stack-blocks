"""Online metrics: rolling means, plot axes, and CSV safety."""
from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from starter.policies.online_metrics import plot_online_metrics, rolling_mean


EPISODES = (
    {"timesteps": 4, "return": 100.0, "success": True, "length": 2},
    {"timesteps": 5, "return": 15.0, "success": False, "length": 3},
    {"timesteps": 9, "return": 100.0, "success": True, "length": 2},
    {"timesteps": 10, "return": 15.0, "success": False, "length": 3},
)
METRICS = (
    {"timesteps": 5, "policy_loss": 0.5, "value_loss": 0.3},
    {"timesteps": 10, "policy_loss": 0.4, "value_loss": 0.2},
)
PLOT_NAMES = ("return_curve.png", "success_rate.png", "episode_length.png", "loss_curve.png")


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RollingMeanTest(unittest.TestCase):
    def test_prefix_warmup_and_empty_input(self):
        np.testing.assert_allclose(rolling_mean([10, 20, 60, 40], 2), [10, 15, 40, 50])
        self.assertEqual(rolling_mean([], 2).size, 0)

    def test_window_must_be_positive_integer(self):
        for window in (0, -1, 2.0, True):
            with self.assertRaises(ValueError):
                rolling_mean([1.0], window)


class PlotOnlineMetricsTest(unittest.TestCase):
    def _fixture(self, root):
        run_dir = root / "run"
        run_dir.mkdir()
        _write_csv(run_dir / "episodes.csv", EPISODES)
        _write_csv(run_dir / "metrics.csv", METRICS)
        return run_dir

    def test_plot_uses_episode_timesteps_and_leaves_csvs_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = self._fixture(Path(temp))
            episodes_before = (run_dir / "episodes.csv").read_bytes()
            metrics_before = (run_dir / "metrics.csv").read_bytes()
            plt.close("all")
            figures_before = plt.get_fignums()
            calls = []
            real_plot = Axes.plot

            def spy(axis, *args, **kwargs):
                calls.append((np.asarray(args[0]), np.asarray(args[1]), kwargs.get("label")))
                return real_plot(axis, *args, **kwargs)

            with patch.object(Axes, "plot", spy):
                plots = plot_online_metrics(run_dir, 2)

            self.assertEqual(len(calls), 7)
            timesteps = [4, 5, 9, 10]
            np.testing.assert_array_equal(calls[0][0], timesteps)
            np.testing.assert_allclose(calls[0][1], [100.0, 15.0, 100.0, 15.0])
            self.assertEqual(calls[0][2], "Completed episode")
            np.testing.assert_array_equal(calls[1][0], timesteps)
            np.testing.assert_allclose(calls[1][1], [100.0, 57.5, 57.5, 57.5])
            np.testing.assert_array_equal(calls[2][0], timesteps)
            np.testing.assert_allclose(calls[2][1], [1.0, 0.5, 0.5, 0.5])
            self.assertIn("Trailing mean", calls[2][2])
            np.testing.assert_array_equal(calls[3][0], timesteps)
            np.testing.assert_allclose(calls[3][1], [2.0, 3.0, 2.0, 3.0])
            np.testing.assert_array_equal(calls[4][0], timesteps)
            np.testing.assert_allclose(calls[4][1], [2.0, 2.5, 2.5, 2.5])
            np.testing.assert_array_equal(calls[5][0], [5, 10])
            np.testing.assert_allclose(calls[5][1], [0.5, 0.4])
            np.testing.assert_array_equal(calls[6][0], [5, 10])
            np.testing.assert_allclose(calls[6][1], [0.3, 0.2])

            self.assertEqual((run_dir / "episodes.csv").read_bytes(), episodes_before)
            self.assertEqual((run_dir / "metrics.csv").read_bytes(), metrics_before)
            self.assertEqual(plt.get_fignums(), figures_before)
            for name in PLOT_NAMES:
                path = plots / name
                self.assertTrue(path.is_file(), name)
                self.assertGreater(path.stat().st_size, 0, name)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n", name)


if __name__ == "__main__":
    unittest.main()
