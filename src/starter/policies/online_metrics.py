"""Plot online training CSVs with Matplotlib; see SOURCES.md for dependencies."""

import argparse
import csv
from pathlib import Path

import numpy as np


def rolling_mean(values, window):
    """Trailing episode mean, including the available prefix during warmup."""
    values = np.asarray(values, dtype=float)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer.")
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    cumulative = np.concatenate(([0.0], values.cumsum()))
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def plot_online_metrics(run_dir, window=100):
    """Render complete episode returns and PPO losses, without mixing their axes."""
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    run_dir = Path(run_dir)
    with (run_dir / "episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    with (run_dir / "metrics.csv").open(newline="") as handle:
        metrics = list(csv.DictReader(handle))
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    blue, orange = "#2a78d6", "#eb6834"
    surface, text, secondary = "#fcfcfb", "#0b0b0b", "#52514e"
    style = {
        "font.family": "DejaVu Sans", "font.size": 9,
        "figure.facecolor": surface, "axes.facecolor": surface,
        "text.color": text, "axes.labelcolor": secondary, "axes.titlecolor": text,
        "xtick.color": secondary, "ytick.color": secondary,
        "axes.edgecolor": "#c3c2b7", "grid.color": "#e1e0d9", "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
    }

    def figure(columns=1):
        fig = Figure(figsize=(7 * columns, 4.5), dpi=150)
        FigureCanvasAgg(fig)
        axes = np.atleast_1d(fig.subplots(1, columns))
        for ax in axes:
            ax.set_xlabel("Environment transitions (all scenes)")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        return fig, axes

    def save(fig, name):
        fig.text(0.02, 0.02, "Training rollouts; official checkpoint evaluation is separate.",
                 color=secondary, fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(plots / name)
        fig.clear()

    def empty(ax, message):
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
                color=secondary)

    def endpoint(ax, x, y):
        ax.annotate(f"{y[-1]:.3g}", (x[-1], y[-1]), xytext=(5, 6),
                    textcoords="offset points", color=text, fontsize=8)
        ax.margins(x=0.1, y=0.15)

    def episode_curve(key, title, ylabel, filename, rate=False):
        fig, (ax,) = figure()
        ax.set(title=title, ylabel=ylabel)
        if episodes:
            x = np.array([int(row["timesteps"]) for row in episodes])
            y = np.array([row[key] == "True" if rate else float(row[key]) for row in episodes], dtype=float)
            mean = rolling_mean(y, window)
            if not rate:
                ax.plot(x, y, color=secondary, alpha=0.4, linewidth=0.7,
                        marker="." if len(y) <= 20 else None, markersize=6, label="Completed episode")
            ax.plot(x, mean, color=blue, linewidth=1.5,
                    marker="o" if len(y) == 1 else None, markersize=6,
                    markeredgecolor=surface, markeredgewidth=1,
                    label=f"Trailing mean (up to {window} episodes)")
            endpoint(ax, x, mean)
            if not rate:
                ax.legend(frameon=False, labelcolor=secondary)
        else:
            empty(ax, "No completed episodes yet")
        if rate:
            ax.set_ylim(-0.03, 1.1)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        save(fig, filename)

    with rc_context(style):
        episode_curve("return", "Training episode return", "Undiscounted sum of environment scores",
                      "return_curve.png")
        episode_curve("success", f"Training success rate · trailing {window} episodes", "Success rate",
                      "success_rate.png", rate=True)
        episode_curve("length", "Training episode length", "Environment steps per completed episode",
                      "episode_length.png")
        fig, axes = figure(columns=2)
        for ax, key, title, color in zip(
            axes, ("policy_loss", "value_loss"), ("PPO policy loss", "PPO value loss"), (blue, orange),
        ):
            ax.set(title=title, ylabel="Loss")
            if metrics:
                x = np.array([int(row["timesteps"]) for row in metrics])
                y = np.array([float(row[key]) for row in metrics])
                ax.plot(x, y, color=color, linewidth=1.5,
                        marker="o" if len(y) <= 20 else None, markersize=6,
                        markeredgecolor=surface, markeredgewidth=1)
                endpoint(ax, x, y)
            else:
                empty(ax, "No completed PPO updates yet")
        save(fig, "loss_curve.png")
    return plots


def main():
    parser = argparse.ArgumentParser(description="Regenerate online training plots from saved CSVs.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    print(plot_online_metrics(args.run_dir, args.window))


if __name__ == "__main__":
    main()
