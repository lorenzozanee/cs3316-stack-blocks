"""Update-level residual PPO diagnostics from the persisted training records."""
from __future__ import annotations

import json
from pathlib import Path


def plot_online_training(run_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    run_dir = Path(run_dir)
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    if not records:
        raise ValueError("No residual PPO updates have been recorded.")
    surface, ink, muted, grid, color = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"
    specs = (("success_rate", "Task success rate"), ("mean_final_score", "Final external score"),
             ("mean_episode_length", "Episode length"), ("policy_loss", "PPO policy loss"),
             ("value_loss", "Value loss"), ("approx_kl", "Approximate policy KL"))
    updates = [record["update"] for record in records]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": ink, "axes.labelcolor": muted, "axes.titlecolor": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.facecolor": surface, "figure.facecolor": surface,
                         "axes.edgecolor": grid, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        fig.suptitle("ACT residual PPO", x=0.04, ha="left", fontsize=17, fontweight="bold")
        for axis, (key, title) in zip(axes.flat, specs):
            values = [record[key] for record in records]
            axis.plot(updates, values, color=color, linewidth=1.2)
            axis.plot(updates[-1], values[-1], "o", color=color, markersize=5,
                      markeredgecolor=surface, markeredgewidth=1)
            axis.annotate(f"{values[-1]:.3g}", (updates[-1], values[-1]), xytext=(7, 0),
                          textcoords="offset points", va="center", fontsize=9, color=ink)
            axis.set_title(title, loc="left", fontsize=11)
            axis.set_xlabel("PPO update")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color=grid, linewidth=0.5)
            axis.set_axisbelow(True)
            span = max(updates[-1] - updates[0], 1)
            axis.set_xlim(updates[0] - 0.03 * span, updates[-1] + 0.2 * span)
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0, 1].set_ylim(0, 105)
        fig.text(0.04, 0.02, "Frozen ACT base | Score 100 is success | Exact values: metrics.csv", color=muted)
        fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.94), h_pad=2.3, w_pad=2)
        try:
            fig.savefig(run_dir / "training_curves.png", dpi=160, facecolor=surface)
            fig.savefig(run_dir / "training_curves.svg", facecolor=surface)
        finally:
            plt.close(fig)
