"""Update-level residual PPO diagnostics from persisted training records.

``metrics.jsonl`` is the source of truth. The static PNG/SVG figures retain the
full update history without smoothing, and ``metrics.csv`` carries the exact
values as flat columns.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from starter.interfaces import PARTIAL_SCORE

VRAM_FIELDS = ("allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib", "total_gib")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def metric_row(record: dict) -> dict:
    """Flatten one record for CSV; CPU runs leave every VRAM column empty."""
    row = {key: value for key, value in record.items() if key != "vram"}
    vram = record["vram"]
    for field in VRAM_FIELDS:
        row[f"vram_{field}"] = vram[field] if vram is not None else None
    return row


def _load_records(run_dir: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("No residual PPO updates have been recorded.")
    return records


def _line(axis, x, values, *, color, style="-", label=None):
    """Draw one full-history line plus a surface-ringed endpoint marker."""
    plotted = [float("nan") if value is None else value for value in values]
    axis.plot(x, plotted, color=color, linestyle=style, linewidth=1.1, label=label,
              solid_capstyle="round", solid_joinstyle="round")
    finite = [index for index, value in enumerate(plotted) if math.isfinite(value)]
    if finite:
        last = finite[-1]
        axis.plot([x[last]], [plotted[last]], marker="o", color=color, linestyle="none",
                  markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.0)
    return plotted


def _endpoint_labels(axis, entries, *, fontsize=8):
    """Direct-label line ends, spacing labels apart when their values converge."""
    entries = [entry for entry in entries if math.isfinite(entry[1])]
    if not entries:
        return
    low, high = axis.get_ylim()
    span = high - low if high > low else 1.0
    ordered = sorted(entries, key=lambda entry: entry[1])
    groups = [[ordered[0]]]
    for entry in ordered[1:]:
        if abs(entry[1] - groups[-1][-1][1]) < 0.075 * span:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    for group in groups:
        for index, (x, y, text) in enumerate(group):
            offset = (index - (len(group) - 1) / 2) * 11
            if len(group) > 1 and group[0][1] <= low + 0.05 * span:
                offset = 4 + index * 11
            elif len(group) > 1 and group[-1][1] >= high - 0.05 * span:
                offset = -4 - (len(group) - 1 - index) * 11
            axis.annotate(text, (x, y), xytext=(7, offset), textcoords="offset points",
                          va="center", fontsize=fontsize, color=INK)


def _legend(axis, **kwargs):
    axis.legend(frameon=False, fontsize=8, **kwargs)


def _format_axis(axis, updates):
    from matplotlib.ticker import MaxNLocator

    span = max(updates[-1] - updates[0], 1)
    axis.set_xlabel("Round-local PPO update")
    axis.set_xlim(updates[0] - max(0.25, 0.03 * span), updates[-1] + max(0.35, 0.18 * span))
    if len(updates) <= 6:
        axis.set_xticks(updates)
    else:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=2))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)


def _loss_scale(axis, values):
    """Log for positive losses, symlog when an exact zero must stay visible."""
    positive = [value for value in values if value > 0]
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    if len(positive) == len(values):
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=7))
    elif positive:
        axis.set_yscale("symlog", linthresh=min(positive), linscale=0.5)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
    axis.yaxis.set_minor_locator(NullLocator())


def plot_online_training(run_dir: str | Path) -> None:
    """Refresh training_curves.png/svg from metrics.jsonl beside the CSV table."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    run_dir = Path(run_dir)
    records = _load_records(run_dir)
    updates = [record["update"] for record in records]

    def values(key):
        return [record[key] for record in records]

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "svg.fonttype": "none",
    }):
        fig, axes = plt.subplots(4, 4, figsize=(18, 14))
        fig.suptitle("ACT residual PPO online training", fontsize=17, fontweight="bold",
                     x=0.04, ha="left")

        # 1. Outcome rates. PARTIAL_SCORE is a peak threshold, so successes count too.
        axis = axes[0, 0]
        axis.set_ylim(0, 1.05)
        entries = []
        for index, (key, name) in enumerate((
            ("success_rate", "Success"),
            ("partial_rate", f"Partial (peak ≥ {PARTIAL_SCORE:g}, includes successes)"),
        )):
            line = _line(axis, updates, values(key), color=COLORS[index], style=("-", "--")[index], label=name)
            entries.append((updates[-1], line[-1], f"{line[-1]:.1%}"))
        _endpoint_labels(axis, entries)
        _legend(axis, loc="best")
        axis.set_title("Episode outcome rates", loc="left", fontsize=11)
        axis.set_ylabel("Percent of episodes")
        axis.yaxis.set_major_formatter(PercentFormatter(1))

        # 2. Final and peak scores.
        axis = axes[0, 1]
        axis.set_ylim(0, 105)
        entries = []
        for index, (key, name) in enumerate((("mean_final_score", "Final"),
                                             ("mean_peak_score", "Peak"))):
            line = _line(axis, updates, values(key), color=COLORS[index], style=("-", "--")[index], label=name)
            entries.append((updates[-1], line[-1], f"{line[-1]:.2f}"))
        _endpoint_labels(axis, entries)
        _legend(axis, loc="best")
        axis.set_title("Episode scores", loc="left", fontsize=11)
        axis.set_ylabel("Score")

        # 3. Episode length.
        axis = axes[0, 2]
        line = _line(axis, updates, values("mean_episode_length"), color=COLORS[0])
        _endpoint_labels(axis, [(updates[-1], line[-1], f"{line[-1]:.1f}")])
        axis.set_title("Episode length", loc="left", fontsize=11)
        axis.set_ylabel("Steps")
        axis.margins(y=0.12)
        axis.set_ylim(bottom=0)

        # 4. Discounted return.
        axis = axes[0, 3]
        line = _line(axis, updates, values("mean_discounted_return"), color=COLORS[0])
        _endpoint_labels(axis, [(updates[-1], line[-1], f"{line[-1]:.3g}")])
        axis.set_title("Mean discounted return", loc="left", fontsize=11)
        axis.set_ylabel("Return")
        axis.margins(y=0.12)
        axis.set_ylim(bottom=0)

        # 5. Policy loss is signed, so it stays on a linear axis.
        axis = axes[1, 0]
        line = _line(axis, updates, values("policy_loss"), color=COLORS[0])
        _endpoint_labels(axis, [(updates[-1], line[-1], f"{line[-1]:.4g}")])
        axis.axhline(0, color=AXIS, linewidth=0.8)
        axis.set_title("Policy loss (signed)", loc="left", fontsize=11)
        axis.set_ylabel("Loss")
        axis.margins(y=0.12)

        # 6. Value loss: exact zeros stay visible through symlog when needed.
        axis = axes[1, 1]
        loss = values("value_loss")
        line = _line(axis, updates, loss, color=COLORS[0])
        _loss_scale(axis, loss)
        _endpoint_labels(axis, [(updates[-1], line[-1], f"{line[-1]:.4g}")])
        axis.set_title("Value loss (zeros preserved)", loc="left", fontsize=11)
        axis.set_ylabel("Loss")

        # 7. Entropy is signed on a linear axis.
        axis = axes[1, 2]
        line = _line(axis, updates, values("entropy"), color=COLORS[0])
        _endpoint_labels(axis, [(updates[-1], line[-1], f"{line[-1]:.4g}")])
        axis.axhline(0, color=AXIS, linewidth=0.8)
        axis.set_title("Policy entropy (signed)", loc="left", fontsize=11)
        axis.set_ylabel("Nats")
        axis.margins(y=0.12)

        # 8. Approximate KL against the fixed neutral threshold.
        axis = axes[1, 3]
        entries = []
        kl = values("approx_kl")
        line = _line(axis, updates, kl, color=COLORS[0], label="Approx KL")
        entries.append((updates[-1], line[-1], f"{line[-1]:.4g}"))
        target = values("target_kl")
        _line(axis, updates, target, color=MUTED, style="--", label="Target KL")
        entries.append((updates[-1], target[-1], f"target {target[-1]:.3g}"))
        _endpoint_labels(axis, entries)
        _legend(axis, loc="best")
        axis.set_title("Approximate KL", loc="left", fontsize=11)
        axis.set_ylabel("KL")
        axis.margins(y=0.12)
        axis.set_ylim(bottom=0)

        # 9. PPO clip fraction.
        axis = axes[2, 0]
        line = _line(axis, updates, values("clip_fraction"), color=COLORS[0])
        _endpoint_labels(axis, [(updates[-1], line[-1], f"{line[-1]:.1%}")])
        axis.set_title("PPO clip fraction", loc="left", fontsize=11)
        axis.set_ylabel("Sample fraction")
        axis.set_ylim(0, 1.05)
        axis.yaxis.set_major_formatter(PercentFormatter(1))

        # 10. Explained variance. ``None`` means return variance was zero: leave a
        # gap and name the undefined state instead of plotting an invented zero.
        axis = axes[2, 1]
        explained = values("explained_variance")
        plotted = _line(axis, updates, explained, color=COLORS[0])
        defined = [(x, value) for x, value in zip(updates, plotted) if math.isfinite(value)]
        if defined:
            _endpoint_labels(axis, [(defined[-1][0], defined[-1][1], f"{defined[-1][1]:.3g}")])
        if any(value is None for value in explained):
            note = ("Undefined: zero return variance" if not defined
                    else "Gaps: undefined (zero return variance)")
            axis.text(0.02, 0.04, note, transform=axis.transAxes, color=MUTED, fontsize=8)
        axis.axhline(0, color=AXIS, linewidth=0.8)
        axis.set_title("Explained variance", loc="left", fontsize=11)
        axis.set_ylabel("Explained variance")
        axis.margins(y=0.12)

        # 11. Raw Gaussian residual standard deviation.
        axis = axes[2, 2]
        entries = []
        for index, (key, name) in enumerate((("policy_std_mean", "Mean"),
                                             ("policy_std_min", "Min"),
                                             ("policy_std_max", "Max"))):
            line = _line(axis, updates, values(key), color=COLORS[index], label=name)
            entries.append((updates[-1], line[-1], f"{line[-1]:.4g}"))
        _endpoint_labels(axis, entries)
        _legend(axis, loc="best")
        axis.set_title("Policy Gaussian std (raw residual)", loc="left", fontsize=11)
        axis.set_ylabel("Std")
        axis.margins(y=0.12)
        axis.set_ylim(bottom=0)

        # 12. Fraction of raw residual elements clipped by |raw| > 1.
        axis = axes[2, 3]
        line = _line(axis, updates, values("residual_clip_fraction"), color=COLORS[0])
        _endpoint_labels(axis, [(updates[-1], line[-1], f"{line[-1]:.1%}")])
        axis.set_title("Residual clip fraction |raw| > 1", loc="left", fontsize=11)
        axis.set_ylabel("Element fraction")
        axis.set_ylim(0, 1.05)
        axis.yaxis.set_major_formatter(PercentFormatter(1))

        # 13. Wall time spent in rollout and optimization, as separate series.
        axis = axes[3, 0]
        entries = []
        for index, (key, name) in enumerate((("rollout_seconds", "Rollout"),
                                             ("optimization_seconds", "Optimization"))):
            line = _line(axis, updates, values(key), color=COLORS[index], label=name)
            entries.append((updates[-1], line[-1], f"{line[-1]:.1f} s"))
        _endpoint_labels(axis, entries)
        _legend(axis, loc="best")
        axis.set_title("Update wall time", loc="left", fontsize=11)
        axis.set_ylabel("Seconds")
        axis.margins(y=0.12)
        axis.set_ylim(bottom=0)

        # 14. Learning rate.
        axis = axes[3, 1]
        line = _line(axis, updates, values("lr"), color=COLORS[0])
        _endpoint_labels(axis, [(updates[-1], line[-1], f"{line[-1]:.3g}")])
        axis.set_title("Learning rate", loc="left", fontsize=11)
        axis.set_ylabel("LR")
        axis.margins(y=0.12)

        # 15. Estimated time remaining in minutes.
        axis = axes[3, 2]
        eta = [value / 60 for value in values("eta_seconds")]
        line = _line(axis, updates, eta, color=COLORS[0])
        _endpoint_labels(axis, [(updates[-1], line[-1], f"{line[-1]:.2f} min")])
        axis.set_title("Estimated time remaining", loc="left", fontsize=11)
        axis.set_ylabel("Minutes")
        axis.margins(y=0.12)
        axis.set_ylim(bottom=0)

        # 16. Learner VRAM, or an explicit CPU annotation when CUDA is absent.
        axis = axes[3, 3]
        gpu_records = [record for record in records if record["vram"] is not None]
        if gpu_records:
            memory_updates = [record["update"] for record in gpu_records]
            entries = []
            series = (("allocated_gib", "Allocated", "-"),
                      ("reserved_gib", "Reserved", "--"),
                      ("peak_allocated_gib", "Peak allocated", ":"))
            for index, ((key, name, style), color) in enumerate(zip(series, COLORS)):
                memory = [record["vram"][key] for record in gpu_records]
                line = _line(axis, memory_updates, memory, color=color, style=style, label=name)
                entries.append((memory_updates[-1], line[-1], f"{line[-1]:.2f}"))
            _endpoint_labels(axis, entries)
            _legend(axis, loc="best")
            axis.margins(y=0.12)
            axis.set_ylim(bottom=0)
        else:
            axis.text(0.5, 0.5, "CPU run: no CUDA VRAM", ha="center", va="center",
                      transform=axis.transAxes, color=MUTED)
        axis.set_title("Learner VRAM (PyTorch only)", loc="left", fontsize=11)
        axis.set_ylabel("GiB")

        for axis in axes.flat:
            _format_axis(axis, updates)

        fig.text(0.04, 0.020, "Raw per-update values, no smoothing | Exact values: metrics.csv",
                 color=MUTED, fontsize=9)
        fig.text(0.04, 0.043,
                 "Training rollout metrics, not held-out evaluation | VRAM is the PyTorch learner only, "
                 "excluding Isaac Sim | VRAM peaks are per update",
                 color=MUTED, fontsize=9)
        fig.tight_layout(rect=(0.01, 0.062, 0.99, 0.945), h_pad=2.4, w_pad=2.0)
        try:
            fig.savefig(run_dir / "training_curves.png", dpi=160, facecolor=SURFACE)
            fig.savefig(run_dir / "training_curves.svg", facecolor=SURFACE)
        finally:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate residual PPO training curves from metrics.jsonl."
    )
    parser.add_argument("run_dir", type=Path)
    plot_online_training(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
