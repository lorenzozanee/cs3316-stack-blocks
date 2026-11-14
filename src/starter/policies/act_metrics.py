"""CSV records and static ACT training curves, without a display server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CSV_COLUMNS = (
    "epoch", "global_step", "loss", "l1", "kl", "lr", "lr_backbone",
    "epoch_seconds", "elapsed_seconds", "eta_seconds",
    "vram_allocated_gib", "vram_reserved_gib", "vram_peak_allocated_gib",
    "vram_peak_reserved_gib",
)


def metric_row(record: dict) -> dict:
    row = {key: record[key] for key in (
        "epoch", "global_step", "lr", "lr_backbone", "epoch_seconds",
        "elapsed_seconds", "eta_seconds",
    )}
    row.update(record["train"])
    for key in ("allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib"):
        row[f"vram_{key}"] = record["vram"][key] if record["vram"] is not None else None
    return row


def plot_training_curves(run_dir: str | Path) -> None:
    """Read metrics.jsonl and refresh PNG/SVG figures alongside the CSV table."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator

    run_dir = Path(run_dir)
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Cannot plot ACT training without recorded epochs.")
    epochs = [record["epoch"] for record in records]
    # Fixed light export palette. CSV supplies the exact table values; line styles
    # and direct labels distinguish the three VRAM series independently of hue.
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    colors = ("#2a78d6", "#eb6834", "#1baf7a")
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": ink, "axes.labelcolor": muted, "axes.titlecolor": ink,
        "xtick.color": muted, "ytick.color": muted,
        "axes.edgecolor": "#c3c2b7", "axes.facecolor": surface,
        "figure.facecolor": surface, "svg.fonttype": "none",
    }):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=False)
        fig.suptitle("ACT offline training", fontsize=17, fontweight="bold", x=0.04, ha="left")

        def draw(axis, x, values, *, color=colors[0], style="-", label=None, endpoint=None):
            axis.plot(x, values, color=color, linestyle=style, linewidth=1.0,
                      solid_capstyle="round", label=label, marker="o", markevery=[len(x) - 1],
                      markersize=4.5, markerfacecolor=color, markeredgecolor=surface, markeredgewidth=0.9)
            if endpoint is not None:
                axis.annotate(endpoint, (x[-1], values[-1]), xytext=(7, 0),
                              textcoords="offset points", color=ink, fontsize=9, va="center")

        loss_specs = (("loss", "Total BC loss"), ("l1", "Normalized L1 loss"), ("kl", "KL divergence"))
        for axis, (key, title) in zip(axes[0], loss_specs):
            values = [record["train"][key] for record in records]
            draw(axis, epochs, values, endpoint=f"{values[-1]:.4g}")
            axis.set_title(title, loc="left", fontsize=11)
            axis.set_ylabel("Loss (log scale)")
            positive = [value for value in values if value > 0]
            if len(positive) == len(values):
                axis.set_yscale("log")
                axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=7))
            elif positive:
                # Preserve exact zero losses, with a linear region near zero.
                axis.set_yscale("symlog", linthresh=min(positive), linscale=0.5)
            else:
                axis.set_ylabel("Loss")
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
            axis.yaxis.set_minor_locator(NullLocator())
            axis.margins(y=0.12)
            if len(values) >= 4:
                first = len(values) // 2
                detail = axis.inset_axes([0.48, 0.51, 0.49, 0.43])
                detail.plot(epochs[first:], values[first:], color=colors[0], linewidth=0.7)
                detail.set_title("Latter half · linear scale", loc="left", fontsize=7, pad=4)
                detail.set_xlim(epochs[first], epochs[-1])
                low, high = min(values[first:]), max(values[first:])
                padding = max((high - low) * 0.08, abs(high) * 0.01, 1e-9)
                detail.set_ylim(max(0, low - padding), high + padding)
                detail.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
                detail.yaxis.set_major_locator(MaxNLocator(nbins=3))
                detail.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
                detail.tick_params(labelsize=6.5, length=2, pad=2)
                detail.spines[["top", "right"]].set_visible(False)
                detail.grid(axis="y", color=grid, linewidth=0.4)
                detail.set_axisbelow(True)
        main_lr = [record["lr"] * 1e6 for record in records]
        backbone_lr = [record["lr_backbone"] * 1e6 for record in records]
        draw(axes[1, 0], epochs, main_lr, label="Main", endpoint=f"{main_lr[-1]:.3f}")
        if backbone_lr != main_lr:
            draw(axes[1, 0], epochs, backbone_lr, color=colors[1], style="--",
                 label="Backbone", endpoint=f"{backbone_lr[-1]:.3f}")
            axes[1, 0].legend(loc="upper right", frameon=False, fontsize=8)
        axes[1, 0].set_title("Learning rate", loc="left", fontsize=11)
        axes[1, 0].set_ylabel("LR (×10⁻⁶)")
        eta = [record["eta_seconds"] / 60 for record in records]
        draw(axes[1, 1], epochs, eta, endpoint=f"{eta[-1]:.2f} min")
        axes[1, 1].set_title("Estimated time remaining", loc="left", fontsize=11)
        axes[1, 1].set_ylabel("Minutes")
        memory_axis = axes[1, 2]
        memory_axis.set_title("Training VRAM", loc="left", fontsize=11)
        memory_axis.set_ylabel("GiB")
        gpu_records = [record for record in records if record["vram"] is not None]
        if gpu_records:
            gpu_epochs = [record["epoch"] for record in gpu_records]
            series = (("allocated_gib", "Allocated", "-"), ("reserved_gib", "Reserved", "--"),
                      ("peak_allocated_gib", "Peak allocated", ":"))
            endpoints = [record["vram"][key] for key, _, _ in series for record in gpu_records[-1:]]
            span = max(max(record["vram"]["reserved_gib"] for record in gpu_records), 0.01)
            for index, ((key, name, style), color) in enumerate(zip(series, colors)):
                values = [record["vram"][key] for record in gpu_records]
                # Use the legend when endpoints converge rather than stacking labels.
                separated = all(abs(endpoints[index] - value) > span * 0.08
                                for other, value in enumerate(endpoints) if other != index)
                endpoint = f"{values[-1]:.2f}" if separated else None
                draw(memory_axis, gpu_epochs, values, color=color, style=style, label=name, endpoint=endpoint)
            memory_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
                               ncols=3, frameon=False, fontsize=8)
        else:
            memory_axis.text(0.5, 0.5, "CPU run: no CUDA VRAM", ha="center", va="center",
                             transform=memory_axis.transAxes, color=muted)
        span = max(epochs[-1] - epochs[0], 1)
        for axis in axes.flat:
            axis.set_xlabel("Epoch")
            axis.set_xlim(epochs[0] - span * 0.03, epochs[-1] + span * 0.18)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color=grid, linewidth=0.5)
            axis.set_axisbelow(True)
        for axis in (axes[1, 1], axes[1, 2]):
            axis.set_ylim(bottom=0)
        axes[1, 0].margins(y=0.12)
        fig.text(0.04, 0.015, "Raw epoch means | Exact values: metrics.csv | Training loss does not measure task success",
                 color=muted, fontsize=9)
        fig.text(0.04, 0.044,
                 "Full training history on log loss axes | Insets: latter half on linear axes | All epoch values retained.",
                 color=muted, fontsize=9)
        fig.tight_layout(rect=(0.01, 0.065, 0.99, 0.94), h_pad=2.3, w_pad=2.0)
        try:
            fig.savefig(run_dir / "training_curves.png", dpi=160, facecolor=surface)
            fig.savefig(run_dir / "training_curves.svg", facecolor=surface)
        finally:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate ACT training curves from metrics.jsonl.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    plot_training_curves(args.run_dir)


if __name__ == "__main__":
    main()
