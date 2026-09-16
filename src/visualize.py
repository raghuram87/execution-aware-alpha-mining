"""Week 4 visual assets: Figure 1 (architecture flow chart) and the
cumulative-return comparison chart used alongside Table 1.

Static, print-oriented matplotlib figures (PDF + PNG at 300dpi) for the FRL
manuscript. Colors are the validated categorical palette from the dataviz
skill: series-1 blue (#2a78d6) and series-2 orange (#eb6834), a CVD-safe
adjacent pair.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config

COLOR_BASELINE = "#2a78d6"       # categorical slot 1 (blue)
COLOR_EXECUTION_AWARE = "#eb6834"  # categorical slot 2 (orange)
COLOR_SEED_ONLY = "#1baf7a"      # categorical slot 3 (aqua) -- no-LLM-feedback control series
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_GRID = "#e3e2dd"


def plot_cumulative_returns(
    net_returns: dict[str, pd.Series],
    out_path: str | Path = config.FIGURES / "fig2_cumulative_returns.png",
    title: str = "Out-of-Sample Cumulative Net-of-Cost Return: Baseline vs. Execution-Aware",
    fold_boundaries: list[pd.Timestamp] | None = None,
) -> Path:
    """net_returns: {"Baseline": series, "Execution-Aware": series, ...} of
    daily net-of-cost returns (full cost model), same convention as
    FactorEvalResult.net_returns / StitchedResult.net_returns. An optional
    "Seed-Only (no LLM)" key is recognized and colored as a third
    categorical series (the no-LLM-feedback control).

    `fold_boundaries`, if given, draws a light vertical line at each
    walk-forward refit date (each fold's test_start) so re-optimization
    points — where the live factor changes — are visible on the stitched
    equity curve."""
    colors = {
        "Baseline": COLOR_BASELINE,
        "Execution-Aware": COLOR_EXECUTION_AWARE,
        "Seed-Only (no LLM)": COLOR_SEED_ONLY,
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for label, series in net_returns.items():
        cum = (1 + series.dropna()).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, linewidth=2.0, color=colors.get(label, COLOR_TEXT), label=label)

    # End-of-line labels, decluttered vertically so two series ending at
    # similar values (e.g. both near -100%) don't render on top of each other.
    ax.relim()
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    min_gap = 0.06 * (ymax - ymin)

    entries = []
    for label, series in net_returns.items():
        cum = (1 + series.dropna()).cumprod() - 1
        entries.append({"label": label, "x": cum.index[-1], "y": float(cum.iloc[-1] * 100)})
    entries.sort(key=lambda e: -e["y"])
    entries[0]["y_adj"] = entries[0]["y"]
    for i in range(1, len(entries)):
        prev_adj = entries[i - 1]["y_adj"]
        entries[i]["y_adj"] = min(entries[i]["y"], prev_adj - min_gap)

    xmin, xmax = ax.get_xlim()
    x_text = xmax + 0.01 * (xmax - xmin)
    for e in entries:
        color = colors.get(e["label"], COLOR_TEXT)
        ax.annotate(
            f"{e['label']}: {e['y']:+.0f}%",
            xy=(e["x"], e["y"]), xycoords="data",
            xytext=(x_text, e["y_adj"]), textcoords="data",
            va="center", ha="left", fontsize=9, color=color, annotation_clip=False,
        )

    if fold_boundaries:
        for b in fold_boundaries:
            ax.axvline(b, color=COLOR_GRID, linewidth=1.0, linestyle="--", zorder=0)
        ax.text(fold_boundaries[0], 1.0, "  refit points", transform=ax.get_xaxis_transform(),
                fontsize=7, color=COLOR_TEXT_SECONDARY, style="italic", va="bottom")

    ax.set_title(title, fontsize=11, color=COLOR_TEXT, pad=12)
    ax.set_ylabel("Cumulative return (%)", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=COLOR_TEXT)
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


_BPS_COLORS = {
    "full_cost_model": "#0b0b0b",   # realistic Corwin-Schultz + impact model, emphasized in black
    "5bps": "#2a78d6",
    "10bps": "#6aa6e8",
    "20bps": "#c7ddf7",
}
_BPS_ORDER = ["full_cost_model", "5bps", "10bps", "20bps"]
_BPS_DISPLAY = {"full_cost_model": "Full cost model", "5bps": "5 bps flat", "10bps": "10 bps flat", "20bps": "20 bps flat"}


def plot_bps_sensitivity(
    stitched_by_mode: dict[str, dict[str, "object"]],
    out_path: str | Path = config.FIGURES / "fig3_bps_sensitivity.png",
    title: str = "Cost Sensitivity: Cumulative Net Return Across Cost Scenarios",
) -> Path:
    """`stitched_by_mode`: {"Baseline": {"full_cost_model": StitchedResult,
    "5bps": StitchedResult, "10bps": ..., "20bps": ...}, "Execution-Aware":
    {...}} -- exactly `WFOExperimentResult.stitched` from walk_forward.py.
    Renders one panel per mode (side by side) so the reader can see, within
    each mode, how much cumulative return erodes as the assumed one-way cost
    rises from the realistic full cost model through flat 5/10/20bps
    scenarios -- and, across panels, how much less sensitive the
    execution-aware factor is to that assumption than the baseline factor."""
    modes = list(stitched_by_mode.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(6.6 * len(modes), 4.2), dpi=300, sharey=True)
    if len(modes) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, mode in zip(axes, modes):
        ax.set_facecolor("white")
        scenarios = stitched_by_mode[mode]
        for label in _BPS_ORDER:
            if label not in scenarios:
                continue
            r = scenarios[label]
            cum = (1 + r.net_returns.dropna()).cumprod() - 1
            lw = 2.2 if label == "full_cost_model" else 1.4
            ax.plot(cum.index, cum.values * 100, linewidth=lw, color=_BPS_COLORS[label],
                     label=f"{_BPS_DISPLAY[label]}: {cum.iloc[-1] * 100:+.0f}%")
        ax.set_title(mode, fontsize=11, color=COLOR_TEXT, pad=10)
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8)
        ax.grid(False, axis="x")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COLOR_GRID)
        ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9)
        ax.legend(frameon=False, loc="lower left", fontsize=8, labelcolor=COLOR_TEXT)

    axes[0].set_ylabel("Cumulative net return (%)", fontsize=10, color=COLOR_TEXT_SECONDARY)
    fig.suptitle(title, fontsize=12, color=COLOR_TEXT, y=1.02)
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grid_cumulative_returns(
    curves: dict[str, dict[str, pd.Series]],
    fold_boundaries: dict[str, list] | None = None,
    out_path: str | Path = config.FIGURES / "fig2_cumulative_returns_grid.png",
    title: str = "Out-of-Sample Cumulative Net Return by Seed-Alpha Category (representative seed)",
) -> Path:
    """Paper Figure 2: one small-multiples panel per seed-alpha category
    (curves.keys(), e.g. reversal/volume/volatility/momentum), each showing
    Baseline vs. Execution-Aware stitched net-of-cost cumulative return for
    a single representative LLM sampling seed -- Table 1 in the manuscript
    carries the full cross-seed mean/std; this figure is illustrative of one
    concrete run per category, not a replacement for that table."""
    categories = list(curves.keys())
    n = len(categories)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 3.6 * nrows), dpi=300, squeeze=False)
    fig.patch.set_facecolor("white")
    colors = {"Baseline": COLOR_BASELINE, "Execution-Aware": COLOR_EXECUTION_AWARE}

    for i, cat in enumerate(categories):
        ax = axes[i // ncols][i % ncols]
        ax.set_facecolor("white")
        for label, series in curves[cat].items():
            cum = (1 + series.dropna()).cumprod() - 1
            ax.plot(cum.index, cum.values * 100, linewidth=1.8, color=colors.get(label, COLOR_TEXT),
                     label=f"{label}: {cum.iloc[-1] * 100:+.0f}%")
        if fold_boundaries and cat in fold_boundaries:
            for b in fold_boundaries[cat]:
                ax.axvline(b, color=COLOR_GRID, linewidth=0.8, linestyle="--", zorder=0)
        ax.set_title(cat.capitalize(), fontsize=10.5, color=COLOR_TEXT, pad=8)
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8)
        ax.grid(False, axis="x")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COLOR_GRID)
        ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=8)
        ax.legend(frameon=False, loc="lower left", fontsize=7.5, labelcolor=COLOR_TEXT)
        if i % ncols == 0:
            ax.set_ylabel("Cumulative net return (%)", fontsize=9, color=COLOR_TEXT_SECONDARY)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=11.5, color=COLOR_TEXT, y=1.01)
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grid_bps_sensitivity(
    summary_by_category: dict[str, dict[str, dict[str, float]]],
    out_path: str | Path = config.FIGURES / "fig3_bps_sensitivity_grid.png",
    title: str = "Net Sharpe vs. Assumed Trading Cost, by Seed-Alpha Category (mean across 3 LLM seeds)",
) -> Path:
    """Paper Figure 3: one small-multiples panel per seed-alpha category,
    each plotting Net Sharpe (full cost model) at three flat-cost book-ends
    plus the realistic full cost model, for Baseline vs. Execution-Aware.

    `summary_by_category`: {category: {mode: {"5bps": x, "10bps": x,
    "20bps": x, "full_cost_model": x}}} -- means across sampling seeds
    (e.g. built from grid_summary.csv), so this figure and Table 1 draw on
    the identical underlying numbers."""
    scenario_order = ["5bps", "10bps", "20bps", "full_cost_model"]
    scenario_labels = ["5bps", "10bps", "20bps", "Full model"]
    categories = list(summary_by_category.keys())
    n = len(categories)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.6 * nrows), dpi=300, squeeze=False)
    fig.patch.set_facecolor("white")
    colors = {"Baseline": COLOR_BASELINE, "Execution-Aware": COLOR_EXECUTION_AWARE}
    x = range(len(scenario_order))

    for i, cat in enumerate(categories):
        ax = axes[i // ncols][i % ncols]
        ax.set_facecolor("white")
        for mode, color in colors.items():
            vals = [summary_by_category[cat][mode][s] for s in scenario_order]
            ax.plot(x, vals, marker="o", markersize=4, linewidth=1.8, color=color, label=mode)
        ax.axhline(0, color=COLOR_TEXT_SECONDARY, linewidth=0.8, linestyle=":", zorder=0)
        ax.set_xticks(list(x))
        ax.set_xticklabels(scenario_labels, fontsize=8)
        ax.set_title(cat.capitalize(), fontsize=10.5, color=COLOR_TEXT, pad=8)
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8)
        ax.grid(False, axis="x")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COLOR_GRID)
        ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=8)
        ax.legend(frameon=False, loc="lower left", fontsize=7.5, labelcolor=COLOR_TEXT)
        if i % ncols == 0:
            ax.set_ylabel("Net Sharpe", fontsize=9, color=COLOR_TEXT_SECONDARY)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=11.5, color=COLOR_TEXT, y=1.01)
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


COLOR_ACCENT = "#4a3aa7"      # categorical slot 7 (violet) -- Fig 1's Cost Engine highlight only;
                               # deliberately NOT slot 2 orange, which means "Execution-Aware" in Figs 2-3
COLOR_ACCENT_FILL = "#efebfa"
COLOR_NEUTRAL_FILL = "#f7f7f5"
COLOR_NEUTRAL_EDGE = "#8a8977"
COLOR_REGION_FILL = "#f8f7fc"
COLOR_REGION_EDGE = "#ded9f0"


def _box(ax, xy, w, h, title, subtitle, facecolor, edgecolor, title_color=None):
    """Two-tier label (bold component name + regular detail line) inside a
    clean rounded card -- consistent corner radius and stroke weight across
    every box is what reads as "designed" rather than default-matplotlib."""
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.018,rounding_size=0.09",
        linewidth=1.4, edgecolor=edgecolor, facecolor=facecolor,
        joinstyle="round",
    )
    ax.add_patch(box)
    cx = xy[0] + w / 2
    title_color = title_color or COLOR_TEXT
    if subtitle:
        ax.text(cx, xy[1] + h * 0.66, title, ha="center", va="center",
                 fontsize=9.2, color=title_color, fontweight="bold")
        ax.text(cx, xy[1] + h * 0.30, subtitle, ha="center", va="center",
                 fontsize=7.8, color=COLOR_TEXT_SECONDARY, linespacing=1.5)
    else:
        ax.text(cx, xy[1] + h / 2, title, ha="center", va="center",
                 fontsize=9.2, color=title_color, fontweight="bold")
    return box


def _arrow(ax, start, end, color=COLOR_TEXT_SECONDARY, connectionstyle="arc3,rad=0.0", label=None, label_pos=None):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.6,
        color=color, connectionstyle=connectionstyle, shrinkA=3, shrinkB=3,
        capstyle="round",
    )
    ax.add_patch(arrow)
    if label:
        lx, ly = label_pos if label_pos else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.8, color=COLOR_TEXT,
                 style="italic", bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                            edgecolor="none", alpha=0.92))


def _pill(ax, xy, text, facecolor, textcolor):
    ax.text(xy[0], xy[1], text, ha="left", va="center", fontsize=7.6, color=textcolor,
             fontweight="bold", family="sans-serif",
             bbox=dict(boxstyle="round,pad=0.32", facecolor=facecolor, edgecolor="none"))


def plot_architecture(out_path: str | Path = config.FIGURES / "fig1_architecture.png") -> Path:
    """Figure 1: LLM outer loop (propose factor, repeated each round) wrapping
    a backtest inner loop (evaluate -> cost engine -> diagnostics), evaluated
    once per candidate."""
    fig, ax = plt.subplots(figsize=(8.6, 5.1), dpi=300)
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    llm_box = (0.5, 4.55, 3.15, 1.15)
    code_box = (6.35, 4.75, 3.25, 0.95)
    eval_box = (6.35, 3.35, 3.25, 0.95)
    cost_box = (6.35, 1.95, 3.25, 0.95)
    diag_box = (0.5, 1.55, 3.15, 1.5)

    # Inner-loop grouping: a soft filled region (not a crude dashed outline)
    # sized to fully contain its three boxes with even padding.
    region_pad = 0.28
    region_xy = (code_box[0] - region_pad, cost_box[1] - region_pad)
    region_w = code_box[2] + 2 * region_pad
    region_h = (code_box[1] + code_box[3]) - cost_box[1] + 2 * region_pad
    ax.add_patch(FancyBboxPatch(
        region_xy, region_w, region_h, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2, edgecolor=COLOR_REGION_EDGE, facecolor=COLOR_REGION_FILL, zorder=0,
    ))
    _pill(ax, (region_xy[0] + 0.18, region_xy[1] + region_h - 0.22),
          "INNER LOOP · per candidate", COLOR_REGION_EDGE, "white")
    _pill(ax, (llm_box[0] + 0.05, llm_box[1] + llm_box[3] + 0.28),
          "OUTER LOOP · N rounds", COLOR_BASELINE, "white")

    _box(ax, llm_box[:2], llm_box[2], llm_box[3], "LLM Agent",
         "system prompt +\nround feedback", "#eaf2fc", COLOR_BASELINE, title_color=COLOR_BASELINE)
    _box(ax, code_box[:2], code_box[2], code_box[3], "factor_code",
         "DSL expression", COLOR_NEUTRAL_FILL, COLOR_NEUTRAL_EDGE)
    _box(ax, eval_box[:2], eval_box[2], eval_box[3], "Factor Evaluator",
         "scores → weights → turnover", COLOR_NEUTRAL_FILL, COLOR_NEUTRAL_EDGE)
    _box(ax, cost_box[:2], cost_box[2], cost_box[3], "Cost Engine",
         "Corwin-Schultz spread +\nsqrt impact model", COLOR_ACCENT_FILL, COLOR_ACCENT, title_color=COLOR_ACCENT)
    _box(ax, diag_box[:2], diag_box[2], diag_box[3], "Diagnostics",
         "Gross_IR, Turnover,\nCost_Impact, Net_IR, Reward", "#eaf2fc", COLOR_BASELINE, title_color=COLOR_BASELINE)

    # outer loop: LLM -> code -> evaluator -> cost -> diagnostics -> back to LLM
    _arrow(ax, (llm_box[0] + llm_box[2], llm_box[1] + llm_box[3] * 0.6),
           (code_box[0], code_box[1] + code_box[3] * 0.5),
           label="propose", label_pos=(4.95, 5.42))
    _arrow(ax, (code_box[0] + code_box[2] / 2, code_box[1]),
           (eval_box[0] + eval_box[2] / 2, eval_box[1] + eval_box[3]),
           label="backtest", label_pos=(code_box[0] + code_box[2] / 2 + 1.05, (code_box[1] + eval_box[1] + eval_box[3]) / 2))
    _arrow(ax, (eval_box[0] + eval_box[2] / 2, eval_box[1]),
           (cost_box[0] + cost_box[2] / 2, cost_box[1] + cost_box[3]),
           label="turnover", label_pos=(eval_box[0] + eval_box[2] / 2 + 1.0, (eval_box[1] + cost_box[1] + cost_box[3]) / 2))
    _arrow(ax, (cost_box[0], cost_box[1] + cost_box[3] * 0.35),
           (diag_box[0] + diag_box[2], diag_box[1] + diag_box[3] * 0.3),
           connectionstyle="arc3,rad=-0.18", label="net returns", label_pos=(5.0, 1.62))
    _arrow(ax, (diag_box[0] + diag_box[2] * 0.8, diag_box[1] + diag_box[3]),
           (llm_box[0] + llm_box[2] * 0.8, llm_box[1]),
           connectionstyle="arc3,rad=0.0", label="feedback (reward)", label_pos=(4.15, 3.2))

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path
