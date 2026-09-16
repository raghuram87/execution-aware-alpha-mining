"""Generate the paper's Figure 2 (cumulative returns, small multiples by
seed-alpha category) and Figure 3 (bps cost sensitivity, small multiples)
from the completed full_grid_2006_2026 run -- no rerun needed, everything
is read from the already-saved per-combo CSVs and grid_summary.csv.

Usage: python scripts/generate_paper_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.visualize import plot_grid_bps_sensitivity, plot_grid_cumulative_returns

GRID_DIR = config.RESULTS / "local_llm_runs" / "full_grid_2006_2026"
CATEGORIES = ["reversal", "volume", "volatility", "momentum"]
REPRESENTATIVE_SEED = 1001
COST_SCENARIOS = ["Net_Sharpe_5bps", "Net_Sharpe_10bps", "Net_Sharpe_20bps", "Net_Sharpe_full_cost_model"]
SCENARIO_KEYS = ["5bps", "10bps", "20bps", "full_cost_model"]


def load_curves() -> tuple[dict, dict]:
    curves, fold_boundaries = {}, {}
    for cat in CATEGORIES:
        combo_dir = GRID_DIR / cat / f"seed_{REPRESENTATIVE_SEED}"
        baseline = pd.read_csv(combo_dir / "net_returns_baseline.csv", index_col=0, parse_dates=True).iloc[:, 0]
        ea = pd.read_csv(combo_dir / "net_returns_execution_aware.csv", index_col=0, parse_dates=True).iloc[:, 0]
        curves[cat] = {"Baseline": baseline, "Execution-Aware": ea}

        folds = pd.read_csv(combo_dir / "wfo_folds.csv")
        test_starts = folds.loc[folds["Mode"] == "Baseline", "Test"].str.split(" to ").str[0]
        fold_boundaries[cat] = pd.to_datetime(test_starts).tolist()
    return curves, fold_boundaries


def load_bps_summary() -> dict:
    summary = pd.read_csv(GRID_DIR / "grid_summary.csv")
    out = {}
    for cat in CATEGORIES:
        out[cat] = {}
        for mode in ("Baseline", "Execution-Aware"):
            rows = summary[(summary["seed_alpha"] == cat) & (summary["mode"] == mode)]
            means = rows[COST_SCENARIOS].mean()
            out[cat][mode] = dict(zip(SCENARIO_KEYS, means.values))
    return out


if __name__ == "__main__":
    curves, fold_boundaries = load_curves()
    path2 = plot_grid_cumulative_returns(
        curves, fold_boundaries=fold_boundaries,
        out_path=config.FIGURES / "fig2_cumulative_returns.png",
    )
    print(f"Wrote {path2} (+ .pdf)")

    bps_summary = load_bps_summary()
    path3 = plot_grid_bps_sensitivity(
        bps_summary, out_path=config.FIGURES / "fig3_bps_sensitivity.png",
    )
    print(f"Wrote {path3} (+ .pdf)")
