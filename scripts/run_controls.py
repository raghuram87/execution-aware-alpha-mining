"""No-search controls: how much of the execution-aware result could be had
without an LLM search, or with portfolio construction alone?

Every control is scored on the same 21 walk-forward test windows and the
same per-fold liquidity-capped universe as the main grid:

  seed_daily          the seed alpha itself, rebalanced daily
  seed_monthly        the seed alpha, rebalanced every 21 trading days
  seed_smoothed       decay_linear(seed, 252), rebalanced daily: the
                      transformation the cost-aware search converged to,
                      applied by hand (chosen after seeing the results, so it
                      favours this control)
  baseline_monthly    each fold's cost-blind winning factor, rebalanced every
                      21 trading days (needs the main grid's wfo_folds files)

Usage: python scripts/run_controls.py [--grid-dir DIR] [--controls ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.pit_universe import get_pit_panels
from src.walk_forward import evaluate_fixed_factor_across_folds, generate_wfo_windows

GRID_DIR = config.RESULTS / "local_llm_runs" / "full_grid_sharadar_2006_2026"
OUT_DIR = config.RESULTS / "local_llm_runs" / "controls_sharadar"
SEEDS = (1001, 2002, 3003)


def summarize(name, alpha, llm_seed, full, flats):
    avg_to = float(full.daily_turnover.mean())
    return {
        "control": name, "seed_alpha": alpha, "llm_seed": llm_seed,
        "Gross_Sharpe": full.Gross_IR, "Net_Sharpe_full_cost_model": full.Net_IR,
        **{f"Net_Sharpe_{int(b)}bps": flats[b].Net_IR for b in config.FLAT_COST_SCENARIOS_BPS},
        "Avg_Daily_Turnover_pct": avg_to * 100,
        "Max_Drawdown": full.max_drawdown, "Cumulative_Return": full.cumulative_return,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", default=str(GRID_DIR))
    ap.add_argument("--controls", nargs="+",
                    default=["seed_daily", "seed_monthly", "seed_smoothed", "baseline_monthly"])
    ap.add_argument("--alphas", nargs="+", default=list(config.SEED_ALPHAS))
    args = ap.parse_args()
    grid_dir = Path(args.grid_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panels = get_pit_panels(fetch_start=config.PIT_FETCH_START, panel_start=config.PIT_FETCH_START, max_tickers=None)
    windows = generate_wfo_windows(panels, config.WFO_TRAIN_YEARS, config.WFO_TEST_YEARS, config.WFO_STEP_YEARS,
                                   train_start=config.PIT_TRAIN_START)
    summary_path = OUT_DIR / "controls_summary.csv"
    rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    done = {(r["control"], r["seed_alpha"], r["llm_seed"]) for r in rows}

    for alpha in args.alphas:
        seed = config.SEED_ALPHAS[alpha]
        jobs = []
        if "seed_daily" in args.controls:
            jobs.append(("seed_daily", 0, seed, 1))
        if "seed_monthly" in args.controls:
            jobs.append(("seed_monthly", 0, seed, 21))
        if "seed_smoothed" in args.controls:
            jobs.append(("seed_smoothed", 0, f"decay_linear({seed}, 252)", 1))
        if "baseline_monthly" in args.controls:
            for s in SEEDS:
                f = grid_dir / alpha / f"seed_{s}" / "wfo_folds_baseline.csv"
                if not f.exists():
                    f = grid_dir / alpha / f"seed_{s}" / "wfo_folds.csv"
                if not f.exists():
                    print(f"[controls] no grid folds for {alpha}/seed_{s}; skipping baseline_monthly")
                    continue
                folds = pd.read_csv(f)
                folds = folds[folds["Mode"] == "Baseline"].sort_values("Fold")
                if len(folds) != len(windows):
                    print(f"[controls] {alpha}/seed_{s}: {len(folds)} baseline folds != {len(windows)} windows; skipping")
                    continue
                jobs.append(("baseline_monthly", s, folds["Factor"].tolist(), 21))

        for name, llm_seed, code, every in jobs:
            if (name, alpha, llm_seed) in done:
                continue
            print(f"[controls] {name} {alpha} seed={llm_seed} rebalance_every={every}", flush=True)
            full, flats = evaluate_fixed_factor_across_folds(code, panels, windows, rebalance_every=every)
            rows.append(summarize(name, alpha, llm_seed, full, flats))
            full.net_returns.to_csv(OUT_DIR / f"net_returns_{name}_{alpha}_{llm_seed}.csv")
            pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(pd.DataFrame(rows).round(3).to_string(index=False))
