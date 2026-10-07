"""Realized trading-cost diagnostics for every grid arm: annualized gross
return, annual cost drag, mean one-way daily turnover and the average
one-way cost per unit of notional traded, re-scored on each fold's test
window from the logged winning factors.

Usage: python scripts/cost_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.factor_eval import evaluate_factor
from src.pit_universe import get_pit_panels
from src.walk_forward import fold_panels, generate_wfo_windows

GRID_DIR = config.RESULTS / "local_llm_runs" / "full_grid_sharadar_2006_2026"
OUT = config.RESULTS / "local_llm_runs" / "cost_diagnostics_sharadar.csv"

if __name__ == "__main__":
    panels = get_pit_panels(fetch_start=config.PIT_FETCH_START, panel_start=config.PIT_FETCH_START, max_tickers=None)
    windows = generate_wfo_windows(panels, config.WFO_TRAIN_YEARS, config.WFO_TEST_YEARS, config.WFO_STEP_YEARS,
                                   train_start=config.PIT_TRAIN_START)
    fold_panel_cache = {w.fold_id: fold_panels(panels, w) for w in windows}
    rows = []
    for f in sorted(GRID_DIR.glob("*/seed_*/wfo_folds.csv")):
        cat, seed = f.parent.parent.name, int(f.parent.name.split("_")[1])
        folds = pd.read_csv(f)
        for mode, g in folds.groupby("Mode"):
            gross, cost, turnover = [], [], []
            for w, code in zip(windows, g.sort_values("Fold")["Factor"]):
                r = evaluate_factor(code, panels=fold_panel_cache[w.fold_id], eval_window=(w.test_start, w.test_end))
                gross.append(r.gross_returns)
                cost.append(r.gross_returns - r.net_returns)
                turnover.append(r.daily_turnover)
            gross, cost, turnover = (pd.concat(x) for x in (gross, cost, turnover))
            rows.append({
                "seed_alpha": cat, "llm_seed": seed, "mode": mode,
                "gross_return_ann": gross.mean() * config.TRADING_DAYS_PER_YEAR,
                "cost_drag_ann": cost.mean() * config.TRADING_DAYS_PER_YEAR,
                "one_way_turnover_daily": turnover.mean(),
                # cost is charged on the full (two-sided) weight change = 2 x one-way turnover
                "cost_bps_per_unit_traded": 1e4 * cost.mean() / (2 * turnover.mean()),
            })
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
