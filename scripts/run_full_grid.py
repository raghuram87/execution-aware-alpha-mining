"""Full robustness grid: 3 LLM sampling seeds x 3 seed alphas (different
economic categories), each run as a full walk-forward optimization (5y
train / 1y test, rolling 1y, first test window anchored at 2006) over the
point-in-time S&P universe (src/pit_universe.py: local DB + yfinance
gap-fill, true historical constituent membership rather than a static
current snapshot).

Each of the 9 (seed, alpha) combinations is a fully independent WFO run
(~20 folds x 2 modes x N_ITERATIONS rounds each) and is checkpointed to its
own directory immediately on completion, with the running grid_summary.csv
rewritten after every combination -- a crash/interrupt partway through
still leaves every already-finished combination's results on disk and in
the summary table.

Usage: python scripts/run_full_grid.py --seeds 1001 2002 3003 --n-iterations 50
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from papers.execution_aware_alpha_mining.src import config
from papers.execution_aware_alpha_mining.src.pit_universe import get_pit_panels
from papers.execution_aware_alpha_mining.src.visualize import plot_bps_sensitivity, plot_cumulative_returns
from papers.execution_aware_alpha_mining.src.walk_forward import run_wfo_experiment, save_wfo_tables

OUT_ROOT = config.RESULTS / "local_llm_runs" / "full_grid_2006_2026"


def run_one_combo(alpha_name, alpha_code, llm_seed, panels, n_iterations, out_root, verbose=True):
    combo_dir = out_root / alpha_name / f"seed_{llm_seed}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    log_dir = combo_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 78}\nALPHA={alpha_name} ({alpha_code})  LLM_SEED={llm_seed}\n{'=' * 78}")
    result = run_wfo_experiment(
        llm_kind="local",
        matched_seed=llm_seed,
        n_iterations=n_iterations,
        panels=panels,
        train_years=config.WFO_TRAIN_YEARS,
        test_years=config.WFO_TEST_YEARS,
        step_years=config.WFO_STEP_YEARS,
        train_start=config.PIT_TRAIN_START,
        seed_alpha=alpha_code,
        log_dir=log_dir,
        verbose=verbose,
    )

    save_wfo_tables(result, out_dir=combo_dir)
    result.table1.to_csv(combo_dir / "table1_wfo.csv")

    fold_boundaries = [w.test_start for w in result.windows]
    plot_cumulative_returns(
        {label: result.stitched[label]["full_cost_model"].net_returns for label in ("Baseline", "Execution-Aware")},
        out_path=combo_dir / "fig2_cumulative_returns.png",
        title=f"OOS Cumulative Net Return -- seed_alpha={alpha_name}, llm_seed={llm_seed}",
        fold_boundaries=fold_boundaries,
    )
    plot_bps_sensitivity(result.stitched, out_path=combo_dir / "fig3_bps_sensitivity.png")

    for label in ("Baseline", "Execution-Aware"):
        result.stitched[label]["full_cost_model"].net_returns.to_csv(
            combo_dir / f"net_returns_{label.lower().replace('-', '_')}.csv"
        )
        result.stitched[label]["full_cost_model"].daily_turnover.to_csv(
            combo_dir / f"daily_turnover_{label.lower().replace('-', '_')}.csv"
        )

    rows = []
    for label in ("Baseline", "Execution-Aware"):
        row = result.table1.loc[label].to_dict()
        row.update({"seed_alpha": alpha_name, "seed_alpha_code": alpha_code, "llm_seed": llm_seed, "mode": label})
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-iterations", type=int, default=config.N_ITERATIONS)
    parser.add_argument("--alphas", nargs="+", default=list(config.SEED_ALPHAS.keys()))
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[run_full_grid] loading full point-in-time panels (cached if already built)...")
    # Uncapped (max_tickers=None): each fold selects its own ~150-ticker
    # universe causally from this full backing panel (walk_forward._run_fold
    # -> pit_universe.select_fold_tickers, ranked on trailing liquidity as
    # of that fold's train_start only) rather than one fixed liquidity-
    # ranked list chosen from the full 1999-2026 sample, which would let an
    # early fold trade among names that only became liquid decades later.
    panels = get_pit_panels(fetch_start=config.PIT_FETCH_START, panel_start=config.PIT_FETCH_START, max_tickers=None)
    print(f"[run_full_grid] panels: {panels['close'].shape[1]} tickers x {panels['close'].shape[0]} days "
          f"({panels['close'].index.min().date()} .. {panels['close'].index.max().date()})")

    summary_path = out_root / "grid_summary.csv"
    all_rows: list[dict] = []
    if summary_path.exists():
        all_rows = pd.read_csv(summary_path).to_dict("records")
        done = {(r["seed_alpha"], r["llm_seed"]) for r in all_rows}
        print(f"[run_full_grid] resuming: {len(done)} (alpha, seed) combos already in summary")
    else:
        done = set()

    for alpha_name in args.alphas:
        alpha_code = config.SEED_ALPHAS[alpha_name]
        for llm_seed in args.seeds:
            if (alpha_name, llm_seed) in done:
                print(f"[run_full_grid] skipping already-completed alpha={alpha_name} seed={llm_seed}")
                continue
            try:
                rows = run_one_combo(alpha_name, alpha_code, llm_seed, panels, args.n_iterations, out_root)
                all_rows.extend(rows)
            except Exception:
                print(f"[run_full_grid] COMBO FAILED: alpha={alpha_name} seed={llm_seed}")
                traceback.print_exc()
                continue
            pd.DataFrame(all_rows).to_csv(summary_path, index=False)
            print(f"[run_full_grid] checkpointed grid_summary.csv ({len(all_rows)} rows)")

    print(f"\n{'=' * 78}\nFULL GRID COMPLETE\n{'=' * 78}")
    print(pd.DataFrame(all_rows).round(4).to_string(index=False))
