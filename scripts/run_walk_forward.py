"""Institutional-standard walk-forward experiment: rolls the (train, test)
window forward across the sample, re-running the full agent search on each
fold's in-sample window and scoring the frozen winner on that fold's
out-of-sample window, then stitches the OOS return streams into one
continuous equity curve per mode.

Usage:
  python scripts/run_walk_forward.py --llm mock                          # no API key needed
  python scripts/run_walk_forward.py --llm claude                        # uses ANTHROPIC_API_KEY
  python scripts/run_walk_forward.py --llm mock --train-years 4 --test-years 1 --step-years 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from papers.execution_aware_alpha_mining.src import config
from papers.execution_aware_alpha_mining.src.data_pipeline import get_panels
from papers.execution_aware_alpha_mining.src.visualize import plot_architecture, plot_cumulative_returns
from papers.execution_aware_alpha_mining.src.walk_forward import run_wfo_experiment, save_wfo_tables

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["mock", "claude", "local"], default="mock")
    parser.add_argument("--n-iterations", type=int, default=config.N_ITERATIONS)
    parser.add_argument("--gamma1", type=float, default=config.GAMMA_1)
    parser.add_argument("--gamma2", type=float, default=config.GAMMA_2)
    parser.add_argument("--train-years", type=int, default=config.WFO_TRAIN_YEARS)
    parser.add_argument("--test-years", type=int, default=config.WFO_TEST_YEARS)
    parser.add_argument("--step-years", type=int, default=config.WFO_STEP_YEARS)
    parser.add_argument("--seed", type=int, default=None,
                         help="seed for the one matched baseline/execution-aware client pair, "
                              "constructed once and reused across all folds; omit for a fresh random seed")
    args = parser.parse_args()

    panels = get_panels()

    result = run_wfo_experiment(
        llm_kind=args.llm, matched_seed=args.seed,
        n_iterations=args.n_iterations, gamma1=args.gamma1, gamma2=args.gamma2,
        panels=panels, train_years=args.train_years, test_years=args.test_years,
        step_years=args.step_years,
    )

    save_wfo_tables(result)
    print(f"\n{len(result.windows)} walk-forward folds:")
    for w in result.windows:
        print(f"  fold {w.fold_id}: train {w.train_start.date()}..{w.train_end.date()} "
              f"-> test {w.test_start.date()}..{w.test_end.date()}")

    print("\n=== Table 1 (walk-forward, stitched out-of-sample) ===")
    print(result.table1.round(4).to_string())

    print("\n=== Per-fold winning factor & OOS diagnostics ===")
    print(result.fold_table.round(4).to_string(index=False))

    net_returns = {
        "Baseline": result.stitched["Baseline"]["full_cost_model"].net_returns,
        "Execution-Aware": result.stitched["Execution-Aware"]["full_cost_model"].net_returns,
    }
    fold_boundaries = [w.test_start for w in result.windows]
    fig_path = plot_cumulative_returns(
        net_returns,
        title="Walk-Forward Stitched OOS Cumulative Net-of-Cost Return",
        fold_boundaries=fold_boundaries,
    )
    arch_path = plot_architecture()
    print(f"\nSaved: {config.TABLES/'table1_wfo.csv'}, {config.TABLES/'table1_wfo.md'}, "
          f"{config.TABLES/'wfo_folds.csv'}, {config.TABLES/'wfo_folds.md'}")
    print(f"Saved: {fig_path} (walk-forward stitched), {arch_path}")
