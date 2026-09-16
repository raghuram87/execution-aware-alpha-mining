"""Ad hoc robustness sweep: run the single-split experiment at temp=0.7
(the validated setting -- see manuscript discussion of temp=0/0.3 dead-loop
failures) across several different matched-pair seeds, sequentially (GPU
memory requires this -- two ~9GB model instances per run), saving each
run's key results for cross-seed comparison.

Usage: python scripts/run_seed_sweep.py --seeds 1001 2002 3003 4004
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from papers.execution_aware_alpha_mining.src.data_pipeline import get_panels
from papers.execution_aware_alpha_mining.src.experiment import run_experiment, save_table1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-iterations", type=int, default=50)
    args = parser.parse_args()

    panels = get_panels()
    out_root = Path(__file__).resolve().parent.parent / "results" / "local_llm_runs" / "temp07_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for seed in args.seeds:
        print(f"\n{'=' * 70}\nSEED {seed}\n{'=' * 70}")
        result = run_experiment(
            llm_kind="local", matched_seed=seed,
            n_iterations=args.n_iterations, panels=panels,
            log_dir="results/logs", verbose=True,
        )
        print(result.table1.round(4).to_string())

        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        result.table1.to_csv(seed_dir / "table1.csv")
        shutil.copy("results/logs/baseline_log.csv", seed_dir / "baseline_log.csv")
        shutil.copy("results/logs/execution_aware_log.csv", seed_dir / "execution_aware_log.csv")

        for label in ("Baseline", "Execution-Aware"):
            row = result.table1.loc[label]
            summary_rows.append({
                "seed": seed, "mode": label, "factor": row["Factor"],
                "gross_sharpe": row["Gross_Sharpe"], "net_sharpe_full": row["Net_Sharpe_full_cost_model"],
                "turnover_pct": row["Annual_Turnover_pct"], "max_dd": row["Max_Drawdown"],
                "cum_ret": row["Cumulative_Return"],
            })

    import pandas as pd
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "sweep_summary.csv", index=False)
    print(f"\n\n{'=' * 70}\nFULL SWEEP SUMMARY\n{'=' * 70}")
    print(summary.round(4).to_string(index=False))
