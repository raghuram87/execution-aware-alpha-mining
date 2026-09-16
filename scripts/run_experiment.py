"""Week 3: run the baseline vs. execution-aware agent-loop experiment end to
end and write Table 1 + the cumulative-return figure.

Usage:
  python scripts/run_experiment.py --llm mock                # no API key needed
  python scripts/run_experiment.py --llm claude               # uses ANTHROPIC_API_KEY
  python scripts/run_experiment.py --llm mock --n-iterations 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from papers.execution_aware_alpha_mining.src import config
from papers.execution_aware_alpha_mining.src.data_pipeline import get_panels
from papers.execution_aware_alpha_mining.src.experiment import run_experiment, save_table1
from papers.execution_aware_alpha_mining.src.visualize import plot_architecture, plot_cumulative_returns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["mock", "claude", "local"], default="mock")
    parser.add_argument("--n-iterations", type=int, default=config.N_ITERATIONS)
    parser.add_argument("--gamma1", type=float, default=config.GAMMA_1)
    parser.add_argument("--gamma2", type=float, default=config.GAMMA_2)
    parser.add_argument("--seed", type=int, default=None,
                         help="shared seed for the baseline/execution-aware matched client pair")
    args = parser.parse_args()

    panels = get_panels()

    result = run_experiment(
        llm_kind=args.llm, matched_seed=args.seed,
        n_iterations=args.n_iterations, gamma1=args.gamma1, gamma2=args.gamma2, panels=panels,
    )

    save_table1(result.table1, name="table1")
    save_table1(result.table_in_sample, name="table1_in_sample")
    print(f"\nSplit date: in-sample <= {result.split_date.date()}, out-of-sample > {result.split_date.date()}")
    print("\n=== Table 1 (out-of-sample / held-out) ===")
    print(result.table1.round(4).to_string())
    print("\n=== In-sample (search window) reference, for the overfitting gap ===")
    print(result.table_in_sample.round(4).to_string())

    net_returns = {
        "Baseline": result.cost_scenarios["Baseline"]["out_of_sample"].net_returns,
        "Execution-Aware": result.cost_scenarios["Execution-Aware"]["out_of_sample"].net_returns,
    }
    fig_path = plot_cumulative_returns(net_returns)
    arch_path = plot_architecture()
    print(f"\nSaved: {config.TABLES/'table1.csv'}, {config.TABLES/'table1.md'}, "
          f"{config.TABLES/'table1_in_sample.csv'}, {config.TABLES/'table1_in_sample.md'}")
    print(f"Saved: {fig_path} (out-of-sample only), {arch_path}")
