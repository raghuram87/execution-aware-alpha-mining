"""Week 3 deliverable: baseline vs. execution-aware benchmarking.

Runs the two agent-loop modes with their feedback restricted to an
in-sample search window, takes each run's best-reward factor, and
re-evaluates it under: the full Corwin-Schultz + square-root impact model,
and flat 5/10/20 bps one-way cost scenarios — separately on the in-sample
window (for reference) and on a held-out out-of-sample window the agent
never saw (the number that should actually be trusted). Table 1 reports the
out-of-sample results; `table_in_sample` is kept alongside it so the
in-sample/out-of-sample gap (i.e. the degree of search overfitting) is
visible rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .agent_loop import AgentLoopResult, run_agent_loop
from .factor_eval import FactorEvalResult, evaluate_factor
from .llm_client import make_matched_pair


@dataclass
class ExperimentResult:
    baseline: AgentLoopResult
    execution_aware: AgentLoopResult
    table1: pd.DataFrame               # out-of-sample (held-out) — the trustworthy numbers
    table_in_sample: pd.DataFrame      # same factors, scored on the search window — shows the overfitting gap
    split_date: pd.Timestamp
    cost_scenarios: dict[str, dict[str, FactorEvalResult]]  # mode -> "in_sample"/"out_of_sample" -> result


def _scenario_row(
    mode: str, code: str, panels: dict[str, pd.DataFrame], cost_params, eval_window: tuple
) -> tuple[dict, FactorEvalResult]:
    full = evaluate_factor(code, panels=panels, cost_params=cost_params, cost_model="full", eval_window=eval_window)
    flat_results = {
        bps: evaluate_factor(
            code, panels=panels, cost_model="flat", flat_bps=bps, eval_window=eval_window
        ).Net_IR
        for bps in config.FLAT_COST_SCENARIOS_BPS
    }
    row = {
        "Mode": mode,
        "Factor": code,
        "Gross_Sharpe": full.Gross_IR,
        **{f"Net_Sharpe_{int(bps)}bps": flat_results[bps] for bps in config.FLAT_COST_SCENARIOS_BPS},
        "Net_Sharpe_full_cost_model": full.Net_IR,
        "Annual_Turnover_pct": full.Turnover * 100,
        "Max_Drawdown": full.max_drawdown,
        "Cumulative_Return": full.cumulative_return,
    }
    return row, full


def run_experiment(
    llm_kind: str = "mock",
    llm_kwargs: dict | None = None,
    matched_seed: int | None = None,
    n_iterations: int = config.N_ITERATIONS,
    gamma1: float = config.GAMMA_1,
    gamma2: float = config.GAMMA_2,
    panels: dict[str, pd.DataFrame] | None = None,
    cost_params: config.CostParams | None = None,
    train_fraction: float = config.TRAIN_FRACTION,
    log_dir: str | Path | None = config.LOGS,
    verbose: bool = True,
) -> ExperimentResult:
    """`llm_kind`/`llm_kwargs` build a *matched pair* of LLM clients (see
    `llm_client.make_matched_pair`) — one per mode, independently
    constructed but identically seeded/parameterized — rather than sharing
    one client instance sequentially across both 50-round runs. That keeps
    baseline and execution-aware apples-to-apples: same seed alpha (both
    bootstrap from `config.SEED_ALPHA`), same LLM parameters, same starting
    random/sampling state, differing only in which reward function shapes
    the search from round 1 onward."""
    if panels is None:
        from .data_pipeline import get_panels
        panels = get_panels()

    from .data_pipeline import train_test_split_date
    split_date = train_test_split_date(panels, train_fraction)
    in_sample_window = (None, split_date)
    out_of_sample_window = (split_date, None)
    if verbose:
        print(f"[experiment] in-sample search window: <= {split_date.date()}  "
              f"| held-out out-of-sample window: > {split_date.date()}")

    baseline_client, ea_client = make_matched_pair(llm_kind, seed=matched_seed, **(llm_kwargs or {}))

    log_dir = Path(log_dir) if log_dir else None
    baseline = run_agent_loop(
        baseline_client, mode="baseline", n_iterations=n_iterations, gamma1=gamma1, gamma2=gamma2,
        panels=panels, cost_params=cost_params, eval_window=in_sample_window,
        log_path=(log_dir / "baseline_log.csv") if log_dir else None, verbose=verbose,
    )
    execution_aware = run_agent_loop(
        ea_client, mode="execution_aware", n_iterations=n_iterations, gamma1=gamma1, gamma2=gamma2,
        panels=panels, cost_params=cost_params, eval_window=in_sample_window,
        log_path=(log_dir / "execution_aware_log.csv") if log_dir else None, verbose=verbose,
    )

    oos_rows, is_rows = [], []
    cost_scenarios: dict[str, dict[str, FactorEvalResult]] = {}
    for label, run in (("Baseline", baseline), ("Execution-Aware", execution_aware)):
        code = run.best_result.factor_code
        is_row, is_full = _scenario_row(label, code, panels, cost_params, in_sample_window)
        oos_row, oos_full = _scenario_row(label, code, panels, cost_params, out_of_sample_window)
        is_rows.append(is_row)
        oos_rows.append(oos_row)
        cost_scenarios[label] = {"in_sample": is_full, "out_of_sample": oos_full}

    table1 = pd.DataFrame(oos_rows).set_index("Mode")
    table_in_sample = pd.DataFrame(is_rows).set_index("Mode")
    return ExperimentResult(
        baseline=baseline, execution_aware=execution_aware,
        table1=table1, table_in_sample=table_in_sample, split_date=split_date,
        cost_scenarios=cost_scenarios,
    )


def save_table1(table1: pd.DataFrame, out_dir: str | Path = config.TABLES, name: str = "table1") -> None:
    out_dir = Path(out_dir)
    table1.to_csv(out_dir / f"{name}.csv")
    with open(out_dir / f"{name}.md", "w") as f:
        f.write(table1.round(4).to_markdown())
