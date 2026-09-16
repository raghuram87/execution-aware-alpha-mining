"""Institutional-standard rolling walk-forward optimization (WFO).

Rather than a single fixed in-sample/out-of-sample split, the discovery
window is rolled forward in time. For each fold:

  Train / agent search : [train_start, train_end]   (LLM re-runs its full
                          search loop from scratch, seeing only diagnostics
                          computed on this window)
  Test / OOS evaluation : [test_start, test_end]     (the winning, now-frozen
                          factor is executed unmodified on this window)

The per-fold OOS return streams are then concatenated in date order into a
single continuous out-of-sample equity curve per mode — the walk-forward
stitched result — which is what Table 1 and Figure 2 should report, since it
reflects many independent re-searches rather than one lucky/unlucky split.

`run_wfo_experiment` constructs one *matched pair* of LLM clients
(`llm_client.make_matched_pair`) once, before the fold loop, and reuses it
across every fold — baseline and execution-aware each get their own
independently-constructed, identically-seeded client, both bootstrapped
from the same `config.SEED_ALPHA` — so the two modes are apples-to-apples
throughout the whole run, not just within one fold. (An earlier version of
this function constructed a fresh pair per fold; besides being 6x more
model loads than necessary for `llm_kind="local"`, it could crash the local
backend with a CUDA OOM, since Python evaluates the new pair's model loads
before releasing the previous fold's clients. See `run_wfo_experiment`'s
docstring for the mechanics.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .agent_loop import run_agent_loop
from .factor_eval import FactorEvalResult, annualized_ir, evaluate_factor, max_drawdown
from .llm_client import LLMClient, make_matched_pair
from .pit_universe import build_fold_universe


@dataclass
class WFOWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_wfo_windows(
    panels: dict[str, pd.DataFrame],
    train_years: int = config.WFO_TRAIN_YEARS,
    test_years: int = config.WFO_TEST_YEARS,
    step_years: int = config.WFO_STEP_YEARS,
    min_test_days: int = config.WFO_MIN_TEST_DAYS,
    train_start: pd.Timestamp | str | None = None,
) -> list[WFOWindow]:
    """Anchor the first train window at the start of the data (or at
    `train_start`, if given -- e.g. a point-in-time panel's raw price
    history extends earlier than the first fold should, purely to warm up
    rolling-window computations; see `pit_universe.get_pit_panels`'s
    `fetch_start`/`panel_start` split) and roll (train, test) forward by
    `step_years` until the test window runs off the end of the sample or
    would be shorter than `min_test_days`."""
    idx = panels["close"].index
    data_start, data_end = idx.min(), idx.max()

    windows: list[WFOWindow] = []
    train_start = pd.Timestamp(train_start) if train_start is not None else data_start
    fold_id = 0
    while True:
        train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1), data_end)

        if test_start > data_end:
            break
        n_test_days = len(idx[(idx >= test_start) & (idx <= test_end)])
        if n_test_days < min_test_days:
            break
        n_train_days = len(idx[(idx >= train_start) & (idx <= train_end)])
        if n_train_days < min_test_days:
            break

        windows.append(WFOWindow(fold_id, train_start, train_end, test_start, test_end))
        train_start = train_start + pd.DateOffset(years=step_years)
        fold_id += 1

    return windows


@dataclass
class WFOFoldResult:
    window: WFOWindow
    mode: str
    factor_code: str
    agent_log: pd.DataFrame = field(repr=False)
    in_sample_result: FactorEvalResult = field(repr=False)
    out_of_sample_result: FactorEvalResult = field(repr=False)          # full cost model
    flat_oos_results: dict[float, FactorEvalResult] = field(repr=False)  # bps -> result


def _run_fold(
    llm_client: LLMClient, mode: str, window: WFOWindow, panels: dict[str, pd.DataFrame],
    n_iterations: int, gamma1: float, gamma2: float, cost_params, log_dir: Path | None, verbose: bool,
    seed_alpha: str = config.SEED_ALPHA,
    fold_universe_size: int | None = 150,
    fold_lookback_days: int = 504,
) -> WFOFoldResult:
    w = window
    if verbose:
        print(f"[wfo:{mode}] fold {w.fold_id}: train {w.train_start.date()}..{w.train_end.date()} "
              f"-> test {w.test_start.date()}..{w.test_end.date()}")

    if fold_universe_size is not None:
        # Causal, periodically-reconstituted per-fold universe: one
        # liquidity reconstitution point per training year plus one at
        # test_start (see pit_universe.build_fold_universe), each ranked
        # only on data strictly before that point -- rather than either (a)
        # a single global ranking over the full 1999-2026 sample (lookahead:
        # a 2001 fold could trade names that only became liquid decades
        # later) or (b) one fixed snapshot at train_start reused through the
        # whole 6-year fold (the test year's universe would then be up to
        # several years stale). Annual reconstitution mirrors how index/
        # universe membership is actually periodically refreshed in
        # practice, without paying for full daily re-screening.
        recon_dates = []
        d = w.train_start
        while d <= w.train_end:
            recon_dates.append(d)
            d = d + pd.DateOffset(years=1)
        recon_dates.append(w.test_start)

        union_tickers, liquid_mask = build_fold_universe(
            panels, recon_dates, w.test_end, fold_universe_size, fold_lookback_days,
        )
        # A name is usable on a given day only if it's BOTH a true
        # point-in-time S&P constituent (panels["eligible"]) AND inside the
        # currently-active liquidity reconstitution block (liquid_mask,
        # which only covers [recon_dates[0], test_end] -- dates before that,
        # kept only for rolling-window warmup, reindex to False and are
        # never in a reported eval_window anyway).
        pit_eligible = panels["eligible"][union_tickers]
        combined_eligible = pit_eligible & liquid_mask.reindex(
            index=pit_eligible.index, columns=union_tickers, fill_value=False
        )
        panels = {k: v[union_tickers] for k, v in panels.items() if k != "eligible"}
        panels["eligible"] = combined_eligible

    # Truncate to dates <= this fold's test_end: evaluate_factor recomputes
    # scores causally over the *entire* passed-in panel every call (no
    # lookahead either way -- see its docstring), so handing it the full,
    # multi-decade point-in-time panel on every one of N_ITERATIONS rounds
    # of every one of ~20 folds means every early fold pays to recompute
    # over 15-20 years of dates it will never look at. Dropping future dates
    # this fold can't see anyway is a pure compute win, not a methodology
    # change -- all rolling-window history *before* test_end is untouched.
    panels_fold = {k: v.loc[:w.test_end] for k, v in panels.items()}
    loop_result = run_agent_loop(
        llm_client, mode=mode, n_iterations=n_iterations, gamma1=gamma1, gamma2=gamma2,
        panels=panels_fold, cost_params=cost_params, eval_window=(w.train_start, w.train_end),
        log_path=(log_dir / f"{mode}_fold{w.fold_id}_log.csv") if log_dir else None,
        verbose=verbose, seed_alpha=seed_alpha,
    )
    code = loop_result.best_result.factor_code
    oos_full = evaluate_factor(
        code, panels=panels_fold, cost_params=cost_params, cost_model="full",
        eval_window=(w.test_start, w.test_end),
    )
    flat_oos = {
        bps: evaluate_factor(
            code, panels=panels_fold, cost_model="flat", flat_bps=bps,
            eval_window=(w.test_start, w.test_end),
        )
        for bps in config.FLAT_COST_SCENARIOS_BPS
    }
    if verbose:
        r = oos_full
        print(f"[wfo:{mode}] fold {w.fold_id} winner: {code}\n"
              f"[wfo:{mode}] fold {w.fold_id} OOS: Gross_IR={r.Gross_IR:.3f} "
              f"Turnover={r.Turnover:.2f} Net_IR={r.Net_IR:.3f}")
    return WFOFoldResult(
        window=w, mode=mode, factor_code=code, agent_log=loop_result.log,
        in_sample_result=loop_result.best_result, out_of_sample_result=oos_full,
        flat_oos_results=flat_oos,
    )


def run_walk_forward(
    llm_client: LLMClient,
    mode: str,
    panels: dict[str, pd.DataFrame] | None = None,
    n_iterations: int = config.N_ITERATIONS,
    gamma1: float = config.GAMMA_1,
    gamma2: float = config.GAMMA_2,
    cost_params: config.CostParams | None = None,
    train_years: int = config.WFO_TRAIN_YEARS,
    test_years: int = config.WFO_TEST_YEARS,
    step_years: int = config.WFO_STEP_YEARS,
    log_dir: str | Path | None = None,
    verbose: bool = True,
    seed_alpha: str = config.SEED_ALPHA,
    train_start: pd.Timestamp | str | None = None,
    fold_universe_size: int | None = 150,
    fold_lookback_days: int = 504,
) -> list[WFOFoldResult]:
    """Run a *single* mode's search independently on every rolling fold with
    one given `llm_client`, scoring each fold's frozen winner on its own
    untouched test window. For the baseline-vs-execution-aware comparison
    (matched, apples-to-apples client pairs per fold), use
    `run_wfo_experiment` instead — this function is for standalone
    single-mode use."""
    if panels is None:
        from .data_pipeline import get_panels
        panels = get_panels()

    windows = generate_wfo_windows(panels, train_years, test_years, step_years, train_start=train_start)
    if not windows:
        raise RuntimeError("No walk-forward folds fit inside the available sample")

    log_dir = Path(log_dir) if log_dir else None
    return [
        _run_fold(llm_client, mode, w, panels, n_iterations, gamma1, gamma2, cost_params, log_dir, verbose,
                  seed_alpha, fold_universe_size, fold_lookback_days)
        for w in windows
    ]


@dataclass
class StitchedResult:
    mode: str
    cost_label: str
    Gross_IR: float
    Turnover: float
    Cost_Impact: float
    Net_IR: float
    max_drawdown: float
    cumulative_return: float
    gross_returns: pd.Series = field(repr=False)
    net_returns: pd.Series = field(repr=False)
    daily_turnover: pd.Series = field(repr=False)
    fold_factors: list[str] = field(repr=False, default_factory=list)


def _concat_sorted(parts: list[pd.Series]) -> pd.Series:
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="first")]


def stitch_folds(
    fold_results: list[WFOFoldResult],
    cost_model: str = "full",
    flat_bps: float | None = None,
) -> StitchedResult:
    """Concatenate each fold's OOS return stream (in date order, using that
    fold's own frozen winning factor) into one continuous equity curve, then
    compute summary statistics on the stitched curve — this is the
    walk-forward analogue of Gross_IR/Turnover/Cost_Impact/Net_IR."""
    if cost_model not in ("full", "flat"):
        raise ValueError("cost_model must be 'full' or 'flat'")
    if cost_model == "flat" and flat_bps is None:
        raise ValueError("flat_bps is required when cost_model='flat'")

    mode = fold_results[0].mode
    gross_parts, net_parts, turnover_parts, codes = [], [], [], []
    for fr in fold_results:
        r = fr.out_of_sample_result if cost_model == "full" else fr.flat_oos_results[flat_bps]
        gross_parts.append(r.gross_returns)
        net_parts.append(r.net_returns)
        turnover_parts.append(r.daily_turnover)
        codes.append(fr.factor_code)

    gross_ret = _concat_sorted(gross_parts)
    net_ret = _concat_sorted(net_parts)
    turnover = _concat_sorted(turnover_parts)
    cost_drag = gross_ret - net_ret

    label = "full_cost_model" if cost_model == "full" else f"{int(flat_bps)}bps"
    return StitchedResult(
        mode=mode,
        cost_label=label,
        Gross_IR=annualized_ir(gross_ret),
        Turnover=float(turnover.mean() * config.TRADING_DAYS_PER_YEAR),
        Cost_Impact=float(cost_drag.mean() * config.TRADING_DAYS_PER_YEAR),
        Net_IR=annualized_ir(net_ret),
        max_drawdown=max_drawdown(net_ret),
        cumulative_return=float((1 + net_ret).prod() - 1),
        gross_returns=gross_ret, net_returns=net_ret, daily_turnover=turnover,
        fold_factors=codes,
    )


def evaluate_fixed_factor_across_folds(
    factor_code: str,
    panels: dict[str, pd.DataFrame],
    windows: list[WFOWindow],
    cost_params: config.CostParams | None = None,
) -> tuple[StitchedResult, dict[float, StitchedResult]]:
    """The no-LLM-feedback control: score ONE fixed, never-refined expression
    (e.g. `config.SEED_ALPHA`) on every fold's OOS test window -- no search,
    no train window, no reward -- then stitch those OOS streams together with
    the exact same `stitch_folds` machinery used for baseline/execution-aware,
    so the comparison is apples-to-apples against the LLM-driven results.
    Returns (full-cost-model StitchedResult, {bps: flat-cost StitchedResult}).
    """
    fold_results = []
    for w in windows:
        oos_full = evaluate_factor(
            factor_code, panels=panels, cost_params=cost_params, cost_model="full",
            eval_window=(w.test_start, w.test_end),
        )
        flat_oos = {
            bps: evaluate_factor(
                factor_code, panels=panels, cost_model="flat", flat_bps=bps,
                eval_window=(w.test_start, w.test_end),
            )
            for bps in config.FLAT_COST_SCENARIOS_BPS
        }
        fold_results.append(WFOFoldResult(
            window=w, mode="seed_only", factor_code=factor_code,
            agent_log=pd.DataFrame(), in_sample_result=oos_full,  # no search occurred; in_sample unused here
            out_of_sample_result=oos_full, flat_oos_results=flat_oos,
        ))
    full = stitch_folds(fold_results, cost_model="full")
    flats = {bps: stitch_folds(fold_results, cost_model="flat", flat_bps=bps) for bps in config.FLAT_COST_SCENARIOS_BPS}
    return full, flats


@dataclass
class WFOExperimentResult:
    windows: list[WFOWindow]
    fold_results: dict[str, list[WFOFoldResult]]      # mode -> per-fold results
    stitched: dict[str, dict[str, StitchedResult]]    # mode -> cost_label -> stitched result
    table1: pd.DataFrame
    fold_table: pd.DataFrame                          # one row per (mode, fold): which factor won, OOS stats


def run_wfo_experiment(
    llm_kind: str = "mock",
    llm_kwargs: dict | None = None,
    matched_seed: int | None = None,
    n_iterations: int = config.N_ITERATIONS,
    gamma1: float = config.GAMMA_1,
    gamma2: float = config.GAMMA_2,
    panels: dict[str, pd.DataFrame] | None = None,
    cost_params: config.CostParams | None = None,
    train_years: int = config.WFO_TRAIN_YEARS,
    test_years: int = config.WFO_TEST_YEARS,
    step_years: int = config.WFO_STEP_YEARS,
    log_dir: str | Path | None = config.LOGS,
    verbose: bool = True,
    seed_alpha: str = config.SEED_ALPHA,
    train_start: pd.Timestamp | str | None = None,
    fold_universe_size: int | None = 150,
    fold_lookback_days: int = 504,
) -> WFOExperimentResult:
    """Constructs ONE matched pair of LLM clients (`llm_client.make_matched_pair`)
    before the fold loop and reuses those same two instances across all
    folds -- baseline and execution-aware each get their own
    independently-constructed, identically-seeded client, called the same
    number of times per fold in lockstep, so the two modes stay
    apples-to-apples (same seed alpha, same LLM parameters, synchronized
    sampling stream) across the *entire* run, not just within one fold.

    This also matters operationally for `llm_kind="local"`: constructing a
    fresh pair *inside* the fold loop was this function's original design,
    and it both reloaded the ~9GB model 12 times (6 folds x 2 clients) for
    no benefit, and could crash with a CUDA OOM -- `baseline_client, ea_client
    = make_matched_pair(...)` evaluates the right-hand side, loading the new
    fold's two models, *before* rebinding the names, so the previous fold's
    two models were still resident on the same GPUs at that moment,
    momentarily needing ~2x the VRAM of one model per card. Constructing
    once avoids this entirely.

    `matched_seed`, if given, seeds the one matched pair (reproducible);
    omit for a fresh random seed (mock still defaults to config.RANDOM_SEED
    if left unset, per `make_matched_pair`)."""
    if panels is None:
        from .data_pipeline import get_panels
        panels = get_panels()

    windows = generate_wfo_windows(panels, train_years, test_years, step_years, train_start=train_start)
    if not windows:
        raise RuntimeError("No walk-forward folds fit inside the available sample")

    log_dir = Path(log_dir) if log_dir else None
    baseline_client, ea_client = make_matched_pair(llm_kind, seed=matched_seed, **(llm_kwargs or {}))
    fold_results: dict[str, list[WFOFoldResult]] = {"baseline": [], "execution_aware": []}
    for w in windows:
        for mode, client in (("baseline", baseline_client), ("execution_aware", ea_client)):
            fold_results[mode].append(_run_fold(
                client, mode, w, panels, n_iterations, gamma1, gamma2, cost_params, log_dir, verbose, seed_alpha,
                fold_universe_size, fold_lookback_days,
            ))

    labels = {"Baseline": "baseline", "Execution-Aware": "execution_aware"}
    stitched: dict[str, dict[str, StitchedResult]] = {}
    table_rows = []
    for label, mode in labels.items():
        frs = fold_results[mode]
        full = stitch_folds(frs, cost_model="full")
        flats = {bps: stitch_folds(frs, cost_model="flat", flat_bps=bps) for bps in config.FLAT_COST_SCENARIOS_BPS}
        stitched[label] = {"full_cost_model": full, **{f"{int(bps)}bps": flats[bps] for bps in config.FLAT_COST_SCENARIOS_BPS}}

        # Avg_Daily_Turnover_pct is the mean one-way daily turnover (% of
        # gross exposure traded per day), not annualized -- Avg_Holding_
        # Period_Days is its inverse (a 5%/day average turnover implies a
        # position is, on average, fully rotated roughly every 20 trading
        # days). Both are computed off the same stitched daily_turnover
        # series `full.Turnover` (annualized) is built from.
        avg_daily_turnover = float(full.daily_turnover.mean())
        holding_period = (1.0 / avg_daily_turnover) if avg_daily_turnover > 0 else float("inf")

        table_rows.append({
            "Mode": label,
            "N_Folds": len(frs),
            "Gross_Sharpe": full.Gross_IR,
            **{f"Net_Sharpe_{int(bps)}bps": flats[bps].Net_IR for bps in config.FLAT_COST_SCENARIOS_BPS},
            "Net_Sharpe_full_cost_model": full.Net_IR,
            "Avg_Daily_Turnover_pct": avg_daily_turnover * 100,
            "Avg_Holding_Period_Days": holding_period,
            "Max_Drawdown": full.max_drawdown,
            "Cumulative_Return": full.cumulative_return,
        })
    table1 = pd.DataFrame(table_rows).set_index("Mode")

    fold_rows = []
    for label, mode in labels.items():
        for fr in fold_results[mode]:
            fold_rows.append({
                "Mode": label,
                "Fold": fr.window.fold_id,
                "Train": f"{fr.window.train_start.date()} to {fr.window.train_end.date()}",
                "Test": f"{fr.window.test_start.date()} to {fr.window.test_end.date()}",
                "Factor": fr.factor_code,
                "OOS_Gross_IR": fr.out_of_sample_result.Gross_IR,
                "OOS_Turnover": fr.out_of_sample_result.Turnover,
                "OOS_Net_IR": fr.out_of_sample_result.Net_IR,
            })
    fold_table = pd.DataFrame(fold_rows)

    return WFOExperimentResult(
        windows=windows,
        fold_results=fold_results, stitched=stitched, table1=table1, fold_table=fold_table,
    )


def save_wfo_tables(result: WFOExperimentResult, out_dir: str | Path = config.TABLES) -> None:
    out_dir = Path(out_dir)
    result.table1.to_csv(out_dir / "table1_wfo.csv")
    with open(out_dir / "table1_wfo.md", "w") as f:
        f.write(result.table1.round(4).to_markdown())
    result.fold_table.to_csv(out_dir / "wfo_folds.csv", index=False)
    with open(out_dir / "wfo_folds.md", "w") as f:
        f.write(result.fold_table.round(4).to_markdown(index=False))
