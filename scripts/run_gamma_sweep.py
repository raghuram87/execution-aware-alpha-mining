"""Reward-decomposition / penalty-sensitivity ablation: does the baseline-
vs-execution-aware effect trace a smooth turnover-vs-Sharpe frontier as a
function of the two penalty weights (gamma1 on Turnover, gamma2 on
Cost_Impact), rather than being an artifact of the one (0.5, 8) combined
weighting used in the main grid?

Single-term sweeps (each holding the other gamma at 0) plus one joint
midpoint, run on the volatility seed-alpha (the paper's headline
net-profitable case) at one LLM sampling seed (1001) -- the (0,0) and
(0.5,8) endpoints already exist in the main full_grid_2006_2026 run and
are not re-run here.

Usage: python scripts/run_gamma_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.pit_universe import get_pit_panels
from src.walk_forward import run_wfo_experiment, save_wfo_tables

ALPHA_NAME = "volatility"
LLM_SEED = 1001
OUT_ROOT = config.RESULTS / "local_llm_runs" / "gamma_sweep_sharadar_volatility_seed1001"

GAMMA_SETTINGS = [
    ("turnover_only", 0.1, 0.0),
    ("turnover_only", 0.25, 0.0),
    ("turnover_only", 0.5, 0.0),
    ("turnover_only", 1.0, 0.0),
    ("cost_only", 0.0, 2.0),
    ("cost_only", 0.0, 4.0),
    ("cost_only", 0.0, 8.0),
    ("cost_only", 0.0, 16.0),
    ("combined_midpoint", 0.25, 4.0),
]


def run_one_setting(kind: str, gamma1: float, gamma2: float, panels, out_root: Path):
    tag = f"{kind}_g1_{gamma1}_g2_{gamma2}"
    combo_dir = out_root / tag
    combo_dir.mkdir(parents=True, exist_ok=True)
    log_dir = combo_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}")
    result = run_wfo_experiment(
        llm_kind="local",
        matched_seed=LLM_SEED,
        n_iterations=config.N_ITERATIONS,
        gamma1=gamma1,
        gamma2=gamma2,
        panels=panels,
        train_years=config.WFO_TRAIN_YEARS,
        test_years=config.WFO_TEST_YEARS,
        step_years=config.WFO_STEP_YEARS,
        train_start=config.PIT_TRAIN_START,
        seed_alpha=config.SEED_ALPHAS[ALPHA_NAME],
        log_dir=log_dir,
        verbose=True,
        modes=("execution_aware",),
    )
    save_wfo_tables(result, out_dir=combo_dir)
    result.stitched["Execution-Aware"]["full_cost_model"].net_returns.to_csv(combo_dir / "net_returns_execution_aware.csv")

    # Only the execution-aware arm: the baseline reward ignores gamma1/gamma2,
    # so its run is identical to the main grid's baseline for this seed.
    row = result.table1.loc["Execution-Aware"].to_dict()
    row.update({"kind": kind, "gamma1": gamma1, "gamma2": gamma2, "seed_alpha": ALPHA_NAME, "llm_seed": LLM_SEED})
    return row


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=int, nargs="+", default=list(range(len(GAMMA_SETTINGS))),
                    help="indices into GAMMA_SETTINGS, to split the sweep across GPUs")
    ap.add_argument("--tag", default="all", help="suffix for this process's summary file")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[gamma_sweep] loading full point-in-time panels (cached if already built)...")
    panels = get_pit_panels(fetch_start=config.PIT_FETCH_START, panel_start=config.PIT_FETCH_START, max_tickers=None)

    summary_path = OUT_ROOT / f"gamma_sweep_summary_{args.tag}.csv"
    rows = []
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict("records")
        done = {(r["gamma1"], r["gamma2"]) for r in rows}
        print(f"[gamma_sweep] resuming: {len(done)} settings already in summary")
    else:
        done = set()

    for kind, g1, g2 in [GAMMA_SETTINGS[i] for i in args.settings]:
        if (g1, g2) in done:
            print(f"[gamma_sweep] skipping already-completed g1={g1} g2={g2}")
            continue
        try:
            row = run_one_setting(kind, g1, g2, panels, OUT_ROOT)
            rows.append(row)
        except Exception:
            import traceback
            print(f"[gamma_sweep] SETTING FAILED: {kind} g1={g1} g2={g2}")
            traceback.print_exc()
            continue
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"[gamma_sweep] checkpointed gamma_sweep_summary.csv ({len(rows)} rows)")

    print(f"\n{'=' * 78}\nGAMMA SWEEP COMPLETE\n{'=' * 78}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
