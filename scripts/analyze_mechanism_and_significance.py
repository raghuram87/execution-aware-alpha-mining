"""Reproduces the manuscript's Table 2 (discovered-factor mechanism stats)
and the cluster-bootstrap statistical significance test (Section 3/4),
entirely from already-completed full_grid_sharadar_2006_2026 run data -- no new
LLM/GPU experiments required.

Usage: python scripts/analyze_mechanism_and_significance.py
"""
from __future__ import annotations

import ast
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.factor_eval import _SMOOTHER_FUNCS, _WINDOWED_FUNCS, count_ast_nodes

GRID_DIR = Path(__file__).resolve().parent.parent / "results" / "local_llm_runs" / "full_grid_sharadar_2006_2026"


def load_all_folds() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(GRID_DIR / "*/seed_*/wfo_folds.csv")):
        m = re.search(r"/([a-z]+)/seed_(\d+)/wfo_folds\.csv$", f.replace("\\", "/"))
        alpha, seed = m.group(1), m.group(2)
        df = pd.read_csv(f)
        df["seed_alpha"] = alpha
        df["llm_seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def extract_lookbacks(code: str) -> list[float]:
    tree = ast.parse(code, mode="eval")
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _WINDOWED_FUNCS:
            if node.args and isinstance(node.args[-1], ast.Constant) and isinstance(node.args[-1].value, (int, float)):
                out.append(node.args[-1].value)
    return out


def smoother_count(code: str) -> int:
    tree = ast.parse(code, mode="eval")
    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SMOOTHER_FUNCS
    )


def mechanism_table(full: pd.DataFrame) -> None:
    full = full.copy()
    full["node_count"] = full["Factor"].apply(count_ast_nodes)
    lookbacks = full["Factor"].apply(extract_lookbacks)
    full["avg_lookback"] = lookbacks.apply(lambda x: sum(x) / len(x) if x else float("nan"))
    full["has_decay"] = full["Factor"].apply(lambda c: "decay_linear" in c)
    full["n_smoothers"] = full["Factor"].apply(smoother_count)

    print("=== Mechanism stats: Baseline vs Execution-Aware (n=252 each) ===")
    summary = full.groupby("Mode").agg(
        mean_node_count=("node_count", "mean"),
        mean_avg_lookback=("avg_lookback", "mean"),
        pct_using_decay_linear=("has_decay", lambda x: 100 * x.mean()),
        mean_n_smoothers=("n_smoothers", "mean"),
    ).round(2)
    print(summary.to_string())

    print("\n=== Table 2: modal winning factor per (category, mode), out of 63 fold-seed instances ===")
    for (cat, mode), g in full.groupby(["seed_alpha", "Mode"]):
        top = g["Factor"].value_counts()
        print(f"\n{cat} / {mode}  (top expr appears {top.iloc[0]}/{len(g)} times)")
        print(" ", top.index[0])


def significance_test(full: pd.DataFrame, treated: str = "Execution-Aware", control: str = "Baseline",
                      n_boot: int = 10_000, seed: int = 42) -> dict:
    """Paired fold-level Net Sharpe difference (treated minus control),
    with a cluster bootstrap over (seed alpha, sampling seed) combinations
    and an exact sign test on the 12 combination-level means."""
    from scipy import stats
    piv = full.pivot_table(index=["seed_alpha", "llm_seed", "Fold"], columns="Mode", values="OOS_Net_IR").reset_index()
    piv = piv.dropna(subset=[treated, control])
    piv["delta"] = piv[treated] - piv[control]

    print(f"\n=== {treated} minus {control}: paired fold-level Net Sharpe delta (n={len(piv)}) ===")
    print("mean:", round(piv["delta"].mean(), 3), " median:", round(piv["delta"].median(), 3))

    rng = np.random.default_rng(seed)
    combo_groups = {(a, s): g["delta"].values for (a, s), g in piv.groupby(["seed_alpha", "llm_seed"])}
    combo_keys = list(combo_groups.keys())
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(len(combo_keys), size=len(combo_keys), replace=True)
        boot.append(np.concatenate([combo_groups[combo_keys[i]] for i in sampled]).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"cluster bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]  (mean {np.mean(boot):.3f})")

    for cat, g in piv.groupby("seed_alpha"):
        print(f"  {cat}: {(g['delta'] > 0).sum()}/{len(g)} folds favor {treated} ({100 * (g['delta'] > 0).mean():.1f}%)")

    cluster_means = piv.groupby(["seed_alpha", "llm_seed"])["delta"].mean()
    n_pos, n = int((cluster_means > 0).sum()), len(cluster_means)
    sign_p = stats.binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue
    print(f"  {n_pos}/{n} combinations positive; exact sign test p = {sign_p:.4f}")
    return {"treated": treated, "control": control, "mean": piv["delta"].mean(), "ci_lo": lo, "ci_hi": hi,
            "n_pos": n_pos, "n": n, "sign_p": sign_p,
            "fold_win_rate": {cat: (g["delta"] > 0).mean() for cat, g in piv.groupby("seed_alpha")}}


if __name__ == "__main__":
    full = load_all_folds()
    print(f"Loaded {len(full)} rows from {GRID_DIR}\n")
    mechanism_table(full)
    for treated, control in [("Execution-Aware", "Baseline"), ("Reward-Only", "Baseline"), ("Execution-Aware", "Reward-Only")]:
        significance_test(full, treated, control)
