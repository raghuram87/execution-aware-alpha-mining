"""Write every manuscript table as a LaTeX fragment (paper/fro/tables/*.tex)
directly from the saved result files, plus a JSON of the headline numbers
quoted in the text (paper/fro/tables/numbers.json), so no figure in the
paper is hand-copied.

Usage: python scripts/make_paper_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_mechanism_and_significance import extract_lookbacks, load_all_folds, significance_test
from run_factor_spanning import FACTORS, load_factors, span
from src import config
from src.factor_eval import count_ast_nodes

RUNS = config.RESULTS / "local_llm_runs"
GRID = RUNS / "full_grid_sharadar_2006_2026"
OUT = ROOT / "paper" / "fro" / "tables"
CATS = ["reversal", "volume", "volatility", "momentum"]
ARMS = ["Baseline", "Reward-Only", "Execution-Aware"]
ARM_TEX = {"Baseline": "Cost-blind", "Reward-Only": "Reward-only", "Execution-Aware": "Execution-aware"}
ARM_FILE = {"Baseline": "baseline", "Reward-Only": "reward_only", "Execution-Aware": "execution_aware"}


def f(x, d=2, sign=False):
    if pd.isna(x):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", "$-$")


def pct(x, d=0, sign=False):
    return f(100 * x, d, sign) + r"\%"


def write(name, body):
    (OUT / name).write_text(body)
    print("wrote", OUT / name)


def table_main(grid):
    rows = []
    for cat in CATS:
        for i, arm in enumerate(ARMS):
            g = grid[(grid.seed_alpha == cat) & (grid["mode"] == arm)]
            m = g.mean(numeric_only=True)
            rows.append(" & ".join([
                cat.capitalize() if i == 0 else "", ARM_TEX[arm],
                f(m.Gross_Sharpe), f"{f(m.Net_Sharpe_full_cost_model)} ({f(g.Net_Sharpe_full_cost_model.std())})",
                f(m.Avg_Daily_Turnover_pct, 1) + r"\%", f(m.Avg_Holding_Period_Days, 1),
                pct(m.Max_Drawdown), pct(m.Cumulative_Return, sign=True),
            ]) + r" \\")
        if cat != CATS[-1]:
            rows.append(r"\addlinespace")
    write("tab_main.tex", r"""\begin{tabular}{llrrrrrr}
\toprule
Seed alpha & Arm & Gross SR & Net SR (s.d.) & Turnover/day & Holding (days) & Max DD & Cum. return \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")


def table_tests(full):
    out, rows = [], []
    for treated, control in [("Reward-Only", "Baseline"), ("Execution-Aware", "Reward-Only"), ("Execution-Aware", "Baseline")]:
        r = significance_test(full, treated, control)
        out.append(r)
        wins = " & ".join(pct(r["fold_win_rate"][c]) for c in CATS)
        rows.append(f"{ARM_TEX[treated]} $-$ {ARM_TEX[control].lower()} & {f(r['mean'], 2, True)} & "
                    f"[{f(r['ci_lo'])}, {f(r['ci_hi'])}] & {r['n_pos']}/{r['n']} & {r['sign_p']:.4f} & {wins}" + r" \\")
    write("tab_tests.tex", r"""\begin{tabular}{lrcccrrrr}
\toprule
 & & & \multicolumn{2}{c}{Combinations} & \multicolumn{4}{c}{Folds won} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-9}
Comparison & $\Delta$ Net SR & 95\% CI & positive & sign-test $p$ & Rev. & Vol. & Volat. & Mom. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    return out


def table_mechanism(full):
    full = full.copy()
    full["nodes"] = full["Factor"].apply(count_ast_nodes)
    full["lookback"] = full["Factor"].apply(lambda c: np.mean(extract_lookbacks(c)) if extract_lookbacks(c) else np.nan)
    full["decay"] = full["Factor"].str.contains("decay_linear")
    rows, stats = [], {}
    for arm in ARMS:
        g = full[full.Mode == arm]
        modal = [g[g.seed_alpha == c]["Factor"].value_counts().iloc[0] for c in CATS]
        stats[arm] = {"decay": g.decay.mean(), "lookback": g.lookback.mean(), "nodes": g.nodes.mean(),
                      "modal_min": min(modal), "modal_max": max(modal)}
        rows.append(f"{ARM_TEX[arm]} & {pct(g.decay.mean())} & {f(g.lookback.mean(), 0)} & {f(g.nodes.mean(), 1)} & "
                    f"{min(modal)}--{max(modal)}" + r" \\")
    write("tab_mechanism.tex", r"""\begin{tabular}{lrrrr}
\toprule
Arm & Uses \texttt{decay\_linear} & Mean lookback (days) & Mean nodes & Modal recurrences (of 63) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    modal_rows = []
    for c in CATS:
        for i, arm in enumerate(ARMS):
            vc = full[(full.Mode == arm) & (full.seed_alpha == c)]["Factor"].value_counts()
            expr = vc.index[0].replace("_", r"\_")
            modal_rows.append(f"{c.capitalize() if i == 0 else ''} & {ARM_TEX[arm]} & \\texttt{{\\footnotesize {expr}}} & {vc.iloc[0]}" + r" \\")
        if c != CATS[-1]:
            modal_rows.append(r"\addlinespace")
    write("tab_modal.tex", r"""\begin{tabular}{llp{0.62\textwidth}r}
\toprule
Seed alpha & Arm & Most frequent winning expression & Count \\
\midrule
""" + "\n".join(modal_rows) + r"""
\bottomrule
\end{tabular}
""")
    return stats


def table_gamma(grid):
    g = pd.concat([pd.read_csv(p) for p in (RUNS / "gamma_sweep_sharadar_volatility_seed1001").glob("gamma_sweep_summary_*.csv")])
    order = {"turnover_only": 0, "cost_only": 1, "combined_midpoint": 2}
    g = g.assign(o=g.kind.map(order)).sort_values(["o", "gamma1", "gamma2"])
    label = {"turnover_only": "Turnover only", "cost_only": "Cost only", "combined_midpoint": "Combined"}
    main = grid[(grid.seed_alpha == "volatility") & (grid.llm_seed == 1001)].set_index("mode")
    def row(name, g1, g2, r):
        return (f"{name} & {f(g1, 2)} & {f(g2, 0)} & {f(r.Gross_Sharpe)} & {f(r.Net_Sharpe_full_cost_model)} & "
                f"{f(r.Avg_Daily_Turnover_pct, 2)}\\% & {pct(r.Cumulative_Return, sign=True)}" + r" \\")
    rows = [row("Cost-blind", 0, 0, main.loc["Baseline"])]
    rows += [row(label[r.kind], r.gamma1, r.gamma2, r) for r in g.itertuples()]
    rows.append(row("Combined (main grid)", config.GAMMA_1, config.GAMMA_2, main.loc["Execution-Aware"]))
    write("tab_gamma.tex", r"""\begin{tabular}{lrrrrrr}
\toprule
Penalty & $\gamma_1$ & $\gamma_2$ & Gross SR & Net SR & Turnover/day & Cum. return \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    return g


def table_controls(grid):
    c = pd.read_csv(RUNS / "controls_sharadar" / "controls_summary.csv")
    cols = [("seed_daily", "Seed alpha, daily"), ("seed_monthly", "Seed alpha, monthly"),
            ("seed_smoothed", r"\texttt{decay\_linear}(seed, 252), daily"), ("baseline_monthly", "Cost-blind winners, monthly")]
    rows = []
    for key, name in cols:
        m = c[c.control == key].groupby("seed_alpha")[["Net_Sharpe_full_cost_model", "Avg_Daily_Turnover_pct"]].mean()
        rows.append(name + " & " + " & ".join(f"{f(m.loc[k].Net_Sharpe_full_cost_model)} & {f(m.loc[k].Avg_Daily_Turnover_pct, 1)}\\%" for k in CATS) + r" \\")
    rows.append(r"\midrule")
    for arm in ["Reward-Only", "Execution-Aware"]:
        m = grid[grid["mode"] == arm].groupby("seed_alpha")[["Net_Sharpe_full_cost_model", "Avg_Daily_Turnover_pct"]].mean()
        rows.append(ARM_TEX[arm] + " search & " + " & ".join(f"{f(m.loc[k].Net_Sharpe_full_cost_model)} & {f(m.loc[k].Avg_Daily_Turnover_pct, 1)}\\%" for k in CATS) + r" \\")
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{k.capitalize()}}}" for k in CATS)
    rules = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4))
    write("tab_controls.tex", r"""\begin{tabular}{lrrrrrrrr}
\toprule
 & """ + head + r""" \\
""" + rules + r"""
Strategy & """ + " & ".join(["Net SR & TO/day"] * 4) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    return c


def pooled(paths):
    s = [pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0] for p in paths]
    return pd.concat(s, axis=1).mean(axis=1)


def table_spanning():
    fac = load_factors()
    rows, res = [], {}
    def add(label, series):
        r = span(series, fac)
        res[label] = r
        rows.append(f"{label} & {f(100 * r['alpha_ann'], 1, True)} & ({f(r['alpha_t'])}) & " +
                    " & ".join(f(r[f'b_{x}']) for x in FACTORS) + f" & {f(r['R2'])}" + r" \\")
    for cat in CATS:
        for arm in ARMS:
            add(f"{cat.capitalize()}: {ARM_TEX[arm].lower()}",
                pooled([GRID / cat / f"seed_{s}" / f"net_returns_{ARM_FILE[arm]}.csv" for s in (1001, 2002, 3003)]))
        if cat != CATS[-1]:
            rows.append(r"\addlinespace")
    rows.append(r"\midrule")
    ctrl = RUNS / "controls_sharadar"
    add(r"Volatility: seed, monthly", pooled([ctrl / "net_returns_seed_monthly_volatility_0.csv"]))
    add(r"Volatility: \texttt{decay\_linear}(seed, 252)", pooled([ctrl / "net_returns_seed_smoothed_volatility_0.csv"]))
    write("tab_spanning.tex", r"""\begin{tabular}{lrrrrrrrrr}
\toprule
Strategy & $\alpha$ (\%/yr) & ($t$) & Mkt & SMB & HML & RMW & CMA & Mom & $R^2$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(GRID / "grid_summary.csv")
    full = load_all_folds()
    table_main(grid)
    tests = table_tests(full)
    mech = table_mechanism(full)
    gamma = table_gamma(grid)
    table_controls(grid)
    spanning = table_spanning()

    vol_ea = grid[(grid.seed_alpha == "volatility") & (grid["mode"] == "Execution-Aware")]
    numbers = {
        "tests": tests, "mechanism": mech,
        "gamma_net_min": gamma.Net_Sharpe_full_cost_model.min(), "gamma_net_max": gamma.Net_Sharpe_full_cost_model.max(),
        "gamma_to_min": gamma.Avg_Daily_Turnover_pct.min(), "gamma_to_max": gamma.Avg_Daily_Turnover_pct.max(),
        "baseline_gross_by_cat": grid[grid["mode"] == "Baseline"].groupby("seed_alpha").Gross_Sharpe.mean().to_dict(),
        "baseline_gross_mean": grid[grid["mode"] == "Baseline"].Gross_Sharpe.mean(),
        "vol_ea_net_range": [vol_ea.Net_Sharpe_full_cost_model.min(), vol_ea.Net_Sharpe_full_cost_model.max()],
        "vol_ea_maxdd_range": [vol_ea.Max_Drawdown.min(), vol_ea.Max_Drawdown.max()],
        "spanning": {k: {kk: v[kk] for kk in ("alpha_ann", "alpha_t", "R2")} for k, v in spanning.items()},
    }
    for arm in ARMS:
        g = grid[grid["mode"] == arm].groupby("seed_alpha")
        numbers[f"turnover_{arm}"] = g.Avg_Daily_Turnover_pct.mean().to_dict()
        numbers[f"net_{arm}"] = g.Net_Sharpe_full_cost_model.mean().to_dict()
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (OUT / "numbers.json").write_text(json.dumps(conv(numbers), indent=1))
    print("wrote", OUT / "numbers.json")
