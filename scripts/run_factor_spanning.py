"""Factor spanning: are the strategies' daily net returns explained by the
Fama-French five factors plus momentum? Reports annualized alpha with a
Newey-West t-statistic, the factor loadings and R^2, for every
(category, seed, arm) in the main grid and for the no-search controls.

Factor data: Kenneth French's data library (daily 5-factor 2x3 and daily
momentum files, saved under data/raw/ff/).

Usage: python scripts/run_factor_spanning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config

FF_DIR = config.ROOT / "data" / "raw" / "ff"
GRID_DIR = config.RESULTS / "local_llm_runs" / "full_grid_sharadar_2006_2026"
CONTROLS_DIR = config.RESULTS / "local_llm_runs" / "controls_sharadar"
OUT = config.RESULTS / "local_llm_runs" / "factor_spanning_sharadar.csv"
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
NW_LAGS = 5


def read_french(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith(","))
    rows = []
    for l in lines[header_idx + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts)
    cols = ["date"] + [c.strip() for c in lines[header_idx].split(",")[1:]]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype(float) / 100.0


def load_factors() -> pd.DataFrame:
    ff5 = read_french(FF_DIR / "F-F_Research_Data_5_Factors_2x3_daily.csv")
    mom = read_french(FF_DIR / "F-F_Momentum_Factor_daily.csv")
    mom.columns = ["Mom"]
    return ff5.join(mom, how="inner")


def ols_newey_west(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS):
    X = np.column_stack([np.ones(len(X)), X])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = xe[l:].T @ xe[:-l]
        S += w * (G + G.T)
    V = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(V))
    r2 = 1.0 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return beta, beta / se, r2


def span(series: pd.Series, factors: pd.DataFrame) -> dict:
    df = pd.concat([series.rename("y"), factors[FACTORS]], axis=1, join="inner").dropna()
    beta, t, r2 = ols_newey_west(df["y"].to_numpy(), df[FACTORS].to_numpy())
    out = {"n_days": len(df), "alpha_ann": beta[0] * config.TRADING_DAYS_PER_YEAR, "alpha_t": t[0], "R2": r2}
    out.update({f"b_{f}": b for f, b in zip(FACTORS, beta[1:])})
    out.update({f"t_{f}": tt for f, tt in zip(FACTORS, t[1:])})
    return out


if __name__ == "__main__":
    factors = load_factors()
    rows = []
    for f in sorted(GRID_DIR.glob("*/seed_*/net_returns_*.csv")):
        cat, seed, arm = f.parent.parent.name, int(f.parent.name.split("_")[1]), f.stem.replace("net_returns_", "")
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        rows.append({"source": "grid", "seed_alpha": cat, "llm_seed": seed, "arm": arm, **span(s, factors)})
    for f in sorted(CONTROLS_DIR.glob("net_returns_*.csv")):
        name = f.stem.replace("net_returns_", "")
        control, cat, seed = name.rsplit("_", 2)
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        rows.append({"source": "control", "seed_alpha": cat, "llm_seed": int(seed), "arm": control, **span(s, factors)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    show = ["source", "seed_alpha", "llm_seed", "arm", "alpha_ann", "alpha_t", "R2"] + [f"b_{f}" for f in FACTORS]
    print(out[show].round(3).to_string(index=False))
