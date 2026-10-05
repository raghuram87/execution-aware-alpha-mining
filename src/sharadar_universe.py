"""Point-in-time S&P 500 panels from the Sharadar Core US Equities bundle.

Membership comes from Sharadar's SP500 table: replaying its added/removed
events (back to 1957) reproduces its own quarterly constituent snapshots
exactly. Prices come from Sharadar's SEP table, which is survivorship-free,
so delisted constituents keep their full price history.

Sharadar quirks handled here:
- `close`, `open`, `high` and `low` are split-adjusted only; `closeadj` also
  adjusts for dividends. Open, high and low are rescaled by closeadj/close so
  every price field is on the same total-return basis.
- Days without trading are filled with the previous close and zero volume,
  so they are masked to NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config

SHARADAR_ROOT = Path("/media/raghuram/crucial/nasdaq_sharadar")
SP500_CSV = SHARADAR_ROOT / "csv" / "SP500.csv"
SEP_PARQUET = SHARADAR_ROOT / "parquet" / "SEP.parquet"
CACHE_DIR = config.ROOT / "data" / "processed_pit_sharadar"


def load_membership_events(path: Path = SP500_CSV) -> pd.DataFrame:
    sp = pd.read_csv(path, parse_dates=["date"])
    ev = sp[sp["action"].isin(["added", "removed"])][["date", "action", "ticker"]]
    return ev.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_daily_eligibility(events: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """(days x ticker) bool. A ticker is a member from its "added" date
    (inclusive) until its next "removed" date (exclusive)."""
    tickers = sorted(events["ticker"].unique())
    out = pd.DataFrame(False, index=days, columns=tickers)
    for t, g in events.groupby("ticker"):
        state = pd.Series(np.nan, index=days)
        for d, a in zip(g["date"], g["action"]):
            pos = days.searchsorted(d)
            if pos < len(days):
                state.iloc[pos] = 1.0 if a == "added" else 0.0
        before = g[g["date"] < days[0]]
        initial = float(before["action"].iloc[-1] == "added") if len(before) else 0.0
        out[t] = state.ffill().fillna(initial).astype(bool).to_numpy()
    return out


def load_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    cols = ["ticker", "date", "open", "high", "low", "close", "volume", "closeadj"]
    sep = pq.read_table(SEP_PARQUET, columns=cols, filters=[("ticker", "in", tickers)]).to_pandas()
    sep["date"] = pd.to_datetime(sep["date"])
    sep = sep[(sep["date"] >= pd.Timestamp(start)) & (sep["date"] <= pd.Timestamp(end))]
    sep = sep[(sep["volume"] > 0) & (sep["close"] > 0) & sep["closeadj"].notna()]
    factor = sep["closeadj"] / sep["close"]
    for f in ("open", "high", "low"):
        sep[f] = sep[f] * factor
    sep["close"] = sep["closeadj"]
    wide = {}
    for f in ("open", "high", "low", "close", "volume"):
        wide[f] = sep.pivot(index="date", columns="ticker", values=f).sort_index()
    return wide


def get_pit_panels_sharadar(
    force_refresh: bool = False,
    fetch_start: str = "1999-06-01",
    end: str | None = None,
    min_active_names: int = 50,
) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["open", "high", "low", "close", "volume", "returns", "dollar_volume", "eligible"]
    paths = {k: CACHE_DIR / f"{k}.parquet" for k in keys}
    if not force_refresh and all(p.exists() for p in paths.values()):
        return {k: pd.read_parquet(p) for k, p in paths.items()}

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    events = load_membership_events()
    probe = build_daily_eligibility(events, pd.bdate_range(fetch_start, end))
    tickers = sorted(probe.columns[probe.any(axis=0)])
    print(f"[sharadar] {len(tickers)} point-in-time S&P 500 constituents between {fetch_start} and {end}")

    wide = load_prices(tickers, fetch_start, end)
    days = wide["close"].index
    eligible = build_daily_eligibility(events, days).reindex(columns=wide["close"].columns, fill_value=False)

    n_active = (eligible & wide["close"].notna()).sum(axis=1)
    keep = n_active[n_active >= min_active_names].index
    panels = {f: wide[f].loc[keep] for f in wide}
    panels["eligible"] = eligible.loc[keep]
    panels["returns"] = panels["close"].pct_change(fill_method=None)
    panels["dollar_volume"] = panels["close"] * panels["volume"]

    for k, p in paths.items():
        panels[k].to_parquet(p)
    cov = (panels["eligible"] & panels["close"].notna()).sum(axis=1) / panels["eligible"].sum(axis=1)
    print(f"[sharadar] panels: {panels['close'].shape[1]} tickers x {len(keep)} days "
          f"({keep.min().date()}..{keep.max().date()}); priced share of members: "
          f"min {cov.min():.3f}, mean {cov.mean():.4f}")
    return panels
