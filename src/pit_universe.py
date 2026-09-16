"""Point-in-time S&P universe construction.

Membership: the fja05680/sp500 historical-components snapshot CSV
(data/raw/sp500_historical_components.csv), ~weekly granularity,
1996-2026 -- forward-filled onto the daily trading calendar to build a
(date x ticker) eligibility mask.

Prices: sourced from a local multi-vendor SQLite archive
(/media/raghuram/seagate/data/dailystockprices.db, tables `tiingo_prices`
then `prices` as fallback -- tiingo has broader delisted-ticker coverage),
with a yfinance gap-fill for (a) tickers absent from the local DB entirely
and (b) the tail past the local DB's ~2021 cutoff for tickers still active
today. Ticker spelling is normalized (dot vs dash for dual-class shares,
e.g. "BF.B"/"BF-B") since the CSV/`prices` table and tiingo/yfinance don't
agree on convention.

Unlike `data_pipeline.build_panels` (a static current-snapshot universe with
a >=90%/98% coverage filter), this module does NOT filter dates/tickers by
coverage against the full backing ticker list -- a stock's raw price history
outside its point-in-time membership window is kept (so rolling windows have
real history the moment it enters the index) and eligibility is instead
carried as a separate `panels["eligible"]` boolean mask, applied to factor
*scores* (not raw prices) in `factor_eval.evaluate_factor`. A trading day is
kept only if at least `MIN_ACTIVE_NAMES` tickers are both eligible and
priced that day.
"""

from __future__ import annotations

import sqlite3
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config

LOCAL_DB_PATH = Path("/media/raghuram/seagate/data/dailystockprices.db")
MEMBERSHIP_CSV = config.DATA_RAW / "sp500_historical_components.csv"
PIT_CACHE_DIR = config.ROOT / "data" / "processed_pit"
PIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PANEL_KEYS = ["open", "high", "low", "close", "volume", "returns", "dollar_volume", "eligible"]
MIN_ACTIVE_NAMES = 50          # min (eligible AND priced) tickers to keep a trading day
MIN_HISTORY_DAYS = 60          # min raw observations to bother keeping a ticker's frame
SQLITE_CHUNK = 400             # stay under SQLite's default 999-variable-per-query limit


def _ticker_candidates(ticker: str) -> list[str]:
    cands = [ticker]
    if "." in ticker:
        cands.append(ticker.replace(".", "-"))
    if "-" in ticker:
        cands.append(ticker.replace("-", "."))
    return list(dict.fromkeys(cands))


def load_membership(csv_path: Path = MEMBERSHIP_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_daily_eligibility(membership: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """(trading_days x ticker) bool, forward-filled from the ~weekly
    membership snapshots onto every trading day (a ticker stays "in" until
    the next snapshot drops it).

    Each snapshot row is materialized as an explicit True/False vector over
    the FULL ticker universe (not just the tickers present that day) before
    ffill runs -- a long-form "presence only" table would never record an
    explicit False on the date a ticker exits the index (it simply stops
    appearing in later rows), so ffill would carry a stale True forward
    forever and the mask would only ever grow. Explicit False per snapshot
    is what lets an exit actually propagate."""
    all_tickers = sorted(set(t for row in membership.itertuples() for t in row.tickers.split(",")))
    snap_dates = membership["date"].tolist()
    mat = pd.DataFrame(False, index=pd.DatetimeIndex(snap_dates), columns=all_tickers)
    for row in membership.itertuples():
        mat.loc[row.date, row.tickers.split(",")] = True
    mat = mat[~mat.index.duplicated(keep="last")].sort_index()

    all_days = pd.DatetimeIndex(sorted(set(mat.index) | set(trading_days)))
    mat = mat.reindex(all_days).ffill().fillna(False)
    return mat.reindex(trading_days).fillna(False).astype(bool)


def _fetch_local(con: sqlite3.Connection, table: str, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    parts = []
    for i in range(0, len(tickers), SQLITE_CHUNK):
        chunk = tickers[i : i + SQLITE_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        q = (f"SELECT ticker, date, open, high, low, close, volume, adjclose, adjvolume "
             f"FROM {table} WHERE ticker IN ({placeholders}) AND date >= ? AND date <= ?")
        part = pd.read_sql_query(q, con, params=[*chunk, start, end])
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "adjclose"])
    df = df[df["close"] > 0]
    # local DB stores raw OHLC + a separately-adjusted close (not fully
    # adjusted OHLC) -- scale open/high/low by the adjclose/close ratio so
    # they stay split/dividend-consistent with the adjusted close we use.
    factor = df["adjclose"] / df["close"]
    df["open"] = df["open"] * factor
    df["high"] = df["high"] * factor
    df["low"] = df["low"] * factor
    df["close"] = df["adjclose"]
    df["volume"] = df["adjvolume"].fillna(df["volume"])
    return df[["ticker", "date", "open", "high", "low", "close", "volume"]]


def _resolve_from_local_db(all_tickers: list[str], start: str, end: str, db_path: Path) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(str(db_path))
    frames: dict[str, pd.DataFrame] = {}
    for table in ("tiingo_prices", "prices"):
        remaining = [t for t in all_tickers if t not in frames]
        if not remaining:
            break
        cand_map = {c: t for t in remaining for c in _ticker_candidates(t)}
        raw = _fetch_local(con, table, list(cand_map.keys()), start, end)
        if raw.empty:
            continue
        raw["orig_ticker"] = raw["ticker"].map(cand_map)
        for orig_t, g in raw.groupby("orig_ticker"):
            if orig_t in frames:
                continue
            g = g.drop_duplicates("date").set_index("date").sort_index()
            if len(g) >= MIN_HISTORY_DAYS:
                frames[orig_t] = g[["open", "high", "low", "close", "volume"]]
    con.close()
    print(f"[pit_universe] local DB resolved {len(frames)}/{len(all_tickers)} tickers")
    return frames


def _yf_gap_fill(frames: dict[str, pd.DataFrame], all_tickers: list[str], start: str, end: str, batch: int = 40) -> None:
    """Fills in (a) tickers entirely missing from the local DB and (b) the
    tail past the local DB's data for tickers still trading today."""
    cutoff = pd.Timestamp(end) - pd.Timedelta(days=15)
    need = [t for t in all_tickers if t not in frames or frames[t].index.max() < cutoff]
    print(f"[pit_universe] yfinance gap-fill for {len(need)} tickers")
    for i in range(0, len(need), batch):
        chunk = need[i : i + batch]
        yf_syms = {t: (t.replace(".", "-") if "." in t else t) for t in chunk}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(
                    list(yf_syms.values()), start=start, end=end, interval="1d",
                    auto_adjust=True, group_by="ticker", threads=True, progress=False,
                )
        except Exception as e:
            print(f"[pit_universe] yfinance batch failed ({e}); skipping {len(chunk)} tickers")
            continue
        for t, sym in yf_syms.items():
            try:
                df = data[sym] if len(yf_syms) > 1 else data
                df = df.dropna(how="all").rename(columns=str.lower)
            except (KeyError, IndexError):
                continue
            needed = {"open", "high", "low", "close", "volume"}
            if not needed.issubset(df.columns):
                continue
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            if df.empty:
                continue
            if t in frames:
                new_rows = df[df.index > frames[t].index.max()]
                if not new_rows.empty:
                    frames[t] = pd.concat([frames[t], new_rows]).sort_index()
            elif len(df) >= MIN_HISTORY_DAYS:
                frames[t] = df
        time.sleep(0.4)


def build_pit_frames(
    start: str = "1999-06-01",
    end: str | None = None,
    membership_csv: Path = MEMBERSHIP_CSV,
    db_path: Path = LOCAL_DB_PATH,
    yf_gap_fill: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Returns (per-ticker OHLCV frames, membership snapshot DataFrame)."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    membership = load_membership(membership_csv)
    membership = membership[(membership["date"] >= pd.Timestamp(start)) & (membership["date"] <= end)]
    all_tickers = sorted(set(t for row in membership["tickers"] for t in row.split(",")))
    print(f"[pit_universe] {len(all_tickers)} distinct point-in-time constituents, "
          f"membership span {membership['date'].min().date()}..{membership['date'].max().date()}")

    frames = _resolve_from_local_db(all_tickers, start, end, db_path)
    if yf_gap_fill:
        _yf_gap_fill(frames, all_tickers, start, end)

    missing = [t for t in all_tickers if t not in frames]
    if missing:
        print(f"[pit_universe] {len(missing)}/{len(all_tickers)} tickers unresolved from any source "
              f"(sample: {missing[:15]})")
    return frames, membership


def assemble_pit_panels(
    frames: dict[str, pd.DataFrame], membership: pd.DataFrame, panel_start: str, end: str,
    max_tickers: int | None = None,
) -> dict[str, pd.DataFrame]:
    close = pd.DataFrame({t: df["close"] for t, df in frames.items()})
    open_ = pd.DataFrame({t: df["open"] for t, df in frames.items()})
    high = pd.DataFrame({t: df["high"] for t, df in frames.items()})
    low = pd.DataFrame({t: df["low"] for t, df in frames.items()})
    volume = pd.DataFrame({t: df["volume"] for t, df in frames.items()})
    for p in (close, open_, high, low, volume):
        p.sort_index(inplace=True)

    idx = close.index
    idx = idx[(idx >= panel_start) & (idx <= end)]
    close, open_, high, low, volume = (p.reindex(idx) for p in (close, open_, high, low, volume))

    eligible = build_daily_eligibility(membership, idx)
    eligible = eligible.reindex(columns=close.columns, fill_value=False)

    if max_tickers is not None and close.shape[1] > max_tickers:
        # Compute-tractability filter: rolling/cross-sectional DSL ops scale
        # with column count, and evaluate_factor recomputes over the full
        # panel every round -- ~1000 point-in-time constituents makes a
        # 21-fold x 2-mode x N_ITERATIONS search loop computationally
        # infeasible. Keep the `max_tickers` most liquid names (mean dollar
        # volume on days each was an actual eligible constituent, so a
        # ticker's thin pre-IPO/post-delisting stub history doesn't count
        # against or for it) -- closer to this project's original SP100
        # scope than the full ~500-name index, while keeping true
        # point-in-time membership (not a static snapshot) for whichever
        # names are kept.
        dv = (close * volume).where(eligible)
        avg_dv = dv.mean(axis=0).sort_values(ascending=False)
        keep_tickers = avg_dv.head(max_tickers).index
        close, open_, high, low, volume, eligible = (
            p[keep_tickers] for p in (close, open_, high, low, volume, eligible)
        )
        print(f"[pit_universe] liquidity filter: kept top {len(keep_tickers)}/{len(avg_dv)} tickers "
              f"by mean $volume while eligible")

    n_active = (eligible & close.notna()).sum(axis=1)
    keep_days = n_active[n_active >= MIN_ACTIVE_NAMES].index
    close, open_, high, low, volume, eligible = (
        p.loc[keep_days] for p in (close, open_, high, low, volume, eligible)
    )

    returns = close.pct_change(fill_method=None)
    dollar_volume = close * volume
    return {
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "returns": returns, "dollar_volume": dollar_volume, "eligible": eligible,
    }


def select_fold_tickers(
    panels: dict[str, pd.DataFrame], as_of: pd.Timestamp, n_tickers: int, lookback_days: int = 504,
) -> list[str]:
    """Causal per-fold universe selection: the top `n_tickers` by trailing
    average dollar volume over the `lookback_days` (~2 trading years)
    strictly BEFORE `as_of` (a fold's train_start), among names eligible
    (true point-in-time S&P constituents) as of `as_of` -- no data at or
    after `as_of` is used to decide which tickers make the cut.

    This replaces a single fixed liquidity-ranked ticker list chosen from
    the *full* 1999-2026 sample (which would let, e.g., a 2001 fold trade
    among names that only became liquid megacaps decades later -- lookahead
    in universe construction, even though the trading signal itself stays
    causal). Each fold instead gets its own universe, as a real point-in-time
    trader would have had to."""
    close, volume, eligible = panels["close"], panels["volume"], panels["eligible"]
    idx = close.index
    history = idx[idx < pd.Timestamp(as_of)]
    window = history[-lookback_days:] if len(history) > lookback_days else history
    if len(window) == 0:
        raise ValueError(f"No price history strictly before {as_of} to select a fold universe from")

    elig_idx = eligible.index
    as_of_row = elig_idx[elig_idx <= pd.Timestamp(as_of)]
    elig_asof = eligible.loc[as_of_row[-1]] if len(as_of_row) else eligible.iloc[0]
    candidates = elig_asof[elig_asof].index

    dollar_volume = (close.loc[window, candidates] * volume.loc[window, candidates])
    avg_dv = dollar_volume.mean(axis=0).dropna().sort_values(ascending=False)
    return list(avg_dv.head(n_tickers).index)


def build_fold_universe(
    panels: dict[str, pd.DataFrame],
    reconstitution_dates: list[pd.Timestamp],
    fold_end: pd.Timestamp,
    n_tickers: int,
    lookback_days: int = 504,
) -> tuple[list[str], pd.DataFrame]:
    """Periodic (annual, one point per WFO fold-year) liquidity
    reconstitution, causal at every point: at each date in
    `reconstitution_dates`, select the top `n_tickers` by trailing
    liquidity using only data strictly before that date
    (`select_fold_tickers`); a ticker's membership in the day-varying
    "liquid" mask holds from its reconstitution date until the next one
    supersedes it. This replaces a single liquidity snapshot fixed at
    train_start (which would leave a fold's later years, and its test
    year, trading a universe that's up to several years stale).

    Returns (union_tickers, liquid_mask) where `liquid_mask` is a
    (date x union_tickers) bool DataFrame spanning
    [reconstitution_dates[0], fold_end] -- the union is typically only
    modestly larger than `n_tickers` (adjacent annual liquidity rankings
    overlap heavily), keeping this cheap relative to the full backing
    panel."""
    dates_sorted = sorted(pd.Timestamp(d) for d in reconstitution_dates)
    selections = {d: select_fold_tickers(panels, d, n_tickers, lookback_days) for d in dates_sorted}
    union_tickers = sorted(set(t for sel in selections.values() for t in sel))

    idx = panels["close"].index
    span = idx[(idx >= dates_sorted[0]) & (idx <= fold_end)]
    mask = pd.DataFrame(False, index=span, columns=union_tickers)
    for i, d in enumerate(dates_sorted):
        end = dates_sorted[i + 1] if i + 1 < len(dates_sorted) else fold_end + pd.Timedelta(days=1)
        block = span[(span >= d) & (span < end)]
        mask.loc[block, selections[d]] = True
    return union_tickers, mask


def _cache_paths(max_tickers: int | None) -> dict[str, Path]:
    suffix = f"_top{max_tickers}" if max_tickers else ""
    cache_dir = PIT_CACHE_DIR.parent / f"processed_pit{suffix}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {k: cache_dir / f"{k}.parquet" for k in PANEL_KEYS}


def get_pit_panels(
    force_refresh: bool = False,
    fetch_start: str = "1999-06-01",
    panel_start: str = "2001-01-01",
    end: str | None = None,
    max_tickers: int | None = 150,
) -> dict[str, pd.DataFrame]:
    """Main entry point. `fetch_start` is when raw price history begins
    (extra ~1.5y of buffer before `panel_start` so rolling windows up to
    ~252 days have real history from day one of the delivered panel);
    `panel_start` is the first date actually in the returned panels (and
    thus, downstream, the first WFO fold's train_start). `max_tickers`
    caps the point-in-time universe to the most liquid names for compute
    tractability -- see `assemble_pit_panels`; pass None for the full,
    uncapped ~500-name index."""
    paths = _cache_paths(max_tickers)
    if not force_refresh and all(p.exists() for p in paths.values()):
        return {k: pd.read_parquet(p) for k, p in paths.items()}

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    frames, membership = build_pit_frames(start=fetch_start, end=end)
    panels = assemble_pit_panels(frames, membership, panel_start, end, max_tickers=max_tickers)
    for k, p in paths.items():
        panels[k].to_parquet(p)
    print(f"[pit_universe] Built PIT panels: {panels['close'].shape[1]} tickers, "
          f"{panels['close'].shape[0]} trading days "
          f"({panels['close'].index.min().date()} to {panels['close'].index.max().date()}), "
          f"eligible/day mean={panels['eligible'].sum(axis=1).mean():.0f}")
    return panels
