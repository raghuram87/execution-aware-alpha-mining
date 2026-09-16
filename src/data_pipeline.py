"""Week 1: data acquisition. Downloads daily OHLCV for a fixed S&P 100 snapshot
via yfinance and builds wide (date x ticker) panels used by the cost engine and
factor evaluator.

Note on survivorship bias: SP100_TICKERS is a *current* constituent snapshot,
not a point-in-time historical membership list. Delisted/replaced names are not
included, which biases returns/turnover mildly upward relative to a true
point-in-time backtest. This is disclosed as a limitation in the manuscript.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd
import yfinance as yf

from . import config

# Fixed S&P 100 snapshot (large-cap US equities), yfinance-compatible symbols
# ("." replaced with "-", e.g. BRK.B -> BRK-B).
SP100_TICKERS = sorted(set([
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "KHC", "KMI", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
    "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]))

FIELDS = ["open", "high", "low", "close", "volume", "returns", "dollar_volume"]


def _download_batch(tickers: list[str], years: int) -> dict[str, pd.DataFrame]:
    period = f"{years}y"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(
            tickers, period=period, interval="1d", auto_adjust=True,
            group_by="ticker", threads=True, progress=False,
        )
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = data[t].dropna(how="all") if len(tickers) > 1 else data.dropna(how="all")
        except (KeyError, IndexError):
            continue
        df = df.rename(columns=str.title)
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < 500:  # require enough history to be usable
            continue
        frames[t] = df
    return frames


def download_universe(
    tickers: list[str] | None = None,
    years: int = config.LOOKBACK_YEARS,
    batch_size: int = 20,
    pause: float = 0.5,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Download raw OHLCV for each ticker. Returns (frames, failed_tickers)."""
    tickers = tickers or SP100_TICKERS
    frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_frames = _download_batch(batch, years)
        frames.update(batch_frames)
        failed.extend([t for t in batch if t not in batch_frames])
        time.sleep(pause)
    return frames, failed


def build_panels(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Turn per-ticker OHLCV frames into aligned wide (date x ticker) panels."""
    close = pd.DataFrame({t: df["Close"] for t, df in frames.items()})
    open_ = pd.DataFrame({t: df["Open"] for t, df in frames.items()})
    high = pd.DataFrame({t: df["High"] for t, df in frames.items()})
    low = pd.DataFrame({t: df["Low"] for t, df in frames.items()})
    volume = pd.DataFrame({t: df["Volume"] for t, df in frames.items()})

    for panel in (close, open_, high, low, volume):
        panel.sort_index(inplace=True)

    # Keep only trading days where at least 90% of the universe has data,
    # then forward-fill isolated single-day gaps (halts) and drop any
    # tickers that still have material missingness.
    coverage = close.notna().mean(axis=1)
    common_days = coverage[coverage >= 0.9].index
    close, open_, high, low, volume = (
        p.loc[common_days].ffill(limit=2) for p in (close, open_, high, low, volume)
    )
    keep = close.columns[close.notna().mean(axis=0) >= 0.98]
    close, open_, high, low, volume = (p[keep] for p in (close, open_, high, low, volume))

    returns = close.pct_change()
    dollar_volume = close * volume

    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "returns": returns, "dollar_volume": dollar_volume,
    }


def _cache_paths() -> dict[str, "config.Path"]:
    return {field: config.DATA_PROCESSED / f"{field}.parquet" for field in FIELDS}


def save_panels(panels: dict[str, pd.DataFrame]) -> None:
    for field, path in _cache_paths().items():
        panels[field].to_parquet(path)


def load_cached_panels() -> dict[str, pd.DataFrame] | None:
    paths = _cache_paths()
    if not all(p.exists() for p in paths.values()):
        return None
    return {field: pd.read_parquet(path) for field, path in paths.items()}


def train_test_split_date(panels: dict[str, pd.DataFrame], train_fraction: float = config.TRAIN_FRACTION) -> pd.Timestamp:
    """Date that splits the sample into an in-sample search window (dates
    <= split_date) and a held-out out-of-sample window (dates > split_date),
    by trading-day count rather than calendar time."""
    idx = panels["close"].index
    split_i = int(len(idx) * train_fraction)
    return idx[split_i]


def get_panels(force_refresh: bool = False, years: int = config.LOOKBACK_YEARS) -> dict[str, pd.DataFrame]:
    """Main entry point: load cached panels or download + build + cache them."""
    if not force_refresh:
        cached = load_cached_panels()
        if cached is not None:
            return cached

    frames, failed = download_universe(years=years)
    if len(frames) < 20:
        raise RuntimeError(
            f"Only {len(frames)} tickers downloaded successfully; aborting. "
            f"Failed: {failed[:10]}..."
        )
    panels = build_panels(frames)
    save_panels(panels)
    if failed:
        print(f"[data_pipeline] {len(failed)}/{len(SP100_TICKERS)} tickers failed/skipped: {failed}")
    print(f"[data_pipeline] Built panels for {panels['close'].shape[1]} tickers, "
          f"{panels['close'].shape[0]} trading days "
          f"({panels['close'].index.min().date()} to {panels['close'].index.max().date()})")
    return panels
