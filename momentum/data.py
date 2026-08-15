"""Market-data access layer (Yahoo Finance via yfinance).

Two kinds of data are needed:

1. Daily adjusted close + volume history  -> price returns, volatility, turnover.
   Fetched in batches with ``yfinance.download`` (one HTTP request per batch).

2. Market capitalisation + shares         -> turnover-ratio filter and weighting.
   Only available per-ticker via ``fast_info``; fetched concurrently and cached.

Both are disk-cached (parquet) so repeated scans within a session are cheap and
the tool degrades gracefully when a few tickers fail.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config

# yfinance logs a warning line per delisted/unknown ticker; quiet it so the UI
# and CLI stay readable. Failed tickers are still surfaced via PriceData.failed.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class PriceData:
    """Bundle of aligned price/volume frames plus per-ticker market data.

    close:   DataFrame indexed by date, one column per Yahoo ticker (adj close)
    volume:  DataFrame indexed by date, one column per Yahoo ticker
    market:  DataFrame indexed by Yahoo ticker with columns
             [market_cap, shares, last_price]
    failed:  tickers that returned no usable price history
    """

    close: pd.DataFrame
    volume: pd.DataFrame
    market: pd.DataFrame
    failed: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Price / volume history
# --------------------------------------------------------------------------- #
def _price_cache_paths(cache_key: str) -> tuple[Path, Path]:
    return (
        config.DATA_DIR / f"close_{cache_key}.parquet",
        config.DATA_DIR / f"volume_{cache_key}.parquet",
    )


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _download_batch(
    tickers: list[str], start: dt.date, end: dt.date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download one batch and return (close, volume) wide frames.

    yfinance returns MultiIndex columns (field, ticker) for multiple tickers and
    flat columns for a single ticker; this normalises both to wide frames whose
    columns are tickers. ``end`` is exclusive in yfinance, so callers pass
    as-of + 1 day to include the as-of date itself.
    """
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:  # single ticker -> flat columns
        close = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()
        close.columns = tickers[:1]
        volume.columns = tickers[:1]

    return close, volume


def fetch_prices(
    tickers: list[str],
    cache_key: str,
    asof: dt.date,
    lookback_days: int = config.PRICE_LOOKBACK_DAYS,
    force_refresh: bool = False,
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (close, volume) wide frames for ``tickers``.

    History spans ``[asof - lookback_days, asof]``. ``cache_key`` should encode
    the universe + as-of date so distinct scans do not collide. ``progress`` is
    an optional callable ``(done, total)`` for UI.
    """
    close_path, volume_path = _price_cache_paths(cache_key)
    if not force_refresh and _is_fresh(close_path, config.PRICE_CACHE_HOURS):
        close = pd.read_parquet(close_path)
        volume = pd.read_parquet(volume_path)
        if progress:
            progress(len(tickers), len(tickers))
        return close, volume

    start = asof - dt.timedelta(days=lookback_days)
    end = asof + dt.timedelta(days=1)  # yfinance end is exclusive

    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    batches = [
        tickers[i : i + config.DOWNLOAD_BATCH_SIZE]
        for i in range(0, len(tickers), config.DOWNLOAD_BATCH_SIZE)
    ]

    closes, volumes, done = [], [], 0
    for batch in batches:
        c, v = _download_batch(batch, start, end)
        if not c.empty:
            closes.append(c)
            volumes.append(v)
        done += len(batch)
        if progress:
            progress(done, len(tickers))

    if not closes:
        return pd.DataFrame(), pd.DataFrame()

    close = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(volumes, axis=1).sort_index()
    # Drop all-NaN columns (tickers Yahoo had nothing for).
    close = close.dropna(axis=1, how="all")
    volume = volume.reindex(columns=close.columns)

    close.to_parquet(close_path)
    volume.to_parquet(volume_path)
    return close, volume


# --------------------------------------------------------------------------- #
# Market cap / shares (per-ticker fast_info)
# --------------------------------------------------------------------------- #
def _marketcap_cache_path(cache_key: str) -> Path:
    return config.DATA_DIR / f"market_{cache_key}.parquet"


def _fetch_one_market(ticker: str) -> dict:
    """Best-effort market snapshot for a single ticker via fast_info."""
    out = {"ticker": ticker, "market_cap": None, "shares": None, "last_price": None}
    try:
        fi = yf.Ticker(ticker).fast_info
        for key, attr in (("market_cap", "market_cap"), ("shares", "shares"), ("last_price", "last_price")):
            try:
                out[key] = fi[attr]
            except Exception:
                pass
    except Exception:
        pass
    return out


def fetch_market_data(
    tickers: list[str],
    cache_key: str,
    force_refresh: bool = False,
    progress=None,
) -> pd.DataFrame:
    """Return per-ticker market data indexed by Yahoo ticker.

    Columns: market_cap, shares, last_price. Missing values are left as NaN so
    downstream filters/weighting can decide how to handle them.
    """
    path = _marketcap_cache_path(cache_key)
    if not force_refresh and _is_fresh(path, config.MARKETCAP_CACHE_HOURS):
        df = pd.read_parquet(path)
        if progress:
            progress(len(tickers), len(tickers))
        return df

    tickers = list(dict.fromkeys(tickers))
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=config.MARKETCAP_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_market, t): t for t in tickers}
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if progress:
                progress(done, len(tickers))

    df = pd.DataFrame(rows).set_index("ticker").sort_index()
    df = df.astype("float64")
    df.to_parquet(path)
    return df


# --------------------------------------------------------------------------- #
# Convenience: fetch everything for a universe
# --------------------------------------------------------------------------- #
def fetch_all(
    tickers: list[str],
    cache_key: str,
    asof: dt.date,
    lookback_days: int = config.PRICE_LOOKBACK_DAYS,
    force_refresh: bool = False,
    price_progress=None,
    market_progress=None,
) -> PriceData:
    """Fetch price history and market data, returning a :class:`PriceData`."""
    close, volume = fetch_prices(
        tickers, cache_key, asof, lookback_days, force_refresh, progress=price_progress
    )
    market = fetch_market_data(tickers, cache_key, force_refresh, progress=market_progress)

    available = set(close.columns)
    failed = [t for t in tickers if t not in available]
    return PriceData(close=close, volume=volume, market=market, failed=failed)
