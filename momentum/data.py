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

    close:     DataFrame indexed by date, one column per Yahoo ticker.
               ADJUSTED close (splits & dividends) — used for all return maths.
    raw_close: DataFrame indexed by date, one column per Yahoo ticker.
               RAW close (the actual traded closing price) — for display. May be
               empty when a caller does not request it.
    volume:    DataFrame indexed by date, one column per Yahoo ticker
    market:    DataFrame indexed by Yahoo ticker with columns
               [market_cap, shares, last_price]
    failed:    tickers that returned no usable price history
    """

    close: pd.DataFrame
    volume: pd.DataFrame
    market: pd.DataFrame
    raw_close: pd.DataFrame = field(default_factory=pd.DataFrame)
    failed: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Price / volume history
# --------------------------------------------------------------------------- #
def _price_cache_paths(cache_key: str) -> tuple[Path, Path, Path]:
    return (
        config.DATA_DIR / f"close_{cache_key}.parquet",
        config.DATA_DIR / f"rawclose_{cache_key}.parquet",
        config.DATA_DIR / f"volume_{cache_key}.parquet",
    )


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _download_batch(
    tickers: list[str], start: dt.date, end: dt.date
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download one batch and return (adj_close, raw_close, volume) wide frames.

    ``auto_adjust=False`` so Yahoo returns both ``Adj Close`` (splits+dividends,
    for return maths) and ``Close`` (the actual traded closing price, for
    display). yfinance returns MultiIndex columns (field, ticker) for multiple
    tickers and flat columns for a single ticker; this normalises both to wide
    frames whose columns are tickers. ``end`` is exclusive in yfinance, so
    callers pass as-of + 1 day to include the as-of date itself.
    """
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    empty = pd.DataFrame()
    if raw is None or raw.empty:
        return empty, empty, empty

    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw["Adj Close"].copy()
        rawc = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:  # single ticker -> flat columns
        adj = raw[["Adj Close"]].copy()
        rawc = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()
        adj.columns = rawc.columns = volume.columns = tickers[:1]

    return adj, rawc, volume


def fetch_prices(
    tickers: list[str],
    cache_key: str,
    asof: dt.date,
    lookback_days: int = config.PRICE_LOOKBACK_DAYS,
    start: dt.date | None = None,
    force_refresh: bool = False,
    cache_hours: float = config.PRICE_CACHE_HOURS,
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (adj_close, raw_close, volume) wide frames for ``tickers``.

    History spans ``[start, asof]``; if ``start`` is omitted it defaults to
    ``asof - lookback_days`` (the ~13-month window used by a single live scan).
    Pass an explicit ``start`` (e.g. 9 years back) to fetch the long history a
    backtest needs — the scanner slices it per as-of internally, so one long
    frame serves every rebalance date.

    ``cache_key`` should encode the universe + history range so distinct scans do
    not collide; ``cache_hours`` controls staleness (deep history is cached for
    much longer than a live scan — see ``config.HISTORY_CACHE_HOURS``).
    ``progress`` is an optional callable ``(done, total)`` for UI.
    """
    close_path, rawclose_path, volume_path = _price_cache_paths(cache_key)
    if (
        not force_refresh
        and _is_fresh(close_path, cache_hours)
        and rawclose_path.exists()
        and volume_path.exists()
    ):
        close = pd.read_parquet(close_path)
        raw_close = pd.read_parquet(rawclose_path)
        volume = pd.read_parquet(volume_path)
        if progress:
            progress(len(tickers), len(tickers))
        return close, raw_close, volume

    if start is None:
        start = asof - dt.timedelta(days=lookback_days)
    end = asof + dt.timedelta(days=1)  # yfinance end is exclusive

    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    batches = [
        tickers[i : i + config.DOWNLOAD_BATCH_SIZE]
        for i in range(0, len(tickers), config.DOWNLOAD_BATCH_SIZE)
    ]

    closes, raws, volumes, done = [], [], [], 0
    for batch in batches:
        c, rc, v = _download_batch(batch, start, end)
        if not c.empty:
            closes.append(c)
            raws.append(rc)
            volumes.append(v)
        done += len(batch)
        if progress:
            progress(done, len(tickers))

    if not closes:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    close = pd.concat(closes, axis=1).sort_index()
    raw_close = pd.concat(raws, axis=1).sort_index()
    volume = pd.concat(volumes, axis=1).sort_index()
    # Drop all-NaN columns (tickers Yahoo had nothing for) and align the others.
    close = close.dropna(axis=1, how="all")
    raw_close = raw_close.reindex(columns=close.columns)
    volume = volume.reindex(columns=close.columns)

    close.to_parquet(close_path)
    raw_close.to_parquet(rawclose_path)
    volume.to_parquet(volume_path)
    return close, raw_close, volume


# --------------------------------------------------------------------------- #
# Benchmark index history (for backtest comparison)
# --------------------------------------------------------------------------- #
def fetch_benchmark(
    ticker: str,
    cache_key: str,
    start: dt.date,
    asof: dt.date,
    force_refresh: bool = False,
    cache_hours: float = config.HISTORY_CACHE_HOURS,
) -> pd.Series:
    """Return an adjusted daily close series for a benchmark index.

    ``ticker`` is a Yahoo index symbol (e.g. ``^NSEI``). Cached to ``data/`` like
    the price frames. Returns an empty Series if Yahoo has nothing.
    """
    path = config.DATA_DIR / f"benchmark_{cache_key}.parquet"
    if not force_refresh and _is_fresh(path, cache_hours):
        return pd.read_parquet(path)["close"]

    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=(asof + dt.timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype="float64", name="close")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # single ticker can still come back wide
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = "close"
    close.to_frame().to_parquet(path)
    return close


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
    start: dt.date | None = None,
    force_refresh: bool = False,
    cache_hours: float = config.PRICE_CACHE_HOURS,
    price_progress=None,
    market_progress=None,
) -> PriceData:
    """Fetch price history and market data, returning a :class:`PriceData`.

    Pass an explicit ``start`` (and a longer ``cache_hours``) for the multi-year
    history a backtest needs; omit it for a fast ~13-month live scan.
    """
    close, raw_close, volume = fetch_prices(
        tickers, cache_key, asof, lookback_days, start=start,
        force_refresh=force_refresh, cache_hours=cache_hours, progress=price_progress,
    )
    market = fetch_market_data(tickers, cache_key, force_refresh, progress=market_progress)

    available = set(close.columns)
    failed = [t for t in tickers if t not in available]
    return PriceData(
        close=close, raw_close=raw_close, volume=volume, market=market, failed=failed
    )
