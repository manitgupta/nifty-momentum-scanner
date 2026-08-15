"""Fetch and assemble the stock universe from NSE index constituent lists.

The Nifty500 Momentum 50 draws from the Nifty 500. We let the user choose which
market-cap segments make up the scan universe:

    Largecap -> Nifty 100
    Midcap   -> Nifty Midcap 150
    Small    -> Nifty Smallcap 250

Each is published by NSE as a CSV with columns:
    Company Name, Industry, Symbol, Series, ISIN Code
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

from . import config


def _cache_path(segment: str) -> Path:
    return config.DATA_DIR / f"constituents_{segment.lower()}.csv"


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def fetch_segment(segment: str, force_refresh: bool = False) -> pd.DataFrame:
    """Return the constituent list for one segment as a tidy DataFrame.

    Columns: Symbol, Company, Industry, ISIN, Segment, YFTicker.
    Downloads from NSE (browser headers required) and caches to ``data/`` for
    ``config.UNIVERSE_CACHE_HOURS``. Falls back to a stale cache if the network
    call fails.
    """
    if segment not in config.SEGMENT_URLS:
        raise ValueError(f"Unknown segment {segment!r}; expected one of {config.SEGMENTS}")

    path = _cache_path(segment)

    if not force_refresh and _is_fresh(path, config.UNIVERSE_CACHE_HOURS):
        raw = pd.read_csv(path)
    else:
        try:
            resp = requests.get(
                config.SEGMENT_URLS[segment],
                headers=config.HTTP_HEADERS,
                timeout=config.HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text))
            raw.to_csv(path, index=False)
        except Exception as exc:  # network/HTTP failure -> fall back gracefully
            if path.exists():  # stale on-disk cache
                raw = pd.read_csv(path)
            else:
                seed = config.SEED_DIR / f"constituents_{segment.lower()}.csv"
                if seed.exists():  # bundled fallback (keeps the app alive on cloud)
                    raw = pd.read_csv(seed)
                else:
                    raise RuntimeError(
                        f"Could not download {segment} constituents; no cache or seed file: {exc}"
                    ) from exc

    return _normalise(raw, segment)


def _normalise(raw: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Standardise NSE's column names and derive the Yahoo Finance ticker."""
    colmap = {
        "Company Name": "Company",
        "Industry": "Industry",
        "Symbol": "Symbol",
        "ISIN Code": "ISIN",
    }
    cols = {c: colmap[c] for c in raw.columns if c in colmap}
    df = raw.rename(columns=cols)

    keep = [c for c in ["Symbol", "Company", "Industry", "ISIN"] if c in df.columns]
    df = df[keep].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df = df[df["Symbol"].ne("") & df["Symbol"].notna()]
    df["Segment"] = segment
    df["YFTicker"] = df["Symbol"] + config.YF_SUFFIX
    return df.reset_index(drop=True)


def build_universe(segments: list[str], force_refresh: bool = False) -> pd.DataFrame:
    """Concatenate the chosen segments into one de-duplicated universe.

    If a symbol appears in more than one selected segment (rare across NSE's
    size buckets, but guarded anyway), the first occurrence in ``config.SEGMENTS``
    order wins so each stock is labelled with a single segment.
    """
    if not segments:
        raise ValueError("Select at least one segment (Largecap / Midcap / Small).")

    ordered = [s for s in config.SEGMENTS if s in segments]
    frames = [fetch_segment(s, force_refresh=force_refresh) for s in ordered]
    universe = pd.concat(frames, ignore_index=True)
    universe = universe.drop_duplicates(subset="Symbol", keep="first").reset_index(drop=True)
    return universe
