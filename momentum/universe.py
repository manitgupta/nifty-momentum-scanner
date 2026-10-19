"""Fetch and assemble the stock universe from NSE index constituent lists.

The Nifty500 Momentum 50 draws from the Nifty 500. We let the user choose which
market-cap segments make up the scan universe:

    Largecap -> Nifty 100
    Midcap   -> Nifty Midcap 150
    Small    -> Nifty Smallcap 250

…or any whole NSE index (broad, sectoral, thematic — see ``config.INDEX_GROUPS``),
a curated list of liquid NSE-listed ETFs, or a free-form list of NSE symbols.

Each NSE index is published as a CSV with columns:
    Company Name, Industry, Symbol, Series, ISIN Code
"""

from __future__ import annotations

import io
import re
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
    if segment == config.ETF_SEGMENT:  # curated, committed list — no download
        return _normalise(pd.read_csv(config.ETF_LIST_PATH), segment)
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
        raise ValueError("Select at least one universe segment or index.")

    ordered = [s for s in config.SEGMENTS if s in segments]
    frames = [fetch_segment(s, force_refresh=force_refresh) for s in ordered]
    universe = pd.concat(frames, ignore_index=True)
    universe = universe.drop_duplicates(subset="Symbol", keep="first").reset_index(drop=True)
    return universe


# --------------------------------------------------------------------------- #
# Custom universes (a user-typed list of NSE symbols — stocks and/or ETFs)
# --------------------------------------------------------------------------- #
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-_]{0,19}$")


def parse_symbols(text: str) -> tuple[list[str], list[str]]:
    """Split free text into (valid NSE symbols, rejected tokens).

    Accepts commas, spaces, semicolons or newlines as separators, is
    case-insensitive and tolerates a trailing ``.NS``. Order is preserved and
    duplicates are dropped.
    """
    good: list[str] = []
    bad: list[str] = []
    for tok in re.split(r"[\s,;]+", text or ""):
        t = tok.strip().upper()
        if not t:
            continue
        if t.endswith(".NS"):
            t = t[:-3]
        if _SYMBOL_RE.match(t):
            if t not in good:
                good.append(t)
        else:
            bad.append(tok.strip())
    return good, bad


def _reference_table() -> pd.DataFrame:
    """Symbol -> Company/Industry/ISIN from the bundled lists (offline lookup)."""
    frames = []
    for seg in ("NiftyTotalMarket", config.ETF_SEGMENT):
        try:
            if seg == config.ETF_SEGMENT:
                frames.append(_normalise(pd.read_csv(config.ETF_LIST_PATH), seg))
            else:
                path = _cache_path(seg)
                src = path if path.exists() else config.SEED_DIR / f"constituents_{seg.lower()}.csv"
                frames.append(_normalise(pd.read_csv(src), seg))
        except Exception:  # a missing reference list only costs us labels
            continue
    if not frames:
        return pd.DataFrame(columns=["Symbol", "Company", "Industry", "ISIN"]).set_index("Symbol")
    ref = pd.concat(frames, ignore_index=True).drop_duplicates("Symbol", keep="last")
    return ref.set_index("Symbol")[[c for c in ("Company", "Industry", "ISIN") if c in ref.columns]]


def build_custom_universe(symbols: list[str]) -> pd.DataFrame:
    """Universe from an explicit list of NSE symbols (stocks or ETFs).

    Company/Industry are filled from the bundled Total Market and ETF lists when
    the symbol is known; unknown symbols keep their ticker as the name and are
    still scanned (Yahoo decides whether they have price data).
    """
    if not symbols:
        raise ValueError("Enter at least one NSE symbol for the custom universe.")
    ref = _reference_table()
    rows = []
    for sym in symbols:
        known = sym in ref.index
        rows.append({
            "Symbol": sym,
            "Company": ref.at[sym, "Company"] if known and "Company" in ref.columns else sym,
            "Industry": ref.at[sym, "Industry"] if known and "Industry" in ref.columns else "Unclassified",
            "ISIN": ref.at[sym, "ISIN"] if known and "ISIN" in ref.columns else "",
            "Segment": config.CUSTOM_SEGMENT,
            "YFTicker": sym + config.YF_SUFFIX,
        })
    return pd.DataFrame(rows)
