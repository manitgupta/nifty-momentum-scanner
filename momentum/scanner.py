"""Core Nifty500 Momentum 50 computation.

Implements the methodology in PDF section 20 as a sequence of pure functions on
pandas frames (no I/O), so the maths is testable in isolation:

    1. compute_momentum      - sigma_p, 6M/12M returns, momentum ratios
    2. apply_eligibility     - listing history + liquidity/turnover-ratio filters
    3. normalize_and_score   - cross-sectional Z-scores -> Normalized Momentum Score
    4. select_and_weight     - top-N selection + free-float-cap x score weighting

``run_scan`` wires them together and returns a :class:`ScanResult`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from . import config


# --------------------------------------------------------------------------- #
# Parameters & results
# --------------------------------------------------------------------------- #
@dataclass
class ScanParams:
    """Tunable knobs for a scan; defaults mirror the published methodology."""

    asof: dt.date
    top_n: int = config.TOP_N
    weight_12m: float = config.WEIGHT_12M
    weight_6m: float = config.WEIGHT_6M
    trading_days: int = config.TRADING_DAYS_PER_YEAR
    zscore_ddof: int = config.ZSCORE_DDOF
    vol_ddof: int = config.VOL_DDOF
    # momentum window (months). Defaults reproduce the published 6M/12M legs;
    # column names keep the 6m/12m labels even when these are retuned.
    lookback_long_months: int = config.MONTHS_12
    lookback_short_months: int = config.MONTHS_6
    skip_months: int = config.SKIP_MONTHS          # "12-1" style: skip recent N months
    vol_lookback_months: int = config.VOL_LOOKBACK_MONTHS
    # eligibility
    min_listing_days: int = config.MIN_LISTING_CALENDAR_DAYS
    min_observations: int = config.MIN_OBSERVATIONS
    liquidity_bottom_pct: float = config.LIQUIDITY_BOTTOM_PERCENTILE
    turnover_lookback_months: int = config.TURNOVER_LOOKBACK_MONTHS
    apply_listing_filter: bool = True
    apply_liquidity_filter: bool = True
    apply_turnover_ratio_filter: bool = True
    # weighting
    cap_absolute: float = config.CAP_ABSOLUTE
    cap_relative: float = config.CAP_RELATIVE
    free_float_factor: float = 1.0  # ff-mcap approx = full mcap * factor (see README)
    # "mcap_score" = methodology (free-float mcap x score, 5%/5x capped); the others
    # ("score", "equal", "inverse_vol") are price-only and uncapped — needed for
    # universes without market caps (ETFs) and identical to the backtest schemes.
    weighting: str = "mcap_score"
    # Diversification: at most this many names per Industry in the selection
    # (None/0 = no cap). Applied while walking down the ranking.
    max_per_sector: int | None = None


@dataclass
class ScanResult:
    results: pd.DataFrame            # full per-stock table (all intermediates)
    selected: pd.DataFrame          # the top-N with weights
    eligible_count: int
    exclusions: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _asof_price(series: pd.Series, when: dt.date) -> float:
    """Last valid close on or before ``when`` (methodology uses last trading day)."""
    s = series.dropna()
    s = s[s.index <= pd.Timestamp(when)]
    return float(s.iloc[-1]) if len(s) else np.nan


def _row_pos(index: pd.DatetimeIndex, when) -> int:
    """Position of the last index entry on/before ``when`` (-1 if none). Index must be sorted."""
    return int(index.searchsorted(pd.Timestamp(when), side="right")) - 1


def _last_valid(values: np.ndarray, valid: np.ndarray, pos: int) -> np.ndarray:
    """Per-column last non-NaN value within rows ``[0, pos]`` (NaN where none).

    Vectorised equivalent of ``frame.iloc[:pos+1].ffill().iloc[-1]`` — it only
    ever reads rows at or before ``pos``, so it cannot look ahead.
    """
    n = values.shape[1]
    if pos < 0 or n == 0:
        return np.full(n, np.nan)
    rev = valid[pos::-1]                       # rows pos, pos-1, …, 0
    has = rev.any(axis=0)
    rows = pos - rev.argmax(axis=0)            # first True in reverse = last valid
    out = values[rows, np.arange(n)]
    return np.where(has, out, np.nan)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame if frame.index.is_monotonic_increasing else frame.sort_index()


def _asof_closes(frame: pd.DataFrame, when: dt.date) -> pd.Series:
    """Per-column last valid close on or before ``when``; empty frame -> empty Series."""
    if frame is None or frame.empty:
        return pd.Series(dtype="float64")
    frame = _sorted(frame)
    vals = frame.to_numpy(dtype="float64")
    out = _last_valid(vals, ~np.isnan(vals), _row_pos(frame.index, when))
    return pd.Series(out, index=frame.columns)


# --------------------------------------------------------------------------- #
# 1. Momentum
# --------------------------------------------------------------------------- #
def compute_momentum(close: pd.DataFrame, params: ScanParams) -> pd.DataFrame:
    """Per-ticker momentum inputs.

    Returns a DataFrame indexed by ticker with:
        ret_6m, ret_12m, sigma, mr_6, mr_12, n_obs, first_date, last_date

    The two return legs span ``[asof - lookback_{long,short}_months, asof - skip_months]``.
    With ``skip_months = 0`` the end point is ``asof`` (methodology default);
    a positive skip reproduces "12-1"-style momentum. Column names retain the
    6m/12m labels regardless of the configured window lengths.
    """
    asof = params.asof
    end = asof - relativedelta(months=params.skip_months)   # skip most recent N months
    date_6m = asof - relativedelta(months=params.lookback_short_months)
    date_12m = asof - relativedelta(months=params.lookback_long_months)
    vol_start = asof - relativedelta(months=params.vol_lookback_months)

    cols = ["ret_6m", "ret_12m", "sigma", "mr_6", "mr_12", "n_obs", "first_date", "last_date"]
    if close is None or close.empty:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="YFTicker"))

    # Vectorised across tickers (one numpy pass per as-of instead of a per-ticker
    # loop) — this is what makes daily/weekly backtests (thousands of scans) fast.
    close = _sorted(close.loc[:, close.notna().any()])   # skip never-traded tickers
    idx = close.index
    vals = close.to_numpy(dtype="float64")
    valid = ~np.isnan(vals)

    p0 = _last_valid(vals, valid, _row_pos(idx, end))
    p6 = _last_valid(vals, valid, _row_pos(idx, date_6m))
    p12 = _last_valid(vals, valid, _row_pos(idx, date_12m))
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_6m = np.where(~np.isnan(p6) & (p6 != 0), p0 / p6 - 1.0, np.nan)
        ret_12m = np.where(~np.isnan(p12) & (p12 != 0), p0 / p12 - 1.0, np.nan)

    # Annualised std-dev of lognormal daily returns over (vol_start, asof]. Each
    # return is taken between consecutive *valid* closes inside the window (gaps
    # are bridged, the first in-window close has no return) — as the per-series
    # ``dropna()`` + ``shift(1)`` formulation did.
    hi = _row_pos(idx, asof)
    lo = int(idx.searchsorted(pd.Timestamp(vol_start), side="right"))
    block = close.iloc[lo:hi + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(block / block.ffill().shift(1))
    n_ret = log_ret.notna().sum(axis=0).to_numpy()
    sigma = log_ret.std(ddof=params.vol_ddof).to_numpy() * np.sqrt(params.trading_days)
    sigma = np.where(n_ret >= 2, sigma, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        pos_sigma = ~np.isnan(sigma) & (sigma > 0)
        mr_6 = np.where(pos_sigma, ret_6m / sigma, np.nan)
        mr_12 = np.where(pos_sigma, ret_12m / sigma, np.nan)

    n = len(idx)
    out = pd.DataFrame(
        {
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "sigma": sigma,
            "mr_6": mr_6,
            "mr_12": mr_12,
            "n_obs": valid[: hi + 1].sum(axis=0).astype(int) if hi >= 0 else np.zeros(vals.shape[1], int),
            "first_date": idx[valid.argmax(axis=0)],
            "last_date": idx[n - 1 - valid[::-1].argmax(axis=0)],
        },
        index=pd.Index(close.columns, name="YFTicker"),
    )
    return out


# --------------------------------------------------------------------------- #
# Liquidity metrics (turnover)
# --------------------------------------------------------------------------- #
def compute_liquidity(
    close: pd.DataFrame, volume: pd.DataFrame, market: pd.DataFrame, params: ScanParams
) -> pd.DataFrame:
    """Per-ticker 6M average daily turnover and turnover ratio.

    turnover_t         = close_t * volume_t   (INR traded value; approximate, as
                         adj-close x raw-volume — adequate for a percentile filter)
    avg_daily_turnover = mean over the trailing ``turnover_lookback_months``
    turnover_ratio     = avg_daily_turnover / free-float market cap
    """
    start = params.asof - relativedelta(months=params.turnover_lookback_months)
    mask = (close.index > pd.Timestamp(start)) & (close.index <= pd.Timestamp(params.asof))

    turnover = (close[mask] * volume[mask])
    avg_turnover = turnover.mean(axis=0)  # per ticker

    ff_mcap = market["market_cap"] * params.free_float_factor
    ratio = avg_turnover / ff_mcap.reindex(avg_turnover.index)

    return pd.DataFrame(
        {"avg_daily_turnover": avg_turnover, "turnover_ratio": ratio, "ff_mcap": ff_mcap}
    )


# --------------------------------------------------------------------------- #
# 2. Eligibility
# --------------------------------------------------------------------------- #
def apply_eligibility(
    df: pd.DataFrame, params: ScanParams
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Flag ineligible stocks and return (annotated_df, exclusion_counts).

    ``df`` must already contain momentum + liquidity columns. Adds boolean
    ``eligible`` and a comma-joined ``exclusion_reasons`` column. Percentile
    filters are computed over the stocks surviving the prior (non-percentile)
    filters, matching how NSE screens within the universe.
    """
    df = df.copy()
    reasons = {t: [] for t in df.index}

    # -- Listing history & computability ------------------------------------ #
    if params.apply_listing_filter:
        listed_days = (pd.Timestamp(params.asof) - df["first_date"]).dt.days
        too_new = listed_days < params.min_listing_days
        too_few = df["n_obs"] < params.min_observations
        for t in df.index[too_new | too_few]:
            reasons[t].append("listing_history<1y")

    # Momentum must be computable regardless of toggle (can't score a NaN).
    no_momentum = df["mr_6"].isna() | df["mr_12"].isna()
    for t in df.index[no_momentum]:
        if "listing_history<1y" not in reasons[t]:
            reasons[t].append("insufficient_price_data")

    def _survivors() -> pd.Index:
        return pd.Index([t for t in df.index if not reasons[t]])

    # -- Liquidity: bottom-10-pct by 6M avg daily turnover ------------------- #
    if params.apply_liquidity_filter:
        surv = _survivors()
        vals = df.loc[surv, "avg_daily_turnover"].dropna()
        if len(vals):
            thresh = np.percentile(vals, params.liquidity_bottom_pct)
            for t in vals.index[vals < thresh]:
                reasons[t].append("low_turnover")

    # -- Turnover ratio: bottom-10-pct -------------------------------------- #
    if params.apply_turnover_ratio_filter:
        surv = _survivors()
        vals = df.loc[surv, "turnover_ratio"].dropna()
        if len(vals):
            thresh = np.percentile(vals, params.liquidity_bottom_pct)
            for t in vals.index[vals < thresh]:
                reasons[t].append("low_turnover_ratio")

    df["exclusion_reasons"] = [", ".join(reasons[t]) for t in df.index]
    df["eligible"] = df["exclusion_reasons"].eq("")

    # Count each reason (a stock may carry more than one).
    counts: dict[str, int] = {}
    for rs in reasons.values():
        for r in rs:
            counts[r] = counts.get(r, 0) + 1
    return df, counts


# --------------------------------------------------------------------------- #
# 3. Normalize & score
# --------------------------------------------------------------------------- #
def normalize_and_score(df: pd.DataFrame, params: ScanParams) -> pd.DataFrame:
    """Compute Z-scores over the eligible universe and the Normalized Momentum Score.

    Z, weighted-average Z and score columns are filled only for eligible rows;
    ineligible rows keep NaN so they never enter the ranking.
    """
    df = df.copy()
    elig = df[df["eligible"]]

    mu12, sd12 = elig["mr_12"].mean(), elig["mr_12"].std(ddof=params.zscore_ddof)
    mu6, sd6 = elig["mr_6"].mean(), elig["mr_6"].std(ddof=params.zscore_ddof)

    # Guard the degenerate case (e.g. a single eligible stock -> zero spread):
    # fall back to an all-NaN Series so downstream `.where` stays vectorised.
    nan_series = pd.Series(np.nan, index=df.index)
    z12 = (df["mr_12"] - mu12) / sd12 if sd12 else nan_series
    z6 = (df["mr_6"] - mu6) / sd6 if sd6 else nan_series
    df["z_12"] = z12.where(df["eligible"])
    df["z_6"] = z6.where(df["eligible"])

    df["waz"] = params.weight_12m * df["z_12"] + params.weight_6m * df["z_6"]

    # Normalized Momentum Score: (1 + WAZ) if WAZ >= 0 else 1 / (1 - WAZ).
    waz = df["waz"]
    df["score"] = np.where(waz >= 0, 1.0 + waz, 1.0 / (1.0 - waz))
    df["score"] = df["score"].where(df["eligible"] & waz.notna())

    df["rank"] = df["score"].rank(ascending=False, method="first")
    return df.sort_values("score", ascending=False, na_position="last")


# --------------------------------------------------------------------------- #
# 4. Selection & weighting
# --------------------------------------------------------------------------- #
def _iterative_cap(raw: pd.Series, cap: pd.Series, max_iter: int) -> pd.Series:
    """Cap weights at per-stock ``cap`` and redistribute excess proportionally.

    Capped stocks are frozen at their cap; the residual is shared among the
    remaining stocks in proportion to their raw (uncapped) weight, iterating
    until no free stock exceeds its cap.
    """
    raw = raw.astype(float).clip(lower=0)
    is_capped = pd.Series(False, index=raw.index)

    w = raw / raw.sum() if raw.sum() > 0 else raw
    for _ in range(max_iter):
        w = pd.Series(0.0, index=raw.index)
        w[is_capped] = cap[is_capped]
        remaining = 1.0 - float(w[is_capped].sum())
        free = ~is_capped
        free_raw = raw[free]
        if free_raw.sum() <= 0 or remaining <= 0:
            break
        w[free] = remaining * free_raw / free_raw.sum()
        newly = free & (w > cap + 1e-12)
        if not newly.any():
            break
        is_capped = is_capped | newly

    # Degenerate fallback: if every stock is capped and the caps cannot sum to 1
    # (only possible with a very small universe), scale up proportionally so the
    # portfolio still sums to 1. Flagged by the caller via cap.sum() < 1.
    total = float(w.sum())
    if 0 < total < 1 - 1e-9:
        w = w / total
    return w


WEIGHTING_SCHEMES = ("mcap_score", "score", "equal", "inverse_vol")


def select_top(
    ranked: pd.DataFrame,
    top_n: int,
    max_per_sector: int | None = None,
    keep=None,
    exit_rank: int | None = None,
) -> list:
    """Choose up to ``top_n`` names from a scored frame, best score first.

    ``max_per_sector`` caps how many names may come from one ``Industry`` (names
    over the cap are skipped and the walk continues down the ranking).

    ``keep`` + ``exit_rank`` implement a *rank buffer* (hysteresis): currently
    held names that still rank within ``exit_rank`` are retained first, and only
    the remaining slots are filled by the best-ranked newcomers. This is how real
    momentum indices damp turnover. Returns tickers in rank order.
    """
    if ranked is None or ranked.empty or top_n <= 0:
        return []
    order = ranked.sort_values("score", ascending=False, kind="mergesort")
    tickers = list(order.index)
    rank = {t: i + 1 for i, t in enumerate(tickers)}
    use_cap = bool(max_per_sector) and "Industry" in order.columns
    industry = order["Industry"].fillna("—").to_dict() if use_cap else {}
    counts: dict = {}
    chosen: list = []
    chosen_set: set = set()

    def _try_add(t) -> None:
        if use_cap and counts.get(industry[t], 0) >= max_per_sector:
            return
        chosen.append(t)
        chosen_set.add(t)
        if use_cap:
            counts[industry[t]] = counts.get(industry[t], 0) + 1

    if keep is not None and exit_rank:
        retained = sorted((t for t in set(keep) if rank.get(t, exit_rank + 1) <= exit_rank), key=rank.get)
        for t in retained:
            if len(chosen) >= top_n:
                break
            _try_add(t)
    for t in tickers:
        if len(chosen) >= top_n:
            break
        if t not in chosen_set:
            _try_add(t)
    return sorted(chosen, key=rank.get)


def scheme_weights(selected: pd.DataFrame, scheme: str) -> pd.Series:
    """Price-only weights over ``selected``: ``score`` (∝ score), ``equal`` or
    ``inverse_vol`` (∝ 1/σ — a simple risk-parity tilt). Degenerate inputs fall
    back to equal weight. Never reads market cap, so it is look-ahead-free."""
    if selected is None or selected.empty:
        return pd.Series(dtype="float64")
    n = len(selected)
    equal = pd.Series(1.0 / n, index=selected.index)
    if scheme == "equal":
        return equal
    if scheme == "inverse_vol":
        sig = selected["sigma"] if "sigma" in selected.columns else pd.Series(np.nan, index=selected.index)
        inv = (1.0 / sig.where(sig > 0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        total = float(inv.sum())
        return inv / total if total > 0 else equal
    s = selected["score"].clip(lower=0.0)
    total = float(s.sum())
    return s / total if total > 0 else equal


def select_and_weight(df: pd.DataFrame, params: ScanParams, keep=None, exit_rank: int | None = None) -> pd.DataFrame:
    """Take the top-N eligible stocks and assign weights.

    Methodology (``weighting="mcap_score"``):
        raw weight  = free-float mcap x Normalized Momentum Score
        cap_i       = min(cap_absolute, cap_relative x free-float-mcap-only weight_i)
    Other schemes use :func:`scheme_weights` (price-only, uncapped).
    ``keep`` / ``exit_rank`` apply a rank buffer around existing holdings (see
    :func:`select_top`) — used when rebalancing a live portfolio.
    """
    scored = df[df["score"].notna()].copy()
    chosen = select_top(scored, params.top_n, params.max_per_sector, keep=keep, exit_rank=exit_rank)
    selected = scored.loc[chosen].copy()
    if selected.empty:
        selected["weight"] = []
        return selected

    if params.weighting != "mcap_score":
        w = scheme_weights(selected, params.weighting)
        ff = selected["ff_mcap"].fillna(0.0) if "ff_mcap" in selected else pd.Series(0.0, index=selected.index)
        selected["raw_weight"] = w
        selected["ffmc_weight"] = ff / ff.sum() if ff.sum() > 0 else np.nan
        selected["cap"] = np.nan
        selected["weight"] = w
        selected["selection_rank"] = range(1, len(selected) + 1)
        selected.attrs["caps_infeasible"] = False
        return selected

    ff = selected["ff_mcap"].fillna(0.0)
    raw = ff * selected["score"]
    if raw.sum() <= 0:  # no mcap data -> fall back to equal-by-score
        raw = selected["score"].clip(lower=0)

    # Free-float-mcap-only weight (the cap reference), relative to the selected set.
    wff = ff / ff.sum() if ff.sum() > 0 else pd.Series(1.0 / len(selected), index=selected.index)
    cap = np.minimum(params.cap_absolute, params.cap_relative * wff)

    selected["raw_weight"] = raw / raw.sum()
    selected["ffmc_weight"] = wff
    selected["cap"] = cap
    selected["weight"] = _iterative_cap(raw, cap, config.CAP_MAX_ITERATIONS)
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected.attrs["caps_infeasible"] = bool(cap.sum() < 1.0 - 1e-9)
    return selected


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_scan(universe: pd.DataFrame, price_data, params: ScanParams) -> ScanResult:
    """Run the full pipeline for a universe and fetched price data.

    ``universe`` has columns Symbol/Company/Industry/Segment/YFTicker.
    ``price_data`` is a :class:`momentum.data.PriceData`.
    """
    close, volume, market = price_data.close, price_data.volume, price_data.market
    notes: list[str] = []
    # The universe defines the eligible set: ignore price columns for tickers
    # outside it (e.g. a stale cache holding names since dropped from the index).
    uni_tickers = set(universe["YFTicker"])
    if close is not None and not close.empty and not set(close.columns) <= uni_tickers:
        keep = [t for t in close.columns if t in uni_tickers]
        close = close[keep]
        volume = volume.reindex(columns=keep) if volume is not None else volume

    momentum = compute_momentum(close, params)
    liquidity = compute_liquidity(close, volume, market, params)

    # Raw closing price on the as-of date (for display). Fall back to adjusted
    # close when a caller did not supply a raw-close frame.
    raw_close = getattr(price_data, "raw_close", None)
    price_source = raw_close if (raw_close is not None and not raw_close.empty) else close
    close_price = _asof_closes(price_source, params.asof).rename("close_price")

    df = momentum.join(liquidity, how="left")
    df = df.join(market, how="left")
    df = df.join(close_price, how="left")

    df, exclusions = apply_eligibility(df, params)
    df = normalize_and_score(df, params)

    # Attach universe metadata.
    meta = universe.set_index("YFTicker")[["Symbol", "Company", "Industry", "Segment"]]
    df = df.join(meta, how="left")

    selected = select_and_weight(df, params)

    eligible_count = int(df["eligible"].sum())
    if selected.attrs.get("caps_infeasible"):
        notes.append(
            "Per-stock caps cannot sum to 100% for this small a selection; "
            "weights were rescaled proportionally (caps effectively non-binding)."
        )
    if price_data.failed:
        notes.append(f"{len(price_data.failed)} ticker(s) had no price data and were skipped.")
    mcap_missing = int(market["market_cap"].isna().sum()) if "market_cap" in market else len(close.columns)
    if mcap_missing and (params.weighting == "mcap_score" or params.apply_turnover_ratio_filter):
        what = "turnover-ratio filter/weights" if params.weighting == "mcap_score" else "turnover-ratio filter"
        notes.append(f"{mcap_missing} ticker(s) missing market cap; {what} approximate for those.")

    return ScanResult(
        results=df,
        selected=selected,
        eligible_count=eligible_count,
        exclusions=exclusions,
        notes=notes,
    )
