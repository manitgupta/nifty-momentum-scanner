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


# --------------------------------------------------------------------------- #
# 1. Momentum
# --------------------------------------------------------------------------- #
def compute_momentum(close: pd.DataFrame, params: ScanParams) -> pd.DataFrame:
    """Per-ticker momentum inputs.

    Returns a DataFrame indexed by ticker with:
        ret_6m, ret_12m, sigma, mr_6, mr_12, n_obs, first_date, last_date
    """
    asof = params.asof
    date_6m = asof - relativedelta(months=config.MONTHS_6)
    date_12m = asof - relativedelta(months=config.MONTHS_12)
    vol_start = asof - relativedelta(years=1)

    rows = []
    for ticker in close.columns:
        series = close[ticker].dropna()
        if series.empty:
            continue

        p0 = _asof_price(series, asof)
        p6 = _asof_price(series, date_6m)
        p12 = _asof_price(series, date_12m)

        ret_6m = p0 / p6 - 1.0 if (p6 and not np.isnan(p6)) else np.nan
        ret_12m = p0 / p12 - 1.0 if (p12 and not np.isnan(p12)) else np.nan

        # Annualised std-dev of lognormal daily returns over the trailing 1 year.
        window = series[(series.index > pd.Timestamp(vol_start)) & (series.index <= pd.Timestamp(asof))]
        log_ret = np.log(window / window.shift(1)).dropna()
        if len(log_ret) >= 2:
            sigma = float(log_ret.std(ddof=params.vol_ddof) * np.sqrt(params.trading_days))
        else:
            sigma = np.nan

        mr_6 = ret_6m / sigma if sigma and not np.isnan(sigma) and sigma > 0 else np.nan
        mr_12 = ret_12m / sigma if sigma and not np.isnan(sigma) and sigma > 0 else np.nan

        rows.append(
            {
                "YFTicker": ticker,
                "ret_6m": ret_6m,
                "ret_12m": ret_12m,
                "sigma": sigma,
                "mr_6": mr_6,
                "mr_12": mr_12,
                "n_obs": int(series[series.index <= pd.Timestamp(asof)].shape[0]),
                "first_date": series.index.min(),
                "last_date": series.index.max(),
            }
        )

    return pd.DataFrame(rows).set_index("YFTicker")


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

    z12 = (df["mr_12"] - mu12) / sd12 if sd12 else np.nan
    z6 = (df["mr_6"] - mu6) / sd6 if sd6 else np.nan
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


def select_and_weight(df: pd.DataFrame, params: ScanParams) -> pd.DataFrame:
    """Take the top-N eligible stocks and assign capped weights.

    raw weight  = free-float mcap x Normalized Momentum Score
    cap_i       = min(cap_absolute, cap_relative x free-float-mcap-only weight_i)
    """
    scored = df[df["score"].notna()].copy()
    selected = scored.nlargest(params.top_n, "score").copy()
    if selected.empty:
        selected["weight"] = []
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

    momentum = compute_momentum(close, params)
    liquidity = compute_liquidity(close, volume, market, params)

    df = momentum.join(liquidity, how="left")
    df = df.join(market, how="left")

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
    if market["market_cap"].isna().any():
        n = int(market["market_cap"].isna().sum())
        notes.append(f"{n} ticker(s) missing market cap; turnover-ratio filter/weights approximate for those.")

    return ScanResult(
        results=df,
        selected=selected,
        eligible_count=eligible_count,
        exclusions=exclusions,
        notes=notes,
    )
