"""Backtesting engine for the momentum strategy.

Turns the point-in-time :func:`momentum.scanner.run_scan` into a rebalancing
backtest over a defined period. The design is deliberately **honest & practical**
about what free data allows:

    * Selection at each rebalance flows through ``run_scan(asof=t)``, which slices
      every price frame to ``index <= t`` — so the momentum ranking is *price-only*
      and free of look-ahead at any historical date.
    * Weights are derived **only** from the (price-only) Normalized Momentum Score
      or equal-weight. The scanner's market-cap-capped ``weight`` column is
      *discarded* because it is built from a live market-cap snapshot (look-ahead),
      and the turnover-ratio eligibility filter (also live-mcap) is forced off.
    * The universe is *today's* index constituents, so results carry
      **survivorship bias** (optimistic). This is surfaced as a permanent note and
      a UI banner, never hidden.

The maths is a sequence of pure functions on pandas frames (no I/O), so it is
unit-testable in isolation; ``run_backtest`` wires them together. Price data is
fetched once (a long history) and ``run_scan`` is called per rebalance on the
same frames — fetch once, compute many.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from . import config
from .scanner import ScanParams, run_scan


# --------------------------------------------------------------------------- #
# Parameters & results
# --------------------------------------------------------------------------- #
@dataclass
class BacktestParams:
    """Configuration for a backtest.

    ``scan_params`` is the single source of truth for *selection* (top_n,
    lookbacks, weights, filters); its ``asof`` is overwritten per rebalance and
    ``apply_turnover_ratio_filter`` is forced off inside the loop.
    """

    scan_params: ScanParams
    start: dt.date                 # requested first-rebalance date (may be pushed later)
    end: dt.date                   # final valuation date (clamped to last trading day <= end)
    rebalance_months: int = 6      # cadence: 1=monthly, 3=quarterly, 6=semiannual, 12=annual
    weighting: str = "score"       # "score" | "equal" — look-ahead-free only; never mcap-cap
    cost_bps: float = 0.0          # bps charged on traded notional Σ|Δw| each rebalance
    rf_annual: float = 0.0         # risk-free p.a. for Sharpe/Sortino (rf=0 default)
    trading_days: int = config.TRADING_DAYS_PER_YEAR
    min_history_months: int | None = None  # override; None -> derived from scan_params


@dataclass
class BacktestResult:
    equity_curve: pd.Series                         # daily, NET of costs, base 1.0 at first rebalance
    benchmark_curve: pd.Series | None               # daily, same index as equity_curve, base 1.0
    drawdown: pd.Series                             # daily strategy drawdown (<= 0)
    period_returns: pd.DataFrame                    # per-rebalance rows (see run_backtest)
    holdings: dict[pd.Timestamp, pd.DataFrame]      # entry date -> selected holdings + weights
    rebalance_dates: list[pd.Timestamp]
    schedule: pd.DataFrame                          # scheduled_date | trading_date | used | reason
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    relative_metrics: dict[str, float]
    notes: list[str] = field(default_factory=list)
    params: BacktestParams | None = None
    benchmark_name: str = ""


# --------------------------------------------------------------------------- #
# Calendar / schedule helpers
# --------------------------------------------------------------------------- #
def _last_trading_day(cal: pd.DatetimeIndex, when) -> pd.Timestamp | None:
    """Last index entry on/before ``when`` (mirrors ``scanner._asof_price``)."""
    ts = pd.Timestamp(when)
    prior = cal[cal <= ts]
    return prior[-1] if len(prior) else None


def _required_history_days(sp: ScanParams) -> int:
    """Calendar days of trailing history a computable scan needs before a rebalance."""
    months = (
        max(sp.lookback_long_months, sp.lookback_short_months, sp.vol_lookback_months)
        + sp.skip_months
    )
    return int(round(months * 30.44)) + 45  # month≈30.44d + holiday buffer


def generate_schedule(
    cal: pd.DatetimeIndex, start: dt.date, end: dt.date, rebalance_months: int
) -> pd.DataFrame:
    """Rebalance schedule: calendar dates stepping by ``rebalance_months`` from
    ``start`` to ``end``, each snapped to the last trading day on/before it.

    Returns a DataFrame with columns scheduled_date, trading_date, used, reason.
    Dates that snap to a trading day already used (holiday collisions) or that
    have no trading day are flagged ``used=False`` with a reason.
    """
    rows: list[dict] = []
    seen: set[pd.Timestamp] = set()
    d = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    step = max(1, int(rebalance_months))
    while d <= end_ts:
        td = _last_trading_day(cal, d)
        used, reason = False, ""
        if td is None:
            reason = "no trading day on/before scheduled date"
        elif td in seen:
            reason = "snapped to an already-used trading day"
        else:
            used = True
            seen.add(td)
        rows.append({"scheduled_date": d, "trading_date": td, "used": used, "reason": reason})
        d = d + relativedelta(months=step)
    return pd.DataFrame(rows, columns=["scheduled_date", "trading_date", "used", "reason"])


def _feasible_first(cal: pd.DatetimeIndex, sp: ScanParams, min_history_months: int | None) -> pd.Timestamp:
    """Earliest trading day with enough trailing history for a computable scan."""
    if len(cal) == 0:
        raise ValueError("Empty price calendar.")
    if min_history_months is not None:
        need_days = int(round(min_history_months * 30.44)) + 45
    else:
        need_days = _required_history_days(sp)
    cutoff = cal[0] + pd.Timedelta(days=need_days)
    later = cal[cal >= cutoff]
    return later[0] if len(later) else cal[-1]


# --------------------------------------------------------------------------- #
# Weighting / pricing helpers
# --------------------------------------------------------------------------- #
def _target_weights(selected: pd.DataFrame, weighting: str) -> pd.Series:
    """Look-ahead-free target weights over ``selected`` (indexed by YFTicker).

    Uses the price-only ``score`` (``score / Σscore``) or equal weight. Never
    reads the scanner's mcap-based ``weight`` column. Empty selection -> empty.
    """
    if selected is None or selected.empty:
        return pd.Series(dtype="float64")
    if weighting == "equal":
        n = len(selected)
        return pd.Series(1.0 / n, index=selected.index)
    s = selected["score"].clip(lower=0.0)
    total = float(s.sum())
    if total > 0:
        return s / total
    return pd.Series(1.0 / len(selected), index=selected.index)  # degenerate -> equal


def _asof_prices(close: pd.DataFrame, tickers, when) -> pd.Series:
    """Per-ticker adjusted close, last valid on/before ``when`` (ffill)."""
    tk = list(tickers)
    if not tk:
        return pd.Series(dtype="float64")
    sub = close.loc[close.index <= pd.Timestamp(when), tk]
    if sub.empty:
        return pd.Series(np.nan, index=tk)
    return sub.ffill().iloc[-1]


def _hold_period_path(
    close: pd.DataFrame, weights: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp
) -> tuple[pd.Series, pd.Series]:
    """Buy-and-hold value path over ``(t0, t1]`` with shares FIXED at ``t0``.

    Shares are set from ``weights`` at the ``t0`` price (so Σ shares·p0 = 1); the
    returned value series is therefore based at ~1.0 at t0 and its last value is
    the exact buy-and-hold period multiple. Prices are ffilled within the period
    so a name that stops trading freezes at its last mark. Also returns the
    drifted end-of-period weights (for turnover measurement).
    """
    tk = list(weights.index)
    p0 = _asof_prices(close, tk, t0)
    shares = weights / p0
    days = close.index[(close.index > t0) & (close.index <= t1)]
    window = close.loc[(close.index >= t0) & (close.index <= t1), tk].copy()
    # Seed the ffill with the entry (as-of) marks so a name with no traded print
    # inside the period (halt/suspension straddling the rebalance) freezes at its
    # entry price rather than being silently valued at zero by ``.sum(axis=1)``.
    if len(window):
        window.iloc[0] = window.iloc[0].fillna(p0)
    window = window.ffill()
    val = window.reindex(days).mul(shares, axis=1).sum(axis=1)
    p1 = _asof_prices(close, tk, t1)
    w_end = shares * p1
    total = float(w_end.sum())
    w_end = w_end / total if total > 0 else pd.Series(dtype="float64")
    return val, w_end


def _turnover_and_cost(
    w_new: pd.Series, w_prev_drift: pd.Series, cost_bps: float
) -> tuple[float, float]:
    """(one-way turnover = 0.5·Σ|Δw|, cost_drag = cost_bps/1e4 · Σ|Δw|) vs drifted prior weights."""
    ix = w_new.index.union(w_prev_drift.index)
    dw = w_new.reindex(ix, fill_value=0.0) - w_prev_drift.reindex(ix, fill_value=0.0)
    traded = float(dw.abs().sum())
    return 0.5 * traded, (cost_bps / 1e4) * traded


def _align_benchmark(
    bench_close: pd.Series | None, daily_index: pd.DatetimeIndex, base_date: pd.Timestamp
) -> pd.Series | None:
    """Reindex benchmark adj-close onto ``daily_index`` (ffill), normalise to 1.0 at ``base_date``."""
    if bench_close is None or len(bench_close) == 0:
        return None
    combined = bench_close.reindex(bench_close.index.union(daily_index)).ffill()
    b = combined.reindex(daily_index)
    base = combined.loc[combined.index <= base_date].dropna()
    if base.empty:
        return None
    normed = b / float(base.iloc[-1])
    normed.name = "benchmark"
    return normed


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def _max_drawdown(equity: pd.Series):
    """(max_dd<=0, peak_date, trough_date, recovery_date|None) from a daily curve."""
    if equity is None or len(equity) < 2:
        return np.nan, None, None, None
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    trough_date = dd.idxmin()
    max_dd = float(dd.loc[trough_date])
    pre = equity.loc[:trough_date]
    peak_date = pre.idxmax()
    peak_val = float(equity.loc[peak_date])
    after = equity.loc[trough_date:]
    recovered = after[after >= peak_val]
    recovery_date = recovered.index[0] if len(recovered) else None
    return max_dd, peak_date, trough_date, recovery_date


def compute_metrics(
    equity: pd.Series,
    period_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    cost_drag: pd.Series | None = None,
    trading_days: int = config.TRADING_DAYS_PER_YEAR,
    rf_annual: float = 0.0,
    rebals_per_year: float | None = None,
) -> dict[str, float]:
    """Analytics for one daily equity curve (+ optional per-period series)."""
    out: dict[str, float] = {}
    if equity is None or len(equity) < 2:
        return out

    r = equity.pct_change().dropna()
    idx = equity.index
    years = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    n = trading_days
    rf_d = (1.0 + rf_annual) ** (1.0 / n) - 1.0

    out["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    out["cagr"] = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(n)) if len(r) >= 2 else np.nan
    out["ann_vol"] = ann_vol
    excess = r - rf_d
    out["sharpe"] = float(excess.mean() * n / ann_vol) if ann_vol and ann_vol > 0 else np.nan
    # Canonical target downside deviation: RMS of shortfalls below the target (rf),
    # measured over ALL observations, not the std of the loss subset about its own mean.
    short = excess.clip(upper=0.0)
    dd_dev = float(np.sqrt((short ** 2).mean()) * np.sqrt(n)) if len(excess) else np.nan
    out["sortino"] = float(excess.mean() * n / dd_dev) if dd_dev and dd_dev > 0 else np.nan

    max_dd, peak, trough, recov = _max_drawdown(equity)
    out["max_dd"] = max_dd
    out["mdd_peak_date"] = peak
    out["mdd_trough_date"] = trough
    out["mdd_recovery_date"] = recov
    out["calmar"] = float(out["cagr"] / abs(max_dd)) if max_dd and max_dd < 0 else np.nan

    out["years"] = float(years)

    if period_returns is not None and len(period_returns):
        pr = period_returns.dropna()
        if len(pr):
            out["hit_rate"] = float((pr > 0).mean())
            out["best_period"] = float(pr.max())
            out["worst_period"] = float(pr.min())
            out["best_period_date"] = pr.idxmax()
            out["worst_period_date"] = pr.idxmin()
        out["n_rebalances"] = int(len(period_returns))

    if turnover is not None and len(turnover):
        # Exclude the initial deploy-from-cash (first entry) from the average.
        interior = turnover.iloc[1:] if len(turnover) > 1 else turnover.iloc[:0]
        avg_to = float(interior.mean()) if len(interior) else np.nan
        out["avg_turnover"] = avg_to
        rpy = rebals_per_year if rebals_per_year else np.nan
        out["annual_turnover"] = float(avg_to * rpy) if avg_to == avg_to and rpy == rpy else np.nan
    if cost_drag is not None and len(cost_drag):
        out["cost_drag_annual"] = float(cost_drag.sum() / years)

    return out


def compute_relative_metrics(
    strat_curve: pd.Series,
    bench_curve: pd.Series | None,
    strat_period: pd.Series | None,
    bench_period: pd.Series | None,
    trading_days: int = config.TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Strategy-vs-benchmark analytics; all NaN when the benchmark is unavailable."""
    keys = ["beta", "alpha_annual", "tracking_error", "info_ratio",
            "up_capture", "down_capture", "corr", "hit_rate_vs_bench"]
    out = {k: np.nan for k in keys}
    if bench_curve is None or len(bench_curve) < 3:
        return out

    joined = pd.concat(
        {"s": strat_curve.pct_change(), "b": bench_curve.pct_change()}, axis=1
    ).dropna()
    if len(joined) >= 2 and joined["b"].var(ddof=1) > 0:
        rs, rb = joined["s"], joined["b"]
        n = trading_days
        beta = float(rs.cov(rb) / rb.var(ddof=1))
        out["beta"] = beta
        out["alpha_annual"] = float((rs.mean() - beta * rb.mean()) * n)
        te = float((rs - rb).std(ddof=1) * np.sqrt(n))
        out["tracking_error"] = te
        cagr_s = (strat_curve.iloc[-1] / strat_curve.iloc[0]) ** (
            365.25 / max((strat_curve.index[-1] - strat_curve.index[0]).days, 1)) - 1
        cagr_b = (bench_curve.iloc[-1] / bench_curve.iloc[0]) ** (
            365.25 / max((bench_curve.index[-1] - bench_curve.index[0]).days, 1)) - 1
        out["info_ratio"] = float((cagr_s - cagr_b) / te) if te > 0 else np.nan
        out["corr"] = float(rs.corr(rb))

    if strat_period is not None and bench_period is not None:
        p = pd.concat({"s": strat_period, "b": bench_period}, axis=1).dropna()
        if len(p):
            up = p[p["b"] > 0]
            dn = p[p["b"] < 0]
            if len(up) and up["b"].mean() != 0:
                out["up_capture"] = float(up["s"].mean() / up["b"].mean())
            if len(dn) and dn["b"].mean() != 0:
                out["down_capture"] = float(dn["s"].mean() / dn["b"].mean())
            out["hit_rate_vs_bench"] = float((p["s"] > p["b"]).mean())
    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _base_notes(params: BacktestParams) -> list[str]:
    return [
        "Universe = TODAY's index constituents → survivorship bias (results are optimistic).",
        f"Weighting = {params.weighting!r} (look-ahead-free); live market-cap capping is NOT used.",
        "Turnover-ratio eligibility filter disabled in backtests (it divides by a live market-cap snapshot).",
    ]


def _empty_result(notes: list[str], params: BacktestParams, schedule: pd.DataFrame | None = None) -> BacktestResult:
    empty = pd.Series(dtype="float64")
    return BacktestResult(
        equity_curve=empty, benchmark_curve=None, drawdown=empty,
        period_returns=pd.DataFrame(), holdings={}, rebalance_dates=[],
        schedule=schedule if schedule is not None else pd.DataFrame(),
        metrics={}, benchmark_metrics={}, relative_metrics={},
        notes=notes, params=params,
    )


def run_backtest(
    universe_df: pd.DataFrame,
    price_data,
    params: BacktestParams,
    benchmark_series: pd.Series | None = None,
    progress=None,
) -> BacktestResult:
    """Run a rebalancing backtest.

    ``universe_df`` has Symbol/Company/Industry/Segment/YFTicker; ``price_data``
    is a :class:`momentum.data.PriceData` covering the full history (its
    ``close`` is adjusted). ``benchmark_series`` is an adjusted-close Series
    (e.g. from ``data.fetch_benchmark``); ``progress`` is ``callable(done, total)``.
    """
    close = price_data.close
    notes = _base_notes(params)
    if close is None or close.empty:
        return _empty_result(notes + ["No price history available."], params)

    cal = close.index.sort_values()
    sp = params.scan_params

    # -- schedule ----------------------------------------------------------- #
    feas = _feasible_first(cal, sp, params.min_history_months)
    first = max(pd.Timestamp(params.start), feas)
    if first > pd.Timestamp(params.start):
        notes.append(f"First rebalance moved to {first.date()} (insufficient prior history before requested start).")
    schedule = generate_schedule(cal, first.date(), params.end, params.rebalance_months)
    rb = [t for t in schedule.loc[schedule["used"], "trading_date"].tolist()]

    end_td = _last_trading_day(cal, params.end)
    if end_td is not None and (not rb or end_td > rb[-1]):
        boundary = rb + [end_td]
    else:
        boundary = list(rb)
    if len(boundary) < 2:
        return _empty_result(
            notes + ["Window too short: need at least one full holding period (≥ 2 boundary dates). "
                     "Try more history years or a shorter rebalance interval."],
            params, schedule,
        )

    # -- PASS A: point-in-time selection (uses only prices <= entry) --------- #
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    meta_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for i, d in enumerate(rb):
        sp_d = replace(sp, asof=d.date(), apply_turnover_ratio_filter=False)
        sel = run_scan(universe_df, price_data, sp_d).selected
        w = _target_weights(sel, params.weighting)
        if not w.empty:  # drop names untradable at entry, renormalise
            p0 = _asof_prices(close, w.index, d)
            good = p0[(p0 > 0) & p0.notna()].index
            if len(good):
                w = w.loc[good] / w.loc[good].sum()
            else:
                w = pd.Series(dtype="float64")
        weights_by_date[d] = w
        meta_by_date[d] = sel
        if progress:
            progress(i + 1, len(rb))

    # -- PASS B: chain daily buy-and-hold multiples -------------------------- #
    E = 1.0
    w_prev = pd.Series(dtype="float64")
    eq: dict[pd.Timestamp, float] = {boundary[0]: 1.0}
    prows: list[dict] = []
    holdings: dict[pd.Timestamp, pd.DataFrame] = {}

    for i in range(len(boundary) - 1):
        t0, t1 = boundary[i], boundary[i + 1]
        w = weights_by_date.get(t0, pd.Series(dtype="float64"))

        one_way, cost = _turnover_and_cost(w, w_prev, params.cost_bps)
        E0 = E
        E *= (1.0 - cost)

        days = cal[(cal > t0) & (cal <= t1)]
        if w.empty or len(days) == 0:
            for d in days:
                eq[d] = E
            w_prev = pd.Series(dtype="float64")
            sret = E / E0 - 1.0
            if w.empty:
                notes.append(f"{t0.date()}: empty selection → held cash for the period.")
        else:
            path, w_prev = _hold_period_path(close, w, t0, t1)
            for d, m in path.items():
                eq[d] = E * float(m)
            E = E * float(path.iloc[-1])
            sret = E / E0 - 1.0

        prows.append({
            "rb_date": t0, "period_start": t0, "period_end": t1,
            "strat_ret": sret, "bench_ret": np.nan,
            "turnover": one_way, "cost_drag": cost,
            "n_holdings": int(len(w)), "empty": bool(w.empty),
        })
        if not w.empty:
            meta = meta_by_date[t0]
            cols = [c for c in ["Symbol", "Company", "Industry", "Segment", "score"] if c in meta.columns]
            holdings[t0] = meta.loc[w.index, cols].assign(weight=w)

    equity = pd.Series(eq).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    equity.name = "portfolio"

    period_returns = pd.DataFrame(prows).set_index("rb_date")

    # -- benchmark alignment ------------------------------------------------- #
    bench_curve = _align_benchmark(benchmark_series, equity.index, boundary[0])
    if benchmark_series is not None and (benchmark_series is None or len(benchmark_series) == 0 or bench_curve is None):
        notes.append("Benchmark data unavailable for this window; relative metrics omitted.")

    bench_period = None
    if bench_curve is not None:
        bench_period_vals = {}
        for _, prow in period_returns.iterrows():
            t0, t1 = prow["period_start"], prow["period_end"]
            b0 = bench_curve.loc[bench_curve.index <= t0].dropna()
            b1 = bench_curve.loc[bench_curve.index <= t1].dropna()
            if len(b0) and len(b1):
                bench_period_vals[prow.name] = float(b1.iloc[-1] / b0.iloc[-1] - 1.0)
        bench_period = pd.Series(bench_period_vals)
        period_returns["bench_ret"] = period_returns.index.map(bench_period).astype(float)

    # -- analytics ----------------------------------------------------------- #
    drawdown = (equity / equity.cummax() - 1.0)
    drawdown.name = "portfolio"
    rebals_per_year = 12.0 / max(1, params.rebalance_months)

    metrics = compute_metrics(
        equity, period_returns["strat_ret"], period_returns["turnover"],
        period_returns["cost_drag"], params.trading_days, params.rf_annual, rebals_per_year,
    )
    benchmark_metrics = {}
    if bench_curve is not None:
        benchmark_metrics = compute_metrics(
            bench_curve,
            bench_period if bench_period is not None else None,
            None, None, params.trading_days, params.rf_annual, rebals_per_year,
        )
    relative_metrics = compute_relative_metrics(
        equity, bench_curve, period_returns["strat_ret"], bench_period, params.trading_days,
    )

    return BacktestResult(
        equity_curve=equity,
        benchmark_curve=bench_curve,
        drawdown=drawdown,
        period_returns=period_returns,
        holdings=holdings,
        rebalance_dates=list(rb),
        schedule=schedule,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        relative_metrics=relative_metrics,
        notes=notes,
        params=params,
    )
