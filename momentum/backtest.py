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
from .scanner import ScanParams, _last_valid, _row_pos, run_scan, scheme_weights, select_top


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
    weighting: str = "score"       # "score" | "equal" | "inverse_vol" — look-ahead-free; never mcap-cap
    cost_bps: float = 0.0          # bps charged on traded notional Σ|Δw| each rebalance
    rf_annual: float = 0.0         # risk-free p.a. for Sharpe/Sortino (rf=0 default)
    trading_days: int = config.TRADING_DAYS_PER_YEAR
    min_history_months: int | None = None  # override; None -> derived from scan_params
    initial_capital: float = config.BACKTEST_INITIAL_CAPITAL  # ₹ notional for the trade ledger
    # Named cadence (key of config.REBALANCE_FREQUENCIES, e.g. "Weekly"); when set
    # it overrides ``rebalance_months`` and unlocks daily / weekly / fortnightly.
    rebalance_freq: str | None = None
    # Rank buffer: a held name is kept while it still ranks <= exit_rank (None = off).
    exit_rank: int | None = None
    # Market-regime filter: at each rebalance, if the regime index closes below its
    # ``regime_ma_days`` SMA, invest only ``regime_exposure_off`` of the book.
    regime_filter: bool = False
    regime_ma_days: int = config.REGIME_MA_DAYS
    regime_exposure_off: float = 0.0
    cash_yield_annual: float = 0.0  # return on uninvested cash (e.g. a liquid fund)


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
    membership_changes: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-rebalance entries/exits
    transactions: pd.DataFrame = field(default_factory=pd.DataFrame)        # per-rebalance trade ledger
    contributions: pd.DataFrame = field(default_factory=pd.DataFrame)       # per-period, per-name P&L
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


def freq_spec(params: BacktestParams) -> tuple[str, int]:
    """(unit, step) cadence of a backtest: a named ``rebalance_freq`` wins, else months."""
    if params.rebalance_freq:
        if params.rebalance_freq not in config.REBALANCE_FREQUENCIES:
            raise ValueError(f"Unknown rebalance frequency {params.rebalance_freq!r}; "
                             f"expected one of {list(config.REBALANCE_FREQUENCIES)}")
        return config.REBALANCE_FREQUENCIES[params.rebalance_freq]
    return "months", max(1, int(params.rebalance_months))


def rebalances_per_year(unit: str, step: int, trading_days: int = config.TRADING_DAYS_PER_YEAR) -> float:
    per_unit = {"days": float(trading_days), "weeks": 52.0, "months": 12.0}[unit]
    return per_unit / max(1, int(step))


def generate_schedule(
    cal: pd.DatetimeIndex, start: dt.date, end: dt.date, rebalance_months: int, unit: str = "months"
) -> pd.DataFrame:
    """Rebalance schedule from ``start`` to ``end``.

    ``unit="months"``/``"weeks"``: calendar dates stepping by ``rebalance_months``
    months/weeks, each snapped to the last trading day on/before it.
    ``unit="days"``: every ``rebalance_months``-th *trading* day (no snapping).

    Returns a DataFrame with columns scheduled_date, trading_date, used, reason.
    Dates that snap to a trading day already used (holiday collisions) or that
    have no trading day are flagged ``used=False`` with a reason.
    """
    cols = ["scheduled_date", "trading_date", "used", "reason"]
    step = max(1, int(rebalance_months))
    end_ts = pd.Timestamp(end)
    if unit == "days":
        first = _last_trading_day(cal, start)
        if first is None:
            later = cal[cal >= pd.Timestamp(start)]
            first = later[0] if len(later) else None
        if first is None:
            return pd.DataFrame(columns=cols)
        days = cal[(cal >= first) & (cal <= end_ts)][::step]
        return pd.DataFrame({"scheduled_date": days, "trading_date": days, "used": True, "reason": ""},
                            columns=cols)
    if unit not in ("months", "weeks"):
        raise ValueError(f"Unknown schedule unit {unit!r}")
    delta = relativedelta(months=step) if unit == "months" else relativedelta(weeks=step)

    rows: list[dict] = []
    seen: set[pd.Timestamp] = set()
    d = pd.Timestamp(start)
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
        d = d + delta
    return pd.DataFrame(rows, columns=cols)


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

    Uses the price-only ``score`` (``score / Σscore``), equal weight, or
    inverse volatility (``1/σ``). Never reads the scanner's mcap-based ``weight``
    column. Empty selection -> empty.
    """
    return scheme_weights(selected, weighting)


def _asof_prices(close: pd.DataFrame, tickers, when) -> pd.Series:
    """Per-ticker adjusted close, last valid on/before ``when`` (ffill)."""
    tk = list(tickers)
    if not tk:
        return pd.Series(dtype="float64")
    pos = _row_pos(close.index, when)
    if pos < 0:
        return pd.Series(np.nan, index=tk)
    vals = close[tk].to_numpy(dtype="float64")[: pos + 1]
    return pd.Series(_last_valid(vals, ~np.isnan(vals), pos), index=tk)


def _cash_growth(n_days: int, daily: float) -> np.ndarray:
    """Cumulative growth factors (1+r)^k for k = 1..n_days."""
    return (1.0 + daily) ** np.arange(1, n_days + 1)


def _hold_period_path(
    close: pd.DataFrame, weights: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp,
    cash: float = 0.0, cash_daily: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Buy-and-hold value path over ``(t0, t1]`` with shares FIXED at ``t0``.

    Shares are set from ``weights`` at the ``t0`` price (so Σ shares·p0 = Σw); an
    optional ``cash`` sleeve (``Σw + cash = 1``) compounds at ``cash_daily`` per
    trading day. The returned value series is based at ~1.0 at t0 and its last
    value is the exact period multiple. Prices are ffilled within the period so
    a name that stops trading freezes at its last mark. Also returns the drifted
    end-of-period weights as fractions of the whole book (for turnover).
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
    cash_end = 0.0
    if cash > 0 and len(days):
        growth = cash * _cash_growth(len(days), cash_daily)
        val = val + growth
        cash_end = float(growth[-1])
    p1 = _asof_prices(close, tk, t1)
    w_end = shares * p1
    total = float(w_end.sum()) + cash_end
    w_end = w_end / total if total > 0 else pd.Series(dtype="float64")
    return val, w_end


# --------------------------------------------------------------------------- #
# Market regime & synthetic benchmark
# --------------------------------------------------------------------------- #
def regime_states(
    series: pd.Series | None, dates, ma_days: int = config.REGIME_MA_DAYS
) -> dict[pd.Timestamp, bool | None]:
    """Risk-on flag per date: index close on/before ``d`` >= its ``ma_days`` SMA.

    Uses only observations on/before each date (no look-ahead). ``None`` where
    there is not yet ``ma_days`` of history (the caller treats that as risk-on).
    """
    out: dict[pd.Timestamp, bool | None] = {pd.Timestamp(d): None for d in dates}
    if series is None or len(series) == 0:
        return out
    s = series.dropna().sort_index()
    sma = s.rolling(int(ma_days), min_periods=int(ma_days)).mean()
    for d in out:
        pos = _row_pos(s.index, d)
        if pos >= 0 and not np.isnan(sma.iloc[pos]):
            out[d] = bool(s.iloc[pos] >= sma.iloc[pos])
    return out


def universe_equal_weight(close: pd.DataFrame) -> pd.Series:
    """Daily-rebalanced equal-weight index of every name in ``close`` (base 1.0).

    Each day's return is the average across names that traded on both days. It
    shares the strategy's universe — and therefore its survivorship bias — so
    strategy-minus-this isolates what the momentum *selection* adds.
    """
    if close is None or close.empty:
        return pd.Series(dtype="float64", name="close")
    rets = close.sort_index().pct_change(fill_method=None)
    daily = rets.mean(axis=1, skipna=True).fillna(0.0)
    level = (1.0 + daily).cumprod()
    level.name = "close"
    return level


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
# Membership changes (entries / exits) & transaction ledger
# --------------------------------------------------------------------------- #
def _symbols(h: pd.DataFrame | None) -> list[str]:
    """Ordered symbols of a holdings frame (falls back to the index/ticker)."""
    if h is None or h.empty:
        return []
    if "Symbol" in h.columns:
        return h["Symbol"].tolist()
    return [str(t) for t in h.index]


def compute_membership_changes(
    holdings: dict[pd.Timestamp, pd.DataFrame], rebalance_dates: list[pd.Timestamp]
) -> pd.DataFrame:
    """Per-rebalance churn: which scrips ENTER and EXIT the momentum portfolio.

    Compares each rebalance's selected names against the previous rebalance's.
    The first rebalance shows every name as an entry (the initial deployment);
    a rebalance that holds cash shows the whole prior book as exits. Returns a
    DataFrame with columns rebalance_date, n_held, n_entered, n_exited, entered,
    exited (the last two are comma-joined symbol strings, sorted).
    """
    rows: list[dict] = []
    prev: set[str] = set()
    for d in rebalance_dates:
        cur_list = _symbols(holdings.get(d))
        cur = set(cur_list)
        entered = sorted(cur - prev)
        exited = sorted(prev - cur)
        rows.append({
            "rebalance_date": d,
            "n_held": len(cur),
            "n_entered": len(entered),
            "n_exited": len(exited),
            "entered": ", ".join(entered),
            "exited": ", ".join(exited),
        })
        prev = cur
    return pd.DataFrame(
        rows,
        columns=["rebalance_date", "n_held", "n_entered", "n_exited", "entered", "exited"],
    )


def _classify_side(w_prev: float, w_new: float, eps: float = 1e-9) -> str:
    """Trade direction for one name given its prior (drifted) and target weight."""
    if w_prev <= eps and w_new > eps:
        return "BUY (new)"
    if w_new <= eps and w_prev > eps:
        return "SELL (exit)"
    if w_new - w_prev > eps:
        return "BUY"
    if w_prev - w_new > eps:
        return "SELL"
    return "HOLD"


def _ledger_rows(
    t0: pd.Timestamp,
    w_new: pd.Series,
    w_prev_drift: pd.Series,
    close: pd.DataFrame,
    e0: float,
    cost_bps: float,
    initial_capital: float,
    meta_map: pd.DataFrame | None,
    meta_now: pd.DataFrame | None,
) -> list[dict]:
    """Per-name trades that turn the drifted prior book into the new targets at ``t0``.

    Notional is denominated in ₹ against a starting capital of ``initial_capital``
    grown by the (unitless) portfolio multiple ``e0`` at this rebalance — so the
    ledger's rupee figures track the "Growth of ₹<capital>" chart. Both prior and
    target values use the *pre-cost* portfolio value ``base`` as the common base,
    so ``Σ traded_value = base·Σ|Δw| = 2·base·turnover`` and the summed per-trade
    cost equals ``base·cost_drag`` — consistent with the engine's turnover/cost.
    Prices are as-of adjusted closes (the series the return maths uses).
    """
    ix = w_new.index.union(w_prev_drift.index)
    if len(ix) == 0:
        return []
    prices = _asof_prices(close, ix, t0)
    base = float(e0) * float(initial_capital)
    rate = cost_bps / 1e4
    rows: list[dict] = []
    for tk in ix:
        wp = float(w_prev_drift.get(tk, 0.0))
        wn = float(w_new.get(tk, 0.0))
        dw = wn - wp
        traded_val = abs(dw) * base
        sym, company = str(tk), ""
        if meta_map is not None and tk in meta_map.index:
            sym = meta_map.at[tk, "Symbol"]
            company = meta_map.at[tk, "Company"]
        score = np.nan
        if meta_now is not None and "score" in meta_now.columns and tk in meta_now.index:
            score = float(meta_now.at[tk, "score"])
        rows.append({
            "rebalance_date": t0,
            "Symbol": sym,
            "Company": company,
            "YFTicker": tk,
            "side": _classify_side(wp, wn),
            "price": float(prices.get(tk, np.nan)),
            "prior_weight": wp,
            "target_weight": wn,
            "delta_weight": dw,
            "prior_value": wp * base,
            "target_value": wn * base,
            "traded_value": traded_val,
            "cost": rate * traded_val,
            "score": score,
        })
    # Traded names first (largest trade on top), then holds — most useful ordering.
    rows.sort(key=lambda r: (r["side"] == "HOLD", -r["traded_value"]))
    return rows


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


# Columns kept when a scan is memoised for reuse across backtests (a sweep).
_RANKED_COLS = ["Symbol", "Company", "Industry", "Segment", "score", "sigma"]


class _RankedScan:
    """Lightweight stand-in for a ScanResult: just the scored, eligible names."""

    def __init__(self, results: pd.DataFrame):
        self.results = results


def _compact_scan(scan) -> _RankedScan:
    res = getattr(scan, "results", None)
    if res is None or "score" not in res.columns:
        return scan  # e.g. a test stub exposing only .selected
    ranked = res[res["score"].notna()]
    return _RankedScan(ranked[[c for c in _RANKED_COLS if c in ranked.columns]].copy())


def _select_from_scan(scan, sp: ScanParams, prev: list | None, exit_rank: int | None) -> pd.DataFrame:
    """Selection at one rebalance: top-N (+ sector cap, + rank buffer) from the
    scan's full ranking. Falls back to ``scan.selected`` when only that exists."""
    res = getattr(scan, "results", None)
    if res is None or "score" not in getattr(res, "columns", []):
        return scan.selected
    ranked = res[res["score"].notna()]
    chosen = select_top(ranked, sp.top_n, sp.max_per_sector, keep=prev, exit_rank=exit_rank)
    return ranked.loc[chosen]


def run_backtest(
    universe_df: pd.DataFrame,
    price_data,
    params: BacktestParams,
    benchmark_series: pd.Series | None = None,
    progress=None,
    regime_series: pd.Series | None = None,
    scan_cache: dict | None = None,
) -> BacktestResult:
    """Run a rebalancing backtest.

    ``universe_df`` has Symbol/Company/Industry/Segment/YFTicker; ``price_data``
    is a :class:`momentum.data.PriceData` covering the full history (its
    ``close`` is adjusted). ``benchmark_series`` is an adjusted-close Series
    (e.g. from ``data.fetch_benchmark``); ``progress`` is ``callable(done, total)``.
    ``regime_series`` drives the regime filter (defaults to the benchmark).
    ``scan_cache`` memoises per-date scans across runs that share every
    ranking-relevant parameter (see :func:`run_sweep`).
    """
    close = price_data.close
    notes = _base_notes(params)
    if close is None or close.empty:
        return _empty_result(notes + ["No price history available."], params)

    cal = close.index.sort_values()
    sp = params.scan_params
    unit, step = freq_spec(params)

    # -- schedule ----------------------------------------------------------- #
    feas = _feasible_first(cal, sp, params.min_history_months)
    first = max(pd.Timestamp(params.start), feas)
    if first > pd.Timestamp(params.start):
        notes.append(f"First rebalance moved to {first.date()} (insufficient prior history before requested start).")
    schedule = generate_schedule(cal, first.date(), params.end, step, unit)
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

    # -- regime ------------------------------------------------------------- #
    regime: dict[pd.Timestamp, bool | None] = {}
    if params.regime_filter:
        src = regime_series if regime_series is not None else benchmark_series
        regime = regime_states(src, rb, params.regime_ma_days)
        if src is None or len(src) == 0:
            notes.append("Regime filter requested but no regime index data — filter not applied.")

    # -- PASS A: point-in-time selection (uses only prices <= entry) --------- #
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    meta_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    prev_names: list | None = None
    for i, d in enumerate(rb):
        scan = scan_cache.get(d) if scan_cache is not None else None
        if scan is None:
            sp_d = replace(sp, asof=d.date(), apply_turnover_ratio_filter=False)
            scan = run_scan(universe_df, price_data, sp_d)
            if scan_cache is not None:
                scan = _compact_scan(scan)
                scan_cache[d] = scan
        sel = _select_from_scan(scan, sp, prev_names, params.exit_rank)
        w = _target_weights(sel, params.weighting)
        if not w.empty:  # drop names untradable at entry, renormalise
            p0 = _asof_prices(close, w.index, d)
            good = p0[(p0 > 0) & p0.notna()].index
            if len(good):
                w = w.loc[good] / w.loc[good].sum()
            else:
                w = pd.Series(dtype="float64")
        if regime.get(d) is False and not w.empty:
            off = float(min(max(params.regime_exposure_off, 0.0), 1.0))
            w = w * off if off > 0 else pd.Series(dtype="float64")
        weights_by_date[d] = w
        meta_by_date[d] = sel
        prev_names = list(w.index) if not w.empty else None
        if progress:
            progress(i + 1, len(rb))

    # Ticker -> Symbol/Company map for the trade ledger (covers exited names too).
    meta_cols = [c for c in ("Symbol", "Company") if c in universe_df.columns]
    meta_map = (
        universe_df.drop_duplicates("YFTicker").set_index("YFTicker")[meta_cols]
        if "YFTicker" in universe_df.columns and len(meta_cols) == 2
        else None
    )

    # -- PASS B: chain daily buy-and-hold multiples -------------------------- #
    cash_daily = (1.0 + params.cash_yield_annual) ** (1.0 / params.trading_days) - 1.0
    E = 1.0
    w_prev = pd.Series(dtype="float64")
    eq: dict[pd.Timestamp, float] = {boundary[0]: 1.0}
    prows: list[dict] = []
    holdings: dict[pd.Timestamp, pd.DataFrame] = {}
    tx_rows: list[dict] = []
    contrib_rows: list[dict] = []

    for i in range(len(boundary) - 1):
        t0, t1 = boundary[i], boundary[i + 1]
        w = weights_by_date.get(t0, pd.Series(dtype="float64"))
        exposure = float(w.sum()) if not w.empty else 0.0

        one_way, cost = _turnover_and_cost(w, w_prev, params.cost_bps)
        E0 = E
        # Ledger uses w_prev (the drifted prior book) BEFORE it is overwritten below.
        tx_rows.extend(_ledger_rows(
            t0, w, w_prev, close, E0, params.cost_bps,
            params.initial_capital, meta_map, meta_by_date.get(t0),
        ))
        E *= (1.0 - cost)

        days = cal[(cal > t0) & (cal <= t1)]
        if w.empty or len(days) == 0:
            growth = _cash_growth(len(days), cash_daily) if len(days) else np.array([])
            for d, g in zip(days, growth):
                eq[d] = E * float(g)
            if len(days):
                E = E * float(growth[-1])
            w_prev = pd.Series(dtype="float64")
            sret = E / E0 - 1.0
            if w.empty:
                why = "regime risk-off" if regime.get(t0) is False else "empty selection"
                notes.append(f"{t0.date()}: {why} → held cash for the period.")
        else:
            E_start = E
            path, w_prev = _hold_period_path(close, w, t0, t1, cash=max(0.0, 1.0 - exposure),
                                             cash_daily=cash_daily)
            for d, m in path.items():
                eq[d] = E * float(m)
            E = E * float(path.iloc[-1])
            sret = E / E0 - 1.0
            p0 = _asof_prices(close, w.index, t0)
            p1 = _asof_prices(close, w.index, t1)
            r = p1 / p0 - 1.0
            sym = meta_map["Symbol"] if meta_map is not None else None
            for tk in w.index:
                contrib = E_start * float(w[tk]) * float(r[tk])
                contrib_rows.append({
                    "period_start": t0, "period_end": t1, "YFTicker": tk,
                    "Symbol": sym.get(tk, tk) if sym is not None else tk,
                    "weight": float(w[tk]), "ret": float(r[tk]),
                    "contribution": contrib, "contribution_inr": contrib * params.initial_capital,
                })

        prows.append({
            "rb_date": t0, "period_start": t0, "period_end": t1,
            "strat_ret": sret, "bench_ret": np.nan,
            "turnover": one_way, "cost_drag": cost,
            "n_holdings": int(len(w)), "empty": bool(w.empty),
            "exposure": exposure, "risk_on": regime.get(t0) if params.regime_filter else None,
        })
        if not w.empty:
            meta = meta_by_date[t0]
            cols = [c for c in ["Symbol", "Company", "Industry", "Segment", "score"] if c in meta.columns]
            holdings[t0] = meta.loc[w.index, cols].assign(weight=w)

    equity = pd.Series(eq).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    equity.name = "portfolio"

    period_returns = pd.DataFrame(prows).set_index("rb_date")
    transactions = pd.DataFrame(tx_rows)
    contributions = pd.DataFrame(contrib_rows)
    # Membership churn over the periods actually simulated (the t0 boundaries).
    membership_changes = compute_membership_changes(holdings, list(boundary[:-1]))

    if params.regime_filter and regime:
        sim = period_returns["risk_on"]
        n_off = int((sim == False).sum())  # noqa: E712 — tri-state (None = no data yet)
        notes.append(
            f"Regime filter ({params.regime_ma_days}-day SMA): risk-off at {n_off} of {len(sim)} "
            f"rebalances (exposure {params.regime_exposure_off:.0%} when risk-off)."
        )
        if sim.isna().any():
            notes.append(f"Regime index lacked {params.regime_ma_days} days of history at "
                         f"{int(sim.isna().sum())} rebalance(s) — treated as risk-on.")
    if params.exit_rank:
        notes.append(f"Rank buffer: holdings kept while ranked ≤ {params.exit_rank}.")

    # -- benchmark alignment ------------------------------------------------- #
    bench_curve = _align_benchmark(benchmark_series, equity.index, boundary[0])
    if benchmark_series is not None and (benchmark_series is None or len(benchmark_series) == 0 or bench_curve is None):
        notes.append("Benchmark data unavailable for this window; relative metrics omitted.")

    bench_period = None
    if bench_curve is not None:
        bvals = bench_curve.to_numpy()
        bidx = bench_curve.index
        bench_period_vals = {}
        for rbd, t0, t1 in zip(period_returns.index, period_returns["period_start"], period_returns["period_end"]):
            i0, i1 = _row_pos(bidx, t0), _row_pos(bidx, t1)
            if i0 >= 0 and i1 >= 0 and np.isfinite(bvals[i0]) and np.isfinite(bvals[i1]):
                bench_period_vals[rbd] = float(bvals[i1] / bvals[i0] - 1.0)
        bench_period = pd.Series(bench_period_vals, dtype="float64")
        period_returns["bench_ret"] = period_returns.index.map(bench_period).astype(float)

    # -- analytics ----------------------------------------------------------- #
    drawdown = (equity / equity.cummax() - 1.0)
    drawdown.name = "portfolio"
    rpy = rebalances_per_year(unit, step, params.trading_days)

    metrics = compute_metrics(
        equity, period_returns["strat_ret"], period_returns["turnover"],
        period_returns["cost_drag"], params.trading_days, params.rf_annual, rpy,
    )
    if params.regime_filter:
        metrics["time_in_market"] = float(period_returns["exposure"].mean())
    benchmark_metrics = {}
    if bench_curve is not None:
        benchmark_metrics = compute_metrics(
            bench_curve,
            bench_period if bench_period is not None else None,
            None, None, params.trading_days, params.rf_annual, rpy,
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
        membership_changes=membership_changes,
        transactions=transactions,
        contributions=contributions,
        notes=notes,
        params=params,
    )


# --------------------------------------------------------------------------- #
# Strategy comparison (parameter sweep)
# --------------------------------------------------------------------------- #
SWEEP_METRICS = ["cagr", "ann_vol", "sharpe", "sortino", "max_dd", "calmar", "annual_turnover",
                 "total_return", "hit_rate"]


def run_sweep(
    universe_df: pd.DataFrame,
    price_data,
    base: BacktestParams,
    top_ns: list[int],
    freqs: list[str],
    benchmark_series: pd.Series | None = None,
    regime_series: pd.Series | None = None,
    progress=None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Backtest every (rebalance frequency × Top-N) combination of ``base``.

    Every other knob is held fixed. The momentum *ranking* at a date does not
    depend on Top-N or cadence, so scans are memoised once per trading date and
    shared across all combinations — a 3×3 grid costs little more than its
    densest single cadence. A rank buffer is scaled with Top-N (same ratio).
    Returns (one metrics row per combination, label -> equity curve).
    """
    cache: dict = {}
    combos = [(f, int(n)) for f in freqs for n in top_ns]
    rows: list[dict] = []
    curves: dict[str, pd.Series] = {}
    base_n = max(1, int(base.scan_params.top_n))
    for k, (f, n) in enumerate(combos):
        exit_rank = None
        if base.exit_rank:
            exit_rank = max(n, int(round(n * base.exit_rank / base_n)))
        p = replace(base, rebalance_freq=f, exit_rank=exit_rank,
                    scan_params=replace(base.scan_params, top_n=n))
        res = run_backtest(universe_df, price_data, p, benchmark_series=benchmark_series,
                           regime_series=regime_series, scan_cache=cache)
        label = f"{f} · Top {n}"
        row = {"label": label, "frequency": f, "top_n": n}
        row.update({m: res.metrics.get(m, np.nan) for m in SWEEP_METRICS})
        row["alpha_annual"] = res.relative_metrics.get("alpha_annual", np.nan)
        row["info_ratio"] = res.relative_metrics.get("info_ratio", np.nan)
        rows.append(row)
        if not res.equity_curve.empty:
            curves[label] = res.equity_curve
        if progress:
            progress(k + 1, len(combos))
    return pd.DataFrame(rows), curves
