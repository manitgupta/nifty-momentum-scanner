"""Unit + integration checks for the backtesting engine.

Run with:  ./venv/bin/python tests/test_backtest.py
(dependency-free of pytest, matching tests/test_scanner.py.)
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from momentum import backtest
from momentum.data import PriceData
from momentum.scanner import ScanParams

PASS, FAIL = "PASS", "FAIL"
_failures = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _failures
    print(f"[{PASS if cond else FAIL}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        _failures += 1


# --------------------------------------------------------------------------- #
# Synthetic data builders
# --------------------------------------------------------------------------- #
def _synthetic_prices(years: int = 4, drifts=None):
    """Deterministic geometric price paths (no randomness) over `years` of bdays."""
    drifts = drifts or {"HIGH.NS": 0.0009, "MID.NS": 0.0004, "LOW.NS": 0.0001}
    end = dt.date(2026, 6, 30)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=int(years * 252))
    close = pd.DataFrame(
        {tk: 100.0 * np.exp(np.cumsum(np.full(len(idx), d))) for tk, d in drifts.items()},
        index=idx,
    )
    volume = pd.DataFrame(1_000_000, index=idx, columns=close.columns)
    market = pd.DataFrame(
        {"market_cap": [5e11] * len(close.columns), "shares": [1e9] * len(close.columns),
         "last_price": [1.0] * len(close.columns)},
        index=close.columns,
    )
    pdata = PriceData(close=close, raw_close=close.copy(), volume=volume, market=market, failed=[])
    universe = pd.DataFrame(
        {"Symbol": [c.replace(".NS", "") for c in close.columns],
         "Company": [c + " Co" for c in close.columns],
         "Industry": ["X"] * len(close.columns),
         "Segment": ["Largecap"] * len(close.columns),
         "YFTicker": list(close.columns)}
    )
    return pdata, universe, idx


def _params(pdata, weighting="score", cost_bps=0.0, freq=3, start=None, end=None, top_n=3):
    idx = pdata.close.index
    start = start or (idx[0] + pd.Timedelta(days=420)).date()
    end = end or idx[-1].date()
    sp = ScanParams(asof=end, top_n=top_n, apply_listing_filter=False,
                    apply_liquidity_filter=False, apply_turnover_ratio_filter=False)
    return backtest.BacktestParams(scan_params=sp, start=start, end=end,
                                   rebalance_months=freq, weighting=weighting, cost_bps=cost_bps)


# --------------------------------------------------------------------------- #
def test_generate_schedule():
    idx = pd.bdate_range("2020-01-01", "2023-12-29")
    sched = backtest.generate_schedule(idx, dt.date(2021, 1, 1), dt.date(2023, 12, 31), 6)
    used = sched[sched["used"]]
    check("schedule: ~6 semiannual dates over 3y", 5 <= len(used) <= 7, f"n={len(used)}")
    check("schedule: trading_date <= scheduled_date",
          bool((used["trading_date"] <= used["scheduled_date"]).all()))
    check("schedule: all trading_dates are real trading days",
          bool(used["trading_date"].isin(idx).all()))
    check("schedule: no duplicate trading days", used["trading_date"].is_unique)

    # A Sunday scheduled date snaps back to the prior Friday.
    sun = dt.date(2021, 1, 3)  # Sunday
    one = backtest.generate_schedule(idx, sun, sun, 6)
    td = one.iloc[0]["trading_date"]
    check("schedule: Sunday snaps to prior trading day", td.weekday() < 5 and td <= pd.Timestamp(sun))


# --------------------------------------------------------------------------- #
def test_target_weights():
    sel = pd.DataFrame({"score": [3.0, 1.0, 1.0]}, index=["A", "B", "C"])
    we = backtest._target_weights(sel, "equal")
    check("weights equal: all 1/N", all(math.isclose(v, 1 / 3, rel_tol=1e-9) for v in we))
    check("weights equal: sum 1", math.isclose(we.sum(), 1.0, rel_tol=1e-12))
    ws = backtest._target_weights(sel, "score")
    check("weights score: proportional", math.isclose(ws["A"], 0.6) and math.isclose(ws["B"], 0.2))
    check("weights score: sum 1", math.isclose(ws.sum(), 1.0, rel_tol=1e-12))
    check("weights empty selection -> empty", backtest._target_weights(sel.iloc[:0], "score").empty)
    neg = pd.DataFrame({"score": [-1.0, -2.0]}, index=["A", "B"])
    check("weights score<=0 -> equal fallback",
          math.isclose(backtest._target_weights(neg, "score")["A"], 0.5, rel_tol=1e-9))


# --------------------------------------------------------------------------- #
def test_turnover_and_cost():
    w0 = pd.Series({"A": 0.5, "B": 0.5})
    empty = pd.Series(dtype="float64")
    ow, cost = backtest._turnover_and_cost(w0, empty, 10.0)
    check("turnover: first deploy one-way = 0.5", math.isclose(ow, 0.5, rel_tol=1e-12))
    check("turnover: first deploy cost = bps*1", math.isclose(cost, 10.0 / 1e4, rel_tol=1e-12))
    ow2, _ = backtest._turnover_and_cost(w0, w0, 10.0)
    check("turnover: identical weights -> 0", math.isclose(ow2, 0.0, abs_tol=1e-12))
    w1 = pd.Series({"A": 0.5, "C": 0.5})  # swap B->C: Σ|Δw| = 1.0 -> one-way 0.5
    ow3, _ = backtest._turnover_and_cost(w1, w0, 0.0)
    check("turnover: half-portfolio swap -> 0.5", math.isclose(ow3, 0.5, rel_tol=1e-12))


# --------------------------------------------------------------------------- #
def test_hold_period_identity():
    idx = pd.bdate_range("2022-01-03", periods=60)
    close = pd.DataFrame({"A.NS": np.linspace(100, 130, 60), "B.NS": np.linspace(50, 45, 60)}, index=idx)
    w = pd.Series({"A.NS": 0.7, "B.NS": 0.3})
    t0, t1 = idx[0], idx[-1]
    path, w_end = backtest._hold_period_path(close, w, t0, t1)
    p0 = close.loc[t0]
    p1 = close.loc[t1]
    expected = float((w * (p1 / p0)).sum())
    check("hold: last value == Σ w·P1/P0 (buy-and-hold identity)",
          math.isclose(path.iloc[-1], expected, rel_tol=1e-9), f"{path.iloc[-1]} vs {expected}")
    check("hold: path starts near 1.0", math.isclose(path.iloc[0], float((w * close.iloc[1] / p0).sum()), rel_tol=1e-9))
    check("hold: drifted end weights sum to 1", math.isclose(w_end.sum(), 1.0, rel_tol=1e-9))
    check("hold: winner's drifted weight rises", w_end["A.NS"] > w["A.NS"])


# --------------------------------------------------------------------------- #
def _reordering_prices(years: int = 6):
    """6-ticker frame where a 'sleeper' only rallies after a mid-timeline cutoff,
    so the momentum ranking (and top-3 membership) reorders across the cutoff."""
    end = dt.date(2026, 6, 30)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=int(years * 252))
    cut = idx[int(len(idx) * 0.6)]

    def path(pre, post):
        d = np.where(idx.values <= cut.to_datetime64(), pre, post)
        return 100.0 * np.exp(np.cumsum(d))

    close = pd.DataFrame({
        "L1.NS": path(0.0011, 0.0011), "L2.NS": path(0.0009, 0.0009),
        "L3.NS": path(0.0007, 0.0007), "M1.NS": path(0.0001, 0.0001),
        "M2.NS": path(0.0001, 0.0001), "SLEEP.NS": path(0.00003, 0.0035),
    }, index=idx)
    vol = pd.DataFrame(1_000_000, index=idx, columns=close.columns)
    market = pd.DataFrame(
        {"market_cap": [5e11] * 6, "shares": [1e9] * 6, "last_price": [1.0] * 6},
        index=close.columns,
    )
    pdata = PriceData(close=close, raw_close=close.copy(), volume=vol, market=market, failed=[])
    uni = pd.DataFrame({
        "Symbol": [c[:-3] for c in close.columns], "Company": [c + " Co" for c in close.columns],
        "Industry": ["X"] * 6, "Segment": ["Largecap"] * 6, "YFTicker": list(close.columns),
    })
    return pdata, uni, idx, cut


def _slice_pdata(pdata, cut):
    c = pdata.close.loc[:cut]
    return PriceData(close=c, raw_close=pdata.raw_close.loc[:cut],
                     volume=pdata.volume.loc[:cut], market=pdata.market, failed=[])


def test_no_lookahead_truncation():
    """Selections/returns at a rebalance must depend ONLY on data up to that date.

    Compares a full-history run against one on data truncated at a cutoff. For any
    rebalance on/before the cutoff, holdings and returns must be bit-identical —
    a real look-ahead leak would make the full run 'see' post-cutoff prices and
    diverge. Uses 6 tickers / top_n=3 with a reordering 'sleeper' so membership
    (not just weights) is exercised.
    """
    pdata, uni, idx, cut = _reordering_prices(years=6)
    sp = ScanParams(asof=idx[-1].date(), top_n=3, apply_listing_filter=False,
                    apply_liquidity_filter=False, apply_turnover_ratio_filter=False)
    start = (idx[0] + pd.Timedelta(days=420)).date()
    p_full = backtest.BacktestParams(scan_params=sp, start=start, end=idx[-1].date(),
                                     rebalance_months=3, weighting="score")
    p_trunc = replace(p_full, end=cut.date())

    r_full = backtest.run_backtest(uni, pdata, p_full)
    r_trunc = backtest.run_backtest(uni, _slice_pdata(pdata, cut), p_trunc)

    shared = [d for d in r_trunc.holdings if d in r_full.holdings]
    check("no-lookahead: shared pre-cutoff rebalances", len(shared) >= 3, f"shared={len(shared)}")
    holdings_match = all(
        list(r_full.holdings[d].index) == list(r_trunc.holdings[d].index)
        and np.allclose(r_full.holdings[d]["weight"].values,
                        r_trunc.holdings[d]["weight"].values, atol=1e-12)
        for d in shared
    )
    check("no-lookahead: holdings identical (truncated vs full data)", holdings_match)
    # The truncated run's LAST period ends at the cutoff (partial), so exclude it;
    # every interior period shares the same (t0, t1] boundary as the full run.
    trunc_interior = list(r_trunc.period_returns.index[:-1])
    compared = [d for d in trunc_interior if d in r_full.period_returns.index]
    rets_match = all(
        math.isclose(r_full.period_returns.loc[d, "strat_ret"],
                     r_trunc.period_returns.loc[d, "strat_ret"], rel_tol=1e-9)
        for d in compared
    )
    check("no-lookahead: pre-cutoff period returns identical", rets_match and len(compared) >= 2,
          f"compared={len(compared)}")

    # The sleeper must NOT appear before it rallies — proves ranking is as-of, not future.
    pre_cut = [d for d in r_full.holdings if d < cut]
    sleeper_early = any("SLEEP.NS" in r_full.holdings[d].index for d in pre_cut)
    check("no-lookahead: sleeper excluded before its rally", not sleeper_early,
          f"pre_cut_dates={len(pre_cut)}")


# --------------------------------------------------------------------------- #
def test_hold_period_boundary_nan_freeze():
    """A holding halted across the rebalance boundary freezes at its entry mark,
    not zero (regression for the boundary-NaN equity-crash bug)."""
    idx = pd.bdate_range("2022-01-03", periods=40)
    close = pd.DataFrame({"A.NS": 100.0, "B.NS": 50.0}, index=idx).astype(float)
    t0, t1 = idx[10], idx[-1]
    # B stops trading across t0 (NaN from t0 through t0+3), resumes at its old price.
    halt = (idx >= t0) & (idx <= idx[13])
    close.loc[halt, "B.NS"] = np.nan
    w = pd.Series({"A.NS": 0.5, "B.NS": 0.5})
    path, w_end = backtest._hold_period_path(close, w, t0, t1)
    check("boundary-nan: no spurious crash (path stays ~1.0)",
          float(path.min()) > 0.99, f"min={float(path.min()):.4f}")
    check("boundary-nan: end value ~1.0", math.isclose(float(path.iloc[-1]), 1.0, rel_tol=1e-6))
    check("boundary-nan: drifted weights still sum to 1", math.isclose(w_end.sum(), 1.0, rel_tol=1e-9))


# --------------------------------------------------------------------------- #
def test_compounding_continuity():
    pdata, uni, idx = _synthetic_prices(years=4)
    res = backtest.run_backtest(uni, pdata, _params(pdata, freq=3))
    eq = res.equity_curve
    check("compound: equity index monotonic increasing", eq.index.is_monotonic_increasing)
    check("compound: no duplicate dates", eq.index.is_unique)
    check("compound: base 1.0", math.isclose(eq.iloc[0], 1.0, rel_tol=1e-12))
    # Per-period: equity ratio across the period == recorded strat_ret (cost=0 here).
    ok = True
    for d, row in res.period_returns.iterrows():
        t0, t1 = row["period_start"], row["period_end"]
        if row["empty"]:
            continue
        ratio = eq.loc[t1] / eq.loc[eq.index[eq.index <= t0][-1]] - 1.0
        if not math.isclose(ratio, row["strat_ret"], rel_tol=1e-7, abs_tol=1e-9):
            ok = False
            break
    check("compound: period ratio == strat_ret", ok)


# --------------------------------------------------------------------------- #
def test_cost_monotonicity():
    pdata, uni, idx = _synthetic_prices(years=4)
    free = backtest.run_backtest(uni, pdata, _params(pdata, freq=3, cost_bps=0.0))
    costly = backtest.run_backtest(uni, pdata, _params(pdata, freq=3, cost_bps=50.0))
    check("cost: costly run ends lower than free run",
          costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1])
    check("cost: annual cost drag > 0", costly.metrics.get("cost_drag_annual", 0) > 0)
    check("cost: free run has ~0 cost drag",
          math.isclose(free.metrics.get("cost_drag_annual", 0.0), 0.0, abs_tol=1e-12))


# --------------------------------------------------------------------------- #
def test_analytics_closed_form():
    # Constant daily-return curve -> known CAGR / vol / drawdown.
    idx = pd.bdate_range("2020-01-01", periods=252 * 3)
    daily = 0.0005
    eq = pd.Series(np.exp(np.cumsum(np.full(len(idx), daily))), index=idx)
    m = backtest.compute_metrics(eq, trading_days=252)
    check("analytics: ann_vol ~ 0 for constant returns", abs(m["ann_vol"]) < 1e-6, f"vol={m['ann_vol']}")
    check("analytics: max_dd == 0 for monotone-up curve", math.isclose(m["max_dd"], 0.0, abs_tol=1e-9))
    exp_cagr = (eq.iloc[-1] / eq.iloc[0]) ** (365.25 / (idx[-1] - idx[0]).days) - 1
    check("analytics: CAGR matches closed form", math.isclose(m["cagr"], exp_cagr, rel_tol=1e-6))

    # V-shaped curve -> drawdown and recovery dates.
    v = pd.Series([1.0, 1.2, 0.6, 0.8, 1.25], index=pd.bdate_range("2021-01-01", periods=5))
    mdd, peak, trough, recov = backtest._max_drawdown(v)
    check("analytics: V max_dd == -0.5", math.isclose(mdd, 0.6 / 1.2 - 1.0, rel_tol=1e-9))
    check("analytics: V peak at the 1.2 point", peak == v.index[1])
    check("analytics: V trough at the 0.6 point", trough == v.index[2])
    check("analytics: V recovers at the 1.25 point", recov == v.index[4])


# --------------------------------------------------------------------------- #
def test_benchmark_alignment():
    pdata, uni, idx = _synthetic_prices(years=4)
    bench = pd.Series(np.linspace(1000, 1600, len(idx)), index=idx)
    res = backtest.run_backtest(uni, pdata, _params(pdata, freq=3), benchmark_series=bench)
    check("benchmark: curve shares strategy index", res.benchmark_curve.index.equals(res.equity_curve.index))
    check("benchmark: base ~1.0 at first rebalance",
          math.isclose(res.benchmark_curve.dropna().iloc[0], 1.0, rel_tol=1e-6))
    check("benchmark: relative metrics populated", np.isfinite(res.relative_metrics["beta"]))
    check("benchmark: bench_ret column filled", res.period_returns["bench_ret"].notna().any())

    # Benchmark with a gap on a rebalance day still aligns via asof-fill.
    gapped = bench.copy()
    rb0 = res.rebalance_dates[1]
    gapped.loc[rb0] = np.nan
    res2 = backtest.run_backtest(uni, pdata, _params(pdata, freq=3), benchmark_series=gapped)
    check("benchmark: gap on rebalance day tolerated", res2.benchmark_curve is not None and
          res2.benchmark_curve.notna().sum() > 0)


# --------------------------------------------------------------------------- #
def test_edge_cases():
    pdata, uni, idx = _synthetic_prices(years=4)

    # Benchmark absent -> None curve, NaN relative metrics, finite strategy metrics.
    res = backtest.run_backtest(uni, pdata, _params(pdata, freq=3), benchmark_series=None)
    check("edge: no benchmark -> None curve", res.benchmark_curve is None)
    check("edge: no benchmark -> NaN relative", math.isnan(res.relative_metrics["beta"]))
    check("edge: strategy metrics still finite", np.isfinite(res.metrics["cagr"]))

    # Empty benchmark Series -> treated as unavailable.
    res_e = backtest.run_backtest(uni, pdata, _params(pdata, freq=3), benchmark_series=pd.Series(dtype=float))
    check("edge: empty benchmark -> None curve", res_e.benchmark_curve is None)

    # Window entirely before the feasibility cutoff -> no usable rebalance ->
    # empty result with an explanatory note, no crash.
    short = _params(pdata, freq=12, start=idx[50].date(), end=idx[100].date())
    res_s = backtest.run_backtest(uni, pdata, short)
    check("edge: pre-feasibility window -> empty equity", res_s.equity_curve.empty)
    check("edge: pre-feasibility window -> explanatory note",
          any("too short" in n.lower() for n in res_s.notes))

    # end on a weekend -> clamped to prior trading day (no crash, curve ends <= end).
    sat = idx[-1] + pd.Timedelta(days=1)
    res_w = backtest.run_backtest(uni, pdata, _params(pdata, freq=3, end=sat.date()))
    check("edge: weekend end clamps to trading day", res_w.equity_curve.index[-1] <= sat)


# --------------------------------------------------------------------------- #
def test_cash_period_via_stub(monkeypatch_run_scan=True):
    """Force every selection empty -> whole backtest holds cash (flat equity)."""
    pdata, uni, idx = _synthetic_prices(years=4)

    class _EmptyScan:
        selected = pd.DataFrame(columns=["Symbol", "Company", "Industry", "Segment", "score", "weight"])

    original = backtest.run_scan
    backtest.run_scan = lambda *a, **k: _EmptyScan()
    try:
        res = backtest.run_backtest(uni, pdata, _params(pdata, freq=6, cost_bps=10.0))
    finally:
        backtest.run_scan = original

    check("cash: equity flat at 1.0 when always empty",
          bool(np.allclose(res.equity_curve.values, 1.0, atol=1e-9)))
    check("cash: all periods flagged empty", bool(res.period_returns["empty"].all()))
    check("cash: no holdings recorded", len(res.holdings) == 0)


# --------------------------------------------------------------------------- #
def test_risk_ratios_closed_form():
    """Sharpe / Sortino / Calmar match hand-computed values on a known curve."""
    idx = pd.bdate_range("2020-01-01", periods=252 * 4)
    t = np.arange(len(idx))
    r = 0.0005 + 0.008 * np.sin(t / 3.0)          # deterministic, with down days
    eq = pd.Series(np.cumprod(1.0 + r), index=idx)
    m = backtest.compute_metrics(eq, trading_days=252, rf_annual=0.0)

    rr = eq.pct_change().dropna()
    exp_vol = float(rr.std(ddof=1) * np.sqrt(252))
    exp_sharpe = float(rr.mean() * 252 / exp_vol)
    short = rr.clip(upper=0.0)
    exp_dd = float(np.sqrt((short ** 2).mean()) * np.sqrt(252))
    exp_sortino = float(rr.mean() * 252 / exp_dd)

    check("risk: ann_vol closed form", math.isclose(m["ann_vol"], exp_vol, rel_tol=1e-9))
    check("risk: sharpe closed form", math.isclose(m["sharpe"], exp_sharpe, rel_tol=1e-9),
          f"{m['sharpe']} vs {exp_sharpe}")
    check("risk: sortino closed form (target downside dev)",
          math.isclose(m["sortino"], exp_sortino, rel_tol=1e-9), f"{m['sortino']} vs {exp_sortino}")
    check("risk: calmar == cagr/|max_dd|",
          math.isclose(m["calmar"], m["cagr"] / abs(m["max_dd"]), rel_tol=1e-9))

    # Sortino is defined with a single down day (old ddof>=2 guard wrongly returned NaN).
    ex = pd.Series([0.03, 0.02, 0.02, -0.04])
    eq2 = pd.Series(np.cumprod(1.0 + ex.values), index=pd.bdate_range("2021-01-01", periods=4))
    m2 = backtest.compute_metrics(eq2, trading_days=252, rf_annual=0.0)
    check("risk: sortino finite with one down day", np.isfinite(m2["sortino"]))

    # Higher rf lowers Sharpe.
    m_rf = backtest.compute_metrics(eq, trading_days=252, rf_annual=0.10)
    check("risk: rf>0 lowers sharpe", m_rf["sharpe"] < m["sharpe"])


# --------------------------------------------------------------------------- #
def test_relative_metrics_closed_form():
    """beta/alpha/capture/corr match closed forms when strat = k · benchmark."""
    idx = pd.bdate_range("2019-01-01", periods=252 * 3)
    t = np.arange(len(idx))
    rb = 0.0004 + 0.01 * np.sin(t / 5.0)
    k = 1.5
    bench = pd.Series(np.cumprod(1.0 + rb), index=idx)
    strat = pd.Series(np.cumprod(1.0 + k * rb), index=idx)

    # Period returns: strat = k · bench per period (3 up, 3 down).
    bench_period = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04, -0.03],
                             index=pd.to_datetime([f"2019-{m:02d}-15" for m in range(3, 9)]))
    strat_period = k * bench_period

    rel = backtest.compute_relative_metrics(strat, bench, strat_period, bench_period, 252)
    check("relative: beta == k", math.isclose(rel["beta"], k, rel_tol=1e-6), str(rel["beta"]))
    check("relative: corr == 1", math.isclose(rel["corr"], 1.0, rel_tol=1e-9))
    check("relative: up_capture == k", math.isclose(rel["up_capture"], k, rel_tol=1e-9))
    check("relative: down_capture == k", math.isclose(rel["down_capture"], k, rel_tol=1e-9))
    # strat beats bench exactly on up periods (3 of 6).
    check("relative: hit_rate_vs_bench == 0.5", math.isclose(rel["hit_rate_vs_bench"], 0.5, rel_tol=1e-9))

    # strat == benchmark -> beta 1, alpha ~0, TE ~0, corr 1.
    same = backtest.compute_relative_metrics(bench, bench, bench_period, bench_period, 252)
    check("relative: identical -> beta 1", math.isclose(same["beta"], 1.0, rel_tol=1e-9))
    check("relative: identical -> alpha ~0", abs(same["alpha_annual"]) < 1e-9, str(same["alpha_annual"]))
    check("relative: identical -> TE ~0", abs(same["tracking_error"]) < 1e-9)


# --------------------------------------------------------------------------- #
def test_cost_continuity_and_cash_reinvest():
    """Cost applied exactly once per boundary; a cash period resets weights so the
    next period pays full-deploy turnover."""
    pdata, uni, idx = _synthetic_prices(years=5)

    # (a) period-ratio identity holds even with costs (single cost application).
    res = backtest.run_backtest(uni, pdata, _params(pdata, freq=3, cost_bps=50.0))
    eq = res.equity_curve
    ok = True
    for d, row in res.period_returns.iterrows():
        if row["empty"]:
            continue
        t0i = eq.index[eq.index <= row["period_start"]][-1]
        ratio = eq.loc[row["period_end"]] / eq.loc[t0i] - 1.0
        if not math.isclose(ratio, row["strat_ret"], rel_tol=1e-7, abs_tol=1e-9):
            ok = False
            break
    check("cost-continuity: net period ratio == strat_ret with costs", ok)

    # (b) force ONE interior rebalance to cash; the next real period must re-deploy.
    normal = backtest.run_backtest(uni, pdata, _params(pdata, freq=6, cost_bps=25.0))
    dates = normal.rebalance_dates
    if len(dates) >= 3:
        target = dates[1]  # an interior rebalance

        class _EmptyScan:
            selected = pd.DataFrame(columns=["Symbol", "Company", "Industry", "Segment", "score", "weight"])

        original = backtest.run_scan
        backtest.run_scan = lambda u, p, params: (_EmptyScan() if params.asof == target.date()
                                                  else original(u, p, params))
        try:
            res2 = backtest.run_backtest(uni, pdata, _params(pdata, freq=6, cost_bps=25.0))
        finally:
            backtest.run_scan = original

        pr = res2.period_returns
        cash_row = pr.loc[target] if target in pr.index else None
        check("cash-reinvest: target period held cash",
              cash_row is not None and bool(cash_row["empty"]))
        # The next period after the cash one re-deploys from cash -> one-way turnover ~0.5.
        after = pr.loc[pr.index > target]
        nxt = after.iloc[0] if len(after) else None
        check("cash-reinvest: next period re-deploys (turnover ~0.5)",
              nxt is not None and math.isclose(float(nxt["turnover"]), 0.5, rel_tol=1e-6),
              f"turnover={None if nxt is None else float(nxt['turnover'])}")
    else:
        check("cash-reinvest: enough rebalances to test", False, "need >=3 rebalances")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_generate_schedule()
    test_target_weights()
    test_turnover_and_cost()
    test_hold_period_identity()
    test_hold_period_boundary_nan_freeze()
    test_no_lookahead_truncation()
    test_compounding_continuity()
    test_cost_monotonicity()
    test_cost_continuity_and_cash_reinvest()
    test_analytics_closed_form()
    test_risk_ratios_closed_form()
    test_relative_metrics_closed_form()
    test_benchmark_alignment()
    test_edge_cases()
    test_cash_period_via_stub()
    print("-" * 50)
    if _failures:
        print(f"{_failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed.")
