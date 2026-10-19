"""Report analytics computed from a finished backtest (pure functions, no I/O).

Everything here takes daily equity curves / result frames from
:mod:`momentum.backtest` and returns tidy pandas objects for the UI:

    calendar_returns     - year × month return grid (+ full-year column)
    annual_returns       - strategy vs benchmark calendar-year returns
    rolling_returns      - trailing N-year annualised return at every date
    rolling_summary      - min / median / max / % positive / % beating benchmark
    drawdown_episodes    - the worst peak → trough → recovery episodes
    simulate_sip / xirr  - monthly SIP (optionally stepped-up) and its XIRR
    sector_exposure      - industry weights at every rebalance
    summarize_contributions - per-stock P&L attribution
    excel_report         - every table in one .xlsx workbook
"""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _clean(curve: pd.Series | None) -> pd.Series:
    if curve is None:
        return pd.Series(dtype="float64")
    return curve.dropna().sort_index()


# --------------------------------------------------------------------------- #
# Calendar returns
# --------------------------------------------------------------------------- #
def _period_returns(curve: pd.Series, freq: str) -> pd.Series:
    """Return per period-end (``"ME"`` month / ``"YE"`` year), first period based
    on the curve's first value so partial first periods are still measured."""
    c = _clean(curve)
    if len(c) < 2:
        return pd.Series(dtype="float64")
    ends = c.resample(freq).last().dropna()
    prev = ends.shift(1)
    prev.iloc[0] = c.iloc[0]
    return ends / prev - 1.0


def calendar_returns(curve: pd.Series) -> pd.DataFrame:
    """Year × month grid of returns (decimals) with a trailing ``Year`` column."""
    m = _period_returns(curve, "ME")
    if m.empty:
        return pd.DataFrame(columns=MONTHS + ["Year"])
    grid = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    table = grid.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))
    table.columns = MONTHS
    y = _period_returns(curve, "YE")
    table["Year"] = pd.Series(y.values, index=y.index.year).reindex(table.index)
    table.index.name = "Year"
    return table


def annual_returns(strategy: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Calendar-year returns of the strategy (and benchmark + excess, if given)."""
    s = _period_returns(strategy, "YE")
    out = pd.DataFrame({"Strategy": pd.Series(s.values, index=s.index.year)})
    if benchmark is not None and len(_clean(benchmark)) >= 2:
        b = _period_returns(benchmark, "YE")
        out["Benchmark"] = pd.Series(b.values, index=b.index.year)
        out["Excess"] = out["Strategy"] - out["Benchmark"]
    out.index.name = "Year"
    return out


# --------------------------------------------------------------------------- #
# Rolling returns
# --------------------------------------------------------------------------- #
def rolling_returns(curve: pd.Series, years: float) -> pd.Series:
    """Annualised return over the trailing ``years`` at every date (CAGR form).

    The start value for date ``t`` is the curve on the last date on/before
    ``t - years``; dates without a full window are omitted.
    """
    c = _clean(curve)
    if len(c) < 2:
        return pd.Series(dtype="float64")
    days = int(round(years * 365.25))
    starts = c.index - pd.Timedelta(days=days)
    pos = c.index.searchsorted(starts, side="right") - 1
    ok = (pos >= 0) & (starts >= c.index[0])
    if not ok.any():
        return pd.Series(dtype="float64")
    v0 = c.values[pos[ok]]
    v1 = c.values[ok]
    span = (c.index[ok] - c.index[pos[ok]]).days / 365.25
    out = pd.Series((v1 / v0) ** (1.0 / np.maximum(span, 1e-9)) - 1.0, index=c.index[ok])
    out.name = f"rolling_{years:g}y"
    return out


def rolling_summary(strat_roll: pd.Series, bench_roll: pd.Series | None = None) -> dict[str, float]:
    """Distribution of rolling returns; ``pct_beat`` needs a benchmark series."""
    s = strat_roll.dropna()
    out = {k: np.nan for k in ("min", "median", "max", "pct_positive", "pct_beat", "n")}
    if s.empty:
        return out
    out.update(min=float(s.min()), median=float(s.median()), max=float(s.max()),
               pct_positive=float((s > 0).mean()), n=int(len(s)))
    if bench_roll is not None and len(bench_roll.dropna()):
        j = pd.concat({"s": s, "b": bench_roll}, axis=1).dropna()
        if len(j):
            out["pct_beat"] = float((j["s"] > j["b"]).mean())
    return out


# --------------------------------------------------------------------------- #
# Drawdown episodes
# --------------------------------------------------------------------------- #
def drawdown_episodes(curve: pd.Series, top: int = 5) -> pd.DataFrame:
    """The ``top`` deepest drawdowns: peak → trough → recovery (NaT if ongoing).

    Columns: peak, trough, recovery, depth (<0), days_to_trough,
    days_to_recover (NaN if ongoing), total_days.
    """
    cols = ["peak", "trough", "recovery", "depth", "days_to_trough", "days_to_recover", "total_days"]
    c = _clean(curve)
    if len(c) < 2:
        return pd.DataFrame(columns=cols)
    dd = c / c.cummax() - 1.0
    under = dd < 0
    rows = []
    i, n = 0, len(c)
    while i < n:
        if not under.iloc[i]:
            i += 1
            continue
        j = i
        while j < n and under.iloc[j]:
            j += 1
        seg = dd.iloc[i:j]
        peak = c.index[i - 1] if i > 0 else c.index[i]
        trough = seg.idxmin()
        recovery = c.index[j] if j < n else pd.NaT
        end = recovery if j < n else c.index[-1]
        rows.append({
            "peak": peak, "trough": trough, "recovery": recovery,
            "depth": float(seg.min()),
            "days_to_trough": int((trough - peak).days),
            "days_to_recover": float((recovery - trough).days) if j < n else np.nan,
            "total_days": int((end - peak).days),
        })
        i = j
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("depth").head(top).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SIP simulation & XIRR
# --------------------------------------------------------------------------- #
def xirr(cashflows: list[tuple]) -> float:
    """Annualised internal rate of return of dated cash flows (Actual/365).

    ``cashflows`` is ``[(date, amount), …]`` with investments negative and the
    final value positive. Solved by bisection (robust, no derivative needed).
    """
    if len(cashflows) < 2:
        return np.nan
    t0 = pd.Timestamp(cashflows[0][0])
    yrs = np.array([(pd.Timestamp(d) - t0).days / 365.0 for d, _ in cashflows])
    amt = np.array([float(a) for _, a in cashflows])
    if not (amt < 0).any() or not (amt > 0).any():
        return np.nan

    def npv(r: float) -> float:
        return float((amt / (1.0 + r) ** yrs).sum())

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9 or (hi - lo) < 1e-12:
            break
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def sip_dates(index: pd.DatetimeIndex, start=None, end=None) -> list[pd.Timestamp]:
    """First trading day of each calendar month within ``[start, end]``."""
    idx = index.sort_values()
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    if len(idx) == 0:
        return []
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).first())


def simulate_sip(
    curve: pd.Series, amount: float, start=None, step_up_pct: float = 0.0,
    dates: list[pd.Timestamp] | None = None,
) -> dict:
    """Invest ``amount`` on the first trading day of every month into ``curve``.

    Each instalment buys units at that day's level (fractional units, no extra
    costs); ``step_up_pct`` raises the instalment by that % every 12 months. Pass
    explicit ``dates`` to run two curves on an identical schedule.
    Returns invested, final_value, gain, xirr, n_instalments and a ``timeline``
    DataFrame (date, invested, value) at each instalment plus the end date.
    """
    c = _clean(curve)
    empty = {"invested": 0.0, "final_value": 0.0, "gain": 0.0, "xirr": np.nan,
             "n_instalments": 0, "timeline": pd.DataFrame(columns=["invested", "value"])}
    if len(c) < 2 or amount <= 0:
        return empty
    when = dates if dates is not None else sip_dates(c.index, start)
    when = [d for d in when if c.index[0] <= d <= c.index[-1]]
    if not when:
        return empty
    units, invested, flows, rows = 0.0, 0.0, [], []
    for k, d in enumerate(when):
        inst = amount * (1.0 + step_up_pct / 100.0) ** (k // 12)
        level = float(c.iloc[c.index.searchsorted(d, side="right") - 1])
        units += inst / level
        invested += inst
        flows.append((d, -inst))
        rows.append({"date": d, "invested": invested, "value": units * level})
    end = c.index[-1]
    final = units * float(c.iloc[-1])
    if end > rows[-1]["date"]:
        rows.append({"date": end, "invested": invested, "value": final})
    flows.append((end, final))
    return {
        "invested": invested, "final_value": final, "gain": final - invested,
        "xirr": xirr(flows), "n_instalments": len(when),
        "timeline": pd.DataFrame(rows).set_index("date"),
    }


# --------------------------------------------------------------------------- #
# Holdings & attribution
# --------------------------------------------------------------------------- #
def sector_exposure(holdings: dict[pd.Timestamp, pd.DataFrame], dates=None) -> pd.DataFrame:
    """Industry weights at each rebalance (rows = dates, columns = industries).

    Pass every simulated rebalance date as ``dates`` so all-cash periods appear
    as zero rows (otherwise a chart would carry the previous mix across them).
    Weights sum to the invested fraction, so 1 − row sum is cash.
    """
    rows = {}
    for d, h in holdings.items():
        if h is None or h.empty or "weight" not in h.columns:
            continue
        ind = h["Industry"].fillna("Unclassified") if "Industry" in h.columns else pd.Series("Unclassified", index=h.index)
        rows[d] = h["weight"].groupby(ind).sum()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).T
    if dates is not None:
        out = out.reindex(out.index.union(pd.DatetimeIndex(dates)))
    out = out.fillna(0.0).sort_index()
    out = out[out.mean().sort_values(ascending=False).index]
    out.index.name = "rebalance_date"
    return out


def summarize_contributions(contribs: pd.DataFrame) -> pd.DataFrame:
    """Per-stock P&L attribution across the whole backtest.

    Columns: Symbol, contribution_inr (₹ P&L on the notional capital),
    contribution (unitless share of the base-1 curve), periods_held, avg_weight,
    win_rate (share of held periods with a positive return), held_return
    (compounded return while held), best_period, worst_period.
    """
    cols = ["Symbol", "contribution_inr", "contribution", "periods_held", "avg_weight",
            "win_rate", "held_return", "best_period", "worst_period"]
    if contribs is None or contribs.empty:
        return pd.DataFrame(columns=cols)
    g = contribs.groupby("Symbol")
    out = pd.DataFrame({
        "contribution_inr": g["contribution_inr"].sum(),
        "contribution": g["contribution"].sum(),
        "periods_held": g.size(),
        "avg_weight": g["weight"].mean(),
        "win_rate": g["ret"].apply(lambda r: float((r > 0).mean())),
        "held_return": g["ret"].apply(lambda r: float((1.0 + r).prod() - 1.0)),
        "best_period": g["ret"].max(),
        "worst_period": g["ret"].min(),
    }).reset_index()
    return out.sort_values("contribution_inr", ascending=False).reset_index(drop=True)[cols]


# --------------------------------------------------------------------------- #
# Excel export
# --------------------------------------------------------------------------- #
def _sheet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Excel cannot store tz-aware datetimes or dicts; flatten for export."""
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.tz_localize(None)
        elif out[c].dtype == object:
            out[c] = out[c].map(lambda v: v if isinstance(v, (str, int, float, bool, type(None),
                                                               pd.Timestamp, dt.date)) else str(v))
    return out


def excel_report(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Write ``{sheet name: frame}`` to an in-memory .xlsx (sheet names ≤31 chars)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd",
                        date_format="yyyy-mm-dd") as xw:
        for name, frame in sheets.items():
            if frame is None:
                continue
            df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            keep_index = not isinstance(df.index, pd.RangeIndex)
            _sheet_safe(df).to_excel(xw, sheet_name=str(name)[:31], index=keep_index)
            ws = xw.sheets[str(name)[:31]]
            ws.set_column(0, max(1, df.shape[1] + int(keep_index)), 14)
    return buf.getvalue()


def metrics_table(metrics: dict, bench_metrics: dict | None, relative: dict | None) -> pd.DataFrame:
    """Flatten metric dicts into a two/three-column sheet for export."""
    keys = list(dict.fromkeys(list(metrics) + list(bench_metrics or {})))
    rows = []
    for k in keys:
        v = metrics.get(k)
        b = (bench_metrics or {}).get(k)
        rows.append({"metric": k, "strategy": _fmt_cell(v), "benchmark": _fmt_cell(b)})
    for k, v in (relative or {}).items():
        rows.append({"metric": f"vs benchmark: {k}", "strategy": _fmt_cell(v), "benchmark": None})
    return pd.DataFrame(rows)


def _fmt_cell(v):
    if isinstance(v, (pd.Timestamp, dt.date)):
        return pd.Timestamp(v).date().isoformat()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v
