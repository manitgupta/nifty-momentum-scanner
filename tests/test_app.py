"""Headless UI tests for the Streamlit app via streamlit.testing.v1.AppTest.

Network access (Yahoo Finance / NSE) is monkeypatched with deterministic
synthetic data, so these run offline and fast. They cover:
  * the app loads with both tabs and the backtest disclaimer,
  * the Scanner tab still works (regression on the tabs refactor),
  * the Backtest tab runs and renders charts/metrics/tables,
  * scan → backtest in one session (no st.stop() leakage across tabs),
  * benchmark-unavailable degrades gracefully,
  * the whole-index universe mode works.

Run with:  ./venv/bin/python tests/test_app.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

from momentum import backtest as bt_mod
from momentum.data import PriceData

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PASS, FAIL = "PASS", "FAIL"
_failures = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _failures
    print(f"[{PASS if cond else FAIL}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        _failures += 1


def ss(at, key):
    """AppTest.session_state has no .get(); read a key or return None."""
    return at.session_state[key] if key in at.session_state else None


# --------------------------------------------------------------------------- #
# Synthetic offline dataset
# --------------------------------------------------------------------------- #
def _dataset():
    today = dt.date.today()
    idx = pd.bdate_range(end=pd.Timestamp(today), periods=11 * 252)
    drifts = {"AAA.NS": 0.0009, "BBB.NS": 0.0007, "CCC.NS": 0.0005,
              "DDD.NS": 0.0003, "EEE.NS": 0.0002, "FFF.NS": 0.0006}
    close = pd.DataFrame(
        {tk: 100.0 * np.exp(np.cumsum(np.full(len(idx), d))) for tk, d in drifts.items()},
        index=idx,
    )
    vol = pd.DataFrame(1_000_000, index=idx, columns=close.columns)
    market = pd.DataFrame(
        {"market_cap": [5e11] * len(close.columns), "shares": [1e9] * len(close.columns),
         "last_price": [1.0] * len(close.columns)},
        index=close.columns,
    )
    pdata = PriceData(close=close, raw_close=close.copy(), volume=vol, market=market, failed=[])
    uni = pd.DataFrame({
        "Symbol": [c[:-3] for c in close.columns],
        "Company": [c + " Co" for c in close.columns],
        "Industry": ["X"] * len(close.columns),
        "Segment": ["Largecap"] * len(close.columns),
        "YFTicker": list(close.columns),
    })
    bench = pd.Series(np.linspace(1000, 3000, len(idx)), index=idx, name="close")
    return pdata, uni, bench


PDATA, UNI, BENCH = _dataset()


def _fake_fetch_all(tickers, **kw):
    for cb in (kw.get("price_progress"), kw.get("market_progress")):
        if cb:
            cb(len(tickers), len(tickers))
    cols = [t for t in PDATA.close.columns if t in set(tickers)] or list(PDATA.close.columns)
    return PriceData(close=PDATA.close[cols], raw_close=PDATA.raw_close[cols],
                     volume=PDATA.volume[cols], market=PDATA.market.loc[cols], failed=[])


def _fake_benchmark(ticker, cache_key, **kw):
    return BENCH.copy()


def _fake_no_benchmark(ticker, cache_key, **kw):
    return pd.Series(dtype="float64", name="close")


def _fake_fetch_all_short(tickers, **kw):
    """Only ~8 months of history — too little for a backtest holding period."""
    for cb in (kw.get("price_progress"), kw.get("market_progress")):
        if cb:
            cb(len(tickers), len(tickers))
    cols = [t for t in PDATA.close.columns if t in set(tickers)] or list(PDATA.close.columns)
    short_idx = PDATA.close.index[-160:]  # ~8 months of trading days
    return PriceData(
        close=PDATA.close.loc[short_idx, cols], raw_close=PDATA.raw_close.loc[short_idx, cols],
        volume=PDATA.volume.loc[short_idx, cols], market=PDATA.market.loc[cols], failed=[],
    )


def _fake_universe(segments, force_refresh=False):
    return UNI.copy()


def _patched():
    return (
        patch("momentum.data.fetch_all", _fake_fetch_all),
        patch("momentum.data.fetch_benchmark", _fake_benchmark),
        patch("momentum.universe.build_universe", _fake_universe),
    )


# --------------------------------------------------------------------------- #
def test_app_loads_tabs():
    at = AppTest.from_file(APP, default_timeout=60).run()
    check("load: no exception", not at.exception, str(at.exception))
    check("load: two tabs", [t.label for t in at.tabs] == ["🔎 Scanner", "🧪 Backtest"],
          str([t.label for t in at.tabs]))
    check("load: backtest disclaimer shown",
          any("Backtest limitations" in w.value for w in at.warning))
    check("load: scan prompt shown",
          any("Run scan" in i.value for i in at.info))


# --------------------------------------------------------------------------- #
def test_scan_happy_path():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.button(key="run_scan").click().run()
        check("scan: no exception", not at.exception, str(at.exception))
        check("scan: session populated", "scan" in at.session_state and at.session_state["scan"])
        check("scan: selection subheader rendered",
              any("by Normalized Momentum Score" in s.value for s in at.subheader))
        check("scan: summary metrics present", len(at.metric) >= 4)
        check("scan: a results table rendered", len(at.dataframe) >= 1)


# --------------------------------------------------------------------------- #
def test_backtest_happy_path():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        at = AppTest.from_file(APP, default_timeout=180).run()
        at.slider(key="bt_years").set_value(5)
        at.selectbox(key="bt_freq").set_value("Semi-annual")
        at.button(key="run_bt").click().run()
        check("backtest: no exception", not at.exception, str(at.exception))
        bt = ss(at, "bt")
        check("backtest: result stored", bool(bt) and bt.get("res") is not None)
        res = bt["res"] if bt else None
        check("backtest: is BacktestResult", isinstance(res, bt_mod.BacktestResult))
        check("backtest: equity curve non-empty", res is not None and not res.equity_curve.empty)
        check("backtest: multiple rebalances", res is not None and len(res.rebalance_dates) >= 3)
        subs = [s.value for s in at.subheader]
        check("backtest: equity/drawdown/metrics subheaders",
              all(x in subs for x in ["Growth of ₹100", "Drawdown", "Performance metrics"]), str(subs))
        check("backtest: KPI metrics rendered", len(at.metric) >= 5)
        check("backtest: metric tables rendered", len(at.dataframe) >= 3)
        check("backtest: disclaimer still shown",
              any("Backtest limitations" in w.value for w in at.warning))


# --------------------------------------------------------------------------- #
def test_scan_then_backtest_no_stop_leak():
    """Both tabs must render in one run — guards against st.stop() leaking across tabs."""
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        at = AppTest.from_file(APP, default_timeout=180).run()
        at.button(key="run_scan").click().run()
        at.slider(key="bt_years").set_value(4)
        at.button(key="run_bt").click().run()
        check("both: no exception", not at.exception, str(at.exception))
        check("both: scan result present", bool(ss(at, "scan")))
        check("both: backtest result present",
              bool(ss(at, "bt")) and ss(at, "bt").get("res") is not None)
        subs = [s.value for s in at.subheader]
        check("both: scan subheader present", any("Normalized Momentum Score" in s for s in subs))
        check("both: backtest subheader present", "Growth of ₹100" in subs)


# --------------------------------------------------------------------------- #
def test_backtest_benchmark_unavailable():
    with patch("momentum.data.fetch_all", _fake_fetch_all), \
         patch("momentum.data.fetch_benchmark", _fake_no_benchmark), \
         patch("momentum.universe.build_universe", _fake_universe):
        at = AppTest.from_file(APP, default_timeout=180).run()
        at.slider(key="bt_years").set_value(4)
        at.button(key="run_bt").click().run()
        check("no-bench: no exception", not at.exception, str(at.exception))
        res = at.session_state["bt"]["res"]
        check("no-bench: benchmark curve is None", res.benchmark_curve is None)
        check("no-bench: strategy still computed", not res.equity_curve.empty)
        check("no-bench: benchmark-gap note recorded",
              any("benchmark" in n.lower() for n in res.notes))


# --------------------------------------------------------------------------- #
def test_backtest_too_short_window():
    """Insufficient history -> engine returns empty -> UI shows an error, no charts."""
    with patch("momentum.data.fetch_all", _fake_fetch_all_short), \
         patch("momentum.data.fetch_benchmark", _fake_benchmark), \
         patch("momentum.universe.build_universe", _fake_universe):
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.slider(key="bt_years").set_value(1)
        at.selectbox(key="bt_freq").set_value("Annual")
        at.button(key="run_bt").click().run()
        check("too-short: no exception", not at.exception, str(at.exception))
        check("too-short: error shown", len(at.error) >= 1)
        subs = [s.value for s in at.subheader]
        check("too-short: no equity-curve subheader", "Growth of ₹100" not in subs, str(subs))


# --------------------------------------------------------------------------- #
def test_scan_no_segments_error():
    """Unchecking every size segment -> the Scanner shows a clear error, no crash."""
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        at = AppTest.from_file(APP, default_timeout=120).run()
        for k in ("seg_large", "seg_mid", "seg_small"):
            at.checkbox(key=k).set_value(False)
        at.run()
        at.button(key="run_scan").click().run()
        check("no-segments: no exception", not at.exception, str(at.exception))
        check("no-segments: segment error shown",
              any("at least one universe segment" in e.value for e in at.error),
              str([e.value for e in at.error]))


# --------------------------------------------------------------------------- #
def test_whole_index_universe_mode():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        at = AppTest.from_file(APP, default_timeout=120)
        at.run()
        at.radio(key="uni_mode").set_value("Whole index").run()
        check("whole-index: no exception after switch", not at.exception, str(at.exception))
        check("whole-index: index selectbox appears",
              any(s.label == "Index universe" for s in at.selectbox))
        # A scan in whole-index mode should still work.
        at.button(key="run_scan").click().run()
        check("whole-index: scan runs", not at.exception and bool(ss(at, "scan")))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_app_loads_tabs()
    test_scan_happy_path()
    test_backtest_happy_path()
    test_scan_then_backtest_no_stop_leak()
    test_backtest_benchmark_unavailable()
    test_backtest_too_short_window()
    test_scan_no_segments_error()
    test_whole_index_universe_mode()
    print("-" * 50)
    if _failures:
        print(f"{_failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed.")
