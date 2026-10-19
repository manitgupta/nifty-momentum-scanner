"""Headless UI tests for the Phase-5 features (Streamlit AppTest, offline).

Network access is monkeypatched: price history is synthesised deterministically
for ANY requested ticker (so ETF / custom / index universes all work), and the
benchmark is a synthetic series. Covers:
  * new universe modes — multi-index, ETF basket, custom symbol list,
  * scanner weighting choice, sector cap, regime banner, portfolio builder and
    rebalance-order generation,
  * daily / weekly / fortnightly rebalancing,
  * risk controls — rank buffer, regime filter (+ cash yield), sector cap,
    inverse-vol weighting, cost presets,
  * benchmark options — Auto (universe-matched), TRI flag, universe equal-weight,
  * the new report sections (calendar / rolling / drawdowns / SIP / attribution /
    sectors) and the Excel export,
  * the Compare-strategies tab.

Run with:  ./venv/bin/python tests/test_app_features.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

from momentum import config
from momentum.data import PriceData

APP = str(Path(__file__).resolve().parent.parent / "app.py")
_failures = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _failures
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        _failures += 1


def ss(at, key):
    return at.session_state[key] if key in at.session_state else None


# --------------------------------------------------------------------------- #
# Synthetic, ticker-agnostic data layer
# --------------------------------------------------------------------------- #
TODAY = dt.date.today()
IDX = pd.bdate_range(end=pd.Timestamp(TODAY), periods=6 * 252)


def _drift(ticker: str) -> float:
    h = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
    return 0.0001 + (h % 1000) / 1000 * 0.0009


def _synth(tickers) -> PriceData:
    tickers = list(dict.fromkeys(tickers))
    t = np.arange(len(IDX))
    close = pd.DataFrame(
        {tk: 100.0 * np.exp(_drift(tk) * t + 0.03 * np.sin(t / (17 + i % 5))) for i, tk in enumerate(tickers)},
        index=IDX,
    )
    vol = pd.DataFrame(1_000_000.0, index=IDX, columns=close.columns)
    market = pd.DataFrame({"market_cap": 5e11, "shares": 1e9, "last_price": 1.0}, index=close.columns)
    return PriceData(close=close, raw_close=close.copy(), volume=vol, market=market, failed=[])


def fake_fetch_all(tickers, **kw):
    for cb in (kw.get("price_progress"), kw.get("market_progress")):
        if cb:
            cb(len(tickers), len(tickers))
    return _synth(tickers)


def _stock_universe(n=12, segment="Largecap"):
    inds = ["Banks", "IT", "Pharma", "Auto"]
    syms = [f"STK{i:02d}" for i in range(n)]
    return pd.DataFrame({
        "Symbol": syms, "Company": [s + " Ltd" for s in syms],
        "Industry": [inds[i % len(inds)] for i in range(n)],
        "Segment": segment, "YFTicker": [s + ".NS" for s in syms],
    })


def fake_build_universe(segments, force_refresh=False):
    if segments == [config.ETF_SEGMENT]:
        etf = pd.read_csv(config.ETF_LIST_PATH)
        return pd.DataFrame({"Symbol": etf["Symbol"], "Company": etf["Company Name"], "Industry": etf["Industry"],
                             "Segment": "ETF", "YFTicker": etf["Symbol"] + ".NS"})
    frames = [_stock_universe(8, seg).assign(Symbol=lambda d, s=seg: s[:3].upper() + d["Symbol"],
                                              YFTicker=lambda d, s=seg: s[:3].upper() + d["Symbol"] + ".NS")
              for seg in segments]
    return pd.concat(frames, ignore_index=True)


BENCH_CALLS: list[dict] = []


def fake_benchmark(ticker, cache_key, **kw):
    BENCH_CALLS.append({"ticker": ticker, "cache_key": cache_key, **kw})
    t = np.arange(len(IDX))
    s = pd.Series(1000.0 * np.exp(0.0003 * t), index=IDX, name="close")
    s.attrs["source"] = "synthetic"
    return s


def fake_falling_benchmark(ticker, cache_key, **kw):
    """Rises for 3 years, then falls — so a 200-day SMA filter turns risk-off."""
    t = np.arange(len(IDX))
    half = len(IDX) // 2
    s = pd.Series(np.where(t < half, 1000.0 * np.exp(0.0006 * t), 1000.0 * np.exp(0.0006 * half - 0.0008 * (t - half))),
                  index=IDX, name="close")
    s.attrs["source"] = "synthetic-falling"
    return s


def _patched(bench=fake_benchmark):
    return (patch("momentum.data.fetch_all", fake_fetch_all),
            patch("momentum.data.fetch_benchmark", bench),
            patch("momentum.universe.build_universe", fake_build_universe))


def _app(timeout=240):
    return AppTest.from_file(APP, default_timeout=timeout).run()


def _run_bt(at, years=3, freq="Quarterly", **settings):
    at.date_input(key="bt_start").set_value(TODAY - dt.timedelta(days=int(years * 365)))
    at.selectbox(key="bt_freq").set_value(freq)
    for key, (kind, val) in settings.items():
        getattr(at, kind)(key=key).set_value(val)
    at.button(key="run_bt").click().run()
    return ss(at, "bt")


# --------------------------------------------------------------------------- #
def test_multi_index_universe():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        at = _app()
        at.radio(key="uni_mode").set_value("Index").run()
        at.multiselect(key="idx_choice").set_value(["Nifty Bank", "Nifty Financial Services", "Nifty Microcap 250"]).run()
        at.button(key="run_scan").click().run()
        check("multi-index: no exception", not at.exception, str(at.exception))
        scan = ss(at, "scan")
        check("multi-index: scan stored", bool(scan))
        check("multi-index: three NSE index keys resolved",
              scan and scan["segments"] == ["NiftyBank", "NiftyFinancialServices", "NiftyMicrocap250"],
              str(scan and scan["segments"]))
        check("multi-index: label names the indices",
              scan and "Nifty Bank" in scan["label"] and "Nifty Microcap 250" in scan["label"])


def test_micro_size_segment():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        BENCH_CALLS.clear()
        at = _app()
        check("micro: universe radio is vertical-friendly (4 options)", len(at.radio(key="uni_mode").options) == 4)
        at.checkbox(key="seg_large").set_value(False)
        at.checkbox(key="seg_micro").set_value(True).run()
        at.button(key="run_scan").click().run()
        check("micro: no exception", not at.exception, str(at.exception))
        scan = ss(at, "scan")
        check("micro: segment key 'Micro'", scan and scan["segments"] == ["Micro"], str(scan and scan["segments"]))
        check("micro: labelled Nifty Microcap 250", scan and scan["label"] == "Nifty Microcap 250")
        check("micro: regime/auto benchmark = Nifty Microcap 250",
              any(c.get("nse_name") == "NIFTY MICROCAP 250" for c in BENCH_CALLS), str(BENCH_CALLS[-1:]))
        for k in ("seg_large", "seg_mid", "seg_small"):
            at.checkbox(key=k).set_value(True)
        at.run()
        BENCH_CALLS.clear()
        at.date_input(key="bt_start").set_value(TODAY - dt.timedelta(days=3 * 365))
        at.button(key="run_bt").click().run()
        bt = ss(at, "bt")
        check("micro: all four segments → Auto benchmark Nifty Total Market",
              bt and bt["bench_name"] == "Nifty Total Market", str(bt and bt["bench_name"]))


def test_etf_universe_scan_and_builder():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        BENCH_CALLS.clear()
        at = _app()
        at.radio(key="uni_mode").set_value("ETFs").run()
        check("etf: basket multiselect shown", any(m.label == "ETFs in the basket" for m in at.multiselect))
        at.multiselect(key="etf_pick").set_value(["NIFTYBEES", "BANKBEES", "GOLDBEES", "ITBEES", "MON100", "SILVERBEES"]).run()
        at.text_input(key="etf_extra").input("MAHKTECH, bad!tok").run()
        at.button(key="run_scan").click().run()
        check("etf: no exception", not at.exception, str(at.exception))
        scan = ss(at, "scan")
        res = scan["result"] if scan else None
        check("etf: universe = 6 picked + 1 extra", scan and scan["universe_size"] == 7, str(scan and scan["universe_size"]))
        check("etf: price-only default weighting (score)", scan and scan["params"].weighting == "score")
        check("etf: default Top N is small (5)", res is not None and len(res.selected) == 5, str(res is not None and len(res.selected)))
        check("etf: turnover-ratio filter off by default", scan and not scan["params"].apply_turnover_ratio_filter)
        cols = [list(df.value.columns) for df in at.dataframe]
        check("etf: portfolio builder table (Shares/Invested ₹)",
              any("Shares" in c and "Invested ₹" in c for c in cols), str(cols))
        check("etf: regime banner shown",
              any("Market regime" in s.value for s in at.success) or any("Market regime" in e.value for e in at.error))
        check("etf: regime used the ETF auto benchmark (Nifty 50)",
              any(c.get("nse_name") == "NIFTY 50" for c in BENCH_CALLS), str(BENCH_CALLS[-1:]))

        # Rebalance an existing portfolio: one name to exit, one to add to, one unknown.
        held_sym = res.selected["Symbol"].iloc[0]
        at.text_area(key="pb_holdings").input(f"{held_sym}, 10\nUNKNOWNX, 5\nGOLDBEES 1").run()
        check("rebalance: no exception", not at.exception, str(at.exception))
        cols = [list(df.value.columns) for df in at.dataframe]
        orders = [df.value for df in at.dataframe if "action" in df.value.columns]
        check("rebalance: orders table rendered", bool(orders), str(cols))
        if orders:
            o = orders[0]
            check("rebalance: unknown holding flagged REVIEW",
                  "REVIEW" in set(o.loc[o["Symbol"] == "UNKNOWNX", "action"]))
            check("rebalance: buys present for new names", (o["action"] == "BUY").any())
        check("rebalance: REVIEW warning shown", any("REVIEW" in w.value for w in at.warning))


def test_custom_universe_and_sector_cap():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        at = _app()
        at.radio(key="uni_mode").set_value("Custom list").run()
        at.text_area(key="custom_syms").input("RELIANCE, HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK, TCS, INFY").run()
        at.number_input(key="max_sector").set_value(1).run()
        at.button(key="run_scan").click().run()
        check("custom: no exception", not at.exception, str(at.exception))
        scan = ss(at, "scan")
        res = scan["result"] if scan else None
        check("custom: 8 symbols in universe", scan and scan["universe_size"] == 8)
        check("custom: company names filled from bundled lists",
              res is not None and res.results["Company"].str.contains("Reliance").any())
        if res is not None and not res.selected.empty:
            per = res.selected["Industry"].value_counts()
            check("custom: sector cap 1 → at most one name per industry", int(per.max()) == 1, str(per.to_dict()))


def test_new_frequencies():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        for freq, years, lo in (("Daily", 1.5, 150), ("Weekly", 2, 60), ("Fortnightly", 2, 30)):
            at = _app()
            bt = _run_bt(at, years=years, freq=freq)
            check(f"{freq.lower()}: no exception", not at.exception, str(at.exception))
            res = bt["res"] if bt else None
            n = len(res.rebalance_dates) if res is not None else 0
            check(f"{freq.lower()}: many rebalances ({n} ≥ {lo})", n >= lo)
            check(f"{freq.lower()}: frequency recorded", bt and bt["freq"] == freq)


def test_risk_controls_and_weighting():
    p1, p2, p3 = _patched(bench=fake_falling_benchmark)
    with p1, p2, p3:
        at = _app()
        bt = _run_bt(at, years=4, freq="Monthly",
                     bt_buffer_on=("checkbox", True), bt_regime_on=("checkbox", True),
                     bt_weighting=("radio", "inverse_vol"), bt_cost_preset=("selectbox", "Statutory only (~11 bps)"),
                     bt_cash_yield=("number_input", 6.0), bt_topn=("number_input", 4))
        check("risk: no exception", not at.exception, str(at.exception))
        res = bt["res"] if bt else None
        p = res.params if res is not None else None
        check("risk: buffer exit rank = 2×TopN", p is not None and p.exit_rank == 8, str(p and p.exit_rank))
        check("risk: regime filter on", p is not None and p.regime_filter)
        check("risk: inverse-vol weighting", p is not None and p.weighting == "inverse_vol")
        check("risk: statutory cost preset = 11 bps", p is not None and p.cost_bps == 11.0)
        check("risk: cash yield 6%", p is not None and abs(p.cash_yield_annual - 0.06) < 1e-12)
        pr = res.period_returns if res is not None else pd.DataFrame()
        check("risk: some rebalances risk-off in the falling market",
              not pr.empty and bool((pr["risk_on"] == False).any()))  # noqa: E712
        check("risk: time-in-market metric < 100%", res is not None and res.metrics.get("time_in_market", 1) < 1)
        check("risk: time-invested KPI rendered", any(m.label == "Time invested" for m in at.metric))


def test_benchmark_options_and_reports():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        BENCH_CALLS.clear()
        at = _app()
        bt = _run_bt(at, years=4, freq="Monthly", bt_tri=("checkbox", True))
        check("bench-auto: no exception", not at.exception, str(at.exception))
        check("bench-auto: Largecap → Nifty 100", bt and bt["bench_name"] == "Nifty 100", str(bt and bt["bench_name"]))
        check("bench-auto: TRI requested from NSE", any(c.get("tri") and c.get("nse_name") == "NIFTY 100" for c in BENCH_CALLS))
        subs = [s.value for s in at.subheader]
        for want in ("📅 Calendar returns", "📈 Rolling returns", "🕳️ Worst drawdowns", "💰 SIP simulator",
                     "🏆 Stock contributions", "🏭 Sector exposure over time"):
            check(f"reports: '{want}' section rendered", want in subs, str(subs))
        check("reports: SIP XIRR metric", any(m.label == "Strategy XIRR" for m in at.metric))
        check("reports: rolling % beating metric", any("beating benchmark" in m.label for m in at.metric))
        cols = [list(df.value.columns) for df in at.dataframe]
        check("reports: contributions table", any("Contribution ₹" in c for c in cols), str(cols))
        check("reports: annual returns table has benchmark column", any("Nifty 100" in c and "Excess" in c for c in cols))

        at.button(key="bt_build_xlsx").click().run()
        check("excel: no exception", not at.exception, str(at.exception))
        x = ss(at, "bt_xlsx")
        check("excel: workbook built (zip magic)", bool(x) and x[1][:2] == b"PK" and len(x[1]) > 5000)

        # Universe equal-weight benchmark — computed, no fetch.
        BENCH_CALLS.clear()
        at.selectbox(key="bt_bench").set_value(config.UNIVERSE_EW_BENCHMARK)
        at.button(key="run_bt").click().run()
        bt2 = ss(at, "bt")
        check("bench-ew: no exception", not at.exception, str(at.exception))
        check("bench-ew: label 'Universe EW'", bt2 and bt2["bench_name"] == "Universe EW")
        check("bench-ew: relative metrics populated",
              bt2 and np.isfinite(bt2["res"].relative_metrics.get("beta", np.nan)))
        check("bench-ew: nothing fetched", not BENCH_CALLS, str(BENCH_CALLS))


def test_compare_tab():
    p1, p2, p3 = _patched()
    with p1, p2, p3:
        at = _app()
        at.date_input(key="bt_start").set_value(TODAY - dt.timedelta(days=4 * 365))
        at.multiselect(key="cmp_freqs").set_value(["Monthly", "Quarterly"])
        at.multiselect(key="cmp_topn").set_value([3, 5])
        at.button(key="run_cmp").click().run()
        check("compare: no exception", not at.exception, str(at.exception))
        cmp = ss(at, "cmp")
        check("compare: 4 combinations", cmp and len(cmp["table"]) == 4, str(cmp and len(cmp["table"])))
        check("compare: curves for each combo", cmp and len(cmp["curves"]) == 4)
        subs = [s.value for s in at.subheader]
        check("compare: heatmap + top-5 sections", any("by frequency × Top N" in s for s in subs)
              and any("Top 5 combinations" in s for s in subs), str(subs))


if __name__ == "__main__":
    test_multi_index_universe()
    test_micro_size_segment()
    test_etf_universe_scan_and_builder()
    test_custom_universe_and_sector_cap()
    test_new_frequencies()
    test_risk_controls_and_weighting()
    test_benchmark_options_and_reports()
    test_compare_tab()
    print("-" * 50)
    if _failures:
        print(f"{_failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed.")
