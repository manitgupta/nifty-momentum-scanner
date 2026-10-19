"""Unit checks for the Phase-5 helper modules (offline, no pytest).

    analytics  - calendar/annual/rolling returns, drawdown episodes, XIRR & SIP,
                 sector exposure, contribution summary, Excel export
    portfolio  - whole-share allocation, holdings parsing, rebalance orders
    universe   - symbol parsing, custom & ETF universes, new index registry
    data       - benchmark source ladder (NSE → snapshot+Yahoo → Yahoo), parsing

Run with:  ./venv/bin/python tests/test_features.py
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from momentum import analytics, config, data, portfolio, universe

_failures = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _failures
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        _failures += 1


# --------------------------------------------------------------------------- #
# analytics
# --------------------------------------------------------------------------- #
def test_calendar_and_annual():
    idx = pd.bdate_range("2021-01-01", "2023-12-29")
    c = pd.Series(np.exp(np.linspace(0, np.log(1.6), len(idx))), index=idx)
    cal = analytics.calendar_returns(c)
    check("calendar: 3 years × 12 months + Year", cal.shape == (3, 13) and list(cal.columns[:12]) == analytics.MONTHS)
    check("calendar: months compound to the year",
          np.allclose((1 + cal[analytics.MONTHS]).prod(axis=1) - 1, cal["Year"], rtol=1e-10))
    check("calendar: years compound to the total",
          math.isclose((1 + cal["Year"]).prod() - 1, c.iloc[-1] / c.iloc[0] - 1, rel_tol=1e-10))
    partial = analytics.calendar_returns(c.loc["2021-03-10":])
    check("calendar: partial first month measured from first value", np.isnan(partial.loc[2021, "Jan"])
          and np.isfinite(partial.loc[2021, "Mar"]))
    ann = analytics.annual_returns(c, c * 2.0)
    check("annual: excess 0 when benchmark is a scaled copy", np.allclose(ann["Excess"], 0.0, atol=1e-12))
    check("calendar: empty input", analytics.calendar_returns(pd.Series(dtype=float)).empty)


def test_rolling_and_summary():
    idx = pd.bdate_range("2018-01-01", "2023-12-29")
    g = 0.12
    days = (idx - idx[0]).days / 365.25
    c = pd.Series((1 + g) ** days, index=idx)
    r1 = analytics.rolling_returns(c, 1)
    check("rolling: constant-growth curve → rolling CAGR ≈ g", np.allclose(r1, g, atol=2e-3), f"{r1.min()}..{r1.max()}")
    check("rolling: first point ≥ 1 year after start", r1.index[0] >= idx[0] + pd.Timedelta(days=365))
    r3 = analytics.rolling_returns(c, 3)
    check("rolling: longer window has fewer points", len(r3) < len(r1))
    b = pd.Series((1 + 0.05) ** days, index=idx)
    s = analytics.rolling_summary(r1, analytics.rolling_returns(b, 1))
    check("rolling summary: beats a 5% benchmark 100% of the time", s["pct_beat"] == 1.0 and s["pct_positive"] == 1.0)
    check("rolling: window longer than history → empty", analytics.rolling_returns(c.iloc[:100], 3).empty)


def test_drawdown_episodes():
    v = pd.Series([1, 1.2, 0.6, 0.8, 1.25, 1.1, 1.3, 1.0], index=pd.bdate_range("2021-01-01", periods=8))
    eps = analytics.drawdown_episodes(v, top=5)
    check("episodes: three episodes, deepest first", len(eps) == 3 and math.isclose(eps.loc[0, "depth"], -0.5))
    check("episodes: dates of the deepest", eps.loc[0, "peak"] == v.index[1] and eps.loc[0, "trough"] == v.index[2]
          and eps.loc[0, "recovery"] == v.index[4])
    ongoing = eps[eps["recovery"].isna()]
    check("episodes: final unrecovered episode has NaT recovery", len(ongoing) == 1 and np.isnan(ongoing["days_to_recover"].iloc[0]))
    check("episodes: monotone curve → none", analytics.drawdown_episodes(pd.Series([1.0, 1.1, 1.2],
          index=pd.bdate_range("2021-01-01", periods=3))).empty)


def test_xirr_and_sip():
    check("xirr: 10% simple case", math.isclose(analytics.xirr([(dt.date(2020, 1, 1), -100), (dt.date(2020, 12, 31), 110)]),
                                                0.10, rel_tol=1e-6))
    flows = [(dt.date(2020, 1, 1), -1000), (dt.date(2021, 1, 1), -1000), (dt.date(2022, 1, 1), 2310)]
    r = analytics.xirr(flows)
    npv = sum(a / (1 + r) ** ((pd.Timestamp(d) - pd.Timestamp(flows[0][0])).days / 365) for d, a in flows)
    check("xirr: NPV at solution ≈ 0", abs(npv) < 1e-6, str(npv))
    check("xirr: no positive flow → NaN", np.isnan(analytics.xirr([(dt.date(2020, 1, 1), -1), (dt.date(2021, 1, 1), -1)])))

    idx = pd.bdate_range("2019-01-01", "2023-12-29")
    days = (idx - idx[0]).days / 365.25
    c = pd.Series(1.15 ** days, index=idx)
    s = analytics.simulate_sip(c, 10_000)
    check("sip: one instalment per month (60)", s["n_instalments"] == 60)
    check("sip: invested = 60 × 10k", math.isclose(s["invested"], 600_000))
    check("sip: XIRR ≈ curve CAGR (15%)", abs(s["xirr"] - 0.15) < 0.003, str(s["xirr"]))
    su = analytics.simulate_sip(c, 10_000, step_up_pct=10)
    check("sip: step-up raises instalments yearly", math.isclose(su["invested"], 10_000 * 12 * sum(1.1 ** k for k in range(5)), rel_tol=1e-9))
    dates = analytics.sip_dates(idx)
    check("sip: first trading day of each month", dates[0] == idx[0] and all(d.day <= 4 for d in dates))
    check("sip: zero amount → empty", analytics.simulate_sip(c, 0)["n_instalments"] == 0)


def test_sector_and_contributions():
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")
    holdings = {
        d1: pd.DataFrame({"Industry": ["Bank", "Bank", "IT"], "weight": [0.3, 0.3, 0.4]}, index=list("ABC")),
        d2: pd.DataFrame({"Industry": ["IT", None], "weight": [0.5, 0.5]}, index=list("CD")),
    }
    sx = analytics.sector_exposure(holdings)
    check("sector: weights per date sum to 1", np.allclose(sx.sum(axis=1), 1.0))
    check("sector: Bank 60% then 0%", math.isclose(sx.loc[d1, "Bank"], 0.6) and sx.loc[d2, "Bank"] == 0)
    check("sector: missing industry → Unclassified", "Unclassified" in sx.columns)
    contribs = pd.DataFrame({
        "Symbol": ["A", "A", "B"], "weight": [0.5, 0.4, 0.5], "ret": [0.10, -0.05, 0.02],
        "contribution": [0.05, -0.02, 0.01], "contribution_inr": [5.0, -2.0, 1.0],
    })
    sm = analytics.summarize_contributions(contribs)
    check("contrib summary: sorted by ₹ contribution", list(sm["Symbol"]) == ["A", "B"])
    a = sm.set_index("Symbol").loc["A"]
    check("contrib summary: totals / periods / win rate", a["contribution_inr"] == 3.0 and a["periods_held"] == 2
          and a["win_rate"] == 0.5)
    check("contrib summary: compounded return while held", math.isclose(a["held_return"], 1.10 * 0.95 - 1))


def test_excel_report():
    sheets = {"Metrics": analytics.metrics_table({"cagr": 0.1, "mdd_peak_date": pd.Timestamp("2024-01-01")},
                                                  {"cagr": 0.08}, {"beta": 1.1}),
              "Equity": pd.DataFrame({"x": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2)),
              "A very long sheet name that exceeds Excel's limit": pd.DataFrame({"a": [1]}),
              "Objects": pd.DataFrame({"o": [{"k": 1}, None]})}
    blob = analytics.excel_report(sheets)
    check("excel: zip container", blob[:2] == b"PK" and len(blob) > 3000)
    back = pd.read_excel(io.BytesIO(blob), sheet_name=None) if _has_reader() else None
    if back is not None:
        check("excel: sheet names truncated to 31 chars", all(len(n) <= 31 for n in back))


def _has_reader() -> bool:
    try:
        import openpyxl  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# portfolio
# --------------------------------------------------------------------------- #
SEL = pd.DataFrame({"Symbol": ["A", "B", "C"], "Company": ["a", "b", "c"],
                    "weight": [0.5, 0.3, 0.2], "close_price": [1000.0, 333.0, 2500.0]})


def test_allocate():
    t, left = portfolio.allocate(SEL, 100_000, top_up=False)
    check("allocate: floor shares", list(t["shares"]) == [50, 90, 8])
    check("allocate: leftover = capital − invested", math.isclose(left, 100_000 - t["invested"].sum()))
    t2, left2 = portfolio.allocate(SEL, 100_000, top_up=True)
    check("allocate: top-up never overspends", left2 >= 0 and t2["invested"].sum() <= 100_000)
    check("allocate: top-up leaves less idle cash", left2 <= left)
    bad = SEL.assign(close_price=[1000.0, np.nan, 0.0])
    t3, _ = portfolio.allocate(bad, 10_000)
    check("allocate: unpriced names get 0 shares", list(t3["shares"])[1:] == [0, 0])


def test_parse_holdings():
    h, bad = portfolio.parse_holdings("reliance, 10\nTCS 5\nINFY:2\nTCS, 3\nM&M\t4\n# c\nnot a line\nX, -1")
    check("holdings: text formats + summing duplicates", h == {"RELIANCE": 10, "TCS": 8, "INFY": 2, "M&M": 4}, str(h))
    check("holdings: unparsable lines reported", "not a line" in bad and "X, -1" in bad)
    csv = b"Instrument,Qty.,Avg. cost\nHDFCBANK,12,1500\nNIFTYBEES.NS,1,000,250\n"
    try:
        h2, _ = portfolio.parse_holdings(csv_bytes=b"Instrument,Qty.\nHDFCBANK,12\nNIFTYBEES.NS,100\n")
        check("holdings: broker CSV (Instrument / Qty.)", h2 == {"HDFCBANK": 12, "NIFTYBEES": 100}, str(h2))
    except ValueError as exc:
        check("holdings: broker CSV (Instrument / Qty.)", False, str(exc))
    try:
        portfolio.parse_holdings(csv_bytes=b"foo,bar\n1,2\n")
        check("holdings: CSV without symbol/qty columns rejected", False)
    except ValueError:
        check("holdings: CSV without symbol/qty columns rejected", True)
    del csv


def test_rebalance_orders():
    current = {"A": 20, "D": 5, "Z": 3}
    orders, s = portfolio.rebalance_orders(SEL, current, {"D": 100.0})
    o = orders.set_index("Symbol")
    check("orders: exit name sold in full", o.loc["D", "action"] == "SELL" and o.loc["D", "qty"] == 5)
    check("orders: unpriced holding flagged REVIEW", o.loc["Z", "action"] == "REVIEW" and s["n_review"] == 1)
    check("orders: value excludes unpriced holding", math.isclose(s["current_value"], 20 * 1000 + 5 * 100))
    after = {sym: int(o.loc[sym, "target_qty"]) for sym in ["A", "B", "C"]}
    spent = sum(after[k] * p for k, p in zip(["A", "B", "C"], [1000.0, 333.0, 2500.0]))
    check("orders: post-trade cash ≥ 0 and consistent", s["cash_after"] >= 0 and math.isclose(s["cash_after"], s["target_value"] - spent))
    check("orders: sells listed first", orders["action"].iloc[0] == "SELL")
    _, s2 = portfolio.rebalance_orders(SEL, current, {"D": 100.0}, extra_cash=50_000)
    check("orders: extra cash raises target value", math.isclose(s2["target_value"], s["target_value"] + 50_000))
    check("lakh format", portfolio.round_lakh(12345678) == "₹1,23,45,678" and portfolio.round_lakh(float("nan")) == "—")


# --------------------------------------------------------------------------- #
# universe
# --------------------------------------------------------------------------- #
def test_universe_helpers():
    good, bad = universe.parse_symbols("reliance, TCS.NS;  m&m\nbajaj-auto  ???, tcs")
    check("parse_symbols: normalised, deduped", good == ["RELIANCE", "TCS", "M&M", "BAJAJ-AUTO"], str(good))
    check("parse_symbols: junk rejected", bad == ["???"])
    cu = universe.build_custom_universe(["RELIANCE", "NIFTYBEES", "NOTREAL"])
    check("custom: yahoo tickers", list(cu["YFTicker"]) == ["RELIANCE.NS", "NIFTYBEES.NS", "NOTREAL.NS"])
    check("custom: names from bundled lists", cu.loc[0, "Company"].startswith("Reliance") and cu.loc[1, "Company"] == "Nifty 50 ETF")
    check("custom: unknown kept as Unclassified", cu.loc[2, "Industry"] == "Unclassified" and cu.loc[2, "Company"] == "NOTREAL")
    try:
        universe.build_custom_universe([])
        check("custom: empty rejected", False)
    except ValueError:
        check("custom: empty rejected", True)
    etf = universe.fetch_segment(config.ETF_SEGMENT)
    check("etf: curated list loads offline (≥40 ETFs)", len(etf) >= 40 and etf["Segment"].eq("ETF").all())
    check("etf: unique symbols", etf["Symbol"].is_unique)
    check("registry: every index key has a seed CSV",
          all((config.SEED_DIR / f"constituents_{k.lower()}.csv").exists() for k in config.SEGMENT_URLS))
    check("registry: every picker label maps to a known key",
          all(k in config.SEGMENT_URLS for k in config.INDEX_CHOICES.values()))
    check("registry: requested universes present",
          {"NiftyMicrocap250", "NiftyBank", "NiftyFinancialServices", "NiftyTotalMarket"} <= set(config.SEGMENT_URLS))
    check("registry: auto-benchmark targets exist", all(v in config.BENCHMARKS for v in config.UNIVERSE_BENCHMARK.values()))
    check("registry: requested benchmarks present",
          {"Nifty Smallcap 250", "Nifty Midcap 150", "Nifty 100", "Nifty Bank", "Nifty Financial Services"}
          <= set(config.BENCHMARKS))
    with patch("momentum.universe.requests.get", side_effect=RuntimeError("offline")):
        seg = universe.fetch_segment("NiftyFinancialServices", force_refresh=True)
        micro = universe.fetch_segment("Micro", force_refresh=True)
    check("universe: NSE offline → seed fallback for a new index", len(seg) >= 15)
    check("universe: Micro size segment loads (≈250 names)", len(micro) >= 240 and micro["Segment"].eq("Micro").all())
    check("registry: Micro is a size segment benchmarked to Microcap 250",
          "Micro" in config.SIZE_SEGMENTS and config.UNIVERSE_BENCHMARK["Micro"] == "Nifty Microcap 250")
    try:
        universe.build_universe(["Largecap", "NotAThing"])
        check("universe: unknown segment → clear error", False)
    except ValueError as exc:
        check("universe: unknown segment → clear error", "NotAThing" in str(exc))


# --------------------------------------------------------------------------- #
# data — benchmark sources
# --------------------------------------------------------------------------- #
def _series(start, n, level=100.0, g=0.001):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(level * np.exp(g * np.arange(n)), index=idx, name="close")


def test_chain_extend():
    base = _series("2024-01-01", 10, 100.0)
    ext = _series("2024-01-08", 10, 50.0)
    out = data._chain_extend(base, ext)
    check("chain: appends only days after base", out.index[-1] == ext.index[-1] and out.index.is_unique)
    anchor = ext[ext.index <= base.index[-1]].iloc[-1]
    nxt = ext[ext.index > base.index[-1]].iloc[0]
    check("chain: preserves ext returns at the splice",
          math.isclose(out[ext.index[ext.index > base.index[-1]][0]] / base.iloc[-1], nxt / anchor, rel_tol=1e-12))
    check("chain: empty ext → base", data._chain_extend(base, pd.Series(dtype=float)).equals(base))


def test_fetch_benchmark_ladder():
    start, asof = dt.date(2024, 1, 1), dt.date(2024, 3, 29)
    nse = _series("2024-01-01", 64, 1000.0)
    yahoo = _series("2023-12-01", 90, 20.0, g=0.002)
    seed = _series("2024-01-01", 40, 1000.0)

    with patch("momentum.data.nse_index_history", return_value=nse) as m_nse, \
         patch("momentum.data._yahoo_history", return_value=yahoo):
        s = data.fetch_benchmark("^X", "t_ladder1", start, asof, force_refresh=True, nse_name="NIFTY X", tri=True)
    check("ladder: NSE first (TRI flag passed)", s.attrs["source"] == "NSE TRI index" and m_nse.call_args.kwargs["tri"])

    with patch("momentum.data.nse_index_history", side_effect=RuntimeError("blocked")), \
         patch("momentum.data._seed_benchmark", return_value=seed), \
         patch("momentum.data._yahoo_history", return_value=yahoo):
        s2 = data.fetch_benchmark("^X", "t_ladder2", start, asof, force_refresh=True, nse_name="NIFTY X", tri=True)
    check("ladder: NSE blocked → snapshot + Yahoo extension",
          "bundled snapshot" in s2.attrs["source"] and "Yahoo" in s2.attrs["source"] and s2.index[-1] > seed.index[-1])

    with patch("momentum.data.nse_index_history", side_effect=RuntimeError("blocked")), \
         patch("momentum.data._seed_benchmark", return_value=pd.Series(dtype=float)), \
         patch("momentum.data._yahoo_history", return_value=yahoo):
        s3 = data.fetch_benchmark("^X", "t_ladder3", start, asof, force_refresh=True, nse_name="NIFTY X", tri=True)
    check("ladder: no snapshot → Yahoo price index", s3.attrs["source"] == "Yahoo price index")
    check("ladder: clipped to [start, asof]", s3.index[0] >= pd.Timestamp(start) and s3.index[-1] <= pd.Timestamp(asof))

    cached = data.fetch_benchmark("^X", "t_ladder3", start, asof)
    check("ladder: cache round-trip keeps the source", cached.attrs.get("source") == "Yahoo price index"
          and len(cached) == len(s3))

    with patch("momentum.data._yahoo_history", return_value=pd.Series(dtype=float)):
        s4 = data.fetch_benchmark("GOLDBEES.NS", "t_ladder4", start, asof, force_refresh=True)
    check("ladder: nothing anywhere → empty", s4.empty)
    for k in range(1, 5):
        (config.DATA_DIR / f"benchmark_t_ladder{k}.parquet").unlink(missing_ok=True)


def test_nse_index_history_parsing():
    rows = [{"HistoricalDate": "03 Jan 2024", "CLOSE": "101.5"}, {"HistoricalDate": "02 Jan 2024", "CLOSE": "100"},
            {"HistoricalDate": "02 Jan 2024", "CLOSE": "100"}, {"HistoricalDate": "04 Jan 2024", "CLOSE": "-"}]

    class _Resp:
        text = json.dumps(rows)

        def raise_for_status(self):
            return None

    with patch("requests.post", return_value=_Resp()) as m:
        s = data.nse_index_history("NIFTY BANK", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    payload = json.loads(m.call_args.kwargs["data"])
    check("nse: payload is JSON-string-in-JSON with name + dd-Mon-yyyy",
          "'name':'NIFTY BANK'" in payload["cinfo"] and "01-Jan-2024" in payload["cinfo"])
    check("nse: browser User-Agent sent", "Mozilla" in m.call_args.kwargs["headers"]["User-Agent"])
    check("nse: sorted, de-duplicated, non-numeric dropped",
          list(s.values) == [100.0, 101.5] and s.index.is_monotonic_increasing)
    class _TriResp(_Resp):
        text = json.dumps([{"Index Name": "NIFTY BANK", "Date": "02 Jan 2024", "TotalReturnsIndex": "5000.5"}])

    with patch("requests.post", return_value=_TriResp()) as m2:
        tri = data.nse_index_history("NIFTY BANK", dt.date(2020, 1, 1), dt.date(2022, 6, 30), tri=True)
    check("nse: long ranges chunked by year", m2.call_count == 3)
    check("nse: TRI endpoint + field parsed", "getTotalReturnIndexString" in m2.call_args.args[0]
          and list(tri.values) == [5000.5])
    with patch("requests.post", return_value=_Resp()):
        try:
            data.nse_index_history("NIFTY BANK", dt.date(2024, 1, 1), dt.date(2024, 1, 5), tri=True)
            check("nse: unexpected format raises a clear error", False)
        except ValueError:
            check("nse: unexpected format raises a clear error", True)


def test_clean_price_spikes():
    idx = pd.bdate_range("2019-12-02", periods=30)
    gold = pd.Series(np.linspace(33.0, 34.5, 30), index=idx)
    gold.iloc[13:15] = gold.iloc[13:15] / 100.0          # Yahoo's 2-day mis-scaled blip (33.6 → 0.336)
    clean = data.clean_price_spikes(gold)
    check("spikes: 2-day ÷100 blip blanked", clean.iloc[13:15].isna().all() and clean.drop(idx[13:15]).notna().all())
    up = pd.Series(np.linspace(100, 101, 30), index=idx)
    up.iloc[20] = 1000.0                                   # single ×10 print
    check("spikes: single ×10 print blanked", np.isnan(data.clean_price_spikes(up).iloc[20]))
    lvl = pd.Series(np.r_[np.full(15, 300.0), np.full(15, 30.0)], index=idx)  # persistent ÷10 (unadjusted split)
    check("spikes: persistent level change kept", data.clean_price_spikes(lvl).notna().all())
    crash = pd.Series(np.r_[np.full(15, 100.0), 100 * 0.8 ** np.arange(1, 16)], index=idx)  # -20%/day limit-downs
    check("spikes: genuine 20%/day slide kept", data.clean_price_spikes(crash).notna().all())
    frame = pd.DataFrame({"A": gold, "B": lvl})
    cf = data.clean_price_spikes(frame)
    check("spikes: frame cleaned column-wise", int(cf["A"].isna().sum()) == 2 and cf["B"].notna().all())
    check("spikes: empty passthrough", data.clean_price_spikes(pd.DataFrame()).empty)


def test_price_cache_filters_and_cleans():
    """Cached frames come back cleaned and limited to the requested tickers."""
    idx = pd.bdate_range("2024-01-01", periods=40)
    a = pd.Series(np.linspace(100, 110, 40), index=idx)
    a.iloc[20] = a.iloc[20] / 100  # a blip baked into an old cache
    frame = pd.DataFrame({"A.NS": a, "B.NS": 50.0, "GONE.NS": 10.0}, index=idx)
    key = "t_cachefilter"
    for prefix in ("close", "rawclose", "volume"):
        frame.to_parquet(config.DATA_DIR / f"{prefix}_{key}.parquet")
    c, rc, v = data.fetch_prices(["A.NS", "B.NS", "NEW.NS"], key, asof=idx[-1].date(), cache_hours=1)
    check("cache: only requested tickers returned (stale names dropped)", list(c.columns) == ["A.NS", "B.NS"]
          and list(rc.columns) == ["A.NS", "B.NS"] and list(v.columns) == ["A.NS", "B.NS"])
    check("cache: blips cleaned on read", np.isnan(c["A.NS"].iloc[20]) and np.isnan(rc["A.NS"].iloc[20]))
    for prefix in ("close", "rawclose", "volume"):
        (config.DATA_DIR / f"{prefix}_{key}.parquet").unlink(missing_ok=True)


def test_portfolio_edge_cases():
    sel = SEL.assign(close_price=[1000.0, 333.0, np.nan])  # C unpriced
    t, _ = portfolio.allocate(sel, 100_000)
    check("allocate: unpriced target's weight redistributed (≥95% invested)", t["invested"].sum() >= 95_000,
          str(t["invested"].sum()))
    orders, s = portfolio.rebalance_orders(sel, {"C": 10, "A": 5}, {})
    o = orders.set_index("Symbol")
    check("orders: held + targeted but unpriced → REVIEW (not SELL)", o.loc["C", "action"] == "REVIEW"
          and o.loc["C", "qty"] == 0)
    o2, s2 = portfolio.rebalance_orders(SEL, {"A": 2}, {}, extra_cash=-5_000)
    check("orders: over-withdrawal clamps to sell-all + reports shortfall",
          s2["target_value"] == 0 and math.isclose(s2["withdrawal_shortfall"], 3_000)
          and o2.set_index("Symbol").loc["A", "action"] == "SELL" and s2["cash_after"] >= 0)


def test_benchmark_snapshot_file():
    ok = data.BENCHMARK_SEED_PATH.exists()
    check("snapshot: bundled benchmarks.parquet exists", ok)
    if ok:
        cols = pd.read_parquet(data.BENCHMARK_SEED_PATH).columns
        need = [f"{spec['nse']}|{k}" for spec in config.BENCHMARKS.values() if spec.get("nse") for k in ("TRI", "PRICE")]
        missing = [c for c in need if c not in cols]
        check("snapshot: TRI + price for every NSE benchmark", not missing, str(missing[:5]))


if __name__ == "__main__":
    test_calendar_and_annual()
    test_rolling_and_summary()
    test_drawdown_episodes()
    test_xirr_and_sip()
    test_sector_and_contributions()
    test_excel_report()
    test_allocate()
    test_parse_holdings()
    test_rebalance_orders()
    test_universe_helpers()
    test_chain_extend()
    test_fetch_benchmark_ladder()
    test_nse_index_history_parsing()
    test_clean_price_spikes()
    test_price_cache_filters_and_cleans()
    test_portfolio_edge_cases()
    test_benchmark_snapshot_file()
    print("-" * 50)
    if _failures:
        print(f"{_failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed.")
