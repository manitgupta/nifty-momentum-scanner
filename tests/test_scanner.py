"""Unit + integration checks for the scanner maths.

Run with:  ./venv/bin/python tests/test_scanner.py
(kept dependency-free of pytest so it runs against the project venv directly).
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from momentum import scanner
from momentum.data import PriceData

PASS, FAIL = "PASS", "FAIL"
_failures = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _failures
    print(f"[{PASS if cond else FAIL}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        _failures += 1


# --------------------------------------------------------------------------- #
def test_asof_price():
    idx = pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-10"])
    s = pd.Series([100.0, 110.0, 121.0], index=idx)
    check("asof_price exact date", scanner._asof_price(s, dt.date(2026, 1, 5)) == 110.0)
    check("asof_price between dates -> last prior", scanner._asof_price(s, dt.date(2026, 1, 7)) == 110.0)
    check("asof_price after last -> last", scanner._asof_price(s, dt.date(2026, 2, 1)) == 121.0)
    check("asof_price before first -> nan", math.isnan(scanner._asof_price(s, dt.date(2025, 1, 1))))


# --------------------------------------------------------------------------- #
def test_score_piecewise():
    """Hand-computed Z -> WAZ -> score for a 3-stock eligible universe."""
    df = pd.DataFrame(
        {
            "mr_12": [1.0, 2.0, 3.0],
            "mr_6": [1.0, 2.0, 3.0],
            "eligible": [True, True, True],
        },
        index=["LOW", "MID", "HIGH"],
    )
    params = scanner.ScanParams(asof=dt.date(2026, 8, 15), zscore_ddof=0)
    out = scanner.normalize_and_score(df, params)

    # mean=2, pop std=sqrt(2/3)=0.81650; z_high=+1.22474, z_low=-1.22474
    z = 1.0 / math.sqrt(2.0 / 3.0)
    check("z_12 HIGH", math.isclose(out.loc["HIGH", "z_12"], z, rel_tol=1e-6))
    check("waz HIGH", math.isclose(out.loc["HIGH", "waz"], z, rel_tol=1e-6))
    # HIGH: WAZ>=0 -> 1+WAZ
    check("score HIGH (1+WAZ)", math.isclose(out.loc["HIGH", "score"], 1.0 + z, rel_tol=1e-6))
    # MID: WAZ=0 -> score 1
    check("score MID == 1", math.isclose(out.loc["MID", "score"], 1.0, rel_tol=1e-9))
    # LOW: WAZ<0 -> 1/(1-WAZ)
    check("score LOW 1/(1-WAZ)", math.isclose(out.loc["LOW", "score"], 1.0 / (1.0 + z), rel_tol=1e-6))
    # ranking
    check("rank HIGH == 1", out.loc["HIGH", "rank"] == 1)
    check("rank LOW == 3", out.loc["LOW", "rank"] == 3)


# --------------------------------------------------------------------------- #
def test_iterative_cap():
    # One dominant stock, cap 40%: excess redistributes evenly to the other three.
    raw = pd.Series([10.0, 1.0, 1.0, 1.0], index=list("ABCD"))
    cap = pd.Series([0.4, 0.4, 0.4, 0.4], index=list("ABCD"))
    w = scanner._iterative_cap(raw, cap, 100)
    check("cap: sums to 1", math.isclose(w.sum(), 1.0, rel_tol=1e-9))
    check("cap: none exceed cap", bool((w <= cap + 1e-9).all()))
    check("cap: A capped at 0.4", math.isclose(w["A"], 0.4, rel_tol=1e-9))
    check("cap: B/C/D == 0.2 each", all(math.isclose(w[k], 0.2, rel_tol=1e-9) for k in "BCD"))

    # Cascade: two dominant stocks both cap.
    raw2 = pd.Series([10.0, 10.0, 1.0], index=list("XYZ"))
    cap2 = pd.Series([0.4, 0.4, 0.4], index=list("XYZ"))
    w2 = scanner._iterative_cap(raw2, cap2, 100)
    check("cap cascade: sums to 1", math.isclose(w2.sum(), 1.0, rel_tol=1e-9))
    check("cap cascade: X,Y at 0.4", math.isclose(w2["X"], 0.4) and math.isclose(w2["Y"], 0.4))
    check("cap cascade: Z == 0.2", math.isclose(w2["Z"], 0.2, rel_tol=1e-9))


# --------------------------------------------------------------------------- #
def test_asof_closes():
    """Per-column as-of close: forward-fills gaps, honours the as-of cutoff."""
    idx = pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-10"])
    frame = pd.DataFrame(
        {"A.NS": [100.0, 110.0, 121.0], "B.NS": [50.0, np.nan, 55.0]}, index=idx
    )
    s = scanner._asof_closes(frame, dt.date(2026, 1, 7))
    check("asof_closes A -> last prior", s["A.NS"] == 110.0)
    check("asof_closes B -> ffill over NaN", s["B.NS"] == 50.0)
    empty = scanner._asof_closes(pd.DataFrame(), dt.date(2026, 1, 7))
    check("asof_closes empty -> empty", empty.empty)


# --------------------------------------------------------------------------- #
def test_skip_month_momentum():
    """skip_months shifts the return end-point back; lookback months are configurable."""
    asof = dt.date(2026, 6, 30)
    idx = pd.bdate_range(end=pd.Timestamp(asof), periods=400)
    # Ramp up then fall in the final ~month, so skipping that month raises the return.
    vals = np.linspace(100.0, 200.0, len(idx))
    vals[-21:] = np.linspace(200.0, 150.0, 21)  # recent drawdown
    close = pd.DataFrame({"X.NS": vals}, index=idx)

    p_no_skip = scanner.compute_momentum(
        close, scanner.ScanParams(asof=asof, skip_months=0, lookback_long_months=12,
                                  lookback_short_months=6)
    )
    p_skip = scanner.compute_momentum(
        close, scanner.ScanParams(asof=asof, skip_months=1, lookback_long_months=12,
                                  lookback_short_months=6)
    )
    check("skip-month raises 12M return after recent drawdown",
          p_skip.loc["X.NS", "ret_12m"] > p_no_skip.loc["X.NS", "ret_12m"],
          f"skip={p_skip.loc['X.NS','ret_12m']:.4f} vs no-skip={p_no_skip.loc['X.NS','ret_12m']:.4f}")

    # Shorter lookback -> smaller ramp captured -> smaller 12M-leg return here.
    p_short_lb = scanner.compute_momentum(
        close, scanner.ScanParams(asof=asof, skip_months=0, lookback_long_months=6,
                                  lookback_short_months=3)
    )
    check("configurable long lookback changes the return leg",
          not math.isclose(p_short_lb.loc["X.NS", "ret_12m"],
                           p_no_skip.loc["X.NS", "ret_12m"], rel_tol=1e-6))


# --------------------------------------------------------------------------- #
def test_close_price_column():
    """run_scan attaches raw close_price; adjusted close drives returns."""
    asof = dt.date(2026, 8, 14)
    idx = pd.bdate_range(end=pd.Timestamp(asof), periods=300)
    adj = pd.DataFrame({"A.NS": np.linspace(100, 200, len(idx))}, index=idx)
    raw = adj * 1.10  # raw close deliberately differs from adjusted
    volume = pd.DataFrame(1_000_000, index=idx, columns=adj.columns)
    market = pd.DataFrame(
        {"market_cap": [5e11], "shares": [1e9], "last_price": [1.0]}, index=adj.columns
    )
    pdata = PriceData(close=adj, raw_close=raw, volume=volume, market=market, failed=[])
    universe = pd.DataFrame(
        {"Symbol": ["A"], "Company": ["A Co"], "Industry": ["X"],
         "Segment": ["Largecap"], "YFTicker": ["A.NS"]}
    )
    params = scanner.ScanParams(
        asof=asof, top_n=1, apply_listing_filter=False,
        apply_liquidity_filter=False, apply_turnover_ratio_filter=False,
    )
    res = scanner.run_scan(universe, pdata, params)
    check("close_price column present", "close_price" in res.results.columns)
    check("close_price uses RAW close (last value)",
          math.isclose(res.results.loc["A.NS", "close_price"], float(raw["A.NS"].iloc[-1]), rel_tol=1e-9))

    # Without a raw_close frame, it falls back to the adjusted close.
    pdata2 = PriceData(close=adj, volume=volume, market=market, failed=[])
    res2 = scanner.run_scan(universe, pdata2, params)
    check("close_price falls back to adjusted when raw absent",
          math.isclose(res2.results.loc["A.NS", "close_price"], float(adj["A.NS"].iloc[-1]), rel_tol=1e-9))


# --------------------------------------------------------------------------- #
def test_integration_ranking():
    """Synthetic prices: strong drift should out-rank flat should out-rank weak."""
    asof = dt.date(2026, 8, 14)
    idx = pd.bdate_range(end=pd.Timestamp(asof), periods=300)
    rng = np.random.default_rng(42)

    def make(drift: float) -> np.ndarray:
        daily = drift + rng.normal(0, 0.01, len(idx))
        return 100.0 * np.exp(np.cumsum(daily))

    close = pd.DataFrame(
        {"STRONG.NS": make(0.003), "MID.NS": make(0.0005), "WEAK.NS": make(-0.002)},
        index=idx,
    )
    volume = pd.DataFrame(1_000_000, index=idx, columns=close.columns)
    market = pd.DataFrame(
        {"market_cap": [5e11, 5e11, 5e11], "shares": [1e9, 1e9, 1e9], "last_price": [1.0, 1.0, 1.0]},
        index=close.columns,
    )
    pdata = PriceData(close=close, volume=volume, market=market, failed=[])

    params = scanner.ScanParams(
        asof=asof,
        top_n=3,
        apply_listing_filter=False,
        apply_liquidity_filter=False,
        apply_turnover_ratio_filter=False,
    )
    universe = pd.DataFrame(
        {
            "Symbol": ["STRONG", "MID", "WEAK"],
            "Company": ["Strong Co", "Mid Co", "Weak Co"],
            "Industry": ["X", "Y", "Z"],
            "Segment": ["Largecap"] * 3,
            "YFTicker": list(close.columns),
        }
    )
    res = scanner.run_scan(universe, pdata, params)

    order = list(res.selected.index)
    check("integration: STRONG ranked first", order[0] == "STRONG.NS", f"order={order}")
    check("integration: WEAK ranked last", order[-1] == "WEAK.NS", f"order={order}")
    check("integration: weights sum to ~1", math.isclose(res.selected["weight"].sum(), 1.0, rel_tol=1e-6))
    check("integration: 3 eligible", res.eligible_count == 3)
    check("integration: positive 12M return for STRONG", res.results.loc["STRONG.NS", "ret_12m"] > 0)
    check("integration: sigma finite", np.isfinite(res.results.loc["STRONG.NS", "sigma"]))


if __name__ == "__main__":
    test_asof_price()
    test_asof_closes()
    test_score_piecewise()
    test_skip_month_momentum()
    test_close_price_column()
    test_iterative_cap()
    test_integration_ranking()
    print("-" * 50)
    if _failures:
        print(f"{_failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed.")
