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


# --------------------------------------------------------------------------- #
def _reference_momentum(close: pd.DataFrame, params) -> pd.DataFrame:
    """The original per-ticker loop, kept verbatim as an oracle for the vectorised version."""
    from dateutil.relativedelta import relativedelta
    asof = params.asof
    end = asof - relativedelta(months=params.skip_months)
    date_6m = asof - relativedelta(months=params.lookback_short_months)
    date_12m = asof - relativedelta(months=params.lookback_long_months)
    vol_start = asof - relativedelta(months=params.vol_lookback_months)
    rows = []
    for ticker in close.columns:
        series = close[ticker].dropna()
        if series.empty:
            continue
        p0 = scanner._asof_price(series, end)
        p6 = scanner._asof_price(series, date_6m)
        p12 = scanner._asof_price(series, date_12m)
        ret_6m = p0 / p6 - 1.0 if (p6 and not np.isnan(p6)) else np.nan
        ret_12m = p0 / p12 - 1.0 if (p12 and not np.isnan(p12)) else np.nan
        window = series[(series.index > pd.Timestamp(vol_start)) & (series.index <= pd.Timestamp(asof))]
        log_ret = np.log(window / window.shift(1)).dropna()
        sigma = float(log_ret.std(ddof=params.vol_ddof) * np.sqrt(params.trading_days)) if len(log_ret) >= 2 else np.nan
        ok = sigma and not np.isnan(sigma) and sigma > 0
        rows.append({"YFTicker": ticker, "ret_6m": ret_6m, "ret_12m": ret_12m, "sigma": sigma,
                     "mr_6": ret_6m / sigma if ok else np.nan, "mr_12": ret_12m / sigma if ok else np.nan,
                     "n_obs": int(series[series.index <= pd.Timestamp(asof)].shape[0]),
                     "first_date": series.index.min(), "last_date": series.index.max()})
    return pd.DataFrame(rows).set_index("YFTicker")


def test_vectorised_momentum_matches_reference():
    """The vectorised compute_momentum must reproduce the per-ticker loop exactly,
    including NaN gaps, late listings, halted names and an all-NaN ticker."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2021-01-01", "2026-06-30")
    n = 40
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, (len(idx), n)), axis=0)),
                         index=idx, columns=[f"T{i:02d}.NS" for i in range(n)])
    for j in range(0, n, 5):                        # late listings
        close.iloc[: rng.integers(50, 1200), j] = np.nan
    for j in range(1, n, 7):                        # random gaps / halts
        close.iloc[rng.choice(len(idx), 60, replace=False), j] = np.nan
    close.iloc[-40:, 3] = np.nan                    # stopped trading before as-of
    close["DEAD.NS"] = np.nan                       # never traded
    close["NEWBIE.NS"] = np.nan
    close.loc[idx[-30]:, "NEWBIE.NS"] = 50.0        # listed after an early as-of
    ok_all = True
    for asof, skip, ddof in ((dt.date(2026, 6, 30), 0, 1), (dt.date(2025, 3, 15), 1, 0), (dt.date(2023, 8, 1), 2, 1)):
        p = scanner.ScanParams(asof=asof, skip_months=skip, vol_ddof=ddof)
        new = scanner.compute_momentum(close, p)
        ref = _reference_momentum(close, p)
        same_index = list(new.index) == list(ref.index)
        num = ["ret_6m", "ret_12m", "sigma", "mr_6", "mr_12"]
        diff = np.nanmax(np.abs(new.loc[ref.index, num].to_numpy(float) - ref[num].to_numpy(float)))
        nan_same = bool((new.loc[ref.index, num].isna().to_numpy() == ref[num].isna().to_numpy()).all())
        ints_same = (new.loc[ref.index, "n_obs"].tolist() == ref["n_obs"].tolist()
                     and new.loc[ref.index, "first_date"].tolist() == ref["first_date"].tolist()
                     and new.loc[ref.index, "last_date"].tolist() == ref["last_date"].tolist())
        ok = same_index and diff < 1e-10 and nan_same and ints_same
        ok_all &= ok
        check(f"vectorised: matches reference loop (asof={asof}, skip={skip}, ddof={ddof})", ok,
              f"index={same_index} maxdiff={diff:.2e} nan={nan_same} ints={ints_same}")
    check("vectorised: all-NaN ticker skipped", "DEAD.NS" not in scanner.compute_momentum(
        close, scanner.ScanParams(asof=dt.date(2026, 6, 30))).index)
    empty = scanner.compute_momentum(pd.DataFrame(), scanner.ScanParams(asof=dt.date(2026, 6, 30)))
    check("vectorised: empty frame -> empty result with columns", empty.empty and "sigma" in empty.columns)


def test_select_top_sector_cap_and_buffer():
    ranked = pd.DataFrame({
        "score": [9, 8, 7, 6, 5, 4, 3],
        "Industry": ["Bank", "Bank", "Bank", "IT", "IT", "Pharma", None],
    }, index=list("ABCDEFG"))
    check("select_top: plain top-3 by score", scanner.select_top(ranked, 3) == ["A", "B", "C"])
    check("select_top: sector cap 1 skips over-represented industries",
          scanner.select_top(ranked, 3, max_per_sector=1) == ["A", "D", "F"])
    check("select_top: cap 2 + NaN industry is its own bucket",
          scanner.select_top(ranked, 6, max_per_sector=2) == ["A", "B", "D", "E", "F", "G"])
    # Buffer: currently hold F (rank 6) and G (rank 7); exit rank 6 keeps F only.
    check("select_top: buffer keeps held names within exit rank",
          scanner.select_top(ranked, 3, keep=["F", "G"], exit_rank=6) == ["A", "B", "F"])
    check("select_top: buffer off -> pure top-N",
          scanner.select_top(ranked, 3, keep=["F", "G"], exit_rank=None) == ["A", "B", "C"])
    check("select_top: retained names still respect top_n",
          scanner.select_top(ranked, 2, keep=list("CDEF"), exit_rank=7) == ["C", "D"])
    check("select_top: empty / zero", scanner.select_top(ranked.iloc[:0], 3) == [] and scanner.select_top(ranked, 0) == [])


def test_scheme_weights_and_weighting_modes():
    sel = pd.DataFrame({"score": [3.0, 1.0], "sigma": [0.2, 0.4]}, index=["A", "B"])
    check("scheme: score ∝ score", math.isclose(scanner.scheme_weights(sel, "score")["A"], 0.75))
    check("scheme: equal", math.isclose(scanner.scheme_weights(sel, "equal")["B"], 0.5))
    iv = scanner.scheme_weights(sel, "inverse_vol")
    check("scheme: inverse-vol ∝ 1/σ (2:1)", math.isclose(iv["A"], 2 / 3) and math.isclose(iv.sum(), 1.0))
    bad = pd.DataFrame({"score": [1.0, 1.0], "sigma": [np.nan, 0.0]}, index=["A", "B"])
    check("scheme: inverse-vol with no usable σ -> equal", math.isclose(scanner.scheme_weights(bad, "inverse_vol")["A"], 0.5))

    df = pd.DataFrame({
        "score": [3.0, 2.0, 1.0, np.nan], "sigma": [0.2, 0.3, 0.4, 0.1], "ff_mcap": [np.nan] * 4,
        "Industry": ["X", "X", "Y", "Y"],
    }, index=["A", "B", "C", "D"])
    for scheme in ("score", "equal", "inverse_vol"):
        p = scanner.ScanParams(asof=dt.date(2026, 1, 1), top_n=3, weighting=scheme)
        out = scanner.select_and_weight(df, p)
        check(f"select_and_weight[{scheme}]: weights sum to 1 with no market caps",
              math.isclose(out["weight"].sum(), 1.0) and list(out.index) == ["A", "B", "C"])
    p = scanner.ScanParams(asof=dt.date(2026, 1, 1), top_n=3, weighting="equal", max_per_sector=1)
    out = scanner.select_and_weight(df, p)
    check("select_and_weight: sector cap applies in the scanner", list(out.index) == ["A", "C"])
    out = scanner.select_and_weight(df, scanner.ScanParams(asof=dt.date(2026, 1, 1), top_n=1, weighting="equal"),
                                    keep=["C"], exit_rank=3)
    check("select_and_weight: live rank buffer keeps a held name", list(out.index) == ["C"])


if __name__ == "__main__":
    test_asof_price()
    test_asof_closes()
    test_score_piecewise()
    test_skip_month_momentum()
    test_close_price_column()
    test_iterative_cap()
    test_integration_ranking()
    test_vectorised_momentum_matches_reference()
    test_select_top_sector_cap_and_buffer()
    test_scheme_weights_and_weighting_modes()
    print("-" * 50)
    if _failures:
        print(f"{_failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed.")
