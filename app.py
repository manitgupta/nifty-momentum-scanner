"""Streamlit UI for the Nifty500 Momentum 50 scanner + backtester.

Two tabs share one sidebar configuration:
    🔎 Scanner  — a point-in-time momentum scan as of any date.
    🧪 Backtest — a rebalancing backtest of the same strategy over a period.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

from momentum import backtest, config, data, scanner, universe

st.set_page_config(page_title="Nifty500 Momentum 50 Scanner", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Cached universe fetch (constituent lists are small and disk-cached anyway)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=config.UNIVERSE_CACHE_HOURS * 3600, show_spinner=False)
def load_universe(segments: tuple[str, ...], force: bool) -> pd.DataFrame:
    return universe.build_universe(list(segments), force_refresh=force)


# --------------------------------------------------------------------------- #
# Sidebar — configuration (shared by both tabs)
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Universe")
uni_mode = st.sidebar.radio(
    "Universe source",
    ["Size segments", "Whole index"],
    horizontal=True,
    key="uni_mode",
    help="Size segments: compose from Nifty 100 / Midcap 150 / Smallcap 250 (their "
    "union ≈ Nifty 500). Whole index: use one published NSE list directly.",
)
if uni_mode == "Size segments":
    st.sidebar.caption("Segments map to NSE size buckets (their union ≈ Nifty 500).")
    seg_large = st.sidebar.checkbox("Largecap · Nifty 100", value=True, key="seg_large")
    seg_mid = st.sidebar.checkbox("Midcap · Nifty Midcap 150", value=False, key="seg_mid")
    seg_small = st.sidebar.checkbox("Small · Nifty Smallcap 250", value=False, key="seg_small")
    selected_segments = [
        s for s, on in [("Largecap", seg_large), ("Midcap", seg_mid), ("Small", seg_small)] if on
    ]
else:
    idx_choice = st.sidebar.selectbox(
        "Index universe",
        ["Nifty 500", "Nifty Total Market (~750)"],
        help="Nifty Total Market ≈ Nifty 500 + Microcap 250. Larger universes take "
        "longer to fetch.",
    )
    selected_segments = ["Nifty500"] if idx_choice.startswith("Nifty 500") else ["NiftyTotalMarket"]

st.sidebar.subheader("As-of date")
asof = st.sidebar.date_input(
    "Scan as-of / backtest end (last trading day on/before):",
    value=dt.date.today(),
    max_value=dt.date.today(),
)

st.sidebar.subheader("Eligibility filters")
apply_listing = st.sidebar.checkbox("Minimum 1-year listing history", value=True)
apply_liquidity = st.sidebar.checkbox("Exclude bottom-10% by 6M avg daily turnover", value=True)
apply_turnover_ratio = st.sidebar.checkbox("Exclude bottom-10% by turnover ratio", value=True)
# Methodology filters that Yahoo/NSE cannot reliably supply — shown, disabled.
st.sidebar.checkbox(
    "Pledged promoter shares ≤ 20%", value=False, disabled=True,
    help="Part of the official methodology, but this data is not available via Yahoo Finance / NSE bulk feeds.",
)
st.sidebar.checkbox(
    "Circuit / price-band hit rule", value=False, disabled=True,
    help="Requires per-day upper/lower circuit flags over 6 months — not reliably sourceable here.",
)
st.sidebar.caption("⚠️ The last two are greyed out: data unavailable (see README → Limitations).")

with st.sidebar.expander("Advanced parameters"):
    top_n = st.number_input("Number of stocks to select (Top N)", 5, 200, config.TOP_N, step=5)

    st.markdown("**Momentum windows**")
    lb_long = st.number_input(
        "Long lookback (months)", 3, 36, config.MONTHS_12, step=1,
        help="Return leg 1. Methodology default: 12 months.",
    )
    lb_short = st.number_input(
        "Short lookback (months)", 1, 24, config.MONTHS_6, step=1,
        help="Return leg 2. Methodology default: 6 months.",
    )
    skip_m = st.number_input(
        "Skip recent months (12-1 style)", 0, 6, config.SKIP_MONTHS, step=1,
        help="Exclude the most recent N months from both legs to avoid short-term "
        "reversal. 0 = methodology default.",
    )
    vol_m = st.number_input(
        "Volatility window (months)", 3, 24, config.VOL_LOOKBACK_MONTHS, step=1,
        help="Trailing window for annualised volatility σₚ. Methodology: 12 months.",
    )

    st.markdown("**Weighting & filters**")
    w12 = st.slider("Weight on long-leg Z-score (%)", 0, 100, int(config.WEIGHT_12M * 100), step=5)
    w6 = 100 - w12
    st.caption(f"Short-leg Z-score weight = {w6}%")
    liq_pct = st.slider("Liquidity bottom-percentile cut", 0, 25, config.LIQUIDITY_BOTTOM_PERCENTILE)
    cap_abs = st.slider("Absolute weight cap (%)", 1, 20, int(config.CAP_ABSOLUTE * 100))
    cap_rel = st.number_input("Relative cap (× ff-mcap weight)", 1.0, 20.0, config.CAP_RELATIVE, step=0.5)
    ff_factor = st.slider(
        "Free-float factor (× market cap)", 0.1, 1.0, 1.0, step=0.05,
        help="Yahoo gives full market cap; scale it here to approximate free-float if desired.",
    )

force_refresh = st.sidebar.checkbox("Force refresh (ignore cache)", value=False)
run = st.sidebar.button("🚀 Run scan", type="primary", width="stretch", key="run_scan")


def _base_scan_params(asof_date: dt.date) -> scanner.ScanParams:
    """Build a ScanParams from the shared sidebar controls (selection knobs)."""
    return scanner.ScanParams(
        asof=asof_date,
        top_n=int(top_n),
        weight_12m=w12 / 100.0,
        weight_6m=w6 / 100.0,
        lookback_long_months=int(lb_long),
        lookback_short_months=int(lb_short),
        skip_months=int(skip_m),
        vol_lookback_months=int(vol_m),
        liquidity_bottom_pct=liq_pct,
        apply_listing_filter=apply_listing,
        apply_liquidity_filter=apply_liquidity,
        apply_turnover_ratio_filter=apply_turnover_ratio,
        cap_absolute=cap_abs / 100.0,
        cap_relative=cap_rel,
        free_float_factor=ff_factor,
    )


def _history_start(end_date: dt.date, extra_years: int = 0) -> dt.date:
    """Earliest fetch date covering the longest configured window (+ backtest span)."""
    span_months = max(int(lb_long), int(lb_short), int(vol_m)) + int(skip_m)
    return (
        end_date
        - relativedelta(years=extra_years)
        - relativedelta(months=span_months)
        - dt.timedelta(days=45)
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📈 Nifty500 Momentum 50 — Scanner & Backtester")
st.markdown(
    "Applies the **Nifty500 Momentum 50** construction rules (NSE methodology §20): "
    "volatility-adjusted 6M & 12M momentum → cross-sectional Z-scores → Normalized "
    "Momentum Score → top-N selection. The **Scanner** ranks as of a date; the "
    "**Backtest** rebalances the strategy over a period."
)


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #
def _run_scan():
    if not selected_segments:
        st.error("Select at least one universe segment (Largecap / Midcap / Small).")
        return None

    with st.status("Fetching data & scanning…", expanded=True) as status:
        st.write(f"Loading universe: {', '.join(selected_segments)}")
        uni = load_universe(tuple(selected_segments), force_refresh)
        tickers = uni["YFTicker"].tolist()
        st.write(f"Universe size: **{len(tickers)}** stocks")

        start = _history_start(asof)
        cache_key = (
            f"{'_'.join(sorted(selected_segments)).lower()}"
            f"_{start.isoformat()}_{asof.isoformat()}"
        )

        pbar = st.progress(0.0, text="Downloading price history…")
        def price_prog(done, total):
            pbar.progress(min(done / total, 1.0), text=f"Price history: {done}/{total}")

        mbar = st.progress(0.0, text="Fetching market caps…")
        def market_prog(done, total):
            mbar.progress(min(done / total, 1.0), text=f"Market caps: {done}/{total}")

        pdata = data.fetch_all(
            tickers, cache_key=cache_key, asof=asof, start=start,
            force_refresh=force_refresh, price_progress=price_prog, market_progress=market_prog,
        )

        st.write("Computing momentum, eligibility, scores & weights…")
        params = _base_scan_params(asof)
        result = scanner.run_scan(uni, pdata, params)
        status.update(label="Scan complete ✅", state="complete", expanded=False)

    return {"result": result, "universe_size": len(tickers), "asof": asof,
            "segments": selected_segments, "top_n": int(top_n),
            "lb_long": int(lb_long), "lb_short": int(lb_short), "skip_m": int(skip_m)}


# --------------------------------------------------------------------------- #
# Display helpers (scanner)
# --------------------------------------------------------------------------- #
def _display_cols(lb_short: int, lb_long: int) -> dict[str, str]:
    """Column key -> label, with the return/Z legs labelled by their actual months."""
    return {
        "selection_rank": "Rank",
        "Symbol": "Symbol",
        "Company": "Company",
        "Segment": "Segment",
        "Industry": "Industry",
        "close_price": "Close (₹)",
        "ret_6m": f"{lb_short}M Return",
        "ret_12m": f"{lb_long}M Return",
        "sigma": "Volatility (σ)",
        "z_6": f"Z {lb_short}M",
        "z_12": f"Z {lb_long}M",
        "score": "Momentum Score",
        "weight": "Weight",
    }


def _format_selected(selected: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    have = [c for c in cols if c in selected.reset_index().columns]
    df = selected.reset_index()[have].rename(columns={k: cols[k] for k in have})
    ret_s, ret_l = cols["ret_6m"], cols["ret_12m"]
    for c in [ret_s, ret_l, "Volatility (σ)", "Weight"]:
        if c in df:
            df[c] = (df[c] * 100).round(2)
    for c in [cols["z_6"], cols["z_12"], "Momentum Score"]:
        if c in df:
            df[c] = df[c].round(3)
    if "Close (₹)" in df:
        df["Close (₹)"] = df["Close (₹)"].round(2)
    return df


PCT_FMT = st.column_config.NumberColumn(format="%.2f%%")


def _colcfg(cols: dict[str, str]) -> dict:
    return {
        cols["ret_6m"]: PCT_FMT, cols["ret_12m"]: PCT_FMT,
        "Volatility (σ)": PCT_FMT, "Weight": PCT_FMT,
        "Close (₹)": st.column_config.NumberColumn(format="%.2f"),
        "Rank": st.column_config.NumberColumn(format="%d"),
    }


def render_scan_tab() -> None:
    # Run inside the tab so status/progress/errors render here, not above the tab bar.
    if run:
        st.session_state["scan"] = _run_scan()

    scan = st.session_state.get("scan")
    if not scan:
        st.info("Configure the scan in the sidebar and press **Run scan**.")
        return

    result: scanner.ScanResult = scan["result"]
    if result is None:
        return

    selected = result.selected
    full = result.results

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe", scan["universe_size"])
    c2.metric("Eligible", result.eligible_count)
    c3.metric("Selected", len(selected))
    c4.metric("As-of", scan["asof"].isoformat())

    for note in result.notes:
        st.warning(note)

    if result.exclusions:
        with st.expander(f"Eligibility exclusions ({sum(result.exclusions.values())} stock-flags)"):
            exdf = pd.DataFrame(
                sorted(result.exclusions.items(), key=lambda kv: -kv[1]), columns=["Reason", "Count"]
            )
            st.dataframe(exdf, hide_index=True, width="stretch")

    if selected.empty:
        st.error("No stocks were selected. Try enabling more segments or relaxing filters.")
        return

    cols = _display_cols(scan["lb_short"], scan["lb_long"])
    skip_note = f" · skip {scan['skip_m']}M" if scan.get("skip_m") else ""
    st.subheader(f"Top {len(selected)} by Normalized Momentum Score")
    st.caption(
        f"Momentum legs: {scan['lb_short']}M & {scan['lb_long']}M lookback{skip_note}. "
        "Close (₹) is the actual traded close on the as-of date; returns use adjusted close."
    )
    disp = _format_selected(selected, cols)
    st.dataframe(disp, hide_index=True, width="stretch", column_config=_colcfg(cols))

    st.download_button(
        "⬇️ Download selection (CSV)",
        disp.to_csv(index=False).encode(),
        file_name=f"momentum50_{'_'.join(scan['segments'])}_{scan['asof'].isoformat()}.csv",
        mime="text/csv",
    )

    st.subheader("Portfolio weights")
    wchart = disp[["Symbol", "Weight"]].set_index("Symbol")
    st.bar_chart(wchart, height=340)

    with st.expander("Full universe detail (all stocks, with eligibility & scores)"):
        ls, ll = scan["lb_short"], scan["lb_long"]
        audit_cols = {
            "Symbol": "Symbol", "Company": "Company", "Segment": "Segment",
            "close_price": "Close (₹)",
            "ret_6m": f"{ls}M Return", "ret_12m": f"{ll}M Return", "sigma": "Volatility (σ)",
            "mr_6": f"MR {ls}M", "mr_12": f"MR {ll}M", "z_6": f"Z {ls}M", "z_12": f"Z {ll}M",
            "score": "Momentum Score", "eligible": "Eligible", "exclusion_reasons": "Exclusion reasons",
        }
        have = [c for c in audit_cols if c in full.columns]
        fdf = full.reset_index()[["YFTicker"] + have].rename(columns=audit_cols)
        for c in [f"{ls}M Return", f"{ll}M Return", "Volatility (σ)"]:
            if c in fdf:
                fdf[c] = (fdf[c] * 100).round(2)
        if "Close (₹)" in fdf:
            fdf["Close (₹)"] = fdf["Close (₹)"].round(2)
        st.dataframe(fdf, hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Download full universe detail (CSV)",
            fdf.to_csv(index=False).encode(),
            file_name=f"momentum_universe_{scan['asof'].isoformat()}.csv",
            mime="text/csv",
        )

    st.caption(
        "Educational tool — not investment advice. Weights use full market cap as a free-float proxy; "
        "pledge & circuit eligibility rules are omitted (data unavailable). See README for full methodology mapping."
    )


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
FREQ_MONTHS = {"Monthly": 1, "Quarterly": 3, "Semi-annual": 6, "Annual": 12}

BACKTEST_DISCLAIMER = (
    "⚠️ **Backtest limitations — read first.** These results are **educational, not "
    "investment advice**, and are structurally *optimistic*:\n"
    "- **Survivorship bias** — the universe is *today's* index constituents, so stocks "
    "later delisted or demoted are silently excluded from history.\n"
    "- **No look-ahead weighting** — weights are **score- or equal-weighted only**. "
    "Live market-cap capping is *not* used (market cap is a current-only snapshot; "
    "applying it to the past would be look-ahead). The turnover-ratio eligibility "
    "filter is disabled for the same reason.\n"
    "- **Costs & liquidity** — modelled only via the optional bps cost input; taxes, "
    "market impact, slippage and capacity limits are not modelled.\n"
    "- **Point-in-time index membership** is not available from free data.\n\n"
    "The momentum *ranking* is price-only and unbiased; the *absolute-return* figures are indicative."
)


def _run_backtest(start, end, freq_months, weighting, top_n_bt, cost_bps, bench_name, rf):
    if not selected_segments:
        st.error("Select at least one universe segment in the sidebar.")
        return None

    end = min(end, dt.date.today())
    if start >= end:
        st.error("Backtest **start** date must be before the **end** date.")
        return None
    hist_start = _history_start(start)

    with st.status("Fetching history & running backtest…", expanded=True) as status:
        st.write(f"Universe: {', '.join(selected_segments)}")
        uni = load_universe(tuple(selected_segments), force_refresh)
        tickers = uni["YFTicker"].tolist()
        st.write(f"Universe size: **{len(tickers)}** · history from {hist_start.isoformat()}")

        ck = f"bt_{'_'.join(sorted(selected_segments)).lower()}_{hist_start.isoformat()}_{end.isoformat()}"

        pbar = st.progress(0.0, text="Downloading price history…")
        def price_prog(done, total):
            pbar.progress(min(done / total, 1.0), text=f"Price history: {done}/{total}")

        mbar = st.progress(0.0, text="Fetching market caps…")
        def market_prog(done, total):
            mbar.progress(min(done / total, 1.0), text=f"Market caps: {done}/{total}")

        pdata = data.fetch_all(
            tickers, cache_key=ck, asof=end, start=hist_start,
            force_refresh=force_refresh, cache_hours=config.HISTORY_CACHE_HOURS,
            price_progress=price_prog, market_progress=market_prog,
        )

        bench_ticker = config.BENCHMARKS[bench_name]
        bench = data.fetch_benchmark(
            bench_ticker, f"{ck}_{bench_ticker}", start=hist_start, asof=end,
            force_refresh=force_refresh,
        )

        st.write("Running rebalances…")
        rbar = st.progress(0.0, text="Backtesting…")
        def reb_prog(done, total):
            rbar.progress(min(done / total, 1.0), text=f"Rebalance {done}/{total}")

        bp = backtest.BacktestParams(
            scan_params=replace(
                _base_scan_params(end),
                apply_turnover_ratio_filter=False,
                top_n=int(top_n_bt),
            ),
            start=start, end=end, rebalance_months=int(freq_months),
            weighting=weighting, cost_bps=float(cost_bps), rf_annual=rf / 100.0,
        )
        res = backtest.run_backtest(uni, pdata, bp, benchmark_series=bench, progress=reb_prog)
        status.update(label="Backtest complete ✅", state="complete", expanded=False)

    return {"res": res, "bench_name": bench_name, "segments": selected_segments,
            "start": start, "end": end, "weighting": weighting,
            "freq_months": int(freq_months), "cost_bps": float(cost_bps),
            "top_n": int(top_n_bt)}


def _pct(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:.{d}f}%"


def _num(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def _dte(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return pd.Timestamp(x).date().isoformat()


PERF_ROWS = [
    ("Total return", "total_return", _pct),
    ("CAGR", "cagr", _pct),
    ("Ann. volatility", "ann_vol", _pct),
    ("Sharpe (rf-adj)", "sharpe", _num),
    ("Sortino", "sortino", _num),
    ("Max drawdown", "max_dd", _pct),
    ("Calmar", "calmar", _num),
    ("Hit rate (periods)", "hit_rate", _pct),
    ("Avg turnover (1-way)", "avg_turnover", _pct),
    ("Annualised turnover", "annual_turnover", _pct),
]

REL_ROWS = [
    ("Beta", "beta", _num),
    ("Alpha (annualised)", "alpha_annual", _pct),
    ("Tracking error", "tracking_error", _pct),
    ("Information ratio", "info_ratio", _num),
    ("Up capture", "up_capture", _num),
    ("Down capture", "down_capture", _num),
    ("Correlation", "corr", _num),
    ("Hit rate vs benchmark", "hit_rate_vs_bench", _pct),
]


def _render_backtest_results(res: backtest.BacktestResult, bench_name: str) -> None:
    m, bm = res.metrics, res.benchmark_metrics
    has_bench = res.benchmark_curve is not None

    # -- KPI row ------------------------------------------------------------ #
    k = st.columns(5)
    def _delta(key, pct=True):
        if not has_bench:
            return None
        p, b = m.get(key), bm.get(key)
        if p is None or b is None or not (np.isfinite(p) and np.isfinite(b)):
            return None
        return f"{(p - b) * 100:+.1f} pp" if pct else f"{p - b:+.2f}"
    k[0].metric("CAGR", _pct(m.get("cagr")), _delta("cagr"))
    k[1].metric("Ann. vol", _pct(m.get("ann_vol")), _delta("ann_vol"), delta_color="inverse")
    k[2].metric("Sharpe", _num(m.get("sharpe")), _delta("sharpe", pct=False))
    k[3].metric("Max DD", _pct(m.get("max_dd")), _delta("max_dd"))
    k[4].metric("Ann. turnover", _pct(m.get("annual_turnover")))

    # -- Equity curve ------------------------------------------------------- #
    st.subheader("Growth of ₹100")
    curve = pd.DataFrame({"Strategy": res.equity_curve * 100.0})
    if has_bench:
        curve[bench_name] = res.benchmark_curve * 100.0
    log_scale = st.checkbox("Log scale", value=False, key="bt_log")
    st.line_chart(np.log10(curve) if log_scale else curve, height=380)
    if log_scale:
        st.caption("Y-axis is log₁₀(₹).")

    # -- Drawdown ----------------------------------------------------------- #
    st.subheader("Drawdown")
    dd = pd.DataFrame({"Strategy": res.drawdown})
    if has_bench:
        dd[bench_name] = res.benchmark_curve / res.benchmark_curve.cummax() - 1.0
    st.area_chart(dd, height=220)

    # -- Metrics tables ----------------------------------------------------- #
    st.subheader("Performance metrics")
    perf = pd.DataFrame([
        {"Metric": lab, "Strategy": fmt(m.get(key)),
         bench_name: (fmt(bm.get(key)) if has_bench else "—")}
        for lab, key, fmt in PERF_ROWS
    ])
    cA, cB = st.columns([3, 2])
    cA.dataframe(perf, hide_index=True, width="stretch")

    rel = pd.DataFrame([
        {"Metric": lab, "Value": fmt(res.relative_metrics.get(key))}
        for lab, key, fmt in REL_ROWS
    ])
    cB.markdown("**Strategy vs benchmark**")
    cB.dataframe(rel, hide_index=True, width="stretch")

    # Drawdown episode dates + run summary (span, rebalances, portfolio size).
    top_n_used = res.params.scan_params.top_n if res.params else None
    holds_note = f", up to {top_n_used} holdings" if top_n_used else ""
    st.caption(
        f"Max drawdown episode — peak {_dte(m.get('mdd_peak_date'))} → "
        f"trough {_dte(m.get('mdd_trough_date'))} → recovery {_dte(m.get('mdd_recovery_date'))}. "
        f"Backtest span {res.equity_curve.index[0].date()} → {res.equity_curve.index[-1].date()} "
        f"({m.get('n_rebalances', 0)} rebalances{holds_note})."
    )

    # -- Per-period returns ------------------------------------------------- #
    with st.expander("Per-rebalance period returns"):
        pr = res.period_returns.reset_index(drop=True)
        tbl = pd.DataFrame({
            "From": pd.to_datetime(pr["period_start"]).dt.date,
            "To": pd.to_datetime(pr["period_end"]).dt.date,
            "Strategy": (pr["strat_ret"] * 100).round(2),
        })
        if "bench_ret" in pr:
            tbl[bench_name] = (pr["bench_ret"] * 100).round(2)
        tbl["Turnover"] = (pr["turnover"] * 100).round(1)
        tbl["Cost"] = (pr["cost_drag"] * 100).round(3)
        tbl["Holdings"] = pr["n_holdings"]
        pct_cfg = {c: PCT_FMT for c in ["Strategy", bench_name, "Turnover", "Cost"] if c in tbl}
        st.dataframe(tbl, hide_index=True, width="stretch", column_config=pct_cfg)

    # -- Rebalance schedule (shows holiday mapping) ------------------------- #
    with st.expander("Rebalance schedule (scheduled → trading day)"):
        st.dataframe(res.schedule, hide_index=True, width="stretch")

    # -- Holdings viewer ---------------------------------------------------- #
    with st.expander("Holdings by rebalance date"):
        dates = res.rebalance_dates
        if not dates:
            st.caption("No holdings recorded (all-cash backtest).")
        else:
            i = st.selectbox(
                "Rebalance date", options=list(range(len(dates))),
                format_func=lambda j: dates[j].date().isoformat(),
                index=len(dates) - 1,
            )
            h = res.holdings.get(dates[i])
            if h is None or h.empty:
                st.caption("Cash held this period (no holdings).")
            else:
                hd = h.reset_index().rename(columns={"index": "YFTicker"})
                if "weight" in hd:
                    hd["weight"] = (hd["weight"] * 100).round(2)
                if "score" in hd:
                    hd["score"] = hd["score"].round(3)
                st.dataframe(
                    hd, hide_index=True, width="stretch",
                    column_config={"weight": PCT_FMT},
                )
                st.bar_chart(h.set_index("Symbol")["weight"] * 100.0, height=280)

    # -- Index membership changes (entries / exits) ------------------------- #
    with st.expander("Index membership changes — scrips entering / leaving"):
        mc = res.membership_changes
        if mc is None or mc.empty:
            st.caption("No membership changes recorded (all-cash backtest).")
        else:
            st.caption(
                "Each rebalance compared with the previous one. The first row is the initial "
                "deployment (every name enters). **Entered** / **Exited** list the scrips that "
                "joined or dropped out of the momentum portfolio as the ranking changed."
            )
            mtbl = pd.DataFrame({
                "Rebalance": pd.to_datetime(mc["rebalance_date"]).dt.date,
                "Held": mc["n_held"],
                "In": mc["n_entered"],
                "Out": mc["n_exited"],
                "Entered": mc["entered"],
                "Exited": mc["exited"],
            })
            st.dataframe(mtbl, hide_index=True, width="stretch")
            st.download_button(
                "⬇️ Membership changes (CSV)", mc.to_csv(index=False).encode(),
                file_name="backtest_membership_changes.csv", mime="text/csv",
                key="dl_membership",
            )

    # -- Transaction ledger (detailed report) ------------------------------- #
    with st.expander("Transaction ledger — detailed trade report"):
        tx = res.transactions
        if tx is None or tx.empty:
            st.caption("No transactions recorded (all-cash backtest).")
        else:
            cap = res.params.initial_capital if res.params else config.BACKTEST_INITIAL_CAPITAL
            st.caption(
                f"Every trade the strategy makes at each rebalance, denominated on a starting "
                f"capital of ₹{cap:,.0f} that compounds with the portfolio (so rupee figures track "
                f"the Growth of ₹{cap:,.0f} chart). Prices are as-of **adjusted** closes; weights "
                "are score/equal (never market-cap-capped). Modelled cost = bps × traded value."
            )
            tx_dates = sorted(tx["rebalance_date"].unique())
            j = st.selectbox(
                "Rebalance date", options=list(range(len(tx_dates))),
                format_func=lambda k: pd.Timestamp(tx_dates[k]).date().isoformat(),
                index=len(tx_dates) - 1, key="bt_tx_date",
            )
            day = tx[tx["rebalance_date"] == tx_dates[j]]
            s = st.columns(3)
            s[0].metric("Trades (excl. holds)", int((day["side"] != "HOLD").sum()))
            s[1].metric("Traded value", f"₹{day['traded_value'].sum():,.2f}")
            s[2].metric("Est. cost", f"₹{day['cost'].sum():,.2f}")
            view = pd.DataFrame({
                "Symbol": day["Symbol"].values,
                "Company": day["Company"].values,
                "Side": day["side"].values,
                "Price (₹)": day["price"].round(2).values,
                "Prior wt %": (day["prior_weight"] * 100).round(2).values,
                "Target wt %": (day["target_weight"] * 100).round(2).values,
                "Δ wt %": (day["delta_weight"] * 100).round(2).values,
                "Prior ₹": day["prior_value"].round(2).values,
                "Target ₹": day["target_value"].round(2).values,
                "Traded ₹": day["traded_value"].round(2).values,
                "Cost ₹": day["cost"].round(4).values,
            })
            st.dataframe(
                view, hide_index=True, width="stretch",
                column_config={
                    "Prior wt %": PCT_FMT, "Target wt %": PCT_FMT, "Δ wt %": PCT_FMT,
                    "Price (₹)": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.download_button(
                "⬇️ Full transaction ledger (CSV)", tx.to_csv(index=False).encode(),
                file_name="backtest_transactions.csv", mime="text/csv",
                key="dl_transactions",
            )

    # -- Downloads ---------------------------------------------------------- #
    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    eq_out = pd.DataFrame({"strategy": res.equity_curve})
    if has_bench:
        eq_out["benchmark"] = res.benchmark_curve
    d1.download_button(
        "⬇️ Equity curve (CSV)", eq_out.to_csv().encode(),
        file_name="backtest_equity.csv", mime="text/csv",
    )
    d2.download_button(
        "⬇️ Period returns (CSV)", res.period_returns.to_csv().encode(),
        file_name="backtest_periods.csv", mime="text/csv",
    )
    if res.holdings:
        flat = pd.concat(
            {d: h for d, h in res.holdings.items()}, names=["rebalance_date", "YFTicker"]
        ).reset_index()
        d3.download_button(
            "⬇️ Holdings history (CSV)", flat.to_csv(index=False).encode(),
            file_name="backtest_holdings.csv", mime="text/csv",
        )

    # -- Notes -------------------------------------------------------------- #
    dynamic = [n for n in res.notes if n not in backtest._base_notes(res.params)]
    if dynamic:
        with st.expander("Run notes"):
            for n in dynamic:
                st.caption(f"• {n}")


def render_backtest_tab() -> None:
    st.warning(BACKTEST_DISCLAIMER)

    c = st.columns(4)
    bt_freq = c[0].selectbox("Rebalance", list(FREQ_MONTHS), index=2, key="bt_freq")
    bt_weighting = c[1].radio(
        "Weighting", ["score", "equal"], index=0, horizontal=True, key="bt_weighting",
        help="Market-cap capping is disabled in backtests (look-ahead).",
    )
    bt_topn = c[2].number_input(
        "Holdings (Top N)", 1, 200, int(top_n), step=1, key="bt_topn",
        help="Backtest only the top-N momentum names each rebalance — e.g. set 10 for a "
        "concentrated top-10 portfolio. Defaults to the sidebar's Top N; every other "
        "parameter is unchanged.",
    )
    bt_cost = c[3].number_input("Cost (bps/side)", 0.0, 100.0, 0.0, step=5.0, key="bt_cost")

    # Explicit performance window — "between two specific dates".
    default_end = min(asof, dt.date.today())
    default_start = default_end - relativedelta(years=config.BACKTEST_DEFAULT_YEARS)
    d = st.columns(2)
    bt_start = d[0].date_input(
        "Backtest start", value=default_start, max_value=default_end, key="bt_start",
        help="Performance is measured between these two dates. The first rebalance may be "
        "pushed later automatically if there isn't enough prior price history for a scan.",
    )
    bt_end = d[1].date_input(
        "Backtest end", value=default_end, max_value=dt.date.today(), key="bt_end",
    )

    c2 = st.columns(2)
    bt_bench = c2[0].selectbox(
        "Benchmark", list(config.BENCHMARKS),
        index=list(config.BENCHMARKS).index(config.DEFAULT_BENCHMARK), key="bt_bench",
    )
    bt_rf = c2[1].number_input("Risk-free % p.a.", 0.0, 15.0, 0.0, step=0.5, key="bt_rf")

    if st.button("🚀 Run backtest", type="primary", width="stretch", key="run_bt"):
        st.session_state["bt"] = _run_backtest(
            bt_start, bt_end, FREQ_MONTHS[bt_freq], bt_weighting, bt_topn, bt_cost, bt_bench, bt_rf
        )

    bt = st.session_state.get("bt")
    if not bt or bt.get("res") is None:
        st.info("Configure the backtest above and press **Run backtest**.")
        return

    res: backtest.BacktestResult = bt["res"]
    if res.equity_curve is None or res.equity_curve.empty:
        for n in res.notes:
            st.warning(n)
        st.error(
            "Backtest produced no results for this configuration — the window is too "
            "short for a full holding period. Increase the history years or shorten the "
            "rebalance interval."
        )
        return

    _render_backtest_results(res, bt["bench_name"])


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_scan, tab_bt = st.tabs(["🔎 Scanner", "🧪 Backtest"])
with tab_scan:
    render_scan_tab()
with tab_bt:
    render_backtest_tab()
