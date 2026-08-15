"""Streamlit UI for the Nifty500 Momentum 50 scanner.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from momentum import config, data, scanner, universe

st.set_page_config(page_title="Nifty500 Momentum 50 Scanner", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Cached universe fetch (constituent lists are small and disk-cached anyway)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=config.UNIVERSE_CACHE_HOURS * 3600, show_spinner=False)
def load_universe(segments: tuple[str, ...], force: bool) -> pd.DataFrame:
    return universe.build_universe(list(segments), force_refresh=force)


# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Scan configuration")

st.sidebar.subheader("Universe")
st.sidebar.caption("Segments map to NSE size buckets (their union ≈ Nifty 500).")
seg_large = st.sidebar.checkbox("Largecap · Nifty 100", value=True)
seg_mid = st.sidebar.checkbox("Midcap · Nifty Midcap 150", value=False)
seg_small = st.sidebar.checkbox("Small · Nifty Smallcap 250", value=False)
selected_segments = [
    s for s, on in [("Largecap", seg_large), ("Midcap", seg_mid), ("Small", seg_small)] if on
]

st.sidebar.subheader("As-of date")
asof = st.sidebar.date_input(
    "Momentum measured as of the last trading day on/before:",
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
    w12 = st.slider("Weight on 12-month Z-score (%)", 0, 100, int(config.WEIGHT_12M * 100), step=5)
    w6 = 100 - w12
    st.caption(f"6-month Z-score weight = {w6}%")
    liq_pct = st.slider("Liquidity bottom-percentile cut", 0, 25, config.LIQUIDITY_BOTTOM_PERCENTILE)
    cap_abs = st.slider("Absolute weight cap (%)", 1, 20, int(config.CAP_ABSOLUTE * 100))
    cap_rel = st.number_input("Relative cap (× ff-mcap weight)", 1.0, 20.0, config.CAP_RELATIVE, step=0.5)
    ff_factor = st.slider(
        "Free-float factor (× market cap)", 0.1, 1.0, 1.0, step=0.05,
        help="Yahoo gives full market cap; scale it here to approximate free-float if desired.",
    )

force_refresh = st.sidebar.checkbox("Force refresh (ignore cache)", value=False)
run = st.sidebar.button("🚀 Run scan", type="primary", width="stretch")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📈 Nifty500 Momentum 50 Scanner")
st.markdown(
    "Scans NSE-listed stocks and applies the **Nifty500 Momentum 50** construction rules "
    "(NSE methodology §20): 6-month & 12-month volatility-adjusted momentum → cross-sectional "
    "Z-scores → Normalized Momentum Score → top-N selection → free-float-cap × score weighting."
)


# --------------------------------------------------------------------------- #
# Run
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

        cache_key = f"{'_'.join(sorted(selected_segments)).lower()}_{asof.isoformat()}"

        pbar = st.progress(0.0, text="Downloading price history…")
        def price_prog(done, total):
            pbar.progress(min(done / total, 1.0), text=f"Price history: {done}/{total}")

        mbar = st.progress(0.0, text="Fetching market caps…")
        def market_prog(done, total):
            mbar.progress(min(done / total, 1.0), text=f"Market caps: {done}/{total}")

        pdata = data.fetch_all(
            tickers, cache_key=cache_key, asof=asof,
            force_refresh=force_refresh, price_progress=price_prog, market_progress=market_prog,
        )

        st.write("Computing momentum, eligibility, scores & weights…")
        params = scanner.ScanParams(
            asof=asof,
            top_n=int(top_n),
            weight_12m=w12 / 100.0,
            weight_6m=w6 / 100.0,
            liquidity_bottom_pct=liq_pct,
            apply_listing_filter=apply_listing,
            apply_liquidity_filter=apply_liquidity,
            apply_turnover_ratio_filter=apply_turnover_ratio,
            cap_absolute=cap_abs / 100.0,
            cap_relative=cap_rel,
            free_float_factor=ff_factor,
        )
        result = scanner.run_scan(uni, pdata, params)
        status.update(label="Scan complete ✅", state="complete", expanded=False)

    return {"result": result, "universe_size": len(tickers), "asof": asof,
            "segments": selected_segments, "top_n": int(top_n)}


if run:
    st.session_state["scan"] = _run_scan()


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
DISPLAY_COLS = {
    "selection_rank": "Rank",
    "Symbol": "Symbol",
    "Company": "Company",
    "Segment": "Segment",
    "Industry": "Industry",
    "ret_6m": "6M Return",
    "ret_12m": "12M Return",
    "sigma": "Volatility (σ)",
    "z_6": "Z 6M",
    "z_12": "Z 12M",
    "score": "Momentum Score",
    "weight": "Weight",
}


def _format_selected(selected: pd.DataFrame) -> pd.DataFrame:
    df = selected.reset_index()[list(DISPLAY_COLS)].rename(columns=DISPLAY_COLS)
    for c in ["6M Return", "12M Return", "Volatility (σ)", "Weight"]:
        df[c] = (df[c] * 100).round(2)
    for c in ["Z 6M", "Z 12M", "Momentum Score"]:
        df[c] = df[c].round(3)
    return df


PCT_FMT = st.column_config.NumberColumn(format="%.2f%%")
COLCFG = {
    "6M Return": PCT_FMT, "12M Return": PCT_FMT, "Volatility (σ)": PCT_FMT, "Weight": PCT_FMT,
    "Rank": st.column_config.NumberColumn(format="%d"),
}


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
scan = st.session_state.get("scan")
if not scan:
    st.info("Configure the scan in the sidebar and press **Run scan**.")
    st.stop()

result: scanner.ScanResult = scan["result"]
if result is None:
    st.stop()

selected = result.selected
full = result.results

# --- Summary metrics ------------------------------------------------------- #
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
    st.stop()

# --- Selected table -------------------------------------------------------- #
st.subheader(f"Top {len(selected)} by Normalized Momentum Score")
disp = _format_selected(selected)
st.dataframe(disp, hide_index=True, width="stretch", column_config=COLCFG)

st.download_button(
    "⬇️ Download selection (CSV)",
    disp.to_csv(index=False).encode(),
    file_name=f"momentum50_{'_'.join(scan['segments'])}_{scan['asof'].isoformat()}.csv",
    mime="text/csv",
)

# --- Weights chart --------------------------------------------------------- #
st.subheader("Portfolio weights")
wchart = disp[["Symbol", "Weight"]].set_index("Symbol")
st.bar_chart(wchart, height=340)

# --- Full universe (audit) ------------------------------------------------- #
with st.expander("Full universe detail (all stocks, with eligibility & scores)"):
    audit_cols = {
        "Symbol": "Symbol", "Company": "Company", "Segment": "Segment",
        "ret_6m": "6M Return", "ret_12m": "12M Return", "sigma": "Volatility (σ)",
        "mr_6": "MR 6M", "mr_12": "MR 12M", "z_6": "Z 6M", "z_12": "Z 12M",
        "score": "Momentum Score", "eligible": "Eligible", "exclusion_reasons": "Exclusion reasons",
    }
    have = [c for c in audit_cols if c in full.columns]
    fdf = full.reset_index()[["YFTicker"] + have].rename(columns=audit_cols)
    for c in ["6M Return", "12M Return", "Volatility (σ)"]:
        if c in fdf:
            fdf[c] = (fdf[c] * 100).round(2)
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
