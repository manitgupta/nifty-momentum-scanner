"""Streamlit UI for the Nifty500 Momentum 50 scanner + backtester.

Three tabs share one sidebar configuration:
    🔎 Scanner  — a point-in-time momentum scan as of any date, plus a portfolio
                  builder that turns it into whole-share orders.
    🧪 Backtest — a rebalancing backtest of the same strategy over a period.
    ⚖️ Compare  — the backtest swept across rebalance frequencies × Top-N.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import replace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

from momentum import analytics, backtest, charts, config, data, portfolio, scanner, universe

st.set_page_config(page_title="Nifty Momentum Scanner & Backtester", page_icon="📈", layout="wide")
alt.data_transformers.disable_max_rows()


# --------------------------------------------------------------------------- #
# Chart palette (reference data-viz palette; light & dark steps of the same hues)
# --------------------------------------------------------------------------- #
def _theme() -> str:
    try:
        t = st.context.theme.type  # type: ignore[attr-defined]
        return "dark" if t == "dark" else "light"
    except Exception:
        return "light"


PAL = charts.PALETTES[_theme()]


# --------------------------------------------------------------------------- #
# Cached universe fetch (constituent lists are small and disk-cached anyway)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=config.UNIVERSE_CACHE_HOURS * 3600, show_spinner=False)
def load_universe(segments: tuple[str, ...], force: bool) -> pd.DataFrame:
    return universe.build_universe(list(segments), force_refresh=force)


@st.cache_data(show_spinner=False)
def etf_catalogue() -> pd.DataFrame:
    df = pd.read_csv(config.ETF_LIST_PATH)
    return df.rename(columns={"Company Name": "Company"})[["Symbol", "Company", "Industry"]]


# --------------------------------------------------------------------------- #
# Sidebar — configuration (shared by all tabs)
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Universe")
UNI_MODES = ["Size segments", "Index", "ETFs", "Custom list"]
uni_mode = st.sidebar.radio(
    "Universe source",
    UNI_MODES,
    horizontal=True,
    key="uni_mode",
    help="Size segments: compose from Nifty 100 / Midcap 150 / Smallcap 250 (their union ≈ "
    "Nifty 500). Index: any NSE broad, sectoral or thematic index (combine several). ETFs: a "
    "curated list of liquid NSE ETFs — momentum rotation across asset classes. Custom list: "
    "type your own NSE symbols (stocks and/or ETFs).",
)
DEFAULT_CUSTOM = "RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, BHARTIARTL, LT, ITC, SBIN, NIFTYBEES, GOLDBEES"
custom_symbols: list[str] = []
etf_symbols: list[str] = []
etf_extra: list[str] = []
if uni_mode == "Size segments":
    st.sidebar.caption("Segments map to NSE size buckets (their union ≈ Nifty 500).")
    seg_large = st.sidebar.checkbox("Largecap · Nifty 100", value=True, key="seg_large")
    seg_mid = st.sidebar.checkbox("Midcap · Nifty Midcap 150", value=False, key="seg_mid")
    seg_small = st.sidebar.checkbox("Small · Nifty Smallcap 250", value=False, key="seg_small")
    selected_segments = [
        s for s, on in [("Largecap", seg_large), ("Midcap", seg_mid), ("Small", seg_small)] if on
    ]
elif uni_mode == "Index":
    idx_labels = st.sidebar.multiselect(
        "Index universe",
        list(config.INDEX_CHOICES),
        default=["Nifty 500"],
        key="idx_choice",
        help="Broad market (Nifty 50 … Microcap 250, Total Market), sectoral (Bank, Financial "
        "Services, IT, Pharma, …) and thematic/strategy indices. Picking several combines "
        "their constituents (de-duplicated). Larger universes take longer to fetch.",
    )
    selected_segments = list(dict.fromkeys(config.INDEX_CHOICES[lab] for lab in idx_labels))
elif uni_mode == "ETFs":
    cat = etf_catalogue()
    names = dict(zip(cat["Symbol"], cat["Company"]))
    etf_symbols = st.sidebar.multiselect(
        "ETFs in the basket",
        cat["Symbol"].tolist(),
        default=cat.loc[cat["Industry"] != "Debt", "Symbol"].tolist(),
        key="etf_pick",
        format_func=lambda s: f"{s} · {names.get(s, '')}",
        help="One liquid ETF per underlying: broad equity, factors, sectors, international and "
        "gold/silver. Gilt/liquid (debt) ETFs are available but left out by default: their tiny "
        "volatility gives them huge volatility-adjusted momentum, so they would top the ranking "
        "almost permanently. Add them only as a deliberate defensive sleeve.",
    )
    extra_txt = st.sidebar.text_input(
        "Add more NSE symbols (optional)", key="etf_extra",
        help="Any other NSE-listed ETF or stock symbols, comma-separated (e.g. MAHKTECH, SETFNIF50).",
    )
    etf_extra, _bad = universe.parse_symbols(extra_txt)
    if _bad:
        st.sidebar.caption(f"Ignored: {', '.join(_bad)}")
    st.sidebar.caption("Tip: ETF rotation works best with a small Top N (3–5) and score/equal weighting.")
    selected_segments = [config.ETF_SEGMENT] if (etf_symbols or etf_extra) else []
else:
    custom_txt = st.sidebar.text_area(
        "NSE symbols (stocks and/or ETFs)", value=DEFAULT_CUSTOM, key="custom_syms", height=110,
        help="Comma/space/newline separated, e.g. RELIANCE, TCS, NIFTYBEES.",
    )
    custom_symbols, _bad = universe.parse_symbols(custom_txt)
    st.sidebar.caption(f"{len(custom_symbols)} symbol(s)" + (f" · ignored: {', '.join(_bad)}" if _bad else ""))
    selected_segments = [config.CUSTOM_SEGMENT] if custom_symbols else []

price_only_universe = uni_mode in ("ETFs", "Custom list")

st.sidebar.subheader("As-of date")
asof = st.sidebar.date_input(
    "Scan as-of / backtest end (last trading day on/before):",
    value=dt.date.today(),
    max_value=dt.date.today(),
)

st.sidebar.subheader("Eligibility filters")
apply_listing = st.sidebar.checkbox("Minimum 1-year listing history", value=True)
apply_liquidity = st.sidebar.checkbox("Exclude bottom-10% by 6M avg daily turnover", value=True)
apply_turnover_ratio = st.sidebar.checkbox(
    "Exclude bottom-10% by turnover ratio", value=not price_only_universe, key=f"tor_{price_only_universe}",
    help="Needs market caps — off by default for ETFs / custom lists (ETFs have none).",
)
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

SCAN_WEIGHTING = {
    "mcap_score": "Methodology (mcap × score, 5%/5× cap)",
    "score": "Score-weighted",
    "equal": "Equal-weight",
    "inverse_vol": "Inverse-volatility",
}

with st.sidebar.expander("Advanced parameters"):
    top_n = st.number_input(
        "Number of stocks to select (Top N)", 1, 200, 5 if price_only_universe else config.TOP_N,
        step=1, key=f"top_n_{price_only_universe}",
    )
    scan_weighting = st.selectbox(
        "Weighting scheme", list(SCAN_WEIGHTING), format_func=SCAN_WEIGHTING.get,
        index=1 if price_only_universe else 0, key=f"scan_w_{price_only_universe}",
        help="Methodology = free-float market cap × momentum score, capped at min(5%, 5× mcap "
        "weight). The others are price-only (no market cap needed — use them for ETFs).",
    )
    max_sector = st.number_input(
        "Max stocks per sector (0 = no cap)", 0, 50, 0, step=1, key="max_sector",
        help="Diversification guard: skip a name once its industry already has this many picks.",
    )

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
        weighting=scan_weighting,
        max_per_sector=int(max_sector) or None,
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
# Universe & benchmark resolution
# --------------------------------------------------------------------------- #
NO_UNIVERSE_MSG = "Select at least one universe segment, index, ETF or symbol in the sidebar."


def load_selected_universe(force: bool) -> pd.DataFrame:
    """The universe described by the sidebar (index lists, ETF basket or custom symbols)."""
    if uni_mode == "Custom list":
        return universe.build_custom_universe(custom_symbols)
    uni = load_universe(tuple(selected_segments), force)
    if uni_mode == "ETFs":
        uni = uni[uni["Symbol"].isin(etf_symbols)]
        extra = [s for s in etf_extra if s not in set(uni["Symbol"])]
        if extra:
            uni = pd.concat([uni, universe.build_custom_universe(extra)], ignore_index=True)
    return uni.reset_index(drop=True)


def universe_tag(uni: pd.DataFrame) -> str:
    """Short, filename-safe cache tag for the current universe."""
    if uni_mode in ("Size segments", "Index"):
        tag = "_".join(sorted(selected_segments)).lower()
        if len(tag) <= 60:
            return tag
    digest = hashlib.md5(",".join(sorted(uni["YFTicker"])).encode()).hexdigest()[:10]
    prefix = {"ETFs": "etf", "Custom list": "custom"}.get(uni_mode, "idx")
    return f"{prefix}_{digest}"


def universe_label() -> str:
    if uni_mode == "ETFs":
        return f"ETF basket ({len(etf_symbols) + len(etf_extra)})"
    if uni_mode == "Custom list":
        return f"Custom list ({len(custom_symbols)})"
    return ", ".join(config.SEGMENT_LABELS.get(s, s) for s in selected_segments)


def auto_benchmark() -> str:
    """The benchmark index that best matches the selected universe."""
    if uni_mode == "ETFs":
        return config.UNIVERSE_BENCHMARK[config.ETF_SEGMENT]
    if uni_mode == "Custom list" or not selected_segments:
        return config.DEFAULT_BENCHMARK
    if len(selected_segments) == 1:
        return config.UNIVERSE_BENCHMARK.get(selected_segments[0], config.DEFAULT_BENCHMARK)
    return config.SEGMENT_COMBO_BENCHMARK.get(frozenset(selected_segments), config.DEFAULT_BENCHMARK)


def fetch_named_benchmark(label: str, start: dt.date, end: dt.date, tri: bool) -> pd.Series:
    spec = config.BENCHMARKS[label]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    kind = "tri" if (tri and spec.get("nse")) else "px"
    return data.fetch_benchmark(
        spec.get("yahoo"), f"{slug}_{kind}_{start.isoformat()}_{end.isoformat()}",
        start=start, asof=end, force_refresh=force_refresh, nse_name=spec.get("nse"), tri=tri,
    )


def resolve_benchmark(choice: str, pdata, start: dt.date, end: dt.date, tri: bool) -> tuple[str, pd.Series]:
    """(display label, series) for a benchmark choice incl. the Auto / EW pseudo-options."""
    if choice == config.UNIVERSE_EW_BENCHMARK:
        s = backtest.universe_equal_weight(pdata.close)
        s.attrs["source"] = "computed daily from the universe's adjusted closes"
        return "Universe EW", s
    label = auto_benchmark() if choice == config.AUTO_BENCHMARK else choice
    return label, fetch_named_benchmark(label, start, end, tri)


# --------------------------------------------------------------------------- #
# Formatting & chart helpers
# --------------------------------------------------------------------------- #
def _pct(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:.{d}f}%"


def _num(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def _dte(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or (x is pd.NaT):
        return "—"
    try:
        return pd.Timestamp(x).date().isoformat()
    except (ValueError, TypeError):
        return "—"


PCT_FMT = st.column_config.NumberColumn(format="%.2f%%")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📈 Nifty Momentum — Scanner & Backtester")
st.markdown(
    "Applies the **Nifty500 Momentum 50** construction rules (NSE methodology §20): "
    "volatility-adjusted 6M & 12M momentum → cross-sectional Z-scores → Normalized "
    "Momentum Score → top-N selection — to any NSE index, sector, ETF basket or your own list. "
    "The **Scanner** ranks as of a date (and builds the order list), the **Backtest** "
    "rebalances the strategy over a period, and **Compare** finds robust settings."
)
with st.expander("ℹ️ How to use this tool"):
    st.markdown(
        "1. **Pick a universe** in the sidebar — size segments, any NSE index (Nifty 50 … "
        "Microcap 250, Bank, Financial Services, IT, …), the ETF basket, or a custom list.\n"
        "2. **Scanner → Run scan** ranks it today (or any past date). Under *Portfolio builder* "
        "enter your capital to get exact share quantities, or paste your current holdings to get "
        "the buy/sell list for this rebalance.\n"
        "3. **Backtest** replays the strategy: choose dates, rebalance frequency (daily → annual), "
        "Top N, costs and a benchmark. *Risk controls* add a rank buffer (less churn), a "
        "market-regime filter (go to cash in downtrends) and a sector cap.\n"
        "4. **Compare** runs the backtest over a grid of frequencies × Top N so you can pick "
        "settings that work across the board, not just one lucky combination.\n\n"
        "Educational tool — not investment advice. Read the backtest limitations on the Backtest tab."
    )


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #
def _market_regime(asof_date: dt.date) -> dict | None:
    """Where the matching index sits vs its 200-day average (for a banner)."""
    label = auto_benchmark()
    try:
        s = fetch_named_benchmark(label, asof_date - dt.timedelta(days=420), asof_date, tri=False)
    except Exception:
        return None
    s = s.dropna()
    if len(s) < config.REGIME_MA_DAYS:
        return None
    sma = float(s.iloc[-config.REGIME_MA_DAYS:].mean())
    last = float(s.iloc[-1])
    return {"label": label, "last": last, "sma": sma, "gap": last / sma - 1.0, "date": s.index[-1]}


def _run_scan():
    if not selected_segments:
        st.error(NO_UNIVERSE_MSG)
        return None

    with st.status("Fetching data & scanning…", expanded=True) as status:
        st.write(f"Loading universe: {universe_label()}")
        try:
            uni = load_selected_universe(force_refresh)
        except ValueError as exc:
            status.update(label="Scan failed", state="error")
            st.error(str(exc))
            return None
        tickers = uni["YFTicker"].tolist()
        st.write(f"Universe size: **{len(tickers)}** instruments")
        if not tickers:
            status.update(label="Scan failed", state="error")
            st.error(NO_UNIVERSE_MSG)
            return None

        start = _history_start(asof)
        cache_key = f"{universe_tag(uni)}_{start.isoformat()}_{asof.isoformat()}"

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
        regime = _market_regime(asof)
        status.update(label="Scan complete ✅", state="complete", expanded=False)

    return {"result": result, "universe_size": len(tickers), "asof": asof,
            "segments": selected_segments, "top_n": int(top_n), "params": params,
            "label": universe_label(), "regime": regime,
            "lb_long": int(lb_long), "lb_short": int(lb_short), "skip_m": int(skip_m)}


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


def _colcfg(cols: dict[str, str]) -> dict:
    return {
        cols["ret_6m"]: PCT_FMT, cols["ret_12m"]: PCT_FMT,
        "Volatility (σ)": PCT_FMT, "Weight": PCT_FMT,
        "Close (₹)": st.column_config.NumberColumn(format="%.2f"),
        "Rank": st.column_config.NumberColumn(format="%d"),
    }


def _render_regime_banner(reg: dict | None) -> None:
    if not reg:
        return
    txt = (f"**{reg['label']}** closed {_pct(abs(reg['gap']))} {'above' if reg['gap'] >= 0 else 'below'} "
           f"its {config.REGIME_MA_DAYS}-day average on {reg['date'].date()} "
           f"({reg['last']:,.0f} vs {reg['sma']:,.0f}).")
    if reg["gap"] >= 0:
        st.success(f"🟢 Market regime: **risk-on** — {txt}")
    else:
        st.error(f"🔴 Market regime: **risk-off** — {txt} Trend-following investors often reduce "
                 "exposure here (see the regime filter in the Backtest tab).")


def _render_portfolio_builder(scan: dict, selected: pd.DataFrame, full: pd.DataFrame) -> None:
    st.subheader("🧾 Portfolio builder")
    st.caption("Turn the selection into an order list at the as-of closing prices. Quantities are whole "
               "shares; actual fills will differ slightly from these closes.")
    b1, b2 = st.columns([1, 2])
    capital = b1.number_input("Capital to invest (₹)", 1_000.0, 1e10, 1_000_000.0, step=50_000.0,
                              key="pb_capital", format="%.0f")
    top_up = b1.checkbox("Spend leftover cash on the most under-weight names", value=True, key="pb_topup")
    alloc, left = portfolio.allocate(selected, float(capital), top_up=top_up)
    with b2:
        m = st.columns(3)
        m[0].metric("Invested", portfolio.round_lakh(float(alloc["invested"].sum())))
        m[1].metric("Cash left", portfolio.round_lakh(left))
        m[2].metric("Names bought", int((alloc["shares"] > 0).sum()))
    view = alloc.assign(
        target_weight=(alloc["target_weight"] * 100).round(2),
        actual_weight=(alloc["actual_weight"] * 100).round(2),
        price=alloc["price"].round(2), target_value=alloc["target_value"].round(0),
        invested=alloc["invested"].round(0),
    ).rename(columns={"price": "Price (₹)", "target_weight": "Target wt %", "target_value": "Target ₹",
                      "shares": "Shares", "invested": "Invested ₹", "actual_weight": "Actual wt %"})
    st.dataframe(view, hide_index=True, width="stretch",
                 column_config={"Target wt %": PCT_FMT, "Actual wt %": PCT_FMT})
    st.download_button("⬇️ Buy list (CSV)", view.to_csv(index=False).encode(),
                       file_name=f"buy_list_{scan['asof'].isoformat()}.csv", mime="text/csv", key="dl_buylist")

    with st.expander("🔁 Rebalance my existing portfolio"):
        st.caption("Paste current holdings (one per line: `SYMBOL, QTY`) or upload a broker holdings CSV "
                   "with Symbol and Qty columns. You'll get the exact sells and buys that move it onto this "
                   "scan's target weights.")
        r1, r2 = st.columns(2)
        hold_txt = r1.text_area("Current holdings", key="pb_holdings", height=150,
                                placeholder="RELIANCE, 25\nTCS, 10\nNIFTYBEES, 300")
        up = r2.file_uploader("…or upload holdings CSV", type=["csv"], key="pb_upload")
        extra_cash = r2.number_input("Cash to add (₹; negative to withdraw)", -1e10, 1e10, 0.0,
                                     step=10_000.0, key="pb_extra", format="%.0f")
        buffer_rank = r2.number_input(
            "Rank buffer: keep holdings ranked within (0 = off)", 0, 500, 0, step=5, key="pb_buffer",
            help="Hold a current position while it still ranks inside this number, instead of selling the "
            "moment it slips out of the Top N. Cuts turnover and costs (e.g. Top 20 with a buffer of 40).")
        try:
            held, bad = portfolio.parse_holdings(hold_txt, up.getvalue() if up is not None else None)
        except ValueError as exc:
            st.error(str(exc))
            return
        if bad:
            st.caption(f"Couldn't read: {', '.join(bad[:8])}")
        if not held:
            return
        target = selected
        res_full = full.reset_index()
        by_sym = res_full.set_index("Symbol") if "Symbol" in res_full.columns else pd.DataFrame()
        if buffer_rank and buffer_rank > 0:
            keep = [by_sym.loc[s, "YFTicker"] for s in held if s in by_sym.index]
            target = scanner.select_and_weight(full, scan["params"], keep=keep, exit_rank=int(buffer_rank))
        prices = {s: float(by_sym.loc[s, "close_price"]) for s in held
                  if s in by_sym.index and pd.notna(by_sym.loc[s, "close_price"])}
        orders, summ = portfolio.rebalance_orders(target, held, prices, extra_cash=float(extra_cash))
        k = st.columns(4)
        k[0].metric("Current value", portfolio.round_lakh(summ["current_value"]))
        k[1].metric("Sell", f"{summ['n_sells']} · {portfolio.round_lakh(summ['sell_value'])}")
        k[2].metric("Buy", f"{summ['n_buys']} · {portfolio.round_lakh(summ['buy_value'])}")
        k[3].metric("Cash after", portfolio.round_lakh(summ["cash_after"]))
        if summ["n_review"]:
            st.warning(f"{summ['n_review']} holding(s) are outside the scanned universe, so they can't be "
                       "priced — marked REVIEW and left untouched.")
        st.dataframe(
            orders.assign(target_weight=(orders["target_weight"] * 100).round(2), price=orders["price"].round(2),
                          value=orders["value"].round(0)),
            hide_index=True, width="stretch", column_config={"target_weight": PCT_FMT},
        )
        st.download_button("⬇️ Rebalance orders (CSV)", orders.to_csv(index=False).encode(),
                           file_name=f"rebalance_orders_{scan['asof'].isoformat()}.csv", mime="text/csv",
                           key="dl_orders")


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

    _render_regime_banner(scan.get("regime"))
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
    params: scanner.ScanParams = scan["params"]
    st.subheader(f"Top {len(selected)} by Normalized Momentum Score")
    st.caption(
        f"{scan.get('label', '')} · momentum legs: {scan['lb_short']}M & {scan['lb_long']}M lookback{skip_note} · "
        f"weighting: {SCAN_WEIGHTING.get(params.weighting, params.weighting)}"
        + (f" · max {params.max_per_sector} per sector" if params.max_per_sector else "")
        + ". Close (₹) is the actual traded close on the as-of date; returns use adjusted close."
    )
    disp = _format_selected(selected, cols)
    st.dataframe(disp, hide_index=True, width="stretch", column_config=_colcfg(cols))

    st.download_button(
        "⬇️ Download selection (CSV)",
        disp.to_csv(index=False).encode(),
        file_name=f"momentum_{'_'.join(scan['segments'])}_{scan['asof'].isoformat()}.csv",
        mime="text/csv",
    )

    w1, w2 = st.columns([3, 2])
    with w1:
        st.subheader("Portfolio weights")
        wchart = disp[["Symbol", "Weight"]].set_index("Symbol")
        st.bar_chart(wchart, height=340, color=PAL["series"][0])
    with w2:
        st.subheader("Sector mix")
        if "Industry" in selected.columns:
            sec = (selected.assign(Industry=selected["Industry"].fillna("Unclassified"))
                   .groupby("Industry").agg(weight=("weight", "sum"), names=("weight", "size"))
                   .reset_index().sort_values("weight", ascending=False))
            st.altair_chart(charts.sector_bar(sec, PAL), width="stretch")

    _render_portfolio_builder(scan, selected, full)

    with st.expander("Full universe detail (all stocks, with eligibility & scores)"):
        ls, ll = scan["lb_short"], scan["lb_long"]
        audit_cols = {
            "Symbol": "Symbol", "Company": "Company", "Segment": "Segment", "Industry": "Industry",
            "close_price": "Close (₹)",
            "ret_6m": f"{ls}M Return", "ret_12m": f"{ll}M Return", "sigma": "Volatility (σ)",
            "mr_6": f"MR {ls}M", "mr_12": f"MR {ll}M", "z_6": f"Z {ls}M", "z_12": f"Z {ll}M",
            "score": "Momentum Score", "rank": "Rank", "eligible": "Eligible",
            "exclusion_reasons": "Exclusion reasons",
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
        "Educational tool — not investment advice. Methodology weights use full market cap as a free-float proxy; "
        "pledge & circuit eligibility rules are omitted (data unavailable). See README for full methodology mapping."
    )


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
FREQS = list(config.REBALANCE_FREQUENCIES)
BT_WEIGHTING = {"score": "Score", "equal": "Equal", "inverse_vol": "Inverse-vol"}
BENCH_OPTIONS = [config.AUTO_BENCHMARK, config.UNIVERSE_EW_BENCHMARK] + list(config.BENCHMARKS)
REGIME_SAME = "Same as benchmark"

BACKTEST_DISCLAIMER = (
    "⚠️ **Backtest limitations — read first.** These results are **educational, not "
    "investment advice**, and are structurally *optimistic*:\n"
    "- **Survivorship bias** — the universe is *today's* index constituents, so stocks "
    "later delisted or demoted are silently excluded from history.\n"
    "- **No look-ahead weighting** — weights are **score-, equal- or inverse-volatility-weighted "
    "only**. Live market-cap capping is *not* used (market cap is a current-only snapshot; "
    "applying it to the past would be look-ahead). The turnover-ratio eligibility "
    "filter is disabled for the same reason.\n"
    "- **Costs & liquidity** — modelled only via the bps cost input; taxes, "
    "market impact, slippage beyond that and capacity limits are not modelled.\n"
    "- **Point-in-time index membership** is not available from free data.\n\n"
    "The momentum *ranking* is price-only and unbiased; the *absolute-return* figures are indicative."
)


def _bt_data(start: dt.date, end: dt.date):
    """Universe + multi-year price history for a backtest window (or None + error)."""
    hist_start = _history_start(start)
    st.write(f"Universe: {universe_label()}")
    uni = load_selected_universe(force_refresh)
    tickers = uni["YFTicker"].tolist()
    if not tickers:
        st.error(NO_UNIVERSE_MSG)
        return None
    st.write(f"Universe size: **{len(tickers)}** · history from {hist_start.isoformat()}")
    ck = f"bt_{universe_tag(uni)}_{hist_start.isoformat()}_{end.isoformat()}"

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
    return uni, pdata, hist_start


def _bt_series(cfg: dict, pdata, hist_start: dt.date, end: dt.date):
    """(bench label, bench series, regime series) for the configured choices."""
    bench_label, bench = resolve_benchmark(cfg["bench"], pdata, hist_start, end, cfg["tri"])
    regime = None
    if cfg["regime_on"]:
        if cfg["regime_index"] == REGIME_SAME:
            regime = bench
        else:
            regime = fetch_named_benchmark(cfg["regime_index"], hist_start, end, tri=False)
    return bench_label, bench, regime


def _bt_params(cfg: dict) -> backtest.BacktestParams:
    sp = replace(
        _base_scan_params(cfg["end"]),
        apply_turnover_ratio_filter=False,
        top_n=int(cfg["top_n"]),
        max_per_sector=int(cfg["max_sector"]) or None,
    )
    return backtest.BacktestParams(
        scan_params=sp, start=cfg["start"], end=cfg["end"], rebalance_freq=cfg["freq"],
        weighting=cfg["weighting"], cost_bps=float(cfg["cost_bps"]), rf_annual=cfg["rf"] / 100.0,
        exit_rank=int(cfg["exit_rank"]) if cfg["buffer_on"] else None,
        regime_filter=bool(cfg["regime_on"]), regime_ma_days=int(cfg["regime_ma"]),
        regime_exposure_off=cfg["regime_off"] / 100.0, cash_yield_annual=cfg["cash_yield"] / 100.0,
    )


def _validate_window(cfg: dict) -> tuple[dt.date, dt.date] | None:
    if not selected_segments:
        st.error(NO_UNIVERSE_MSG)
        return None
    end = min(cfg["end"], dt.date.today())
    if cfg["start"] >= end:
        st.error("Backtest **start** date must be before the **end** date.")
        return None
    return cfg["start"], end


def _run_backtest(cfg: dict):
    window = _validate_window(cfg)
    if window is None:
        return None
    start, end = window
    cfg = {**cfg, "end": end}

    with st.status("Fetching history & running backtest…", expanded=True) as status:
        got = _bt_data(start, end)
        if got is None:
            status.update(label="Backtest failed", state="error")
            return None
        uni, pdata, hist_start = got
        bench_label, bench, regime = _bt_series(cfg, pdata, hist_start, end)

        st.write("Running rebalances…")
        rbar = st.progress(0.0, text="Backtesting…")
        def reb_prog(done, total):
            rbar.progress(min(done / total, 1.0), text=f"Rebalance {done}/{total}")

        bp = _bt_params(cfg)
        res = backtest.run_backtest(uni, pdata, bp, benchmark_series=bench, progress=reb_prog,
                                    regime_series=regime)
        res.benchmark_name = bench_label
        status.update(label="Backtest complete ✅", state="complete", expanded=False)

    return {"res": res, "bench_name": bench_label, "bench_source": bench.attrs.get("source", "") if bench is not None else "",
            "segments": selected_segments, "label": universe_label(),
            "start": start, "end": end, "weighting": cfg["weighting"], "freq": cfg["freq"],
            "cost_bps": float(cfg["cost_bps"]), "top_n": int(cfg["top_n"]), "cfg": cfg}


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


def _render_overview(res: backtest.BacktestResult, bench_name: str, bench_source: str) -> None:
    m, bm = res.metrics, res.benchmark_metrics
    has_bench = res.benchmark_curve is not None

    # -- KPI row ------------------------------------------------------------ #
    k = st.columns(6 if "time_in_market" in m else 5)
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
    if "time_in_market" in m:
        k[5].metric("Time invested", _pct(m.get("time_in_market"), 0))
    if has_bench and bench_source:
        st.caption(f"Benchmark **{bench_name}** — source: {bench_source}. KPI deltas are vs the benchmark.")

    # -- Equity curve ------------------------------------------------------- #
    st.subheader("Growth of ₹100")
    curve = pd.DataFrame({"Strategy": res.equity_curve * 100.0})
    if has_bench:
        curve[bench_name] = res.benchmark_curve * 100.0
    log_scale = st.checkbox("Log scale", value=False, key="bt_log")
    st.altair_chart(charts.line_chart(curve, charts.strat_bench_colors(PAL, bench_name if has_bench else None),
                                      "₹ (start = 100)", PAL, ",.0f", log=log_scale), width="stretch")

    # -- Drawdown ----------------------------------------------------------- #
    st.subheader("Drawdown")
    dd = pd.DataFrame({"Strategy": res.drawdown})
    if has_bench:
        dd[bench_name] = res.benchmark_curve / res.benchmark_curve.cummax() - 1.0
    st.altair_chart(charts.line_chart(dd, charts.strat_bench_colors(PAL, bench_name if has_bench else None),
                                      "Drawdown", PAL, ".0%", height=220), width="stretch")

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

    dynamic = [n for n in res.notes if n not in backtest._base_notes(res.params)]
    if dynamic:
        with st.expander(f"Run notes ({len(dynamic)})"):
            for n in dynamic[:200]:
                st.caption(f"• {n}")
            if len(dynamic) > 200:
                st.caption(f"… and {len(dynamic) - 200} more.")


def _render_returns_analysis(res: backtest.BacktestResult, bench_name: str) -> None:
    has_bench = res.benchmark_curve is not None
    bench = res.benchmark_curve if has_bench else None

    st.subheader("📅 Calendar returns")
    cal = analytics.calendar_returns(res.equity_curve)
    if cal.empty:
        st.caption("Not enough history for calendar returns.")
    else:
        st.altair_chart(charts.calendar_heatmap(cal, PAL), width="stretch")
        ann = analytics.annual_returns(res.equity_curve, bench)
        if has_bench:
            ann = ann.rename(columns={"Benchmark": bench_name})
        st.dataframe((ann * 100).round(2), width="stretch",
                     column_config={c: PCT_FMT for c in ann.columns})
        st.caption("Blue = gain, red = loss; first/last years are partial. Full-year figures in the table.")

    st.subheader("📈 Rolling returns")
    span_years = (res.equity_curve.index[-1] - res.equity_curve.index[0]).days / 365.25
    choices = [y for y in (1, 3, 5) if y < span_years] or [1]
    win = st.radio("Window", choices, horizontal=True, format_func=lambda y: f"{y}-year", key="bt_roll")
    roll_s = analytics.rolling_returns(res.equity_curve, win)
    roll_b = analytics.rolling_returns(bench, win) if has_bench else None
    if roll_s.empty:
        st.caption("The backtest is shorter than this window.")
    else:
        wide = pd.DataFrame({"Strategy": roll_s})
        if roll_b is not None and not roll_b.empty:
            wide[bench_name] = roll_b
        st.altair_chart(charts.line_chart(wide, charts.strat_bench_colors(PAL, bench_name if has_bench else None),
                                          f"Annualised {win}-yr return", PAL, ".0%", height=280), width="stretch")
        summ = analytics.rolling_summary(roll_s, roll_b)
        k = st.columns(5)
        k[0].metric("Median", _pct(summ["median"]))
        k[1].metric("Worst", _pct(summ["min"]))
        k[2].metric("Best", _pct(summ["max"]))
        k[3].metric("% windows positive", _pct(summ["pct_positive"], 0))
        k[4].metric("% windows beating benchmark", _pct(summ["pct_beat"], 0))

    st.subheader("🕳️ Worst drawdowns")
    eps = analytics.drawdown_episodes(res.equity_curve, top=5)
    if eps.empty:
        st.caption("No drawdowns.")
    else:
        st.dataframe(pd.DataFrame({
            "Peak": eps["peak"].map(_dte), "Trough": eps["trough"].map(_dte),
            "Recovered": eps["recovery"].map(lambda d: _dte(d) if pd.notna(d) else "not yet"),
            "Depth": (eps["depth"] * 100).round(1),
            "Days falling": eps["days_to_trough"],
            "Days to recover": eps["days_to_recover"],
            "Total days": eps["total_days"],
        }), hide_index=True, width="stretch", column_config={"Depth": PCT_FMT})


def _render_sip(res: backtest.BacktestResult, bench_name: str) -> None:
    st.subheader("💰 SIP simulator")
    st.caption("A monthly SIP into the strategy (first trading day of each month) vs the same SIP into "
               "the benchmark. Assumes fractional units and no extra costs on the inflows.")
    c = st.columns(3)
    amount = c[0].number_input("Monthly SIP (₹)", 500.0, 1e8, config.SIP_DEFAULT_AMOUNT, step=1_000.0,
                               key="sip_amt", format="%.0f")
    step_up = c[1].number_input("Annual step-up (%)", 0.0, 50.0, 0.0, step=5.0, key="sip_step",
                                help="Raise the instalment by this % every 12 months (a 'step-up SIP').")
    has_bench = res.benchmark_curve is not None
    dates = analytics.sip_dates(res.equity_curve.index)
    s = analytics.simulate_sip(res.equity_curve, amount, step_up_pct=step_up, dates=dates)
    b = analytics.simulate_sip(res.benchmark_curve, amount, step_up_pct=step_up, dates=dates) if has_bench else None
    c[2].metric("Instalments", s["n_instalments"])
    if not s["n_instalments"]:
        st.caption("Backtest too short for a SIP.")
        return
    k = st.columns(4)
    k[0].metric("Total invested", portfolio.round_lakh(s["invested"]))
    k[1].metric("Strategy value", portfolio.round_lakh(s["final_value"]),
                f"{s['final_value'] / s['invested'] - 1:+.0%}")
    k[2].metric("Strategy XIRR", _pct(s["xirr"]))
    if b is not None and b["n_instalments"]:
        k[3].metric(f"{bench_name} XIRR", _pct(b["xirr"]),
                    f"value {portfolio.round_lakh(b['final_value'])}", delta_color="off")
    tl = pd.DataFrame({"Strategy": s["timeline"]["value"], "Invested": s["timeline"]["invested"]})
    colors = {"Strategy": PAL["series"][0]}
    if b is not None and b["n_instalments"]:
        tl[bench_name] = b["timeline"]["value"]
        colors[bench_name] = PAL["series"][1]
    colors["Invested"] = PAL["muted"]
    st.altair_chart(charts.line_chart(tl, colors, "₹", PAL, ",.0f", height=300, dashed=("Invested",)),
                    width="stretch")


def _render_attribution(res: backtest.BacktestResult) -> None:
    st.subheader("🏆 Stock contributions")
    summ = analytics.summarize_contributions(res.contributions)
    if summ.empty:
        st.caption("No holdings — nothing to attribute.")
    else:
        cap = res.params.initial_capital if res.params else config.BACKTEST_INITIAL_CAPITAL
        st.caption(f"P&L each name added on a ₹{cap:,.0f} starting capital (before costs), summed over every "
                   "period it was held. Win rate = share of its holding periods that made money.")
        top = pd.concat([summ.head(10), summ.tail(10)]).drop_duplicates("Symbol")
        st.altair_chart(charts.diverging_bars(top, "Symbol", "contribution_inr", ",.1f",
                                              f"Contribution (₹ on ₹{cap:,.0f})", PAL,
                                              height=max(200, 22 * len(top))), width="stretch")
        view = summ.assign(
            avg_weight=(summ["avg_weight"] * 100).round(2), win_rate=(summ["win_rate"] * 100).round(0),
            held_return=(summ["held_return"] * 100).round(1), best_period=(summ["best_period"] * 100).round(1),
            worst_period=(summ["worst_period"] * 100).round(1), contribution_inr=summ["contribution_inr"].round(2),
        ).drop(columns=["contribution"]).rename(columns={
            "contribution_inr": "Contribution ₹", "periods_held": "Periods held", "avg_weight": "Avg wt %",
            "win_rate": "Win rate %", "held_return": "Return while held %", "best_period": "Best period %",
            "worst_period": "Worst period %"})
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={c: PCT_FMT for c in ["Avg wt %", "Win rate %", "Return while held %",
                                                         "Best period %", "Worst period %"]})
        pr = res.contributions
        st.caption(f"Picks hit rate: {_pct(float((pr['ret'] > 0).mean()), 0)} of {len(pr):,} position-periods "
                   f"were profitable · {summ.shape[0]} distinct names held.")

    st.subheader("🏭 Sector exposure over time")
    sx = analytics.sector_exposure(res.holdings, res.period_returns.index)
    if sx.empty:
        st.caption("No holdings recorded.")
        return
    st.altair_chart(charts.sector_area(sx, PAL), width="stretch")
    st.caption("Industry weights at each rebalance (top 7 + Other). Blank space = cash (regime filter or "
               "empty selections).")
    with st.expander("Sector weights table"):
        st.dataframe((sx * 100).round(1).rename_axis("Rebalance").reset_index(), hide_index=True,
                     width="stretch")


def _render_holdings_trades(res: backtest.BacktestResult, bench_name: str) -> None:
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
        if "exposure" in pr:
            tbl["Invested"] = (pr["exposure"] * 100).round(0)
        pct_cfg = {c: PCT_FMT for c in ["Strategy", bench_name, "Turnover", "Cost", "Invested"] if c in tbl}
        st.dataframe(tbl, hide_index=True, width="stretch", column_config=pct_cfg)

    # -- Rebalance schedule (shows holiday mapping) ------------------------- #
    with st.expander("Rebalance schedule (scheduled → trading day)"):
        st.dataframe(res.schedule, hide_index=True, width="stretch")

    # -- Holdings viewer ---------------------------------------------------- #
    with st.expander("Holdings by rebalance date"):
        dates = sorted(res.holdings)
        if not dates:
            st.caption("No holdings recorded (all-cash backtest).")
        else:
            i = st.selectbox(
                "Rebalance date", options=list(range(len(dates))),
                format_func=lambda j: dates[j].date().isoformat(),
                index=len(dates) - 1, key="bt_hold_date",
            )
            h = res.holdings.get(dates[i])
            hd = h.reset_index().rename(columns={"index": "YFTicker"})
            if "weight" in hd:
                hd["weight"] = (hd["weight"] * 100).round(2)
            if "score" in hd:
                hd["score"] = hd["score"].round(3)
            st.dataframe(hd, hide_index=True, width="stretch", column_config={"weight": PCT_FMT})
            st.bar_chart(h.set_index("Symbol")["weight"] * 100.0, height=280, color=PAL["series"][0])

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
                "are score/equal/inverse-vol (never market-cap-capped). Modelled cost = bps × traded value."
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


def _excel_sheets(bt: dict) -> dict[str, pd.DataFrame]:
    res: backtest.BacktestResult = bt["res"]
    bench = res.benchmark_curve
    cfg = bt.get("cfg", {})
    settings = pd.DataFrame({"setting": list(cfg) + ["universe", "benchmark source"],
                             "value": [str(v) for v in cfg.values()] + [bt.get("label", ""), bt.get("bench_source", "")]})
    eq = pd.DataFrame({"strategy": res.equity_curve})
    if bench is not None:
        eq["benchmark"] = bench
    holdings = (pd.concat({d: h for d, h in res.holdings.items()}, names=["rebalance_date", "YFTicker"]).reset_index()
                if res.holdings else pd.DataFrame())
    return {
        "Settings": settings,
        "Metrics": analytics.metrics_table(res.metrics, res.benchmark_metrics, res.relative_metrics),
        "Equity curve": eq,
        "Calendar returns": analytics.calendar_returns(res.equity_curve),
        "Annual returns": analytics.annual_returns(res.equity_curve, bench),
        "Drawdowns": analytics.drawdown_episodes(res.equity_curve, top=10),
        "Period returns": res.period_returns,
        "Holdings": holdings,
        "Membership changes": res.membership_changes,
        "Transactions": res.transactions,
        "Contributions": analytics.summarize_contributions(res.contributions),
        "Sector exposure": analytics.sector_exposure(res.holdings, res.period_returns.index),
        "Notes": pd.DataFrame({"note": res.notes}),
    }


def _render_downloads(bt: dict) -> None:
    res: backtest.BacktestResult = bt["res"]
    has_bench = res.benchmark_curve is not None
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

    st.markdown("**Full report (Excel)** — settings, metrics, equity, calendar & annual returns, drawdowns, "
                "periods, holdings, entries/exits, trades, contributions and sector exposure in one workbook.")
    token = id(res)
    cached = st.session_state.get("bt_xlsx")
    if st.button("📦 Build Excel report", key="bt_build_xlsx"):
        with st.spinner("Writing workbook…"):
            st.session_state["bt_xlsx"] = (token, analytics.excel_report(_excel_sheets(bt)))
        cached = st.session_state["bt_xlsx"]
    if cached and cached[0] == token:
        st.download_button("⬇️ Download Excel report", cached[1], file_name="momentum_backtest_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_xlsx")


def _render_backtest_results(bt: dict) -> None:
    res: backtest.BacktestResult = bt["res"]
    bench_name = bt["bench_name"]
    st.caption(f"**{bt.get('label', '')}** · {bt.get('freq', '')} rebalance · Top {bt.get('top_n', '')} · "
               f"{BT_WEIGHTING.get(bt.get('weighting'), bt.get('weighting'))} weighting · "
               f"{bt.get('cost_bps', 0):g} bps/side")
    tabs = st.tabs(["📊 Overview", "📅 Returns analysis", "💰 SIP", "🏆 Attribution & sectors",
                    "📒 Holdings & trades", "⬇️ Downloads"])
    with tabs[0]:
        _render_overview(res, bench_name, bt.get("bench_source", ""))
    with tabs[1]:
        _render_returns_analysis(res, bench_name)
    with tabs[2]:
        _render_sip(res, bench_name)
    with tabs[3]:
        _render_attribution(res)
    with tabs[4]:
        _render_holdings_trades(res, bench_name)
    with tabs[5]:
        _render_downloads(bt)


def render_backtest_controls() -> dict:
    """Backtest settings widgets → a plain dict (shared with the Compare tab)."""
    c = st.columns(4)
    bt_freq = c[0].selectbox(
        "Rebalance", FREQS, index=FREQS.index(config.DEFAULT_REBALANCE), key="bt_freq",
        help="How often the portfolio is re-ranked and traded. Daily/weekly react fastest but "
        "trade the most — pair them with a rank buffer and realistic costs.",
    )
    bt_weighting = c[1].radio(
        "Weighting", list(BT_WEIGHTING), format_func=BT_WEIGHTING.get, index=0, horizontal=True,
        key="bt_weighting",
        help="Score ∝ momentum score; Equal = 1/N; Inverse-vol ∝ 1/volatility (calmer names get more). "
        "Market-cap capping is disabled in backtests (look-ahead).",
    )
    bt_topn = c[2].number_input(
        "Holdings (Top N)", 1, 200, int(top_n), step=1, key="bt_topn_etf" if price_only_universe else "bt_topn",
        help="Backtest only the top-N momentum names each rebalance — e.g. set 10 for a "
        "concentrated top-10 portfolio. Defaults to the sidebar's Top N; every other "
        "parameter is unchanged.",
    )
    preset = c[3].selectbox("Trading costs", list(config.COST_PRESETS), index=0, key="bt_cost_preset",
                            help="Per-side cost on traded value. Statutory ≈ STT + stamp duty + exchange/SEBI "
                            "fees + GST on Indian delivery trades.")
    if config.COST_PRESETS[preset] is None:
        bt_cost = c[3].number_input("Cost (bps/side)", 0.0, 200.0, 20.0, step=5.0, key="bt_cost")
    else:
        bt_cost = float(config.COST_PRESETS[preset])

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

    c2 = st.columns([2, 1, 1])
    bt_bench = c2[0].selectbox(
        "Benchmark", BENCH_OPTIONS, index=0, key="bt_bench",
        help=f"Auto picks the index matching your universe (now: {auto_benchmark()}). "
        "'Universe equal-weight' holds every stock in your universe equally — it shares the same "
        "survivorship bias, so the gap to it is the momentum selection's real value-add.",
    )
    bt_tri = c2[1].checkbox(
        "Total Return Index (TRI)", value=True, key="bt_tri",
        help="Compare against the dividend-reinvested TRI (fair: the strategy uses dividend-adjusted "
        "prices). Off = price index, which flatters the strategy by the dividend yield.",
    )
    bt_rf = c2[2].number_input("Risk-free % p.a.", 0.0, 15.0, 0.0, step=0.5, key="bt_rf")

    with st.expander("🛡️ Risk controls & portfolio rules"):
        r = st.columns(3)
        buffer_on = r[0].checkbox("Rank buffer", value=False, key="bt_buffer_on",
                                  help="Keep a holding while it still ranks within the exit rank, instead of "
                                  "selling as soon as it leaves the Top N. Real momentum indices do this to cut churn.")
        buffer_mult = r[0].number_input("Exit rank = Top N ×", 1.0, 10.0, 2.0, step=0.5, key="bt_buffer_mult",
                                        disabled=not buffer_on)
        exit_rank = max(int(bt_topn), int(round(float(buffer_mult) * int(bt_topn))))
        if buffer_on:
            r[0].caption(f"→ hold names while ranked ≤ {exit_rank}")
        bt_max_sector = r[1].number_input("Max stocks per sector (0 = no cap)", 0, 50, int(max_sector), step=1,
                                          key="bt_max_sector")
        cash_yield = r[2].number_input("Cash yield % p.a.", 0.0, 15.0, 0.0, step=0.5, key="bt_cash_yield",
                                       help="Return earned on uninvested cash (regime risk-off or empty "
                                       "selections) — e.g. ~6% for a liquid fund/ETF.")
        g = st.columns(3)
        regime_on = g[0].checkbox(
            "Market-regime filter", value=False, key="bt_regime_on",
            help="At each rebalance, if the regime index closes below its moving average, cut exposure "
            "(to cash by default). A classic trend filter that softens deep bear-market drawdowns.",
        )
        regime_ma = g[0].number_input("Moving average (days)", 20, 400, config.REGIME_MA_DAYS, step=10,
                                      key="bt_regime_ma", disabled=not regime_on)
        regime_index = g[1].selectbox("Regime index", [REGIME_SAME] + [b for b, s in config.BENCHMARKS.items() if s.get("nse")],
                                      key="bt_regime_index", disabled=not regime_on)
        regime_off = g[2].slider("Exposure when risk-off (%)", 0, 100, 0, step=10, key="bt_regime_off",
                                 disabled=not regime_on)
    if bt_freq in ("Daily", "Weekly"):
        st.caption(f"⏱️ {bt_freq} rebalancing runs a scan every {'day' if bt_freq == 'Daily' else 'week'} — "
                   "expect ~30–90 s for large universes over many years.")
    return {
        "start": bt_start, "end": bt_end, "freq": bt_freq, "weighting": bt_weighting, "top_n": int(bt_topn),
        "cost_bps": float(bt_cost), "bench": bt_bench, "tri": bool(bt_tri), "rf": float(bt_rf),
        "buffer_on": bool(buffer_on), "exit_rank": int(exit_rank), "max_sector": int(bt_max_sector),
        "regime_on": bool(regime_on), "regime_ma": int(regime_ma), "regime_index": regime_index,
        "regime_off": float(regime_off), "cash_yield": float(cash_yield),
    }


def render_backtest_tab() -> dict:
    st.warning(BACKTEST_DISCLAIMER)
    cfg = render_backtest_controls()

    if st.button("🚀 Run backtest", type="primary", width="stretch", key="run_bt"):
        st.session_state["bt"] = _run_backtest(cfg)

    bt = st.session_state.get("bt")
    if not bt or bt.get("res") is None:
        st.info("Configure the backtest above and press **Run backtest**.")
        return cfg

    res: backtest.BacktestResult = bt["res"]
    if res.equity_curve is None or res.equity_curve.empty:
        for n in res.notes:
            st.warning(n)
        st.error(
            "Backtest produced no results for this configuration — the window is too "
            "short for a full holding period. Increase the history years or shorten the "
            "rebalance interval."
        )
        return cfg

    _render_backtest_results(bt)
    return cfg


# --------------------------------------------------------------------------- #
# Compare strategies
# --------------------------------------------------------------------------- #
METRIC_LABELS = {
    "sharpe": "Sharpe", "cagr": "CAGR", "calmar": "Calmar", "sortino": "Sortino",
    "max_dd": "Max drawdown", "ann_vol": "Volatility", "annual_turnover": "Turnover / yr",
    "total_return": "Total return", "hit_rate": "Hit rate", "alpha_annual": "Alpha", "info_ratio": "Info ratio",
}
PCT_METRICS = {"cagr", "max_dd", "ann_vol", "annual_turnover", "total_return", "hit_rate", "alpha_annual"}
LOWER_IS_BETTER = {"ann_vol", "annual_turnover"}


def _run_compare(cfg: dict, freqs: list[str], tops: list[int]):
    window = _validate_window(cfg)
    if window is None:
        return None
    if not freqs or not tops:
        st.error("Pick at least one rebalance frequency and one Top N value.")
        return None
    start, end = window
    cfg = {**cfg, "end": end}
    with st.status("Running strategy comparison…", expanded=True) as status:
        got = _bt_data(start, end)
        if got is None:
            status.update(label="Comparison failed", state="error")
            return None
        uni, pdata, hist_start = got
        bench_label, bench, regime = _bt_series(cfg, pdata, hist_start, end)
        bar = st.progress(0.0, text="Backtesting combinations…")
        def prog(done, total):
            bar.progress(min(done / total, 1.0), text=f"Combination {done}/{total}")
        table, curves = backtest.run_sweep(uni, pdata, _bt_params(cfg), [int(t) for t in tops], freqs,
                                           benchmark_series=bench, regime_series=regime, progress=prog)
        bench_curve = backtest._align_benchmark(bench, next(iter(curves.values())).index,
                                                next(iter(curves.values())).index[0]) if curves else None
        status.update(label="Comparison complete ✅", state="complete", expanded=False)
    return {"table": table, "curves": curves, "bench_name": bench_label, "bench_curve": bench_curve,
            "label": universe_label(), "cfg": cfg}


def render_compare_tab(cfg: dict) -> None:
    st.markdown(
        "Runs the **Backtest tab's settings** (universe, dates, weighting, costs, benchmark and risk "
        "controls) across a grid of rebalance frequencies × Top N. Prefer settings that do well across "
        "neighbouring cells — a lone bright cell is more likely luck than edge."
    )
    c = st.columns([2, 2, 1])
    freqs = c[0].multiselect("Rebalance frequencies", FREQS, default=["Monthly", "Quarterly", "Semi-annual"],
                             key="cmp_freqs")
    tops = c[1].multiselect("Top N values", [3, 5, 10, 15, 20, 25, 30, 40, 50],
                            default=[5, 10, 20] if price_only_universe else [10, 20, 30], key="cmp_topn")
    metric = c[2].selectbox("Colour by", list(METRIC_LABELS), format_func=METRIC_LABELS.get, key="cmp_metric")
    n_combo = len(freqs) * len(tops)
    st.caption(f"{n_combo} combination(s). Scans are shared across combinations, so cost ≈ the densest cadence."
               + (" Daily cadences are slow on big universes." if "Daily" in freqs else ""))
    if st.button("⚖️ Run comparison", type="primary", key="run_cmp"):
        st.session_state["cmp"] = _run_compare(cfg, freqs, tops)

    cmp = st.session_state.get("cmp")
    if not cmp:
        st.info("Pick the grid above and press **Run comparison**.")
        return
    table: pd.DataFrame = cmp["table"]
    if table.empty or table["cagr"].isna().all():
        st.error("No combination produced results — the window may be too short.")
        return

    st.subheader(f"{METRIC_LABELS[metric]} by frequency × Top N")
    fmt = ".1%" if metric in PCT_METRICS else ".2f"
    order = [f for f in FREQS if f in set(table["frequency"])]
    tips = {m: (".1%" if m in PCT_METRICS else ".2f") for m in ("cagr", "sharpe", "max_dd", "annual_turnover")}
    st.altair_chart(charts.metric_heatmap(table, metric, order, PAL, fmt, metric in LOWER_IS_BETTER, tips),
                    width="stretch")

    view = table.copy()
    for m in PCT_METRICS & set(view.columns):
        view[m] = (view[m] * 100).round(2)
    for m in ("sharpe", "sortino", "calmar", "info_ratio"):
        if m in view:
            view[m] = view[m].round(2)
    view = view.drop(columns=["label"]).rename(columns={**METRIC_LABELS, "frequency": "Frequency", "top_n": "Top N"})
    st.dataframe(view.sort_values(METRIC_LABELS[metric], ascending=metric in LOWER_IS_BETTER),
                 hide_index=True, width="stretch",
                 column_config={METRIC_LABELS[m]: PCT_FMT for m in PCT_METRICS if METRIC_LABELS[m] in view})

    st.subheader("Top 5 combinations — growth of ₹100")
    best = table.sort_values(metric, ascending=metric in LOWER_IS_BETTER).head(5)["label"].tolist()
    wide = pd.DataFrame({lab: cmp["curves"][lab] * 100.0 for lab in best if lab in cmp["curves"]})
    colors = {lab: PAL["series"][i] for i, lab in enumerate(wide.columns)}
    if cmp.get("bench_curve") is not None:
        wide[cmp["bench_name"]] = cmp["bench_curve"] * 100.0
        colors[cmp["bench_name"]] = PAL["muted"]
    st.altair_chart(charts.line_chart(wide, colors, "₹ (start = 100)", PAL, ",.0f", height=360,
                                      dashed=(cmp["bench_name"],)), width="stretch")
    st.download_button("⬇️ Comparison table (CSV)", table.to_csv(index=False).encode(),
                       file_name="strategy_comparison.csv", mime="text/csv", key="dl_cmp")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_scan, tab_bt, tab_cmp = st.tabs(["🔎 Scanner", "🧪 Backtest", "⚖️ Compare strategies"])
with tab_scan:
    render_scan_tab()
with tab_bt:
    _bt_cfg = render_backtest_tab()
with tab_cmp:
    render_compare_tab(_bt_cfg)
