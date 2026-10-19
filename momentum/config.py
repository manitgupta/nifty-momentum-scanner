"""Central configuration for the Nifty500 Momentum 50 scanner.

All tunable constants live here so the methodology parameters (PDF section 20,
"Nifty500 Momentum 50") are visible in one place and easy to audit.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths / caching
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Bundled fallback copies of the constituent lists, committed to the repo. Used
# when NSE is unreachable (e.g. cloud IPs are sometimes blocked) and no fresh
# on-disk cache exists — so the app still works on Streamlit Cloud.
SEED_DIR = Path(__file__).resolve().parent / "seed_data"

# How long a cached artefact is considered fresh, in hours.
UNIVERSE_CACHE_HOURS = 24
PRICE_CACHE_HOURS = 6
MARKETCAP_CACHE_HOURS = 24

# --------------------------------------------------------------------------- #
# Universe (index constituents) — NSE publishes these as plain CSV downloads.
#
# Two flavours are offered:
#   * Size segments — Largecap = Nifty 100, Midcap = Nifty Midcap 150,
#     Small = Nifty Smallcap 250. Their union is (approximately) the Nifty 500.
#   * Whole-index universes — the Nifty 500 and the Nifty Total Market (~750
#     names, i.e. Nifty 500 + Microcap 250) downloaded directly as a single list.
#
# All share NSE's column layout (Company Name, Industry, Symbol, Series, ISIN).
# --------------------------------------------------------------------------- #
NSE_BASE = "https://nsearchives.nseindia.com/content/indices"

# Size buckets whose union approximates the Nifty 500.
SIZE_SEGMENT_URLS: dict[str, str] = {
    "Largecap": f"{NSE_BASE}/ind_nifty100list.csv",
    "Midcap": f"{NSE_BASE}/ind_niftymidcap150list.csv",
    "Small": f"{NSE_BASE}/ind_niftysmallcap250list.csv",
}

# Whole-index universes downloaded as one CSV each: broad-market, sectoral,
# thematic and strategy indices. Every key has a committed seed CSV fallback
# (``seed_data/constituents_<key.lower()>.csv``).
INDEX_UNIVERSE_URLS: dict[str, str] = {
    "Nifty500": f"{NSE_BASE}/ind_nifty500list.csv",
    "NiftyTotalMarket": f"{NSE_BASE}/ind_niftytotalmarket_list.csv",
    # broad market
    "Nifty50": f"{NSE_BASE}/ind_nifty50list.csv",
    "NiftyNext50": f"{NSE_BASE}/ind_niftynext50list.csv",
    "Nifty200": f"{NSE_BASE}/ind_nifty200list.csv",
    "NiftyMidcap100": f"{NSE_BASE}/ind_niftymidcap100list.csv",
    "NiftySmallcap100": f"{NSE_BASE}/ind_niftysmallcap100list.csv",
    "NiftyMicrocap250": f"{NSE_BASE}/ind_niftymicrocap250_list.csv",
    "NiftyLargeMidcap250": f"{NSE_BASE}/ind_niftylargemidcap250list.csv",
    "NiftyMidSmallcap400": f"{NSE_BASE}/ind_niftymidsmallcap400list.csv",
    # sectoral
    "NiftyBank": f"{NSE_BASE}/ind_niftybanklist.csv",
    "NiftyFinancialServices": f"{NSE_BASE}/ind_niftyfinancelist.csv",
    "NiftyPrivateBank": f"{NSE_BASE}/ind_nifty_privatebanklist.csv",
    "NiftyPSUBank": f"{NSE_BASE}/ind_niftypsubanklist.csv",
    "NiftyIT": f"{NSE_BASE}/ind_niftyitlist.csv",
    "NiftyPharma": f"{NSE_BASE}/ind_niftypharmalist.csv",
    "NiftyHealthcare": f"{NSE_BASE}/ind_niftyhealthcarelist.csv",
    "NiftyAuto": f"{NSE_BASE}/ind_niftyautolist.csv",
    "NiftyFMCG": f"{NSE_BASE}/ind_niftyfmcglist.csv",
    "NiftyMetal": f"{NSE_BASE}/ind_niftymetallist.csv",
    "NiftyRealty": f"{NSE_BASE}/ind_niftyrealtylist.csv",
    "NiftyEnergy": f"{NSE_BASE}/ind_niftyenergylist.csv",
    "NiftyOilGas": f"{NSE_BASE}/ind_niftyoilgaslist.csv",
    "NiftyMedia": f"{NSE_BASE}/ind_niftymedialist.csv",
    "NiftyConsumerDurables": f"{NSE_BASE}/ind_niftyconsumerdurableslist.csv",
    # thematic & strategy
    "NiftyInfra": f"{NSE_BASE}/ind_niftyinfralist.csv",
    "NiftyIndiaConsumption": f"{NSE_BASE}/ind_niftyconsumptionlist.csv",
    "NiftyCPSE": f"{NSE_BASE}/ind_niftycpselist.csv",
    "NiftyPSE": f"{NSE_BASE}/ind_niftypselist.csv",
    "NiftyIndiaDefence": f"{NSE_BASE}/ind_niftyindiadefence_list.csv",
    "NiftyCommodities": f"{NSE_BASE}/ind_niftycommoditieslist.csv",
    "NiftyMNC": f"{NSE_BASE}/ind_niftymnclist.csv",
    "NiftyServices": f"{NSE_BASE}/ind_niftyservicelist.csv",
    "Nifty200Momentum30": f"{NSE_BASE}/ind_nifty200Momentum30_list.csv",
    "Nifty500Momentum50": f"{NSE_BASE}/ind_nifty500Momentum50_list.csv",
}

# Combined lookup; size segments first so that when a stock appears in several
# selected universes the size-bucket label wins during de-duplication.
SEGMENT_URLS: dict[str, str] = {**SIZE_SEGMENT_URLS, **INDEX_UNIVERSE_URLS}

# Non-NSE-index universes: a curated ETF list (committed CSV) and a free-form
# list of NSE symbols typed by the user.
ETF_SEGMENT = "ETF"
CUSTOM_SEGMENT = "Custom"
ETF_LIST_PATH = SEED_DIR / "etfs.csv"

SEGMENTS = list(SEGMENT_URLS.keys()) + [ETF_SEGMENT]
SIZE_SEGMENTS = list(SIZE_SEGMENT_URLS.keys())
INDEX_UNIVERSES = list(INDEX_UNIVERSE_URLS.keys())

# Human labels for the "Index" universe picker, grouped. Nifty 100 / Midcap 150 /
# Smallcap 250 reuse the size-segment keys (same NSE lists).
INDEX_GROUPS: dict[str, dict[str, str]] = {
    "Broad market": {
        "Nifty 50": "Nifty50",
        "Nifty Next 50": "NiftyNext50",
        "Nifty 100": "Largecap",
        "Nifty 200": "Nifty200",
        "Nifty 500": "Nifty500",
        "Nifty Midcap 150": "Midcap",
        "Nifty Midcap 100": "NiftyMidcap100",
        "Nifty Smallcap 250": "Small",
        "Nifty Smallcap 100": "NiftySmallcap100",
        "Nifty Microcap 250": "NiftyMicrocap250",
        "Nifty LargeMidcap 250": "NiftyLargeMidcap250",
        "Nifty MidSmallcap 400": "NiftyMidSmallcap400",
        "Nifty Total Market (~750)": "NiftyTotalMarket",
    },
    "Sectoral": {
        "Nifty Bank": "NiftyBank",
        "Nifty Financial Services": "NiftyFinancialServices",
        "Nifty Private Bank": "NiftyPrivateBank",
        "Nifty PSU Bank": "NiftyPSUBank",
        "Nifty IT": "NiftyIT",
        "Nifty Pharma": "NiftyPharma",
        "Nifty Healthcare": "NiftyHealthcare",
        "Nifty Auto": "NiftyAuto",
        "Nifty FMCG": "NiftyFMCG",
        "Nifty Metal": "NiftyMetal",
        "Nifty Realty": "NiftyRealty",
        "Nifty Energy": "NiftyEnergy",
        "Nifty Oil & Gas": "NiftyOilGas",
        "Nifty Media": "NiftyMedia",
        "Nifty Consumer Durables": "NiftyConsumerDurables",
    },
    "Thematic & strategy": {
        "Nifty Infrastructure": "NiftyInfra",
        "Nifty India Consumption": "NiftyIndiaConsumption",
        "Nifty CPSE": "NiftyCPSE",
        "Nifty PSE": "NiftyPSE",
        "Nifty India Defence": "NiftyIndiaDefence",
        "Nifty Commodities": "NiftyCommodities",
        "Nifty MNC": "NiftyMNC",
        "Nifty Services Sector": "NiftyServices",
        "Nifty200 Momentum 30": "Nifty200Momentum30",
        "Nifty500 Momentum 50": "Nifty500Momentum50",
    },
}
INDEX_CHOICES: dict[str, str] = {lab: key for grp in INDEX_GROUPS.values() for lab, key in grp.items()}
SEGMENT_LABELS: dict[str, str] = {
    **{key: lab for lab, key in INDEX_CHOICES.items()},
    ETF_SEGMENT: "ETFs", CUSTOM_SEGMENT: "Custom list",
}

# Yahoo Finance ticker suffix for NSE-listed equities.
YF_SUFFIX = ".NS"

# --------------------------------------------------------------------------- #
# Benchmark indices for backtests. Each entry may carry:
#   nse   — index name on niftyindices.com (full daily history since ~2005, and
#           a Total Return Index (TRI) series that includes dividends);
#   yahoo — a Yahoo Finance symbol (price-only) used as a fallback, or as the
#           only source for ETF-based benchmarks.
# The strategy is valued on dividend-ADJUSTED closes, so the TRI is the
# apples-to-apples comparison; a price index flatters the strategy by the
# dividend yield (~1-1.5% a year).
# --------------------------------------------------------------------------- #
NIFTYINDICES_URL = "https://www.niftyindices.com/BackPage"
BENCHMARKS: dict[str, dict[str, str]] = {
    "Nifty 50": {"nse": "NIFTY 50", "yahoo": "^NSEI"},
    "Nifty Next 50": {"nse": "NIFTY NEXT 50", "yahoo": "^NSMIDCP"},
    "Nifty 100": {"nse": "NIFTY 100", "yahoo": "^CNX100"},
    "Nifty 200": {"nse": "NIFTY 200", "yahoo": "^CNX200"},
    "Nifty 500": {"nse": "NIFTY 500", "yahoo": "^CRSLDX"},
    "Nifty Midcap 150": {"nse": "NIFTY MIDCAP 150", "yahoo": "NIFTYMIDCAP150.NS"},
    "Nifty Midcap 100": {"nse": "NIFTY MIDCAP 100"},
    "Nifty Smallcap 250": {"nse": "NIFTY SMALLCAP 250", "yahoo": "NIFTYSMLCAP250.NS"},
    "Nifty Smallcap 100": {"nse": "NIFTY SMALLCAP 100"},
    "Nifty Microcap 250": {"nse": "NIFTY MICROCAP 250"},
    "Nifty LargeMidcap 250": {"nse": "NIFTY LARGEMIDCAP 250"},
    "Nifty MidSmallcap 400": {"nse": "NIFTY MIDSMALLCAP 400"},
    "Nifty Total Market": {"nse": "NIFTY TOTAL MARKET"},
    "Nifty Bank": {"nse": "NIFTY BANK", "yahoo": "^NSEBANK"},
    "Nifty Financial Services": {"nse": "NIFTY FINANCIAL SERVICES"},
    "Nifty Private Bank": {"nse": "NIFTY PRIVATE BANK"},
    "Nifty PSU Bank": {"nse": "NIFTY PSU BANK"},
    "Nifty IT": {"nse": "NIFTY IT", "yahoo": "^CNXIT"},
    "Nifty Pharma": {"nse": "NIFTY PHARMA", "yahoo": "^CNXPHARMA"},
    "Nifty Healthcare": {"nse": "NIFTY HEALTHCARE"},
    "Nifty Auto": {"nse": "NIFTY AUTO"},
    "Nifty FMCG": {"nse": "NIFTY FMCG"},
    "Nifty Metal": {"nse": "NIFTY METAL"},
    "Nifty Realty": {"nse": "NIFTY REALTY"},
    "Nifty Energy": {"nse": "NIFTY ENERGY"},
    "Nifty Infrastructure": {"nse": "NIFTY INFRASTRUCTURE"},
    "Nifty200 Momentum 30": {"nse": "NIFTY200 MOMENTUM 30"},
    "Nifty500 Momentum 50": {"nse": "NIFTY500 MOMENTUM 50"},
    "Gold (GOLDBEES ETF)": {"yahoo": "GOLDBEES.NS"},
    "Nasdaq 100 (MON100 ETF)": {"yahoo": "MON100.NS"},
}
DEFAULT_BENCHMARK = "Nifty 500"
# Pseudo-benchmarks resolved in the app rather than fetched by name.
AUTO_BENCHMARK = "Auto (match universe)"
UNIVERSE_EW_BENCHMARK = "Universe equal-weight (same stocks)"

# Which benchmark "Auto" resolves to for a single selected universe.
UNIVERSE_BENCHMARK: dict[str, str] = {
    "Largecap": "Nifty 100", "Midcap": "Nifty Midcap 150", "Small": "Nifty Smallcap 250",
    "Nifty500": "Nifty 500", "NiftyTotalMarket": "Nifty Total Market",
    "Nifty50": "Nifty 50", "NiftyNext50": "Nifty Next 50", "Nifty200": "Nifty 200",
    "NiftyMidcap100": "Nifty Midcap 100", "NiftySmallcap100": "Nifty Smallcap 100",
    "NiftyMicrocap250": "Nifty Microcap 250", "NiftyLargeMidcap250": "Nifty LargeMidcap 250",
    "NiftyMidSmallcap400": "Nifty MidSmallcap 400",
    "NiftyBank": "Nifty Bank", "NiftyFinancialServices": "Nifty Financial Services",
    "NiftyPrivateBank": "Nifty Private Bank", "NiftyPSUBank": "Nifty PSU Bank",
    "NiftyIT": "Nifty IT", "NiftyPharma": "Nifty Pharma", "NiftyHealthcare": "Nifty Healthcare",
    "NiftyAuto": "Nifty Auto", "NiftyFMCG": "Nifty FMCG", "NiftyMetal": "Nifty Metal",
    "NiftyRealty": "Nifty Realty", "NiftyEnergy": "Nifty Energy", "NiftyInfra": "Nifty Infrastructure",
    "Nifty200Momentum30": "Nifty200 Momentum 30", "Nifty500Momentum50": "Nifty500 Momentum 50",
    ETF_SEGMENT: "Nifty 50",
}
# Size-segment combinations map to the matching composite index.
SEGMENT_COMBO_BENCHMARK: dict[frozenset, str] = {
    frozenset({"Largecap", "Midcap"}): "Nifty LargeMidcap 250",
    frozenset({"Midcap", "Small"}): "Nifty MidSmallcap 400",
    frozenset({"Largecap", "Midcap", "Small"}): "Nifty 500",
}

# Browser-like headers; NSE archives reject requests without a User-Agent.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
HTTP_TIMEOUT = 30

# --------------------------------------------------------------------------- #
# Momentum methodology parameters (PDF section 20)
# --------------------------------------------------------------------------- #
TRADING_DAYS_PER_YEAR = 252  # for annualising daily-return volatility (sigma_p)
MONTHS_12 = 12               # long-leg momentum lookback (methodology: 12 months)
MONTHS_6 = 6                 # short-leg momentum lookback (methodology: 6 months)

# Skip-month ("12-1" momentum): exclude the most recent N months from BOTH
# lookbacks to sidestep short-term reversal. 0 = methodology default (no skip).
SKIP_MONTHS = 0

# Trailing window (months) over which sigma_p is measured. Methodology uses the
# trailing year; exposed so it can be tuned independently of the lookbacks.
VOL_LOOKBACK_MONTHS = 12

# Weighted Average Z Score = W12 * Z12 + W6 * Z6  (methodology uses 50/50).
WEIGHT_12M = 0.50
WEIGHT_6M = 0.50

TOP_N = 50  # "Top 50 stocks with the highest Normalized Momentum Score are selected"

# Weight capping: "capped at the lower of 5% or 5 times the weight of the stock
# in the index based only on free-float market capitalization".
CAP_ABSOLUTE = 0.05  # 5%
CAP_RELATIVE = 5.0   # 5x the free-float-mcap-only weight
CAP_MAX_ITERATIONS = 100

# Eligibility parameters.
MIN_LISTING_CALENDAR_DAYS = 365   # "minimum listing history of 1 year"
MIN_OBSERVATIONS = 200            # data-quality floor: valid daily closes required in window
LIQUIDITY_BOTTOM_PERCENTILE = 10  # drop bottom 10 percentile on liquidity metrics
TURNOVER_LOOKBACK_MONTHS = 6      # "6 month average daily turnover"

# Degrees of freedom for the trailing daily-return volatility (sigma_p). 1 = sample.
VOL_DDOF = 1

# Cross-sectional std-dev degrees of freedom used when computing Z-scores across
# the eligible universe. 0 = population, 1 = sample. NSE does not specify; with a
# universe of 100-500 names the choice barely affects the ranking.
ZSCORE_DDOF = 0

# --------------------------------------------------------------------------- #
# Data-fetch tuning
# --------------------------------------------------------------------------- #
# Extra calendar buffer beyond 12 months so the 12M-ago trading day is always
# present even across long holidays; ~15 months of history is fetched for a
# single live scan.
PRICE_LOOKBACK_DAYS = 400
MARKETCAP_MAX_WORKERS = 12  # threads for per-ticker fast_info calls
DOWNLOAD_BATCH_SIZE = 100   # tickers per yfinance.download batch

# --------------------------------------------------------------------------- #
# Backtesting
# --------------------------------------------------------------------------- #
# Default depth of price history to fetch for backtests. The request is "more
# than 7-8 years"; 9 years gives a comfortable margin and still leaves a full
# 12-month lookback available before the first rebalance.
BACKTEST_DEFAULT_YEARS = 9
BACKTEST_MIN_YEARS = 1
BACKTEST_MAX_YEARS = 15
# Notional starting capital (₹) used to denominate the transaction ledger and the
# "Growth of ₹100" chart. Purely a display scale — the equity curve itself is
# unitless (base 1.0), so changing this never affects returns or metrics.
BACKTEST_INITIAL_CAPITAL = 100.0
# Long-history downloads are cached far longer than the 6h live-scan cache: the
# deep past does not change, so re-fetching multi-year frames is wasteful.
HISTORY_CACHE_HOURS = 24 * 7

# Rebalance cadences: label -> (unit, step). "days" steps through actual trading
# days; "weeks"/"months" step through the calendar and snap each date back to
# the last trading day on/before it.
REBALANCE_FREQUENCIES: dict[str, tuple[str, int]] = {
    "Daily": ("days", 1),
    "Weekly": ("weeks", 1),
    "Fortnightly": ("weeks", 2),
    "Monthly": ("months", 1),
    "Quarterly": ("months", 3),
    "Semi-annual": ("months", 6),
    "Annual": ("months", 12),
}
DEFAULT_REBALANCE = "Semi-annual"

# Per-side trading-cost presets (bps of traded value) for Indian equity delivery.
# Statutory ≈ STT 0.1% + stamp duty 0.015% (buy) + exchange/SEBI fees + 18% GST
# on fees ≈ 11 bps per side on average; slippage/brokerage come on top.
COST_PRESETS: dict[str, float | None] = {
    "Frictionless (0 bps)": 0.0,
    "Statutory only (~11 bps)": 11.0,
    "Discount broker + slippage (~20 bps)": 20.0,
    "Full-service broker (~50 bps)": 50.0,
    "Custom": None,
}

# Market-regime (trend) filter defaults: be invested only while the regime index
# closes above its N-day simple moving average at a rebalance.
REGIME_MA_DAYS = 200

# SIP simulator defaults (monthly instalment, ₹).
SIP_DEFAULT_AMOUNT = 10_000.0
