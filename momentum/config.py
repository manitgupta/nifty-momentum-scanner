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
# Largecap = Nifty 100, Midcap = Nifty Midcap 150, Small = Nifty Smallcap 250.
# Their union is (approximately) the Nifty 500, which is the index's base universe.
# --------------------------------------------------------------------------- #
NSE_BASE = "https://nsearchives.nseindia.com/content/indices"
SEGMENT_URLS: dict[str, str] = {
    "Largecap": f"{NSE_BASE}/ind_nifty100list.csv",
    "Midcap": f"{NSE_BASE}/ind_niftymidcap150list.csv",
    "Small": f"{NSE_BASE}/ind_niftysmallcap250list.csv",
}
SEGMENTS = list(SEGMENT_URLS.keys())

# Yahoo Finance ticker suffix for NSE-listed equities.
YF_SUFFIX = ".NS"

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
MONTHS_12 = 12               # 12-month momentum lookback
MONTHS_6 = 6                 # 6-month momentum lookback

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
# present even across long holidays; ~15 months of history is fetched.
PRICE_LOOKBACK_DAYS = 400
MARKETCAP_MAX_WORKERS = 12  # threads for per-ticker fast_info calls
DOWNLOAD_BATCH_SIZE = 100   # tickers per yfinance.download batch
