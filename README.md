# Nifty500 Momentum 50 Scanner

A utility that scans NSE-listed Indian stocks and applies the **Nifty500 Momentum 50**
index construction rules (NSE *Methodology Document*, §20), with a lightweight
Streamlit UI for configuring and running the scan.

It ranks a configurable universe (Largecap / Midcap / Small) by a
**volatility-adjusted momentum score**, selects the top N, and computes
free-float-cap × score portfolio weights with the methodology's 5% / 5× capping.

> ⚠️ **Educational tool, not investment advice.** It reproduces the *published
> methodology* as faithfully as freely-available data allows — see
> [Limitations & deviations](#limitations--deviations).

---

## What it does

1. **Builds the universe** from NSE index constituent lists. Choose either
   composable **size segments** or a **whole index** in one pick:

   | Universe source | NSE index          | Size |
   |-----------------|--------------------|------|
   | Largecap        | Nifty 100          | 100  |
   | Midcap          | Nifty Midcap 150   | 150  |
   | Small           | Nifty Smallcap 250 | 250  |
   | Nifty 500       | Nifty 500          | 500  |
   | Nifty Total Market | Nifty Total Market | ~750 |

   The size segments' union ≈ the Nifty 500 (the index's base universe). The
   **selected universe defines the eligible set** over which Z-scores are
   normalized and the top N are chosen.

2. **Applies eligibility filters** (those computable from price/volume):
   minimum 1-year listing history, exclude bottom-10-percentile by 6-month
   average daily turnover, and exclude bottom-10-percentile by turnover ratio.

3. **Scores momentum** exactly per the methodology (see mapping below). The
   momentum windows are **tunable** from the sidebar — long/short lookback
   months, a *skip-month* toggle for "12-1"-style momentum, the volatility
   window, and the long/short Z-score weighting.

4. **Selects & weights** the top N, capping each weight at `min(5%, 5×` its
   free-float-mcap-only weight`)` and redistributing the excess.

Results are shown as a ranked, sortable table — including each stock's **actual
closing price** on the as-of date — with a weights chart and CSV downloads (both
the selection and the full audited universe).

The app has two tabs sharing one sidebar configuration:

- **🔎 Scanner** — the point-in-time scan described above.
- **🧪 Backtest** — rebalances the strategy over a defined period (monthly /
  quarterly / semi-annual / annual), on **multi-year history** (up to 15 years),
  and compares it to a **benchmark** (Nifty 50 / Nifty 500). It reports an equity
  curve (growth of ₹100), drawdown, and a full metrics panel — CAGR, annualised
  volatility, Sharpe, Sortino, max drawdown (with peak/trough/recovery dates),
  Calmar, turnover, hit rate, and vs-benchmark beta / alpha / tracking error /
  information ratio / up-down capture — plus per-rebalance period returns, the
  rebalance schedule, a holdings viewer, and CSV exports. Optional per-side
  trading-cost (bps) modelling is included.

### Backtest honesty (important)

The backtest is deliberately **look-ahead-free**: selection at each rebalance
uses only prices up to that date, and weights are **score- or equal-weighted
only** — the live market-cap capping is *not* used historically (market cap is a
current-only snapshot). The turnover-ratio eligibility filter is disabled in
backtests for the same reason. The universe is *today's* constituents, so results
carry **survivorship bias**. These limitations are shown as a permanent disclaimer
on the Backtest tab and in every result's notes. The momentum *ranking* is
unbiased; the *absolute-return* figures are indicative. See
[Limitations](#limitations--deviations).

---

## Setup

Requires Python 3.11–3.13 (developed on 3.13). A virtual environment is used:

```bash
cd nifty-momentum
python3.13 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

> **Note on Python 3.14:** at time of writing some dependencies (e.g. `pyarrow`)
> ship wheels for 3.13 more reliably than 3.14. If you hit a build error on
> 3.14, create the venv with `python3.13` as shown above.

## Run

```bash
./venv/bin/streamlit run app.py
```

Then open the URL it prints (default <http://localhost:8501>), pick your
universe segments in the sidebar, and press **🚀 Run scan**.

The first scan downloads ~13 months of daily prices plus per-ticker market caps
from Yahoo Finance; results are disk-cached under `data/` so subsequent scans are
fast. Use **Force refresh** to bypass the cache.

## Deploy as a web app (Streamlit Community Cloud)

Give non-technical users a plain link — no install, any OS or phone.

1. Push this repo to GitHub (already done if you cloned it from there).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **Create app → Deploy a public app from GitHub**, then select:
   - **Repository:** `manitgupta/nifty-momentum-scanner`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - (Optional) **Advanced settings → Python version:** 3.13
4. Click **Deploy**. First build installs `requirements.txt` (a few minutes);
   afterwards you get a shareable `https://<app>.streamlit.app` URL.

The app ships with bundled fallback constituent lists (`momentum/seed_data/`), so
it still loads the universe even if NSE blocks the cloud's IP. Price data comes
from Yahoo Finance at scan time; on shared cloud IPs Yahoo may occasionally
rate-limit large (500-stock) scans — rerun, or start with a single segment.

## Run the tests

```bash
./venv/bin/python tests/test_scanner.py    # scanner maths
./venv/bin/python tests/test_backtest.py   # backtest engine
./venv/bin/python tests/test_app.py        # Streamlit UI (headless AppTest, offline)
```

- **`test_scanner.py`** — as-of price lookup, the Z-score → Normalized Momentum
  Score piecewise formula (both branches), the iterative weight-capping algorithm,
  skip-month / configurable-lookback momentum, the raw closing-price column, and an
  end-to-end ranking sanity check on synthetic prices.
- **`test_backtest.py`** — rebalance scheduling (incl. holiday snapping), the
  buy-and-hold return identity, a **no-look-ahead invariance** check, compounding
  continuity, transaction-cost monotonicity, closed-form analytics
  (CAGR/vol/drawdown), benchmark alignment, and edge cases (missing benchmark,
  pre-feasibility window, cash periods).
- **`test_app.py`** — drives the Streamlit app headlessly with
  `streamlit.testing.v1.AppTest` and monkeypatched (offline) data: both tabs load,
  the Scanner still works, the Backtest runs and renders charts/metrics/tables, a
  scan→backtest session shares no `st.stop()` leakage, and benchmark-unavailable
  degrades gracefully.

---

## Methodology → code mapping (PDF §20)

| Methodology step | Where |
|---|---|
| σₚ = annualised std-dev of lognormal daily returns over 1 year | `scanner.compute_momentum` |
| 12M return = `P(asof)/P(asof−12m) − 1`; 6M analogously | `scanner.compute_momentum` |
| Momentum Ratio `MR = return / σₚ` (6M & 12M) | `scanner.compute_momentum` |
| Z-score `(MR − μ)/σ` across the eligible universe | `scanner.normalize_and_score` |
| Weighted-Avg Z = `0.5·Z₁₂ + 0.5·Z₆` (configurable) | `scanner.normalize_and_score` |
| Normalized Momentum Score `= 1+WAZ` (WAZ≥0) else `1/(1−WAZ)` | `scanner.normalize_and_score` |
| Top 50 by score | `scanner.select_and_weight` |
| Weight = free-float-mcap × score, capped at `min(5%, 5×ff-mcap-weight)` | `scanner.select_and_weight` / `_iterative_cap` |
| ≥1-year listing history; bottom-10% turnover; bottom-10% turnover ratio | `scanner.apply_eligibility` |

All parameters live in `momentum/config.py` and are overridable from the UI.

---

## Project structure

```
nifty-momentum/
├── app.py                 # Streamlit UI (Scanner + Backtest tabs)
├── requirements.txt
├── momentum/
│   ├── config.py          # methodology parameters + data-source config
│   ├── universe.py        # NSE constituent lists -> eligible universe
│   ├── data.py            # Yahoo Finance price/volume + market-cap + benchmark (disk-cached)
│   ├── scanner.py         # eligibility, momentum, scoring, selection, weighting
│   └── backtest.py        # rebalancing backtest engine + analytics (look-ahead-free)
├── tests/
│   ├── test_scanner.py    # scanner maths
│   ├── test_backtest.py   # backtest engine
│   └── test_app.py        # headless Streamlit UI (AppTest)
└── data/                  # on-disk cache (gitignored)
```

---

## Limitations & deviations

The core momentum ranking is reproduced faithfully. Some methodology inputs are
not available through free data sources, so they are handled transparently:

- **Pledged promoter shares > 20% filter** — *not applied.* Shown as a disabled
  toggle in the UI. This data is not in Yahoo Finance and NSE does not expose it
  as a clean bulk feed.
- **Circuit / price-band hit rule** — *not applied.* Requires per-day upper/lower
  circuit flags over 6 months; not reliably sourceable here. Shown disabled.
- **Free-float market cap** — approximated by **full market cap** (Yahoo's
  `fast_info.market_cap`). A *Free-float factor* slider lets you scale it. This
  affects the turnover-ratio filter and the weighting, not the stock ranking.
- **Turnover** uses adjusted-close × volume as a proxy for traded value —
  adequate for a relative percentile filter.
- **Universe for normalization** — Z-scores are computed over the *selected
  segments*, not always the full Nifty 500, so choosing only Midcap (say)
  produces a self-contained midcap momentum scan.
- **Live as-of vs. semi-annual rebalance** — the official index reconstitutes
  semi-annually (Jun/Dec) with buffer rules; this tool computes a fresh snapshot
  as of any date you choose and does not model the turnover-reducing buffer.

### Backtesting caveats (free-data honesty)

The momentum **ranking** is price-only, so it is unbiased at any historical date.
Two limitations mean an *absolute-return* backtest should be read as indicative,
not precise — both are surfaced in the UI rather than hidden:

- **Survivorship bias** — the universe uses *today's* index constituents. A
  backtest of, say, 2018 therefore excludes stocks later delisted or demoted,
  which flatters returns. True point-in-time NSE membership is not in a free
  feed. (You can supply historical constituent lists to reduce this.)
- **Current-snapshot market cap** — free-float mcap is only available *now*
  (`fast_info`), so historical mcap-cap weighting would look ahead. Backtests
  therefore default to **price-only weighting** (score- or equal-weight); the
  5%/5× free-float capping is applied to live snapshots only.

Data sources: index constituents from NSE
(`nsearchives.nseindia.com/content/indices/*.csv`) and prices/market caps from
Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance).
