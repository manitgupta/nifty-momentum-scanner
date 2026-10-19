# Nifty500 Momentum 50 Scanner

A utility that scans NSE-listed Indian stocks and ETFs and applies the **Nifty500 Momentum 50**
index construction rules (NSE *Methodology Document*, §20), with a Streamlit UI for
scanning, building an order list, backtesting and comparing strategy settings.

It ranks a configurable universe — size segments, any NSE broad / sectoral / thematic
index, a curated ETF basket or your own symbol list — by a **volatility-adjusted
momentum score**, selects the top N, and weights them either by the methodology's
free-float-cap × score with 5% / 5× capping, or by price-only schemes (score, equal,
inverse-volatility).

> ⚠️ **Educational tool, not investment advice.** It reproduces the *published
> methodology* as faithfully as freely-available data allows — see
> [Limitations & deviations](#limitations--deviations).

---

## What it does

1. **Builds the universe.** Four sources, picked in the sidebar:

   | Universe source | What you get |
   |-----------------|--------------|
   | **Size segments** | Largecap (Nifty 100) / Midcap (Midcap 150) / Small (Smallcap 250) / Micro (Microcap 250) — Large + Mid + Small ≈ Nifty 500; all four ≈ Nifty Total Market |
   | **Index** (multi-select) | *Broad:* Nifty 50, Next 50, 100, 200, 500, Midcap 150/100, Smallcap 250/100, **Microcap 250**, LargeMidcap 250, MidSmallcap 400, **Total Market (~750)**. *Sectoral:* **Bank**, **Financial Services**, Private Bank, PSU Bank, IT, Pharma, Healthcare, Auto, FMCG, Metal, Realty, Energy, Oil & Gas, Media, Consumer Durables. *Thematic/strategy:* Infrastructure, India Consumption, CPSE, PSE, India Defence, Commodities, MNC, Services, Nifty200 Momentum 30, Nifty500 Momentum 50. Several indices combine (de-duplicated). |
   | **ETFs** | A curated basket of 43 liquid NSE ETFs — one per underlying: broad equity, factor (momentum, alpha, low-vol, quality, value), sector, international (Nasdaq 100, FANG+, S&P 500 Top 50, Hang Seng), gold, silver, gilt and liquid. Pick any subset and add more symbols — a **custom ETF portfolio** for momentum rotation. |
   | **Custom list** | Any NSE symbols you type (stocks and/or ETFs). Names/industries are filled from the bundled lists. |

   The **selected universe defines the eligible set** over which Z-scores are
   normalized and the top N are chosen. Every NSE index list has a committed
   fallback copy in `momentum/seed_data/` (NSE sometimes blocks cloud IPs).

2. **Applies eligibility filters** (those computable from price/volume):
   minimum 1-year listing history, exclude bottom-10-percentile by 6-month
   average daily turnover, and exclude bottom-10-percentile by turnover ratio.

3. **Scores momentum** exactly per the methodology (see mapping below). The
   momentum windows are **tunable** from the sidebar — long/short lookback
   months, a *skip-month* toggle for "12-1"-style momentum, the volatility
   window, and the long/short Z-score weighting.

4. **Selects & weights** the top N — optionally with **at most N names per
   sector** — using the methodology weighting (capping each weight at `min(5%, 5×`
   its free-float-mcap-only weight`)` and redistributing the excess) or a
   price-only scheme: **score**, **equal** or **inverse-volatility** (the default
   for ETFs / custom lists, which have no market caps).

Results are shown as a ranked, sortable table — including each stock's **actual
closing price** on the as-of date — with a weights chart, a sector-mix chart, a
**market-regime banner** (is the matching index above its 200-day average?) and
CSV downloads (both the selection and the full audited universe).

The app has three tabs sharing one sidebar configuration:

- **🔎 Scanner** — the point-in-time scan described above, plus a **portfolio
  builder**: enter your capital to get whole-share quantities at the as-of closes
  (leftover cash is topped up into the most under-weight names), or paste / upload
  your **current holdings** to get the exact **sell and buy orders** for this
  rebalance — with an optional rank buffer so positions still ranked inside, say,
  the top 40 are kept rather than churned.
- **🧪 Backtest** — rebalances the strategy over a defined period (**daily /
  weekly / fortnightly / monthly / quarterly / semi-annual / annual**), on
  multi-year history, and compares it to a **benchmark** — any of ~30 NSE indices
  (Nifty 50, Next 50, 100, 200, 500, Midcap 150/100, **Smallcap 250**/100,
  **Microcap 250**, Total Market, **Bank**, **Financial Services**, sector indices,
  Nifty500 Momentum 50, …), Gold / Nasdaq-100 ETFs, an **Auto** option that matches
  your universe, or a **universe equal-weight** benchmark (same stocks, same
  survivorship bias — so the gap is the momentum selection's true value-add). NSE
  benchmarks default to the **Total Return Index (TRI)**, the fair comparison for
  a strategy valued on dividend-adjusted prices. It reports an equity curve
  (growth of ₹100), drawdown, and a full metrics panel — CAGR, annualised
  volatility, Sharpe, Sortino, max drawdown (with peak/trough/recovery dates),
  Calmar, turnover, hit rate, and vs-benchmark beta / alpha / tracking error /
  information ratio / up-down capture.

  **Risk controls:** a **rank buffer** (hold a name until it drops below rank X —
  slashes turnover, especially for weekly/daily cadences), a **market-regime
  filter** (cut exposure to cash or a chosen % when an index closes below its
  N-day moving average; idle cash can earn a yield), a **sector cap**, and
  **Indian cost presets** (statutory ≈ 11 bps/side, discount broker + slippage,
  full-service, or custom).

  **Report tabs:** *Overview* (KPIs, charts, metrics); *Returns analysis* — a
  **calendar-returns heatmap** with annual returns vs the benchmark, **rolling
  1/3/5-year returns** with % of windows positive / beating the benchmark, and the
  **worst drawdown episodes**; *SIP* — a monthly (optionally step-up) **SIP
  simulator with XIRR** vs the benchmark; *Attribution & sectors* — **per-stock
  P&L contribution** (top winners/losers, win rate, return while held) and
  **sector exposure over time**; *Holdings & trades* — period returns, schedule,
  holdings, entries/exits and the transaction ledger; *Downloads* — CSVs and a
  one-click **Excel report** with every table.
- **⚖️ Compare strategies** — runs the Backtest tab's settings over a grid of
  rebalance frequencies × Top N and shows a heatmap of the chosen metric
  (Sharpe, CAGR, Calmar, max drawdown, …), a full metrics table and the top-5
  equity curves. Scans are shared across the grid, so a 3×3 comparison costs
  little more than one backtest.

  The Backtest tab also lets you:
  - **Pick an exact date window** — choose the **start** and **end** dates to
    measure performance between any two specific dates (the first rebalance is
    auto-nudged later only if there isn't enough prior history for a scan).
  - **Set the portfolio size (Top N)** — backtest only the top-N momentum names
    each rebalance (e.g. **10** for a concentrated top-10 portfolio); every other
    parameter is unchanged.
  - **See index churn** — a per-rebalance **entries / exits** table naming the
    scrips that joined or dropped out of the momentum portfolio as the ranking
    changed (with a CSV export).
  - **Read the transaction ledger** — a detailed per-rebalance trade report: side
    (buy / sell / new / exit), as-of price, prior vs target weight, and the rupee
    prior / target / traded value and modelled cost of each trade, denominated on
    a ₹100 starting capital that compounds with the portfolio (with a CSV export).

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
./venv/bin/python tests/test_scanner.py       # scanner maths
./venv/bin/python tests/test_backtest.py      # backtest engine
./venv/bin/python tests/test_features.py      # analytics / portfolio / universe / benchmark sources
./venv/bin/python tests/test_app.py           # Streamlit UI (headless AppTest, offline)
./venv/bin/python tests/test_app_features.py  # Streamlit UI — new universes, risk controls, reports, Compare
```

The tests are dependency-free (a tiny hand-rolled `check()` harness, no pytest) so
they run against the app's own `requirements.txt`. To measure coverage, install the
optional dev tool and run under it:

```bash
./venv/bin/python -m pip install -r requirements-dev.txt
for t in scanner backtest features app app_features; do
  ./venv/bin/python -m coverage run --append --branch --source=momentum,app tests/test_$t.py
done
./venv/bin/python -m coverage report -m
```

- **`test_scanner.py`** — as-of price lookup, the Z-score → Normalized Momentum
  Score piecewise formula (both branches), the iterative weight-capping algorithm,
  skip-month / configurable-lookback momentum, the raw closing-price column, and an
  end-to-end ranking sanity check on synthetic prices.
- **`test_backtest.py`** — rebalance scheduling (incl. holiday snapping), the
  buy-and-hold return identity, a **no-look-ahead invariance** check, compounding
  continuity, transaction-cost monotonicity, closed-form analytics
  (CAGR/vol/drawdown), benchmark alignment, edge cases (missing benchmark,
  pre-feasibility window, cash periods), and — for the detailed backtest report —
  **membership churn** (entries/exits), trade-side classification, and the
  **transaction ledger** including a reconciliation that its rupee trades sum back
  to the engine's own turnover and cost.
- **`test_scanner.py`** also checks the **vectorised** momentum maths against the
  original per-ticker loop (an oracle) on messy data — gaps, late listings,
  halts — plus sector caps, the rank buffer and the price-only weighting schemes.
- **`test_backtest.py`** also covers daily / weekly / fortnightly schedules, the
  regime filter (risk-off → cash, partial exposure, cash yield), the rank buffer
  (lower turnover), sector caps, inverse-vol weights, the **per-stock attribution
  reconciling exactly to the equity curve**, the parameter sweep matching
  standalone runs, and **no look-ahead with every new feature switched on**.
- **`test_features.py`** — calendar / rolling returns, drawdown episodes, XIRR & SIP
  (closed forms), sector exposure, contribution summaries, the Excel export,
  whole-share allocation, holdings parsing (text + broker CSV), rebalance orders,
  symbol parsing / custom & ETF universes, and the benchmark **source ladder**
  (NSE → bundled snapshot + Yahoo → Yahoo) with the network patched out.
- **`test_app_features.py`** — headless UI runs of every new path: multi-index,
  ETF and custom universes, the portfolio builder and rebalance orders, the new
  cadences, risk controls, benchmark options (Auto / TRI / universe EW), every
  report section, the Excel export and the Compare tab.
- **`test_app.py`** — drives the Streamlit app headlessly with
  `streamlit.testing.v1.AppTest` and monkeypatched (offline) data: both tabs load,
  the Scanner still works, the Backtest runs and renders charts/metrics/tables, a
  scan→backtest session shares no `st.stop()` leakage, benchmark-unavailable
  degrades gracefully, and the new controls work — the **start/end date window**,
  the **Top-N** portfolio-size control, and the entries/exits + transaction-ledger
  tables.

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
├── app.py                 # Streamlit UI (Scanner / Backtest / Compare tabs)
├── requirements.txt
├── momentum/
│   ├── config.py          # methodology parameters, universe/benchmark registries, presets
│   ├── universe.py        # NSE index lists, curated ETFs, custom symbol lists
│   ├── data.py            # Yahoo prices/market caps + NSE (niftyindices) benchmarks, disk-cached
│   ├── scanner.py         # eligibility, momentum (vectorised), scoring, selection, weighting
│   ├── backtest.py        # rebalancing engine, regime/buffer/sector rules, sweeps (look-ahead-free)
│   ├── analytics.py       # calendar/rolling returns, drawdowns, SIP/XIRR, attribution, Excel
│   ├── portfolio.py       # whole-share allocation + rebalance order lists
│   └── seed_data/         # offline fallbacks: index lists, etfs.csv, benchmarks.parquet
├── scripts/
│   └── refresh_seeds.py   # regenerate the seed constituent lists + benchmark snapshot
├── tests/                 # dependency-free check() suites (see "Run the tests")
└── data/                  # on-disk cache (gitignored)
```

### Refreshing the bundled data

`momentum/seed_data/` holds offline fallbacks so the hosted app keeps working when
NSE or niftyindices.com block cloud IPs: every index constituent list and a
daily TRI + price snapshot of every NSE benchmark since 2005 (when the live
source is unreachable the snapshot is chain-extended with Yahoo where a symbol
exists). Refresh them occasionally:

```bash
./venv/bin/python scripts/refresh_seeds.py               # constituents + benchmark snapshot
./venv/bin/python scripts/refresh_seeds.py --only bench  # just the benchmark snapshot
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

- **ETFs & custom lists** — ETFs have no market cap, so the methodology weighting
  and the turnover-ratio filter don't apply; the app defaults to score weighting
  and switches that filter off. Liquid-fund ETFs that pay daily dividends as units
  show a flat price on Yahoo, so the basket uses a growth-option liquid ETF.
- **Benchmarks** — NSE TRI/price indices come from niftyindices.com (back-calculated
  before each index's launch date, e.g. Microcap 250 and the momentum indices).
  The regime filter and the scanner banner use the price index.
- **Bad Yahoo prints** — Yahoo occasionally serves a few days of wrongly-scaled
  closes (e.g. several ETFs at 1/10–1/100 of their price around Dec 2019). Closes
  more than 3× away from their centred 9-day median are dropped and forward-filled
  (`data.clean_price_spikes`); persistent level changes are never touched.
- **Daily / weekly rebalancing** — modelled at the close with the configured cost;
  real-world slippage at that turnover can be much higher, so use realistic costs
  and/or the rank buffer.

Data sources: index constituents from NSE
(`nsearchives.nseindia.com/content/indices/*.csv`), benchmark TRI/price history
from niftyindices.com, and prices/market caps from Yahoo Finance via
[`yfinance`](https://github.com/ranaroussi/yfinance).
