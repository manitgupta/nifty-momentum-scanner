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

1. **Builds the universe** from NSE index constituent lists:
   | Checkbox   | NSE index          | Size |
   |------------|--------------------|------|
   | Largecap   | Nifty 100          | 100  |
   | Midcap     | Nifty Midcap 150   | 150  |
   | Small      | Nifty Smallcap 250 | 250  |

   Their union ≈ the Nifty 500 (the index's base universe). The **selected
   segments define the eligible universe** over which Z-scores are normalized
   and the top N are chosen.

2. **Applies eligibility filters** (those computable from price/volume):
   minimum 1-year listing history, exclude bottom-10-percentile by 6-month
   average daily turnover, and exclude bottom-10-percentile by turnover ratio.

3. **Scores momentum** exactly per the methodology (see mapping below).

4. **Selects & weights** the top N, capping each weight at `min(5%, 5×` its
   free-float-mcap-only weight`)` and redistributing the excess.

Results are shown as a ranked, sortable table with a weights chart and CSV
downloads (both the selection and the full audited universe).

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
./venv/bin/python tests/test_scanner.py
```

Covers the as-of price lookup, the Z-score → Normalized Momentum Score piecewise
formula (both branches), the iterative weight-capping algorithm, and an
end-to-end ranking sanity check on synthetic prices.

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
├── app.py                 # Streamlit UI
├── requirements.txt
├── momentum/
│   ├── config.py          # methodology parameters + data-source config
│   ├── universe.py        # NSE constituent lists -> eligible universe
│   ├── data.py            # Yahoo Finance price/volume + market-cap (disk-cached)
│   └── scanner.py         # eligibility, momentum, scoring, selection, weighting
├── tests/
│   └── test_scanner.py    # unit + integration checks
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

Data sources: index constituents from NSE
(`nsearchives.nseindia.com/content/indices/*.csv`) and prices/market caps from
Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance).
