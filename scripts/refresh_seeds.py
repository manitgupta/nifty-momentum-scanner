"""Regenerate the committed offline fallbacks in ``momentum/seed_data/``.

    ./venv/bin/python scripts/refresh_seeds.py              # constituents + benchmarks
    ./venv/bin/python scripts/refresh_seeds.py --only bench # just the benchmark snapshot

* ``constituents_<key>.csv`` — NSE index constituent lists (used when NSE blocks
  the cloud's IP and no fresh on-disk cache exists).
* ``benchmarks.parquet`` — daily TRI + price history of every NSE benchmark in
  ``config.BENCHMARKS`` from niftyindices.com (used when that site is
  unreachable; the app chain-extends it with Yahoo where a symbol exists).

The curated ``etfs.csv`` is maintained by hand (one liquid ETF per underlying).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from momentum import config, data  # noqa: E402

BENCH_START = dt.date(2005, 1, 1)


def refresh_constituents() -> None:
    for key, url in config.SEGMENT_URLS.items():
        r = requests.get(url, headers=config.HTTP_HEADERS, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if not {"Symbol", "Company Name"} <= set(df.columns):
            print(f"  ! {key}: unexpected columns {list(df.columns)} — kept old seed")
            continue
        df.to_csv(config.SEED_DIR / f"constituents_{key.lower()}.csv", index=False)
        print(f"  {key:24s} {len(df):4d} names")


def refresh_benchmarks() -> None:
    today = dt.date.today()
    cols: dict[str, pd.Series] = {}
    for label, spec in config.BENCHMARKS.items():
        name = spec.get("nse")
        if not name:
            continue
        for tri in (True, False):
            s = data.nse_index_history(name, BENCH_START, today, tri=tri)
            tag = "TRI" if tri else "PRICE"
            if s.empty:
                print(f"  ! {label} [{tag}]: no data for NSE name {name!r}")
                continue
            cols[f"{name}|{tag}"] = s
            print(f"  {label:26s} {tag:5s} {s.index[0].date()} → {s.index[-1].date()} ({len(s)} rows)")
    frame = pd.DataFrame(cols).sort_index()
    frame.index.name = "date"
    frame.to_parquet(data.BENCHMARK_SEED_PATH, compression="zstd")
    size_kb = data.BENCHMARK_SEED_PATH.stat().st_size / 1024
    print(f"wrote {data.BENCHMARK_SEED_PATH.name}: {frame.shape[1]} series, {size_kb:.0f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["constituents", "bench"])
    args = ap.parse_args()
    if args.only in (None, "constituents"):
        print("Constituents:")
        refresh_constituents()
    if args.only in (None, "bench"):
        print("Benchmarks:")
        refresh_benchmarks()
