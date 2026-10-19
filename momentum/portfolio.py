"""Turn a momentum selection into an actionable order list (pure functions).

    allocate         - invest ₹ capital across the selection in whole shares
    parse_holdings   - read "SYMBOL, QTY" text / a CSV into {symbol: qty}
    rebalance_orders - exact BUY / SELL / HOLD quantities that move a current
                       portfolio onto the new target weights

Prices are the actual traded closes on the scan date (``close_price``), so the
quantities are what an investor would place for the next session.
"""

from __future__ import annotations

import io
import math
import re

import numpy as np
import pandas as pd


def _targets(selected: pd.DataFrame) -> pd.DataFrame:
    """Normalise a scanner selection to Symbol / Company / weight / price rows."""
    df = selected.reset_index()
    price_col = "close_price" if "close_price" in df.columns else "price"
    out = pd.DataFrame({
        "Symbol": df["Symbol"].astype(str),
        "Company": df["Company"] if "Company" in df.columns else df["Symbol"],
        "weight": df["weight"].astype(float),
        "price": pd.to_numeric(df[price_col], errors="coerce"),
    })
    total = out["weight"].sum()
    if total > 0:
        out["weight"] = out["weight"] / total
    return out


def allocate(selected: pd.DataFrame, capital: float, top_up: bool = True) -> tuple[pd.DataFrame, float]:
    """Whole-share allocation of ``capital`` (₹) across ``selected`` weights.

    Each name gets ``floor(capital × weight / price)`` shares. With ``top_up`` the
    leftover cash is then spent greedily, one share at a time, on the name whose
    actual weight is furthest below target (and affordable) — so very little cash
    is left idle. Names with no usable price get 0 shares.
    Returns (table, leftover_cash). Table columns: Symbol, Company, price,
    target_weight, target_value, shares, invested, actual_weight.
    """
    t = _targets(selected)
    ok = t["price"].gt(0) & t["price"].notna()
    target_value = capital * t["weight"]
    shares = np.where(ok, np.floor(target_value / t["price"].where(ok, 1.0)), 0).astype(int)
    invested = shares * t["price"].fillna(0.0)
    cash = float(capital - invested.sum())

    if top_up and ok.any():
        price = t["price"].where(ok, np.inf).to_numpy()
        tv = target_value.to_numpy()
        sh = shares.astype(float)
        for _ in range(10_000):  # bounded; each pass buys one share
            gap = tv - sh * np.where(np.isfinite(price), price, 0.0)
            gap = np.where(price <= cash + 1e-9, gap, -np.inf)
            k = int(np.argmax(gap))
            if not np.isfinite(gap[k]) or gap[k] <= 0:
                break
            sh[k] += 1
            cash -= float(price[k])
        shares = sh.astype(int)
        invested = shares * t["price"].fillna(0.0)

    total_inv = float(invested.sum())
    table = pd.DataFrame({
        "Symbol": t["Symbol"], "Company": t["Company"], "price": t["price"],
        "target_weight": t["weight"], "target_value": target_value,
        "shares": shares, "invested": invested,
        "actual_weight": invested / total_inv if total_inv > 0 else 0.0,
    })
    return table, float(capital - total_inv)


_HOLDING_RE = re.compile(r"^\s*([A-Za-z0-9&\-_.]+)\s*[,;:\t ]\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*$")


def parse_holdings(text: str | None = None, csv_bytes: bytes | None = None) -> tuple[dict[str, int], list[str]]:
    """Parse current holdings into ``{SYMBOL: qty}`` plus a list of unparsed lines.

    Text: one holding per line — ``RELIANCE, 10`` / ``TCS 5`` / ``INFY:12``.
    CSV: needs a symbol column (Symbol/Instrument/Scrip/Ticker) and a quantity
    column (Qty/Quantity/Shares/Units) — e.g. a broker holdings export.
    Quantities for a repeated symbol are summed; ``.NS`` suffixes are stripped.
    """
    out: dict[str, int] = {}
    bad: list[str] = []

    def _add(sym: str, qty) -> None:
        s = str(sym).strip().upper()
        if s.endswith(".NS"):
            s = s[:-3]
        q = int(float(str(qty).replace(",", "")))
        if s and q > 0:
            out[s] = out.get(s, 0) + q

    if csv_bytes:
        df = pd.read_csv(io.BytesIO(csv_bytes))
        lower = {c.lower().strip(): c for c in df.columns}
        sym_col = next((lower[c] for c in ("symbol", "instrument", "scrip", "ticker", "tradingsymbol") if c in lower), None)
        qty_col = next((lower[c] for c in ("qty", "quantity", "shares", "units", "qty.", "net qty") if c in lower), None)
        if sym_col is None or qty_col is None:
            raise ValueError("CSV needs a Symbol column and a Qty/Quantity column.")
        for sym, qty in zip(df[sym_col], df[qty_col]):
            try:
                _add(sym, qty)
            except (TypeError, ValueError):
                bad.append(f"{sym},{qty}")
    for line in (text or "").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = _HOLDING_RE.match(line)
        if m:
            _add(m.group(1), m.group(2))
        else:
            bad.append(line.strip())
    return out, bad


def rebalance_orders(
    selected: pd.DataFrame,
    current: dict[str, int],
    prices: dict[str, float],
    extra_cash: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Orders that move ``current`` holdings onto the ``selected`` target weights.

    Portfolio value = Σ current qty × price (+ ``extra_cash`` to add, or negative
    to withdraw). Target quantities are whole-share allocations of that value.
    Held names outside the target are sold in full; a held name with no known
    price is flagged ``REVIEW`` (it cannot be valued) and left untouched.
    Returns (orders, summary) — orders columns: Symbol, action, qty, price,
    value, current_qty, target_qty, target_weight.
    """
    t = _targets(selected)
    px = {**prices, **{s: p for s, p in zip(t["Symbol"], t["price"]) if pd.notna(p)}}
    held_value = sum(q * px[s] for s, q in current.items() if s in px and pd.notna(px[s]))
    total = held_value + float(extra_cash)
    alloc, leftover = allocate(selected, max(total, 0.0))
    target_qty = dict(zip(alloc["Symbol"], alloc["shares"]))
    target_w = dict(zip(alloc["Symbol"], alloc["target_weight"]))

    rows = []
    for sym in list(dict.fromkeys(list(t["Symbol"]) + list(current))):
        cur = int(current.get(sym, 0))
        price = px.get(sym, np.nan)
        if sym not in target_qty and (price is None or not np.isfinite(price)):
            rows.append({"Symbol": sym, "action": "REVIEW", "qty": 0, "price": np.nan, "value": np.nan,
                         "current_qty": cur, "target_qty": np.nan, "target_weight": 0.0})
            continue
        tgt = int(target_qty.get(sym, 0))
        delta = tgt - cur
        action = "BUY" if delta > 0 else ("SELL" if delta < 0 else "HOLD")
        rows.append({"Symbol": sym, "action": action, "qty": abs(delta), "price": price,
                     "value": abs(delta) * price, "current_qty": cur, "target_qty": tgt,
                     "target_weight": float(target_w.get(sym, 0.0))})
    orders = pd.DataFrame(rows, columns=["Symbol", "action", "qty", "price", "value",
                                         "current_qty", "target_qty", "target_weight"])
    order_rank = {"SELL": 0, "BUY": 1, "HOLD": 2, "REVIEW": 3}
    orders = orders.sort_values(["action", "value"], key=lambda c: c.map(order_rank) if c.name == "action" else -c.fillna(0))
    buys = float(orders.loc[orders["action"] == "BUY", "value"].sum())
    sells = float(orders.loc[orders["action"] == "SELL", "value"].sum())
    summary = {
        "current_value": float(held_value), "extra_cash": float(extra_cash), "target_value": float(total),
        "buy_value": buys, "sell_value": sells, "cash_after": float(total - (alloc["invested"].sum())),
        "n_buys": int((orders["action"] == "BUY").sum()), "n_sells": int((orders["action"] == "SELL").sum()),
        "n_review": int((orders["action"] == "REVIEW").sum()),
    }
    return orders.reset_index(drop=True), summary


def round_lakh(x: float) -> str:
    """₹ in Indian digit grouping (12,34,567)."""
    if x is None or not math.isfinite(x):
        return "—"
    neg = x < 0
    s = f"{abs(x):.0f}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return ("-" if neg else "") + "₹" + s
