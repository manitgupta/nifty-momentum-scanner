"""Nifty500 Momentum 50 scanner package.

Modules:
    config    - methodology parameters and data-source configuration
    universe  - NSE index constituents, the curated ETF list and custom symbol lists
    data      - price/volume + market cap (Yahoo Finance) and benchmark indices (NSE/Yahoo)
    scanner   - eligibility filters, momentum scoring, selection and weighting
    backtest  - look-ahead-free rebalancing backtests and parameter sweeps
    analytics - report tables: calendar/rolling returns, drawdowns, SIP/XIRR, attribution
    portfolio - whole-share allocation and rebalance order lists
"""

__version__ = "0.5.0"
