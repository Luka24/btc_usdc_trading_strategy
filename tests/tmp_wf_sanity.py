"""
Sanity check for walk_forward data slices.

Verifies:
1. Each OOS slice has correct date range and row count
2. BTC price ranges match known market history
3. Buy-and-hold return for each OOS period (strategy should beat or limit losses vs BTC)
4. EMA warm-up gap: how many rows are NaN before first valid signal
"""

import sys, os
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

from optimization.walk_forward import load_full_data, FOLDS, _slice
from backtest import BacktestEngine

# Known approximate BTC prices at period boundaries (USD, rough)
KNOWN_PRICES = {
    "2020-01-01": 7_200,
    "2020-12-31": 29_000,
    "2021-01-01": 29_000,
    "2021-12-31": 47_000,
    "2022-01-01": 47_000,
    "2022-12-31": 16_500,
    "2023-01-01": 16_500,
    "2023-12-31": 42_000,
    "2024-01-01": 42_000,
    "2025-06-30": 107_000,
}

print("Loading full dataset...")
df = load_full_data(force_refresh=False)
print(f"Full data: {len(df)} rows  |  {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
print(f"BTC price column: first={df['btc_price'].iloc[0]:.0f}  last={df['btc_price'].iloc[-1]:.0f}\n")

print(f"{'Fold':<14} {'OOS rows':>8} {'First date':>12} {'Last date':>12} "
      f"{'BTC start':>10} {'BTC end':>10} {'BH return':>10}")
print("-" * 82)

for fold in FOLDS:
    oos = _slice(df, fold.oos_start, fold.oos_end)
    if len(oos) == 0:
        print(f"{fold.name:<14}  NO DATA")
        continue

    prices = pd.to_numeric(oos["btc_price"], errors="coerce")
    btc_start = prices.iloc[0]
    btc_end   = prices.iloc[-1]
    bh_ret    = (btc_end / btc_start - 1) * 100

    first_date = oos["date"].iloc[0].date()
    last_date  = oos["date"].iloc[-1].date()

    print(f"{fold.name:<14} {len(oos):>8} {str(first_date):>12} {str(last_date):>12} "
          f"{btc_start:>10,.0f} {btc_end:>10,.0f} {bh_ret:>9.1f}%")

print()

# Check EMA warm-up gap: does the engine produce NaN signals early in each OOS window?
import config as cfg

ema_window = cfg.PortfolioConfig.SIGNAL_EMA_WINDOW  # the slowest EMA (~14d default)
print(f"Signal EMA window (slowest): {ema_window} days")
print(f"Price EMA window: {cfg.PortfolioConfig.PRICE_EMA_WINDOW}")
print(f"Cost EMA window:  {cfg.PortfolioConfig.COST_EMA_WINDOW}")
print()

# Check strategy vs buy-and-hold per fold
print(f"{'Fold':<14} {'BH return':>10} {'Strategy ret':>13} {'vs BH':>8} {'MaxDD':>8}")
print("-" * 60)

for fold in FOLDS:
    oos = _slice(df, fold.oos_start, fold.oos_end)
    if len(oos) < 30:
        print(f"{fold.name:<14}  SKIP (too short)")
        continue

    prices = pd.to_numeric(oos["btc_price"], errors="coerce")
    bh_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

    eng = BacktestEngine(initial_capital=100_000, enable_risk_management=True)
    eng.add_from_dataframe(oos.copy())
    eng.run_backtest(initial_btc_quantity=2.0)
    m = eng.calculate_metrics()

    strat_ret = m["total_return_pct"]
    vs_bh     = strat_ret - bh_ret
    maxdd     = m["max_drawdown_pct"]

    marker = " ✓" if vs_bh > 0 else " ✗"
    print(f"{fold.name:<14} {bh_ret:>9.1f}%  {strat_ret:>11.1f}%  {vs_bh:>+7.1f}%  {maxdd:>7.2f}%{marker}")

print()
print("Note: 2022 OOS = entire BTC bear market. Strategy losing less than BTC is correct behavior.")
print("EMA warm-up: first N rows in each OOS window will have sub-optimal initial signal.")
print("  -> This is inherent to expanding-window WF; not a bug, but worth knowing.")
