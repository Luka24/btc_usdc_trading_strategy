"""
Walk-forward validation harness.

Loads full dataset once, slices into expanding-window folds,
runs BacktestEngine on each OOS window, and returns per-fold metrics.

Folds (expanding train, fixed OOS):
  1: train 2017-2019, OOS 2020
  2: train 2017-2020, OOS 2021
  3: train 2017-2021, OOS 2022
  4: train 2017-2022, OOS 2023
  5: train 2017-2023, OOS 2024-mid-2025

Hands-off test set: 2025-07-01 onward — never touched during optimization.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest import BacktestEngine
from data_fetcher import DataFetcher

# ---------------------------------------------------------------------------
# Fold definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    name: str
    train_start: str
    train_end: str   # inclusive
    oos_start: str
    oos_end: str     # inclusive


FOLDS: list[Fold] = [
    Fold("f1_oos2020", "2017-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    Fold("f2_oos2021", "2017-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    Fold("f3_oos2022", "2017-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    Fold("f4_oos2023", "2017-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    Fold("f5_oos2024", "2017-01-01", "2023-12-31", "2024-01-01", "2025-06-30"),
]

# Hands-off: never used during optimization
TEST_FOLD = Fold("test_oos",  "2017-01-01", "2025-06-30", "2025-07-01", "2026-12-31")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_cached_full_data: Optional[pd.DataFrame] = None


def load_full_data(force_refresh: bool = False) -> pd.DataFrame:
    """Load and cache the full dataset (called once per process)."""
    global _cached_full_data
    if _cached_full_data is None or force_refresh:
        raw = DataFetcher.fetch_combined_data(
            days=3650, use_real_data=True, force_refresh=force_refresh
        )
        raw["date"] = pd.to_datetime(raw["date"])
        _cached_full_data = raw.sort_values("date").reset_index(drop=True)
    return _cached_full_data


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Single-fold runner
# ---------------------------------------------------------------------------

def run_fold(
    df_full: pd.DataFrame,
    fold: Fold,
    initial_capital: float = 100_000.0,
    initial_btc: float = 2.0,
    enable_risk_management: bool = True,
) -> dict:
    """
    Run backtest on the OOS window of a single fold.

    Returns dict with sortino_ratio, sharpe_ratio, max_drawdown_pct,
    total_return_pct, and fold metadata.
    """
    oos_df = _slice(df_full, fold.oos_start, fold.oos_end)

    if len(oos_df) < 30:
        # Not enough data — return sentinel
        return {
            "fold": fold.name,
            "sortino_ratio": -99.0,
            "sharpe_ratio": -99.0,
            "max_drawdown_pct": -99.0,
            "total_return_pct": -99.0,
            "n_days": len(oos_df),
            "error": "insufficient_data",
        }

    eng = BacktestEngine(
        initial_capital=initial_capital,
        enable_risk_management=enable_risk_management,
    )
    eng.add_from_dataframe(oos_df)
    eng.run_backtest(initial_btc_quantity=initial_btc)
    m = eng.calculate_metrics()

    return {
        "fold": fold.name,
        "oos_start": fold.oos_start,
        "oos_end": fold.oos_end,
        "n_days": len(oos_df),
        "sortino_ratio": m["sortino_ratio"],
        "sharpe_ratio": m["sharpe_ratio"],
        "max_drawdown_pct": m["max_drawdown_pct"],
        "total_return_pct": m["total_return_pct"],
    }


# ---------------------------------------------------------------------------
# Full walk-forward run
# ---------------------------------------------------------------------------

def run_walk_forward(
    df_full: Optional[pd.DataFrame] = None,
    folds: Optional[list[Fold]] = None,
    initial_capital: float = 100_000.0,
    initial_btc: float = 2.0,
) -> dict:
    """
    Run all folds and return summary statistics.

    Returns:
        dict with per-fold results and aggregate mean/std of OOS Sortino.
    """
    if df_full is None:
        df_full = load_full_data()
    if folds is None:
        folds = FOLDS

    results = [
        run_fold(df_full, fold, initial_capital, initial_btc)
        for fold in folds
    ]

    sortinos = [r["sortino_ratio"] for r in results if r.get("error") is None]
    sharpes  = [r["sharpe_ratio"]  for r in results if r.get("error") is None]
    max_dds  = [r["max_drawdown_pct"] for r in results if r.get("error") is None]

    return {
        "folds": results,
        "mean_oos_sortino": float(np.mean(sortinos)) if sortinos else -99.0,
        "std_oos_sortino":  float(np.std(sortinos))  if sortinos else 99.0,
        "mean_oos_sharpe":  float(np.mean(sharpes))  if sharpes  else -99.0,
        "worst_oos_drawdown": float(min(max_dds))    if max_dds  else -99.0,
    }


# ---------------------------------------------------------------------------
# CLI usage: python optimization/walk_forward.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    df = load_full_data()
    print(f"Data loaded: {len(df)} rows, {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}\n")

    summary = run_walk_forward(df_full=df)

    print(f"{'Fold':<15} {'OOS Start':<12} {'OOS End':<12} {'Days':>5} {'Sortino':>8} {'Sharpe':>8} {'MaxDD':>8} {'Return':>8}")
    print("-" * 82)
    for r in summary["folds"]:
        err = r.get("error", "")
        if err:
            print(f"{r['fold']:<15} {'ERR':>12} {'':12} {r['n_days']:>5} {'N/A':>8}")
        else:
            print(
                f"{r['fold']:<15} {r['oos_start']:<12} {r['oos_end']:<12}"
                f" {r['n_days']:>5} {r['sortino_ratio']:>8.3f}"
                f" {r['sharpe_ratio']:>8.3f} {r['max_drawdown_pct']:>8.2f}"
                f" {r['total_return_pct']:>8.2f}"
            )

    print("-" * 82)
    print(f"{'Mean OOS Sortino:':<30} {summary['mean_oos_sortino']:.3f}  (std {summary['std_oos_sortino']:.3f})")
    print(f"{'Mean OOS Sharpe:':<30} {summary['mean_oos_sharpe']:.3f}")
    print(f"{'Worst OOS MaxDD:':<30} {summary['worst_oos_drawdown']:.2f}%")
