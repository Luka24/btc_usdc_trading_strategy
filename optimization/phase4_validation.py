"""
Phase 4 — Validation report.

Checks:
  1. Train vs OOS Sortino per fold (overfit ratio)
  2. Parameter importance from Optuna SQLite studies
  3. TEST set evaluation (hands-off, run once)
  4. Benchmark table: Strategy vs BTC buy-and-hold vs 50/50 static
  5. Rolling 30-day Sharpe across all OOS data

Usage:
  python optimization/phase4_validation.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any

# Ensure stdout can handle any characters on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from backtest import BacktestEngine
from optimization.walk_forward import (
    FOLDS, TEST_FOLD, Fold, _slice, load_full_data, run_fold
)

BEST_PARAMS_PATH = os.path.join(CURRENT_DIR, "best_params.json")
DB_A = os.path.join(CURRENT_DIR, "study_phase_a.db")
DB_B = os.path.join(CURRENT_DIR, "study_phase_b.db")
DB_C = os.path.join(CURRENT_DIR, "study_phase_c.db")

SEP = "-" * 70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_apply_best_params() -> dict[str, Any]:
    with open(BEST_PARAMS_PATH) as f:
        bp = json.load(f)
    ga, gb, gc = bp["group_a"], bp["group_b"], bp["group_c"]
    cfg.SignalConfig.PRICE_EMA_WINDOW       = ga["PRICE_EMA_WINDOW"]
    cfg.SignalConfig.COST_EMA_WINDOW        = ga["COST_EMA_WINDOW"]
    cfg.SignalConfig.SIGNAL_EMA_WINDOW      = ga["SIGNAL_EMA_WINDOW"]
    cfg.PortfolioConfig.TREND_FILTER_WINDOW = ga["TREND_FILTER_WINDOW"]
    cfg.PortfolioConfig.RSI_OVERSOLD        = ga["RSI_OVERSOLD"]
    cfg.PortfolioConfig.VOL_TARGET          = ga["VOL_TARGET"]
    cfg.RiskManagementConfig.DD_THRESHOLDS.update(gb["DD_THRESHOLDS"])
    cfg.RiskManagementConfig.VOL_THRESHOLDS.update(gb["VOL_THRESHOLDS"])
    cfg.PortfolioConfig.HASH_RIBBON_CAP_MULT = gc["HASH_RIBBON_CAP_MULT"]
    return bp


def perf_metrics(value_series: pd.Series) -> dict:
    """Compute Sharpe, Sortino, Calmar, MaxDD, AnnReturn from daily total-value series."""
    rets = value_series.pct_change().dropna()
    if len(rets) < 5:
        return dict(sharpe=0.0, sortino=0.0, calmar=0.0, maxdd=0.0, ann_return=0.0)

    ann_ret = (value_series.iloc[-1] / value_series.iloc[0]) ** (252 / len(rets)) - 1
    sharpe  = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    neg = rets[rets < 0]
    sortino = float(rets.mean() / neg.std() * np.sqrt(252)) if len(neg) > 1 and neg.std() > 0 else 0.0

    cummax = value_series.cummax()
    maxdd  = float(((value_series - cummax) / cummax).min())
    calmar = float(ann_ret / abs(maxdd)) if maxdd < 0 else 0.0

    return dict(
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        calmar=round(calmar, 3),
        maxdd=round(maxdd * 100, 2),
        ann_return=round(ann_ret * 100, 2),
    )


def run_train_fold(df_full: pd.DataFrame, fold: Fold) -> dict:
    """In-sample backtest on train window only (no warmup trick needed)."""
    train_df = _slice(df_full, fold.train_start, fold.train_end)
    if len(train_df) < 60:
        return {"sortino": -99.0, "sharpe": -99.0}
    eng = BacktestEngine(initial_capital=100_000, enable_risk_management=True)
    eng.add_from_dataframe(train_df)
    eng.run_backtest(initial_btc_quantity=2.0)
    pf = eng.portfolio_manager.get_portfolio_dataframe()
    m = perf_metrics(pf["total_value"])
    return m


def bnh_metrics(df_full: pd.DataFrame, start: str, end: str,
                initial_capital: float = 100_000.0, initial_btc: float = 2.0) -> dict:
    """Buy-and-hold: put everything into BTC on day 1 and never sell."""
    sl = _slice(df_full, start, end)
    if len(sl) < 2:
        return dict(sharpe=0.0, sortino=0.0, calmar=0.0, maxdd=0.0, ann_return=0.0)
    p = sl["btc_price"].astype(float).values
    # total starting wealth = initial_capital + initial_btc * first_price
    total_wealth = initial_capital + initial_btc * p[0]
    btc_qty = total_wealth / p[0]
    value = pd.Series(btc_qty * p, index=range(len(p)), dtype=float)
    return perf_metrics(value)


def static5050_metrics(df_full: pd.DataFrame, start: str, end: str,
                       initial_capital: float = 100_000.0, initial_btc: float = 2.0) -> dict:
    """Static 50/50: put half in BTC, half in USDC, never rebalance."""
    sl = _slice(df_full, start, end)
    if len(sl) < 2:
        return dict(sharpe=0.0, sortino=0.0, calmar=0.0, maxdd=0.0, ann_return=0.0)
    p = sl["btc_price"].astype(float).values
    total_wealth = initial_capital + initial_btc * p[0]
    half = total_wealth / 2.0
    btc_qty = half / p[0]
    usdc = half
    value = pd.Series(btc_qty * p + usdc, dtype=float)
    return perf_metrics(value)


def load_study_importance(db_path: str, study_name: str) -> dict[str, float]:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        if len(study.trials) < 5:
            return {}
        imp = optuna.importance.get_param_importances(study)
        return dict(imp)
    except Exception as e:
        return {"error": str(e)}


def rolling_sharpe(value_series: pd.Series, window: int = 30) -> pd.Series:
    rets = value_series.pct_change()
    roll_mean = rets.rolling(window).mean()
    roll_std  = rets.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).dropna()


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_overfit(df: pd.DataFrame) -> list[float]:
    print(f"\n{SEP}")
    print("1. TRAIN vs OOS — OVERFIT CHECK")
    print(SEP)
    print(f"{'Fold':<15} {'Train Sortino':>14} {'OOS Sortino':>12} {'Ratio OOS/Train':>16}")
    print("-" * 60)

    train_sortinos, oos_sortinos = [], []
    for fold in FOLDS:
        train_m = run_train_fold(df, fold)
        oos_r   = run_fold(df, fold)
        ts = train_m["sortino"]
        os_ = oos_r["sortino_ratio"]
        ratio = os_ / ts if ts > 0 else float("nan")
        flag = "" if (ratio >= 0.70 or np.isnan(ratio)) else "  << low"
        print(f"{fold.name:<15} {ts:>14.3f} {os_:>12.3f} {ratio:>16.2f}{flag}")
        train_sortinos.append(ts)
        oos_sortinos.append(os_)

    mean_train = float(np.mean(train_sortinos))
    mean_oos   = float(np.mean(oos_sortinos))
    mean_ratio = mean_oos / mean_train if mean_train > 0 else float("nan")
    verdict = "OK — no severe overfit" if mean_ratio >= 0.70 else "WARNING — possible overfit"

    print("-" * 60)
    print(f"{'MEAN':<15} {mean_train:>14.3f} {mean_oos:>12.3f} {mean_ratio:>16.2f}  [{verdict}]")
    print(f"\n  Rule of thumb: OOS/Train ratio ≥ 0.70 is acceptable.")
    return oos_sortinos


def section_importance():
    print(f"\n{SEP}")
    print("2. PARAMETER IMPORTANCE (Optuna fANOVA)")
    print(SEP)

    for label, db, name in [
        ("Phase A — core signal",  DB_A, "phase_a"),
        ("Phase B — risk manager", DB_B, "phase_b"),
        ("Phase C — overlays",     DB_C, "phase_c"),
    ]:
        if not os.path.exists(db):
            print(f"  {label}: study DB not found, skipping.")
            continue
        imp = load_study_importance(db, name)
        if "error" in imp:
            print(f"  {label}: {imp['error']}")
            continue
        print(f"\n  {label}:")
        for param, val in sorted(imp.items(), key=lambda x: -x[1]):
            bar = "|" * max(1, int(val * 30))
            print(f"    {param:<30} {val:.3f}  {bar}")


def section_test(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("3. TEST SET — HANDS-OFF EVALUATION")
    print(f"   Period: {TEST_FOLD.oos_start} -> {TEST_FOLD.oos_end}")
    print(SEP)

    r = run_fold(df, TEST_FOLD)

    # Strategy portfolio series for richer metrics
    oos_sl  = _slice(df, TEST_FOLD.oos_start, TEST_FOLD.oos_end)
    warmup  = _slice(df, TEST_FOLD.train_start, TEST_FOLD.oos_end)
    eng = BacktestEngine(initial_capital=100_000, enable_risk_management=True)
    eng.add_from_dataframe(warmup)
    eng.run_backtest(initial_btc_quantity=2.0)
    pf = eng.portfolio_manager.get_portfolio_dataframe()
    oos_pf = pf[pf.index >= pd.Timestamp(TEST_FOLD.oos_start)]["total_value"]
    strat_m = perf_metrics(oos_pf)

    bnh_m   = bnh_metrics(df, TEST_FOLD.oos_start, TEST_FOLD.oos_end)
    s5050_m = static5050_metrics(df, TEST_FOLD.oos_start, TEST_FOLD.oos_end)

    print(f"\n  {'Metric':<16} {'Strategy':>10} {'BTC B&H':>10} {'50/50':>10}")
    print(f"  {'-'*48}")
    for key, label in [("sortino","Sortino"), ("sharpe","Sharpe"), ("calmar","Calmar"),
                        ("maxdd","Max DD (%)"), ("ann_return","Ann. Return %")]:
        sv  = strat_m[key]
        bv  = bnh_m[key]
        fv  = s5050_m[key]
        print(f"  {label:<16} {sv:>10.3f} {bv:>10.3f} {fv:>10.3f}")
    print(f"\n  Test days: {r['n_days']}")
    return strat_m


def section_full_oos_benchmark(df: pd.DataFrame, oos_sortinos: list[float]):
    print(f"\n{SEP}")
    print("4. BENCHMARK — FULL OOS PERIOD (2020-01-01 → 2025-06-30)")
    print(SEP)

    OOS_START = "2020-01-01"
    OOS_END   = "2025-06-30"

    warmup = _slice(df, "2017-01-01", OOS_END)
    eng = BacktestEngine(initial_capital=100_000, enable_risk_management=True)
    eng.add_from_dataframe(warmup)
    eng.run_backtest(initial_btc_quantity=2.0)
    pf = eng.portfolio_manager.get_portfolio_dataframe()
    oos_pf = pf[pf.index >= pd.Timestamp(OOS_START)]["total_value"]
    strat_m = perf_metrics(oos_pf)
    bnh_m   = bnh_metrics(df, OOS_START, OOS_END)
    s5050_m = static5050_metrics(df, OOS_START, OOS_END)

    print(f"\n  {'Metric':<16} {'Strategy':>10} {'BTC B&H':>10} {'50/50':>10}")
    print(f"  {'-'*48}")
    for key, label in [("sortino","Sortino"), ("sharpe","Sharpe"), ("calmar","Calmar"),
                        ("maxdd","Max DD (%)"), ("ann_return","Ann. Return %")]:
        print(f"  {label:<16} {strat_m[key]:>10.3f} {bnh_m[key]:>10.3f} {s5050_m[key]:>10.3f}")

    # Rolling Sharpe summary
    rs = rolling_sharpe(oos_pf, window=30)
    print(f"\n  Rolling 30-day Sharpe (across OOS 2020-2025):")
    print(f"    Median : {rs.median():.3f}")
    print(f"    Min    : {rs.min():.3f}  (worst month)")
    print(f"    Max    : {rs.max():.3f}  (best month)")
    print(f"    % positive months: {(rs > 0).mean()*100:.1f}%")


def section_summary(bp: dict):
    print(f"\n{SEP}")
    print("5. BEST PARAMETERS USED")
    print(SEP)
    ga, gb, gc = bp["group_a"], bp["group_b"], bp["group_c"]
    print(f"  Walk-forward OOS Sortino (from optimizer): {bp['final_sortino']:.4f}\n")
    print("  Group A — Core signal:")
    for k, v in ga.items():
        print(f"    {k:<28} = {v}")
    print("  Group B — Risk thresholds:")
    for k, v in gb.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"    {k}[{kk}]{'':>5} = {round(vv, 4)}")
        else:
            print(f"    {k} = {v}")
    print("  Group C — Overlays:")
    for k, v in gc.items():
        print(f"    {k:<28} = {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE 4 — VALIDATION REPORT")
    print("=" * 70)

    print("\nLoading best params...")
    bp = load_and_apply_best_params()

    print("Loading data...")
    df = load_full_data()
    print(f"  {len(df)} rows  ({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")

    oos_sortinos = section_overfit(df)
    section_importance()
    section_test(df)
    section_full_oos_benchmark(df, oos_sortinos)
    section_summary(bp)

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
