"""
ML-lite optimizer for a small subset of parameters.

Focus:
- RATIO_BUY_THRESHOLD
- RATIO_SELL_THRESHOLD
- POSITION_TABLE weights (bins unchanged)

Approach:
- Build ratio features from 10y data
- Fit a simple ridge regression (closed-form) to predict forward returns
- Derive thresholds from predicted return by ratio bins
- Map predicted return by ratio bins to portfolio weights

This keeps the rest of the config unchanged.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import BacktestConfig, ProductionCostConfig, SignalConfig, PortfolioConfig
from data_fetcher import DataFetcher
from production_cost import ProductionCostSeries
from portfolio import PortfolioManager


def build_ratio_frame(forward_days: int) -> pd.DataFrame:
    """
    Build a feature frame with ratio and forward returns.
    """
    raw = DataFetcher.fetch_combined_data(
        days=BacktestConfig.DAYS_TO_FETCH,
        use_real_data=BacktestConfig.USE_REAL_DATA,
    )

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    cost_series = ProductionCostSeries(ema_window=ProductionCostConfig.EMA_WINDOW)
    cost_series.add_from_dataframe(df[["date", "hashrate_eh_per_s"]])

    cost_ema = cost_series.smooth_with_ema().rename("cost_ema").reset_index()
    merged = pd.merge(df, cost_ema, left_on="date", right_on="date", how="inner")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["ratio"] = merged["btc_price"] / merged["cost_ema"]

    merged["ratio_ema"] = merged["ratio"].ewm(
        span=SignalConfig.SIGNAL_EMA_WINDOW, adjust=False
    ).mean()
    merged["price_ema"] = merged["btc_price"].ewm(
        span=SignalConfig.PRICE_EMA_WINDOW, adjust=False
    ).mean()
    merged["cost_ema_slow"] = merged["cost_ema"].ewm(
        span=SignalConfig.COST_EMA_WINDOW, adjust=False
    ).mean()

    merged["ratio_change_7d"] = merged["ratio"].pct_change(7)
    merged["price_change_7d"] = merged["btc_price"].pct_change(7)

    merged["forward_return"] = (
        merged["btc_price"].shift(-forward_days) / merged["btc_price"] - 1.0
    )

    merged = merged.dropna().reset_index(drop=True)

    return merged


def fit_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=l2, fit_intercept=True)
    model.fit(X_scaled, y)
    return model, scaler


def predict_ridge(
    X: np.ndarray,
    model: Ridge,
    scaler: StandardScaler,
) -> np.ndarray:
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def derive_thresholds(
    ratio: pd.Series,
    pred: np.ndarray,
    bins: int = 80,
    positive_target: float = 0.0,
    negative_target: float = 0.0,
) -> tuple[float, float]:
    df = pd.DataFrame({"ratio": ratio.values, "pred": pred})
    df = df.sort_values("ratio")

    df["bin"] = pd.qcut(df["ratio"], q=bins, duplicates="drop")
    grouped = df.groupby("bin").agg(
        ratio_min=("ratio", "min"),
        ratio_max=("ratio", "max"),
        pred_mean=("pred", "mean"),
    )

    buy_candidates = grouped[grouped["pred_mean"] >= positive_target]
    sell_candidates = grouped[grouped["pred_mean"] <= negative_target]

    buy_threshold = buy_candidates["ratio_max"].max() if not buy_candidates.empty else np.nan
    sell_threshold = sell_candidates["ratio_min"].min() if not sell_candidates.empty else np.nan

    if pd.isna(buy_threshold) or pd.isna(sell_threshold) or buy_threshold >= sell_threshold:
        buy_threshold = df["ratio"].quantile(0.35)
        sell_threshold = df["ratio"].quantile(0.65)

    return float(buy_threshold), float(sell_threshold)


def build_position_table(
    ratio: pd.Series,
    pred: np.ndarray,
    baseline_table: Iterable[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    bins = list(baseline_table)
    min_weight = min(w for _, _, w in bins)
    max_weight = max(w for _, _, w in bins)

    bin_means: list[float] = []
    for lo, hi, _ in bins:
        mask = (ratio >= lo) & (ratio <= hi)
        if mask.any():
            bin_means.append(float(pred[mask].mean()))
        else:
            bin_means.append(np.nan)

    overall_mean = float(np.nanmean(bin_means)) if not np.isnan(np.nanmean(bin_means)) else 0.0
    bin_means = [overall_mean if np.isnan(v) else v for v in bin_means]

    min_pred = min(bin_means)
    max_pred = max(bin_means)

    if np.isclose(min_pred, max_pred):
        weights = [w for _, _, w in bins]
    else:
        weights = [
            min_weight + (v - min_pred) / (max_pred - min_pred) * (max_weight - min_weight)
            for v in bin_means
        ]

    # Enforce monotonic non-increasing weights (cheap ratio -> higher weight)
    monotonic_weights = []
    prev = 1.0
    for w in weights:
        w = min(w, prev)
        w = max(0.0, min(1.0, w))
        monotonic_weights.append(w)
        prev = w

    new_table = []
    for (lo, hi, _), w in zip(bins, monotonic_weights):
        new_table.append((float(lo), float(hi), float(w)))

    return new_table


def run_backtest_with_params(
    frame: pd.DataFrame,
    position_table: list[tuple[float, float, float]],
    buy_threshold: float,
    sell_threshold: float,
) -> dict:
    pm = PortfolioManager(
        initial_capital=PortfolioConfig.INITIAL_PORTFOLIO_USD,
        position_table=position_table,
        max_daily_change=PortfolioConfig.MAX_DAILY_WEIGHT_CHANGE,
    )

    first_price = frame.iloc[0]["btc_price"]
    pm.initialize_holdings(first_price, initial_btc_quantity=2.0)

    for _, row in frame.iterrows():
        price = float(row["btc_price"])
        ratio = float(row["ratio"])

        rebalance = pm.rebalance(ratio, enforce_limit=True)
        pm.execute_rebalance(price, rebalance["actual_weight"])

        if ratio < buy_threshold:
            signal = "BUY"
        elif ratio > sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        pm.add_to_history(
            row["date"].strftime("%Y-%m-%d"),
            price,
            pm.btc_quantity,
            ratio,
            signal,
        )

    portfolio_df = pm.get_portfolio_dataframe()
    return pm.calculate_metrics(portfolio_df)


def format_table(table: list[tuple[float, float, float]]) -> str:
    lines = []
    for lo, hi, w in table:
        lines.append(f"    ({lo:.3f}, {hi:.3f}, {w:.3f}),")
    return "\n".join(lines)


def update_config(
    config_path: str,
    buy_threshold: float,
    sell_threshold: float,
    position_table: list[tuple[float, float, float]],
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = re.sub(
        r"RATIO_BUY_THRESHOLD\s*=\s*[-0-9.]+",
        f"RATIO_BUY_THRESHOLD = {buy_threshold:.4f}",
        text,
    )
    text = re.sub(
        r"RATIO_SELL_THRESHOLD\s*=\s*[-0-9.]+",
        f"RATIO_SELL_THRESHOLD = {sell_threshold:.4f}",
        text,
    )

    table_block = "\n".join(
        ["    POSITION_TABLE = ["]
        + [format_table(position_table)]
        + ["    ]"]
    )

    text = re.sub(
        r"POSITION_TABLE\s*=\s*\[(?:.|\n)*?\]",
        table_block,
        text,
        count=1,
    )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML-lite parameter optimizer with k-fold CV")
    parser.add_argument("--forward-days", type=int, default=7)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    frame = build_ratio_frame(forward_days=args.forward_days)

    feature_cols = [
        "ratio",
        "ratio_ema",
        "price_ema",
        "cost_ema_slow",
        "ratio_change_7d",
        "price_change_7d",
    ]

    X = frame[feature_cols].values
    y = frame["forward_return"].values
    ratio = frame["ratio"].values

    # k-fold time series split
    n = len(frame)
    fold_size = n // args.k_folds
    
    all_baseline_test = []
    all_optimized_test = []
    best_params = {"buy": 0.90, "sell": 1.10, "table": PortfolioConfig.POSITION_TABLE}
    best_score = -np.inf

    print(f"\n[CROSS-VALIDATION] {args.k_folds}-fold time series split")
    print(f"Fold size: {fold_size} samples\n")

    for fold in range(args.k_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < args.k_folds - 1 else n

        X_test_fold = X[test_start:test_end]
        y_test_fold = y[test_start:test_end]
        ratio_test_fold = ratio[test_start:test_end]

        X_train_fold = np.concatenate([X[:test_start], X[test_end:]])
        y_train_fold = np.concatenate([y[:test_start], y[test_end:]])
        ratio_train_fold = np.concatenate([ratio[:test_start], ratio[test_end:]])

        # Fit model
        model, scaler = fit_ridge_regression(X_train_fold, y_train_fold, l2=args.l2)
        pred_train_fold = predict_ridge(X_train_fold, model, scaler)
        pred_test_fold = predict_ridge(X_test_fold, model, scaler)

        # Derive params from train fold
        buy_thresh, sell_thresh = derive_thresholds(
            pd.Series(ratio_train_fold), pred_train_fold
        )
        opt_table = build_position_table(
            pd.Series(ratio_train_fold),
            pred_train_fold,
            baseline_table=PortfolioConfig.POSITION_TABLE,
        )

        # Test on hold-out fold
        test_frame_fold = frame.iloc[test_start:test_end]
        
        baseline_metrics = run_backtest_with_params(
            test_frame_fold,
            position_table=PortfolioConfig.POSITION_TABLE,
            buy_threshold=SignalConfig.RATIO_BUY_THRESHOLD,
            sell_threshold=SignalConfig.RATIO_SELL_THRESHOLD,
        )

        optimized_metrics = run_backtest_with_params(
            test_frame_fold,
            position_table=opt_table,
            buy_threshold=buy_thresh,
            sell_threshold=sell_thresh,
        )

        all_baseline_test.append(baseline_metrics)
        all_optimized_test.append(optimized_metrics)

        # Track best fold
        opt_score = optimized_metrics.get("sharpe_ratio", 0)
        if opt_score > best_score:
            best_score = opt_score
            best_params = {
                "buy": buy_thresh,
                "sell": sell_thresh,
                "table": opt_table,
            }

        print(f"[FOLD {fold + 1}] Baseline Sharpe: {baseline_metrics['sharpe_ratio']:.4f} | "
              f"Optimized Sharpe: {optimized_metrics['sharpe_ratio']:.4f}")

    # Average metrics across folds
    avg_baseline = {
        k: np.mean([m[k] for m in all_baseline_test])
        for k in all_baseline_test[0].keys()
    }
    avg_optimized = {
        k: np.mean([m[k] for m in all_optimized_test])
        for k in all_optimized_test[0].keys()
    }

    print("\n" + "=" * 70)
    print("[CROSS-VALIDATION RESULTS] (averaged across folds)")
    print("=" * 70)

    print("\n[RECOMMENDED] RATIO THRESHOLDS (from best fold)")
    print(f"  RATIO_BUY_THRESHOLD  = {best_params['buy']:.4f}")
    print(f"  RATIO_SELL_THRESHOLD = {best_params['sell']:.4f}")

    print("\n[RECOMMENDED] POSITION_TABLE:")
    print(format_table(best_params["table"]))

    print("\n[AVG METRICS] BASELINE (test folds):")
    for k, v in avg_baseline.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    print("\n[AVG METRICS] OPTIMIZED (test folds):")
    for k, v in avg_optimized.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    # Comparison
    print("\n[IMPROVEMENT]")
    print(f"  Sharpe: {avg_baseline['sharpe_ratio']:.4f} → {avg_optimized['sharpe_ratio']:.4f} "
          f"({(avg_optimized['sharpe_ratio'] - avg_baseline['sharpe_ratio']) / avg_baseline['sharpe_ratio'] * 100:+.2f}%)")
    print(f"  Max DD: {avg_baseline['max_drawdown_pct']:.2f}% → {avg_optimized['max_drawdown_pct']:.2f}% "
          f"({avg_optimized['max_drawdown_pct'] - avg_baseline['max_drawdown_pct']:+.2f}pp)")
    print(f"  Total Return: {avg_baseline['total_return_pct']:.2f}% → {avg_optimized['total_return_pct']:.2f}% "
          f"({avg_optimized['total_return_pct'] - avg_baseline['total_return_pct']:+.2f}pp)")

    if args.write_config:
        config_path = os.path.join(PROJECT_ROOT, "config.py")
        update_config(
            config_path=config_path,
            buy_threshold=best_params["buy"],
            sell_threshold=best_params["sell"],
            position_table=best_params["table"],
        )
        print(f"\n[OK] Updated config.py with optimized parameters")


if __name__ == "__main__":
    main()
