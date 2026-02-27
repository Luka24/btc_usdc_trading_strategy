"""
Ablation study: which overlays / components actually contribute?

Tests 8 configurations (baseline + ablations) on the full OOS 2020-2025 windows.
Best params from best_params.json are loaded first; each ablation disables one
feature by clamping its config value to "off".

Output: table showing OOS Sortino and MaxDD per configuration, delta vs baseline.

Usage:
    python optimization/ablation_study.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from optimization.walk_forward import FOLDS, load_full_data, run_fold

BEST_PARAMS_PATH = os.path.join(CURRENT_DIR, "best_params.json")
SEP = "=" * 70


# ── Load & apply best params ─────────────────────────────────────────────────
def _load_best_params() -> dict:
    with open(BEST_PARAMS_PATH) as f:
        return json.load(f)


def _apply_best_params(bp: dict) -> None:
    ga, gb, gc = bp["group_a"], bp["group_b"], bp["group_c"]
    cfg.SignalConfig.PRICE_EMA_WINDOW        = ga["PRICE_EMA_WINDOW"]
    cfg.SignalConfig.COST_EMA_WINDOW         = ga["COST_EMA_WINDOW"]
    cfg.SignalConfig.SIGNAL_EMA_WINDOW       = ga["SIGNAL_EMA_WINDOW"]
    cfg.PortfolioConfig.TREND_FILTER_WINDOW  = ga["TREND_FILTER_WINDOW"]
    cfg.PortfolioConfig.RSI_OVERSOLD         = ga["RSI_OVERSOLD"]
    cfg.PortfolioConfig.VOL_TARGET           = ga["VOL_TARGET"]
    cfg.RiskManagementConfig.DD_THRESHOLDS.update(gb["DD_THRESHOLDS"])
    cfg.RiskManagementConfig.VOL_THRESHOLDS.update(gb["VOL_THRESHOLDS"])
    cfg.PortfolioConfig.HASH_RIBBON_CAP_MULT = gc["HASH_RIBBON_CAP_MULT"]


# ── Metric helpers ────────────────────────────────────────────────────────────
def _metrics_from_folds(df_full: pd.DataFrame) -> dict:
    """Run all 5 OOS folds, return mean Sortino and mean MaxDD."""
    sortinos, maxdds = [], []
    for fold in FOLDS:
        r = run_fold(df_full, fold, enable_risk_management=True)
        sortinos.append(r.get("sortino_ratio", 0.0))
        maxdds.append(r.get("max_drawdown_pct", 0.0))
    return {
        "sortino": float(np.mean(sortinos)),
        "per_fold": sortinos,
        "maxdd":    float(np.mean(maxdds)),
    }


# ── Context manager: temporarily override config attrs ───────────────────────
from contextlib import contextmanager

@contextmanager
def _override(**kwargs):
    """Temporarily set config class attributes, restore on exit."""
    saves = {}
    for key, val in kwargs.items():
        for cls in (cfg.SignalConfig, cfg.PortfolioConfig, cfg.RiskManagementConfig):
            if hasattr(cls, key):
                saves[(cls, key)] = copy.deepcopy(getattr(cls, key))
                setattr(cls, key, val)
                break
    try:
        yield
    finally:
        for (cls, key), val in saves.items():
            setattr(cls, key, copy.deepcopy(val))


# ── Define ablation configurations ───────────────────────────────────────────
# Each entry: (label, description, context_manager_factory)
# factory returns a context manager that disables one feature

def _ablations():
    """Return list of (label, description, ctx_factory) tuples."""
    return [
        # ── Signal overlays ────────────────────────────────────────────────
        (
            "No RSI boost",
            "RSI_OVERSOLD=999 (never fires)",
            lambda: _override(RSI_OVERSOLD=999),
        ),
        (
            "No trend filter",
            "TREND_BEAR_CAP=1.0 (price<EMA no longer caps exposure)",
            lambda: _override(TREND_BEAR_CAP=1.0),
        ),
        (
            "No vol scaling",
            "VOL_TARGET=0 (vol scaling disabled)",
            lambda: _override(VOL_TARGET=0),
        ),
        (
            "No hash ribbon",
            "HASH_RIBBON_CAP_MULT=1.0 (miner cap signal disabled)",
            lambda: _override(HASH_RIBBON_CAP_MULT=1.0),
        ),

        # ── Risk management ────────────────────────────────────────────────
        (
            "No risk manager",
            "enable_risk_management=False (DD/vol/VaR mode caps disabled)",
            None,  # handled specially below
        ),
        # ── Signal speed ──────────────────────────────────────────────────
        (
            "Slow signal (EMA=14)",
            "SIGNAL_EMA_WINDOW=14 (default, vs optimized=3)",
            lambda: _override(SIGNAL_EMA_WINDOW=14),
        ),
        (
            "Fast signal (EMA=3, opt)",
            "SIGNAL_EMA_WINDOW=3 reference — same as best_params",
            lambda: _override(SIGNAL_EMA_WINDOW=3),  # no-op, confirms baseline
        ),
    ]


# ── Run ablations ─────────────────────────────────────────────────────────────
def run_all(df_full: pd.DataFrame) -> list[dict]:
    results = []

    # Baseline: best_params, all features on
    print("  Running baseline ...", flush=True)
    base = _metrics_from_folds(df_full)
    results.append({"label": "BASELINE (best_params)", "desc": "all features on", **base})

    for label, desc, ctx_factory in _ablations():
        print(f"  Running: {label} ...", flush=True)
        if label == "No risk manager":
            # Special: pass enable_risk_management=False to each fold
            sortinos, maxdds = [], []
            for fold in FOLDS:
                r = run_fold(df_full, fold, enable_risk_management=False)
                sortinos.append(r.get("sortino_ratio", 0.0))
                maxdds.append(r.get("max_drawdown_pct", 0.0))
            m = {"sortino": float(np.mean(sortinos)), "per_fold": sortinos,
                 "maxdd": float(np.mean(maxdds))}
        else:
            ctx = ctx_factory()
            with ctx:
                m = _metrics_from_folds(df_full)
        results.append({"label": label, "desc": desc, **m})

    return results


# ── Print results ─────────────────────────────────────────────────────────────
def print_results(results: list[dict]) -> None:
    baseline_s = results[0]["sortino"]
    baseline_d = results[0]["maxdd"]

    print(f"\n{SEP}")
    print("ABLATION STUDY — OOS 2020-mid2025 (5 folds)")
    print(SEP)
    print(
        f"{'Configuration':<30} {'Sortino':>8} {'Delta':>8} {'MaxDD%':>8} {'Delta':>8}  Description"
    )
    print("-" * 90)

    for r in results:
        ds = r["sortino"] - baseline_s
        dd_delta = r["maxdd"] - baseline_d
        ds_str = f"{ds:+.3f}" if r["label"] != "BASELINE (best_params)" else "  ref  "
        dd_delta_str = f"{dd_delta:+.1f}" if r["label"] != "BASELINE (best_params)" else "  ref  "
        print(
            f"{r['label']:<30} {r['sortino']:>8.3f} {ds_str:>8} {r['maxdd']:>7.1f}%"
            f" {dd_delta_str:>8}  {r['desc']}"
        )

    print(f"\n{SEP}")
    print("PER-FOLD SORTINO BREAKDOWN")
    print(SEP)
    fold_names = [f.name for f in FOLDS]
    header = f"{'Configuration':<30}" + "".join(f" {n:>12}" for n in fold_names) + f" {'MEAN':>8}"
    print(header)
    print("-" * (30 + 12 * len(fold_names) + 9))
    for r in results:
        row = f"{r['label']:<30}"
        for v in r["per_fold"]:
            row += f" {v:>12.3f}"
        row += f" {r['sortino']:>8.3f}"
        print(row)

    print(f"\n{SEP}")
    print("INTERPRETATION GUIDE")
    print(SEP)
    print("  Delta > +0.05  : feature adds meaningful value — keep it")
    print("  Delta -0.05..+0.05: feature is noise — consider removing")
    print("  Delta < -0.05  : disabling feature HELPS — feature is harmful")
    print()
    # Highlight findings
    meaningful = [r for r in results[1:] if abs(r["sortino"] - baseline_s) > 0.05]
    if meaningful:
        print("  Notable findings:")
        for r in meaningful:
            ds = r["sortino"] - baseline_s
            verdict = "HARMFUL (worth removing)" if ds > 0.05 else "ADDS VALUE (keep)"
            print(f"    {r['label']}: {ds:+.3f} Sortino  -> removing is {verdict}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(SEP)
    print("ABLATION STUDY: BTC/USDC strategy component evaluation")
    print(SEP)

    # Load best params
    if not os.path.exists(BEST_PARAMS_PATH):
        print("ERROR: best_params.json not found. Run optimizer first.")
        sys.exit(1)
    bp = _load_best_params()
    _apply_best_params(bp)
    print(f"Loaded best_params.json (Sortino={bp['final_sortino']:.3f})")

    # Load data
    print("Loading data ...")
    df_full = load_full_data()
    print(f"Data: {len(df_full)} rows, {df_full['date'].min()} to {df_full['date'].max()}")
    print()

    # Run
    results = run_all(df_full)
    print_results(results)


if __name__ == "__main__":
    main()
