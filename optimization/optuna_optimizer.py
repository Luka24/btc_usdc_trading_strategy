"""
Optuna-based three-phase parameter optimizer.

Optimizes on mean OOS Sortino ratio across all 5 walk-forward folds.

Three sequential phases (expanding-parameter strategy):
  Phase A  — Core signal   (6 params),  B+C held at defaults  (~45 trials)
  Phase B  — Risk manager  (6 params),  A fixed at best-A     (~35 trials)
  Phase C  — Overlays      (2 params),  A+B fixed at best     (~20 trials)

Hands-off test set (2025-07-01 → today) is NEVER touched here.

Usage
-----
  # Full three-phase run (default trials per phase):
  python optimization/optuna_optimizer.py

  # Fast smoke test:
  python optimization/optuna_optimizer.py --phase a --trials 5 --smoke

  # Run a single phase:
  python optimization/optuna_optimizer.py --phase b --trials 80

  # After all phases: print best params and write to best_params.json:
  python optimization/optuna_optimizer.py --report

Outputs
-------
  optimization/best_params.json   – machine-readable best params (all phases)
  optimization/study_phase_a.db   – SQLite Optuna study (phase A)
  optimization/study_phase_b.db   – SQLite Optuna study (phase B)
  optimization/study_phase_c.db   – SQLite Optuna study (phase C)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from optimization.walk_forward import FOLDS, load_full_data, run_fold  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optuna_opt")

OUT_DIR = CURRENT_DIR
BEST_PARAMS_PATH = os.path.join(OUT_DIR, "best_params.json")

# Snapshot defaults once at import so _patched_config can restore them after each trial.
_DEFAULT_SIGNAL: dict[str, Any] = {}
_DEFAULT_PORTFOLIO: dict[str, Any] = {}
_DEFAULT_RISK: dict[str, Any] = {}


def _snapshot_defaults() -> None:
    global _DEFAULT_SIGNAL, _DEFAULT_PORTFOLIO, _DEFAULT_RISK
    _DEFAULT_SIGNAL = {
        "PRICE_EMA_WINDOW": cfg.SignalConfig.PRICE_EMA_WINDOW,
        "COST_EMA_WINDOW": cfg.SignalConfig.COST_EMA_WINDOW,
        "SIGNAL_EMA_WINDOW": cfg.SignalConfig.SIGNAL_EMA_WINDOW,
    }
    _DEFAULT_PORTFOLIO = {
        "TREND_FILTER_WINDOW": cfg.PortfolioConfig.TREND_FILTER_WINDOW,
        "RSI_OVERSOLD": cfg.PortfolioConfig.RSI_OVERSOLD,
        "VOL_TARGET": cfg.PortfolioConfig.VOL_TARGET,
        "HASH_RIBBON_CAP_MULT": cfg.PortfolioConfig.HASH_RIBBON_CAP_MULT,
    }
    _DEFAULT_RISK = {
        "DD_THRESHOLDS": copy.deepcopy(cfg.RiskManagementConfig.DD_THRESHOLDS),
        "VOL_THRESHOLDS": copy.deepcopy(cfg.RiskManagementConfig.VOL_THRESHOLDS),
        "VAR_THRESHOLDS": copy.deepcopy(cfg.RiskManagementConfig.VAR_THRESHOLDS),
    }


_snapshot_defaults()


@contextmanager
def _patched_config(overrides: dict[str, Any]):
    """Temporarily patch config classes with overrides, then restore. Not thread-safe."""
    for key, val in overrides.items():
        if hasattr(cfg.SignalConfig, key):
            setattr(cfg.SignalConfig, key, val)
        elif hasattr(cfg.PortfolioConfig, key):
            setattr(cfg.PortfolioConfig, key, val)
        elif hasattr(cfg.RiskManagementConfig, key):
            setattr(cfg.RiskManagementConfig, key, val)
        else:
            raise KeyError(f"Unknown config key: {key}")
    try:
        yield
    finally:
        for k, v in _DEFAULT_SIGNAL.items():
            setattr(cfg.SignalConfig, k, v)
        for k, v in _DEFAULT_PORTFOLIO.items():
            setattr(cfg.PortfolioConfig, k, copy.deepcopy(v))
        for k, v in _DEFAULT_RISK.items():
            setattr(cfg.RiskManagementConfig, k, copy.deepcopy(v))


def _mean_oos_sortino(
    df_full,
    overrides: dict[str, Any],
    folds=None,
    penalise_bad_folds: bool = True,
) -> float:
    """Run all walk-forward folds with the given overrides, return mean OOS Sortino."""
    if folds is None:
        folds = FOLDS
    try:
        with _patched_config(overrides):
            sortinos: list[float] = []
            for fold in folds:
                result = run_fold(df_full, fold, enable_risk_management=True)
                if result.get("error"):
                    sortinos.append(-5.0)
                else:
                    sortinos.append(float(result["sortino_ratio"]))

        if not sortinos:
            return -99.0

        mean_s = float(np.mean(sortinos))
        std_s = float(np.std(sortinos))

        if penalise_bad_folds:
            # penalise folds that go deeply negative and high cross-fold variance
            worst = min(sortinos)
            mean_s -= max(0.0, -worst) * 0.3
            mean_s -= std_s * 0.15

        return mean_s

    except Exception as exc:
        log.debug("Trial failed: %s", exc)
        return -99.0


# Group default dicts — used when a phase is skipped
GROUP_A_DEFAULTS: dict[str, Any] = {
    "PRICE_EMA_WINDOW": cfg.SignalConfig.PRICE_EMA_WINDOW,
    "COST_EMA_WINDOW": cfg.SignalConfig.COST_EMA_WINDOW,
    "SIGNAL_EMA_WINDOW": cfg.SignalConfig.SIGNAL_EMA_WINDOW,
    "TREND_FILTER_WINDOW": cfg.PortfolioConfig.TREND_FILTER_WINDOW,
    "RSI_OVERSOLD": cfg.PortfolioConfig.RSI_OVERSOLD,
    "VOL_TARGET": cfg.PortfolioConfig.VOL_TARGET,
}

GROUP_B_DEFAULTS: dict[str, Any] = {
    "DD_THRESHOLDS": copy.deepcopy(cfg.RiskManagementConfig.DD_THRESHOLDS),
    "VOL_THRESHOLDS": copy.deepcopy(cfg.RiskManagementConfig.VOL_THRESHOLDS),
    "VAR_THRESHOLDS": copy.deepcopy(cfg.RiskManagementConfig.VAR_THRESHOLDS),
}

GROUP_C_DEFAULTS: dict[str, Any] = {
    "HASH_RIBBON_CAP_MULT": cfg.PortfolioConfig.HASH_RIBBON_CAP_MULT,
}


def _suggest_group_a(trial) -> dict[str, Any]:
    return {
        "PRICE_EMA_WINDOW": trial.suggest_int("price_ema_window", 7, 50),
        "COST_EMA_WINDOW": trial.suggest_int("cost_ema_window", 10, 60),
        "SIGNAL_EMA_WINDOW": trial.suggest_int("signal_ema_window", 3, 30),
        "TREND_FILTER_WINDOW": trial.suggest_int("trend_filter_window", 100, 400, step=10),
        "RSI_OVERSOLD": trial.suggest_int("rsi_oversold", 20, 40),
        "VOL_TARGET": trial.suggest_float("vol_target", 0.20, 0.80, step=0.05),
    }


def _suggest_group_b(trial) -> dict[str, Any]:
    # Sample 3 unconstrained values per metric, then sort to enforce ordering.
    # Fixed bounds avoid Optuna's TPE warnings about inconsistent distributions.
    dd_raw = sorted(
        [
            trial.suggest_float("dd_raw_1", -0.60, -0.04, step=0.01),
            trial.suggest_float("dd_raw_2", -0.60, -0.04, step=0.01),
            trial.suggest_float("dd_raw_3", -0.60, -0.04, step=0.01),
        ],
        reverse=True,
    )
    dd_caution   = dd_raw[0]
    dd_risk_off  = min(dd_raw[1], dd_caution  - 0.03)
    dd_emergency = min(dd_raw[2], dd_risk_off - 0.05)

    vol_raw = sorted(
        [
            trial.suggest_float("vol_raw_1", 0.40, 2.20, step=0.05),
            trial.suggest_float("vol_raw_2", 0.40, 2.20, step=0.05),
            trial.suggest_float("vol_raw_3", 0.40, 2.20, step=0.05),
        ]
    )
    vol_caution   = vol_raw[0]
    vol_risk_off  = max(vol_raw[1], vol_caution  + 0.10)
    vol_emergency = max(vol_raw[2], vol_risk_off + 0.10)

    return {
        "DD_THRESHOLDS": {
            "CAUTION":   dd_caution,
            "RISK_OFF":  dd_risk_off,
            "EMERGENCY": dd_emergency,
        },
        "VOL_THRESHOLDS": {
            "CAUTION":   vol_caution,
            "RISK_OFF":  vol_risk_off,
            "EMERGENCY": vol_emergency,
        },
        "VAR_THRESHOLDS": copy.deepcopy(cfg.RiskManagementConfig.VAR_THRESHOLDS),
    }


def _suggest_group_c(trial) -> dict[str, Any]:
    return {
        "HASH_RIBBON_CAP_MULT": trial.suggest_float("hash_ribbon_cap_mult", 0.00, 0.40, step=0.05),
    }


def _make_study(name: str, storage: Optional[str] = None):
    """Create (or resume) an Optuna study. Pass storage='MEMORY' for in-memory (smoke tests)."""
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("optuna not installed — run: pip install optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    in_memory = storage == "MEMORY"
    kwargs: dict[str, Any] = {
        "study_name": name,
        "direction": "maximize",
        "load_if_exists": not in_memory,
    }
    if not in_memory and storage:
        kwargs["storage"] = storage
    return optuna.create_study(**kwargs)


def run_phase_a(
    df_full,
    n_trials: int = 45,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Phase A: optimize core signal params with B+C at defaults."""
    storage = "MEMORY" if (db_path is None or db_path == "MEMORY") else f"sqlite:///{db_path}"
    study = _make_study("phase_a", storage)
    remaining = max(0, n_trials - len(study.trials))

    log.info("Phase A — core signal  (%d trials, %d already done)", n_trials, n_trials - remaining)
    log.info("  params: PRICE_EMA_WINDOW, COST_EMA_WINDOW, SIGNAL_EMA_WINDOW,")
    log.info("          TREND_FILTER_WINDOW, RSI_OVERSOLD, VOL_TARGET")

    b_fixed = copy.deepcopy(GROUP_B_DEFAULTS)
    c_fixed = copy.deepcopy(GROUP_C_DEFAULTS)

    def objective(trial):
        return _mean_oos_sortino(df_full, {**_suggest_group_a(trial), **b_fixed, **c_fixed})

    t0 = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=1,
                       show_progress_bar=False, callbacks=[_log_callback])
    log.info("Phase A done in %.0fs", time.time() - t0)

    p = study.best_trial.params
    best_a = {
        "PRICE_EMA_WINDOW": p["price_ema_window"],
        "COST_EMA_WINDOW": p["cost_ema_window"],
        "SIGNAL_EMA_WINDOW": p["signal_ema_window"],
        "TREND_FILTER_WINDOW": p["trend_filter_window"],
        "RSI_OVERSOLD": p["rsi_oversold"],
        "VOL_TARGET": p["vol_target"],
    }
    _log_params("Phase A best", best_a, study.best_value)
    return best_a


def run_phase_b(
    df_full,
    best_a: dict[str, Any],
    n_trials: int = 35,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Phase B: optimize risk manager thresholds with A fixed."""
    storage = "MEMORY" if (db_path is None or db_path == "MEMORY") else f"sqlite:///{db_path}"
    study = _make_study("phase_b", storage)
    remaining = max(0, n_trials - len(study.trials))

    log.info("Phase B — risk manager  (%d trials, %d already done)", n_trials, n_trials - remaining)
    log.info("  params: DD_THRESHOLDS x3, VOL_THRESHOLDS x3")

    a_fixed = copy.deepcopy(best_a)
    c_fixed = copy.deepcopy(GROUP_C_DEFAULTS)

    def objective(trial):
        return _mean_oos_sortino(df_full, {**a_fixed, **_suggest_group_b(trial), **c_fixed})

    t0 = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=1,
                       show_progress_bar=False, callbacks=[_log_callback])
    log.info("Phase B done in %.0fs", time.time() - t0)

    best = study.best_trial

    p = best.params
    # Re-apply same ordering logic used in _suggest_group_b
    dd_sorted = sorted([p["dd_raw_1"], p["dd_raw_2"], p["dd_raw_3"]], reverse=True)
    dd_caution   = dd_sorted[0]
    dd_risk_off  = min(dd_sorted[1], dd_caution  - 0.03)
    dd_emergency = min(dd_sorted[2], dd_risk_off - 0.05)

    vol_sorted = sorted([p["vol_raw_1"], p["vol_raw_2"], p["vol_raw_3"]])
    vol_caution   = vol_sorted[0]
    vol_risk_off  = max(vol_sorted[1], vol_caution  + 0.10)
    vol_emergency = max(vol_sorted[2], vol_risk_off + 0.10)

    best_b: dict[str, Any] = {
        "DD_THRESHOLDS": {
            "CAUTION": dd_caution,
            "RISK_OFF": dd_risk_off,
            "EMERGENCY": dd_emergency,
        },
        "VOL_THRESHOLDS": {
            "CAUTION": vol_caution,
            "RISK_OFF": vol_risk_off,
            "EMERGENCY": vol_emergency,
        },
        "VAR_THRESHOLDS": copy.deepcopy(cfg.RiskManagementConfig.VAR_THRESHOLDS),
    }
    _log_params("Phase B best", best_b, study.best_value)
    return best_b


def run_phase_c(
    df_full,
    best_a: dict[str, Any],
    best_b: dict[str, Any],
    n_trials: int = 20,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Phase C: optimize overlay multipliers with A+B fixed."""
    storage = "MEMORY" if (db_path is None or db_path == "MEMORY") else f"sqlite:///{db_path}"
    study = _make_study("phase_c", storage)
    remaining = max(0, n_trials - len(study.trials))

    log.info("Phase C — overlays  (%d trials, %d already done)", n_trials, n_trials - remaining)
    log.info("  params: HASH_RIBBON_CAP_MULT")

    a_fixed = copy.deepcopy(best_a)
    b_fixed = copy.deepcopy(best_b)

    def objective(trial):
        return _mean_oos_sortino(df_full, {**a_fixed, **b_fixed, **_suggest_group_c(trial)})

    t0 = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=1,
                       show_progress_bar=False, callbacks=[_log_callback])
    log.info("Phase C done in %.0fs", time.time() - t0)

    best = study.best_trial

    p = best.params
    best_c: dict[str, Any] = {
        "HASH_RIBBON_CAP_MULT": p["hash_ribbon_cap_mult"],
    }
    _log_params("Phase C best", best_c, study.best_value)
    return best_c


def _log_callback(study, trial):
    if trial.number % 10 == 0:
        val = trial.value if trial.value is not None else float("nan")
        log.info("  trial #%d  value=%.3f  best=%.3f", trial.number, val, study.best_value)


def _log_params(label: str, params: dict, score: float) -> None:
    log.info("%s (Sortino=%.3f):", label, score)
    for k, v in params.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                log.info("    %s[%s] = %s", k, kk, vv)
        else:
            log.info("    %s = %s", k, v)


def save_best_params(best_a, best_b, best_c, final_sortino: float) -> None:
    payload = {
        "final_sortino": final_sortino,
        "group_a": _serialise(best_a),
        "group_b": _serialise(best_b),
        "group_c": _serialise(best_c),
    }
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Saved to %s", BEST_PARAMS_PATH)


def _serialise(d: dict) -> dict:
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {str(kk): float(vv) if isinstance(vv, float) else vv for kk, vv in v.items()}
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def load_best_params() -> Optional[dict]:
    if not os.path.exists(BEST_PARAMS_PATH):
        return None
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_best_params_to_live_config(best_a: dict, best_b: dict, best_c: dict) -> None:
    """Patch the in-process config with best params. Call before dashboard/backtest."""
    for k, v in best_a.items():
        if hasattr(cfg.SignalConfig, k):
            setattr(cfg.SignalConfig, k, v)
        elif hasattr(cfg.PortfolioConfig, k):
            setattr(cfg.PortfolioConfig, k, v)

    if "DD_THRESHOLDS" in best_b:
        cfg.RiskManagementConfig.DD_THRESHOLDS.update(best_b["DD_THRESHOLDS"])
    if "VOL_THRESHOLDS" in best_b:
        cfg.RiskManagementConfig.VOL_THRESHOLDS.update(best_b["VOL_THRESHOLDS"])
    if "VAR_THRESHOLDS" in best_b:
        cfg.RiskManagementConfig.VAR_THRESHOLDS.update(best_b["VAR_THRESHOLDS"])

    for k, v in best_c.items():
        setattr(cfg.PortfolioConfig, k, v)

    log.info("Config patched with best params.")


def final_combined_sortino(df_full, best_a, best_b, best_c) -> float:
    """Evaluate the combined best A+B+C on all 5 folds and return mean Sortino."""
    overrides = {**best_a, **best_b, **best_c}
    return _mean_oos_sortino(df_full, overrides, penalise_bad_folds=False)


def report() -> None:
    data = load_best_params()
    if data is None:
        print("No best_params.json found. Run optimization first.")
        return
    print(f"\nBest params — OOS Sortino = {data['final_sortino']:.4f}")
    print("=" * 55)
    for group in ("group_a", "group_b", "group_c"):
        print(f"\n{group.upper()}:")
        for k, v in data[group].items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"  {k}[{kk}] = {vv}")
            else:
                print(f"  {k} = {v}")
    print()


def _parse_args():
    p = argparse.ArgumentParser(description="Optuna walk-forward optimizer for BTC/USDC strategy")
    p.add_argument("--phase", choices=["a", "b", "c", "all"], default="all")
    p.add_argument("--trials", type=int, default=None, help="Override trial count for chosen phase")
    p.add_argument("--smoke", action="store_true", help="3 trials/phase, no DB (quick test)")
    p.add_argument("--report", action="store_true", help="Print best_params.json and exit")
    p.add_argument("--force-refresh", action="store_true", help="Re-download market data")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.report:
        report()
        return

    if args.smoke:
        n_a, n_b, n_c = 3, 3, 3
        db_a = db_b = db_c = "MEMORY"
        log.info("Smoke test mode: 3 trials per phase, in-memory only")
    else:
        n_a = args.trials or 45
        n_b = args.trials or 35
        n_c = args.trials or 20
        db_a = os.path.join(OUT_DIR, "study_phase_a.db")
        db_b = os.path.join(OUT_DIR, "study_phase_b.db")
        db_c = os.path.join(OUT_DIR, "study_phase_c.db")

    log.info("Loading data...")
    t0 = time.time()
    df_full = load_full_data(force_refresh=args.force_refresh)
    log.info(
        "Loaded %d rows (%s to %s) in %.1fs",
        len(df_full),
        df_full["date"].iloc[0].date(),
        df_full["date"].iloc[-1].date(),
        time.time() - t0,
    )

    saved = load_best_params()

    if args.phase in ("a", "all"):
        best_a = run_phase_a(df_full, n_trials=n_a, db_path=db_a)
    else:
        best_a = saved["group_a"] if saved else dict(GROUP_A_DEFAULTS)
        log.info("Phase A skipped, using %s params", "saved" if saved else "default")

    if args.phase in ("b", "all"):
        best_b = run_phase_b(df_full, best_a, n_trials=n_b, db_path=db_b)
    else:
        best_b = saved["group_b"] if saved else dict(GROUP_B_DEFAULTS)
        log.info("Phase B skipped, using %s params", "saved" if saved else "default")

    if args.phase in ("c", "all"):
        best_c = run_phase_c(df_full, best_a, best_b, n_trials=n_c, db_path=db_c)
    else:
        best_c = saved["group_c"] if saved else dict(GROUP_C_DEFAULTS)
        log.info("Phase C skipped, using %s params", "saved" if saved else "default")

    if args.smoke:
        log.info("Smoke done — skipping final eval and save.")
    else:
        combined = final_combined_sortino(df_full, best_a, best_b, best_c)
        log.info("Final combined A+B+C — mean OOS Sortino (unpenalised): %.4f", combined)
        save_best_params(best_a, best_b, best_c, combined)

    log.info("Done.")


if __name__ == "__main__":
    main()
