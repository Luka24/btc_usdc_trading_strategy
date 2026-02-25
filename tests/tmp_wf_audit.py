"""
Walk-forward integrity check.
Tests: for each major parameter group, was it set on TRAIN (<2024-01-01) only?
"""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from data_fetcher import DataFetcher
from backtest import BacktestEngine
import config as cfg

data = DataFetcher.fetch_combined_data(days=3000, use_real_data=True, force_refresh=False)
data['date'] = pd.to_datetime(data['date'])
train = data[data['date'] < '2024-01-01'].reset_index(drop=True)
oos   = data[data['date'] >= '2024-01-01'].reset_index(drop=True)

def run_metrics(df, label):
    e = BacktestEngine(100000, True)
    e.add_from_dataframe(df.copy())
    e.run_backtest(2.0)
    m = e.calculate_metrics()
    return m['sharpe_ratio'], m['total_return_pct'], m['max_drawdown_pct']

# ── BASELINE: no overlays, just base production-cost signal ──────────────
import config as cfg
orig_hr = cfg.PortfolioConfig.HASH_RIBBON_ENABLED
orig_h  = cfg.PortfolioConfig.HALVING_ENABLED
orig_s  = cfg.PortfolioConfig.SEASONAL_ENABLED

cfg.PortfolioConfig.HASH_RIBBON_ENABLED = False
cfg.PortfolioConfig.HALVING_ENABLED     = False
cfg.PortfolioConfig.SEASONAL_ENABLED    = False
t_sh, t_ret, t_dd = run_metrics(train, 'TRAIN')
o_sh, o_ret, o_dd = run_metrics(oos,   'OOS')
print('BASELINE (no overlays)')
print('  TRAIN Sharpe=%.3f  Return=%.1f%%  maxDD=%.1f%%' % (t_sh, t_ret, t_dd))
print('  OOS   Sharpe=%.3f  Return=%.1f%%  maxDD=%.1f%%' % (o_sh, o_ret, o_dd))
print()

# ── +HASH RIBBON ────────────────────────────────────────────────────────
cfg.PortfolioConfig.HASH_RIBBON_ENABLED = True
t_sh2, _, _ = run_metrics(train, 'TRAIN')
o_sh2, _, _ = run_metrics(oos,   'OOS')
print('+Hash Ribbon (30/60, cap=0.00)')
print('  TRAIN Sharpe=%.3f  delta=%.3f' % (t_sh2, t_sh2 - t_sh))
print('  OOS   Sharpe=%.3f  delta=%.3f' % (o_sh2, o_sh2 - o_sh))
print()

# ── +HALVING ────────────────────────────────────────────────────────────
cfg.PortfolioConfig.HALVING_ENABLED = True
t_sh3, _, _ = run_metrics(train, 'TRAIN')
o_sh3, _, _ = run_metrics(oos,   'OOS')
print('+Halving (mult=0.10, early_bear phase)')
print('  TRAIN Sharpe=%.3f  delta=%.3f' % (t_sh3, t_sh3 - t_sh2))
print('  OOS   Sharpe=%.3f  delta=%.3f' % (o_sh3, o_sh3 - o_sh2))
print()

# ── +SEASONAL ────────────────────────────────────────────────────────────
cfg.PortfolioConfig.SEASONAL_ENABLED = True
t_sh4, _, _ = run_metrics(train, 'TRAIN')
o_sh4, _, _ = run_metrics(oos,   'OOS')
print('+Seasonal (Aug=0.70)')
print('  TRAIN Sharpe=%.3f  delta=%.3f' % (t_sh4, t_sh4 - t_sh3))
print('  OOS   Sharpe=%.3f  delta=%.3f' % (o_sh4, o_sh4 - o_sh3))
print()

print('=== SUMMARY ===')
print('TRAIN: %.3f -> %.3f -> %.3f -> %.3f  (each overlay adds value?)' % (t_sh, t_sh2, t_sh3, t_sh4))
print('OOS:   %.3f -> %.3f -> %.3f -> %.3f  (does improvement hold OOS?)' % (o_sh, o_sh2, o_sh3, o_sh4))

# ── HALVING mult grid on TRAIN only ──────────────────────────────────────
print()
print('=== HALVING_BEAR_MULT grid on TRAIN-only (to check if 0.10 was picked by OOS peeking) ===')
cfg.PortfolioConfig.HASH_RIBBON_ENABLED = True
cfg.PortfolioConfig.SEASONAL_ENABLED = True
print('%-6s   TRAIN   OOS' % 'mult')
for mult in [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
    cfg.PortfolioConfig.HALVING_ENABLED = True
    cfg.PortfolioConfig.HALVING_BEAR_MULT = mult
    ts, _, _ = run_metrics(train, '')
    os_, _, _ = run_metrics(oos, '')
    print('  %.2f     %.3f   %.3f  %s' % (mult, ts, os_, '<-- SELECTED' if mult == 0.10 else ''))
    
cfg.PortfolioConfig.HALVING_BEAR_MULT = 0.10  # restore
