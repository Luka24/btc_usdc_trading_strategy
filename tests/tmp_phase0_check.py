"""Phase 0 sanity check — verify sortino_ratio is in calculate_metrics output."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import DataFetcher
from backtest import BacktestEngine

data = DataFetcher.fetch_combined_data(days=365, use_real_data=True, force_refresh=False)
eng = BacktestEngine(100_000, True)
eng.add_from_dataframe(data)
eng.run_backtest(2.0)
m = eng.calculate_metrics()

keys = ["sortino_ratio", "sharpe_ratio", "max_drawdown_pct"]
for k in keys:
    present = k in m
    val = round(m[k], 3) if present else "MISSING"
    print(f"{k}: {val}")

missing = [k for k in keys if k not in m]
if missing:
    print(f"\nFAIL — missing keys: {missing}")
    sys.exit(1)
else:
    print("\nFaza 0 OK — all three metrics present")
