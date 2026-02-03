from backtest import BacktestEngine
from data_fetcher import DataFetcher
import pandas as pd

# Fetch 10 days of real data
data = DataFetcher.fetch_combined_data(days=10, use_real_data=True)

print("Input data:")
print(data.head(10))

# Run backtest
engine = BacktestEngine(initial_capital=100_000)

# Check first few days
print("\n\nBacktest data (first 5 days):")
for _, row in data.iterrows():
    engine.add_daily_data(row['date'], row['btc_price'], row['hashrate_eh_per_s'])
    
    # Check what was computed
    backtest_rows = engine.backtest_data.tail(1)
    if not backtest_rows.empty:
        last = backtest_rows.iloc[0]
        print(f"{last['date'].date()}: Price=${last['btc_price']:.0f}, "
              f"RawCost=${last['production_cost']:.0f}, "
              f"SmoothedCost=${last['production_cost_smoothed']:.0f}, "
              f"Ratio={last['signal_ratio']:.2f}, "
              f"Signal={last['signal']}")
