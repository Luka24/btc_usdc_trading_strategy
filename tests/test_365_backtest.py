#!/usr/bin/env python3
"""Run 365-day backtest with new improvements"""

from data_fetcher import DataFetcher
from backtest import BacktestEngine
import pandas as pd

print("\n" + "="*70)
print("365-DAY BACKTEST WITH IMPROVED SYSTEM")
print("="*70)

# Fetch 365 days of data
print("\nFetching 365 days of data...")
data = DataFetcher.fetch_combined_data(days=365, use_real_data=True)

print(f"\n✓ Loaded {len(data)} days of data")
print(f"  Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
print(f"  BTC price: ${data['btc_price'].min():.0f} - ${data['btc_price'].max():.0f}")
print(f"  Hashrate: {data['hashrate_eh_per_s'].min():.1f} - {data['hashrate_eh_per_s'].max():.1f} EH/s")

# Run backtest
print("\nRunning backtest...")
engine = BacktestEngine(initial_capital=100_000)
engine.add_from_dataframe(data)

# Check block rewards
unique_rewards = sorted(engine.backtest_data['block_reward'].unique())
print(f"\n✓ Block rewards in backtest: {unique_rewards}")

# Signal analysis
print("\nSignal Analysis:")
signal_dist = engine.backtest_data['signal'].value_counts()
for signal, count in sorted(signal_dist.items()):
    pct = 100 * count / len(engine.backtest_data)
    print(f"  {signal}: {count} days ({pct:.1f}%)")

# Cost analysis
print("\nProduction Cost Analysis:")
print(f"  Range: ${engine.backtest_data['production_cost'].min():.0f} - ${engine.backtest_data['production_cost'].max():.0f}")
print(f"  Average: ${engine.backtest_data['production_cost'].mean():.0f}")

# Price to cost ratio
print("\nPrice/Cost Ratio Analysis:")
min_ratio = engine.backtest_data['signal_ratio'].min()
max_ratio = engine.backtest_data['signal_ratio'].max()
avg_ratio = engine.backtest_data['signal_ratio'].mean()
print(f"  Range: {min_ratio:.2f} - {max_ratio:.2f}")
print(f"  Average: {avg_ratio:.2f}")

# Run trading simulation
print("\nRunning trading simulation...")
results_df = engine.run_backtest(initial_btc_quantity=2.0)

# Portfolio metrics
final_row = results_df.iloc[-1]
initial_cap = 100_000
final_value = final_row['total_value']
return_pct = ((final_value - initial_cap) / initial_cap) * 100

print(f"\nPortfolio Results:")
print(f"  Initial: ${initial_cap:,.0f}")
print(f"  Final: ${final_value:,.0f}")
print(f"  Return: {return_pct:.2f}%")
if 'btc_weight' in results_df.columns:
    print(f"  Final BTC weight: {final_row['btc_weight']:.1f}%")
    print(f"  Final USDC weight: {final_row['usdc_weight']:.1f}%")

# Price movement
price_change = ((data['btc_price'].iloc[-1] - data['btc_price'].iloc[0]) / data['btc_price'].iloc[0]) * 100
print(f"\nMarket Context:")
print(f"  BTC price change: {price_change:+.2f}%")
print(f"  Starting price: ${data['btc_price'].iloc[0]:.0f}")
print(f"  Ending price: ${data['btc_price'].iloc[-1]:.0f}")

print("\n" + "="*70)
print("BACKTEST COMPLETE")
print("="*70)
