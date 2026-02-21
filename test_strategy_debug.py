"""
Test script to debug strategy execution
"""
import pandas as pd
import numpy as np
from strategy import TradingStrategy
from data_fetcher import DataFetcher

# Fetch some test data
print("Fetching data...")
data = DataFetcher.fetch_combined_data(days=400, use_real_data=True, force_refresh=False)
print(f"Got {len(data)} days of data")
print(f"Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")

# Initialize strategy
strategy = TradingStrategy(initial_capital=100_000)
prices = np.array(data['btc_price'].values, dtype=float)

# Run for 200 days (after warmup)
print("\n" + "="*70)
print("RUNNING STRATEGY")
print("="*70)

confirmed_count = 0
executed_count = 0

for idx in range(200, min(250, len(data))):  # Just 50 days for testing
    row = data.iloc[idx]
    returns_30 = np.diff(prices[max(0, idx-30):idx+1]) / prices[max(0, idx-30):idx]
    
    result = strategy.run_daily_cycle(
        date=row['date'],
        btc_price=row['btc_price'],
        production_cost=row['production_cost'],
        prices_last_200=prices[:idx+1],
        prices_last_90=prices[max(0, idx-90):idx+1],
        daily_returns_30=returns_30
    )
    
    if result['confirmed']:
        confirmed_count += 1
        print(f"\n[{row['date']}] Price: ${row['btc_price']:,.0f}")
        print(f"  Target: {result['target_btc']*100:.1f}% BTC")
        print(f"  Current: {result['current_position_btc']*100:.1f}% BTC")
        print(f"  Confirmed: {result['confirmation_reason']}")
        print(f"  Score: {result['score_adjusted']:.2f}")
        
    if result['executed']:
        executed_count += 1
        print(f"  ✅ EXECUTED: {result['execution_stage']}")

# Get log
log_df = strategy.get_execution_log_df()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total days simulated: {len(log_df)}")
print(f"Confirmations: {confirmed_count}")
print(f"Executions: {executed_count}")
print(f"\nTarget BTC weight distribution:")
print(log_df['target_btc'].value_counts().sort_index())
print(f"\nCurrent position changes: {(log_df['current_position_btc'].diff().abs() > 0.001).sum()}")

# Show first/last positions
print(f"\nFirst position: {log_df['current_position_btc'].iloc[0]*100:.1f}% BTC")
print(f"Last position: {log_df['current_position_btc'].iloc[-1]*100:.1f}% BTC")

# Calculate simple portfolio value
initial_price = log_df['btc_price'].iloc[0]
final_price = log_df['btc_price'].iloc[-1]
initial_btc_weight = log_df['current_position_btc'].iloc[0]

# Buy & Hold return
bh_return = ((final_price - initial_price) / initial_price) * 100

print(f"\nBTC Buy & Hold return: {bh_return:+.2f}%")
print(f"Price: ${initial_price:,.0f} → ${final_price:,.0f}")
