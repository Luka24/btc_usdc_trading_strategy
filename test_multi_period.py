"""
Test Strategy on Multiple Time Periods
=======================================
Runs backtest on different dataset lengths to validate strategy performance
across various market conditions and timeframes.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from strategy import TradingStrategy
from production_cost import BTCProductionCostCalculator

print("=" * 70)
print("TESTING STRATEGY ON MULTIPLE TIME PERIODS")
print("=" * 70)

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

periods = [
    ('365d', 'combined_data_365d.csv'),
    ('1491d', 'combined_data_1491d.csv'),
    ('3685d', 'combined_data_3685d.csv')
]

results_summary = []

for period_name, filename in periods:
    print(f"\n{'=' * 70}")
    print(f"Testing: {period_name}")
    print(f"{'=' * 70}")
    
    try:
        # Load data
        filepath = os.path.join('data', filename)
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate production cost if not present
        if 'production_cost' not in df.columns:
            print("Calculating production costs...")
            cost_calc = BTCProductionCostCalculator()
            production_costs = []
            for idx, row in df.iterrows():
                cost_calc.date = row['date']
                cost_calc.energy_price = cost_calc.__class__.__init__.__defaults__[0] if hasattr(cost_calc.__class__.__init__, '__defaults__') else 0.06
                cost_calc.efficiency = cost_calc.__class__.__init__.__defaults__[1] if hasattr(cost_calc.__class__.__init__, '__defaults__') else 30
                cost_calc.block_reward = cost_calc.__class__.__init__.__defaults__[2] if hasattr(cost_calc.__class__.__init__, '__defaults__') else 6.25
                
                # Reinitialize with date for historical parameters
                cost_calc_temp = BTCProductionCostCalculator(date=row['date'])
                cost = cost_calc_temp.calculate_total_cost_per_btc(
                    hashrate_eh_per_s=row['hashrate_eh_per_s']
                )
                production_costs.append(cost)
                
                if idx % 200 == 0:
                    print(f"  Progress: {idx}/{len(df)}")
            
            df['production_cost'] = production_costs
            print(f"  Completed! Avg cost: ${np.mean(production_costs):,.2f}")
        
        # Get actual date range
        start_date = df['date'].min()
        end_date = df['date'].max()
        actual_days = len(df)
        
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Total days: {actual_days} (~{actual_days/365:.1f} years)")
        
        # Initialize strategy
        strategy = TradingStrategy(initial_capital=100_000)
        
        # Prepare data arrays
        btc_prices = df['btc_price'].values
        production_costs = df['production_cost'].values
        
        execution_log = []
        
        # Run backtest
        print("Running backtest...")
        for i in range(200, len(df)):
            date = df.iloc[i]['date']
            btc_price = df.iloc[i]['btc_price']
            production_cost = df.iloc[i]['production_cost']
            
            # Prepare lookback windows
            prices_last_200 = btc_prices[max(0, i-200):i+1]
            prices_last_90 = btc_prices[max(0, i-90):i+1]
            daily_returns_30 = np.diff(btc_prices[max(0, i-30):i+1]) / btc_prices[max(0, i-30):i]
            
            # Run daily cycle
            result = strategy.run_daily_cycle(
                date=date,
                btc_price=btc_price,
                production_cost=production_cost,
                prices_last_200=prices_last_200.tolist(),
                prices_last_90=prices_last_90.tolist(),
                daily_returns_30=daily_returns_30
            )
            
            execution_log.append(result)
            
            if (i - 200) % 200 == 0:
                print(f"  Progress: {i-200}/{len(df)-200}")
        
        # Calculate performance
        exec_df = pd.DataFrame(execution_log)
        
        initial_capital = 100_000
        exec_df['portfolio_value'] = initial_capital * (
            exec_df['current_position_btc'] * (exec_df['btc_price'] / exec_df['btc_price'].iloc[0]) +
            exec_df['current_position_usdc']
        )
        
        total_return = ((exec_df['portfolio_value'].iloc[-1] / exec_df['portfolio_value'].iloc[0]) - 1) * 100
        bh_return = ((exec_df['btc_price'].iloc[-1] / exec_df['btc_price'].iloc[0]) - 1) * 100
        
        trades = exec_df['executed'].sum()
        confirmations = exec_df['confirmed'].sum()
        
        # Calculate max drawdown
        cummax = exec_df['portfolio_value'].cummax()
        drawdown = (exec_df['portfolio_value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified - assumes 0 risk-free rate)
        returns = exec_df['portfolio_value'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        
        print(f"\nRESULTS:")
        print(f"  Strategy Return: {total_return:+.2f}%")
        print(f"  Buy & Hold Return: {bh_return:+.2f}%")
        print(f"  Outperformance: {total_return - bh_return:+.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Trades: {trades}")
        print(f"  Confirmations: {confirmations} ({confirmations/len(exec_df)*100:.1f}%)")
        print(f"  Final BTC: {exec_df['current_position_btc'].iloc[-1]*100:.1f}%")
        print(f"  Final Value: ${exec_df['portfolio_value'].iloc[-1]:,.2f}")
        
        results_summary.append({
            'period': period_name,
            'days': actual_days,
            'years': actual_days / 365,
            'start': start_date.date(),
            'end': end_date.date(),
            'strategy_return': total_return,
            'bh_return': bh_return,
            'outperformance': total_return - bh_return,
            'max_dd': max_drawdown,
            'sharpe': sharpe_ratio,
            'trades': trades,
            'confirmations': confirmations,
            'conf_rate': confirmations/len(exec_df)*100
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n\n{'=' * 70}")
print("COMPREHENSIVE SUMMARY")
print(f"{'=' * 70}\n")
print(f"{'Period':<12} {'Years':<6} {'Strategy':<10} {'B&H':<10} {'Outperf':<10} {'MaxDD':<8} {'Sharpe':<7} {'Trades':<7}")
print(f"{'-'*78}")

for r in results_summary:
    print(f"{r['period']:<12} {r['years']:<6.1f} {r['strategy_return']:>+9.2f}% {r['bh_return']:>+9.2f}% {r['outperformance']:>+9.2f}% {r['max_dd']:>7.2f}% {r['sharpe']:>6.2f} {r['trades']:>6}")

print(f"\n{'=' * 70}")
print("ANALYSIS")
print(f"{'=' * 70}")

if len(results_summary) > 0:
    avg_outperf = np.mean([r['outperformance'] for r in results_summary])
    avg_sharpe = np.mean([r['sharpe'] for r in results_summary])
    
    print(f"\nAverage Outperformance: {avg_outperf:+.2f}%")
    print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    
    if avg_outperf > 0:
        print(f"\nStrategy consistently OUTPERFORMS Buy & Hold across all periods!")
    else:
        print(f"\nStrategy underperforms - needs further optimization.")
        
    # Check consistency
    all_positive = all(r['outperformance'] > 0 for r in results_summary)
    if all_positive:
        print(f"EXCELLENT: Positive outperformance in ALL tested periods!")
    else:
        negative_periods = [r['period'] for r in results_summary if r['outperformance'] <= 0]
        print(f"WARNING: Underperformance in: {', '.join(negative_periods)}")

print(f"\n{'=' * 70}")
