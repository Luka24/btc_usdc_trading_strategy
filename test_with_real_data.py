"""
Test with REAL hashrate data to validate production cost model
"""
import pandas as pd
import sys
sys.path.append('c:\\Users\\lukap\\Documents\\IOCNCOMI naloga\\btc_trading_strategy')

from production_cost import ProductionCostSeries

# Load real hashrate data
hashrate_df = pd.read_csv('c:/Users/lukap/Documents/IOCNCOMI naloga/data/hashrate_3650d.csv')
hashrate_df['date'] = pd.to_datetime(hashrate_df['date'])

# Calculate costs for all dates
cost_series = ProductionCostSeries(ema_window=30)
cost_series.add_from_dataframe(hashrate_df)

# Get monthly averages for comparison
cost_series.data.reset_index(inplace=True)
cost_series.data['year_month'] = cost_series.data['date'].dt.to_period('M')
monthly_costs = cost_series.data.groupby('year_month').agg({
    'total_cost_usd': 'mean',
    'energy_cost_usd': 'mean',
    'electricity_price': 'first',
    'miner_efficiency': 'first',
    'hashrate_eh_per_s': 'mean'
}).round(0)

# Target costs
target_data = """
2016-01,650
2016-07,850
2017-01,1050
2017-07,2100
2017-12,3844
2018-01,4200
2018-07,6800
2018-12,5200
2019-01,4500
2019-07,7100
2019-12,6900
2020-01,7500
2020-07,9500
2020-12,15200
2021-01,20500
2021-07,15500
2021-12,25000
2022-01,25500
2022-07,24500
2022-12,22500
2023-01,23000
2023-07,31500
2023-12,38000
2024-01,41500
2024-04,55000
2024-05,91000
2024-07,70000
2024-12,88000
2025-01,91500
2025-07,88000
2025-12,108000
2026-01,102000
"""

target_dict = {}
for line in target_data.strip().split('\n'):
    if line:
        ym, cost = line.split(',')
        target_dict[ym] = float(cost)

print(f"{'Month':<10} | {'Target':<10} | {'Calculated':<12} | {'Diff %':<8} | {'Hash(EH/s)':<12}")
print("-" * 70)

for ym_str, target in target_dict.items():
    ym_period = pd.Period(ym_str, freq='M')
    
    if ym_period in monthly_costs.index:
        calc = monthly_costs.loc[ym_period, 'total_cost_usd']
        hashrate = monthly_costs.loc[ym_period, 'hashrate_eh_per_s']
        diff_pct = ((calc - target) / target) * 100
        
        print(f"{ym_str:<10} | ${target:>8,.0f} | ${calc:>10,.0f} | {diff_pct:>7.1f}% | {hashrate:>10.1f}")
    else:
        print(f"{ym_str:<10} | ${target:>8,.0f} | {'N/A':>10} | {'N/A':>7} | {'N/A':>10}")

print("\n" + "="*70)
print("SUMMARY:")
print(f"Model uses:")
print(f"  - Historical network average efficiency (not best miners)")
print(f"  - Realistic electricity prices with facility overhead")
print(f"  - Overhead factor of 1.50 (electricity is 67% of total cost)")
print(f"  - Actual halvings accounted for")
