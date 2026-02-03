"""
Test script to compare calculated costs with target estimates
"""
import pandas as pd
import sys
sys.path.append('c:\\Users\\lukap\\Documents\\IOCNCOMI naloga\\btc_trading_strategy')

from production_cost import BTCProductionCostCalculator
from config import HistoricalParameters

# Target cost estimates (monthly averages)
target_costs = """
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

# Sample hashrate data (approximate historical values in EH/s)
hashrate_data = {
    "2016-01": 1.0,
    "2016-07": 1.5,
    "2017-01": 2.5,
    "2017-07": 5.0,
    "2017-12": 12.0,
    "2018-01": 19.0,
    "2018-07": 40.0,
    "2018-12": 40.0,
    "2019-01": 42.0,
    "2019-07": 70.0,
    "2019-12": 90.0,
    "2020-01": 110.0,
    "2020-07": 120.0,
    "2020-12": 150.0,
    "2021-01": 150.0,
    "2021-07": 110.0,
    "2021-12": 180.0,
    "2022-01": 200.0,
    "2022-07": 220.0,
    "2022-12": 250.0,
    "2023-01": 270.0,
    "2023-07": 380.0,
    "2023-12": 500.0,
    "2024-01": 550.0,
    "2024-04": 600.0,
    "2024-05": 650.0,
    "2024-07": 600.0,
    "2024-12": 750.0,
    "2025-01": 780.0,
    "2025-07": 700.0,
    "2025-12": 850.0,
    "2026-01": 900.0,
}

print(f"{'Date':<12} | {'Target':<12} | {'Calculated':<12} | {'Diff %':<10} | {'Elec':<6} | {'Eff':<6}")
print("-" * 75)

for line in target_costs.strip().split('\n'):
    if not line:
        continue
    date_str, target_str = line.split(',')
    target = float(target_str)
    
    # Get approximate hashrate
    hashrate = hashrate_data.get(date_str, 100.0)
    
    # Calculate with our model (append -01 for full date format)
    full_date = f"{date_str}-01"
    calc = BTCProductionCostCalculator(date=full_date)
    calculated = calc.calculate_total_cost_per_btc(hashrate)
    
    # Difference
    diff_pct = ((calculated - target) / target) * 100
    
    print(f"{date_str:<12} | ${target:>10,.0f} | ${calculated:>10,.0f} | {diff_pct:>8.1f}% | {calc.energy_price:.3f} | {calc.efficiency:>5.0f}")

print("\nNote: Differences mainly due to hashrate approximations")
print("The model parameters (efficiency & electricity) are now more realistic")
