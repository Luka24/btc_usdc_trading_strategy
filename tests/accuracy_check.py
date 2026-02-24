"""
Quick accuracy check for key periods
"""
import pandas as pd
import sys
sys.path.append('c:\\Users\\lukap\\Documents\\IOCNCOMI naloga\\btc_trading_strategy')

from production_cost import ProductionCostSeries

# Load real hashrate data
hashrate_df = pd.read_csv('c:/Users/lukap/Documents/IOCNCOMI naloga/data/hashrate_3650d.csv')
hashrate_df['date'] = pd.to_datetime(hashrate_df['date'])

# Calculate costs
cost_series = ProductionCostSeries(ema_window=30)
cost_series.add_from_dataframe(hashrate_df)

# Get monthly averages
cost_series.data.reset_index(inplace=True)
cost_series.data['year_month'] = cost_series.data['date'].dt.to_period('M')
monthly_costs = cost_series.data.groupby('year_month').agg({
    'total_cost_usd': 'mean',
}).round(0)

# Key test points
tests = [
    ("2017-07", 2100),
    ("2017-12", 3844),
    ("2018-07", 6800),
    ("2019-01", 4500),
    ("2020-01", 7500),
    ("2021-01", 20500),
    ("2022-07", 24500),
    ("2023-07", 31500),
    ("2024-01", 41500),
    ("2024-04", 55000),
    ("2024-05", 91000),
    ("2025-01", 91500),
    ("2026-01", 102000),
]

print("Key Period Accuracy Check")
print("=" * 60)
errors = []
for ym_str, target in tests:
    ym_period = pd.Period(ym_str, freq='M')
    if ym_period in monthly_costs.index:
        calc = monthly_costs.loc[ym_period, 'total_cost_usd']
        diff_pct = ((calc - target) / target) * 100
        errors.append(abs(diff_pct))
        status = "✓" if abs(diff_pct) < 15 else "•"
        print(f"{status} {ym_str}: Target ${target:>6,.0f} | Calc ${calc:>6,.0f} | {diff_pct:>6.1f}%")

print("=" * 60)
print(f"Average absolute error: {sum(errors)/len(errors):.1f}%")
print(f"Errors < 10%: {sum(1 for e in errors if e < 10)}/{len(errors)}")
print(f"Errors < 15%: {sum(1 for e in errors if e < 15)}/{len(errors)}")
print(f"Errors < 20%: {sum(1 for e in errors if e < 20)}/{len(errors)}")
