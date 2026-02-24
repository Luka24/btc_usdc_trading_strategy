from production_cost import ProductionCostSeries
import pandas as pd

# Test with real data
cost_series = ProductionCostSeries()

# Add some daily data
dates_prices = [
    ('2025-11-04', 106521),
    ('2025-11-05', 101635),
    ('2025-11-06', 103877),
    ('2025-11-07', 101322),
]

hashrate = 520  # EH/s

print("Raw costs:")
for date, price in dates_prices:
    cost_data = cost_series.add_daily_data(date, hashrate)
    print(f"  {date}: Raw cost = ${cost_data['total_cost']:.2f}, Price = ${price}")

print("\nSmoothed costs:")
smoothed = cost_series.smooth_with_ema()
for i, (date, price) in enumerate(dates_prices):
    if i < len(smoothed):
        ema_cost = smoothed.iloc[i]
        ratio = price / ema_cost
        print(f"  {date}: EMA cost = ${ema_cost:.2f}, Ratio = {ratio:.2f}")
