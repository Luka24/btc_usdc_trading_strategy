#!/usr/bin/env python3
"""Test dynamic halving detection"""

from data_fetcher import DataFetcher
from datetime import datetime

# Test halving detection
print('='*70)
print('HALVING DETECTION')
print('='*70)
print()

current_reward = DataFetcher.get_current_block_reward()
print(f'Current block reward (today): {current_reward} BTC')
print()

halving_dates = [
    (datetime(2012, 11, 28), 50.0),
    (datetime(2016, 7, 9), 25.0),
    (datetime(2020, 5, 11), 12.5),
    (datetime(2024, 4, 20), 6.25),
    (datetime(2028, 4, 20), 3.125),
]

print('Halving history:')
for halving_date, reward in halving_dates:
    date_str = halving_date.strftime('%Y-%m-%d')
    print(f'  {date_str}: {reward} BTC')
