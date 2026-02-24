#!/usr/bin/env python3
"""Test combined data fetching and caching"""

from data_fetcher import DataFetcher
import os

print("\n" + "="*70)
print("TEST 1: Fetch 30 days of data (should fetch from APIs)")
print("="*70)

# First call - should fetch from APIs
data1 = DataFetcher.fetch_combined_data(days=30, use_real_data=True)

print("\n" + "="*70)
print("TEST 2: Fetch same 30 days again (should load from cache)")
print("="*70)

# Second call - should load from cache
data2 = DataFetcher.fetch_combined_data(days=30, use_real_data=True)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Data rows: {len(data1)}")
print(f"Date range: {data1['date'].iloc[0]} to {data1['date'].iloc[-1]}")
print(f"BTC price range: ${data1['btc_price'].min():.2f} - ${data1['btc_price'].max():.2f}")
print(f"Hashrate range: {data1['hashrate_eh_per_s'].min():.1f} - {data1['hashrate_eh_per_s'].max():.1f} EH/s")

# Check cache file
cache_path = os.path.join(DataFetcher.DATA_DIR, f"combined_data_30d.csv")
if os.path.exists(cache_path):
    file_size = os.path.getsize(cache_path)
    print(f"\nCache file size: {file_size:,} bytes")
