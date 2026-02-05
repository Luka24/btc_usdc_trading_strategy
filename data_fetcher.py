"""
Data Fetcher Module: Fetch Historical BTC Data
===============================================
Combines data from:
- CoinGecko API: BTC prices
- Glassnode/Blockchain.com API: Hashrate data
- On-chain data: Mining halving dates
- Caches data locally in data/ folder
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import os
import json
from config import ProductionCostConfig as CostConfig

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARNING] yfinance not installed. Install with: pip install yfinance")


class CoinGeckoFetcher:
    """Fetch BTC historical prices from CoinGecko API or Yahoo Finance"""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    DATA_DIR = "data"
    
    @staticmethod
    def _get_cache_path(days: int, currency: str = "usd") -> str:
        """Get cache file path"""
        return os.path.join(CoinGeckoFetcher.DATA_DIR, f"btc_prices_{days}d_{currency}.csv")
    
    @staticmethod
    def _load_from_cache(days: int) -> Optional[pd.DataFrame]:
        """Load prices from cache if they exist"""
        cache_path = CoinGeckoFetcher._get_cache_path(days)
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"   [CACHE] Loaded prices from {cache_path}")
            return df
        return None
    
    @staticmethod
    def _save_to_cache(df: pd.DataFrame, days: int) -> None:
        """Save prices to cache"""
        os.makedirs(CoinGeckoFetcher.DATA_DIR, exist_ok=True)
        cache_path = CoinGeckoFetcher._get_cache_path(days)
        df.to_csv(cache_path, index=False)
        print(f"   [CACHE] Saved prices to {cache_path}")
    
    @staticmethod
    def fetch_btc_prices_yfinance(days: int = 365) -> pd.DataFrame:
        """
        Fetch BTC prices from Yahoo Finance (no API limits for historical data).
        
        Args:
            days (int): Number of days to fetch
            
        Returns:
            pd.DataFrame: DataFrame with columns [date, btc_price]
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance package is required. Install: pip install yfinance")
        
        print(f"[Yahoo Finance] Fetching {days} days of BTC prices...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            # Fetch BTC-USD data
            btc = yf.Ticker("BTC-USD")
            hist = btc.history(start=start_date.strftime('%Y-%m-%d'), 
                              end=end_date.strftime('%Y-%m-%d'))
            
            if hist.empty:
                raise ValueError("No data returned from Yahoo Finance")
            
            # Convert to required format
            df = pd.DataFrame({
                'date': hist.index.date,
                'btc_price': hist['Close'].values
            })
            
            df = df.reset_index(drop=True)
            
            print(f"   [OK] Fetched {len(df)} days of data")
            print(f"   - From: {df['date'].iloc[0]}")
            print(f"   - To: {df['date'].iloc[-1]}")
            print(f"   - Price (first): ${df['btc_price'].iloc[0]:,.2f}")
            print(f"   - Price (last): ${df['btc_price'].iloc[-1]:,.2f}")
            
            return df
            
        except Exception as e:
            print(f"   [ERROR] Yahoo Finance fetch failed: {e}")
            raise
    
    @staticmethod
    def fetch_btc_prices(days: int = 365, vs_currency: str = "usd") -> pd.DataFrame:
        """
        Fetch historical BTC prices with caching.
        Uses Yahoo Finance for longer periods (>365 days) and CoinGecko for shorter.
        
        Args:
            days (int): Number of days to fetch
            vs_currency (str): Currency (usd, eur, etc.)
            
        Returns:
            pd.DataFrame: DataFrame with columns [date, btc_price]
        """
        # Try cache first
        cached = CoinGeckoFetcher._load_from_cache(days)
        if cached is not None:
            return cached
        
        # For periods > 365 days, prefer Yahoo Finance (no API limits)
        if days > 365 and YFINANCE_AVAILABLE:
            df = CoinGeckoFetcher.fetch_btc_prices_yfinance(days)
            CoinGeckoFetcher._save_to_cache(df, days)
            return df
        
        print(f"[CoinGecko] Fetching {days} days of BTC prices...")
        
        requested_days = days
        days_param = "max" if requested_days > 365 else requested_days
        
        url = f"{CoinGeckoFetcher.BASE_URL}/coins/bitcoin/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days_param,
            "interval": "daily"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp_ms', 'btc_price'])
            
            # Convert timestamp to date
            df['date'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            df['date'] = df['date'].dt.date
            
            df = df[['date', 'btc_price']].reset_index(drop=True)
            
            # If we fetched max history, trim to requested days
            if requested_days > 0 and len(df) > requested_days:
                df = df.tail(requested_days).reset_index(drop=True)
                print(f"   [INFO] Trimmed to last {requested_days} days")
            
            print(f"   [OK] Fetched {len(df)} days of data")
            print(f"   - From: {df['date'].iloc[0]}")
            print(f"   - To: {df['date'].iloc[-1]}")
            print(f"   - Price (first): ${df['btc_price'].iloc[0]:.2f}")
            print(f"   - Price (last): ${df['btc_price'].iloc[-1]:.2f}")
            
            # Save to cache
            CoinGeckoFetcher._save_to_cache(df, days)
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"   [ERROR] CoinGecko request failed: {e}")
            
            # Fallback to Yahoo Finance if available
            if YFINANCE_AVAILABLE:
                print(f"   [FALLBACK] Trying Yahoo Finance...")
                df = CoinGeckoFetcher.fetch_btc_prices_yfinance(days)
                CoinGeckoFetcher._save_to_cache(df, days)
                return df
            raise


class BlockchainFetcher:
    """Fetch hashrate data from free APIs (Blockchain.com, Glassnode)"""
    
    DATA_DIR = "data"
    
    @staticmethod
    def _get_cache_path(days: int) -> str:
        """Get cache file path"""
        return os.path.join(BlockchainFetcher.DATA_DIR, f"hashrate_{days}d.csv")
    
    @staticmethod
    def _load_from_cache(days: int) -> Optional[pd.DataFrame]:
        """Load hashrate from cache if exists"""
        cache_path = BlockchainFetcher._get_cache_path(days)
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"   [CACHE] Loaded hashrate from {cache_path}")
            return df
        return None
    
    @staticmethod
    def _save_to_cache(df: pd.DataFrame, days: int) -> None:
        """Save hashrate to cache"""
        os.makedirs(BlockchainFetcher.DATA_DIR, exist_ok=True)
        cache_path = BlockchainFetcher._get_cache_path(days)
        df.to_csv(cache_path, index=False)
        print(f"   [CACHE] Saved hashrate to {cache_path}")
    
    @staticmethod
    def fetch_hashrate(days: int = 365) -> pd.DataFrame:
        """
        Fetch historical hashrate from Blockchain.com with caching.
        
        Uses JSON endpoint from blockchain.info charts API (free, no auth needed).
        
        Args:
            days (int): Number of days to fetch
            
        Returns:
            pd.DataFrame: DataFrame with columns [date, hashrate_eh_per_s]
        """
        # Try cache first
        cached = BlockchainFetcher._load_from_cache(days)
        if cached is not None:
            return cached
        
        print(f"[Blockchain.com] Fetching {days} days of hashrate data...")
        
        try:
            # Try JSON endpoint for hash rate with timespan
            # timespan: 1year, 2years, 3years, 5years, all
            if days <= 365:
                timespan = "1year"
            elif days <= 730:
                timespan = "2years"
            elif days <= 1095:
                timespan = "3years"
            elif days <= 1825:
                timespan = "5years"
            else:
                timespan = "all"
            
            url = f"https://api.blockchain.info/charts/hash-rate?timespan={timespan}&format=json&cors=true"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract values (each entry is [timestamp, hashrate_h_per_s])
            values = data.get('values', [])
            
            if not values:
                raise ValueError("No data in response")
            
            # Convert to DataFrame
            # Note: API returns TH/s (terahashes per second), not H/s
            df_data = []
            for entry in values:
                timestamp = entry.get('x')  # Unix timestamp
                hashrate_th_per_s = entry.get('y')  # Terahashes per second
                
                if timestamp and hashrate_th_per_s:
                    date = pd.to_datetime(timestamp, unit='s')
                    hashrate_eh_per_s = hashrate_th_per_s / 1e6  # Convert TH/s to EH/s (1 EH = 1M TH)
                    df_data.append({'date': date, 'hashrate_eh_per_s': hashrate_eh_per_s})
            
            df = pd.DataFrame(df_data)
            
            # Keep only last N days
            if len(df) > 0:
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days)
                df = df[df['date'].dt.date >= start_date]
            
            if len(df) > 0:
                print(f"   [OK] Fetched {len(df)} days of hashrate data")
                print(f"   - From: {df['date'].min().date()}")
                print(f"   - To: {df['date'].max().date()}")
                print(f"   - Average: {df['hashrate_eh_per_s'].mean():.1f} EH/s")
                print(f"   - Min: {df['hashrate_eh_per_s'].min():.1f} EH/s")
                print(f"   - Max: {df['hashrate_eh_per_s'].max():.1f} EH/s")
                
                # Save to cache
                BlockchainFetcher._save_to_cache(df, days)
                
                return df.reset_index(drop=True)
            else:
                raise ValueError("No data in range")
            
        except Exception as e:
            print(f"   [ERROR] Could not fetch hashrate from Blockchain.com: {e}")
            print(f"   [ERROR] API data is required - estimation is disabled!")
            print(f"   Possible solutions:")
            print(f"     1. Check internet connection")
            print(f"     2. Try smaller timespan (days parameter)")
            print(f"     3. Wait and retry (API rate limit)")
            print(f"     4. Use alternative API (requires implementation)")
            raise RuntimeError(f"Failed to fetch hashrate data from API. Estimation disabled. Error: {e}")
    
    @staticmethod
    def _estimate_hashrate(start_date: 'datetime.date', days: int) -> pd.DataFrame:
        """
        Estimate hashrate based on historical anchor points and interpolation.
        Uses real historical difficulty data converted to hashrate.
        """
        dates = pd.date_range(start=start_date, periods=days, freq="D")
        
        print(f"   [INFO] Hashrate data - using historical interpolation model...")
        
        # Historical anchor points (derived from real difficulty data)
        # Difficulty to Hashrate: hashrate ≈ difficulty * 2^32 / 600 / 1e18
        historical_points = [
            (datetime(2016, 1, 1), 0.72),   # 103T difficulty
            (datetime(2017, 1, 1), 2.2),    # 317T difficulty
            (datetime(2018, 1, 1), 13.3),   # 1.9P difficulty
            (datetime(2019, 1, 1), 39.2),   # 5.6P difficulty
            (datetime(2020, 1, 1), 96.6),   # 13.8P difficulty
            (datetime(2021, 1, 1), 144.2),  # 20.6P difficulty
            (datetime(2022, 1, 1), 170.1),  # 24.3P difficulty
            (datetime(2023, 1, 1), 247.1),  # 35.3P difficulty
            (datetime(2024, 1, 1), 511.0),  # 73P difficulty
            (datetime(2025, 1, 1), 616.0),  # 88P difficulty
            (datetime(2026, 2, 1), 805.0),  # 115P difficulty
        ]
        
        # Convert to pandas for interpolation
        anchor_dates = [p[0] for p in historical_points]
        anchor_hashrates = [p[1] for p in historical_points]
        
        # Create interpolation function
        from scipy.interpolate import interp1d
        anchor_timestamps = [d.timestamp() for d in anchor_dates]
        interp_func = interp1d(anchor_timestamps, anchor_hashrates, 
                               kind='linear', fill_value='extrapolate')
        
        # Interpolate for all dates
        date_timestamps = [d.timestamp() for d in dates]
        base_hashrates = interp_func(date_timestamps)
        
        # Add realistic daily variations (±5% noise)
        daily_variation = np.random.normal(0, base_hashrates * 0.02, days)
        
        # Add difficulty adjustment cycles (every ~2 weeks)
        cycle_variation = base_hashrates * 0.03 * np.sin(np.linspace(0, days/14 * 2*np.pi, days))
        
        hashrates = base_hashrates + daily_variation + cycle_variation
        hashrates = np.maximum(hashrates, 0.5)  # Minimum realistic hashrate
        
        df = pd.DataFrame({
            'date': [d.date() for d in dates],
            'hashrate_eh_per_s': hashrates,
        })
        
        print(f"   [OK] Estimated {len(df)} days of hashrate data")
        print(f"   - Average: {df['hashrate_eh_per_s'].mean():.1f} EH/s")
        print(f"   - Min: {df['hashrate_eh_per_s'].min():.1f} EH/s")
        print(f"   - Max: {df['hashrate_eh_per_s'].max():.1f} EH/s")
        print(f"   - StdDev: {df['hashrate_eh_per_s'].std():.1f} EH/s")
        
        return df


class DataFetcher:
    """Main class for fetching combined data with caching"""
    
    DATA_DIR = "data"
    
    @staticmethod
    def _get_cache_path(days: int) -> str:
        """Get cache file path for combined data"""
        return os.path.join(DataFetcher.DATA_DIR, f"combined_data_{days}d.csv")
    
    @staticmethod
    def _load_from_cache(days: int) -> Optional[pd.DataFrame]:
        """Load combined data from cache if exists"""
        cache_path = DataFetcher._get_cache_path(days)
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"   [CACHE] Loaded combined data from {cache_path}")
            return df
        return None
    
    @staticmethod
    def _save_to_cache(df: pd.DataFrame, days: int) -> None:
        """Save combined data to cache"""
        os.makedirs(DataFetcher.DATA_DIR, exist_ok=True)
        cache_path = DataFetcher._get_cache_path(days)
        df.to_csv(cache_path, index=False)
        print(f"   [CACHE] Saved combined data to {cache_path}")
    
    @staticmethod
    def get_current_block_reward() -> float:
        """
        Get current BTC block reward based on halving history.
        Halvings: 2012-11-28 (50->25), 2016-07-09 (25->12.5), 
                  2020-05-11 (12.5->6.25), 2024-04-20 (6.25->3.125)
        """
        halving_dates = [
            (datetime.strptime(date_str, '%Y-%m-%d'), reward)
            for date_str, reward in CostConfig.HALVING_SCHEDULE
        ]
        
        current = datetime.now()
        for halving_date, reward in halving_dates:
            if current < halving_date:
                # Return reward from the previous halving
                idx = halving_dates.index((halving_date, reward))
                if idx == 0:
                    return CostConfig.PRE_HALVING_REWARD  # Before first halving
                else:
                    return halving_dates[idx - 1][1]
        
        # Current date is after all known halvings, return latest
        return halving_dates[-1][1]
    
    @staticmethod
    def fetch_combined_data(days: int = 365, 
                           use_real_data: bool = True,
                           force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch combined data (prices + hashrate) with caching support.
        
        Args:
            days (int): Number of days
            use_real_data (bool): Whether to use real API data
            force_refresh (bool): If True, bypass cache and fetch fresh data
            
        Returns:
            pd.DataFrame: Combined data
        """
        
        print("\n" + "="*70)
        print("DATA FETCHING")
        print("="*70)
        
        if not use_real_data:
            print("\n[MODE] Synthetic data (for testing)")
            return DataFetcher._generate_synthetic_data(days)
        
        print("\n[MODE] Real historical data\n")
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_data = DataFetcher._load_from_cache(days)
            if cached_data is not None:
                print(f"   [OK] {len(cached_data)} days of combined data (from cache)")
                print(f"   - Dates: {cached_data['date'].iloc[0]} to {cached_data['date'].iloc[-1]}")
                return cached_data
        else:
            print("   [REFRESH] Bypassing cache, fetching fresh data...")
        
        # Fetch BTC prices
        df_prices = CoinGeckoFetcher.fetch_btc_prices(days)
        
        print()
        
        # Fetch hashrate
        df_hashrate = BlockchainFetcher.fetch_hashrate(days)
        
        # Combine data
        print("\n[MERGE] Combining data...")
        df_prices['date'] = pd.to_datetime(df_prices['date'])
        df_hashrate['date'] = pd.to_datetime(df_hashrate['date'])
        
        # Use left merge to keep all price dates
        df_combined = pd.merge(df_prices, df_hashrate, on='date', how='left')
        
        # Interpolate hashrate values for missing dates (linear interpolation)
        df_combined['hashrate_eh_per_s'] = df_combined['hashrate_eh_per_s'].interpolate(method='linear')
        
        # If first/last values are still NaN, forward/backfill (pandas 2.0+ syntax)
        df_combined['hashrate_eh_per_s'] = df_combined['hashrate_eh_per_s'].ffill()
        df_combined['hashrate_eh_per_s'] = df_combined['hashrate_eh_per_s'].bfill()
        
        # Drop any remaining rows with NaN (shouldn't happen, but just in case)
        df_combined = df_combined.dropna()
        
        # Format as required by backtest engine
        df_combined['date'] = df_combined['date'].dt.strftime('%Y-%m-%d')
        df_combined = df_combined[['date', 'btc_price', 'hashrate_eh_per_s']]
        
        print(f"   [OK] {len(df_combined)} days of combined data")
        print(f"   - Dates: {df_combined['date'].iloc[0]} to {df_combined['date'].iloc[-1]}")
        print(f"   [INFO] Hashrate values interpolated linearly for missing dates")
        
        # Save to cache
        DataFetcher._save_to_cache(df_combined, days)
        
        return df_combined
    
    @staticmethod
    def _generate_synthetic_data(days: int = 365) -> pd.DataFrame:
        """
        Generate synthetic data (fallback for API issues).
        """
        dates = pd.date_range(start="2023-01-01", periods=days, freq="D")
        
        # BTC prices
        base_price = 42000
        cycle1 = np.sin(np.linspace(0, 4*np.pi, days)) * 5000
        trend = np.linspace(0, 8000, days)
        volatility = np.random.normal(0, 2500, days)
        btc_prices = base_price + cycle1 + trend + volatility
        btc_prices = np.maximum(btc_prices, 10000)
        
        # Hashrate
        base_hashrate = 400
        hashrate_cycle = np.sin(np.linspace(0, 4*np.pi, days)) * 40
        hashrate_trend = np.linspace(0, 100, days)
        hashrate_noise = np.random.normal(0, 10, days)
        hashrates = base_hashrate + hashrate_cycle + hashrate_trend + hashrate_noise
        hashrates = np.maximum(hashrates, 100)
        
        df = pd.DataFrame({
            'date': [d.strftime('%Y-%m-%d') for d in dates],
            'btc_price': btc_prices,
            'hashrate_eh_per_s': hashrates,
        })
        
        print(f"   [OK] Generated {len(df)} days of synthetic data")
        
        return df


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    # Fetch real data
    data = DataFetcher.fetch_combined_data(days=30, use_real_data=True)
    
    print("\n" + "="*70)
    print("DATA (FIRST 5 ROWS)")
    print("="*70)
    print(data.head().to_string())
    
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"BTC Price - Min: ${data['btc_price'].min():.2f}, "
          f"Max: ${data['btc_price'].max():.2f}, "
          f"Average: ${data['btc_price'].mean():.2f}")
    print(f"Hashrate - Min: {data['hashrate_eh_per_s'].min():.1f} EH/s, "
          f"Max: {data['hashrate_eh_per_s'].max():.1f} EH/s, "
          f"Average: {data['hashrate_eh_per_s'].mean():.1f} EH/s")
