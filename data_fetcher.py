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
    def fetch_btc_prices(days: int = 365, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch historical BTC prices with caching. Uses Yahoo Finance for >365 days, CoinGecko otherwise."""
        # Try cache first unless force refresh
        if not force_refresh:
            cached = CoinGeckoFetcher._load_from_cache(days)
            if cached is not None:
                return cached
        
        # For periods > 365 days, prefer Yahoo Finance (no API limits)
        if days > 365 and YFINANCE_AVAILABLE:
            df = CoinGeckoFetcher.fetch_btc_prices_yfinance(days)
            CoinGeckoFetcher._save_to_cache(df, days)
            return df
        
        print(f"[CoinGecko] Fetching {days} days of BTC prices...")
        
        days_param = "max" if days > 365 else days
        
        url = f"{CoinGeckoFetcher.BASE_URL}/coins/bitcoin/market_chart"
        params = {
            "vs_currency": "usd",
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
            
            # Trim to requested days if CoinGecko returned more than asked
            if len(df) > days:
                df = df.tail(days).reset_index(drop=True)
                print(f"   [INFO] Trimmed to last {days} days")
            
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
    def fetch_hashrate(days: int = 365, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical hashrate from Blockchain.com with caching.
        
        Uses JSON endpoint from blockchain.info charts API (free, no auth needed).
        
        Args:
            days (int): Number of days to fetch
            
        Returns:
            pd.DataFrame: DataFrame with columns [date, hashrate_eh_per_s]
        """
        # Try cache first unless force refresh
        if not force_refresh:
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


class MVRVFetcher:
    """
    Fetch BTC MVRV ratio from CoinMetrics Community API and compute Z-score.
    MVRV = Market Cap / Realized Cap  (free, no API key required)
    MVRV Z-score = rolling Z-normalization → identifies cycle tops & bottoms.
    """

    BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    DATA_DIR = "data"
    CACHE_FILE = "mvrv_data.csv"

    @staticmethod
    def _get_cache_path() -> str:
        return os.path.join(MVRVFetcher.DATA_DIR, MVRVFetcher.CACHE_FILE)

    @staticmethod
    def _load_from_cache(min_end_date: 'pd.Timestamp') -> 'Optional[pd.DataFrame]':
        path = MVRVFetcher._get_cache_path()
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            # Invalidate if cache is stale (last row older than yesterday)
            if df['date'].max() >= min_end_date - pd.Timedelta(days=2):
                print(f"   [CACHE] Loaded MVRV data from {path} ({len(df)} rows, "
                      f"{df['date'].min().date()} – {df['date'].max().date()})")
                return df
            print(f"   [CACHE] MVRV cache is stale – refetching...")
        return None

    @staticmethod
    def _save_to_cache(df: pd.DataFrame) -> None:
        os.makedirs(MVRVFetcher.DATA_DIR, exist_ok=True)
        path = MVRVFetcher._get_cache_path()
        df.to_csv(path, index=False)
        print(f"   [CACHE] Saved MVRV data to {path}")

    @staticmethod
    def fetch_mvrv_z(days: int, z_window: int = 730, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch MVRV ratio from CoinMetrics and return Z-score column.

        Args:
            days:         Number of strategy days needed
            z_window:     Rolling window (days) for Z-score computation
            force_refresh: Bypass cache

        Returns:
            DataFrame with columns [date, mvrv, mvrv_z]
        """
        print(f"[CoinMetrics] Fetching MVRV Z-score ({days}d strategy + {z_window}d Z-warmup)...")

        end_dt   = pd.Timestamp.now().normalize()
        start_dt_cache_check = end_dt - pd.Timedelta(days=2)

        if not force_refresh:
            cached = MVRVFetcher._load_from_cache(start_dt_cache_check)
            if cached is not None:
                return cached

        # Fetch enough history for the Z-score rolling window + strategy period
        total_days = days + z_window + 60  # extra buffer
        start_date = (datetime.now() - timedelta(days=total_days)).strftime('%Y-%m-%d')
        end_date   = datetime.now().strftime('%Y-%m-%d')

        params = {
            'assets':      'btc',
            'metrics':     'CapMVRVCur',
            'frequency':   '1d',
            'start_time':  start_date,
            'end_time':    end_date,
            'page_size':   10000,
        }

        try:
            r = requests.get(MVRVFetcher.BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            raw = r.json().get('data', [])

            rows = []
            for item in raw:
                # CoinMetrics returns ISO-8601 UTC strings; convert to tz-naive
                date_ts = pd.to_datetime(item['time']).tz_convert(None).normalize()
                mvrv_val = float(item['CapMVRVCur'])
                rows.append({'date': date_ts, 'mvrv': mvrv_val})

            df = pd.DataFrame(rows)
            df = (
                df.sort_values('date')
                  .drop_duplicates(subset=['date'], keep='last')
                  .reset_index(drop=True)
            )

            # Rolling Z-score (min 60 days for meaningful normalisation)
            roll_mean = df['mvrv'].rolling(window=z_window, min_periods=60).mean()
            roll_std  = df['mvrv'].rolling(window=z_window, min_periods=60).std(ddof=0)
            df['mvrv_z'] = (df['mvrv'] - roll_mean) / roll_std.replace(0, np.nan)
            df['mvrv_z'] = df['mvrv_z'].fillna(0.0)  # neutral during short warmup

            print(f"   [OK] {len(df)} days of MVRV data fetched")
            print(f"   - MVRV   : {df['mvrv'].min():.2f} – {df['mvrv'].max():.2f}")
            print(f"   - MVRV Z : {df['mvrv_z'].min():.2f} – {df['mvrv_z'].max():.2f}")

            MVRVFetcher._save_to_cache(df)
            return df

        except Exception as e:
            print(f"   [WARN] Could not fetch MVRV from CoinMetrics: {e}")
            print(f"   [WARN] Continuing with MVRV disabled (neutral weight everywhere)")
            return pd.DataFrame(columns=['date', 'mvrv', 'mvrv_z'])


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
            raise RuntimeError("Synthetic data mode is not supported in this strategy version.")
        
        print("\n[MODE] Real historical data\n")
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_data = DataFetcher._load_from_cache(days)
            if cached_data is not None:
                # Invalidate combined cache if MVRV column missing (schema upgrade)
                if 'mvrv_z' not in cached_data.columns:
                    print("   [CACHE] Combined cache missing mvrv_z – rebuilding...")
                else:
                    print(f"   [OK] {len(cached_data)} days of combined data (from cache)")
                    print(f"   - Dates: {cached_data['date'].iloc[0]} to {cached_data['date'].iloc[-1]}")
                    return cached_data
        else:
            print("   [REFRESH] Bypassing cache, fetching fresh data...")
        
        # Fetch BTC prices
        df_prices = CoinGeckoFetcher.fetch_btc_prices(days, force_refresh=force_refresh)
        
        print()
        
        # Fetch hashrate
        df_hashrate = BlockchainFetcher.fetch_hashrate(days, force_refresh=force_refresh)
        
        # Combine data
        print("\n[MERGE] Combining data...")
        df_prices['date'] = pd.to_datetime(df_prices['date'])
        df_hashrate['date'] = pd.to_datetime(df_hashrate['date'])

        # Normalize duplicates before merge (keep latest record per date)
        df_prices = df_prices.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        df_hashrate = df_hashrate.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        
        # Use left merge to keep all price dates
        df_combined = pd.merge(df_prices, df_hashrate, on='date', how='left')
        
        # Interpolate hashrate values for missing dates (linear interpolation)
        df_combined['hashrate_eh_per_s'] = df_combined['hashrate_eh_per_s'].interpolate(method='linear')
        
        # If first/last values are still NaN, forward/backfill (pandas 2.0+ syntax)
        df_combined['hashrate_eh_per_s'] = df_combined['hashrate_eh_per_s'].ffill()
        df_combined['hashrate_eh_per_s'] = df_combined['hashrate_eh_per_s'].bfill()
        
        # Drop any remaining rows with NaN (shouldn't happen, but just in case)
        df_combined = df_combined.dropna()
        
        # ── MVRV Z-score (on-chain valuation cycle signal) ──────────────────
        from config import PortfolioConfig as _PC
        if getattr(_PC, 'MVRV_ENABLED', False):
            print("\n[MVRV] Fetching MVRV Z-score...")
            df_mvrv = MVRVFetcher.fetch_mvrv_z(
                days=days,
                z_window=getattr(_PC, 'MVRV_Z_WINDOW', 730),
                force_refresh=force_refresh,
            )
            if not df_mvrv.empty:
                df_mvrv['date'] = pd.to_datetime(df_mvrv['date']).dt.normalize()
                df_mvrv = df_mvrv[['date', 'mvrv_z']]
                df_combined['date'] = pd.to_datetime(df_combined['date']).dt.normalize()
                df_combined = pd.merge(df_combined, df_mvrv, on='date', how='left')
                df_combined['mvrv_z'] = df_combined['mvrv_z'].fillna(0.0)
                print(f"   [OK] MVRV Z merged ({df_combined['mvrv_z'].notna().sum()} non-null rows)")
            else:
                df_combined['mvrv_z'] = 0.0
                print("   [WARN] MVRV data unavailable – using neutral (0.0)")
        else:
            df_combined['mvrv_z'] = 0.0
        # ─────────────────────────────────────────────────────────────────────

        # Format as required by backtest engine
        df_combined['date'] = pd.to_datetime(df_combined['date']).dt.strftime('%Y-%m-%d')
        df_combined = df_combined[['date', 'btc_price', 'hashrate_eh_per_s', 'mvrv_z']]

        print(f"   [OK] {len(df_combined)} days of combined data")
        print(f"   - Dates: {df_combined['date'].iloc[0]} to {df_combined['date'].iloc[-1]}")
        print(f"   [INFO] Hashrate values interpolated linearly for missing dates")

        # Save to cache
        DataFetcher._save_to_cache(df_combined, days)

        return df_combined
    



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
