import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional
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
    BASE_URL = "https://api.coingecko.com/api/v3"
    DATA_DIR = "data"
    
    @staticmethod
    def _get_cache_path(days: int, currency: str = "usd") -> str:
        return os.path.join(CoinGeckoFetcher.DATA_DIR, f"btc_prices_{days}d_{currency}.csv")
    
    @staticmethod
    def _load_from_cache(days: int) -> Optional[pd.DataFrame]:
        cache_path = CoinGeckoFetcher._get_cache_path(days)
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"   [CACHE] Loaded prices from {cache_path}")
            return df
        return None
    
    @staticmethod
    def _save_to_cache(df: pd.DataFrame, days: int) -> None:
        os.makedirs(CoinGeckoFetcher.DATA_DIR, exist_ok=True)
        cache_path = CoinGeckoFetcher._get_cache_path(days)
        df.to_csv(cache_path, index=False)
        print(f"   [CACHE] Saved prices to {cache_path}")
    
    @staticmethod
    def fetch_btc_prices_yfinance(days: int = 365) -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance package is required. Install: pip install yfinance")
        
        print(f"[Yahoo Finance] Fetching {days} days of BTC prices...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            btc = yf.Ticker("BTC-USD")
            hist = btc.history(start=start_date.strftime('%Y-%m-%d'), 
                              end=end_date.strftime('%Y-%m-%d'))
            
            if hist.empty:
                raise ValueError("No data returned from Yahoo Finance")
            
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
    DATA_DIR = "data"
    
    @staticmethod
    def _get_cache_path(days: int) -> str:
        return os.path.join(BlockchainFetcher.DATA_DIR, f"hashrate_{days}d.csv")
    
    @staticmethod
    def _load_from_cache(days: int) -> Optional[pd.DataFrame]:
        cache_path = BlockchainFetcher._get_cache_path(days)
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"   [CACHE] Loaded hashrate from {cache_path}")
            return df
        return None
    
    @staticmethod
    def _save_to_cache(df: pd.DataFrame, days: int) -> None:
        os.makedirs(BlockchainFetcher.DATA_DIR, exist_ok=True)
        cache_path = BlockchainFetcher._get_cache_path(days)
        df.to_csv(cache_path, index=False)
        print(f"   [CACHE] Saved hashrate to {cache_path}")
    
    @staticmethod
    def fetch_hashrate(days: int = 365, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch historical hashrate from blockchain.info charts API (free, no auth)."""
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
            
            # API returns TH/s (terahashes per second)
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


class DataFetcher:
    DATA_DIR = "data"
    
    @staticmethod
    def _get_cache_path(days: int) -> str:
        return os.path.join(DataFetcher.DATA_DIR, f"combined_data_{days}d.csv")
    
    @staticmethod
    def _load_from_cache(days: int) -> Optional[pd.DataFrame]:
        cache_path = DataFetcher._get_cache_path(days)
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"   [CACHE] Loaded combined data from {cache_path}")
            return df
        return None
    
    @staticmethod
    def _save_to_cache(df: pd.DataFrame, days: int) -> None:
        os.makedirs(DataFetcher.DATA_DIR, exist_ok=True)
        cache_path = DataFetcher._get_cache_path(days)
        df.to_csv(cache_path, index=False)
        print(f"   [CACHE] Saved combined data to {cache_path}")
    
    @staticmethod
    def fetch_combined_data(days: int = 365, 
                           use_real_data: bool = True,
                           force_refresh: bool = False) -> pd.DataFrame:
        print("\n" + "="*70)
        print("DATA FETCHING")
        print("="*70)
        
        if not use_real_data:
            raise RuntimeError("Synthetic data mode is not supported in this strategy version.")
        
        print("\n[MODE] Real historical data\n")
        
        if not force_refresh:
            cached_data = DataFetcher._load_from_cache(days)
            if cached_data is not None:
                print(f"   [OK] {len(cached_data)} days of combined data (from cache)")
                print(f"   - Dates: {cached_data['date'].iloc[0]} to {cached_data['date'].iloc[-1]}")
                return cached_data
        else:
            print("   [REFRESH] Bypassing cache, fetching fresh data...")
        
        df_prices = CoinGeckoFetcher.fetch_btc_prices(days, force_refresh=force_refresh)
        
        print()
        
        df_hashrate = BlockchainFetcher.fetch_hashrate(days, force_refresh=force_refresh)
        
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
        
        df_combined = df_combined.dropna()

        df_combined['date'] = pd.to_datetime(df_combined['date']).dt.strftime('%Y-%m-%d')
        df_combined = df_combined[['date', 'btc_price', 'hashrate_eh_per_s']]

        print(f"   [OK] {len(df_combined)} days of combined data")
        print(f"   - Dates: {df_combined['date'].iloc[0]} to {df_combined['date'].iloc[-1]}")
        print(f"   [INFO] Hashrate values interpolated linearly for missing dates")

        # Save to cache
        DataFetcher._save_to_cache(df_combined, days)

        return df_combined
    



if __name__ == "__main__":
    data = DataFetcher.fetch_combined_data(days=30, use_real_data=True)
    print(data.head().to_string())
    print(f"BTC Price  min/avg/max: ${data['btc_price'].min():.2f} / ${data['btc_price'].mean():.2f} / ${data['btc_price'].max():.2f}")
    print(f"Hashrate   min/avg/max: {data['hashrate_eh_per_s'].min():.1f} / {data['hashrate_eh_per_s'].mean():.1f} / {data['hashrate_eh_per_s'].max():.1f} EH/s")
