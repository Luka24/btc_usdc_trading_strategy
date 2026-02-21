import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    ProductionCostConfig, SignalConfig, PortfolioConfig,
    RiskManagementConfig, BacktestConfig
)
from production_cost import BTCProductionCostCalculator, ProductionCostSeries
from strategy import TradingStrategy
from data_fetcher import DataFetcher





def main():
    print("\n" + "="*70)
    print("BTC/USDC ADAPTIVE ALLOCATION STRATEGY - PROFESSIONAL EDITION")
    print("="*70)
    
    print("\n[1] Fetching data...")
    data = DataFetcher.fetch_combined_data(
        days=BacktestConfig.DAYS_TO_FETCH, 
        use_real_data=BacktestConfig.USE_REAL_DATA
    )
    print(f"    ✓ {len(data)} days loaded")
    
    print("\n[2] Initializing strategy...")
    strategy = TradingStrategy(initial_capital=100_000)
    print(f"    ✓ Strategy initialized (Professional rules v3.0)")
    print(f"      - Signal smoothing: Trend(SMA3), Momentum(EMA0.4), Cost(SMA7)")
    print(f"      - Multi-layer confirmation: Zone/Magnitude/Directional/Extreme")
    print(f"      - Staged execution: 2-day split (60/40)")
    
    print("\n[3] Running daily trading cycles...")
    
    # Prepare price histories
    prices = data['btc_price'].values
    costs = data['production_cost'].values
    
    cycle_count = 0
    for idx in range(200, len(data)):  # Need 200 days for SMA
        date = data.iloc[idx]['date']
        btc_price = data.iloc[idx]['btc_price']
        production_cost = data.iloc[idx]['production_cost']
        
        # Get recent data for calculations
        prices_200 = prices[:idx+1]
        prices_90 = prices[max(0, idx-90):idx+1]
        returns_30 = np.diff(prices[max(0, idx-30):idx+1]) / prices[max(0, idx-30):idx]
        
        # Run strategy cycle
        result = strategy.run_daily_cycle(
            date=date,
            btc_price=btc_price,
            production_cost=production_cost,
            prices_last_200=prices_200,
            prices_last_90=prices_90,
            daily_returns_30=returns_30
        )
        
        cycle_count += 1
        if cycle_count % 500 == 0:
            print(f"    ✓ Processed {cycle_count} cycles... (Current: {date})")
    
    print(f"    ✓ Total cycles: {cycle_count}")
    
    print("\n[4] Generating summary report...")
    summary = strategy.get_summary()
    print(summary)
    
    print("\n[5] Exporting execution log...")
    log_df = strategy.get_execution_log_df()
    os.makedirs('results', exist_ok=True)
    
    # Save execution log
    log_filename = f"results/strategy_execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_df.to_csv(log_filename, index=False)
    print(f"    ✓ Saved: {log_filename}")
    
    # Save summary
    summary_filename = f"results/strategy_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"    ✓ Saved: {summary_filename}")
    
    print("\n[6] Generating analysis plots...")
    _plot_strategy_analysis(log_df, data)
    
    print("\n" + "="*70)
    print("COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nCheck results/ folder for outputs:")
    print(f"  - Execution log CSV")
    print(f"  - Summary report TXT")
    print(f"  - Analysis plots PNG")
    print("\n")


def _plot_strategy_analysis(log_df: pd.DataFrame, data_df: pd.DataFrame):
    """Generate analysis plots."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # ===== Plot 1: Price vs Targets =====
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(log_df['date'], log_df['btc_price'], 'b-', linewidth=2, label='BTC Price', alpha=0.8)
    ax1.plot(log_df['date'], log_df['production_cost'], color='gray', linewidth=1, alpha=0.5, label='Production Cost')
    
    # Color code by target allocation
    colors_map = {1.0: 'darkgreen', 0.7: 'green', 0.4: 'yellow', 0.2: 'orange', 0.0: 'red'}
    for target, color in colors_map.items():
        mask = log_df['target_btc'] == target
        ax1.scatter(log_df[mask]['date'], log_df[mask]['btc_price'], 
                   c=color, s=20, alpha=0.6, label=f'Target {target:.0%}')
    
    ax1.set_ylabel('USD', fontweight='bold')
    ax1.set_title('BTC Price vs Production Cost & Target Allocation', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: Ensemble Score Evolution =====
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(log_df['date'], log_df['score_raw'], 'gray', alpha=0.3, linewidth=1, label='Raw')
    ax2.plot(log_df['date'], log_df['score_smooth'], 'orange', alpha=0.6, linewidth=1.5, label='Smooth')
    ax2.plot(log_df['date'], log_df['score_adjusted'], 'darkblue', linewidth=2, label='Adjusted')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.3)
    ax2.axhline(y=-2.0, color='red', linestyle='--', alpha=0.3)
    ax2.fill_between(log_df['date'], -2.0, 2.0, alpha=0.05, color='gray')
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Ensemble Score Evolution', fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ===== Plot 3: Position Evolution =====
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.fill_between(log_df['date'], 0, log_df['current_position_btc'], 
                    label='BTC %', alpha=0.7, color='orange')
    ax3.fill_between(log_df['date'], log_df['current_position_btc'], 1, 
                    label='USDC %', alpha=0.7, color='steelblue')
    ax3.plot(log_df['date'], log_df['target_btc'], 'darkred', linewidth=1.5, 
            linestyle='--', label='Target', alpha=0.8)
    ax3.set_ylabel('Allocation', fontweight='bold')
    ax3.set_title('Position Evolution', fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ===== Plot 4: Volatility Regime =====
    ax4 = fig.add_subplot(gs[2, 0])
    colors = {'LOW': 'green', 'NORMAL': 'blue', 'HIGH': 'red'}
    for regime in ['LOW', 'NORMAL', 'HIGH']:
        mask = log_df['vol_regime'] == regime
        ax4.scatter(log_df[mask]['date'], log_df[mask]['annual_vol'], 
                   c=colors[regime], label=regime, alpha=0.6, s=20)
    ax4.axhline(y=0.50, color='green', linestyle='--', alpha=0.3)
    ax4.axhline(y=0.80, color='red', linestyle='--', alpha=0.3)
    ax4.set_ylabel('Annual Vol', fontweight='bold')
    ax4.set_title('Volatility Regime', fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # ===== Plot 5: Confirmation Success Rate =====
    ax5 = fig.add_subplot(gs[2, 1])
    confirmed_sum = log_df['confirmed'].sum()
    total = len(log_df)
    not_confirmed = total - confirmed_sum
    
    ax5.bar(['Confirmed', 'Not Confirmed'], [confirmed_sum, not_confirmed], 
           color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Count', fontweight='bold')
    ax5.set_title(f'Confirmation Results (Total: {total})', fontweight='bold')
    
    # Add percentage labels
    for i, v in enumerate([confirmed_sum, not_confirmed]):
        pct = (v / total) * 100
        ax5.text(i, v + 10, f'{v}\n({pct:.1f}%)', ha='center', fontweight='bold')
    
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = f'results/strategy_analysis_{ts}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    ✓ Saved: {fn}")


if __name__ == "__main__":
    main()
