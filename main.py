import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import BacktestConfig
from backtest import BacktestEngine
from risk_manager import RiskManager
from data_fetcher import DataFetcher


def gen_report(engine, rm):
    m = engine.calculate_metrics()
    pdf = engine.portfolio_manager.get_portfolio_dataframe()
    
    r = f"""
BTC/USDC TRADING STRATEGY REPORT
{'-'*70}

Period: {pdf.index[0].strftime('%Y-%m-%d')} to {pdf.index[-1].strftime('%Y-%m-%d')}
Days: {len(pdf)} ({len(pdf)/365:.1f} years)

Capital: ${pdf['total_value'].iloc[0]:,.0f} -> ${pdf['total_value'].iloc[-1]:,.0f}
Return: {m['total_return_pct']:.2f}%
Sharpe: {m['sharpe_ratio']:.3f}
Max DD: {m['max_drawdown_pct']:.2f}%
Win rate: {m['win_rate_pct']:.1f}%

BUY: {m['buy_signals']} | SELL: {m['sell_signals']} | HOLD: {m['hold_signals']}
Trades: {m['num_trades']}

BTC weight: {pdf['btc_weight'].mean():.1%} (min: {pdf['btc_weight'].min():.1%}, max: {pdf['btc_weight'].max():.1%})

Risk controls: Drawdown (-20%), Volatility (80%), VaR (-4%), Liquidity (>$300M)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'-'*70}
"""
    return r


def plot_analysis(engine):
    rdf = engine.backtest_data
    pdf = engine.portfolio_manager.get_portfolio_dataframe()

    comp_path = 'results/mining_cost_comparison.csv'
    mcdf = None
    if os.path.exists(comp_path):
        try:
            mcdf = pd.read_csv(comp_path)
            mcdf['date'] = pd.to_datetime(mcdf['date'])
            if mcdf.empty or 'model_cost_usd' not in mcdf.columns:
                mcdf = None
        except Exception:
            mcdf = None
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(rdf['date'], rdf['btc_price'], 'b-', linewidth=2, label='BTC Price')
    ax1.plot(rdf['date'], rdf['production_cost'], color='gray', linewidth=1, alpha=0.7, label='Cost')
    ax1.plot(rdf['date'], rdf['production_cost_smoothed'], 'r--', linewidth=2, label='Cost (EMA)')
    
    if mcdf is not None:
        ax1.plot(mcdf['date'], mcdf['model_cost_usd'], color='purple', linewidth=2, linestyle=':', alpha=0.8)
        ax1.plot(mcdf['date'], mcdf['real_cost_usd'], color='green', linewidth=2, linestyle='-.')
    
    ax1.fill_between(rdf['date'], 
                      rdf['production_cost_smoothed'] * 0.9,
                      rdf['production_cost_smoothed'] * 1.1,
                      alpha=0.1, color='orange')
    ax1.set_ylabel('USD')
    ax1.set_title('Price vs Cost', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    col = {'BUY': 'g', 'SELL': 'r', 'HOLD': 'gray'}
    for sig in ['BUY', 'SELL', 'HOLD']:
        mask = rdf['signal'] == sig
        ax2.scatter(rdf[mask]['date'], rdf[mask]['signal_ratio'], c=col[sig], label=sig, alpha=0.6, s=20)
    ax2.axhline(y=0.90, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=1.10, color='r', linestyle='--', alpha=0.5)
    ax2.set_title('Signals')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(pdf.index, pdf['total_value'], 'darkgreen', linewidth=2.5)
    ax3.fill_between(pdf.index, pdf['total_value'].iloc[0], pdf['total_value'], alpha=0.3, color='darkgreen')
    ax3.set_title('Portfolio Value')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.fill_between(pdf.index, 0, pdf['btc_weight'], label='BTC', alpha=0.7, color='orange')
    ax4.fill_between(pdf.index, pdf['btc_weight'], 1, label='USDC', alpha=0.7, color='blue')
    ax4.set_title('Allocation')
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 1])
    ret = pdf['total_value'].pct_change() * 100
    ax5.hist(ret.dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax5.axvline(x=ret.mean(), color='red', linestyle='--', linewidth=2)
    ax5.set_title('Returns Distribution')
    ax5.grid(True, alpha=0.3, axis='y')
    
    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = f'results/strategy_analysis_{ts}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return fig


def main():
    print("\nBTC/USDC Trading Strategy")
    print("-" * 40)
    
    print("\n[1] Fetching data...")
    data = DataFetcher.fetch_combined_data(
        days=BacktestConfig.DAYS_TO_FETCH, 
        use_real_data=BacktestConfig.USE_REAL_DATA
    )
    print(f"    {len(data)} days loaded")
    
    print("\n[2] Running backtest...")
    engine = BacktestEngine(initial_capital=100_000)
    engine.add_from_dataframe(data)
    pdf = engine.run_backtest(initial_btc_quantity=2.0)
    print(f"    Backtest complete")
    
    print("\n[3] Calculating metrics...")
    m = engine.calculate_metrics()
    print(f"    Return: {m['total_return_pct']:.2f}% | Sharpe: {m['sharpe_ratio']:.3f} | DD: {m['max_drawdown_pct']:.2f}%")
    
    print("\n[4] Risk evaluation...")
    rm = RiskManager()
    rets = pdf['total_value'].pct_change().dropna()
    for r in rets.tail(30):
        rm.update_returns(r)
    print(f"    Vol: {rm.get_volatility():.2%} | VaR: {rm.calculate_var():.2%}")
    
    print("\n[5] Generating report...")
    report = gen_report(engine, rm)
    print(report)
    os.makedirs('results', exist_ok=True)
    fn = f"results/strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n[6] Generating graphs...")
    plot_analysis(engine)
    print("\n[7] Exporting results...")
    engine.export_results()
    print("\nDone! Check results/ folder for outputs.\n")


if __name__ == "__main__":
    main()
