"""
BTC Trading Strategy Dashboard - AUTO-UPDATING
================================================
Automatically updates data via API and displays charts.
Last updated: Feb 13, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import glob
from datetime import datetime, timedelta
import sys
import os
import io

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
import json
from data_fetcher import DataFetcher
from backtest import BacktestEngine
from risk_manager import RiskManager
from config import BacktestConfig
import config as _cfg

# Apply optimized parameters from best_params.json if available
_bp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization', 'best_params.json')
if os.path.exists(_bp_path):
    with open(_bp_path) as _f:
        _bp = json.load(_f)
    _ga, _gb, _gc = _bp['group_a'], _bp['group_b'], _bp['group_c']
    _cfg.SignalConfig.PRICE_EMA_WINDOW       = _ga['PRICE_EMA_WINDOW']
    _cfg.SignalConfig.COST_EMA_WINDOW        = _ga['COST_EMA_WINDOW']
    _cfg.SignalConfig.SIGNAL_EMA_WINDOW      = _ga['SIGNAL_EMA_WINDOW']
    _cfg.PortfolioConfig.TREND_FILTER_WINDOW = _ga['TREND_FILTER_WINDOW']
    _cfg.PortfolioConfig.RSI_OVERSOLD        = _ga['RSI_OVERSOLD']
    _cfg.PortfolioConfig.VOL_TARGET          = _ga['VOL_TARGET']
    _cfg.RiskManagementConfig.DD_THRESHOLDS.update(_gb['DD_THRESHOLDS'])
    _cfg.RiskManagementConfig.VOL_THRESHOLDS.update(_gb['VOL_THRESHOLDS'])
    _cfg.PortfolioConfig.HASH_RIBBON_CAP_MULT = _gc['HASH_RIBBON_CAP_MULT']
    _BEST_PARAMS_LOADED = True
    _BEST_PARAMS_SORTINO = _bp.get('final_sortino', None)
else:
    _BEST_PARAMS_LOADED = False
    _BEST_PARAMS_SORTINO = None

# Page config
st.set_page_config(
    page_title="Dashboard",
    page_icon="BTC",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force sidebar to stay collapsed with CSS and JavaScript
st.markdown(
    """
<style>
/* Force hide sidebar */
section[data-testid="stSidebar"] {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
}
/* Keep it hidden even when expanded */
section[data-testid="stSidebar"][aria-expanded="true"] {
    display: none !important;
}
/* Hide the collapse/expand button */
button[kind="header"] {
    display: none !important;
}
.stButton > button {
    background-color: #4f6f52;
    color: #ffffff;
    border: 1px solid #3f5f42;
    border-radius: 6px;
    height: 40px;
    margin-top: 22px;
}
.stButton > button:hover {
    background-color: #3f5f42;
    color: #ffffff;
    border: 1px solid #2f4f32;
}
h3 {
    color: #3b4637;
    font-weight: 600;
    background: #f4f6f2;
    padding: 10px 18px 10px 18px !important;
    border-radius: 6px;
    border: 1px solid #e3e8df;
    line-height: 1.35;
    display: flex;
    align-items: center;
    min-height: 44px;
    margin: 10px 0 8px 0;
}
h4 {
    color: #4a5446;
    font-weight: 600;
    background: #f8f9f5;
    padding: 8px 16px 8px 16px !important;
    border-radius: 6px;
    border: 1px solid #e7ece3;
    line-height: 1.35;
    display: flex;
    align-items: center;
    min-height: 36px;
    margin: 8px 0 6px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

def fetch_and_run_backtest(start_date_str, end_date_str, force_refresh=False, use_risk_management=True):
    """Fetch fresh data from APIs and run backtest."""
    
    # Create cache key
    cache_key = f"{start_date_str}_{end_date_str}_{use_risk_management}"
    
    # If force refresh, bypass cache
    if force_refresh:
        return _do_fetch_and_backtest(start_date_str, end_date_str, force_refresh=True, use_risk_management=use_risk_management)
    
    # Otherwise use cached version
    return _do_fetch_and_backtest_cached(start_date_str, end_date_str, use_risk_management)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def _do_fetch_and_backtest_cached(start_date_str, end_date_str, use_risk_management=True):
    """Cached version of fetch and backtest."""
    return _do_fetch_and_backtest(start_date_str, end_date_str, force_refresh=False, use_risk_management=use_risk_management)

def _do_fetch_and_backtest(start_date_str, end_date_str, force_refresh=False, use_risk_management=True):
    """Actual fetch and backtest implementation."""
    with st.spinner('Fetching fresh data from APIs...'):
        # Calculate required days based on date range
        from datetime import datetime
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
        required_days = (end_dt - start_dt).days + 1

        # Fetch data for the required period (add buffer for API limits)
        # If the end date is in the past, expand lookback to cover the historical window
        today = datetime.now().date()
        if end_dt.date() < today:
            lookback_days = (today - start_dt.date()).days + 1
            fetch_days = max(lookback_days, required_days + 120, 3000)
        else:
            # Always fetch 3000+ days: Blockchain.com returns 4-day sampled (sparse)
            # hashrate for large windows → linear interpolation creates smooth values.
            # Shorter fetches return daily noisy data → false Hash Ribbon signals.
            # 3000d ensures the validated hash-ribbon signal quality (TRAIN 1.061, OOS 0.924).
            fetch_days = max(required_days + 120, 3000)
        
        data = DataFetcher.fetch_combined_data(
            days=fetch_days,
            use_real_data=True,
            force_refresh=force_refresh
        )
        
        # Run backtest
        engine = BacktestEngine(initial_capital=100_000, enable_risk_management=use_risk_management)
        engine.add_from_dataframe(data)
        portfolio_df = engine.run_backtest(initial_btc_quantity=2.0)
        
        # Calculate metrics
        metrics = engine.calculate_metrics()
        
        # Risk manager
        rm = RiskManager()
        daily_returns = portfolio_df['total_value'].pct_change().dropna()
        for ret in daily_returns.tail(30):
            rm.update_returns(ret)
        
        return engine, portfolio_df, metrics, rm


def calculate_display_metrics(engine, portfolio_df, metrics, rm):
    """Calculate metrics for display."""
    results_df = engine.backtest_data
    
    return {
        'total_return': metrics['total_return_pct'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'max_drawdown': metrics['max_drawdown_pct'],
        'volatility': rm.get_volatility() * 100,
        'var_99': rm.calculate_var() * 100,
        'total_trades': metrics.get('buy_signals', 0) + metrics.get('sell_signals', 0),
        'win_rate': metrics.get('win_rate_pct', 0),
        'initial_value': portfolio_df['total_value'].iloc[0],
        'final_value': portfolio_df['total_value'].iloc[-1],
        'days': len(results_df),
        'buy_signals': metrics.get('buy_signals', 0),
        'sell_signals': metrics.get('sell_signals', 0),
        'hold_signals': metrics.get('hold_signals', 0),
    }


def build_trades_dataframe(portfolio_df_filtered, results_df_filtered):
    """Build a portfolio snapshot dataframe with all daily data."""
    try:
        # Start with portfolio data
        export_df = portfolio_df_filtered.reset_index()
        if 'index' in export_df.columns:
            export_df = export_df.rename(columns={'index': 'date'})
        
        # Ensure date is datetime
        export_df['date'] = pd.to_datetime(export_df['date'])
        
        # Calculate daily changes
        export_df['btc_quantity_change'] = export_df['btc_quantity'].diff()
        export_df['transaction'] = np.where(
            export_df['btc_quantity_change'] > 0, 'BUY',
            np.where(export_df['btc_quantity_change'] < 0, 'SELL', 'HOLD')
        )
        export_df['trade_value_usd'] = (export_df['btc_quantity_change'].abs() * export_df['btc_price']).round(2)
        
        # Add signal info if available
        if 'signal' in results_df_filtered.columns and 'signal_ratio' in results_df_filtered.columns:
            try:
                signal_map = results_df_filtered[['date', 'signal', 'signal_ratio', 'btc_price', 'production_cost_smoothed']].copy()
                signal_map['date'] = pd.to_datetime(signal_map['date'])
                export_df = export_df.merge(signal_map, on='date', how='left', suffixes=('', '_sig'))
            except Exception:
                export_df['signal'] = np.nan
                export_df['signal_ratio'] = np.nan
        else:
            export_df['signal'] = np.nan
            export_df['signal_ratio'] = np.nan
        
        # Select and order columns
        cols_to_export = [
            'date', 'btc_price', 'transaction', 'btc_quantity_change', 'trade_value_usd',
            'btc_quantity', 'btc_value', 'usdc_value', 'total_value',
            'signal', 'signal_ratio'
        ]
        cols_available = [c for c in cols_to_export if c in export_df.columns]
        
        return export_df[cols_available].sort_values('date')
    except Exception as e:
        # Return empty dataframe on error
        return pd.DataFrame()


def create_pdf_report(display_metrics, results_df_filtered, portfolio_df_filtered, start_date, end_date):
    """Create a simple PDF report and return bytes."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    title = "BTC Trading Strategy Report"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, title)

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 75, f"Period: {start_date} → {end_date}")
    c.drawString(50, height - 92, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 130
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Key Metrics")
    c.setFont("Helvetica", 11)
    y -= 20

    metric_lines = [
        f"Total Return: {display_metrics['total_return']:.2f}%",
        f"Sharpe Ratio: {display_metrics['sharpe_ratio']:.3f}",
        f"Sortino Ratio: {display_metrics.get('sortino_ratio', 0):.3f}",
        f"Max Drawdown: {display_metrics['max_drawdown']:.2f}%",
        f"Volatility (30d): {display_metrics['volatility']:.2f}%",
        f"Win Rate: {display_metrics['win_rate']:.1f}%",
        f"Total Trades: {display_metrics['total_trades']}",
        f"BUY / SELL / HOLD: {display_metrics['buy_signals']} / {display_metrics['sell_signals']} / {display_metrics['hold_signals']}",
        f"Initial Value: ${display_metrics['initial_value']:,.2f}",
        f"Final Value: ${display_metrics['final_value']:,.2f}",
        f"Days Analyzed: {display_metrics['days']}",
    ]

    for line in metric_lines:
        c.drawString(50, y, line)
        y -= 16

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Data Summary")
    c.setFont("Helvetica", 11)
    y -= 20

    data_lines = [
        f"BTC Price range: ${results_df_filtered['btc_price'].min():,.2f} → ${results_df_filtered['btc_price'].max():,.2f}",
        f"Production cost (EMA) range: ${results_df_filtered['production_cost_smoothed'].min():,.2f} → ${results_df_filtered['production_cost_smoothed'].max():,.2f}",
        f"Portfolio value range: ${portfolio_df_filtered['total_value'].min():,.2f} → ${portfolio_df_filtered['total_value'].max():,.2f}",
    ]

    for line in data_lines:
        c.drawString(50, y, line)
        y -= 16

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# Header
st.title("BTC/USDC Mean-Reversion Strategy")
st.markdown("Live data via CoinGecko and Blockchain.info | Updates hourly")

st.markdown(
    """
    <div style="border:1px solid #E6E9EF; background:#F7F9FC; padding:16px 18px; border-radius:12px; margin:8px 0 14px 0;">
        <div style="font-size:18px; font-weight:700; color:#3b4637;">Strategy Overview</div>
        <div style="margin-top:6px; color:#3C4858; font-size:15px;">
            A systematic mean-reversion strategy that sizes BTC exposure based on how far the price deviates
            from its estimated production cost, with multiple independent risk filters running on top.
            Full methodology and parameter details are in the Strategy page.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if st.button("Full Strategy Details →", type="primary"):
    st.session_state["show_strategy_page"] = True
    st.rerun()

if st.session_state.get("show_strategy_page"):
    from strategy_page import render_strategy_page
    render_strategy_page()
    st.stop()

st.markdown("---")

# Risk engine is always active
use_risk_management = True

st.markdown("---")
# Date Selection - Main Page
st.markdown("### 📅 Choose Time Period")
st.markdown("_We have data from 2016. Default time is 4 years back_")

col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])

default_end_date = datetime.now().date()
default_start_date = (datetime.now() - timedelta(days=1460)).date()  # ~4 years

with col1:
    start_date = st.date_input(
        "Start Date",
        value=default_start_date,
        min_value=datetime(2016, 1, 1).date(),
        max_value=default_end_date
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=default_end_date,
        min_value=start_date,
        max_value=default_end_date
    )

with col3:
    # Calculate days
    days_selected = (end_date - start_date).days
    st.metric("Days Total", f"{days_selected}", delta=None)

with col4:
    force_refresh = st.button("Get New Data", type="primary")
    if force_refresh:
        st.cache_data.clear()
        st.session_state['force_refresh'] = True
        st.rerun()

with col5:
    st.markdown(
        f"<div style='text-align:right; color:#5a6256; font-size:14px; padding-top:28px;'>"
        f"Last update: {datetime.now().strftime('%H:%M:%S')}"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# Fetch data and run backtest
try:
    force_refresh_flag = st.session_state.get('force_refresh', False)
    if force_refresh_flag:
        st.session_state['force_refresh'] = False  # Reset flag
    
    engine, portfolio_df, metrics_raw, rm = fetch_and_run_backtest(
        str(start_date), 
        str(end_date),
        force_refresh=force_refresh_flag,
        use_risk_management=use_risk_management
    )
    results_df = engine.backtest_data.copy()
    
    # Show total data available
    data_start = pd.to_datetime(results_df['date'].min()).strftime('%Y-%m-%d')
    data_end = pd.to_datetime(results_df['date'].max()).strftime('%Y-%m-%d')
    days_available = len(results_df)
    days_requested = (end_date - start_date).days + 1
    
    st.info(f"We have data: {days_available} days ({data_start} → {data_end})")
    
    # Warn if cached data is insufficient
    if days_available < days_requested * 0.9:  # Allow 10% tolerance
        st.warning(
            f"⚠️ **We have only {days_available} days, but you want {days_requested} days.**\n\n"
            f"Click **'Get New Data'** button for fresh data from internet (BTC price + hashrate).\n\n"
            f"Note: Hashrate data comes every 4 days. We make it daily."
        )
    
    # Filter data based on selected date range
    results_df['date'] = pd.to_datetime(results_df['date'])
    mask = (results_df['date'].dt.date >= start_date) & (results_df['date'].dt.date <= end_date)
    results_df_filtered = results_df[mask].copy()
    
    portfolio_df_all = portfolio_df.copy()
    portfolio_df_filtered = portfolio_df_all[
        (portfolio_df_all.index.date >= start_date) & 
        (portfolio_df_all.index.date <= end_date)
    ].copy()
    
    if len(results_df_filtered) == 0 or len(portfolio_df_filtered) == 0:
        st.error("Sorry, no data for this time!")
        st.stop()
    
    # Recalculate metrics for filtered period
    initial_value = portfolio_df_filtered['total_value'].iloc[0]
    final_value = portfolio_df_filtered['total_value'].iloc[-1]
    total_return = ((final_value - initial_value) / initial_value) * 100
    
    daily_returns = portfolio_df_filtered['total_value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
    downside = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 1 and downside.std() > 0 else 0

    cummax = portfolio_df_filtered['total_value'].cummax()
    drawdown = ((portfolio_df_filtered['total_value'] - cummax) / cummax) * 100
    max_dd = drawdown.min()
    
    display_metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'volatility': daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0,
        'var_99': daily_returns.quantile(0.01) * 100 if len(daily_returns) > 0 else 0,
        'total_trades': (results_df_filtered['signal'] == 'BUY').sum() + (results_df_filtered['signal'] == 'SELL').sum(),
        'win_rate': ((daily_returns > 0).sum() / len(daily_returns) * 100) if len(daily_returns) > 0 else 0,
        'initial_value': initial_value,
        'final_value': final_value,
        'days': len(results_df_filtered),
        'buy_signals': (results_df_filtered['signal'] == 'BUY').sum(),
        'sell_signals': (results_df_filtered['signal'] == 'SELL').sum(),
        'hold_signals': (results_df_filtered['signal'] == 'HOLD').sum(),
    }

    trades_export_df = build_trades_dataframe(portfolio_df_filtered, results_df_filtered)
    pdf_report_bytes = None
    if REPORTLAB_AVAILABLE:
        pdf_report_bytes = create_pdf_report(
            display_metrics,
            results_df_filtered,
            portfolio_df_filtered,
            start_date,
            end_date
        )
    
    # Use filtered data for all graphs
    results_df = results_df_filtered
    
    filtered_start = results_df_filtered['date'].min().strftime('%Y-%m-%d')
    filtered_end = results_df_filtered['date'].max().strftime('%Y-%m-%d')
    
    st.success(f"Data ready! {len(results_df_filtered)} days ({filtered_start} → {filtered_end})")

    
except Exception as e:
    st.error(f"Problem with data: {str(e)}")
    st.info("Maybe check your internet connection")
    st.stop()

# Metrics Row
if _BEST_PARAMS_LOADED:
    st.success(f"✅ Optimized parameters loaded (walk-forward OOS Sortino = {_BEST_PARAMS_SORTINO:.3f})")
st.markdown("### Performance Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Total Return",
        f"{display_metrics['total_return']:.2f}%",
        delta=f"${display_metrics['final_value'] - display_metrics['initial_value']:,.0f}",
        help=(
            "Total portfolio return over the selected period.\n"
            "Formula: (Final value − Initial value) / Initial value × 100\n"
            "The dollar delta shows the absolute gain or loss in USD."
        )
    )

with col2:
    st.metric(
        "Sharpe Ratio",
        f"{display_metrics['sharpe_ratio']:.3f}",
        delta=None,
        help=(
            "Risk-adjusted return relative to total volatility (up and down moves).\n"
            "Formula: (mean daily return / std daily return) × √252\n"
            "Limitation: penalises large upside moves the same as losses. "
            "That is why Sortino is used as the primary optimisation target here."
        )
    )

with col3:
    st.metric(
        "Sortino Ratio",
        f"{display_metrics['sortino_ratio']:.3f}",
        delta=None,
        help=(
            "Risk-adjusted return relative to downside volatility only.\n"
            "Formula: (mean daily return / std of negative daily returns) × √252\n"
            "This is the primary metric the strategy parameters were optimised for."
        )
    )

with col4:
    st.metric(
        "Max Drawdown",
        f"{display_metrics['max_drawdown']:.2f}%",
        delta=None,
        delta_color="inverse",
        help=(
            "Largest peak-to-trough decline over the selected period.\n"
            "Formula: (portfolio value − running peak) / running peak × 100\n"
            "The risk engine activates: CAUTION at −12%, RISK_OFF at −20%, EMERGENCY at −35%."
        )
    )

with col5:
    st.metric(
        "Volatility (Ann.)",
        f"{display_metrics['volatility']:.2f}%",
        delta=None,
        help=(
            "Annualised standard deviation of daily portfolio returns.\n"
            "Formula: std(daily returns) × √252 × 100\n"
            "The strategy targets ~40% annualised vol via dynamic position scaling. "
            "Raw BTC typically runs at 60–80% annualised vol."
        )
    )

with col6:
    st.metric(
        "Win Rate",
        f"{display_metrics['win_rate']:.1f}%",
        delta=None,
        help=(
            "Percentage of calendar days with a positive portfolio return.\n"
            "Formula: (days with return > 0) / total days × 100\n"
            "A low win rate is normal for a mean-reversion strategy: "
            "it profits from fewer but larger moves, not from being right every day."
        )
    )

st.markdown("---")
st.markdown("### Download Files")
col1, col2 = st.columns(2)

with col1:
    if REPORTLAB_AVAILABLE and pdf_report_bytes:
        st.download_button(
            "Get PDF File",
            data=pdf_report_bytes,
            file_name=f"btc_strategy_report_{start_date}_{end_date}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.button("Get PDF File", disabled=True, use_container_width=True)
        st.info("Need reportlab for PDF: pip install reportlab")

with col2:
    trades_csv = trades_export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Get CSV File",
        data=trades_csv,
        file_name=f"btc_trades_{start_date}_{end_date}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")

# Charts
st.markdown("### Chart Analysis")
st.markdown("""
**Portfolio value** = BTC position + USDC position. The backtest opens with up to 2 BTC (remaining capital in USDC at the first day's price). Changing the date range reruns the backtest — data is cached for 1 hour.
""")

with st.expander("How this strategy works", expanded=False):
    st.markdown("""
    #### Core signal — Price vs. Mining Cost ratio
    Bitcoin's production cost is estimated daily from network hashrate and electricity prices.
    When price drops well below what it costs to mine a coin, miners operate at a loss — something
    that cannot go on indefinitely. The ratio (price ÷ cost) is used to set BTC allocation each day:

    | Ratio | BTC allocation |
    |---|---|
    | < 0.80 | **100%** |
    | 0.80 – 0.90 | 85% |
    | 0.90 – 1.00 | 70% |
    | 1.00 – 1.10 | 50% |
    | 1.10 – 1.25 | 30% |
    | > 1.25 | **0%** |

    **Smoothing (concrete numbers):** Before computing the ratio, price is smoothed with a
    **30-day exponential moving average (EMA)** and mining cost with a **90-day EMA**.
    An EMA with window N moves by a factor of 2/(N+1) toward the current value each day —
    so the 30-day price EMA shifts at most ~6.5% per day toward raw price,
    and the 90-day cost EMA shifts at most ~2.2% per day.
    In practice the ratio can only move a small fraction each day, so the portfolio allocation
    rarely changes by more than 2–3 percentage points on a normal trading day.
    The strategy reacts to sustained trends over weeks, not to single-day price moves.

    ---
    #### What actually drives results — ablation study
    Each component was removed one at a time and the out-of-sample Sortino ratio measured.
    Baseline with everything active: **Sortino = 2.443**.

    | Component removed | Change in Sortino | Takeaway |
    |---|---|---|
    | Signal EMA smoothing | −0.43 | **Biggest driver by far.** Wrong EMA windows kill most of the edge. |
    | Hash Ribbon filter | −0.34 | **Second biggest.** Without it the strategy holds through miner capitulation crashes. |
    | Trend filter (360-day EMA) | −0.26 | **Meaningful.** Without it the strategy stays long through prolonged bear markets. |
    | RSI oversold boost | −0.07 | Small but consistent across folds. Kept. |
    | 4-mode drawdown engine | −0.02 | Near-zero Sortino impact — but it hard-caps exposure during extreme stress (2018, 2022). Kept for tail risk. |
    | Volatility scaling | −0.02 | Near-zero Sortino impact — but it keeps annualised vol near the 40% target. Kept for risk management. |

    ---
    #### Indirect risk controls
    These don't generate trades — they reduce or zero out the allocation when conditions are bad:

    - **Hash Ribbon (30/60-day hashrate SMA crossover):** when the 30-day hashrate average drops
      below the 60-day average, miners are turning off machines because mining is unprofitable.
      Allocation is cut to 0% until the cross-back. Periods fixed from mining literature; not optimised.

    - **Trend filter (360-day EMA):** if BTC is below its long-term moving average, the strategy
      treats it as a bear market regime and holds 0% BTC regardless of the ratio.
      Window fixed from technical analysis literature.

    - **Volatility scaling:** realised portfolio volatility is computed daily. When it exceeds
      the 40% annualised target, the position is scaled down proportionally.
      The 40% target was optimised; raw BTC vol typically runs at 60–80%.

    - **RSI oversold boost:** when the 14-day RSI drops below 30, the target allocation
      gets a ×1.30 multiplier to push harder into oversold conditions.
      The threshold (30) and multiplier (1.30) were fixed from momentum literature.

    ---
    #### Direct risk controls — 4-mode drawdown engine
    Tracks the portfolio's rolling peak-to-trough decline and hard-caps BTC exposure:

    | Mode | Triggers at | Max BTC | Stays elevated for |
    |---|---|---|---|
    | NORMAL | — | 100% | — |
    | CAUTION | −12% | 75% | 2 days after recovery |
    | RISK_OFF | −20% | 45% | 3 days after recovery |
    | EMERGENCY | −35% | 5% | 5 days after recovery |

    Sticky recovery prevents flip-flopping: the mode stays elevated for a few days after the
    drawdown heals, so it does not immediately re-open a full position into a volatile bounce.
    Thresholds were optimised in walk-forward validation; recovery windows were fixed.

    Ablation shows this layer barely changes Sortino (−0.02), but it prevented catastrophic
    losses in 2018 and 2022 by keeping exposure near-zero for weeks during those crashes.

    ---
    #### How parameters were chosen
    - **Optimised by walk-forward cross-validation** (multiple in-sample/out-of-sample folds):
      EMA windows for price, cost, and signal; trend filter window; RSI oversold threshold;
      drawdown thresholds; volatility target.
    - **Fixed from established literature:** RSI period (14 days), hashrate SMA windows (30/60 days),
      RSI multiplier (1.30), trend filter structure.
    - **Tested as direct risk managers, then dropped:**
      stop-loss at −15% (triggered on normal pullbacks, missed the recovery),
      take-profit at +25% (cut bull-run gains prematurely),
      trailing stop at −10% (caused whipsaw in high-vol regimes).
      The 4-mode engine replaced all of these because it adjusts the *exposure level gradually*
      rather than forcing binary entry/exit events.

    ---
    #### Sentiment signals tested
    - **RSI (14-day):** kept. Ablation: −0.07 Sortino when removed, small but consistent.
      The +30% position boost adds a gentle push into oversold conditions (e.g. March 2020).
    - **Fear and Greed Index:** tested, dropped. Data before 2019 was reconstructed backwards
      from incomplete records — which inflated backtest results artificially. Not reliable.
    - **SOPR (Spent Output Profit Ratio):** tested, dropped. Reliable daily data only from ~2020.
      Without the full training window it could not be validated properly.
    """)
st.markdown("---")


# 1. BTC Price vs Production Cost
st.markdown("""
#### Bitcoin Price vs Mining Cost
The primary signal. Daily mining cost (gray line, raw estimate) is smoothed with a
**90-day EMA** (red dashed line) — that smoothed value is what goes into the ratio.
BTC price is smoothed with a **30-day EMA** before the ratio is calculated.

**Concrete numbers:** an EMA with window N shifts by 2/(N+1) toward the current value each day.
The 30-day price EMA moves at most ~6.5% per day; the 90-day cost EMA at most ~2.2% per day.
Even on a day with a large price move, the ratio and therefore the allocation can only shift a
small fraction — typically 2–3 percentage points. The strategy reacts to trends over weeks,
not to day-to-day swings.

The orange band is ±10% around the smoothed cost. That is the neutral zone (50–70% BTC allocation).
Below the band = strategy is increasingly long; above it = increasingly reducing exposure.
""")

fig_main = go.Figure()

fig_main.add_trace(go.Scatter(
    x=results_df['date'],
    y=results_df['btc_price'],
    mode='lines',
    name='BTC Price',
    line=dict(color='#0066FF', width=2.5)
))

fig_main.add_trace(go.Scatter(
    x=results_df['date'],
    y=results_df['production_cost'],
    mode='lines',
    name='Mining Cost (Real)',
    line=dict(color='gray', width=1),
    opacity=0.5
))

fig_main.add_trace(go.Scatter(
    x=results_df['date'],
    y=results_df['production_cost_smoothed'],
    mode='lines',
    name='Mining Cost (Smooth)',
    line=dict(color='#FF0000', width=2, dash='dash')
))

# Add ±10% buffer
upper_buffer = results_df['production_cost_smoothed'] * 1.1
lower_buffer = results_df['production_cost_smoothed'] * 0.9

fig_main.add_trace(go.Scatter(
    x=results_df['date'],
    y=upper_buffer,
    mode='lines',
    name='±10% Zone',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig_main.add_trace(go.Scatter(
    x=results_df['date'],
    y=lower_buffer,
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 165, 0, 0.1)',
    line=dict(width=0),
    name='±10% Zone',
    hoverinfo='skip'
))

fig_main.update_layout(
    title="BTC Price vs Mining Cost",
    xaxis_title="Date",
    yaxis_title="USD",
    template="plotly_white",
    height=500,
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
)

st.plotly_chart(fig_main, use_container_width=True)

# 2. Total Portfolio Value
st.markdown("""
#### Total Portfolio Value
Strategy equity curve (green) vs. a simple **buy-and-hold BTC** benchmark (blue dotted line) — 
buying the same initial dollar amount of BTC on day one and never rebalancing.

The strategy does not trade in and out of BTC in discrete blocks. Instead, it adjusts the 
*allocation* continuously: more BTC when the ratio is low (price cheap relative to cost), 
less or none when the signal deteriorates or a risk control fires. In practice the allocation 
rarely swings by more than a few percentage points per day because both the signal and the 
risk engine change gradually.

The typical tradeoff: the strategy **lags the benchmark during strong bull runs** (it is rarely 
at 100% allocation at the top) but **absorbs significantly less drawdown in bear markets** 
(filters cut exposure before the worst of the decline). Over a full cycle, reducing peak-to-trough 
loss from 70%+ to ∼30–40% means a far shorter path back to new highs, which compounds 
into better risk-adjusted returns even if raw returns are similar.
""")

fig_portfolio = go.Figure()

initial_value = portfolio_df_filtered['total_value'].iloc[0]
initial_price = results_df_filtered['btc_price'].iloc[0]
buy_and_hold_btc = initial_value / initial_price
buy_and_hold_values = buy_and_hold_btc * results_df_filtered['btc_price']

fig_portfolio.add_trace(go.Scatter(
    x=portfolio_df_filtered.index,
    y=portfolio_df_filtered['total_value'],
    mode='lines',
    name='Strategy',
    line=dict(color='#006400', width=2.5),
    fill='tozeroy',
    fillcolor='rgba(0, 100, 0, 0.15)'
))

fig_portfolio.add_trace(go.Scatter(
    x=results_df_filtered['date'],
    y=buy_and_hold_values,
    mode='lines',
    name='Buy & Hold BTC',
    line=dict(color='#1f77b4', width=2, dash='dot')
))

fig_portfolio.update_layout(
    title="Total Portfolio Value — Strategy vs. Buy & Hold",
    xaxis_title="Date",
    yaxis_title="Portfolio Value (USD)",
    template="plotly_white",
    height=480,
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
    yaxis=dict(tickprefix='$', separatethousands=True)
)

st.plotly_chart(fig_portfolio, use_container_width=True)

# 3. Row: Portfolio Allocation + Returns Distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### BTC / USDC Allocation Over Time
    Daily split between BTC (orange) and USDC (blue), showing how the portfolio allocation shifts
    as the signal and risk controls change.

    The BTC weight is the output of five stacked layers, applied in order:
    1. Ratio signal — sets the base target (0–100% across 6 steps)
    2. Hash Ribbon filter — forces 0% during miner capitulation
    3. Trend filter (250-day moving average) — forces 0% in bear market regime
    4. Risk engine cap — CAUTION: max 75%, RISK_OFF: max 45%, EMERGENCY: max 5%
    5. Volatility scaling — reduces exposure proportionally when realised vol exceeds 40% target

    Sharp drops to 0% reflect one of the protective filters closing the position.
    """)
    
    # Portfolio allocation
    fig_alloc = go.Figure()
    
    fig_alloc.add_trace(go.Scatter(
        x=portfolio_df_filtered.index,
        y=portfolio_df_filtered['btc_weight'],
        mode='lines',
        name='BTC',
        line=dict(width=0),
        stackgroup='one',
        fillcolor='rgba(255, 165, 0, 0.7)'
    ))
    
    fig_alloc.add_trace(go.Scatter(
        x=portfolio_df_filtered.index,
        y=1 - portfolio_df_filtered['btc_weight'],
        mode='lines',
        name='USDC',
        line=dict(width=0),
        stackgroup='one',
        fillcolor='rgba(0, 0, 255, 0.7)'
    ))
    
    fig_alloc.update_layout(
        title="BTC / USDC Allocation",
        xaxis_title="Date",
        yaxis_title="Allocation (%)",
        template="plotly_white",
        height=400,
        yaxis=dict(range=[0, 1]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_alloc, use_container_width=True)

with col2:
    st.markdown("""
    #### Daily Return Distribution
    Histogram of how often the portfolio gained or lost a given percentage on each calendar day.

    - **Red dashed line** — mean daily return
    - **Wider spread** — higher realised volatility; the strategy targets ~40% annualised via vol scaling
    - **Right skew** — more large gains than large losses, which is desirable for a long-biased strategy

    Both Sharpe and Sortino are calculated from this distribution.
    Sharpe uses the full standard deviation; Sortino uses only the downside half —
    which is why Sortino is the primary optimisation target here.
    """)
    
    # Returns distribution
    returns = portfolio_df_filtered['total_value'].pct_change() * 100
    returns_clean = returns.dropna()
    
    fig_returns = go.Figure()
    
    fig_returns.add_trace(go.Histogram(
        x=returns_clean,
        nbinsx=50,
        name='Daily Change',
        marker=dict(color='skyblue', line=dict(color='black', width=1)),
        opacity=0.7
    ))
    
    # Add mean line
    mean_return = returns_clean.mean()
    fig_returns.add_vline(x=mean_return, line_dash="dash", line_color="red", line_width=2)
    
    fig_returns.update_layout(
        title="Daily Return Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Number of Days",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_returns, use_container_width=True)

# 4. Hash Ribbon Chart
st.markdown("""
#### Hash Ribbon — Miner Capitulation Filter
Network hashrate with two moving averages: **30-day fast SMA** (blue) and **60-day slow SMA** (red).

When the fast SMA drops below the slow SMA, miners are switching off machines because mining is
unprofitable at current prices — a historically reliable early warning of sustained price drops
(most clearly visible in the 2018 and 2022 bear markets).

**Orange shading** marks capitulation zones (fast < slow): during these periods the strategy locks
BTC exposure to 0% regardless of the price/cost ratio, preventing large drawdowns during the
worst part of bear markets.
""")

if 'hashrate' in results_df.columns and 'hr_fast' in results_df.columns and 'hr_slow' in results_df.columns:
    fig_hr = go.Figure()

    # Shade capitulation periods
    cap_mask = results_df['hr_fast'] < results_df['hr_slow']
    in_cap = False
    cap_start = None
    for i, (date, is_cap) in enumerate(zip(results_df['date'], cap_mask)):
        if is_cap and not in_cap:
            cap_start = date
            in_cap = True
        elif not is_cap and in_cap:
            fig_hr.add_vrect(
                x0=cap_start, x1=date,
                fillcolor='rgba(255, 140, 0, 0.18)',
                layer='below', line_width=0
            )
            in_cap = False
    if in_cap:
        fig_hr.add_vrect(
            x0=cap_start, x1=results_df['date'].iloc[-1],
            fillcolor='rgba(255, 140, 0, 0.18)',
            layer='below', line_width=0
        )

    fig_hr.add_trace(go.Scatter(
        x=results_df['date'], y=results_df['hashrate'],
        mode='lines', name='Hashrate',
        line=dict(color='rgba(100,120,140,0.4)', width=1)
    ))
    fig_hr.add_trace(go.Scatter(
        x=results_df['date'], y=results_df['hr_fast'],
        mode='lines', name='30-day SMA (fast)',
        line=dict(color='#1f77b4', width=2)
    ))
    fig_hr.add_trace(go.Scatter(
        x=results_df['date'], y=results_df['hr_slow'],
        mode='lines', name='60-day SMA (slow)',
        line=dict(color='#e74c3c', width=2)
    ))

    fig_hr.update_layout(
        title="Hash Ribbon — Miner Capitulation Filter",
        xaxis_title="Date",
        yaxis_title="Hashrate (EH/s)",
        template="plotly_white",
        height=420,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    st.plotly_chart(fig_hr, use_container_width=True)
else:
    st.info("Hashrate data not available in this backtest run.")

# 5. Drawdown Chart
st.markdown("""
#### Portfolio Drawdown
Rolling peak-to-trough decline of the portfolio value over time. This is the primary metric
the risk engine is designed to contain.

The four risk mode thresholds are marked by the drawdown levels they monitor:
- **CAUTION** activates at −12% → caps BTC exposure at 75%
- **RISK_OFF** activates at −20% → caps BTC exposure at 45%
- **EMERGENCY** activates at −35% → caps BTC exposure at 5%

A smaller max drawdown relative to buy-and-hold means a shorter recovery path back to a new
high-water mark — which directly improves long-run compounded performance.
""")

cummax = portfolio_df_filtered['total_value'].cummax()
drawdown = ((portfolio_df_filtered['total_value'] - cummax) / cummax) * 100

fig_dd = go.Figure()

fig_dd.add_trace(go.Scatter(
    x=portfolio_df_filtered.index,
    y=drawdown,
    mode='lines',
    name='Loss %',
    line=dict(color='#FF6B6B', width=2),
    fill='tozeroy',
    fillcolor='rgba(255, 107, 107, 0.3)'
))

fig_dd.update_layout(
    title="Portfolio Drawdown",
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    template="plotly_white",
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_dd, use_container_width=True)

# Trading Signals Summary
st.markdown("### Signal Breakdown")
st.markdown("""
Count of BUY, SELL, and HOLD signals across the selected period.
The strategy produces a signal each day based on the ratio, but most days are HOLD
because both the ratio and the risk filters change gradually — large allocation shifts
only happen when the ratio moves through a threshold or a filter turns on or off.
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("BUY Signals", display_metrics['buy_signals'])
with col2:
    st.metric("SELL Signals", display_metrics['sell_signals'])
with col3:
    st.metric("HOLD Signals", display_metrics['hold_signals'])
with col4:
    st.metric("Total Trades", display_metrics['total_trades'])

# Signal distribution pie chart
signal_counts = results_df['signal'].value_counts()
color_map = {'BUY': '#2ecc71', 'SELL': '#e74c3c', 'HOLD': '#95a5a6'}
pie_colors = [color_map.get(lbl, '#95a5a6') for lbl in signal_counts.index]

fig_pie = go.Figure(data=[go.Pie(
    labels=signal_counts.index,
    values=signal_counts.values,
    hole=0.4,
    marker_colors=pie_colors
)])

fig_pie.update_layout(
    title="Signal Distribution",
    template="plotly_white",
    height=300
)

st.plotly_chart(fig_pie, use_container_width=True)

# Daily data table
st.markdown(f"### Daily Data ({len(results_df)} trading days)")
st.markdown("""
Full day-by-day breakdown for the selected period showing BTC price, smoothed mining cost,
the signal generated, and the price-to-cost ratio. Sort or filter to inspect specific periods,
or cross-reference with the charts above to understand individual allocation decisions.
""")
recent_cols = ['date', 'btc_price', 'production_cost_smoothed', 'signal', 'signal_ratio']
recent_df = results_df[recent_cols].copy()

# Format numbers
recent_df['btc_price'] = recent_df['btc_price'].apply(lambda x: f"${x:,.2f}")
recent_df['production_cost_smoothed'] = recent_df['production_cost_smoothed'].apply(lambda x: f"${x:,.2f}")
recent_df['signal_ratio'] = recent_df['signal_ratio'].apply(lambda x: f"{x:.3f}")

# Rename columns
recent_df.columns = ['Date', 'BTC Price (USD)', 'Mining Cost EMA (USD)', 'Signal', 'Price / Cost Ratio']

st.dataframe(recent_df.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666;'>
        <p><strong>Data sources:</strong> CoinGecko (BTC price) &amp; Blockchain.info (network hashrate)</p>
        <p>Data cached for 1 hour | Click "Get New Data" to force a refresh</p>
        <p>Last page load: <strong>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)
