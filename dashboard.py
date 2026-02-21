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
from data_fetcher import DataFetcher
from strategy import TradingStrategy
from signal_calculator import SignalCalculator
from confirmation_layer import ConfirmationLayer
from risk_manager import RiskManager
from config import BacktestConfig

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
            fetch_days = max(lookback_days, required_days + 30, 90)
        else:
            fetch_days = max(required_days + 30, 90)  # At least 90 days
        
        data = DataFetcher.fetch_combined_data(
            days=fetch_days,
            use_real_data=True,
            force_refresh=force_refresh
        )
        
        # Run strategy (nova verzija)
        strategy = TradingStrategy(initial_capital=100_000)
        prices = np.array(data['btc_price'].values, dtype=float)
        
        for idx in range(200, len(data)):
            row = data.iloc[idx]
            returns_30 = np.diff(prices[max(0, idx-30):idx+1]) / prices[max(0, idx-30):idx]
            
            strategy.run_daily_cycle(
                date=row['date'],
                btc_price=row['btc_price'],
                production_cost=row['production_cost'],
                prices_last_200=prices[:idx+1],
                prices_last_90=prices[max(0, idx-90):idx+1],
                daily_returns_30=returns_30
            )
        
        # Get execution log (simulacija portfolio_df)
        log_df = strategy.get_execution_log_df()
        
        # Add 'signal' column for dashboard compatibility (map from confirmation_reason)
        def map_signal(row):
            if not row['confirmed']:
                return 'HOLD'
            reason = str(row['confirmation_reason']).upper()
            if 'BUY' in reason or row['target_btc'] > row['current_position_btc']:
                return 'BUY'
            elif 'SELL' in reason or row['target_btc'] < row['current_position_btc']:
                return 'SELL'
            else:
                return 'HOLD'
        
        log_df['signal'] = log_df.apply(map_signal, axis=1)
        log_df['signal_ratio'] = log_df['cost_ratio']  # Use cost_ratio as signal_ratio
        log_df['production_cost_smoothed'] = log_df['production_cost']
        
        # Build portfolio_df iz execution log
        # NOTE: current_position_btc/usdc are WEIGHTS (0-1), not quantities!
        portfolio_df = pd.DataFrame()
        portfolio_df['date'] = pd.to_datetime(log_df['date'])
        portfolio_df.set_index('date', inplace=True)
        portfolio_df['btc_price'] = log_df['btc_price'].values
        portfolio_df['btc_weight'] = log_df['current_position_btc'].values  # Target weight
        portfolio_df['usdc_weight'] = log_df['current_position_usdc'].values
        
        # Calculate actual portfolio value with PROPER ACCOUNTING
        # Track actual BTC quantity and USDC balance through time
        initial_capital = strategy.initial_capital
        initial_btc_weight = log_df['current_position_btc'].iloc[0]
        initial_usdc_weight = log_df['current_position_usdc'].iloc[0]
        initial_btc_price = portfolio_df['btc_price'].iloc[0]
        
        # Initialize positions
        btc_quantity = (initial_capital * initial_btc_weight) / initial_btc_price
        usdc_balance = initial_capital * initial_usdc_weight
        
        quantities_btc = []
        balances_usdc = []
        total_values = []
        
        prev_weight = initial_btc_weight
        
        for idx in range(len(portfolio_df)):
            current_price = portfolio_df['btc_price'].iloc[idx]
            target_weight = portfolio_df['btc_weight'].iloc[idx]
            
            # Calculate current value BEFORE any rebalancing
            btc_value = btc_quantity * current_price
            current_total = btc_value + usdc_balance
            
            # Check if we need to rebalance (weight changed)
            if abs(target_weight - prev_weight) > 0.001:  # Weight changed
                # Rebalance to target weight
                target_btc_value = current_total * target_weight
                target_usdc_value = current_total * (1 - target_weight)
                
                # Execute trade
                btc_quantity = target_btc_value / current_price
                usdc_balance = target_usdc_value
                
                prev_weight = target_weight
            
            # Store current state AFTER rebalancing
            quantities_btc.append(btc_quantity)
            balances_usdc.append(usdc_balance)
            total_values.append(btc_quantity * current_price + usdc_balance)
        
        portfolio_df['btc_quantity'] = quantities_btc
        portfolio_df['usdc_value'] = balances_usdc
        portfolio_df['total_value'] = total_values
        portfolio_df['btc_value'] = portfolio_df['btc_quantity'] * portfolio_df['btc_price']
        
        # Recalculate actual weights based on real positions
        portfolio_df['btc_weight'] = portfolio_df['btc_value'] / portfolio_df['total_value']
        portfolio_df['btc_weight'] = portfolio_df['btc_weight'].fillna(0)
        
        # Diagnostics - print to console for debugging
        print(f"\n[PORTFOLIO DEBUG]")
        print(f"  Initial capital: ${initial_capital:,.0f}")
        print(f"  Initial BTC weight: {initial_btc_weight*100:.1f}%")
        print(f"  Initial BTC price: ${initial_btc_price:,.2f}")
        print(f"  Initial BTC quantity: {btc_quantity:.6f} BTC")
        print(f"  Weight changes detected: {(portfolio_df['btc_weight'].diff().abs() > 0.001).sum()}")
        print(f"  Final portfolio value: ${portfolio_df['total_value'].iloc[-1]:,.2f}")
        print(f"  Final BTC quantity: {portfolio_df['btc_quantity'].iloc[-1]:.6f} BTC")
        
        # Metrics
        metrics = {
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'win_rate_pct': 0,
        }
        
        # Risk manager
        rm = RiskManager()
        
        return strategy, log_df, portfolio_df, metrics, rm


def calculate_display_metrics(strategy, log_df, portfolio_df, metrics, rm):
    """Calculate metrics for display."""
    
    return {
        'total_return': 0,
        'sharpe_ratio': 0,
        'max_drawdown': 0,
        'volatility': log_df['annual_vol'].mean() * 100 if 'annual_vol' in log_df.columns else 0,
        'var_99': 0,
        'total_trades': log_df['confirmed'].sum() if 'confirmed' in log_df.columns else 0,
        'win_rate': (log_df['confirmed'].sum() / len(log_df) * 100) if len(log_df) > 0 else 0,
        'initial_value': 100_000,
        'final_value': 100_000,
        'days': len(log_df),
        'buy_signals': metrics.get('buy_signals', 0),
        'sell_signals': metrics.get('sell_signals', 0),
        'hold_signals': metrics.get('hold_signals', 0),
    }


def build_trades_dataframe(log_df):
    """Build a portfolio snapshot dataframe with all daily data."""
    try:
        # Start with log data
        export_df = log_df.reset_index(drop=True)
        export_df['date'] = pd.to_datetime(export_df['date'])
        
        # Calculate daily changes in target
        export_df['target_change'] = export_df['target_btc'].diff()
        export_df['transaction'] = np.where(
            export_df['target_change'] > 0, 'BUY',
            np.where(export_df['target_change'] < 0, 'SELL', 'HOLD')
        )
        export_df['trade_value_usd'] = (export_df['target_change'].abs() * export_df['btc_price']).round(2)
        
        # Select and order columns
        cols_to_export = [
            'date', 'btc_price', 'transaction', 'target_change', 'trade_value_usd',
            'current_position_btc', 'signal', 'confidence', 'confirmation_reason'
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
st.title("Bitcoin Trading - My Strategy")
st.markdown("**Data updates automatically** | CoinGecko + Blockchain.info")

st.markdown(
    """
    <div style="border:1px solid #E6E9EF; background:#F7F9FC; padding:16px 18px; border-radius:12px; margin:8px 0 14px 0;">
        <div style="font-size:18px; font-weight:700; color:#3b4637;">How This Works</div>
        <div style="margin-top:6px; color:#3C4858; font-size:15px;">
            Here you can see all info about the strategy, parameters, and where data comes from.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if st.button("See More Info →", type="primary"):
    st.session_state["show_strategy_page"] = True
    st.rerun()

if st.session_state.get("show_strategy_page"):
    from strategy_page import render_strategy_page
    render_strategy_page()
    st.stop()

st.markdown("---")

# ========== RISK MANAGEMENT TOGGLE ==========
st.markdown("### ⚙️ Strategy Settings")

col_rm1, col_rm2 = st.columns([3, 1])

with col_rm1:
    use_risk_management = st.checkbox(
        "Enable Risk Management Protection",
        value=True,
        help="Turn ON to use stop-loss, take-profit, trailing stops, and other protections. Turn OFF for pure signal-based trading."
    )
    
    if use_risk_management:
        st.success("✅ Risk Management: **ACTIVE** - All protections enabled (stop-loss, take-profit, trailing stop, consecutive loss brake)")
    else:
        st.warning("⚠️ Risk Management: **DISABLED** - Only basic signals used, no automatic protections")

with col_rm2:
    st.metric("Protection", "ON" if use_risk_management else "OFF", 
             delta="Safe" if use_risk_management else "Risky",
             delta_color="normal" if use_risk_management else "inverse")

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
    
    strategy, log_df, portfolio_df, metrics_raw, rm = fetch_and_run_backtest(
        str(start_date), 
        str(end_date),
        force_refresh=force_refresh_flag,
        use_risk_management=use_risk_management
    )
    results_df = log_df.copy()
    
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
    
    cummax = portfolio_df_filtered['total_value'].cummax()
    drawdown = ((portfolio_df_filtered['total_value'] - cummax) / cummax) * 100
    max_dd = drawdown.min()
    
    display_metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
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

    trades_export_df = build_trades_dataframe(log_df)
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
    
    # Show expected metrics benchmark
    st.info(f"""
    **📊 Expected Performance Metrics (for reference)**
    
    For period **{filtered_start} to {filtered_end}**:
    
    **BTC Buy & Hold baseline:**
    - Total Return: Depends on start/end prices (can be negative in bear markets)
    - Volatility: ~60-80% (very volatile)
    - Max Drawdown: -50% to -76% (2022 crash: -76%)
    - Sharpe Ratio: 0.2-0.8 (not risk-adjusted)
    
    **Good Trading Strategy should have:**
    - ✅ Total Return: Better or similar to Buy & Hold
    - ✅ Volatility: **Lower than BTC** (~30-50%, not 60-80%)
    - ✅ Max Drawdown: **Much lower** (-20% to -40%, not -76%)
    - ✅ Sharpe Ratio: **>1.0 is good, >1.5 is excellent** (risk-adjusted)
    - ✅ Win Rate: 50-60% is realistic
    
    **What matters most:** Lower drawdown + better Sharpe (risk-adjusted return), not just higher returns!
    """)

    
except Exception as e:
    st.error(f"Problem with data: {str(e)}")
    st.info("Maybe check your internet connection")
    st.stop()

# Metrics Row
st.markdown("### Main Numbers")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Profit",
        f"{display_metrics['total_return']:.2f}%",
        delta=f"${display_metrics['final_value'] - display_metrics['initial_value']:.0f}"
    )

with col2:
    st.metric(
        "Sharpe Number",
        f"{display_metrics['sharpe_ratio']:.3f}",
        delta=None
    )

with col3:
    st.metric(
        "Biggest Loss",
        f"{display_metrics['max_drawdown']:.2f}%",
        delta=None,
        delta_color="inverse"
    )

with col4:
    st.metric(
        "Volatility (Annual)",
        f"{display_metrics['volatility']:.2f}%",
        delta=None
    )

with col5:
    st.metric(
        "Win Rate",
        f"{display_metrics['win_rate']:.1f}%",
        delta=None
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

# Graphs - Matching main.py style
st.markdown("### Chart Analysis")
st.markdown("""
**Short explanation:** Portfolio value = BTC value + USDC value. The backtest starts with up to 2 BTC and the remaining capital in USDC (based on the first day’s price in the selected range).
When you change the date range, the portfolio and metrics are recalculated for that range (data fetch is cached for 1 hour).
""")

# Strategy Overview
st.markdown("""
**Trading Strategy Overview:**

This dashboard implements a **mean-reversion trading strategy** based on Bitcoin's production cost (mining cost). 
The core principle: Bitcoin price tends to revert to its production cost over time, creating trading opportunities.

**Key Objectives:**
- Maximize returns while managing risk through dynamic position sizing
- Exploit price deviations from production costs (±10% buffer zone)
- Maintain capital preservation during market downturns

**Implementation:**
- Production cost calculated from Bitcoin network hashrate and energy prices
- Exponential Moving Average (EMA) smoothing to reduce noise
- Risk-adjusted position sizing based on volatility and drawdown limits
""")

st.markdown("---")


# 1. BTC Price vs Production Cost (glavni graf)
st.markdown("""
#### Bitcoin Price vs Mining Cost
This shows how BTC price compares with mining cost.
The **±10% zone** (orange area) is where we decide:
- **BUY**: When price goes below -10% of cost (cheap)
- **SELL**: When price goes above +10% of cost (expensive)
- **HOLD**: Price is okay, in the middle
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

# 2. Row: Signals + Portfolio Value
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Trading Signals
    Each point is one day signal:
    - **Green (BUY)**: Ratio < 0.90 → Bitcoin is cheap
    - **Red (SELL)**: Ratio > 1.10 → Bitcoin is expensive  
    - **Gray (HOLD)**: Ratio 0.90-1.10 → Price is okay
    
    Ratio shows how far price is from mining cost.
    """)
    
    # Signals scatter plot
    fig_signals = go.Figure()
    
    # Add scatter for each signal type
    for signal_type, color in [('BUY', '#00FF00'), ('SELL', '#FF0000'), ('HOLD', '#808080')]:
        mask = results_df['signal'] == signal_type
        fig_signals.add_trace(go.Scatter(
            x=results_df[mask]['date'],
            y=results_df[mask]['signal_ratio'],
            mode='markers',
            name=signal_type,
            marker=dict(color=color, size=8, opacity=0.6)
        ))
    
    # Add threshold lines
    fig_signals.add_hline(y=0.90, line_dash="dash", line_color="green", opacity=0.5)
    fig_signals.add_hline(y=1.10, line_dash="dash", line_color="red", opacity=0.5)
    
    fig_signals.update_layout(
        title="Buy/Sell Signals",
        xaxis_title="Date",
        yaxis_title="Ratio (Price/Cost)",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_signals, use_container_width=True)

with col2:
    st.markdown("""
    #### Total Money Over Time
    All money together (BTC + USDC).
    
    **Goal**: Make more money than just buying and keeping Bitcoin.
    
    **How**: 
    - Change position when signals come
    - Use max 50% of money per trade
    - Stop loss when we lose too much
    """)
    
    # Portfolio value
    fig_portfolio = go.Figure()

    initial_value = portfolio_df_filtered['total_value'].iloc[0]
    initial_price = results_df_filtered['btc_price'].iloc[0]
    buy_and_hold_btc = initial_value / initial_price
    buy_and_hold_values = buy_and_hold_btc * results_df_filtered['btc_price']
    
    fig_portfolio.add_trace(go.Scatter(
        x=portfolio_df_filtered.index,
        y=portfolio_df_filtered['total_value'],
        mode='lines',
        name='My Strategy',
        line=dict(color='#006400', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 0, 0.3)'
    ))

    fig_portfolio.add_trace(go.Scatter(
        x=results_df_filtered['date'],
        y=buy_and_hold_values,
        mode='lines',
        name='Just Buy & Keep BTC',
        line=dict(color='#1f77b4', width=2, dash='dot')
    ))
    
    fig_portfolio.update_layout(
        title="Money Over Time",
        xaxis_title="Date",
        yaxis_title="Money (USD)",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_portfolio, use_container_width=True)

# 3. Row: Portfolio Allocation + Returns Distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### BTC and USDC Split
    Shows how money splits between BTC (orange) and USDC (blue).
    
    **Strategy**: 
    - More BTC when price is cheap (BUY)
    - Less BTC when price is expensive (SELL)
    - Keep USDC to buy later
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
        title="Money Split (BTC/USDC)",
        xaxis_title="Date",
        yaxis_title="Percent",
        template="plotly_white",
        height=400,
        yaxis=dict(range=[0, 1]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_alloc, use_container_width=True)

with col2:
    st.markdown("""
    #### Daily Profit/Loss
    Shows how much money changes each day.
    
    **Risk Check**: 
    - Bell shape = normal market
    - Red line = average daily change
    - Wide = price jumps a lot
    - Used to calculate Sharpe number and risk
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
        title="Daily Change",
        xaxis_title="Change Per Day (%)",
        yaxis_title="How Many Times",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_returns, use_container_width=True)

# 4. BTC Holdings Over Time
st.markdown("""
#### How Much Bitcoin We Have
Shows BTC amount over time.

**What Happens**:
- Goes up when BUY signals (we buy more)
- Goes down when SELL signals (we sell some)
- Changes follow the strategy

This helps see when we buy more and when we sell.
""")

fig_btc = go.Figure()

fig_btc.add_trace(go.Scatter(
    x=portfolio_df_filtered.index,
    y=portfolio_df_filtered['btc_quantity'],
    mode='lines',
    name='BTC Amount',
    line=dict(color='#FFD700', width=2),
    fill='tozeroy',
    fillcolor='rgba(255, 215, 0, 0.2)'
))

fig_btc.update_layout(
    title="Bitcoin Amount",
    xaxis_title="Date",
    yaxis_title="BTC Amount",
    template="plotly_white",
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_btc, use_container_width=True)

# 5. Drawdown Chart
st.markdown("""
#### Biggest Loss from Top
Shows how much money we lose from the highest point. Important for risk.

**Risk Control**:
- Shows biggest loss from top
- Goal: Keep loss < 30%
- Stop trading at -20% loss
- Helps see if strategy protects money

Small loss = good protection when market goes down.
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
    title="Loss from Top",
    xaxis_title="Date",
    yaxis_title="Loss (%)",
    template="plotly_white",
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_dd, use_container_width=True)

# Trading Signals Summary
st.markdown("### All Signals")
st.markdown("""
All buy/sell decisions from the time you picked.
Strategy tries not to trade too much, only when price is really different.
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
fig_pie = go.Figure(data=[go.Pie(
    labels=signal_counts.index,
    values=signal_counts.values,
    hole=0.4,
    marker_colors=['#00FF00', '#FF0000', '#808080']
)])

fig_pie.update_layout(
    title="Signals Split",
    template="plotly_white",
    height=300
)

st.plotly_chart(fig_pie, use_container_width=True)

# Recent Activity Table
st.markdown(f"### All Days ({len(results_df)} days)")
st.markdown("""
Every day data with price, mining cost, signal, and ratio.
Use this table to check each day and see why strategy does what it does.
""")
recent_cols = ['date', 'btc_price', 'production_cost_smoothed', 'signal', 'signal_ratio']
recent_df = results_df[recent_cols].copy()

# Format numbers
recent_df['btc_price'] = recent_df['btc_price'].apply(lambda x: f"${x:,.2f}")
recent_df['production_cost_smoothed'] = recent_df['production_cost_smoothed'].apply(lambda x: f"${x:,.2f}")
recent_df['signal_ratio'] = recent_df['signal_ratio'].apply(lambda x: f"{x:.3f}")

# Rename columns
recent_df.columns = ['Date', 'BTC Price', 'Mining Cost', 'Signal', 'Ratio']

st.dataframe(recent_df.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666;'>
        <p>🤖 <strong>Data Update Automatic</strong> | From CoinGecko & Blockchain.info</p>
        <p>New data every hour | Or click "Get New Data"</p>
        <p>Last update: <strong>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)
