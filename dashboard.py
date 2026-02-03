"""
BTC Trading Strategy Dashboard - AUTO-UPDATING
================================================
Avtomatsko posodablja podatke preko API-ja in prikazuje grafe.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import glob
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from data_fetcher import DataFetcher
from backtest import BacktestEngine
from risk_manager import RiskManager
from config import BacktestConfig

# Page config
st.set_page_config(
    page_title="BTC Trading Strategy Dashboard - LIVE",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_and_run_backtest():
    """Fetch fresh data from APIs and run backtest."""
    with st.spinner('🔄 Fetching fresh data from APIs...'):
        # Fetch real data
        data = DataFetcher.fetch_combined_data(
            days=BacktestConfig.DAYS_TO_FETCH,
            use_real_data=True
        )
        
        # Run backtest
        engine = BacktestEngine(initial_capital=100_000)
        engine.add_from_dataframe(data)
        portfolio_df = engine.run_backtest(initial_btc_quantity=2.0)
        
        # Calculate metrics
        metrics = engine.calculate_backtest_metrics()
        
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
        'volatility': rm.calculate_volatility() * 100,
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

# Header
st.title("₿ BTC Trading Strategy Dashboard - LIVE")
st.markdown("**Avtomatsko posodabljanje podatkov iz API-jev** | CoinGecko + Blockchain.info")
st.markdown("---")

# Refresh button
col1, col2, col3 = st.columns([2, 1, 1])
with col2:
    if st.button("🔄 Refresh Data Now", type="primary"):
        st.cache_data.clear()
        st.rerun()

with col3:
    st.markdown(f"**Last update:** {datetime.now().strftime('%H:%M:%S')}")

# Fetch data and run backtest
try:
    engine, portfolio_df, metrics_raw, rm = fetch_and_run_backtest()
    results_df = engine.backtest_data
    display_metrics = calculate_display_metrics(engine, portfolio_df, metrics_raw, rm)
    
    st.success(f"✅ Data fetched successfully! {display_metrics['days']} days analyzed")
    
except Exception as e:
    st.error(f"❌ Error fetching data: {str(e)}")
    st.info("💡 Tip: Check your internet connection or API limits")
    st.stop()

# Metrics Row
st.markdown("### 📈 Ključne Metrike")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Return",
        f"{display_metrics['total_return']:.2f}%",
        delta=f"${display_metrics['final_value'] - display_metrics['initial_value']:.0f}"
    )

with col2:
    st.metric(
        "Sharpe Ratio",
        f"{display_metrics['sharpe_ratio']:.3f}",
        delta=None
    )

with col3:
    st.metric(
        "Max Drawdown",
        f"{display_metrics['max_drawdown']:.2f}%",
        delta=None,
        delta_color="inverse"
    )

with col4:
    st.metric(
        "Volatility (30d)",
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

# Graphs - Matching main.py style
st.markdown("### 📊 Graf Analiza (kot v main.py)")

# 1. BTC Price vs Production Cost (glavni graf)
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
    name='Cost (Raw)',
    line=dict(color='gray', width=1),
    opacity=0.5
))

fig_main.add_trace(go.Scatter(
    x=results_df['date'],
    y=results_df['production_cost_smoothed'],
    mode='lines',
    name='Cost (EMA Smoothed)',
    line=dict(color='#FF0000', width=2, dash='dash')
))

# Add ±10% buffer
upper_buffer = results_df['production_cost_smoothed'] * 1.1
lower_buffer = results_df['production_cost_smoothed'] * 0.9

fig_main.add_trace(go.Scatter(
    x=results_df['date'],
    y=upper_buffer,
    mode='lines',
    name='±10% Buffer',
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
    name='±10% Buffer',
    hoverinfo='skip'
))

fig_main.update_layout(
    title="BTC Price vs. Production Costs",
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
        title="Trading Signals",
        xaxis_title="Date",
        yaxis_title="Ratio (Price/Cost)",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_signals, use_container_width=True)

with col2:
    # Portfolio value
    fig_portfolio = go.Figure()
    
    fig_portfolio.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['total_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#006400', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 0, 0.3)'
    ))
    
    fig_portfolio.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value (USD)",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_portfolio, use_container_width=True)

# 3. Row: Portfolio Allocation + Returns Distribution
col1, col2 = st.columns(2)

with col1:
    # Portfolio allocation
    fig_alloc = go.Figure()
    
    fig_alloc.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['btc_weight'],
        mode='lines',
        name='BTC',
        line=dict(width=0),
        stackgroup='one',
        fillcolor='rgba(255, 165, 0, 0.7)'
    ))
    
    fig_alloc.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=1 - portfolio_df['btc_weight'],
        mode='lines',
        name='USDC',
        line=dict(width=0),
        stackgroup='one',
        fillcolor='rgba(0, 0, 255, 0.7)'
    ))
    
    fig_alloc.update_layout(
        title="Portfolio Allocation (BTC/USDC)",
        xaxis_title="Date",
        yaxis_title="Weight",
        template="plotly_white",
        height=400,
        yaxis=dict(range=[0, 1]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_alloc, use_container_width=True)

with col2:
    # Returns distribution
    returns = portfolio_df['total_value'].pct_change() * 100
    returns_clean = returns.dropna()
    
    fig_returns = go.Figure()
    
    fig_returns.add_trace(go.Histogram(
        x=returns_clean,
        nbinsx=50,
        name='Daily Returns',
        marker=dict(color='skyblue', line=dict(color='black', width=1)),
        opacity=0.7
    ))
    
    # Add mean line
    mean_return = returns_clean.mean()
    fig_returns.add_vline(x=mean_return, line_dash="dash", line_color="red", line_width=2)
    
    fig_returns.update_layout(
        title="Distribution of Daily Returns",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_returns, use_container_width=True)

# 4. BTC Holdings Over Time
fig_btc = go.Figure()

fig_btc.add_trace(go.Scatter(
    x=portfolio_df.index,
    y=portfolio_df['btc_quantity'],
    mode='lines',
    name='BTC Balance',
    line=dict(color='#FFD700', width=2),
    fill='tozeroy',
    fillcolor='rgba(255, 215, 0, 0.2)'
))

fig_btc.update_layout(
    title="BTC Holdings Over Time",
    xaxis_title="Date",
    yaxis_title="BTC Quantity",
    template="plotly_white",
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_btc, use_container_width=True)

# 5. Drawdown Chart
cummax = portfolio_df['total_value'].cummax()
drawdown = ((portfolio_df['total_value'] - cummax) / cummax) * 100

fig_dd = go.Figure()

fig_dd.add_trace(go.Scatter(
    x=portfolio_df.index,
    y=drawdown,
    mode='lines',
    name='Drawdown',
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
st.markdown("### 🎯 Trading Signali")

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
    title="Signal Distribution",
    template="plotly_white",
    height=300
)

st.plotly_chart(fig_pie, use_container_width=True)

# Recent Activity Table
st.markdown("### 📋 Zadnjih 20 Trading Dni")
recent_cols = ['date', 'btc_price', 'production_cost_smoothed', 'signal', 'signal_ratio']
recent_df = results_df[recent_cols].tail(20).copy()

# Format numbers
recent_df['btc_price'] = recent_df['btc_price'].apply(lambda x: f"${x:,.2f}")
recent_df['production_cost_smoothed'] = recent_df['production_cost_smoothed'].apply(lambda x: f"${x:,.2f}")
recent_df['signal_ratio'] = recent_df['signal_ratio'].apply(lambda x: f"{x:.3f}")

# Rename columns
recent_df.columns = ['Date', 'BTC Price', 'Production Cost', 'Signal', 'Ratio']

st.dataframe(recent_df.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666;'>
        <p>🤖 <strong>AUTO-UPDATING Dashboard</strong> | Data z CoinGecko & Blockchain.info API</p>
        <p>Podatki se avtomatsko osvežijo vsako uro | Ročno: klikni "🔄 Refresh Data Now"</p>
        <p>Zadnja posodobitev: <strong>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)
