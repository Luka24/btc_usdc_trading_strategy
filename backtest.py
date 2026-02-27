import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
from enum import Enum

from production_cost import ProductionCostSeries, get_block_reward_for_date
from portfolio import PortfolioManager
from risk_manager import RiskManager
from config import SignalConfig, PortfolioConfig


class Signal(Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class BacktestEngine:
    """Backtest engine implementing the professional combined strategy."""

    def __init__(self, initial_capital=100_000, enable_risk_management=True):
        self.initial_capital = initial_capital
        self.cost_series = ProductionCostSeries(ema_window=SignalConfig.COST_EMA_WINDOW)
        self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
        self.risk_manager = RiskManager()

        self.enable_risk_management = enable_risk_management
        self.backtest_data = pd.DataFrame()

    def _signal_weight_from_ratio(self, ratio_ema: float) -> float:
        for min_ratio, max_ratio, weight in SignalConfig.POSITION_TABLE:
            if min_ratio <= ratio_ema < max_ratio:
                return weight
        return SignalConfig.POSITION_TABLE[-1][2]

    def _signal_label_from_weight(self, current_w: float, target_w: float) -> Signal:
        if target_w > current_w:
            return Signal.BUY
        if target_w < current_w:
            return Signal.SELL
        return Signal.HOLD

    def _recompute_emas(self) -> None:
        self.backtest_data["price_ema"] = (
            self.backtest_data["btc_price"].ewm(span=SignalConfig.PRICE_EMA_WINDOW, adjust=False).mean()
        )
        self.backtest_data["production_cost_smoothed"] = (
            self.backtest_data["production_cost"].ewm(span=SignalConfig.COST_EMA_WINDOW, adjust=False).mean()
        )
        self.backtest_data["signal_ratio"] = (
            self.backtest_data["price_ema"] / self.backtest_data["production_cost_smoothed"]
        )
        # Step-function lookup on the ratio
        raw_weight = self.backtest_data["signal_ratio"].apply(self._signal_weight_from_ratio)
        # Smooth the hard band jumps so position ramps gradually between levels
        self.backtest_data["signal_weight"] = (
            raw_weight.ewm(span=SignalConfig.SIGNAL_EMA_WINDOW, adjust=False).mean()
        )
        # 200-day trend filter EMA
        self.backtest_data["trend_ema"] = (
            self.backtest_data["btc_price"].ewm(span=PortfolioConfig.TREND_FILTER_WINDOW, adjust=False).mean()
        )
        # RSI (14-day Wilder method via EWM)
        delta = self.backtest_data["btc_price"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=PortfolioConfig.RSI_WINDOW, adjust=False).mean()
        avg_loss = loss.ewm(span=PortfolioConfig.RSI_WINDOW, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.backtest_data["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
        # Hash Ribbon: fast/slow SMA of hashrate
        hr = self.backtest_data["hashrate"].astype(float)
        fast = PortfolioConfig.HASH_RIBBON_FAST
        slow = PortfolioConfig.HASH_RIBBON_SLOW
        self.backtest_data["hr_fast"] = hr.rolling(fast, min_periods=fast).mean()
        self.backtest_data["hr_slow"] = hr.rolling(slow, min_periods=slow).mean()

    def add_daily_data(self, date, btc_price, hashrate_eh_per_s,
                       _skip_recompute: bool = False):
        block_reward = get_block_reward_for_date(date)
        cost_data = self.cost_series.add_daily_data(date, hashrate_eh_per_s)

        row = {
            "date": pd.to_datetime(date),
            "btc_price": float(btc_price),
            "hashrate": float(hashrate_eh_per_s),
            "production_cost": float(cost_data["total_cost"]),
            "production_cost_smoothed": float(cost_data["total_cost"]),
            "price_ema": float(btc_price),
            "signal_ratio": 0.0,
            "signal_weight": 0.5,
            "signal": Signal.HOLD.value,
            "block_reward": block_reward,
            "electricity_price": cost_data["electricity_price"],
            "miner_efficiency": cost_data["miner_efficiency"],
            "risk_mode": "NORMAL",
            "dd_pct": 0.0,
            "vol_pct": 0.0,
            "var_pct": 0.0,
            "mode_cap": 1.0,
            "w_signal": 0.5,
            "w_final_target": 0.5,
            "w_final_exec": 0.5,
            "recovery_counter": 0,
            "trade_executed": False,
            "trade_size_weight": 0.0,
            "trend_ema": float(btc_price),
            "rsi": 50.0,
        }

        self.backtest_data = pd.concat([self.backtest_data, pd.DataFrame([row])], ignore_index=True)
        if not _skip_recompute:
            self._recompute_emas()

    def add_from_dataframe(self, df):
        """Batch-load a DataFrame, O(n) – builds row list first, concat once."""
        rows = []
        for _, row in df.iterrows():
            block_reward = get_block_reward_for_date(row["date"])
            cost_data = self.cost_series.add_daily_data(row["date"], row["hashrate_eh_per_s"])
            rows.append({
                "date": pd.to_datetime(row["date"]),
                "btc_price": float(row["btc_price"]),
                "hashrate": float(row["hashrate_eh_per_s"]),
                "production_cost": float(cost_data["total_cost"]),
                "production_cost_smoothed": float(cost_data["total_cost"]),
                "price_ema": float(row["btc_price"]),
                "signal_ratio": 0.0,
                "signal_weight": 0.5,
                "signal": Signal.HOLD.value,
                "block_reward": block_reward,
                "electricity_price": cost_data["electricity_price"],
                "miner_efficiency": cost_data["miner_efficiency"],
                "risk_mode": "NORMAL",
                "dd_pct": 0.0,
                "vol_pct": 0.0,
                "var_pct": 0.0,
                "mode_cap": 1.0,
                "w_signal": 0.5,
                "w_final_target": 0.5,
                "w_final_exec": 0.5,
                "recovery_counter": 0,
                "trade_executed": False,
                "trade_size_weight": 0.0,
                "trend_ema": float(row["btc_price"]),
                "rsi": 50.0,
            })
        self.backtest_data = pd.DataFrame(rows)
        # Single O(n) EMA pass for the full dataset
        self._recompute_emas()

    def get_portfolio_value(self, btc_price):
        portfolio = self.portfolio_manager.calculate_portfolio_value(
            btc_price,
            self.portfolio_manager.btc_quantity,
        )
        return portfolio["total_value"]

    def run_backtest(self, initial_btc_quantity=2.0):
        if self.backtest_data.empty:
            return self.portfolio_manager.get_portfolio_dataframe()

        first_price = self.backtest_data.iloc[0]["btc_price"]
        self.portfolio_manager.initialize_holdings(first_price, initial_btc_quantity)

        prev_nav = None
        prev_price: float | None = None
        _btc_returns: deque = deque(maxlen=PortfolioConfig.VOL_SCALING_WINDOW)

        for idx, row in self.backtest_data.iterrows():
            date = row["date"]
            price = float(row["btc_price"])

            # track daily returns for vol scaling
            if prev_price is not None:
                _btc_returns.append((price - prev_price) / prev_price)
            prev_price = price
            ratio_ema = float(row["signal_ratio"])

            nav_before = self.get_portfolio_value(price)
            daily_return = 0.0 if prev_nav is None or prev_nav == 0 else (nav_before - prev_nav) / prev_nav

            w_signal = float(row["signal_weight"])

            # ── RSI overlay ──────────────────────────────────────────────────
            rsi = float(row["rsi"])
            rsi_mult = PortfolioConfig.RSI_BOOST if rsi < PortfolioConfig.RSI_OVERSOLD else 1.0
            w_signal = float(np.clip(w_signal * rsi_mult, 0.0, 1.0))
            # ─────────────────────────────────────────────────────────────────

            # ── Hash Ribbon overlay (miner capitulation = hr_fast < hr_slow) ──
            if PortfolioConfig.HASH_RIBBON_ENABLED:
                hr_fast_val = float(row["hr_fast"])
                hr_slow_val = float(row["hr_slow"])
                if hr_fast_val < hr_slow_val:   # capitulation: miners exiting network
                    w_signal = float(np.clip(w_signal * PortfolioConfig.HASH_RIBBON_CAP_MULT, 0.0, 1.0))
            # ─────────────────────────────────────────────────────────────────

            # ── Vol scaling (target annualised volatility) ─────────────────
            if PortfolioConfig.VOL_TARGET > 0 and len(_btc_returns) >= 5:
                realized_vol = float(np.std(_btc_returns, ddof=0)) * np.sqrt(252)
                if realized_vol > 0:
                    raw_scalar = PortfolioConfig.VOL_TARGET / realized_vol
                    vol_scalar = float(np.clip(raw_scalar, PortfolioConfig.VOL_SCALE_MIN, PortfolioConfig.VOL_SCALE_MAX))
                else:
                    vol_scalar = 1.0
            else:
                vol_scalar = 1.0
            w_signal = float(np.clip(w_signal * vol_scalar, 0.0, 1.0))
            # ─────────────────────────────────────────────────────────────────

            # ── Trend filter (200-day EMA regime) ───────────────────────────
            trend_ema = float(row["trend_ema"])
            trend_cap = PortfolioConfig.TREND_BEAR_CAP if price < trend_ema else 1.0
            # ─────────────────────────────────────────────────────────────────

            if self.enable_risk_management:
                risk = self.risk_manager.evaluate(nav=nav_before, daily_return=daily_return)
                mode_cap = float(risk["mode_cap"])
            else:
                risk = {
                    "mode": "NORMAL",
                    "dd": 0.0,
                    "vol": 0.0,
                    "var_loss_pct": 0.0,
                    "recovery_counter": 0,
                }
                mode_cap = 1.0

            w_final_target = min(w_signal, mode_cap, trend_cap)

            current_portfolio = self.portfolio_manager.calculate_portfolio_value(price, self.portfolio_manager.btc_quantity)
            current_weight = current_portfolio["btc_weight"]
            delta = w_final_target - current_weight

            if abs(delta) < PortfolioConfig.MIN_REBALANCE_THRESHOLD:
                w_exec = current_weight
                trade_executed = False
            else:
                step = np.sign(delta) * min(abs(delta), PortfolioConfig.MAX_DAILY_WEIGHT_CHANGE)
                w_exec = float(np.clip(current_weight + step, 0.0, 1.0))
                trade_executed = True

            trade_size_weight = abs(w_exec - current_weight)
            signal = self._signal_label_from_weight(current_weight, w_exec)

            self.portfolio_manager.execute_rebalance(price, w_exec)

            nav_after = self.get_portfolio_value(price)
            prev_nav = nav_after

            self.portfolio_manager.add_to_history(
                date.strftime("%Y-%m-%d"),
                price,
                self.portfolio_manager.btc_quantity,
                ratio_ema,
                signal.value,
            )

            self.backtest_data.at[idx, "signal"] = signal.value
            self.backtest_data.at[idx, "risk_mode"] = risk["mode"]
            self.backtest_data.at[idx, "dd_pct"] = risk["dd"] * 100
            self.backtest_data.at[idx, "vol_pct"] = risk["vol"] * 100
            self.backtest_data.at[idx, "var_pct"] = risk["var_loss_pct"] * 100
            self.backtest_data.at[idx, "mode_cap"] = mode_cap
            self.backtest_data.at[idx, "w_signal"] = w_signal
            self.backtest_data.at[idx, "w_final_target"] = w_final_target
            self.backtest_data.at[idx, "w_final_exec"] = w_exec
            self.backtest_data.at[idx, "recovery_counter"] = risk["recovery_counter"]
            self.backtest_data.at[idx, "trade_executed"] = trade_executed
            self.backtest_data.at[idx, "trade_size_weight"] = trade_size_weight

        return self.portfolio_manager.get_portfolio_dataframe()

    def calculate_metrics(self):
        portfolio_df = self.portfolio_manager.get_portfolio_dataframe()
        metrics = self.portfolio_manager.calculate_metrics(portfolio_df)

        buy = (self.backtest_data["signal"] == "BUY").sum()
        sell = (self.backtest_data["signal"] == "SELL").sum()
        hold = (self.backtest_data["signal"] == "HOLD").sum()

        pdf = portfolio_df.copy()
        pdf["daily_return"] = pdf["total_value"].pct_change()
        wins = (pdf["daily_return"] > 0).sum()
        total_days = len(pdf) - 1
        win_rate = (wins / total_days * 100) if total_days > 0 else 0

        return {
            **metrics,
            "buy_signals": buy,
            "sell_signals": sell,
            "hold_signals": hold,
            "win_rate_pct": win_rate,
            "data_points": len(self.backtest_data),
        }

    def generate_report(self):
        metrics = self.calculate_metrics()
        pdf = self.portfolio_manager.get_portfolio_dataframe()

        return f"""
{'='*70}
BACKTEST REPORT: BTC/USDC Trading Strategy
{'='*70}

PERIOD
Start:              {pdf.index[0].strftime('%Y-%m-%d')}
End:                {pdf.index[-1].strftime('%Y-%m-%d')}
Days:                 {len(pdf)}

CAPITAL
Initial:             ${self.initial_capital:>15,.2f}
Final:               ${metrics['final_value']:>15,.2f}
Return:              {metrics['total_return_pct']:>15.2f}%

PERFORMANCE
Avg daily return:     {metrics['avg_daily_return_pct']:>12.2f}%
Volatility:           {metrics['daily_volatility_pct']:>12.2f}%
Sharpe:               {metrics['sharpe_ratio']:>12.2f}

RISK
Max drawdown:         {metrics['max_drawdown_pct']:>12.2f}%

SIGNALS
Buy:                  {metrics['buy_signals']:>15}
Sell:                 {metrics['sell_signals']:>15}
Hold:                 {metrics['hold_signals']:>15}
Total trades:         {metrics['num_trades']:>15}
Win rate:             {metrics['win_rate_pct']:>15.2f}%

FINAL ALLOCATION
BTC:                  {metrics['final_btc_weight']:>15.1%}
USDC:                 {1 - metrics['final_btc_weight']:>15.1%}

{'='*70}
"""

    def export_results(self, filename=None):
        import os

        os.makedirs("results", exist_ok=True)

        if not filename:
            filename = f"results/backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        self.backtest_data.to_csv(filename, index=False)
        print(f"Results exported to: {filename}")
