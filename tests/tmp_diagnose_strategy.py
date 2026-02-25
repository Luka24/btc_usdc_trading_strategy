from data_fetcher import DataFetcher
from backtest import BacktestEngine
import pandas as pd


def run(force_refresh: bool):
    data = DataFetcher.fetch_combined_data(days=1460, use_real_data=True, force_refresh=force_refresh)
    engine = BacktestEngine(initial_capital=100_000, enable_risk_management=True)
    engine.add_from_dataframe(data)
    engine.run_backtest(initial_btc_quantity=2.0)
    metrics = engine.calculate_metrics()
    bt = engine.backtest_data

    print(f"--- force_refresh={force_refresh} ---")
    unique_dates = pd.to_datetime(data['date']).nunique()
    print("rows", len(data), "unique_dates", unique_dates, "dup_dates", len(data) - unique_dates)
    print("range", data['date'].iloc[0], data['date'].iloc[-1])
    print("return", round(metrics['total_return_pct'], 2), "sharpe", round(metrics['sharpe_ratio'], 3), "maxDD", round(metrics['max_drawdown_pct'], 2))
    print("signal_ratio min/med/max", round(bt['signal_ratio'].min(), 3), round(bt['signal_ratio'].median(), 3), round(bt['signal_ratio'].max(), 3))
    print("w_signal dist", bt['w_signal'].value_counts().sort_index().to_dict())
    print("risk_mode dist", bt['risk_mode'].value_counts().to_dict())
    print("trades", int(bt['trade_executed'].sum()), "BUY", int((bt['signal'] == 'BUY').sum()), "SELL", int((bt['signal'] == 'SELL').sum()), "HOLD", int((bt['signal'] == 'HOLD').sum()))


if __name__ == "__main__":
    run(False)
    run(True)
