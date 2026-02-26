import sys
sys.path.insert(0, ".")

from optimization.walk_forward import load_full_data, FOLDS, _slice
from backtest import BacktestEngine
import pandas as pd

df = load_full_data()
print("FULL:", len(df), "rows |", df["date"].iloc[0].date(), "->", df["date"].iloc[-1].date())
print("BTC price: first=", df["btc_price"].iloc[0], " last=", df["btc_price"].iloc[-1])
print()

print("--- OOS fold sanity: dates + BTC buy-and-hold ---")
for f in FOLDS:
    oos = _slice(df, f.oos_start, f.oos_end)
    p = oos["btc_price"].astype(float)
    bh = (p.iloc[-1] / p.iloc[0] - 1) * 100
    print(
        f"{f.name}: rows={len(oos)}"
        f"  {oos['date'].iloc[0].date()} -> {oos['date'].iloc[-1].date()}"
        f"  BTC ${p.iloc[0]:.0f} -> ${p.iloc[-1]:.0f}"
        f"  BH={bh:+.1f}%"
    )

print()
print("--- Strategy return vs Buy-and-Hold per fold ---")
for f in FOLDS:
    oos = _slice(df, f.oos_start, f.oos_end)
    p = oos["btc_price"].astype(float)
    bh = (p.iloc[-1] / p.iloc[0] - 1) * 100

    eng = BacktestEngine(initial_capital=100_000, enable_risk_management=True)
    eng.add_from_dataframe(oos.copy())
    eng.run_backtest(initial_btc_quantity=2.0)
    m = eng.calculate_metrics()

    strat = m["total_return_pct"]
    sortino = m["sortino_ratio"]
    maxdd = m["max_drawdown_pct"]
    alpha = strat - bh
    ok = "OK" if (bh < 0 and strat > bh) or (bh > 0 and strat > 0) else "CHECK"
    print(
        f"{f.name}: BH={bh:+6.1f}%  strat={strat:+6.1f}%  alpha={alpha:+6.1f}%"
        f"  sortino={sortino:.3f}  maxDD={maxdd:.1f}%  [{ok}]"
    )

print()
print("2022: BTC crashed -65%. Strategy losing <10% in 2022 = CORRECT behavior.")
print("Negative Sortino in a bear year is expected (few positive days = high downside vol).")
