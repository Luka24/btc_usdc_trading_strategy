"""Compare model production costs vs provided real monthly estimates."""
import io
import pandas as pd
from data_fetcher import DataFetcher
from backtest import BacktestEngine

REAL_CSV = """Datum,Ocena_Stroskov_USD
2016-01,650
2016-02,680
2016-03,700
2016-04,720
2016-05,750
2016-06,780
2016-07,850
2016-08,820
2016-09,800
2016-10,850
2016-11,880
2016-12,920
2017-01,1050
2017-02,1100
2017-03,1200
2017-04,1350
2017-05,1500
2017-06,1800
2017-07,2100
2017-08,2500
2017-09,2800
2017-10,3200
2017-11,3500
2017-12,3844
2018-01,4200
2018-02,4800
2018-03,5200
2018-04,5500
2018-05,6100
2018-06,6500
2018-07,6800
2018-08,7100
2018-09,7000
2018-10,6500
2018-11,5800
2018-12,5200
2019-01,4500
2019-02,4200
2019-03,4400
2019-04,4800
2019-05,5200
2019-06,6500
2019-07,7100
2019-08,7300
2019-09,7200
2019-10,6800
2019-11,6500
2019-12,6900
2020-01,7500
2020-02,7800
2020-03,6500
2020-04,7200
2020-05,8500
2020-06,9200
2020-07,9500
2020-08,10500
2020-09,11200
2020-10,12500
2020-11,13800
2020-12,15200
2021-01,20500
2021-02,22000
2021-03,23500
2021-04,28000
2021-05,31500
2021-06,22000
2021-07,15500
2021-08,17000
2021-09,19500
2021-10,21000
2021-11,23500
2021-12,25000
2022-01,25500
2022-02,24500
2022-03,26000
2022-04,27500
2022-05,28000
2022-06,29500
2022-07,24500
2022-08,22000
2022-09,21500
2022-10,23000
2022-11,24000
2022-12,22500
2023-01,23000
2023-02,21500
2023-03,24500
2023-04,26000
2023-05,28500
2023-06,30000
2023-07,31500
2023-08,32000
2023-09,30500
2023-10,33000
2023-11,35000
2023-12,38000
2024-01,41500
2024-02,42000
2024-03,45000
2024-04,55000
2024-05,91000
2024-06,75000
2024-07,70000
2024-08,78000
2024-09,80500
2024-10,72000
2024-11,81000
2024-12,88000
2025-01,91500
2025-02,94000
2025-03,86000
2025-04,92000
2025-05,98000
2025-06,90000
2025-07,88000
2025-08,95000
2025-09,102000
2025-10,112000
2025-11,105000
2025-12,108000
2026-01,102000
"""


def load_real_series() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(REAL_CSV))
    df["date"] = pd.to_datetime(df["Datum"] + "-01")
    df = df[["date", "Ocena_Stroskov_USD"]].rename(
        columns={"Ocena_Stroskov_USD": "real_cost_usd"}
    )
    return df


def load_model_series() -> pd.DataFrame:
    data = DataFetcher.fetch_combined_data(days=3650, use_real_data=True)
    engine = BacktestEngine(initial_capital=100_000)
    engine.add_from_dataframe(data)
    engine.run_backtest(initial_btc_quantity=1.0)
    model_df = engine.backtest_data[["date", "production_cost"]].copy()
    model_df["date"] = pd.to_datetime(model_df["date"])
    model_df["month"] = model_df["date"].dt.to_period("M").dt.to_timestamp()
    model_monthly = (
        model_df.groupby("month")["production_cost"].mean().reset_index()
        .rename(columns={"month": "date", "production_cost": "model_cost_usd"})
    )
    return model_monthly


def compare_series() -> pd.DataFrame:
    real_df = load_real_series()
    model_df = load_model_series()

    merged = pd.merge(real_df, model_df, on="date", how="inner")
    merged["diff_usd"] = merged["model_cost_usd"] - merged["real_cost_usd"]
    merged["diff_pct"] = (merged["diff_usd"] / merged["real_cost_usd"]) * 100
    return merged


def print_summary(df: pd.DataFrame) -> None:
    mae = df["diff_usd"].abs().mean()
    mape = df["diff_pct"].abs().mean()
    max_over = df.loc[df["diff_usd"].idxmax()]
    max_under = df.loc[df["diff_usd"].idxmin()]

    print("\n" + "=" * 80)
    print("REAL VS MODEL COST COMPARISON")
    print("=" * 80)
    print(f"Rows compared: {len(df)}")
    print(f"MAE (USD): {mae:,.0f}")
    print(f"MAPE (%): {mape:.2f}%")
    print("\nLargest overestimate:")
    print(f"  {max_over['date'].strftime('%Y-%m')}: +${max_over['diff_usd']:,.0f} ({max_over['diff_pct']:.2f}%)")
    print("Largest underestimate:")
    print(f"  {max_under['date'].strftime('%Y-%m')}: ${max_under['diff_usd']:,.0f} ({max_under['diff_pct']:.2f}%)")


def main() -> None:
    df = compare_series()
    print_summary(df)

    # Save comparison CSV
    out_path = "results/real_vs_model_costs_monthly.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Print first/last 6 rows
    print("\nFirst 6 rows:")
    print(df.head(6).to_string(index=False))
    print("\nLast 6 rows:")
    print(df.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
