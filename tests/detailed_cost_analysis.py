"""
Dnevni izračun stroškov rudarjenja (API hashrate)
=================================================
Uporablja samo:
- data_fetcher (API hashrate)
- config (cena elektrike, učinkovitost, overhead, halving)
"""

import io
import os
from datetime import datetime

import pandas as pd

from data_fetcher import BlockchainFetcher
from config import HistoricalParameters, ProductionCostConfig


REAL_COSTS_CSV = """Datum,Ocena_Stroskov_USD
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


def get_block_reward_for_date(date_input) -> float:
    """Pridobi pravilno nagrado za blok glede na datum (halving schedule iz config)."""
    schedule = [
        (datetime.strptime(date_str, "%Y-%m-%d"), reward)
        for date_str, reward in ProductionCostConfig.HALVING_SCHEDULE
    ]
    dt = pd.Timestamp(date_input).to_pydatetime()

    for idx, (halving_date, reward) in enumerate(schedule):
        if dt < halving_date:
            if idx == 0:
                return ProductionCostConfig.PRE_HALVING_REWARD
            return schedule[idx - 1][1]
    return schedule[-1][1]


def calculate_daily_costs(start_date: str, end_date: str) -> pd.DataFrame:
    """Dnevni izračun stroškov iz API hashrate podatkov."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    days = (end - start).days + 1

    df_hashrate = BlockchainFetcher.fetch_hashrate(days=days)
    df_hashrate["date"] = pd.to_datetime(df_hashrate["date"])
    df_hashrate = df_hashrate.set_index("date").sort_index()

    daily_index = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame(index=daily_index)
    df = df.join(df_hashrate[["hashrate_eh_per_s"]], how="left")
    df["hashrate_eh_per_s"] = df["hashrate_eh_per_s"].interpolate(method="time")
    df["hashrate_eh_per_s"] = df["hashrate_eh_per_s"].ffill().bfill()
    df = df.reset_index().rename(columns={"index": "date"})

    if df.empty:
        raise RuntimeError("Ni podatkov za izbrano obdobje.")

    results = []
    for _, row in df.iterrows():
        date = row["date"]
        hashrate_eh_per_s = row["hashrate_eh_per_s"]

        electricity_price = HistoricalParameters.get_electricity_price(date)
        efficiency_j_th = HistoricalParameters.get_miner_efficiency(date)
        block_reward = get_block_reward_for_date(date)

        hashrate_h_per_s = hashrate_eh_per_s * 1e18
        total_hashes_per_day = hashrate_h_per_s * 86400
        energy_joules = (total_hashes_per_day * efficiency_j_th) / 1e12
        energy_kwh = energy_joules / 3.6e6

        btc_per_day = ProductionCostConfig.BLOCKS_PER_DAY * block_reward
        energy_cost_per_btc = (energy_kwh * electricity_price) / btc_per_day
        total_cost_per_btc = energy_cost_per_btc * ProductionCostConfig.OVERHEAD_FACTOR

        results.append(
            {
                "date": date,
                "hashrate_eh_per_s": hashrate_eh_per_s,
                "electricity_price": electricity_price,
                "miner_efficiency_j_th": efficiency_j_th,
                "block_reward": block_reward,
                "energy_cost_usd": energy_cost_per_btc,
                "total_cost_usd": total_cost_per_btc,
            }
        )

    return pd.DataFrame(results)


def load_real_costs() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(REAL_COSTS_CSV))
    df["date"] = pd.to_datetime(df["Datum"] + "-01")
    return df[["date", "Ocena_Stroskov_USD"]].rename(
        columns={"Ocena_Stroskov_USD": "real_cost_usd"}
    )


def compare_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    monthly = daily_df.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    model_monthly = (
        monthly.groupby("month")["total_cost_usd"].mean().reset_index()
        .rename(columns={"month": "date", "total_cost_usd": "model_cost_usd"})
    )

    real_df = load_real_costs()
    merged = pd.merge(real_df, model_monthly, on="date", how="inner")
    merged["diff_usd"] = merged["model_cost_usd"] - merged["real_cost_usd"]
    merged["diff_pct"] = (merged["diff_usd"] / merged["real_cost_usd"]) * 100
    return merged


def main() -> None:
    start_date = "2016-01-01"
    end_date = "2026-02-01"

    os.makedirs("results", exist_ok=True)

    print("[START] Dnevni izračun stroškov (API hashrate)")
    daily_df = calculate_daily_costs(start_date, end_date)

    daily_out = "results/mining_costs_daily.csv"
    daily_df.to_csv(daily_out, index=False)
    print(f"[OK] Dnevni podatki shranjeni: {daily_out}")

    comparison_df = compare_monthly(daily_df)
    comparison_out = "results/mining_cost_comparison.csv"
    comparison_df.to_csv(comparison_out, index=False)
    print(f"[OK] Primerjava (mesečno) shranjena: {comparison_out}")

    mae = comparison_df["diff_usd"].abs().mean()
    mape = comparison_df["diff_pct"].abs().mean()
    print(f"[SUMMARY] MAE: ${mae:,.0f} | MAPE: {mape:.2f}%")

    print("\nPrvih 5 vrstic:")
    print(comparison_df.head(5).to_string(index=False))
    print("\nZadnjih 5 vrstic:")
    print(comparison_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
