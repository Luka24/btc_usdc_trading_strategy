# BTC/USDC Trading Strategy (Production Cost)

## Overview
Automated BTC/USDC strategy driven by Bitcoin production cost and risk controls. Core components:
- Production cost model (dynamic electricity + network-average efficiency)
- Signal generation (price / cost ratio)
- Portfolio rebalancing (BTC/USDC allocation table)
- Backtest engine with risk management

## What is current (2026-02)
- **Production cost model validated** against real mining costs (2016–2026):
    - Correlation $R = 0.990$, $R^2 = 0.980$
    - Average accuracy ~76% for 2024–2026
- **Dynamic parameters** (historical, not static):
    - Electricity prices: step function by year (2016–2026)
    - Network-average miner efficiency: step function by year (2016–2026)
- **All-in cost formula**:
    - Total cost = Energy cost × OVERHEAD_FACTOR (default 1.40)
    - Block reward is date-aware (halving schedule)

## Quick start
- Run the full backtest: `python main.py`
- Run production cost validation:
    - `python detailed_cost_analysis.py`
    - `python analyze_cost_accuracy.py`
    - `python show_cost_comparison.py`

## Key files
- config.py — all parameters (signals, portfolio, risk, cost model)
- production_cost.py — production cost calculation and time series
- backtest.py — backtest engine (signals + portfolio)
- main.py — orchestration + report + plots
- data_fetcher.py — data from CoinGecko + Blockchain.com
- results/ — backtest outputs and reports

## Production cost model (current)
- Dynamic electricity price and network-average efficiency per year
- Date-aware block reward (halvings)
- Energy cost derived from network hashrate and efficiency
- Total cost = Energy cost × OVERHEAD_FACTOR (1.40)

## Signal logic (current defaults)
- BUY when ratio < 0.90
- HOLD when 0.90 ≤ ratio ≤ 1.10
- SELL when ratio > 1.10

## Portfolio allocation (current defaults)
- Ratio < 0.85 → 100% BTC
- 0.85–0.95 → 70% BTC
- 0.95–1.05 → 50% BTC
- 1.05–1.20 → 30% BTC
- > 1.20 → 5% BTC

## Data Sources

### Electricity Prices
**Source**: US Energy Information Administration (EIA)  
**URL**: https://www.eia.gov/electricity/monthly/  
**Data**: Average Retail Price of Electricity to Ultimate Customers

**Historical Data Used**:
```
2015: $0.0429/kWh    2020: $0.0427/kWh (COVID dip)
2016: $0.0411/kWh    2021: $0.0540/kWh (Energy crisis)
2017: $0.0430/kWh    2022: $0.0734/kWh (Peak crisis)
2018: $0.0435/kWh    2023: $0.0569/kWh (Stabilizing)
2019: $0.0432/kWh    2024: $0.0459/kWh (Recent)
2025: $0.0475/kWh (Forecast)    2026: $0.0490/kWh (Forecast)
```

**Why US Data**: Post-China ban (2021), US became largest mining hub (~35% global hashrate). Most transparent and reliable official pricing data. Linear interpolation between years.

**Regional Context**:
- Texas (20-25% US mining): $0.035-0.045/kWh
- Appalachia (15-20%): $0.04-0.06/kWh
- Pacific NW (10-15%): $0.08-0.12/kWh

### Miner Hardware Efficiency
**Source**: Bitmain AntMiner Official Specifications  
**URL**: https://www.bitmain.com/products/hardware  
**Secondary**: Bitcoin Magazine, ASIC Miner Value (https://asicminervalue.com/)

**Hardware Evolution Timeline**:
```
2015-2016: AntMiner S7 (500 J/TH), S9 (300 J/TH)
2017-2018: AntMiner S9i (250 J/TH), S11 (100 J/TH)
2019:      AntMiner S17 (55 J/TH)
2020-2021: AntMiner S19 (50 J/TH), S19 Pro (40 J/TH)
2022:      AntMiner S19 XP (32 J/TH)
2023-2024: AntMiner S21 (21 J/TH), S21 Pro (21 J/TH)
2025-2026: Next Gen (18 J/TH), (15 J/TH predicted)
```

**Interpolation Method**: Exponential (Moore's Law)  
- Efficiency improves ~2× every 2 years
- Matches semiconductor physics (ASIC chip improvements)
- Historical validation: 500 J/TH (2015) → 21 J/TH (2024) = ~24× in 9 years
- Formula: $E_t = E_0 \times (E_1/E_0)^{(t-t_0)/(t_1-t_0)}$

### Bitcoin Network Data
**Hashrate**: Blockchain.com API  
**Price**: CoinGecko API (free tier, no auth required)  
**Block Reward**: Bitcoin Protocol specification  
**Blocks/Day**: 144 (10 min target, ±1% modern era)

**Halving Schedule**:
```
2012-11-28: 50 → 25 BTC/block
2016-07-09: 25 → 12.5 BTC/block
2020-05-11: 12.5 → 6.25 BTC/block
2024-04-20: 6.25 → 3.125 BTC/block
2028-04-20: 3.125 → 1.5625 BTC/block (predicted)
```

## Production Cost Calculation

### Formula
```python
# Daily energy consumption
hashes_per_day = hashrate_EH * 1e18 * 86400
energy_kwh = (hashes_per_day * efficiency_J_TH / 1e12) / 3.6e6

# Cost per BTC
btc_per_day = 144 * block_reward
energy_cost_per_btc = (energy_kwh * electricity_price) / btc_per_day
total_cost_per_btc = energy_cost_per_btc * OVERHEAD_FACTOR
```

### Model Parameters (Updated February 2026)

**Overhead Factor**: 1.50 (increased from 1.40)
- Electricity: 67% of total cost
- Other costs: 33% (CAPEX, cooling, facilities, personnel, maintenance)

**Network Average Efficiency** (J/TH by year):
```
2016: 250    2020: 60     2024: 32
2017: 150    2021: 50     2025: 28
2018: 100    2022: 45     2026: 26
2019: 55     2023: 38
```

**Electricity Prices** ($/kWh by year):
```
2016: 0.08    2020: 0.065    2024: 0.046
2017: 0.075   2021: 0.054    2025: 0.048
2018: 0.07    2022: 0.073    2026: 0.049
2019: 0.068   2023: 0.057
```

### Validation Results (Real vs Model Costs)

**Best Matches** (< 10% error):
- 2022-07: +0.3% ✓ **Near perfect**
- 2019-01: +5.7% ✓
- 2019-07: +5.8% ✓
- 2020-01: +7.6% ✓
- 2021-01: -6.0% ✓
- 2024-01: +7.5% ✓
- 2024-05: +9.3% ✓

**Acceptable Matches** (10-30% error):
- 2017-12: -22.2%
- 2023-01: +21.3%
- 2025-01: +21.5%
- 2026-01: +30.0%

**Overall Performance**:
- Correlation: R = 0.990, R² = 0.980
- Average accuracy: ~76% for 2024-2026 period
- Data quality: ±15-25% accuracy (sufficient for trading strategy)

**Known Limitations**:
- 2020-07 to 2020-12: Overestimation during COVID disruption
- Late 2024-2025: Overestimation post-halving (market adjustments)
- Regional variations not captured (US average used)
- Hashrate estimated from difficulty, not direct measurement

## Documentation
- 00_START_HERE.md — fast intro
- INDEX.md — navigation
- strategy_notebook.ipynb — interactive analysis
