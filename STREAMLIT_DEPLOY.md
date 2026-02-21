# Streamlit Cloud Deployment Guide

## 🚀 Kako deployat na Streamlit Cloud

### 1. Pripravi GitHub repo (DONE ✓)
Tvoj code je že v GitHub repoju: `Luka24/btc_usdc_trading_strategy`

### 2. Pojdi na Streamlit Cloud
1. Pojdi na: https://share.streamlit.io
2. Sign in z GitHub accountom
3. Klikni "New app"

### 3. Deployment settings
```
Repository: Luka24/btc_usdc_trading_strategy
Branch: main
Main file path: btc_trading_strategy/dashboard.py
```

### 4. Advanced settings (optional)
- Python version: 3.10 ali 3.11
- Secrets: Ne potrebuješ (vse je public data)

### 5. Deploy! 🎉
Klikni "Deploy" in čakaj 2-3 minute da se builda.

---

## 📊 Dashboard Features

### ✅ Že implementirano:
- **Real-time data fetching** (CoinGecko & Yahoo Finance)
- **Multi-period backtests** (365d, 1491d, 3685d)
- **Performance metrics** (Sharpe, returns, drawdown)
- **Interactive charts** (Plotly)
- **Signal analysis** (trend, momentum, production cost)
- **Trade execution log**
- **Optimizirana strategija** (Sharpe 1.11 na 10 let!)

### 🎯 Rezultati strategije:
- **Bear market (1y)**: +38% outperformance vs Buy&Hold
- **Mixed market (4y)**: Sharpe 0.50 (odlično za choppy market!)
- **Bull market (10y)**: Sharpe 1.11 (TOP-TIER!) 🏆

---

## 🔧 Lokalno testiranje

```bash
cd btc_trading_strategy
streamlit run dashboard.py
```

Dashboard bo na: http://localhost:8501

---

## 📦 Requirements

Vsi packagei so že v `requirements.txt`:
- streamlit
- pandas
- numpy  
- plotly
- requests
- yfinance

---

## 🐛 Troubleshooting

**Problem**: "Module not found"
**Fix**: Preveri da so vsi filei v `btc_trading_strategy/` folderju

**Problem**: "No data"
**Fix**: Dashboard bo avtomatsko fetchal data iz API-jev

**Problem**: "Config errors"
**Fix**: ✓ Že fixed - `.streamlit/config.toml` je posodobljen

---

## 📱 Po deploymentu

Ko je deployment končan, dobiš URL tipa:
```
https://your-app-name.streamlit.app
```

Lahko ga share-aš!

---

## 🔄 Updates

Ko pushneš nove spremembe na GitHub (main branch), se dashboard avtomatsko rebuildा!

```bash
git add .
git commit -m "Update strategy parameters"
git push origin main
```

Streamlit Cloud bo avtomatsko detektiral spremembe in rebuildal app.

---

Enjoy! 🚀
