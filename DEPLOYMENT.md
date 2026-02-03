# 🚀 Deployment Navodila - BTC Trading Dashboard

## **LOKALNO TESTIRANJE** (najprej to naredi!)

### 1. Instaliraj Streamlit
```bash
pip install streamlit plotly
```

### 2. Poženi Dashboard
```bash
cd btc_trading_strategy
streamlit run dashboard.py
```

Dashboard se odpre na: **http://localhost:8501**

---

## **ZASTONJ HOSTING NA STREAMLIT CLOUD** ☁️

### Korak 1: Pripravi GitHub Repository

1. **Naredi GitHub račun** (če ga še nimaš): https://github.com
2. **Naredi nov repository** (lahko private ali public)
3. **Upload-aj te file:**
   - `dashboard.py`
   - `requirements.txt`
   - `results/` folder (z vsemi CSV files)

### Korak 2: Deploy na Streamlit Cloud

1. Pojdi na: **https://streamlit.io/cloud**
2. Klikni **"Sign up"** in se prijavi z GitHub računom
3. Klikni **"New app"**
4. Izberi:
   - **Repository**: tvoj GitHub repo
   - **Branch**: main
   - **Main file path**: `btc_trading_strategy/dashboard.py`
5. Klikni **"Deploy!"**

**✅ GOTOVO!** Tvoja stran bo dostopna na: `https://[tvoj-username]-btc-trading.streamlit.app`

---

## 🔄 Kako Posodobiš Dashboard

### Način 1: Ročno (enostavno)
1. Poženi `main.py` lokalno
2. Nov CSV file se ustvari v `results/`
3. Upload-aj nov CSV na GitHub (v `results/` folder)
4. Streamlit Cloud **avtomatsko** posodobi stran v ~1 minuti

### Način 2: GitHub Desktop (še lažje)
1. Namesti **GitHub Desktop**: https://desktop.github.com
2. Clone-aj tvoj repository
3. Ko poženeš `main.py`, se nov CSV ustvari
4. V GitHub Desktop klikni **"Commit"** in **"Push"**
5. Dashboard se avtomatsko posodobi

---

## 💡 POMEMBNO

- **Zastonj je do 1GB storage** (dovolj za ~10,000 CSV files!)
- **Dashboard se posodobi sam** ko upload-aš nove rezultate
- **Ni treba nič spreminjat** v kodi
- Lahko **deliš link** s komerkoli - javno dostopno

---

## 🎯 Kaj Dashboard Prikazuje

✅ **Portfolio Value** graf - kako raste/pada vrednost  
✅ **BTC Price vs Production Cost** - kdaj kupuješ/prodajaš  
✅ **Signal Ratio** - kako močan je signal  
✅ **Trading Signals** pie chart - koliko buy/sell/hold  
✅ **BTC Holdings** - koliko BTC imaš skozi čas  
✅ **Drawdown** - največji padci  
✅ **Zadnjih 20 dni** tabela  

---

## 🆘 Če Kaj Ne Dela

1. **Dashboard ne kaže podatkov?**
   - Preveri da je CSV file v `results/` folderju
   - Poženi `main.py` če še nimaš rezultatov

2. **Deployment failed?**
   - Preveri da je `requirements.txt` v root folderju
   - Preveri da je path do `dashboard.py` pravilen

3. **Grafi so prazni?**
   - CSV mora imeti pravilne column names (date, btc_price, portfolio_value, itd.)

---

## 📱 Bonus: Responsive Design

Dashboard dela tudi na **telefonu** in **tabletu** - lahko vidiš rezultate kjerkoli! 📊
