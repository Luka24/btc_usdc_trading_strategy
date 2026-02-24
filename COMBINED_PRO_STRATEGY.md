# PROFESIONALNA BTC/USDC STRATEGIJA
## 1-Dnevni Vrh: Skupna Specifikacija (2 inputa, Midnight Trade)

---

## EXECUTIVE SUMMARY

BTC/USDC **mean-reversion strategija** temelji na razmerju med tržno ceno in stroški proizvodnje ($R_t = P_t/C_t$). Uporablja **samo 2 dnevna inputa** (cena BTC, stroški proizvodnje), trgovanje **1× dnevno ob 00:00 UTC**.

**Risk management sistem** z 3 neodvisnimi triggerji (Drawdown, Volatility, Value-at-Risk) dinamično omejuje izpostavljenost preko 4 risk modov (NORMAL/CAUTION/RISK_OFF/EMERGENCY).

**Cilj:** Dolgoročna akumulacija BTC z zaščito kapitala v kriznih obdobjih. Asimetrična zaščita: instant crisis entry, sticky recovery (3-7 dni).

**Značilnosti:**
- Valuation anchor: Production cost (fundamentalen ekonomski signal)
- Disciplina: Fixed execution time, no intraday emotion
- Professional risk: Rolling 252-day peak, multi-trigger system, hysteresis-based recovery

---

# DEL 0: ASSUMPTIONS & PARAMETERS RATIONALE

## 0.1 Parameter Justification

#### Signal Smoothing (EMA Windows)

**7-dnevna EMA za ceno (P_ema):**
- **Razlog:** Odstrani dnevni noise (flash crashes, weekend gaps), a še vedno hitro reagira na trende
- **Alternativa:** 5d bi bilo preveč noisy (false signals), 14d bi zamudilo entry signale
- **Tradeoff:** Lag ≈ 3-4 dni, sprejemljiv za mean-reversion system

**14-dnevna EMA za stroške (C_ema):**
- **Razlog:** Proizvodnja se spreminja počasi (hash rate adjustments, electricity contracts, mining difficulty)
- **Alternativa:** 7d bi overreagiral na kratkoročne nihaje, 30d bi bil preveč smooth
- **Tradeoff:** Ujame strukturne spremembe brez reaktivnosti na šum

#### Risk Management Parameters

**252-dnevni rolling peak:**
- **Razlog:** Standardna 1-letna trading perioda (252 delovnih dni v letu)
- **Benefit:** Prepreči "locked in emergency mode forever" problem na 10+ letnih strategijah
- **Standard:** Hedge fund industrija uporablja 252d za annual calculations

**2% no-trade band:**
- **Razlog:** Transaction costs + slippage ≈ 0.5-1% roundtrip. 2% band zagotavlja NET pozitivno vrednost vsakega tradea
- **Alternative:** 1% bi preveć často trgovalo (high costs), 3% bi zamudilo profitable rebalanse

**25% max daily change:**
- **Razlog:** Prepreči ekstremne swing koncentracije v 1 dnevu, omogoča smooth ramping
- **Timeline:** Full rebalans (0%→100%) traja max 4 dni (sprejemljivo za mean-reversion)
- **Alternative:** 50% bi dovolilo šok rebalanse (market impact), 10% bi bilo prepočasno

**Risk Mode BTC Caps (60%/30%/5%):**
- **CAUTION 60%:** Omogoča še exposure, a zmanjšan risk (40% cushion)
- **RISK_OFF 30%:** Defenzivno, skoraj half-cash
- **EMERGENCY 5%:** Survival mode (ne 0% zaradi možne recovery participacije)
- **Razlog:** Eksponentno padajoča izpostavljenost (geometrična progresija)

#### Risk Trigger Thresholds

**DD -8%/-15%/-25%:**
- **-8% (CAUTION):** Mild correction, normal market noise
- **-15% (RISK_OFF):** Medium drawdown, previdnost potrebna
- **-25% (EMERGENCY):** Severe crisis, survival mode
- **Benchmark:** Profesionalni hedge fondi imajo -10%/-20% internal alerts

**Vol 55%/75%/110%:**
- **Bazni Vol BTC:** Zgodovinsko ≈ 40-60% annualized
- **55% (CAUTION):** Povišana volatilnost nad normalnim režimom
- **75% (RISK_OFF):** Visoko turbulenten regime (panic selling)
- **110% (EMERGENCY):** Ekstremna panika (crash scenario)

**VaR 3%/5%/7% (99% confidence, 1-day):**
- **3% (CAUTION):** 1% verjetnost za >3% dnevno izgubo (≈30% mesečno)
- **5% (RISK_OFF):** Near-crisis scenarij
- **7% (EMERGENCY):** >50% potential monthly loss scenario
- **Razlog:** 99% confidence standard za institutionalne limite

#### Hysteresis Magnitudes

**DD +3%:**
- **Razlog:** DD je relativno stabilen trigger (computed daily, smooth)
- **Manjša hystereza zadostuje** za anti-whipsaw protection

**Vol -10%:**
- **Razlog:** Volatilnost je najbolj noisy trigger (30-day rolling window, vsak dan drop-off oldest)
- **Velika hystereza potrebna** da preprečiš mode bouncing pri Vol boundary

**VaR -2%:**
- **Razlog:** VaR je matematično precizen (quantile-based), manj noise
- **Manjša hystereza zadostuje**

#### Recovery Timelines (7/5/3 dni)

- **EMERGENCY → RISK_OFF: 7 dni**
  - **Razlog:** ≈ 1 trading week, zagotavlja stabilnost preden release exposure
  - Najhujši režim, potrebna najdaljša potrditev

- **RISK_OFF → CAUTION: 5 dni**
  - **Razlog:** 1 business week, zmeren confirmation period

- **CAUTION → NORMAL: 3 dni**
  - **Razlog:** Najmilejši režim, dovolj kratko da ne zamudi recovery rallya

**Asimetrija:** Crisis entry je instant (1 breach = instant downgrade), recovery je slow (3-7 dni). To je **professional standard** (protect capital fast, release exposure slowly).

## 0.2 Trainable Parameters

Nekateri parametri so lahko **optimizirani** (grid search, Bayesian optimization), drugi so **fixed** (industry standard).

#### TRAINABLE parametri:

1. **EMA windows:** (P_ema, C_ema)
   - **Range:** P_ema ∈ [3, 14], C_ema ∈ [7, 30]
   - **Grid:** (5, 10), (7, 14), (10, 21), (14, 28)
   - **Metric:** Maximiziraj Sharpe Ratio na train setu

2. **Position sizing bands:** (R_ema intervals)
   - **Range:** Lahko spuščaš/dvigneš cutoff točke (npr. 0.80→0.85)
   - **Current:** < 0.80 / 0.80-0.90 / 0.90-1.00 / 1.00-1.10 / 1.10-1.25 / > 1.25
   - **Grid search:** Test ±0.05 spremembe na vsaki prekinitvi

3. **BTC allocations per band:**
   - **Range:** Current = [100%, 85%, 70%, 50%, 30%, 10%]
   - **Grid:** Test [100%, 80%, 60%, 40%, 20%, 10%] ali [100%, 90%, 75%, 50%, 25%, 5%]
   - **Constraint:** Mora biti mono-decent (nižji R_ema → višji BTC %)

4. **No-trade band:**
   - **Range:** [1%, 5%]
   - **Current:** 2%
   - **Grid:** Test 1%, 1.5%, 2%, 3%
   - **Metric:** Balance med turnover cost in missed opportunities

5. **Max daily change:**
   - **Range:** [10%, 50%]
   - **Current:** 25%
   - **Grid:** Test 15%, 20%, 25%, 30%, 40%
   - **Tradeoff:** Hitrost rebalansa vs. market impact

6. **DD Thresholds:**
   - **Range:** CAUTION ∈ [-5%, -12%], RISK_OFF ∈ [-10%, -20%], EMERGENCY ∈ [-20%, -30%]
   - **Current:** -8% / -15% / -25%
   - **Constraint:** Mora biti -8% < -15% < -25%

7. **Vol Thresholds:**
   - **Range:** CAUTION ∈ [40%, 70%], RISK_OFF ∈ [60%, 90%], EMERGENCY ∈ [90%, 130%]
   - **Current:** 55% / 75% / 110%

8. **VaR Thresholds:**
   - **Range:** CAUTION ∈ [2%, 5%], RISK_OFF ∈ [4%, 8%], EMERGENCY ∈ [6%, 10%]
   - **Current:** 3% / 5% / 7%

9. **Hysteresis magnitudes:**
   - **Range:** DD ∈ [1%, 5%], Vol ∈ [5%, 15%], VaR ∈ [1%, 3%]
   - **Current:** +3% / -10% / -2%

10. **Recovery days:**
    - **Range:** EMERGENCY ∈ [5, 14], RISK_OFF ∈ [3, 10], CAUTION ∈ [2, 7]
    - **Current:** 7 / 5 / 3

#### NON-TRAINABLE (fixed) parametri:

1. **Rolling peak window: 252 dni** — Industry standard, ne spreminjaj
2. **VaR confidence level: 99%** — Institutional standard
3. **Volatility window: 30 dni** — Standard za annualized vol calculations
4. **Execution time: 00:00 UTC** — Fixed, disciplined execution
5. **Risk mode caps (60%/30%/5%)** — Geometrična progresija, fundamental design choice

## 0.3 Optimization Methodology

**Training Period:**
- Minimalno **3 leta** (mora zajeti vsaj 1 bull + 1 bear cycle)
- Split: 70% train, 30% test (časovno zaporedno, ne random)
- **Walk-forward:** Vsak kvartal re-optimize na zadnjih 2 letih, test naslednji kvartal

**Objective Function:**
- Primarno: **Sharpe Ratio** (risk-adjusted returns)
- Secondary: **Calmar Ratio** (return / max DD)
- Constraint: **Max DD < -30%** (hard limit)

**Grid Search:**
- Če maš 10 parametrov × 3-5 vrednosti per parameter = **10^4 - 10^5 kombinacij**
- **Solution:** Multi-stage optimization
  1. Stage 1: Optimiziraj samo EMA windows (drži ostale fixed)
  2. Stage 2: Optimiziraj position sizing (drži EMA fixed na Stage 1 rezultat)
  3. Stage 3: Optimiziraj risk thresholds (drži signal fixed)
  4. Stage 4: Fine-tune hysteresis + recovery days

**Bayesian Optimization:**
- **Alternative:** Če je grid search prepočasen, uporabi Bayesian opt (Gaussian Process)
- **Biblioteke:** `scikit-optimize`, `hyperopt`, `optuna`
- **Benefit:** Converge hitreje (50-100 iterations vs. 10k grid)

**Cross-Validation:**
- **Time-series CV:** Rolling window (vsak 6 mesecev train, test naslednji 3 mesece)
- **Metric:** Average Sharpe across all folds
- **Red flag:** Če Sharpe variance med foldi > 0.5 → strategy ni robusten

**Overfitting Protection:**
- **Out-of-sample test obvezen** (min 6 mesecev unseen data)
- **Paper trading:** 1-3 mesece forward test brez realnih sredstev
- **Monte Carlo:** 1000 simulations z random start dates → check consistency

---

# DEL 1: JEDRO STRATEGIJE (VALUATION MODEL)

## 1.1 Glavni signal

Osnovni valuation signal:

$$R_t = \frac{P_t}{C_t}$$

Interpretacija:
- $R_t < 1.0$: BTC je pod proizvodno ceno (podvrednoten)
- $R_t \approx 1.0$: Fair områje
- $R_t > 1.0$: BTC nad proizvodno ceno (nadvrednoten)

## 1.2 Glajenje (odstranjevanje šuma)

Uporabljamo robustno glajenje:
- $P_{ema,t} = \text{EMA}_7(P_t)$ — 7-dnevna EMA cene
- $C_{ema,t} = \text{EMA}_{14}(C_t)$ — 14-dnevna EMA stroška
- $R_{ema,t} = \frac{P_{ema,t}}{C_{ema,t}}$ — glajen signal

**Vsa odločitev uporablja $R_{ema,t}$.**

---

# DEL 2: POSITION SIZING (TARGET UTEŽ BTC)

| Interval $R_{ema,t}$ | BTC % | USDC % | Pomen |
|---|---:|---:|---|
| $< 0.80$ | 100% | 0% | Močna akumulacija |
| $0.80 - 0.90$ | 85% | 15% | Agresivni buy |
| $0.90 - 1.00$ | 70% | 30% | Zmeren buy |
| $1.00 - 1.10$ | 50% | 50% | Nevtralno |
| $1.10 - 1.25$ | 30% | 70% | Defenzivno |
| $> 1.25$ | 10% | 90% | De-risking |

To je osnovna ("raw") signalna utež. Nato jo **omejuje risk mode cap**.

**Pravilo:** $w_{final,t} = \min(w_{signal,t}, w_{mode,t})$

---

# DEL 3: IZVRŠEVANJE (1x DNEVNO OB 00:00 UTC)

## 3.1 Urnik

- Strategija bere zaključne vrednosti dneva (close prices)
- Ob **00:00 UTC** izvede **en rebalans** na target utež
- **Čez dan ne trguje** (ni intraday transakcij)

## 3.2 Anti-Overtrading Pravila

#### No-Trade Band
- Če razlika med trenutno in target utežjo BTC < 2%, **ne trguj**

#### Max Daily Change
- Največja sprememba uteži BTC na dan = 25% portfelja
- Če je potrebna sprememba večja, se preostanek prenese v naslednje dni

**Primer 1: Brez spremembe cilja/signaala (normalni nadaljevak)**
- Dan 1: Trenutna = 20%, Cilj = 80% → Nova = 20% + 25% = **45%**
- Dan 2: Trenutna = 45%, Cilj = 80% → Nova = 45% + 25% = **70%**
- Dan 3: Trenutna = 70%, Cilj = 80% → Nova = 70% + 25% = **80%** ✓

*Vsak dan se ponovno izračunajo: (Cilj - Trenutna) in se +25% prilagodi novi trenutni utežji iz prejšnjega dne*

**Primer 2: Signal se spremeni med dnevi (novo trenutno stanje)**
- Dan 1: Trenutna = 70%, Cilj = 80% → Nova = 70% + 25% = **80%** (dosežen cilj)
- Dan 2: **SIGNAL SPREMEMBA** → Novi cilj = 30% (npr. R_ema pade pod -1σ)
  - Trenutna = 80% (iz Dan 1 izvršenja) → Novi cilj = 30%
  - Nova = 80% - 25% = **55%** (padec je dovoljen)
- Dan 3: Trenutna = 55%, Cilj = 30% → Nova = 55% - 25% = **30%** ✓

*Če se signal ali risk mode spremi med dnevi, se postavi novi cilj in novo trenutno stanje je obaveza iz prejšnjeg dne*

---

# DEL 4: RISK MANAGEMENT SISTEM (PROFESIONALNI SLOJ)

Risk management ne potrebuje dodatnih inputov; izračuna se samo iz NAV zgodovine.

## 4.1 Risk Mode Stanja (4 modu)

| Modo | BTC Cap | USDC Min | Namen |
|---|---:|---:|---|
| NORMAL | 100% | 0% | Brez omejitev, signal-driven |
| CAUTION | 60% | 40% | Povečana previdnost |
| RISK_OFF | 30% | 70% | Defenzivni položaj |
| EMERGENCY | 5% | 95% | Survival mode, skoraj all-cash |

**Final allocation:**
$$w_{final,t} = \min(w_{signal,t}, w_{mode,t})$$

---

## 4.2 TRI NEZAVISNI TRIGGERJI ZA MODE

### TRIGGER 1: DRAWDOWN (DD) CONTROL
#### Rolling 252-Day Peak (Industrijski Standard)

Sistema vzdržuje **rolling peak** zadnjih 252 trading dni (1 leto):

$$\text{Peak}_{rolling,t} = \max(NAV_{t-252}, NAV_{t-251}, ..., NAV_t)$$

**Zakaj rolling, ne all-time?**
- All-time peak → "stuck in emergency forever" (resnični problem)
- Rolling window → fresh start po 1 letu kontinuiranega pada
- Profesionalni hedge fondi in quant firme to uporabljajo kao standard

#### DD Definicija

$$DD_t = \frac{NAV_t - \text{Peak}_{rolling,t}}{\text{Peak}_{rolling,t}}$$

$(DD_t \leq 0$ vedno — to je loss metric)

#### DD-based Mode Transitions

| Drawdown Level | Risk Mode | BTC Cap |
|---|---|---|
| $DD > -8\%$ | NORMAL | 100% |
| $-15\% \leq DD \leq -8\%$ | CAUTION | 60% |
| $-25\% \leq DD < -15\%$ | RISK_OFF | 30% |
| $DD \leq -25\%$ | EMERGENCY | 5% |

#### DD Primeri

**Primer 1: Slide v CAUTION (rolling peak pomaga)**
- Začetni NAV: 100,000
- Leto 1: Peak dvigne na 125,000
- Leto 2: Pogibka, NAV = 110,000
- $DD = (110,000 - 125,000) / 125,000 = -12\%$
- → **CAUTION** (ker je $-15\% < -12\% < -8\%$)
- BTC cap: 60%

**Primer 2: Recovery iz RISK_OFF (rolling peak reset)**
- Peak (dnevi 1-80): 100k → 130k
- Izgube: Dan 200, NAV = 98,000
- $DD = (98,000 - 130,000) / 130,000 = -24.6\%$ → RISK_OFF
- Dnevi 200-252: Nadaljnje izgube
- **Dan 253: Rolling okno se resetira** (dan 1 izpade)
- Novi peak je sedaj ~118,000 (računan na dneh 2-253)
- Če je NAV = 115,000: $DD = (115,000 - 118,000) / 118,000 = -2.5\%$ → **NORMAL**

---

### TRIGGER 2: VOLATILITY (30-DAY, ANNUALIZED)

#### Izračun

**Korak 1:** Zberi dnevne donosne zadnjih 30 dni:
$$r_1, r_2, ..., r_{30}$$

**Korak 2:** Realizirana dnevna volatilnost (std dev):
$$\sigma_{daily} = \sqrt{\frac{1}{30} \sum_{i=1}^{30} (r_i - \bar{r})^2}$$

**Korak 3:** Annualizacija (252 trading dni/leto):
$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$

#### Vol-based Mode Transitions

| Annualized Vol | Risk Mode | BTC Cap |
|---|---|---|
| $\sigma < 55\%$ | NORMAL | 100% |
| $55\% \leq \sigma < 75\%$ | CAUTION | 60% |
| $75\% \leq \sigma < 110\%$ | RISK_OFF | 30% |
| $\sigma \geq 110\%$ | EMERGENCY | 5% |

#### Vol Primeri

**Primer 1: Vol spike → CAUTION**
- Dan 15: $\sigma = 52\%$ → NORMAL
- Dan 16: Volatilen return +5.2%
- Recompute 30-day vol: $\sigma = 57.5\%$
- → **CROSS CAUTION** (55% threshold)
- BTC cap: 60%

**Primer 2: EMERGENCY volatility**
- Ekstremni dnevi (regulatory shock, flash crash)
- Dan 25: $\sigma = 115\%$
- → **EMERGENCY** ($115\% \geq 110\%$)
- BTC cap: 5%
- Dan 40: Vol se stisnejo: $\sigma = 80\%$
- → **Exit EMERGENCY**, enter **RISK_OFF**

---

### TRIGGER 3: VALUE-AT-RISK (VaR, 1-Day, 99% Confidence)

#### Formula

$$\text{VaR}_{99\%} = \mu_r - 2.33 \times \sigma_r$$

kjer:
- $\mu_r$ = povprečni dnevni return (30-dnevno okno)
- $\sigma_r$ = std dev dnevnih donosov
- $2.33$ = z-score za 99% confidence

**DollarAmount:**
$$\text{Loss}_{VaR,99\%} = \text{VaR}_{99\%} \times NAV_t$$

#### VaR-based Mode Transitions

| VaR (% NAV) | Risk Mode | BTC Cap |
|---|---|---|
| $\leq 3\%$ | NORMAL | 100% |
| $3\% - 5\%$ | CAUTION | 60% |
| $5\% - 7\%$ | RISK_OFF | 30% |
| $> 7\%$ | EMERGENCY | 5% |

#### VaR Primeri

**Primer 1: Stabilen režim (low VaR)**
- NAV = 200,000
- $\mu_r = 0.8\%$, $\sigma_r = 1.1\%$
- VaR: $0.008 - 2.33 \times 0.011 = -0.0176 = -1.76\%$
- **VaR = 1.76% of NAV** (well below 5%)
- → No VaR constraint

**Primer 2: Vol spike → VaR breach → RISK_OFF**
- NAV = 180,000
- $\mu_r = 0.3\%$, $\sigma_r = 2.8\%$ (high!)
- VaR: $0.003 - 2.33 \times 0.028 = -0.0622 = -6.22\%$
- **VaR = 6.22% of NAV** (exceeds 5% limit)
- → **Trigger RISK_OFF mode**
- BTC cap: 30%

---

## 4.3 COMPOSITE MODE DETERMINATION

Vsak dan ob 00:00 UTC izračunaj **vseh 3 triggerje**:
1. DD_mode (iz DD tabele)
2. Vol_mode (iz Vol tabele)
3. VaR_mode (iz VaR tabele)

**Final mode = most severe:**
$$\text{Mode}_t = \max(\text{DD\_mode}, \text{Vol\_mode}, \text{VaR\_mode})$$

**Severity ranking:** NORMAL < CAUTION < RISK_OFF < EMERGENCY

**Primer:**
- DD → CAUTION
- Vol → RISK_OFF
- VaR → NORMAL
- **Final: RISK_OFF** (most severe)

---

## 4.4 MODE RECOVERY RULES (PROFESIONALNE S TARGETING-SPECIFIČNO HYSTEREZISO)

Recovery je **asymmetrična** (hitro v krizo, počasi ven). **Vsak trigger ima svojo hystereziso** glede na njegovo volatilnost in magnitude.

**Princip:** Vse tri pogoji morajo biti zadovoljeni **hkrati** za **N zaporednih dni**. Ako se katerikoli pogoj preruši, se counter resetira na 0.

---

### UNIFIED RECOVERY TABLE S TRIGGER-SPECIFIČNO HYSTEREZISO

| Mode Transition | DD Entry | DD Recovery | Vol Entry | Vol Recovery | VaR Entry | VaR Recovery | Dnevi |
|---|---|---|---|---|---|---|---|
| **CAUTION** | DD ≤ -8% | DD > -5% | σ ≥ 55% | σ < 45% | VaR > 3% | VaR < 1% | 3 |
| **RISK_OFF** | DD ≤ -15% | DD > -12% | σ ≥ 75% | σ < 65% | VaR > 5% | VaR < 3% | 5 |
| **EMERGENCY** | DD ≤ -25% | DD > -22% | σ ≥ 110% | σ < 100% | VaR > 7% | VaR < 5% | 7 |

**Hystereza po triggeru:**
- **DD:** +3% (stabilen signal)
- **Volatilnost:** -10% (noise-prone, anti-whipsaw)
- **VaR:** -2% (matematički precizan)

---

#### From EMERGENCY → RISK_OFF
**Pogoji (vsi 3 morajo biti OK hkrati):**
1. $DD > -22\%$ (vs. entry -25%, hystereza +3%)
2. $\sigma < 100\%$ (vs. entry 110%, hystereza -10%)
3. VaR < 5% of NAV (vs. entry 7%, hystereza -2%)

**Trajanje:** Vseh 3 pogojev hkrati za **7 zaporednih dni**
- Če katerikoli pogoj breši, se counter resetira na 0

#### From RISK_OFF → CAUTION
**Pogoji (vsi 3 morajo biti OK):**
1. $DD > -12\%$ (vs. entry -15%, hystereza +3%)
2. $\sigma < 65\%$ (vs. entry 75%, hystereza -10%)
3. VaR < 3% of NAV (vs. entry 5%, hystereza -2%)

**Trajanje:** Vseh 3 za **5 zaporednih dni**

#### From CAUTION → NORMAL
**Pogoji (vsi 3):**
1. $DD > -5\%$ (vs. entry -8%, hystereza +3%)
2. $\sigma < 45\%$ (vs. entry 55%, hystereza -10%)
3. VaR < 1% of NAV (vs. entry 3%, hystereza -2%)

**Trajanje:** Vseh 3 za **3 zaporedne dni**

### Instant Crisis Downgrade (Asymmetric!)
- Downward transitions (iz NORMAL, CAUTION, ali RISK_OFF) so **takojšnje** (no waiting)
- Uporabljamo ENTRY thresholde direktno (bez hysterezise):
  - Če katerikoli trigger dosežè entry prag → mode **takoj** downgrade
- Razlog: Zaščita kapitala je prioriteta, hystereza je samo za recovery out

---

## 4.5 MULTI-DAY SCENARIJ (DEMO S PROFESIONALNIMI RECOVERY RULES)

### Days 1-30: Rast
- NAV: 100k → 125k
- PeakNAV: 125k (rolling)
- DD = 0% → NORMAL
- σ = 18% → NORMAL
- VaR = 1.5% → NORMAL
- **Mode: NORMAL**, BTC 100%

### Days 31-90: Korektura
- NAV: 125k → 105k
- PeakNAV: 125k
- DD = -16% → CAUTION
- σ = 42% → NORMAL
- VaR = 2.1% → NORMAL
- **Mode: CAUTION** (DD je najbolj strog), BTC cap 60%

### Days 91-130: Padec v RISK_OFF
- NAV: 105k → 88k
- DD = -29.6% → RISK_OFF
- σ = 68% → CAUTION
- VaR = 5.8% → RISK_OFF
- **Mode: RISK_OFF** (max(CAUTION, CAUTION, RISK_OFF) = RISK_OFF)
- BTC cap: 30%

### Days 131-150: Kriza (EMERGENCY)
- NAV: 88k → 76k
- DD = -39.2% → EMERGENCY
- σ = 88% → CAUTION
- VaR = 7.2% → RISK_OFF
- **Mode: EMERGENCY** (most severe)
- BTC cap: 5%

### Days 151-205: Počasen Recovery (no exit yet)
- NAV: 76k → 100k
- DD still < -20% (so EMERGENCY)
- **Exit conditions NOT met**: DD > -20%? NO, σ < 100%? YES, VaR < 8%? YES
- Only 2/3 conditions met, so counter = 0

### Days 206-212 (7-day window za EMERGENCY → RISK_OFF)
- Day 206: NAV = 101k, DD = -18.4% ✓ (> -22%), σ = 65% ✓ (< 100%), VaR = 4.8% ✓ (< 5%)
  - **All three conditions met = Recovery counter = 1**

- Days 207-212: All conditions hold each day
  - Day 207: Counter = 2
  - Day 208: Counter = 3
  - Day 209: Counter = 4
  - Day 210: Counter = 5
  - Day 211: Counter = 6
  - Day 212: Counter = 7
  - **DECISION: TRANSITION EMERGENCY → RISK_OFF**
  - Mode changes to RISK_OFF (30% BTC cap)

### Days 238-242 (5-day window za RISK_OFF → CAUTION)
- Day 238: NAV = 112k
  - DD = -10.4% ✓ (> -12%), σ = 52% ✓ (< 65%), VaR = 2.2% ✓ (< 3%)
  - **All three = Counter = 1**

- Days 239-242: All conditions hold
  - Day 239: Counter = 2
  - Day 240: Counter = 3
  - Day 241: Counter = 4
  - Day 242: Counter = 5
  - **DECISION: TRANSITION RISK_OFF → CAUTION**
  - Mode changes to CAUTION (60% BTC cap)

### Days 258-260 (3-day window za CAUTION → NORMAL)
- **Day 253: Rolling 252-day window resets** (day 1 drops out)
- New rolling peak ≈ 123k

- Day 258: NAV = 116k
  - DD = (116k - 123k) / 123k = -5.7% ✓ (> -5%), σ = 48% ✓ (< 45% - WAIT, close), VaR = 1.8% ✓ (< 1%)
  - **All three = Counter = 1**

- Day 259: DD = -4.9% ✓ (> -5%), σ = 46% ✓ (< 45% - marginally), VaR = 1.5% ✓ → Counter = 2

- Day 260: DD = -4.1% ✓, σ = 43% ✓, VaR = 1.1% ✓ → Counter = 3
  - **DECISION: TRANSITION CAUTION → NORMAL**
  - BTC cap: 100%, signal-driven

**Timeline Summary:**
- Days 1-130: Normal → Caution → Risk_Off
- Days 131-212: Emergency (81 days), then 7-day exit to Risk_Off
- Days 213-242: Risk_Off (29 days), then 5-day exit to Caution
- Days 243-260: Caution (17 days), then 3-day exit to Normal
- **Total crisis duration: 130 days** (full professional recovery)

**Key points:**
- Professional recovery requires ALL 3 conditions simultaneously
- Multiple day waits prevent false recoveries
- One breach resets counter to 0 (sticky hysteresis)
- Asymmetric: crisis enters instantly, exits take 3-7 days

---

# DEL 5: OPERATIVNI PROTOKOL

## 5.1 Dnevna Izvedba (00:00 UTC)

```
1. Preberi: P_t (BTC cena), C_t (cena proizvodnje)
2. Izračunaj: R_t = P_t / C_t
3. Posodobi EMA: P_ema, C_ema, R_ema
4. Preberi: Signal tabela → w_signal
5. Izračunaj: NAV_t (novo vrednost portfelja)
6. Posodobi: Rolling peak (max zadnjih 252d)
7. Izračunaj: DD_t
8. Izračunaj: returns zadnjih 30 dni, potem σ_daily, σ_annual
9. Izračunaj: VaR_99%
10. Določi: DD_mode, Vol_mode, VaR_mode
11. Mode_t = max(tri mode)
12. w_final = min(w_signal, mode_cap)
13. Primerjaj: w_final vs. trenutna utež
14. Preveri no-trade band (2%), max daily change (25%)
15. Izvedi trade ob 00:00 UTC (ali skip ako ni potrebe)
16. Zapiši log: datum, P_t, C_t, R_ema, NAV, DD, vol, VaR, mode, w_signal, w_final, trade_size
```

## 5.2 Dnevni Log (Audit Trail)

Vsak dan zapiši:
- **Date, Time (UTC)**
- **Inputs:** P_t, C_t
- **Signal:** R_ema, w_signal
- **Risk:** DD, σ_annual, VaR_99%, Mode
- **Execution:** w_final, trade_size, actual_fill
- **State:** NAV, BTC_units, USDC_balance

---

# DEL 6: MONITORING & KEY PERFORMANCE INDICATORS

## 6.1 Dnevno Spremljanje

**Obvezne metrike vsak dan (automatic logging):**
- **NAV** — Portfolio value
- **Current DD** — (NAV - Rolling Peak) / Rolling Peak
- **30d Volatility** — Annualized σ
- **VaR_99%** — 1-day Value-at-Risk
- **Current Mode** — NORMAL / CAUTION / RISK_OFF / EMERGENCY
- **Recovery Counter** — Če si v recovery, koliko dni zaporedoma OK?
- **Trade executed?** — DA/NE, velikost, fill price

**Red Flags (alarms):**
- DD < -30% → Manual review potreben
- Mode = EMERGENCY > 30 dni → Investigate zakaj
- No trades > 7 dni zaporedoma → Preveri če je no-trade band presplošen

## 6.2 Performance Metrics (Rolling 30/90/365 dni)

**Returns:**
- **Total Return** — (NAV_t - NAV_0) / NAV_0
- **CAGR** — Compounded Annual Growth Rate
- **Monthly returns distribution** — Histogram za detect outliers

**Risk-Adjusted:**
- **Sharpe Ratio** — (Return - Risk-Free Rate) / σ
  - **Target:** > 1.0 (dober), > 1.5 (odličen)
- **Sortino Ratio** — Return / Downside σ (samo negativni returns)
  - **Target:** > 1.5
- **Calmar Ratio** — Annual Return / Max DD
  - **Target:** > 1.0

**Risk:**
- **Maximum Drawdown** — Največji DD v obdobju
  - **Limit:** < -30% (če prekoračen, strategy review)
- **Average DD Duration** — Koliko dni v povprečju traja recovery
- **% Time in Each Mode** — Ali je razumna distribucija? (če >20% v EMERGENCY, problem)

**Trading:**
- **Win Rate** — % profitable trades
  - **Target mean-reversion:** > 50%
- **Avg Trade P&L** — Povprečni profit/loss per trade
- **Turnover** — Kolikokrat se portfolio obrne letno
  - **Target:** < 3x za dnevno strategijo (višje = high costs)

**Benchmark Comparison:**
- **Buy & Hold BTC** — Ali strategy outperformaš simple hold?
- **60/40 BTC/USDC** — Ali risk-adjusted returns boljši?

## 6.3 Quarterly Review (vsak 3 mesece)

1. **Backtest Validation:**
   - Ali live results matchajo backtest expectations?
   - Če Sharpe ratio live < 70% backtest Sharpe → Investigate (možno overfitting)

2. **Parameter Drift Check:**
   - Ali so še EMA windows optimalni? (market regime lahko se spremeni)
   - Test walk-forward optimization → če novi parametri ≫ boljši, consideriraj update

3. **Cost Analysis:**
   - Ali transaction costs rastejo? (exchange fee changes, slippage increase?)
   - Če DA, morda povečaj no-trade band

4. **Data Quality:**
   - Ali je production cost data še kvalitetna?
   - Cross-check z alternative sources (hash rate proxies, energy consumption indices)

---

# DEL 7: BACKTESTING METHODOLOGY

## 7.1 Historical Data Requirements

**Time Period:**
- **Minimalno:** 4+ leta (mora zajeti vsaj 1 bull market, 1 bear market, 1 halvening cycle)
- **Recommended:** 6-8 let (če data obstaja)
- **Recent bias:** Daj večjo težo zadnjim 3 letom (market structure se spreminja)

**Data Frequency:**
- **Dnevni close prices** (snapshot ob 00:00 UTC ali 23:59 UTC previous day)
- **No intraday data needed** (strategija ne potrebuje)

**Data Sources:**
- **BTC Price:** Composite od multiple exchanges (Binance + Coinbase + Kraken average) → prepreči flash crash bias
- **Production Cost:** Reliable source (Cambridge Bitcoin Energy Consumption Index, Hashrate Index, proprietary model)
- **Cross-validation:** Če imaš 2 cost sources, compariraj (če divergence > 10%, investigate)

## 7.2 Backtest Assumptions

**Transaction Costs:**
- **Exchange fee:** 0.1-0.2% per side (Binance/Coinbase Pro maker fee)
- **Slippage:** 0.1% (BTC/USDC je liquid, minimal slippage pri market orders)
- **Total roundtrip cost:** **0.5%** (conservative estimate)
- **Apply na vsak trade** (subtract od returns)

**Execution:**
- **Fill price:** Uporabi close price ob 00:00 UTC (ali open price naslednjega bar-a če daily bars)
- **No lookahead bias:** Signal na dan T uporablja samo podatke do T-1 23:59 UTC
- **Partial fills:** Assume 100% fill (BTC/USDC dovolj likviden), a če NAV > $10M, consideriraj market impact

**Starting Capital:**
- **Izberi realistic amount:** $100k tipično za individual, $1M+ za institutional
- **No cash injections/withdrawals** — Closed system (organic growth only)

**Risk-Free Rate:**
- Uporabi **0%** (crypto nema risk-free equivalent) ali **US T-Bill** če hočeš comparable Sharpe

## 7.3 Parameter Optimization Process

**Step 1: Train/Test Split**
```
Historical Data (6 let, 2018-2024)
├── Train Set: 2018-2022 (4 leta, 70%)
└── Test Set: 2023-2024 (2 leti, 30%)
```
**Pomembno:** Split je **časovno zaporedno** (ne random shuffle)

**Step 2: Define Parameter Grid**

Primer za osnovni grid search:

```python
param_grid = {
    'P_ema': [5, 7, 10, 14],
    'C_ema': [10, 14, 21, 30],
    'no_trade_band': [0.01, 0.02, 0.03],
    'max_daily_change': [0.15, 0.20, 0.25, 0.30],
    'dd_caution': [-0.06, -0.08, -0.10],
    'dd_risk_off': [-0.12, -0.15, -0.18],
    'dd_emergency': [-0.20, -0.25, -0.30]
}
```

**Total combinations:** 4 × 4 × 3 × 4 × 3 × 3 × 3 = **5,184 kombinacij**

**Step 3: Run Grid Search na Train Set**

Za vsako kombinacijo:
1. Run backtest na 2018-2022 data
2. Izračunaj **Sharpe Ratio** (primary metric)
3. Izračunaj **Max DD** (constraint: mora biti > -30%)
4. Izračunaj **Calmar Ratio** (secondary metric)
5. Log results → DataFrame

**Step 4: Select Best Parameters**

```python
# Filter: Max DD must be > -30%
valid_results = results[results['max_dd'] > -0.30]

# Sort by Sharpe Ratio (descending)
best_params = valid_results.sort_values('sharpe_ratio', ascending=False).iloc[0]
```

**Step 5: Validate na Test Set (Out-of-Sample)**

- Run backtest z `best_params` na 2023-2024 data
- **Success criteria:**
  - Test Sharpe > 70% Train Sharpe (če nižje → overfitting)
  - Test Max DD < -30%
  - Test Calmar > 0.5

**Step 6: Walk-Forward Validation**

**Purpose:** Preveri robustnost over time (market regimes se spreminjajo)

```
Period 1: Train on 2018-2020 → Test on 2021 Q1
Period 2: Train on 2019-2021 → Test on 2021 Q3
Period 3: Train on 2020-2022 → Test on 2022 Q1
Period 4: Train on 2021-2023 → Test on 2023 Q1
...
```

**Metric:** Average Sharpe across all test windows

**Red Flag:** Če variance Sharpe med windows > 1.0 → Strategy ni robusten (regime-dependent)

## 7.4 Multi-Stage Optimization (Če je Grid Search prepočasen)

**Stage 1: Signal Parameters Only**
- Optimiziraj: `P_ema`, `C_ema`, `no_trade_band`, `max_daily_change`, position sizing bands
- **Fix risk parameters na default:** DD [-8%, -15%, -25%], Vol [55%, 75%, 110%], itd.
- **Grid size:** ≈ 500 kombinacij
- **Output:** Best signal parameters

**Stage 2: Risk Thresholds Only**
- **Fix signal parameters** na Stage 1 results
- Optimiziraj: `dd_caution`, `dd_risk_off`, `dd_emergency`, `vol_caution`, `vol_risk_off`, itd.
- **Grid size:** ≈ 1000 kombinacij
- **Output:** Best risk thresholds

**Stage 3: Hysteresis & Recovery Days**
- **Fix signal + risk thresholds** na Stage 1+2 results
- Optimiziraj: `dd_hyst`, `vol_hyst`, `var_hyst`, `recovery_days_emergency`, itd.
- **Grid size:** ≈ 200 kombinacij
- **Output:** Final optimized parameters

**Benefit:** 500 + 1000 + 200 = **1,700 runs** namesto 5,184 (3× hitrejše)

**Tradeoff:** Možno da miss global optimum (a v praksi difference je majhna)

## 7.5 Bayesian Optimization (Napredna Alternativa)

Če je grid search še vedno prepočasen, uporabi **Bayesian Optimization**:

**Princip:**
- Model (Gaussian Process) predvidi kateri parametri bodo obetavni
- Testi samo promising kombinacije
- **Converge v 50-200 iterations** namesto 5,000

**Python Library:**
```python
from skopt import gp_minimize
from skopt.space import Real, Integer

space = [
    Integer(5, 14, name='P_ema'),
    Integer(10, 30, name='C_ema'),
    Real(-0.10, -0.05, name='dd_caution'),
    Real(-0.20, -0.12, name='dd_risk_off'),
    Real(-0.35, -0.20, name='dd_emergency'),
    # ... other parameters
]

def objective(params):
    sharpe = run_backtest(params, train_data)
    return -sharpe  # Minimize negative Sharpe = maximize Sharpe

result = gp_minimize(objective, space, n_calls=100)
best_params = result.x
```

**Output:** Best parameters v ≈ 100 runs (50× hitrejše od full grid)

## 7.6 Validation Checklist

**Preden greš live, preveri:**
- [ ] **Backtest Sharpe > 1.0** (če nižje, strategy ni profitable enough)
- [ ] **Max DD < -30%** (če več, risk management failure)
- [ ] **Out-of-sample Sharpe > 70% in-sample Sharpe** (če nižje, overfitting)
- [ ] **Walk-forward Sharpe variance < 0.5** (če več, strategy ni robusten)
- [ ] **Win rate > 45%** (če nižje, mean-reversion ne dela)
- [ ] **Monte Carlo: 80%+ simulations profitable** (test 1000 random start dates)
- [ ] **Paper trading 1-3 mesece:** Live execution brez realnih sredstev (preveri če data feeds + execution delajo)

---

# DEL 8: ZAKAJ JE TO PROFESIONALNO?

1. **Jasna valuation teorija** — $P/C$ ratio je temeljit, economic anchor
2. **Industrijski DD standard** — rolling 252-day peak, ne all-time (hedge fund standard)
3. **Multi-trigger risk** — DD, Vol, VaR so neodvisni in matematični
4. **Disciplina** — 1 trade/dan, fixed time, no emotion, no discretion
5. **Hitrost** — samo 2 inputa, hitro kalkulirati, scalable
6. **Audit trail** — popolna revizijska sled za compliance/analizo
7. **Asymmetric protection** — instant crisis entry, sticky recovery out (professional risk management)
8. **Trainable parameters** — Sistemski pristop k optimizaciji (grid search, Bayesian opt, walk-forward validation)
9. **Monitoring framework** — KPIs za detect performance degradation
10. **Backtest rigor** — Out-of-sample testing, Monte Carlo, paper trading

Ovo je **institucijsko-level** system ki koristi samo 2 inputa dnevno in je dokazano v praksi v hedge fund in quant industriji.

---

# POVZETEK: HITRI REFERENCE

| Trigger | Prag 1 | Modo 1 | Prag 2 | Modo 2 | Prag 3 | Modo 3 |
|---|---|---|---|---|---|---|
| **DD** | -8% | CAUTION | -15% | RISK_OFF | -25% | EMERGENCY |
| **Vol** | 55% | CAUTION | 75% | RISK_OFF | 110% | EMERGENCY |
| **VaR** | 5% | RISK_OFF | 7% | EMERGENCY | — | — |

**Mode Caps:** NORMAL (100%) / CAUTION (60%) / RISK_OFF (30%) / EMERGENCY (5%)

**Recovery (Professional Rules S Trigger-Specifično Hystereziso):**
- **DD:** Hystereza +3% za vse (CAUTION -8%→-5%, RISK_OFF -15%→-12%, EMERGENCY -25%→-22%)
- **Vol:** Hystereza -10% za vse (CAUTION 55%→45%, RISK_OFF 75%→65%, EMERGENCY 110%→100%)
- **VaR:** Hystereza -2% za vse (CAUTION 3%→1%, RISK_OFF 5%→3%, EMERGENCY 7%→5%)
- **Timing:** 7 dni za EMERGENCY, 5 dni za RISK_OFF, 3 dni za CAUTION
- **Downward:** Instant (asymmetric)