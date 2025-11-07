# Phase 2 Pragmatic: Mejoras con FRED + APIs Confiables

**Objetivo:** Mejorar el sistema de detección de estrés de liquidez usando **SOLO** APIs confiables (sin scraping).

---

## 🎯 TOP 10 MEJORAS PRIORITARIAS

### **TIER 1: Critical - Agrega AHORA (30 min implementación)**

#### 1. **Commercial Paper Spreads** 🔴 CRÍTICO
```yaml
Series FRED:
  DCPF3M: "3-Month AA Financial Commercial Paper"
  DCPN3M: "3-Month AA Nonfinancial Commercial Paper"

Por qué importa:
- CP es el "canary in the coal mine" de estrés de liquidez
- Se dispara DÍAS antes de crisis (2008, 2020)
- Mide estrés en funding corporativo

Implementación:
  spread_cp_tbill = DCPF3M - TB3MS
  if spread_cp_tbill > 100bp: ALERTA ROJA
```

#### 2. **Discount Window Borrowing** 🔴 CRÍTICO
```yaml
Series FRED:
  WLODLL: "Fed Discount Window Borrowing"
  H41RESPPALDKNWW: "Primary Credit Outstanding"

Por qué importa:
- Banks ONLY use discount window when desperate
- Spike = Crisis inmediata (SVB 2023)
- Leading indicator #1 de stress banking

Implementación:
  if WLODLL > $5B: ALERTA MÁXIMA
  delta_discount = diff(WLODLL)
  if delta_discount > $1B: CRISIS MODE
```

#### 3. **Repo Market Rates** 🟠 HIGH VALUE
```yaml
Series FRED:
  BGCR: "Broad General Collateral Rate"
  SOFR30DAYAVG: "30-Day Average SOFR"
  SOFR90DAYAVG: "90-Day Average SOFR"

Por qué importa:
- Repo = plomería del sistema
- BGCR vs SOFR spread = collateral scarcity
- Averaging reduce noise

Implementación:
  repo_spread = BGCR - SOFR
  term_premium = SOFR90DAYAVG - SOFR
```

---

### **TIER 2: High Value - Agrega Esta Semana (2 horas)**

#### 4. **Credit Curve Detail** 🟠 HIGH VALUE
```yaml
Series FRED:
  BAMLC0A0CM: "AAA Corporate OAS"
  BAMLC0A4CBBB: "BBB Corporate OAS"
  BAMLH0A1HYBB: "BB High Yield OAS"
  BAMLH0A3HYC: "CCC & Lower OAS"

Por qué importa:
- Credit curve inversion = stress
- BBB-AAA widening = flight to quality
- CCC spike = distress selling

Implementación:
  credit_curve = {
    'AAA': BAMLC0A0CM,
    'BBB': BAMLC0A4CBBB,
    'BB': BAMLH0A1HYBB,
    'CCC': BAMLH0A3HYC
  }

  BBB_AAA_spread = BAMLC0A4CBBB - BAMLC0A0CM
  if BBB_AAA_spread > 150bp: WARNING
```

#### 5. **Dollar Strength Index** 🟠 IMPORTANT
```yaml
Series FRED:
  DTWEXBGS: "Trade Weighted Dollar Index (Broad)"
  DTWEXBMGS: "Trade Weighted Dollar Index (Major)"

Por qué importa:
- Strong dollar = USD shortage globally
- Correlates with offshore USD stress
- Leading indicator for EM stress

Implementación:
  dollar_strength_zscore = zscore(DTWEXBGS, window=252)
  if dollar_strength_zscore > 2: USD_SHORTAGE_ALERT
```

#### 6. **Breakeven Inflation** 🟡 MEDIUM
```yaml
Series FRED:
  T10YIE: "10-Year Breakeven Inflation"
  T5YIE: "5-Year Breakeven Inflation"
  DFII10: "10-Year TIPS Spread"

Por qué importa:
- Breakeven collapse = deflation fear
- Real rates = nominal - breakeven
- Flight to safety indicator

Implementación:
  real_rate_10y = DGS10 - T10YIE
  breakeven_slope = T10YIE - T5YIE
```

---

### **TIER 3: Nice to Have - Próximo Mes**

#### 7. **Bank Credit Aggregates**
```yaml
TOTBKCR: "Total Bank Credit"
BUSLOANS: "C&I Loans"
REALLN: "Real Estate Loans"
DPSACBW027SBOG: "Broker-Dealer Credit"
```

#### 8. **Term Structure Detail**
```yaml
T10Y3M: "10Y-3M Spread"
DGS30: "30-Year Treasury"
DGS5: "5-Year Treasury"
```

#### 9. **Leading Economic Indicators**
```yaml
INDPRO: "Industrial Production"
ICSA: "Initial Jobless Claims"
UNRATE: "Unemployment Rate"
```

#### 10. **Safe Haven Assets**
```yaml
GOLDAMGBD228NLBM: "Gold Price"
DCOILWTICO: "WTI Crude Oil"
```

---

## 📊 IMPACTO ESTIMADO POR TIER

| Tier | Series | Tiempo | Mejora Rating | Costo |
|------|--------|--------|---------------|-------|
| **Tier 1** | 10 | 30 min | 3.5 → 4.0 | $0 |
| **Tier 2** | 15 | 2 hrs | 4.0 → 4.3 | $0 |
| **Tier 3** | 20 | 4 hrs | 4.3 → 4.5 | $0 |

---

## 🚀 IMPLEMENTACIÓN RÁPIDA (Tier 1)

### Paso 1: Actualiza `series_map.yaml`

```yaml
# Commercial Paper (CRÍTICO)
stress_indicators:
  CP_FINANCIAL_3M:
    code: "DCPF3M"
    description: "3-Month AA Financial Commercial Paper Rate"
    category: "money_markets"

  CP_NONFINANCIAL_3M:
    code: "DCPN3M"
    description: "3-Month AA Nonfinancial Commercial Paper Rate"
    category: "money_markets"

# Discount Window (CRÍTICO)
core_plumbing:
  DISCOUNT_WINDOW:
    code: "WLODLL"
    description: "Federal Reserve Discount Window Borrowing"
    category: "fed_facilities"

  PRIMARY_CREDIT:
    code: "H41RESPPALDKNWW"
    description: "Primary Credit Outstanding"
    category: "fed_facilities"

# Repo Rates (HIGH VALUE)
reference_rates:
  BGCR:
    code: "BGCR"
    description: "Broad General Collateral Rate"
    category: "repo"

  SOFR_30D_AVG:
    code: "SOFR30DAYAVG"
    description: "30-Day Average SOFR"
    category: "repo"
```

### Paso 2: Agrega Features Derivadas

```python
# En fred_client.py compute_derived_features()

# Commercial Paper Spread
if "CP_FINANCIAL_3M" in df.columns and "TB3MS" in df.columns:
    df["cp_tbill_spread"] = df["CP_FINANCIAL_3M"] - df["TB3MS"]

# Discount Window Alarm
if "DISCOUNT_WINDOW" in df.columns:
    df["discount_window_alarm"] = (df["DISCOUNT_WINDOW"] > 5000).astype(int)

# Repo Spread
if "BGCR" in df.columns and "SOFR" in df.columns:
    df["bgcr_sofr_spread"] = df["BGCR"] - df["SOFR"]
```

### Paso 3: Agrega Alertas en Dashboard

```python
# En app.py

# CRITICAL ALERT: Discount Window
if df["DISCOUNT_WINDOW"].iloc[-1] > 5000:
    st.error("🚨 CRISIS ALERT: Discount Window Usage > $5B")

# WARNING: CP Spread
if df["cp_tbill_spread"].iloc[-1] > 100:
    st.warning("⚠️ WARNING: CP-TBill Spread > 100bp")

# INFO: Repo Stress
if df["bgcr_sofr_spread"].iloc[-1] > 10:
    st.info("📊 Elevated Repo Spread")
```

---

## 🎁 BONUS: Treasury Direct API (Más fácil que FRED)

### Datos Únicos que FRED No Tiene

```python
import requests

# Operating Cash Balance (TGA real-time)
url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/cash_balance"
params = {
    "sort": "-record_date",
    "page[size]": "100"
}
response = requests.get(url, params=params)
data = response.json()

# Federal Tax Deposits (daily flows)
url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/tax_deposits"
```

**Ventajas sobre FRED:**
- ✅ Datos más recientes (same-day)
- ✅ Granularidad diaria (no weekly)
- ✅ Incluye intraday updates
- ✅ Sin límite de rate

---

## 📈 COMPARACIÓN: Antes vs Después

### Sistema Actual (20 series FRED)
```
Rating: 3.5/5.0
Cobertura:
- Plomería básica: ✅
- Stress indicators: ✅
- Money markets: ⚠️ (incompleto)
- Banking stress: ❌
- Credit markets: ⚠️ (solo HY)
```

### Con Tier 1 (30 series FRED)
```
Rating: 4.0/5.0
Cobertura:
- Plomería básica: ✅✅
- Stress indicators: ✅✅
- Money markets: ✅ (CP + Repo)
- Banking stress: ✅ (Discount Window)
- Credit markets: ✅ (Credit curve)
```

### Con Tier 1+2 (45 series FRED)
```
Rating: 4.3/5.0
Cobertura:
- Plomería básica: ✅✅✅
- Stress indicators: ✅✅✅
- Money markets: ✅✅ (full coverage)
- Banking stress: ✅✅
- Credit markets: ✅✅ (AAA to CCC)
- FX markets: ✅ (Dollar index)
```

---

## 🎯 RECOMENDACIÓN FINAL

**Empieza con Tier 1 (10 series):**
1. Commercial Paper spreads
2. Discount Window
3. BGCR & SOFR averages

**Tiempo total:** 30 minutos
**Mejora rating:** 3.5 → 4.0
**Costo:** $0

**Luego agrega Tier 2** cuando tengas tiempo.

---

## 🔗 RECURSOS

**FRED Series Search:**
https://fred.stlouisfed.org/categories

**Treasury Direct API Docs:**
https://fiscaldata.treasury.gov/api-documentation/

**Alpha Vantage Registration:**
https://www.alphavantage.co/support/#api-key

**yfinance Docs:**
https://pypi.org/project/yfinance/
