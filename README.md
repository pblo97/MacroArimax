# Liquidity Stress Detection System (MacroArimax)

## Sistema Avanzado de Detección de Estrés de Liquidez

Sistema completo de monitoreo y predicción de estrés de liquidez macro, con horizonte de alerta temprana de 1-10 días. Combina múltiples metodologías estadísticas avanzadas para detectar tensiones en el "plumbing" del sistema financiero.

---

## 🎯 Objetivo Operativo

**Alerta temprana de estrés de liquidez** con tres outputs principales:

1. **Semáforo de estrés**: Probabilidad calibrada (0-1) de risk-off en próximos 1-10 días
2. **Mapa de drenajes**: Grafo dinámico que identifica quién drena/inyecta liquidez y por qué
3. **Overlay operativo**: Recomendación de posicionamiento (beta objetivo / exposición neta)

---

## 📊 Arquitectura del Sistema

```
macro_plumbing/
├── data/                    # Ingesta de datos
│   ├── fred_client.py      # Cliente FRED con cache incremental
│   └── series_map.yaml     # Configuración de series
│
├── features/                # Feature engineering
│   ├── transforms.py       # Z-scores, ROC, ATR-norm, flags
│   ├── net_liquidity.py    # Net Liquidity (Yardeni-style)
│   └── leadlag.py          # Correlaciones cruzadas lead-lag
│
├── models/                  # Modelos estadísticos
│   ├── dfm_kalman.py       # Dynamic Factor Model + Kalman Filter
│   ├── hmm_global.py       # HMM/Markov 2 regímenes
│   ├── cusum_ewma.py       # CUSUM + EWMA control charts
│   ├── changepoints.py     # Detección de breaks estructurales
│   ├── anomalies.py        # IsolationForest (outliers multivariados)
│   ├── nowcast.py          # Logit/Quantile para prob(risk-off)
│   └── fusion.py           # BMA + calibración (Platt/isotónica)
│
├── graph/                   # Análisis de red
│   ├── graph_builder.py    # Constructor de grafo de liquidez
│   ├── graph_markov.py     # HMM local por nodo + contagio
│   └── graph_analytics.py  # PageRank, min-cut, CUSUM por nodo
│
├── backtest/                # Validación
│   ├── walkforward.py      # Walk-forward validation
│   └── metrics.py          # IC, AUROC, Brier, Q4-Q1, robustez
│
└── app/                     # UI
    └── app.py               # Streamlit app (5 tabs)
```

---

## 🔬 Metodologías Implementadas

### 1. Feature Engineering
- **Core plumbing**: SOFR, EFFR, OBFR, TGCR (niveles y spreads)
- **Fed Balance Sheet**: ON RRP, TGA, Reservas, QT
- **Stress indicators**: NFCI, STLFSI4, HY OAS, Term Spread (T10Y2Y), VIX
- **Derivados**: Z-scores rolling, rate-of-change, vol-adjusted moves, quarter-end flags
- **Net Liquidity**: NL = Reservas - TGA - ON RRP (y deltas)

### 2. Detección de Régimen/Evento (Ensemble)

#### **DFM + Kalman** (Factor latente suave)
- Dynamic Factor Model para estimar factor latente de liquidez
- Kalman Filter para smoothing y filtrado óptimo
- Más estable que PCA estático

```python
from macro_plumbing.models.dfm_kalman import fit_dfm_liquidity

filtered, smoothed, model = fit_dfm_liquidity(indicators_df)
factor_z = model.get_factor_zscore(window=252)
```

#### **HMM / Markov Switching** (Régimen global)
- 2 regímenes: calmo / tenso
- Switching variance para capturar cambios en volatilidad

```python
from macro_plumbing.models.hmm_global import fit_liquidity_hmm

prob_stress, model = fit_liquidity_hmm(y=returns, X=liquidity_factor)
```

#### **CUSUM + EWMA** (Detección rápida)
- CUSUM para detectar desvíos persistentes de la media
- EWMA para control chart con bandas (semáforo)

```python
from macro_plumbing.models.cusum_ewma import CUSUM, EWMA

cusum = CUSUM(k=0.5, h=4.0)
alarms = cusum.get_signals(spread_series)
```

#### **Change-points** (Breaks estructurales)
- Ruptures library (Pelt/Binary Segmentation)
- Identifica eventos como picos TGA, fines de QT

```python
from macro_plumbing.models.changepoints import detect_changepoints

breakpoints = detect_changepoints(series, method='pelt', penalty=10.0)
```

#### **IsolationForest** (Anomalías multivariadas)
- Detecta cuando varias "tuberías" se desalinean simultáneamente
- Útil para combinaciones raras de plumbing

```python
from macro_plumbing.models.anomalies import detect_anomalies

anomalies = detect_anomalies(features_df, contamination=0.05)
```

#### **Quantile/Logit** (Bridge a resultados)
- Mapea señales a probabilidades calibradas de risk-off
- Target: SPX ER < 0, ΔHY OAS > 0 en próximos 1-5 días

```python
from macro_plumbing.models.nowcast import nowcast_risk_off

probs, model = nowcast_risk_off(X=features, returns=spx_returns, horizon=5)
```

### 3. Fusión de Señales (Score Maestro)

**BMA/Ensemble** con ponderaciones robustas:

```python
from macro_plumbing.models.fusion import SignalFusion

fusion = SignalFusion(method='weighted_average', calibration='isotonic')
fusion.fit(signals_df, target=risk_off_binary)

# Fused probability
prob_stress = fusion.transform(signals_df)
```

**Calibración**: Platt scaling o isotónica para mapear scores → Prob(risk-off)

---

## 🌐 Grafo de Liquidez

### Nodos (Entidades)
- Tesoro (TGA)
- Fed (QT/ON RRP/SRF)
- Bancos (Reservas/FHLB)
- MMFs
- Dealers
- GSEs, Hedge Funds, REITs, BDC/Private Credit
- UST Market

### Aristas (Flujos)
- Dirigidas, peso = flujo efectivo o "presión"
- Color: verde (inyección), rojo (drenaje)
- Grosor ∝ |z-score|

### Markov sobre el Grafo

**Markov global**: HMM sobre factor de liquidez → colorea todo el grafo (calmo/tenso)

**Markov local**: Cada nodo tiene estado (OK/Tenso) vía HMM univariante

**Contagio**: Random walk 1-paso para estimar propagación de tensión entre nodos

```python
from macro_plumbing.graph.graph_builder import build_liquidity_graph

graph = build_liquidity_graph(fred_data_df)
nodes_df, edges_df = graph.to_dataframe()

sinks = graph.get_sinks(top_n=5)  # Top drenadores
sources = graph.get_sources(top_n=5)  # Top inyectores
```

---

## 📈 Backtest & Robustez

### Walk-Forward Validation

```python
from macro_plumbing.backtest.walkforward import WalkForwardValidator

validator = WalkForwardValidator(train_window=252, test_window=63)
results = validator.validate(X, y, model_func)
```

### Métricas

- **IC (Spearman)**: Correlación de rangos predicción vs actual
- **AUROC**: Área bajo curva ROC
- **Brier Score**: Error cuadrático medio de probabilidades
- **Q4-Q1 Spread**: Diferencia de retornos entre cuartiles extremos
- **Hit Rate**: Accuracy binaria
- **Sharpe/Sortino**: Ratios de overlay

```python
from macro_plumbing.backtest.metrics import compute_all_metrics

metrics = compute_all_metrics(predictions, actuals, returns)
# Returns: IC, AUROC, Brier, Q4-Q1, etc.
```

### Robustez
- **Jackknife**: Quita 1 señal y verifica degradación
- **Sensitivity**: Estabilidad de umbrales y pesos
- **Stress-tests**: Performance en crisis conocidas

---

## 🖥️ Aplicación Streamlit

### 5 Tabs:

1. **🚦 Semáforo**: Estado actual (🔴🟡🟢), score de estrés, métricas clave
2. **📊 Detalle Señales**: Desglose de cada componente (DFM, CUSUM, anomalías, etc.)
3. **🔗 Mapa de Drenajes**: Grafo interactivo de flujos de liquidez
4. **📈 Backtest**: Métricas OOS, walk-forward results
5. **🔍 Explicabilidad**: Atribución (SHAP o descomposición lineal), drivers actuales

### Ejecución

```bash
# Desde la raíz del proyecto
streamlit run macro_plumbing/app/app.py
```

---

## 🚀 Instalación

### 1. Clonar repositorio

```bash
git clone https://github.com/pblo97/MacroArimax.git
cd MacroArimax
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar API Key de FRED

Crear archivo `.streamlit/secrets.toml`:

```toml
FRED_API_KEY = "tu_api_key_aqui"
```

O setear variable de entorno:

```bash
export FRED_API_KEY="tu_api_key_aqui"
```

Obtener API key gratuita: https://fred.stlouisfed.org/docs/api/api_key.html

---

## 📦 Dependencias Principales

```
pandas>=2.0.0
numpy>=1.24.0
statsmodels>=0.14.0      # DFM, HMM, Quantile Regression
scikit-learn>=1.3.0      # PCA, IsolationForest, calibración
ruptures>=1.1.8          # Change-point detection
networkx>=3.1            # Grafo de liquidez
arch>=6.2.0              # GARCH (opcional)
fredapi>=0.5.1           # FRED data
plotly>=5.18.0           # Visualizaciones
streamlit>=1.28.0        # Web app
shap>=0.44.0             # Explicabilidad (opcional)
```

---

## 🔥 Casos de Uso

### 1. Monitoring Diario
- Ejecutar app Streamlit cada mañana
- Revisar semáforo y score de estrés
- Identificar drivers en tab de Explicabilidad

### 2. Análisis de Régimen
- Usar HMM global para identificar cambios de régimen
- Correlacionar con eventos macro (FOMC, QE/QT, crisis)

### 3. Construcción de Overlay
- Usar prob(stress) para ajustar beta
- Reducir exposición cuando prob > 0.6
- Incrementar cuando prob < 0.3

### 4. Research de Señales
- Usar lead-lag scanner para identificar indicadores líderes
- Backtest walk-forward para validar nuevas señales
- Agregar al ensemble con pesos optimizados

---

## 🧪 Testing

```bash
# Run unit tests (cuando estén implementados)
pytest macro_plumbing/tests/

# Run example scripts
python macro_plumbing/data/fred_client.py
python macro_plumbing/features/net_liquidity.py
python macro_plumbing/models/dfm_kalman.py
```

---

## 📚 Referencias Técnicas

### Papers & Methodology
- **Dynamic Factor Models**: Stock & Watson (2002), "Forecasting Using Principal Components"
- **Kalman Filter**: Hamilton (1994), "Time Series Analysis"
- **Markov Switching**: Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series"
- **CUSUM**: Page (1954), "Continuous Inspection Schemes"
- **IsolationForest**: Liu et al. (2008), "Isolation Forest"
- **Change-point Detection**: Killick et al. (2012), "Optimal Detection of Changepoints With a Linear Computational Cost"

### Market Microstructure
- **Net Liquidity**: Yardeni Research methodology
- **Repo Plumbing**: Pozsar (2014), "Shadow Banking: The Money View"
- **Fed Balance Sheet**: Singh (2020), "Reserves, Repo, and Other Plumbing"

---

## 🛠️ Desarrollo Futuro

### Próximas Mejoras
- [ ] Completar `graph_markov.py` y `graph_analytics.py` con Markov local
- [ ] Integrar SHAP para explicabilidad avanzada
- [ ] Añadir FX markets y commodities como features adicionales
- [ ] Implementar estrategia de trading automatizada (overlay)
- [ ] API REST para integración con otros sistemas
- [ ] Dashboard en tiempo real (WebSocket para updates)

### Optimizaciones
- [ ] Caching más agresivo (Redis)
- [ ] Paralelización de modelos (joblib/dask)
- [ ] GPU acceleration para modelos pesados

---

## 👥 Contribuir

1. Fork el repositorio
2. Crear branch (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

---

## 📄 Licencia

MIT License - Ver archivo [LICENSE](LICENSE) para detalles.

---

## 🙏 Agradecimientos

- **FRED (Federal Reserve Economic Data)** por la API gratuita de datos macro
- **Statsmodels** team por las implementaciones de state-space models
- **Ruptures** library para change-point detection
- Comunidad de **QuantFinance** en Twitter/X por insights de mercado

---

## 📧 Contacto

**Autor**: Pablo
**GitHub**: [@pblo97](https://github.com/pblo97)
**Proyecto**: [MacroArimax](https://github.com/pblo97/MacroArimax)

---

## ⚠️ Disclaimer

**Este sistema es solo para fines educativos e investigación.**

No constituye asesoramiento de inversión. Los mercados financieros son inherentemente impredecibles. Siempre consulte con un profesional financiero antes de tomar decisiones de inversión.

El código se proporciona "tal cual", sin garantías de ningún tipo.

---

**Happy stress detection!** 🌊📈
