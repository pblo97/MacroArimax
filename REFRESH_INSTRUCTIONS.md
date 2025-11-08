# 🔄 Instrucciones para Refrescar el Modelo

## Problema

El modelo en tu sesión de Streamlit fue entrenado con el **código antiguo** (5 features).
Ahora el código usa **3 features**, pero Streamlit tiene el módulo cacheado en memoria.

## Evidencia

✅ **Código actualizado** (correcto):
```python
# crisis_classifier.py ahora usa:
features = ['cp_tbill_spread', 'T10Y2Y', 'NFCI']  # 3 features ✅
```

❌ **UI mostrando** (antiguo):
```
Feature | Current Value
VIX | 19.50          ← NO debería aparecer
HY_OAS | 3.13        ← NO debería aparecer
cp_tbill_spread | 0.03
T10Y2Y | 0.56
NFCI | -0.51
```

## Solución

### Opción 1: Reiniciar Streamlit (Recomendado)

1. **Detén** el servidor de Streamlit (Ctrl+C en la terminal)
2. **Limpia cache de Python**:
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
   find . -type f -name "*.pyc" -delete 2>/dev/null
   ```
3. **Inicia** Streamlit de nuevo:
   ```bash
   streamlit run macro_plumbing/app/app.py
   ```

### Opción 2: Botón Retrain en el UI

1. En el UI de Streamlit, ve al tab **"🤖 Crisis Predictor"**
2. Scroll hasta el final
3. Click en **"🔄 Retrain Model"**
4. El modelo se reentrenará con los **3 features nuevos**

### Opción 3: Borrar modelo manualmente

```bash
rm -f macro_plumbing/models/trained_crisis_predictor.pkl
```

Luego refresca el UI (F5).

## Verificación

Después de refrescar, verifica en **"🔍 Prediction Explanation"**:

### ✅ Debe mostrar SOLO 3 features:
```
Feature          | Current Value | Coefficient | Effect
cp_tbill_spread  | 0.03         | +X.XXX     | ↑ Crisis
T10Y2Y           | 0.56         | +X.XXX     | ...
NFCI             | -0.51        | +X.XXX     | ...
```

### ❌ NO debe mostrar:
- VIX
- HY_OAS

## VIF Scores (Confirmado)

Tu análisis VIF muestra **ZERO multicolinealidad**:

```
Feature           VIF    Status
cp_tbill_spread   1.24   ✅ Excelente
T10Y2Y            1.96   ✅ Excelente
NFCI              1.99   ✅ Excelente
```

Todos VIF < 2 → **Independencia perfecta** ✅

## Problema de Unidades (Separado)

También detecté un problema de **unidades** en los thresholds:

```python
# ACTUAL (en crisis_classifier.py):
cp_tbill_spread > 1.0   # Espera % decimal (1.0 = 100%)
HY_OAS > 8.0           # Espera % decimal (8.0 = 800%)

# PERO tu data está en:
cp_tbill_spread = 0.03  # 3 bps (0.03%)
HY_OAS = 3.13          # 313 bps (3.13%)
```

Esto significa que los thresholds están **100x muy altos**. Pero eso es un problema separado - primero asegúrate de tener solo 3 features.

## Resumen

1. ✅ Código correcto (3 features)
2. ✅ VIF correcto (todos < 2)
3. ❌ UI mostrando 5 features → **Reinicia Streamlit**
4. ⚠️  Unidades incorrectas → Problema separado (lo arreglaremos después)
