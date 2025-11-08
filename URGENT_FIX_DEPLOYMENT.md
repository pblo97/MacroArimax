# ⚠️ URGENT: Cómo Desplegar el Fix de Burbujas Gigantes

## El Problema

Tu Streamlit Cloud está apuntando al branch `main`, pero el **FIX de las burbujas gigantes** está en:
- Branch: `claude/liquidity-stress-detection-system-011CUoKdxAbMy1259QPRQkZV`
- Commit: `9cdd600` - "Fix graph visualization: prevent giant node overlap"

## Solución Inmediata (3 Opciones)

### 🚀 Opción 1: Cambiar Branch en Streamlit Cloud (MÁS RÁPIDO - 2 minutos)

1. **Ve a**: https://share.streamlit.io/ (o tu dashboard de Streamlit Cloud)
2. **Encuentra tu app**: MacroArimax
3. **Click** en ⚙️ Settings (esquina superior derecha)
4. **Advanced Settings** → **Branch**
5. **Cambia de** `main` **a**: `claude/liquidity-stress-detection-system-011CUoKdxAbMy1259QPRQkZV`
6. **Click "Save"**
7. **Espera 2-3 minutos** para redespliegue

✅ **Resultado**: La app se redesplegará con el fix y verás los nodos correctamente dimensionados.

---

### 📋 Opción 2: Crear Pull Request (RECOMENDADO - 5 minutos)

1. **Ve a tu repositorio** en GitHub/GitLab
2. **Click "New Pull Request"**
3. **Configuración**:
   - Base: `main`
   - Compare: `claude/liquidity-stress-detection-system-011CUoKdxAbMy1259QPRQkZV`
4. **Título**: "Fix: Prevent giant node overlap in graph visualization"
5. **Descripción**: (opcional - puedes copiar de DEPLOYMENT_INSTRUCTIONS.md)
6. **Create Pull Request**
7. **Merge Pull Request** (si tienes permisos)
8. **Espera redespliegue automático** (2-3 minutos)

✅ **Resultado**: Los cambios quedan en `main` permanentemente y la app se actualiza.

---

### 💻 Opción 3: Merge Local (Si tienes acceso git - 3 minutos)

En tu computadora local:

```bash
# 1. Ir a main
git checkout main

# 2. Traer últimos cambios
git pull origin main

# 3. Hacer merge del fix
git merge claude/liquidity-stress-detection-system-011CUoKdxAbMy1259QPRQkZV

# 4. Pushear a main
git push origin main
```

✅ **Resultado**: Los cambios quedan en `main` y Streamlit Cloud se actualiza automáticamente.

---

## Qué Verás Después del Fix

### ANTES (ACTUAL - MAL) ❌:
```
Insurance_Pensions (35,000B) → 1,181 pixels 😱
  Cubre TODO el grafo
  No se ven otros nodos
```

### DESPUÉS (CON FIX) ✅:
```
Fed (3,493B)              → 28px  ✅
Treasury (635B)           → 24px  ✅
Banks (3,493B)            → 28px  ✅
Dealers (1,000B)          → 25px  ✅
Hedge_Funds (4,000B)      → 28px  ✅
Asset_Managers (25,000B)  → 32px  ✅
Insurance_Pensions (35,000B) → 33px  ✅

Todos los nodos visibles, sin superposición
```

## Cómo Verificar que Funcionó

1. **Abre tu app** en Streamlit Cloud
2. **Ve a** "Análisis Avanzado de Red de Liquidez" (Tab 3)
3. **Deberías ver**:
   - ✅ Todos los nodos visibles (no burbujas gigantes)
   - ✅ Nodos entre 24-33 pixels
   - ✅ Más espacio entre nodos
   - ✅ Grafo más alto (800px)
   - ✅ Banner: "🚀 Showing Enhanced Graph with all 4 phases"

## Si Sigue Sin Funcionar

1. **Clear cache del browser**: Ctrl+Shift+R (Chrome) o Cmd+Shift+R (Mac)
2. **Reboot app** en Streamlit Cloud: Settings → Reboot
3. **Verificar branch** en Streamlit Cloud: Settings → Advanced → Branch debe ser el feature branch o main (con merge)

## Archivos Modificados en el Fix

- `macro_plumbing/graph/visualization.py`:
  - Línea 30: Spring layout k=3 (vs k=2)
  - Línea 48: Edge width /100 (vs /50)
  - Línea 106-111: Logarithmic node sizing
  - Línea 141-145: Better margins + aspect ratio
  - Línea 211-212: Edge width reduced
  - Línea 313-319: Logarithmic sizing (enhanced)
  - Línea 442-462: Wider layout spacing

## Urgencia

⚠️ **ALTA**: El grafo actual es inutilizable por las burbujas gigantes.

✅ **Tiempo estimado para fix**: 2-5 minutos con cualquier opción.

---

**Contacto**: Si ninguna opción funciona, responde con un screenshot del error específico.
