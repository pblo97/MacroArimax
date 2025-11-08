# 🚨 INSTRUCCIONES URGENTES: Cómo Ver el Fix

## El Problema

El **FIX YA ESTÁ EN MAIN** ✅ pero Streamlit Cloud muestra la versión vieja en cache.

He agregado debug prints que dirán:
```
🔄 Visualization module loaded - VERSION 2.0 (Fixed node overlap)
📊 Insurance_Pensions: balance=35000.0B → size=33.0px (LOG SCALE ✓)
```

## ⚡ Solución Inmediata (Elige UNA opción)

### Opción 1: Reboot en Streamlit Cloud (2 minutos)

1. **Ve a**: https://share.streamlit.io/
2. **Encuentra** tu app "MacroArimax"
3. **Click** en ⋮ (tres puntos) o ⚙️ Settings
4. **Click "Reboot app"**
5. **Espera 2-3 minutos**
6. **Refresca el browser** (Ctrl+Shift+R o Cmd+Shift+R)

✅ **Verifica**: Deberías ver en los logs de Streamlit el mensaje:
```
🔄 Visualization module loaded - VERSION 2.0
```

---

### Opción 2: Clear Cache del Browser

1. **Abre** tu app en Streamlit
2. **Presiona**: 
   - Windows/Linux: `Ctrl + Shift + R`
   - Mac: `Cmd + Shift + R`
3. **Si eso no funciona**:
   - Chrome: `Ctrl+Shift+Delete` → Clear cache → Reload
   - Firefox: `Ctrl+Shift+Delete` → Clear cache → Reload

---

### Opción 3: Clear Streamlit Cache (desde la app)

1. **Abre** tu app
2. **Presiona** `c` en el teclado
3. **Click "Clear cache"**
4. **Reload** la página

---

## Cómo Verificar que Funcionó

### ✅ En los Logs de Streamlit Cloud:

Deberías ver:
```
🔄 Visualization module loaded - VERSION 2.0 (Fixed node overlap)
📊 Fed: balance=3493.4B → size=27.7px (LOG SCALE ✓)
📊 Banks: balance=3493.4B → size=27.7px (LOG SCALE ✓)  
📊 Hedge_Funds: balance=4000.0B → size=28.0px (LOG SCALE ✓)
📊 Asset_Managers: balance=25000.0B → size=32.0px (LOG SCALE ✓)
📊 Insurance_Pensions: balance=35000.0B → size=32.7px (LOG SCALE ✓)
```

### ✅ En el Grafo:

- Todos los nodos visibles (tamaños 24-35px)
- No hay burbujas gigantes
- Puedes ver todos los nombres de nodos
- Más espacio entre nodos

### ❌ Si SIGUE MAL:

En los logs verías tamaños como:
```
Insurance_Pensions: balance=35000.0B → size=1181.0px  ❌
```

**Esto significa**: Streamlit está usando versión vieja en cache.

---

## Ver los Logs en Streamlit Cloud

1. **Dashboard** → Tu app
2. **Click** en "Manage app" 
3. **Scroll down** a "Logs"
4. **Busca** el mensaje `🔄 Visualization module loaded`

---

## Si NADA Funciona

Último recurso (100% efectivo):

1. **Delete** la app en Streamlit Cloud
2. **Re-deploy** desde cero:
   - Repository: pblo97/MacroArimax
   - Branch: `main`
   - Main file: macro_plumbing/app/app.py
3. **Deploy**

Esto forzará a Streamlit a descargar todo de nuevo.

---

## Resumen Técnico

**El código correcto YA ESTÁ en main** (verificado):
- Línea 114: `size = 10 + 5 * math.log10(abs(balance) + 1)`  
- Línea 115: `final_size = min(size, 30)`

**El problema es SOLO cache/deployment**, no el código.

---

**Próximo paso**: Reboot app en Streamlit Cloud y espera 3 minutos.
