import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Opcional (volatilidad) ---
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False

# ----------------- UI CONFIG -----------------
st.set_page_config(page_title="Macro Monitor (Streamlit)", layout="wide")

st.title("📊 Macro Monitor — Z-Score compuesto + Regímenes + Overlay")
st.caption("Inversiones - Macro | Compuesto con HY OAS y Term Spread | Markov | Overlay OOS | (Opcional) GARCH")

# ----------------- SIDEBAR -------------------
with st.sidebar:
    st.header("⚙️ Parámetros")
    mode = st.radio("Modo de uso", ["Generar desde FRED", "Subir CSV ya generado"], index=0)
    freq = st.selectbox("Frecuencia", ["Semanal (W)", "Mensual (M)"], index=0)
    freq_key = "W" if freq.startswith("Semanal") else "M"
    start_date = st.date_input("Fecha inicio", value=pd.to_datetime("2010-01-01"))
    roll_z_w = st.slider("Ventana Z-score semanal (semanas)", 26, 78, 52)
    roll_z_m = st.slider("Ventana Z-score mensual (meses)", 18, 60, 36)
    st.markdown("---")
    st.markdown("**Umbrales iniciales (puedes optimizarlos luego)**")
    thr_comp_init = st.number_input("Umbral COMPOSITE (z)", value=0.0, step=0.1, format="%.2f")
    thr_prob_init = st.number_input("Umbral Prob. Estrés Markov", value=0.40, step=0.05, format="%.2f")
    st.markdown("---")
    use_garch = st.checkbox("Usar GARCH (si 'arch' está instalado)", value=False and HAVE_ARCH)
    annual_target_vol = st.number_input("Target Vol anual (p.ej. 0.15)", value=0.15, step=0.01, format="%.2f")
    st.markdown("---")
    st.info("🔑 Agrega tu FRED API key en `st.secrets['FRED_API_KEY']`", icon="🔐")
    st.caption("En Streamlit Cloud, configúralo en la sección Secrets.\nLocal: crea `.streamlit/secrets.toml` con FRED_API_KEY=\"tu_key\"")

# ----------------- UTILS ---------------------
def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def zscore(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window, min_periods=window)
    return (s - roll.mean()) / roll.std()

def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return (x.mean() / x.std()) if x.std() != 0 else np.nan

# ----------------- DATA ----------------------
@st.cache_data(show_spinner=True)
def fetch_fred_series(start_dt: pd.Timestamp) -> pd.DataFrame:
    from fredapi import Fred
    fred = Fred(api_key=st.secrets.get("qGDE52LhIJ9CQSyRwKpAzjLXeLP4Pwkt", ""))
    series = {
        "SOFR": "SOFR",
        "RRP": "RRPONTSYD",
        "NFCI": "NFCI",
        "EFFR": "EFFR",
        "OBFR": "OBFR",
        "SP500": "SP500",
        "TGA": "WTREGEN",
        "STLFSI4": "STLFSI4",
        "TB3MS": "TB3MS",
        "DGS3MO": "DGS3MO",
        "BAMLH0A0HYM2": "BAMLH0A0HYM2",  # HY OAS (%)
        "T10Y2Y": "T10Y2Y",               # 10y-2y (pp)
    }
    df = pd.DataFrame()
    for name, code in series.items():
        try:
            df[name] = fred.get_series(code)
        except Exception as e:
            st.warning(f"FRED {name}: {e}")
    df.index.name = "Date"
    df = df.loc[df.index >= pd.to_datetime(start_dt)]
    dfd = df.resample("D").last().ffill()
    dfd["diferencial_colateral"] = dfd["EFFR"] - dfd["SOFR"]
    dfd["sofr_spread"] = dfd["OBFR"] - dfd["SOFR"]
    dfd = dfd.rename(columns={"RRP": "Reverse_Repo_Volume", "TGA": "WTREGEN"})
    return dfd

def build_composite(dfd: pd.DataFrame, freq_key: str, roll_z_w: int, roll_z_m: int) -> pd.Series:
    FACTORES = {
        "NFCI":                  {"w": 0.20, "sign": +1, "diff": False, "winsor": True},
        "STLFSI4":               {"w": 0.15, "sign": +1, "diff": False, "winsor": True},
        "BAMLH0A0HYM2":          {"w": 0.20, "sign": +1, "diff": False, "winsor": True},
        "T10Y2Y":                {"w": 0.10, "sign": -1, "diff": False, "winsor": True},
        "diferencial_colateral": {"w": 0.10, "sign": +1, "diff": False, "winsor": True},
        "Reverse_Repo_Volume":   {"w": 0.15, "sign": +1, "diff": True,  "winsor": True},
        "WTREGEN":               {"w": 0.10, "sign": +1, "diff": True,  "winsor": True},
        "sofr_spread":           {"w": 0.10, "sign": +1, "diff": False, "winsor": True},
    }
    if freq_key == "W":
        df = dfd.resample("W").last().ffill()
        window = roll_z_w
    else:
        df = dfd.resample("M").last().ffill()
        window = roll_z_m

    cols = []
    for fac, cfg in FACTORES.items():
        if fac not in df.columns:
            continue
        s = df[fac].astype(float)
        if cfg.get("diff", False):
            s = s.diff()
        if cfg.get("winsor", True):
            s = winsorize(s)
        z = zscore(s, window)
        cols.append(cfg["sign"] * cfg["w"] * z)
    comp = pd.concat(cols, axis=1).sum(axis=1)
    comp.name = "COMPOSITE_Z"
    return comp

def composite_pca(dfd: pd.DataFrame, freq_key: str, roll_z_w: int, roll_z_m: int) -> pd.Series:
    if freq_key == "W":
        df = dfd.resample("W").last().ffill()
        window = roll_z_w
    else:
        df = dfd.resample("M").last().ffill()
        window = roll_z_m

    FACTORES = ["NFCI","STLFSI4","BAMLH0A0HYM2","T10Y2Y","diferencial_colateral","Reverse_Repo_Volume","WTREGEN","sofr_spread"]
    Zs = []
    names = []
    for fac in FACTORES:
        if fac not in df.columns: 
            continue
        s = df[fac].astype(float)
        if fac in ["Reverse_Repo_Volume","WTREGEN"]:
            s = s.diff()
        s = winsorize(s)
        Zs.append(zscore(s, window))
        names.append(fac)
    Z = pd.concat(Zs, axis=1).dropna()
    Z.columns = names
    pc1 = pd.Series(PCA(n_components=1).fit_transform(Z).ravel(), index=Z.index, name="COMPOSITE_PCA")
    pc1 = zscore(pc1, window)
    return pc1

def equity_premium(dfd: pd.DataFrame, freq_key: str) -> pd.Series:
    if freq_key == "M":
        sp = dfd["SP500"].resample("M").last().dropna()
        ret = np.log(sp).diff()
        rf = (dfd["TB3MS"]/100/12).resample("M").last().reindex(sp.index).ffill()
    else:
        sp = dfd["SP500"].resample("W").last().dropna()
        ret = np.log(sp).diff()
        rf = (dfd["DGS3MO"]/100/52).resample("W").last().reindex(sp.index).ffill()
    y = (ret - rf).dropna()
    y.name = "Excess_Ret"
    return y

def markov_two_regimes(y: pd.Series, comp_l1: pd.Series):
    df = pd.concat([y, comp_l1], axis=1).dropna()
    sc = StandardScaler()
    arr = sc.fit_transform(df.values)
    dfs = pd.DataFrame(arr, index=df.index, columns=[y.name, comp_l1.name])
    try:
        mod = MarkovRegression(dfs[y.name], exog=dfs[[comp_l1.name]], k_regimes=2, trend='c', switching_variance=True)
        res = mod.fit(method='lbfgs', maxiter=1000, disp=False)
        prob_reg0 = res.smoothed_marginal_probabilities[0]
        return prob_reg0.rename("P_reg0"), res
    except Exception as e:
        st.warning(f"Markov error: {e}")
        return None, None

def overlay_gridsearch(y, composite, prob_stress=None,
                       comp_grid=np.arange(-0.5, 1.01, 0.05),
                       pst_grid=np.arange(0.4, 0.91, 0.05),
                       split=0.7):
    idx = y.index.intersection(composite.index)
    y = y.loc[idx].dropna()
    comp = composite.reindex(y.index).ffill()

    n = len(y)
    cut = int(n*split)
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
    c_tr,  c_te = comp.iloc[:cut], comp.iloc[cut:]

    pst_tr = pst_te = pd.Series(False, index=y.index)
    if prob_stress is not None:
        pst = prob_stress.reindex(y.index).ffill()
        pst_tr, pst_te = pst.iloc[:cut], pst.iloc[cut:]

    best = {"sh_te": -np.inf, "thr_comp": None, "thr_prob": None}
    for tc in comp_grid:
        for tp in (pst_grid if prob_stress is not None else [1.1]):
            sig_te = ~((c_te > tc) | (pst_te > tp))
            sh_te  = sharpe(y_te * sig_te.astype(int))
            if sh_te > best["sh_te"]:
                best = {"sh_te": sh_te, "thr_comp": tc, "thr_prob": None if prob_stress is None else tp}

    tc = best["thr_comp"]; tp = best["thr_prob"] if best["thr_prob"] is not None else 1.1
    pst_all = (prob_stress.reindex(y.index).ffill() if prob_stress is not None else pd.Series(0, index=y.index))
    sig_all = ~((comp > tc) | (pst_all > tp))
    ret_filt = y * sig_all.astype(int)

    return best, sig_all.rename("Overlay_Signal"), ret_filt.rename("Ret_Filtered")

def garch_vol_forecast(y: pd.Series, comp_l1: pd.Series):
    if not HAVE_ARCH:
        return None
    df = pd.concat([y, comp_l1], axis=1).dropna()
    X = sm.add_constant(df[['COMP_L1']])
    try:
        am = arch_model(df[y.name]*100, mean='ARX', lags=0, x=X[['COMP_L1']],
                        vol='GARCH', p=1, o=0, q=1, dist='t')
        res = am.fit(disp='off')
        vol = (res.conditional_volatility/100.0).rename("VolForecast")
        return vol.reindex(y.index)
    except Exception as e:
        st.warning(f"GARCH error: {e}")
        return None

# ----------------- MAIN FLOW ----------------
if mode == "Subir CSV ya generado":
    st.subheader("📤 Cargar bundle CSV")
    up = st.file_uploader("Selecciona tu macro_monitor_bundle.csv", type=["csv"])
    if up:
        df_in = pd.read_csv(up, parse_dates=True, index_col=0)
        st.success(f"Cargado: {df_in.shape[0]} filas, {df_in.shape[1]} columnas")
        st.dataframe(df_in.tail(10))
        # Gráficos rápidos
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(df_in.reset_index(), x=df_in.index.name, y=["COMPOSITE_Z","COMPOSITE_PCA"], title="Composite (Weighted vs PCA)"), use_container_width=True)
        with c2:
            if "P_reg0" in df_in.columns:
                st.plotly_chart(px.line(df_in.reset_index(), x=df_in.index.name, y="P_reg0", title="Prob. Régimen 0 (calma)"), use_container_width=True)
        # KPIs
        if {"Ret_Filtered","Excess_Ret"}.issubset(df_in.columns):
            k1, k2, k3 = st.columns(3)
            sharpe_naive = sharpe(df_in["Excess_Ret"])
            sharpe_filt  = sharpe(df_in["Ret_Filtered"])
            k1.metric("Sharpe naive", f"{sharpe_naive:.3f}")
            k2.metric("Sharpe filtrado", f"{sharpe_filt:.3f}")
            k3.metric("Mejora", f"{(sharpe_filt - sharpe_naive):.3f}")
        st.stop()
    else:
        st.info("Sube un CSV para visualizar.")
        st.stop()

# --- Generar desde FRED ---
st.subheader("⬇️ Descargando FRED y calculando…")
if st.button("Ejecutar pipeline"):
    with st.spinner("Obteniendo series FRED…"):
        dfd = fetch_fred_series(pd.to_datetime(start_date))

    with st.spinner("Construyendo compuestos y equity premium…"):
        comp_w = build_composite(dfd, freq_key, roll_z_w, roll_z_m)
        comp_p = composite_pca(dfd, freq_key, roll_z_w, roll_z_m)
        y = equity_premium(dfd, freq_key)

    # Lag 3 sugerido por Granger
    comp_l3 = comp_w.shift(3).rename("COMP_L3").reindex(y.index)

    with st.spinner("Markov (2 regímenes)…"):
        prob_reg0, ms_res = markov_two_regimes(y, comp_l3.rename("COMP_L1"))
        prob_stress = (1 - prob_reg0) if prob_reg0 is not None else None

    with st.spinner("Overlay OOS (grid)…"):
        best, signal, ret_filt = overlay_gridsearch(
            y=y,
            composite=comp_l3,          # usamos L3
            prob_stress=prob_stress,
            comp_grid=np.arange(-0.5, 1.01, 0.05),
            pst_grid=np.arange(0.4, 0.91, 0.05),
            split=0.7
        )
    st.success(f"Overlay óptimo OOS → Sharpe={best['sh_te']:.3f} | thr_comp={best['thr_comp']} | thr_prob={best['thr_prob']}")

    volf = None
    if use_garch:
        with st.spinner("Pronóstico de volatilidad (GARCH)…"):
            volf = garch_vol_forecast(y, comp_l3.rename("COMP_L1"))

    # KPIs
    sharpe_naive = sharpe(y)
    sharpe_filtered = sharpe(ret_filt)
    k1, k2, k3 = st.columns(3)
    k1.metric("Sharpe naive", f"{sharpe_naive:.3f}")
    k2.metric("Sharpe filtrado (overlay)", f"{sharpe_filtered:.3f}")
    k3.metric("Mejora", f"{(sharpe_filtered - sharpe_naive):.3f}")

    # Gráficos
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(pd.concat([comp_w, comp_p], axis=1).reset_index(),
                                x="Date", y=["COMPOSITE_Z","COMPOSITE_PCA"],
                                title="Composite (Weighted vs PCA)"), use_container_width=True)
    with c2:
        if prob_reg0 is not None:
            st.plotly_chart(px.line(prob_reg0.reset_index(), x="Date", y="P_reg0",
                                    title="Probabilidad Régimen 0 (calma)"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.line(signal.reset_index(), x="Date", y="Overlay_Signal", title="Señal Overlay (0/1)"), use_container_width=True)
    with c4:
        df_ret = pd.concat([y.rename("Excess_Ret"), ret_filt], axis=1).dropna()
        df_ret["EQ_naive"] = (1 + df_ret["Excess_Ret"]).cumprod()
        df_ret["EQ_filtered"] = (1 + df_ret["Ret_Filtered"]).cumprod()
        st.plotly_chart(px.line(df_ret.reset_index(), x="Date", y=["EQ_naive","EQ_filtered"], title="Curva de capital"), use_container_width=True)

    # --- Construir bundle y descarga ---
    bundle = pd.concat({
        "COMPOSITE_Z": comp_w,
        "COMPOSITE_PCA": comp_p,
        "COMP_L3": comp_l3,
        "P_reg0": prob_reg0 if prob_reg0 is not None else pd.Series(index=y.index, dtype=float),
        "Overlay_Signal": signal.astype(int),
        "VolForecast": volf if volf is not None else pd.Series(index=y.index, dtype=float),
        "Ret_Filtered": ret_filt,
        "Excess_Ret": y
    }, axis=1)

    st.markdown("### 📦 Descargar `macro_monitor_bundle.csv`")
    st.download_button(
        label="💾 Descargar CSV",
        data=bundle.to_csv(index=True).encode("utf-8"),
        file_name="macro_monitor_bundle.csv",
        mime="text/csv"
    )

    st.dataframe(bundle.tail(10))
else:
    st.info("Configura la API key de FRED, ajusta parámetros y haz clic en **Ejecutar pipeline**.")