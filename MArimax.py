# streamlit_app.py
import os, re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Opcional (volatilidad) ---
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False

# ================= UI CONFIG =================
st.set_page_config(page_title="Macro Monitor (Streamlit)", layout="wide")
st.title("📊 Macro Monitor — Z-Score compuesto + Regímenes + Overlay")
st.caption("Compuesto con HY OAS y Term Spread | Markov | Overlay OOS | Métricas | Sensibilidad | (Opcional) GARCH")

# ================= HELPERS ===================
def is_valid_fred_key(key: str) -> bool:
    return isinstance(key, str) and re.fullmatch(r"[a-z0-9]{32}", key or "") is not None

def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.dropna().empty: return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def zscore(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window, min_periods=window)
    return (s - roll.mean()) / roll.std()

def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return (x.mean() / x.std()) if x.std() != 0 else np.nan

def sortino(x: pd.Series) -> float:
    x = x.dropna()
    dn = x[x < 0]
    sd = dn.std()
    return (x.mean() / sd) if sd and sd != 0 else np.nan

def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()

def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    dd.name = f"DD_{returns.name or 'ret'}"
    return dd

def max_drawdown(returns: pd.Series) -> float:
    return drawdown_series(returns).min()

def percent_on(signal: pd.Series) -> float:
    s = signal.dropna().astype(int)
    return s.mean()

def minmax_robust(s: pd.Series) -> pd.Series:
    q1, q99 = s.quantile(0.01), s.quantile(0.99)
    return ((s - q1) / (q99 - q1)).clip(0, 1)

def plotly_line_safe(df: pd.DataFrame, y_cols, title: str, st_obj=st):
    if isinstance(y_cols, str): y_cols = [y_cols]
    y_ok = [c for c in y_cols if c in df.columns]
    if not y_ok:
        st_obj.warning(f"No hay columnas válidas para graficar: {y_cols}")
        return
    dfx = df[y_ok].dropna(how="all")
    if dfx.empty:
        st_obj.warning(f"Sin datos válidos para graficar {y_ok}.")
        return
    if not isinstance(dfx.index, pd.DatetimeIndex):
        dfx.index = pd.to_datetime(dfx.index, errors="coerce")
    dfx = dfx.loc[dfx.index.notna()].reset_index()
    date_col = "Date" if "Date" in dfx.columns else dfx.columns[0]
    st_obj.plotly_chart(px.line(dfx, x=date_col, y=y_ok, title=title), use_container_width=True)

def plot_return_hist(df_ret: pd.DataFrame, cols: list, title: str):
    # apila retornos en formato largo
    df_long = df_ret[cols].dropna().melt(var_name="Serie", value_name="Ret")
    st.plotly_chart(px.histogram(df_long, x="Ret", color="Serie", nbins=60,
                                 barmode="overlay", opacity=0.55,
                                 title=title), use_container_width=True)

# ================= SIDEBAR ===================
with st.sidebar:
    st.header("⚙️ Parámetros")
    mode = st.radio("Modo de uso", ["Generar desde FRED", "Subir CSV ya generado"], index=0)
    freq = st.selectbox("Frecuencia", ["Semanal (W)", "Mensual (M)"], index=0)
    freq_key = "W" if freq.startswith("Semanal") else "M"
    start_date = st.date_input("Fecha inicio", value=pd.to_datetime("2010-01-01"))
    roll_z_w = st.slider("Ventana Z-score semanal (semanas)", 26, 78, 52)
    roll_z_m = st.slider("Ventana Z-score mensual (meses)", 18, 60, 36)

    st.markdown("**Overlay (0/1)**")
    thr_comp_init = st.number_input("Umbral COMPOSITE (z)", value=0.0, step=0.1, format="%.2f")
    thr_prob_init = st.number_input("Umbral Prob. Estrés Markov", value=0.40, step=0.05, format="%.2f")

    st.markdown("**Exposición continua (opcional)**")
    use_cont = st.checkbox("Usar exposición continua", value=False)
    alpha = st.slider("Peso α (COMPOSITE)", 0.0, 1.0, 0.6, 0.05)
    beta  = st.slider("Peso β (ProbEstrés)", 0.0, 1.0, 0.4, 0.05)
    ema_k = st.slider("EMA smoothing (periodos)", 1, 24, 6)

    st.markdown("**Sensibilidad**")
    do_sens = st.checkbox("Ejecutar grid de sensibilidad", value=True)
    st.caption("Prueba lags (mensual L=1–3), umbrales y ventanas de z-score.")

    st.markdown("---")
    use_garch = st.checkbox("Usar GARCH (si 'arch' está instalado)", value=False and HAVE_ARCH)
    st.caption("Con frecuencia semanal: vol semanal; anualiza con √52 si lo necesitas.")
    st.markdown("---")
    fred_key_default = st.secrets.get("FRED_API_KEY", "")
    fred_key = st.text_input("FRED API key", value=fred_key_default, type="password")
    st.caption("Configura en .streamlit/secrets.toml: FRED_API_KEY=\"...\"")

# ================ DATA =======================
@st.cache_data(show_spinner=True)
def fetch_fred_series(start_dt: pd.Timestamp, api_key: str) -> pd.DataFrame:
    if not is_valid_fred_key(api_key):
        st.error("❌ FRED API key inválida (32 chars minúscula/alfanumérica).")
        st.stop()
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    series = {
        "SOFR": "SOFR", "RRP": "RRPONTSYD", "NFCI": "NFCI", "EFFR": "EFFR",
        "OBFR": "OBFR", "SP500": "SP500", "TGA": "WTREGEN", "STLFSI4": "STLFSI4",
        "TB3MS": "TB3MS", "DGS3MO": "DGS3MO", "BAMLH0A0HYM2": "BAMLH0A0HYM2",
        "T10Y2Y": "T10Y2Y",
    }
    df, failures = pd.DataFrame(), []
    for name, code in series.items():
        try:
            s = fred.get_series(code)
            if s is not None and not s.empty: df[name] = s
            else: failures.append(name)
        except Exception:
            failures.append(name)
    if df.empty:
        st.error("❌ No se pudo descargar ninguna serie.")
        st.stop()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[df.index.notna()]
    df = df.loc[df.index >= pd.to_datetime(start_dt)]
    dfd = df.resample("D").last().ffill()
    dfd["diferencial_colateral"] = dfd["EFFR"] - dfd["SOFR"]
    dfd["sofr_spread"] = dfd["OBFR"] - dfd["SOFR"]
    dfd = dfd.rename(columns={"RRP": "Reverse_Repo_Volume", "TGA": "WTREGEN"})
    return dfd

def build_composite(dfd: pd.DataFrame, freq_key: str, roll_z_w: int, roll_z_m: int) -> pd.Series:
    FACTORES = {
        "NFCI": {"w": 0.20, "sign": +1, "diff": False},
        "STLFSI4": {"w": 0.15, "sign": +1, "diff": False},
        "BAMLH0A0HYM2": {"w": 0.20, "sign": +1, "diff": False},
        "T10Y2Y": {"w": 0.10, "sign": -1, "diff": False},
        "diferencial_colateral": {"w": 0.10, "sign": +1, "diff": False},
        "Reverse_Repo_Volume": {"w": 0.15, "sign": +1, "diff": True},
        "WTREGEN": {"w": 0.10, "sign": +1, "diff": True},
        "sofr_spread": {"w": 0.10, "sign": +1, "diff": False},
    }
    if freq_key == "W":
        df = dfd.resample("W").last().ffill(); window = roll_z_w
    else:
        df = dfd.resample("M").last().ffill(); window = roll_z_m
    cols = []
    for fac, cfg in FACTORES.items():
        if fac not in df.columns: continue
        s = df[fac].astype(float)
        if cfg["diff"]: s = s.diff()
        s = winsorize(s); z = zscore(s, window)
        cols.append(cfg["sign"] * cfg["w"] * z)
    comp = pd.concat(cols, axis=1).sum(axis=1).rename("COMPOSITE_Z")
    return comp

def composite_pca(dfd: pd.DataFrame, freq_key: str, roll_z_w: int, roll_z_m: int) -> pd.Series:
    if freq_key == "W":
        df = dfd.resample("W").last().ffill(); window = roll_z_w
    else:
        df = dfd.resample("M").last().ffill(); window = roll_z_m
    FACTORES = ["NFCI","STLFSI4","BAMLH0A0HYM2","T10Y2Y","diferencial_colateral","Reverse_Repo_Volume","WTREGEN","sofr_spread"]
    Zs, names = [], []
    for fac in FACTORES:
        if fac not in df.columns: continue
        s = df[fac].astype(float)
        if fac in ["Reverse_Repo_Volume","WTREGEN"]: s = s.diff()
        s = winsorize(s); Zs.append(zscore(s, window)); names.append(fac)
    Z = pd.concat(Zs, axis=1).dropna(); Z.columns = names
    pc1 = pd.Series(PCA(n_components=1).fit_transform(Z).ravel(), index=Z.index, name="COMPOSITE_PCA")
    return zscore(pc1, window)

def equity_premium(dfd: pd.DataFrame, freq_key: str) -> pd.Series:
    if freq_key == "M":
        sp = dfd["SP500"].resample("M").last().dropna()
        ret = np.log(sp).diff(); rf = (dfd["TB3MS"]/100/12).resample("M").last().reindex(sp.index).ffill()
    else:
        sp = dfd["SP500"].resample("W").last().dropna()
        ret = np.log(sp).diff(); rf = (dfd["DGS3MO"]/100/52).resample("W").last().reindex(sp.index).ffill()
    return (ret - rf).dropna().rename("Excess_Ret")

def markov_two_regimes(y: pd.Series, comp_l1: pd.Series):
    df = pd.concat([y, comp_l1], axis=1).dropna()
    sc = StandardScaler(); arr = sc.fit_transform(df.values)
    dfs = pd.DataFrame(arr, index=df.index, columns=[y.name, comp_l1.name])
    try:
        mod = MarkovRegression(dfs[y.name], exog=dfs[[comp_l1.name]], k_regimes=2, trend='c', switching_variance=True)
        res = mod.fit(method='lbfgs', maxiter=1000, disp=False)
        return res.smoothed_marginal_probabilities[0].rename("P_reg0"), res
    except Exception as e:
        st.warning(f"Markov error: {e}")
        return None, None

def overlay_from_thresholds(y, comp_l, prob_stress, thr_comp, thr_prob):
    sig = ~((comp_l > thr_comp) | (prob_stress > thr_prob))
    sig = sig.reindex(y.index).ffill().astype(int).rename("Overlay_Signal")
    return sig, (y * sig).rename("Ret_Filtered")

def overlay_gridsearch(y, composite, prob_stress=None,
                       comp_grid=np.arange(-0.5, 1.01, 0.05),
                       pst_grid=np.arange(0.4, 0.91, 0.05),
                       split=0.7):
    idx = y.index.intersection(composite.index)
    y = y.loc[idx].dropna(); comp = composite.reindex(y.index).ffill()
    n = len(y); cut = int(n*split)
    y_te, c_te = y.iloc[cut:], comp.iloc[cut:]
    pst_te = (prob_stress.reindex(y.index).ffill().iloc[cut:] if prob_stress is not None else pd.Series(0, index=y_te.index))
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

def continuous_exposure(y, comp_l, prob_stress, alpha=0.6, beta=0.4, ema_k=6):
    x = alpha*minmax_robust(comp_l) + beta*minmax_robust(prob_stress)
    expo = (1 - x).clip(0, 1)
    if ema_k > 1:
        expo = expo.ewm(span=ema_k, adjust=False).mean()
    expo = expo.reindex(y.index).ffill().rename("Exposure")
    return expo, (y * expo).rename("Ret_Filtered_cont")

def garch_vol_forecast(y: pd.Series, comp_l1: pd.Series):
    if not HAVE_ARCH: return None
    df = pd.concat([y, comp_l1], axis=1).dropna()
    X = sm.add_constant(df[['COMP_L1']])
    try:
        am = arch_model(df[y.name]*100, mean='ARX', lags=0, x=X[['COMP_L1']], vol='GARCH', p=1, o=0, q=1, dist='t')
        res = am.fit(disp='off')
        return (res.conditional_volatility/100.0).rename("VolForecast").reindex(y.index)
    except Exception as e:
        st.warning(f"GARCH error: {e}")
        return None

# ================ MAIN FLOW =================
if mode == "Subir CSV ya generado":
    st.subheader("📤 Cargar bundle CSV")
    up = st.file_uploader("Selecciona tu macro_monitor_bundle.csv", type=["csv"])
    if up:
        df_in = pd.read_csv(up, parse_dates=True, index_col=0)
        st.success(f"Cargado: {df_in.shape[0]} filas, {df_in.shape[1]} columnas")
        st.dataframe(df_in.tail(10))
        st.stop()
    else:
        st.info("Sube un CSV para visualizar.")
        st.stop()

st.subheader("⬇️ Descargando FRED y calculando…")
if st.button("Ejecutar pipeline"):
    # --------- DATA ----------
    with st.spinner("Obteniendo series FRED…"):
        dfd = fetch_fred_series(pd.to_datetime(start_date), api_key=fred_key)

    with st.spinner("Compuestos y equity premium…"):
        comp_w = build_composite(dfd, freq_key, roll_z_w, roll_z_m)
        comp_p = composite_pca(dfd, freq_key, roll_z_w, roll_z_m)
        common_idx = comp_w.index.union(comp_p.index)
        comp_w = comp_w.reindex(common_idx); comp_p = comp_p.reindex(common_idx)
        y = equity_premium(dfd, freq_key)

    # --------- LAGS ----------
    lag_list = [1,2,3] if freq_key == "M" else [1]
    results_rows = []
    # guardaremos retornos del mejor lag para gráficos extra
    best_pack = None
    best_sh = -np.inf

    for L in lag_list:
        comp_l = comp_w.shift(L).rename("COMP_L")
        with st.spinner(f"Markov (lag {L})…"):
            prob_reg0, _ = markov_two_regimes(y, comp_l.rename("COMP_L1"))
            prob_stress = (1 - prob_reg0) if prob_reg0 is not None else pd.Series(index=y.index, dtype=float)

        # Overlay grid OOS
        with st.spinner(f"Overlay grid OOS (lag {L})…"):
            best, signal, ret_filt = overlay_gridsearch(
                y=y, composite=comp_l, prob_stress=prob_stress,
                comp_grid=np.arange(-0.5, 1.01, 0.05),
                pst_grid=np.arange(0.4, 0.91, 0.05),
                split=0.7
            )

        # Exposición continua (opcional)
        if use_cont:
            expo, ret_cont = continuous_exposure(y, comp_l, prob_stress, alpha=alpha, beta=beta, ema_k=ema_k)

        # ====== MÉTRICAS ======
        sh_naive = sharpe(y); so_naive = sortino(y); mdd_naive = max_drawdown(y)
        sh_filt  = sharpe(ret_filt); so_filt  = sortino(ret_filt); mdd_filt  = max_drawdown(ret_filt)
        p_on = percent_on(signal); p_off = 1 - p_on

        row = {
            "Lag": L,
            "thr_comp": best["thr_comp"], "thr_prob": best["thr_prob"],
            "Sharpe_naive": sh_naive, "Sharpe_filtered": sh_filt,
            "Sortino_naive": so_naive, "Sortino_filtered": so_filt,
            "MaxDD_naive": mdd_naive, "MaxDD_filtered": mdd_filt,
            "%ON": p_on, "%OFF": p_off
        }

        if use_cont:
            sh_cont = sharpe(ret_cont); so_cont = sortino(ret_cont); mdd_cont = max_drawdown(ret_cont)
            row.update({"Sharpe_cont": sh_cont, "Sortino_cont": so_cont, "MaxDD_cont": mdd_cont})

        results_rows.append(row)

        # Guardar pack si es el mejor por Sharpe filtrado
        if sh_filt > best_sh:
            best_sh = sh_filt
            best_pack = {
                "L": L, "signal": signal, "ret_filt": ret_filt,
                "prob_reg0": prob_reg0, "comp_w": comp_w, "comp_p": comp_p
            }
            if use_cont:
                best_pack.update({"expo": expo, "ret_cont": ret_cont})

    # ====== PLOTS PRINCIPALES (del mejor lag) ======
    L = best_pack["L"]
    signal = best_pack["signal"]
    ret_filt = best_pack["ret_filt"]
    prob_reg0 = best_pack["prob_reg0"]
    comp_w = best_pack["comp_w"]; comp_p = best_pack["comp_p"]

    c1, c2 = st.columns(2)
    with c1:
        df_comp = pd.concat([comp_w.rename("COMPOSITE_Z"), comp_p.rename("COMPOSITE_PCA")], axis=1)
        plotly_line_safe(df_comp, ["COMPOSITE_Z","COMPOSITE_PCA"], "Composite (Weighted vs PCA)")
    with c2:
        if prob_reg0 is not None and not prob_reg0.dropna().empty:
            plotly_line_safe(prob_reg0.to_frame("P_reg0"), "P_reg0", f"Probabilidad Régimen 0 (calma) — Lag {L}")

    c3, c4 = st.columns(2)
    with c3:
        plotly_line_safe(signal.to_frame("Overlay_Signal"), "Overlay_Signal", f"Señal Overlay (0/1) — Lag {L}")
    with c4:
        df_ret = pd.concat([y.rename("Excess_Ret"), ret_filt], axis=1).dropna()
        df_plot = pd.DataFrame(index=df_ret.index)
        df_plot["EQ_naive"] = equity_curve(df_ret["Excess_Ret"])
        df_plot["EQ_filtered"] = equity_curve(df_ret["Ret_Filtered"])
        if use_cont and "ret_cont" in best_pack:
            df_plot["EQ_cont"] = equity_curve(best_pack["ret_cont"])
        plotly_line_safe(df_plot, list(df_plot.columns), "Curva de capital")

    # ====== GRÁFICOS EXTRA: Distribución & Drawdown ======
    st.markdown("### 📊 Distribución de retornos (hist)")
    ret_cols = ["Excess_Ret", "Ret_Filtered"]
    if use_cont and "ret_cont" in best_pack: ret_cols += ["Ret_Filtered_cont"]
    plot_return_hist(pd.concat([y.rename("Excess_Ret"), ret_filt.rename("Ret_Filtered"),
                                (best_pack.get("ret_cont") or pd.Series(dtype=float)).rename("Ret_Filtered_cont")], axis=1),
                     [c for c in ret_cols if c is not None], "Distribución de retornos")

    st.markdown("### 📉 Drawdowns")
    dd_df = pd.DataFrame({
        "DD_naive": drawdown_series(y),
        "DD_filtered": drawdown_series(ret_filt)
    })
    if use_cont and "ret_cont" in best_pack:
        dd_df["DD_cont"] = drawdown_series(best_pack["ret_cont"])
    plotly_line_safe(dd_df, list(dd_df.columns), "Curva de drawdown (pico a valle)")

    # ====== TABLA DE RESULTADOS ======
    res_df = pd.DataFrame(results_rows)
    st.markdown("### 📈 Métricas por lag")
    st.dataframe(res_df.set_index("Lag").round(3))

    # ====== SENSIBILIDAD (opcional): ventanas de z-score ======
    if do_sens:
        st.markdown("### 🔎 Sensibilidad: ventanas de z-score y lags")
        win_list = ([24,36,48,60] if freq_key == "M" else [26,39,52,65,78])
        sens_rows = []
        for w in win_list:
            comp_tmp = build_composite(dfd, freq_key, roll_z_w=w if freq_key=="W" else roll_z_w,
                                       roll_z_m=w if freq_key=="M" else roll_z_m)
            for L in ([1,2,3] if freq_key=="M" else [1]):
                comp_l = comp_tmp.shift(L)
                prob_reg0, _ = markov_two_regimes(y, comp_l.rename("COMP_L1"))
                prob_stress = (1 - prob_reg0) if prob_reg0 is not None else pd.Series(index=y.index, dtype=float)
                best, _, ret_filt_tmp = overlay_gridsearch(y, comp_l, prob_stress,
                                                           comp_grid=np.arange(-0.5, 1.01, 0.1),
                                                           pst_grid=np.arange(0.4, 0.91, 0.1),
                                                           split=0.7)
                sens_rows.append({"window": w, "lag": L, "Sharpe_filtered": sharpe(ret_filt_tmp),
                                  "thr_comp": best["thr_comp"], "thr_prob": best["thr_prob"]})
        st.dataframe(pd.DataFrame(sens_rows).sort_values("Sharpe_filtered", ascending=False).round(3))

    # ====== DESCARGA BUNDLE (mejor lag) ======
    volf = garch_vol_forecast(y, comp_w.shift(best_pack["L"]).rename("COMP_L1")) if use_garch else None
    bundle = pd.concat({
        "COMPOSITE_Z": comp_w,
        "COMPOSITE_PCA": comp_p,
        "COMP_L": comp_w.shift(best_pack["L"]),
        "P_reg0": prob_reg0 if prob_reg0 is not None else pd.Series(index=y.index, dtype=float),
        "ProbStress": (1 - prob_reg0) if prob_reg0 is not None else pd.Series(index=y.index, dtype=float),
        "Overlay_Signal": signal.astype(int),
        "Ret_Filtered": ret_filt,
        "Excess_Ret": y,
        "VolForecast": volf if volf is not None else pd.Series(index=y.index, dtype=float),
        "DD_naive": drawdown_series(y),
        "DD_filtered": drawdown_series(ret_filt),
    }, axis=1)

    st.markdown("### 📦 Descargar `macro_monitor_bundle.csv`")
    st.download_button("💾 Descargar CSV",
        data=bundle.to_csv(index=True).encode("utf-8"),
        file_name="macro_monitor_bundle.csv",
        mime="text/csv")
else:
    st.info("Configura la API key de FRED, ajusta parámetros y haz clic en **Ejecutar pipeline**.")
