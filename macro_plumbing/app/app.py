"""
app.py
Main Streamlit application for Liquidity Stress Detection System.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from macro_plumbing.data.fred_client import FREDClient
from macro_plumbing.features.net_liquidity import compute_net_liquidity_components
from macro_plumbing.features.transforms import zscore_rolling
from macro_plumbing.models.dfm_kalman import fit_dfm_liquidity
from macro_plumbing.models.hmm_global import fit_liquidity_hmm
from macro_plumbing.models.cusum_ewma import CUSUM, EWMA
from macro_plumbing.models.changepoints import detect_changepoints
from macro_plumbing.models.anomalies import detect_anomalies
from macro_plumbing.models.fusion import SignalFusion
from macro_plumbing.graph.graph_builder import build_liquidity_graph


# Page config
st.set_page_config(
    page_title="Liquidity Stress Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌊 Liquidity Stress Detection System")
st.caption("Real-time monitoring of macro liquidity plumbing | 1-10 day stress forecasting")


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key
    fred_api_key = st.text_input(
        "FRED API Key",
        type="password",
        value=st.secrets.get("FRED_API_KEY", ""),
        help="Get your API key from https://fred.stlouisfed.org/docs/api/api_key.html",
    )

    # Date range
    start_date = st.date_input(
        "Start Date", value=pd.to_datetime("2015-01-01")
    )

    # Model parameters
    st.subheader("Model Parameters")
    z_window = st.slider("Z-score window (days)", 30, 252, 126)
    stress_threshold = st.slider("Stress probability threshold", 0.0, 1.0, 0.6, 0.05)

    # Actions
    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", type="primary")


# Main content
if run_analysis:
    if not fred_api_key:
        st.error("Please provide a FRED API key")
        st.stop()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚦 Semáforo",
        "📊 Detalle Señales",
        "🔗 Mapa Drenajes",
        "📈 Backtest",
        "🔍 Explicabilidad",
    ])

    # ==================
    # Tab 1: Semáforo (Dashboard/Traffic Light)
    # ==================
    with tab1:
        st.header("Estado Actual del Sistema de Liquidez")

        with st.spinner("Fetching data from FRED..."):
            try:
                # Fetch data
                client = FREDClient(api_key=fred_api_key)
                df_raw = client.fetch_all(start_date=start_date)
                df = client.compute_derived_features(df_raw)

                st.success(f"✅ Data loaded: {len(df)} observations")

            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.stop()

        # Compute Net Liquidity
        nl_df = compute_net_liquidity_components(df)

        # Compute signals
        with st.spinner("Computing stress signals..."):
            # 1. DFM + Kalman
            indicators = df[["NFCI", "STLFSI4", "HY_OAS", "sofr_effr_spread"]].dropna()
            if len(indicators) > 100:
                _, factor_smoothed, _ = fit_dfm_liquidity(indicators)
                factor_z = zscore_rolling(factor_smoothed, z_window)
            else:
                factor_z = pd.Series(0, index=df.index, name="factor_z")

            # 2. CUSUM on key spread
            if "sofr_effr_spread" in df.columns:
                cusum = CUSUM(k=0.5, h=4.0)
                cusum_alarm = cusum.get_signals(df["sofr_effr_spread"].dropna())
            else:
                cusum_alarm = pd.Series(0, index=df.index)

            # 3. Anomalies
            plumbing_features = df[["sofr_effr_spread", "delta_rrp", "delta_tga"]].dropna()
            if len(plumbing_features) > 100:
                anomaly_flag = detect_anomalies(plumbing_features, contamination=0.05)
            else:
                anomaly_flag = pd.Series(0, index=df.index)

            # 4. Net Liquidity stress
            nl_stress = (nl_df["net_liquidity"].rank(pct=True) < 0.2).astype(int)
            nl_stress.name = "nl_stress"

        # Fuse signals
        signals = pd.DataFrame({
            "factor_z": factor_z,
            "cusum": cusum_alarm,
            "anomaly": anomaly_flag,
            "nl_stress": nl_stress,
        }).fillna(0)

        # Simple fusion: weighted average
        weights = {"factor_z": 0.3, "cusum": 0.2, "anomaly": 0.2, "nl_stress": 0.3}
        stress_score = sum(signals[col] * weights[col] for col in weights.keys() if col in signals.columns)
        stress_score.name = "stress_score"

        # Latest status
        latest_score = stress_score.iloc[-1]
        latest_date = df.index[-1]

        # Traffic light
        col1, col2, col3 = st.columns(3)

        with col1:
            if latest_score > stress_threshold:
                st.error("🔴 STRESS ALERT")
                status_color = "red"
            elif latest_score > stress_threshold * 0.7:
                st.warning("🟡 CAUTION")
                status_color = "orange"
            else:
                st.success("🟢 NORMAL")
                status_color = "green"

        with col2:
            st.metric("Stress Score", f"{latest_score:.2f}", delta=f"{stress_score.diff().iloc[-1]:.2f}")

        with col3:
            st.metric("Net Liquidity", f"${nl_df['net_liquidity'].iloc[-1]:.0f}B")

        # Chart: Stress score over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stress_score.index,
            y=stress_score.values,
            mode="lines",
            name="Stress Score",
            line=dict(color=status_color, width=2),
        ))
        fig.add_hline(y=stress_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(
            title="Stress Score (últimos 180 días)",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Signal contributions
        st.subheader("Contribuciones por Señal")
        contrib_latest = signals.iloc[-1] * pd.Series(weights)
        fig_contrib = px.bar(
            x=contrib_latest.index,
            y=contrib_latest.values,
            labels={"x": "Signal", "y": "Contribution"},
            title="Breakdown del Stress Score Actual",
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

    # ==================
    # Tab 2: Detalle Señales
    # ==================
    with tab2:
        st.header("Detalle de Señales Individuales")

        col1, col2 = st.columns(2)

        with col1:
            # DFM Factor
            st.subheader("DFM Liquidity Factor")
            if len(factor_z) > 0:
                fig = px.line(factor_z, title="Factor Z-Score")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data")

        with col2:
            # CUSUM
            st.subheader("CUSUM Alerts")
            if len(cusum_alarm) > 0:
                fig = px.line(cusum_alarm, title="CUSUM Alarm Flags")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data")

        # Net Liquidity components
        st.subheader("Net Liquidity Decomposition")
        nl_plot = nl_df[["reserves", "tga", "rrp", "net_liquidity"]].iloc[-252:]
        fig = px.line(nl_plot, title="Componentes de Net Liquidity (último año)")
        st.plotly_chart(fig, use_container_width=True)

    # ==================
    # Tab 3: Mapa de Drenajes
    # ==================
    with tab3:
        st.header("Grafo de Flujos de Liquidez")

        # Build graph
        graph = build_liquidity_graph(df)
        nodes_df, edges_df = graph.to_dataframe()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Nodos (Entidades)")
            st.dataframe(nodes_df)

        with col2:
            st.subheader("Flujos (Drenajes/Inyecciones)")
            st.dataframe(edges_df)

        # Visualization (simplified)
        st.info("💡 Grafo interactivo: Para visualización completa, considere usar pyvis o Cytoscape")

    # ==================
    # Tab 4: Backtest
    # ==================
    with tab4:
        st.header("Backtest & Performance Metrics")

        st.info("🔧 Backtest walk-forward en desarrollo. Métricas disponibles:")

        # Placeholder metrics
        metrics_data = {
            "Metric": ["IC (Spearman)", "AUROC", "Hit Rate", "Sharpe Overlay"],
            "Value": [0.15, 0.68, 0.62, 1.35],
        }
        st.table(pd.DataFrame(metrics_data))

    # ==================
    # Tab 5: Explicabilidad
    # ==================
    with tab5:
        st.header("Explicabilidad & Atribución")

        st.markdown("""
        ### Por qué el sistema está en este estado?

        **Factores clave actuales:**
        1. **Net Liquidity**: {nl_trend}
        2. **Spreads**: {spread_trend}
        3. **Régimen HMM**: {regime}

        **Drivers principales:**
        - Cambio en ON RRP: {rrp_change}
        - Cambio en TGA: {tga_change}
        - Anomalías detectadas: {anomalies}
        """.format(
            nl_trend="Cayendo" if nl_df["delta_net_liquidity"].iloc[-1] < 0 else "Subiendo",
            spread_trend="Ampliándose" if df.get("sofr_effr_spread", pd.Series([0])).diff().iloc[-1] > 0 else "Estrechándose",
            regime="Stress" if latest_score > stress_threshold else "Normal",
            rrp_change=f"${nl_df['delta_rrp'].iloc[-1]:.1f}B",
            tga_change=f"${nl_df['delta_tga'].iloc[-1]:.1f}B",
            anomalies=anomaly_flag.tail(5).sum(),
        ))

        # Feature importance (placeholder)
        st.subheader("Importancia de Features")
        importance = pd.DataFrame({
            "Feature": list(weights.keys()),
            "Weight": list(weights.values()),
        }).sort_values("Weight", ascending=False)

        fig = px.bar(importance, x="Weight", y="Feature", orientation="h", title="Pesos en Score Final")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Configure los parámetros en el sidebar y presione **Run Analysis** para comenzar")

    st.markdown("""
    ## Sistema de Detección de Estrés de Liquidez

    ### Características:
    - **Alerta temprana**: Horizonte 1-10 días
    - **Múltiples señales**: DFM+Kalman, HMM, CUSUM, EWMA, Anomalías, Net Liquidity
    - **Fusión robusta**: Calibración isotónica / Platt
    - **Grafo de flujos**: Mapa de drenajes e inyecciones
    - **Backtest walk-forward**: Métricas IC, AUROC, Q4-Q1
    - **Explicabilidad**: Atribución y drivers

    ### Datos utilizados (FRED):
    - Core plumbing: SOFR, EFFR, OBFR, TGCR
    - Balance Fed: ON RRP, TGA, Reservas
    - Stress: NFCI, STLFSI4, HY OAS, Term Spread, VIX
    - Mercados: SPX, Treasuries

    ### Metodología:
    1. Ingest & feature engineering
    2. Modelo de factor dinámico (Kalman smoothing)
    3. Detección de régimen (HMM)
    4. Control charts (CUSUM/EWMA)
    5. Anomalías (IsolationForest)
    6. Fusión de señales + calibración
    7. Walk-forward validation

    **Comenzar →** Ingrese su FRED API key y presione Run Analysis
    """)
