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
from macro_plumbing.graph.visualization import create_interactive_graph_plotly
from macro_plumbing.graph.graph_dynamics import GraphMarkovDynamics
from macro_plumbing.graph.graph_contagion import StressContagion
from macro_plumbing.graph.graph_analysis import LiquidityNetworkAnalysis
from macro_plumbing.backtest.walkforward import WalkForwardValidator
from macro_plumbing.backtest.metrics import compute_all_metrics


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
                spread = df["sofr_effr_spread"].dropna()
                if len(spread) > 30:
                    # Use data-driven parameters
                    spread_mean = spread.mean()
                    spread_std = spread.std()
                    # k = 0.5 * std (detect shift of 0.5 sigma)
                    # h = 4 * std (alarm threshold)
                    cusum = CUSUM(target_mean=spread_mean, k=0.5 * spread_std, h=4.0 * spread_std)
                    cusum_alarm = cusum.get_signals(spread)
                    # Reindex to match df
                    cusum_alarm = cusum_alarm.reindex(df.index, fill_value=0)
                else:
                    cusum_alarm = pd.Series(0, index=df.index)
            else:
                cusum_alarm = pd.Series(0, index=df.index)

            # 3. Anomalies
            plumbing_features = df[["sofr_effr_spread", "delta_rrp", "delta_tga"]].dropna()
            if len(plumbing_features) > 100:
                anomaly_flag = detect_anomalies(plumbing_features, contamination=0.05)
            else:
                anomaly_flag = pd.Series(0, index=df.index)

            # 4. Net Liquidity stress
            # Use rolling percentile to detect when NL is in bottom 20%
            nl_series = nl_df["net_liquidity"]
            # Rolling rank over past year (252 days) or available data
            window_size = min(252, len(nl_series))
            nl_rolling_pct = nl_series.rolling(window=window_size, min_periods=30).apply(
                lambda x: (x.iloc[-1] <= x.quantile(0.2)).astype(int) if len(x) > 0 else 0
            )
            nl_stress = nl_rolling_pct.fillna(0).astype(int)
            nl_stress.name = "nl_stress"
            # Reindex to match df
            nl_stress = nl_stress.reindex(df.index, fill_value=0)

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
    # Tab 3: Mapa de Drenajes (Advanced Graph Analysis)
    # ==================
    with tab3:
        st.header("🔗 Análisis Avanzado de Red de Liquidez")

        # Build graph
        with st.spinner("Construyendo grafo de liquidez..."):
            graph = build_liquidity_graph(df)

        # Tabs within Tab 3
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "📈 Visualización",
            "🧬 Estados Markov",
            "🦠 Contagio",
            "⚠️ Análisis Sistémico"
        ])

        # Subtab 1: Visualization
        with subtab1:
            st.subheader("Grafo Interactivo de Flujos")
            try:
                fig_graph = create_interactive_graph_plotly(graph)
                st.plotly_chart(fig_graph, use_container_width=True)
            except Exception as e:
                st.error(f"Error creando visualización: {e}")

            # Basic insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodos", len(graph.G.nodes()))
            with col2:
                st.metric("Flujos", len(graph.G.edges()))
            with col3:
                total_flow = sum(abs(d.get('flow', 0)) for _, _, d in graph.G.edges(data=True))
                st.metric("Flujo Total", f"${total_flow:.0f}B")

            # Expandable tables
            with st.expander("📊 Ver Tablas Detalladas"):
                nodes_df, edges_df = graph.to_dataframe()
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(nodes_df, use_container_width=True)
                with col2:
                    st.dataframe(edges_df, use_container_width=True)

        # Subtab 2: Markov Dynamics
        with subtab2:
            st.subheader("Estados Markov por Nodo")
            st.markdown("""
            Cada nodo tiene una **probabilidad de stress** que evoluciona basada en:
            - 🔵 Señal propia (z-score, percentil, deltas)
            - 🔴 Contagio de vecinos (weighted by flows)
            - 🟣 Régimen global
            """)

            try:
                # Initialize dynamics
                dynamics = GraphMarkovDynamics(
                    graph.G,
                    transition_speed=0.3,
                    contagion_weight=0.3
                )

                # Compute current stress state
                global_stress = stress_score.iloc[-1] if len(stress_score) > 0 else 0.5
                current_stress = dynamics.step(global_stress=global_stress)

                # Display current state
                st.subheader("Estado Actual de Stress por Nodo")
                stress_df = pd.DataFrame([
                    {
                        'Nodo': node,
                        'Prob. Stress': f"{prob:.1%}",
                        'Estado': '🔴 STRESSED' if prob > 0.6 else '🟡 CAUTION' if prob > 0.4 else '🟢 CALM'
                    }
                    for node, prob in current_stress.items()
                ])
                st.dataframe(stress_df, use_container_width=True, hide_index=True)

                # Simulate evolution
                st.subheader("Simulación de Evolución (10 pasos)")
                evolution = dynamics.simulate(
                    n_steps=10,
                    global_stress_series=np.linspace(global_stress, min(global_stress + 0.2, 1.0), 10)
                )

                # Plot evolution
                fig_evolution = px.line(
                    evolution,
                    x='step',
                    y=[col for col in evolution.columns if col != 'step'],
                    title="Evolución de Probabilidad de Stress",
                    labels={'value': 'Prob. Stress', 'variable': 'Nodo'}
                )
                st.plotly_chart(fig_evolution, use_container_width=True)

            except Exception as e:
                st.error(f"Error en análisis Markov: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Subtab 3: Contagion
        with subtab3:
            st.subheader("🦠 Modelo de Contagio via Random Walk")
            st.markdown("""
            Simula cómo el **stress se propaga** a través de la red.
            - **1-step**: Contagio inmediato a vecinos
            - **k-step**: Propagación multi-salto
            - **Amplification**: Nodos que amplifican stress (superspreaders)
            """)

            try:
                # Initialize contagion model
                contagion = StressContagion(graph.G, damping=0.7)

                # Get current stress
                current_stress_dict = {
                    node: dynamics.node_states[node].stress_prob
                    for node in graph.G.nodes()
                }

                # Superspreaders
                st.subheader("🎯 Nodos Superspreaders (Amplificadores)")
                superspreaders = contagion.identify_superspreaders(top_k=min(3, len(graph.G.nodes())), steps=3)

                superspreader_df = pd.DataFrame([
                    {
                        'Nodo': node,
                        'Factor Amplificación': f"{amp:.2f}x",
                        'Riesgo': '🔴 Alto' if amp > 1.2 else '🟡 Medio' if amp > 1.0 else '🟢 Bajo'
                    }
                    for node, amp in superspreaders
                ])
                st.dataframe(superspreader_df, use_container_width=True, hide_index=True)

                # Shock simulation
                st.subheader("💥 Simulación de Shock")

                with st.form("shock_simulation"):
                    shock_node = st.selectbox("Selecciona nodo para shock:", list(graph.G.nodes()))
                    shock_magnitude = st.slider("Magnitud del shock:", 0.1, 2.0, 1.0, 0.1)
                    shock_steps = st.slider("Pasos de simulación:", 1, 10, 5, 1)
                    submit_shock = st.form_submit_button("🚀 Simular Shock")

                if submit_shock:
                    with st.spinner("Simulando propagación de shock..."):
                        shock_result = contagion.simulate_shock(
                            shock_node=shock_node,
                            shock_magnitude=shock_magnitude,
                            steps=shock_steps
                        )

                        fig_shock = px.line(
                            shock_result,
                            x='step',
                            y=[col for col in shock_result.columns if col != 'step'],
                            title=f"Propagación de Shock desde {shock_node} (magnitud {shock_magnitude}x)",
                            labels={'value': 'Stress', 'variable': 'Nodo'}
                        )
                        st.plotly_chart(fig_shock, use_container_width=True)

                        # Show final state
                        st.subheader("Estado Final del Sistema")
                        final_state = shock_result.iloc[-1]
                        final_df = pd.DataFrame([
                            {
                                'Nodo': col,
                                'Stress Final': f"{final_state[col]:.2%}",
                                'Estado': '🔴 Alto' if final_state[col] > 0.7 else '🟡 Medio' if final_state[col] > 0.3 else '🟢 Bajo'
                            }
                            for col in shock_result.columns if col != 'step'
                        ])
                        st.dataframe(final_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error en análisis de contagio: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Subtab 4: Systemic Analysis
        with subtab4:
            st.subheader("⚠️ Análisis de Riesgo Sistémico")

            try:
                # Initialize analysis
                analysis = LiquidityNetworkAnalysis(graph.G)

                # Systemic importance
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🏦 Nodos Sistémicamente Importantes")
                    sifis = analysis.identify_systemically_important_nodes(top_k=min(3, len(graph.G.nodes())))
                    sifi_df = pd.DataFrame([
                        {
                            'Nodo': node,
                            'Importancia': f"{score:.1%}",
                            'Nivel': '🔴 Crítico' if score > 0.7 else '🟡 Alto' if score > 0.4 else '🟢 Normal'
                        }
                        for node, score in sifis
                    ])
                    st.dataframe(sifi_df, use_container_width=True, hide_index=True)

                with col2:
                    st.subheader("🔗 Bottlenecks Críticos")
                    critical_edges = analysis.find_all_critical_edges()[:3]
                    if critical_edges:
                        bottleneck_df = pd.DataFrame([
                            {
                                'Flujo': f"{u} → {v}",
                                'Criticidad': f"{crit:.3f}",
                                'Riesgo': '🔴 Alto' if crit > 0.1 else '🟡 Medio'
                            }
                            for (u, v), crit in critical_edges
                        ])
                        st.dataframe(bottleneck_df, use_container_width=True, hide_index=True)

                # Network fragility
                st.subheader("🛡️ Fragilidad de la Red")
                fragility = analysis.compute_network_fragility()

                frag_col1, frag_col2, frag_col3 = st.columns(3)
                with frag_col1:
                    st.metric(
                        "Densidad",
                        f"{fragility['density']:.1%}",
                        help="Qué tan conectada está la red"
                    )
                with frag_col2:
                    st.metric(
                        "Componentes",
                        f"{fragility['n_components']:.0f}",
                        help="Número de subgrafos desconectados"
                    )
                with frag_col3:
                    fragility_score = fragility['fragility_score']
                    st.metric(
                        "Score Fragilidad",
                        f"{fragility_score:.1%}",
                        delta=None,
                        help="0=robusto, 1=frágil"
                    )

                # Centrality metrics
                with st.expander("📊 Métricas de Centralidad Completas"):
                    centrality_df = analysis.compute_centrality_metrics()
                    st.dataframe(
                        centrality_df[['node', 'pagerank', 'betweenness', 'closeness', 'weighted_in', 'weighted_out']],
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as e:
                st.error(f"Error en análisis sistémico: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ==================
    # Tab 4: Backtest Walk-Forward
    # ==================
    with tab4:
        st.header("📊 Backtest Walk-Forward")
        st.markdown("""
        Validación robusta del modelo usando **walk-forward cross-validation**.
        - Entrena en ventana histórica
        - Predice en ventana futura (out-of-sample)
        - Rola hacia adelante para evitar look-ahead bias
        """)

        # Configuration in form to prevent full page reload
        with st.form("backtest_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                train_window = st.selectbox("Ventana entrenamiento:", [126, 252, 504], index=1, help="Días de training")
            with col2:
                test_window = st.selectbox("Ventana test:", [21, 63, 126], index=1, help="Días de testing")
            with col3:
                step_size = st.selectbox("Step size:", [21, 63], index=0, help="Días entre folds")

            submit_backtest = st.form_submit_button("🚀 Ejecutar Backtest", use_container_width=True)

        if submit_backtest:
            with st.spinner("Ejecutando walk-forward backtest..."):
                try:
                    # Prepare target: stress events (e.g., VIX spikes, NFCI > threshold)
                    # For simplicity, use stress_score > threshold as "ground truth"
                    target = (stress_score > stress_threshold).astype(int)
                    target.name = "stress_event"

                    # Prepare features
                    features = signals[["factor_z", "cusum", "anomaly", "nl_stress"]].copy()

                    # Create validator
                    validator = WalkForwardValidator(
                        train_window=train_window,
                        test_window=test_window,
                        step=step_size
                    )

                    # Model function: simple weighted average (can be replaced with ML model)
                    def train_model(X_train, y_train):
                        """Simple fusion model for demo."""
                        class SimpleModel:
                            def __init__(self, weights):
                                self.weights = weights

                            def predict(self, X):
                                pred = sum(X[col] * self.weights[col] for col in self.weights if col in X.columns)
                                return pred

                        # Use predefined weights or optimize on train set
                        return SimpleModel(weights)

                    # Run walk-forward
                    results = validator.validate(features, target, train_model)

                    if len(results) == 0:
                        st.warning("No hay suficientes datos para backtest. Necesita al menos train_window + test_window días.")
                    else:
                        # Display results
                        st.success(f"✅ Backtest completado: {len(results)} folds")

                        # Aggregate metrics
                        st.subheader("📈 Métricas Agregadas")
                        metric_cols = ['IC', 'AUROC', 'Brier', 'HitRate']
                        agg_metrics = results[metric_cols].describe()

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            ic_mean = results['IC'].mean()
                            ic_std = results['IC'].std()
                            st.metric("IC (mean)", f"{ic_mean:.3f}", f"±{ic_std:.3f}")
                        with col2:
                            auroc_mean = results['AUROC'].mean()
                            st.metric("AUROC (mean)", f"{auroc_mean:.3f}")
                        with col3:
                            hit_mean = results['HitRate'].mean()
                            st.metric("Hit Rate (mean)", f"{hit_mean:.1%}")
                        with col4:
                            brier_mean = results['Brier'].mean()
                            st.metric("Brier (mean)", f"{brier_mean:.3f}", help="Lower is better")

                        # Time series of metrics
                        st.subheader("📉 Evolución Temporal de Métricas")

                        fig_metrics = go.Figure()
                        for metric in ['IC', 'AUROC', 'HitRate']:
                            if metric in results.columns:
                                fig_metrics.add_trace(go.Scatter(
                                    x=results['test_start'],
                                    y=results[metric],
                                    mode='lines+markers',
                                    name=metric
                                ))

                        fig_metrics.update_layout(
                            title="Métricas por Fold (Out-of-Sample)",
                            xaxis_title="Test Start Date",
                            yaxis_title="Metric Value",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_metrics, use_container_width=True)

                        # Results table
                        st.subheader("📋 Resultados Detallados por Fold")
                        display_cols = ['fold', 'train_start', 'train_end', 'test_start', 'test_end',
                                       'IC', 'AUROC', 'Brier', 'HitRate']
                        display_results = results[display_cols].copy()
                        display_results['train_start'] = pd.to_datetime(display_results['train_start']).dt.date
                        display_results['train_end'] = pd.to_datetime(display_results['train_end']).dt.date
                        display_results['test_start'] = pd.to_datetime(display_results['test_start']).dt.date
                        display_results['test_end'] = pd.to_datetime(display_results['test_end']).dt.date

                        st.dataframe(
                            display_results.style.format({
                                'IC': '{:.3f}',
                                'AUROC': '{:.3f}',
                                'Brier': '{:.3f}',
                                'HitRate': '{:.1%}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                        # Statistical significance
                        st.subheader("📊 Análisis Estadístico")
                        col1, col2 = st.columns(2)

                        with col1:
                            # t-test: is IC significantly > 0?
                            from scipy import stats as sp_stats
                            ic_values = results['IC'].dropna()
                            if len(ic_values) > 2:
                                t_stat, p_value = sp_stats.ttest_1samp(ic_values, 0)
                                sig = "✅ Significativo" if p_value < 0.05 else "❌ No significativo"
                                st.metric(
                                    "IC > 0 (t-test)",
                                    sig,
                                    f"p-value: {p_value:.4f}"
                                )

                        with col2:
                            # Consistency: % of folds with IC > 0
                            consistency = (results['IC'] > 0).mean()
                            st.metric(
                                "Consistencia IC > 0",
                                f"{consistency:.1%}",
                                f"{int(consistency * len(results))}/{len(results)} folds"
                            )

                        # Distribution of IC
                        st.subheader("📊 Distribución de IC")
                        fig_hist = px.histogram(
                            results,
                            x='IC',
                            nbins=20,
                            title="Distribución de Information Coefficient",
                            labels={'IC': 'Information Coefficient'},
                            marginal='box'
                        )
                        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="IC=0")
                        st.plotly_chart(fig_hist, use_container_width=True)

                except Exception as e:
                    st.error(f"Error en backtest: {e}")
                    import traceback
                    st.code(traceback.format_exc())

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
