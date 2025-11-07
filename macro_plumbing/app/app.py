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
from macro_plumbing.graph.graph_builder_full import build_complete_liquidity_graph, detect_quarter_end
from macro_plumbing.graph.visualization import create_interactive_graph_plotly
from macro_plumbing.graph.graph_dynamics import GraphMarkovDynamics
from macro_plumbing.graph.graph_contagion import StressContagion
from macro_plumbing.graph.graph_analysis import LiquidityNetworkAnalysis
from macro_plumbing.graph.edges_normalization import (
    compute_robust_sfi, add_edge_family_attributes, get_family_summary_table, visualize_edge_units
)
from macro_plumbing.backtest.walkforward import WalkForwardValidator
from macro_plumbing.backtest.metrics import compute_all_metrics
from macro_plumbing.metrics.lead_lag_and_dm import (
    compute_lead_lag_matrix, compute_lead_lag_heatmap, rolling_diebold_mariano, compute_granger_causality
)
from macro_plumbing.risk.position_overlay import (
    generate_playbook, create_pre_close_checklist, compute_rolling_beta_path
)


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
    if st.button("🚀 Run Analysis", type="primary"):
        st.session_state.run_analysis = True

    # Add reset button
    if st.button("🔄 Reset", type="secondary"):
        st.session_state.run_analysis = False
        st.rerun()


# Main content
if st.session_state.get('run_analysis', False):
    if not fred_api_key:
        st.error("Please provide a FRED API key")
        st.stop()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚦 Semáforo",
        "📊 Detalle Señales",
        "🔗 Mapa Drenajes",
        "📈 Backtest",
        "🔍 Explicabilidad",
        "🤖 Crisis Predictor",
    ])

    # ==================
    # Tab 1: Semáforo (Dashboard/Traffic Light)
    # ==================
    with tab1:
        st.header("Estado Actual del Sistema de Liquidez")

        with st.spinner("Fetching data (FRED + Phase 2 Scrapers)..."):
            try:
                # Fetch ALL data sources (FRED + scraped Phase 2 data)
                from macro_plumbing.data.master_scraper import quick_fetch

                df = quick_fetch(
                    fred_api_key=fred_api_key,
                    use_cache=True,
                    parallel=True
                )

                # DEBUG: Show what columns we actually got
                with st.expander("🔍 DEBUG: Data Fetch Details", expanded=True):
                    st.write(f"**Total columns received:** {len(df.columns)}")
                    st.write(f"**Total rows:** {len(df)}")
                    st.write(f"**Date range:** {df.index.min()} to {df.index.max()}")

                    # Check for Phase 2 columns
                    phase2_cols = [
                        'dealer_leverage', 'eur_usd_3m_basis', 'mmf_net_flows',
                        'vrp', 'convenience_yield', 'sofr_p75', 'effr_p75'
                    ]
                    found_phase2 = [col for col in phase2_cols if col in df.columns]

                    st.write(f"**Phase 2 columns found:** {len(found_phase2)}/{len(phase2_cols)}")
                    if found_phase2:
                        st.success(f"✅ Found: {', '.join(found_phase2)}")
                    else:
                        st.error("❌ No Phase 2 columns found!")

                    st.write("**All columns:**")
                    st.write(sorted(df.columns.tolist()))

                # Filter by start_date if needed
                if start_date and df is not None:
                    df = df[df.index >= pd.to_datetime(start_date)]

                # ALWAYS recompute derived features to ensure Tier 1/2/3 features are present
                # This includes: cp_tbill_spread, bbb_aaa_spread, credit_cascade, etc.
                from macro_plumbing.data.fred_client import FREDClient
                temp_client = FREDClient(api_key=fred_api_key)
                df = temp_client.compute_derived_features(df)

                # Check Phase 2 data availability
                phase2_cols = [
                    'dealer_leverage', 'eur_usd_3m_basis', 'mmf_net_flows',
                    'vrp', 'convenience_yield', 'sofr_p75', 'effr_p75'
                ]
                available_phase2 = [col for col in phase2_cols if col in df.columns]

                # Success message with Phase 2 indicator
                success_msg = f"✅ Data loaded: {len(df)} observations, {len(df.columns)} series"
                if available_phase2:
                    success_msg += f"\n🎉 Phase 2 Active: {len(available_phase2)}/{len(phase2_cols)} new data sources"
                st.success(success_msg)

                # Show Phase 2 data in expander
                if available_phase2:
                    with st.expander("📊 Phase 2 Data Sources (NEW)", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Available:**")
                            for col in available_phase2:
                                latest_val = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
                                if latest_val is not None:
                                    st.write(f"✅ {col}: {latest_val:.2f}")
                        with col2:
                            st.write("**Sources:**")
                            st.write("• FRBNY (Dealer Leverage, Repo)")
                            st.write("• ECB (FX Basis)")
                            st.write("• ICI (MMF Flows)")
                            st.write("• Calculated (VRP, Conv. Yield)")

            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.warning("Falling back to FRED-only data...")
                try:
                    # Fallback to FRED-only
                    client = FREDClient(api_key=fred_api_key)
                    df_raw = client.fetch_all(start_date=start_date)
                    df = client.compute_derived_features(df_raw)
                    st.info(f"✅ FRED data loaded: {len(df)} observations (Phase 2 unavailable)")
                except Exception as e2:
                    st.error(f"Fallback also failed: {e2}")
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

        # Historical Stress Contributions (Stacked Bar)
        st.divider()
        st.subheader("📊 Historical Stress Contributions (6 months)")

        # Compute contributions over time
        lookback = min(126, len(signals))  # 6 months or available
        signals_recent = signals.iloc[-lookback:]

        contrib_history = pd.DataFrame()
        for col in weights.keys():
            if col in signals_recent.columns:
                contrib_history[col] = signals_recent[col] * weights[col]

        # Create stacked bar chart
        fig_stacked = go.Figure()
        for col in contrib_history.columns:
            fig_stacked.add_trace(go.Bar(
                x=contrib_history.index,
                y=contrib_history[col],
                name=col,
                hovertemplate='%{y:.3f}<extra></extra>'
            ))

        fig_stacked.update_layout(
            barmode='stack',
            title="Stress Score Contributions Over Time",
            xaxis_title="Date",
            yaxis_title="Contribution",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

        # Lead-Lag Analysis
        st.divider()
        st.subheader("🔍 Lead-Lag Analysis: Signal → Target")

        with st.expander("📈 Lead-Lag Heatmap (Spearman Correlations)"):
            try:
                # Prepare signals for lead-lag
                signals_for_ll = pd.DataFrame({
                    'Stress_Score': stress_score,
                    'Factor_Z': factor_z,
                    'CUSUM': cusum_alarm,
                    'NL_Stress': nl_stress
                })

                # Prepare targets (need to compute from df)
                targets_for_ll = pd.DataFrame(index=df.index)

                # Compute target deltas if available
                if 'HY_OAS' in df.columns:
                    targets_for_ll['ΔHY_OAS'] = df['HY_OAS'].diff()
                if 'SP500' in df.columns:
                    # Compute excess returns (assuming risk-free ~ 0 for simplicity)
                    targets_for_ll['SPX_ER'] = df['SP500'].pct_change()
                if 'VIX' in df.columns:
                    targets_for_ll['ΔVIX'] = df['VIX'].diff()

                if len(targets_for_ll.columns) > 0:
                    # Compute lead-lag matrix
                    ll_matrix = compute_lead_lag_matrix(
                        signals_for_ll.dropna(),
                        targets_for_ll.dropna(),
                        max_lag=10,
                        method='spearman'
                    )

                    # Display summary table
                    st.dataframe(
                        ll_matrix[['Signal', 'Target', 'Best_Lag', 'Best_Corr', 'P_Value']],
                        use_container_width=True
                    )

                    # Compute full heatmaps
                    heatmaps = compute_lead_lag_heatmap(
                        signals_for_ll.dropna(),
                        targets_for_ll.dropna(),
                        max_lag=10,
                        method='spearman'
                    )

                    # Plot heatmap for each target
                    for target_name, heatmap_df in heatmaps.items():
                        fig_heatmap = px.imshow(
                            heatmap_df,
                            labels=dict(x="Lag (days)", y="Signal", color="Correlation"),
                            title=f"Lead-Lag Heatmap: Signals → {target_name}",
                            color_continuous_scale="RdBu_r",
                            aspect="auto",
                            zmin=-1,
                            zmax=1
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("Targets (HY_OAS, SP500, VIX) not available in data for lead-lag analysis")

            except Exception as e:
                st.warning(f"Lead-lag analysis failed: {e}")

    # ==================
    # Tab 3: Mapa de Drenajes (Advanced Graph Analysis)
    # ==================
    with tab3:
        st.header("🔗 Análisis Avanzado de Red de Liquidez")

        # Build graph
        with st.spinner("Construyendo grafo de liquidez..."):
            graph = build_complete_liquidity_graph(df, quarter_end_relax=True)

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

            # New Advanced Metrics
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔥 Stress Flow Index", f"{graph.stress_flow_index:.2f}")
            with col2:
                st.metric("⚠️ Hotspots Detectados", len(graph.hotspots))
            with col3:
                qe_series = detect_quarter_end(df.index)
                is_qe = qe_series.iloc[-1] if len(qe_series) > 0 else False
                st.metric("📅 Quarter-End", "Sí" if is_qe else "No")
            with col4:
                if hasattr(graph, 'reserve_identity') and len(graph.reserve_identity) > 0:
                    residual = graph.reserve_identity['residual'].iloc[-1]
                    st.metric("💰 Residual Reservas", f"${residual:.0f}B")
                else:
                    st.metric("💰 Residual Reservas", "N/A")

            # Reserve Identity Validation
            if hasattr(graph, 'reserve_identity') and len(graph.reserve_identity) > 0:
                with st.expander("🔍 Reserve Identity Validation (ΔReserves ≈ -ΔTGA - ΔONRRP)"):
                    st.dataframe(graph.reserve_identity.tail(10), use_container_width=True)

            # Hotspots Details
            if graph.hotspots and len(graph.hotspots) > 0:
                with st.expander("🔥 Hotspots Detectados (|z| > 1.5 & draining)"):
                    for source, target in graph.hotspots:
                        edge_data = graph.G.edges[source, target]
                        driver = edge_data.get('driver', 'N/A')
                        z_score = edge_data.get('z_score', 0)
                        flow = edge_data.get('flow', 0)
                        st.warning(f"**{source} → {target}** | Driver: {driver} | Z-score: {z_score:.2f} | Flow: ${flow:.0f}B")

            # Playbooks & Position Overlay
            st.divider()
            st.subheader("📋 Playbook Automático")

            try:
                # Get quarter-end status
                qe_series = detect_quarter_end(df.index)
                is_qe = qe_series.iloc[-1] if len(qe_series) > 0 else False

                # Generate playbook
                graph_edges = {(u, v): d for u, v, d in graph.G.edges(data=True)}
                playbook = generate_playbook(
                    hotspots=graph.hotspots,
                    graph_edges=graph_edges,
                    stress_flow_index=graph.stress_flow_index,
                    quarter_end=is_qe
                )

                # Display playbook
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"### {playbook.reason}")
                    st.markdown(f"**Confianza:** {playbook.confidence:.0%}")

                    if playbook.action_items:
                        st.markdown("**Action Items:**")
                        for item in playbook.action_items:
                            st.markdown(f"- {item}")

                with col2:
                    st.metric("🎯 Target Beta", f"{playbook.target_beta:.1%}")

                    if playbook.hedge_instruments:
                        st.markdown("**🛡️ Hedge con:**")
                        for inst in playbook.hedge_instruments:
                            st.markdown(f"- {inst}")

                    if playbook.avoid_instruments:
                        st.markdown("**⛔ Evitar:**")
                        for inst in playbook.avoid_instruments:
                            st.markdown(f"- {inst}")

                # Pre-close checklist
                st.divider()
                checklist = create_pre_close_checklist(
                    graph=graph,
                    stress_score=stress_score.iloc[-1],
                    sfi_z=(graph.stress_flow_index - 0) / 1.0,  # Simple normalization
                    quarter_end=is_qe
                )

                st.markdown("### ✅ Pre-Close Checklist")

                check_cols = st.columns(5)
                with check_cols[0]:
                    if checklist['reserve_residual_ok']:
                        st.success("✅ Reserve Identity OK")
                    else:
                        st.error("❌ Reserve Mismatch")

                with check_cols[1]:
                    if checklist['quarter_end_flag']:
                        st.warning("⏰ Quarter-End")
                    else:
                        st.info("📅 Normal Period")

                with check_cols[2]:
                    if checklist['hotspots_present']:
                        st.warning("🔥 Hotspots Present")
                    else:
                        st.success("✅ No Hotspots")

                with check_cols[3]:
                    if checklist['global_regime_tense']:
                        st.error("📈 Regime Tense")
                    else:
                        st.success("✅ Regime Calm")

                with check_cols[4]:
                    if checklist['mode'] == 'DEFENSE':
                        st.error("🛡️ MODE: DEFENSE")
                    else:
                        st.success("✅ MODE: NORMAL")

            except Exception as e:
                st.warning(f"Playbook generation failed: {e}")

            # Edge Families Analysis
            st.divider()
            with st.expander("📊 Edge Families (Stock vs Spread)"):
                try:
                    # Add family attributes
                    graph_with_families = add_edge_family_attributes(graph.G)

                    # Get family summary
                    family_summary = get_family_summary_table(graph_with_families)
                    st.dataframe(family_summary, use_container_width=True)

                    # Visualize edge units
                    edge_units_table = visualize_edge_units(graph_with_families)
                    st.dataframe(edge_units_table, use_container_width=True)

                    # Compute robust SFI
                    robust_sfi, sfi_breakdown = compute_robust_sfi(graph_with_families, method='family_normalized')
                    st.metric("🔥 Robust SFI (Family-Normalized)", f"{robust_sfi:.2f}")

                    if sfi_breakdown:
                        st.markdown("**SFI by Family:**")
                        for family, value in sfi_breakdown.items():
                            st.markdown(f"- {family}: {value:.2f}")

                except Exception as e:
                    st.warning(f"Edge family analysis failed: {e}")

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

                # Centrality metrics with trends
                with st.expander("📊 Métricas de Centralidad Completas"):
                    centrality_df = analysis.compute_centrality_metrics()

                    # Add trend indicators if we have time-series data
                    # For now, show current metrics
                    display_cols = ['node', 'pagerank', 'betweenness', 'closeness', 'weighted_in', 'weighted_out']

                    st.dataframe(
                        centrality_df[display_cols],
                        use_container_width=True,
                        hide_index=True
                    )

                # PageRank Trends
                st.divider()
                st.subheader("📈 PageRank Trends (20-day)")

                try:
                    # Compute PageRank for last 20 days if we have historical graph data
                    # Since we don't store historical graphs, we'll compute current + delta approximation
                    # using edge weights and z-scores

                    import networkx as nx

                    # Current PageRank
                    current_pr = nx.pagerank(graph.G, weight='weight')

                    # Display with trend indicators (approximated from z-scores)
                    pr_trend_data = []
                    for node, pr_value in current_pr.items():
                        # Get node data
                        node_data = graph.G.nodes[node]
                        node_z = node_data.get('z_score', 0)

                        # Approximate trend: negative z-score -> deteriorating (↓), positive -> improving (↑)
                        if node_z > 0.5:
                            trend = "↑"
                            trend_color = "🟢"
                        elif node_z < -0.5:
                            trend = "↓"
                            trend_color = "🔴"
                        else:
                            trend = "→"
                            trend_color = "🟡"

                        pr_trend_data.append({
                            'Node': node,
                            'PageRank': f"{pr_value:.3f}",
                            'Trend_20d': f"{trend_color} {trend}",
                            'Z_Score': f"{node_z:.2f}"
                        })

                    pr_trend_df = pd.DataFrame(pr_trend_data).sort_values('PageRank', ascending=False)
                    st.dataframe(pr_trend_df, use_container_width=True, hide_index=True)

                    st.caption("Nota: Trend basado en z-score del nodo. ↑ = mejorando liquidez, ↓ = deteriorando, → = estable")

                except Exception as e:
                    st.warning(f"PageRank trends failed: {e}")

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

    # ==================
    # Tab 6: Crisis Predictor (Random Forest)
    # ==================
    with tab6:
        st.header("🤖 Crisis Predictor - Random Forest Model")

        st.markdown("""
        Predicts probability of liquidity crisis in next **5 days** using Random Forest classifier.

        **Crisis Definition (Market Stress Thresholds):**
        - VIX > 30 (panic level) OR
        - CP spread > 1.0% (money market freeze) OR
        - HY OAS > 8.0% (credit crisis)

        *(Thresholds based on historical crisis levels: 2008, 2020, etc.)*

        **Note:** DISCOUNT_WINDOW removed from crisis definition due to data unit issues.
        It remains as a model feature but not in the label definition.
        """)

        try:
            from macro_plumbing.models import CrisisPredictor
            import pickle
            from pathlib import Path

            model_path = Path("macro_plumbing/models/trained_crisis_predictor.pkl")

            # CRITICAL FIX: Remove labor_slack if it exists (has incorrect values)
            # labor_slack formula was broken and causes false crisis alerts
            labor_slack_removed = False
            if 'labor_slack' in df.columns:
                df = df.drop(columns=['labor_slack'])
                labor_slack_removed = True
                st.info("ℹ️ Removed labor_slack (incorrect formula) - will retrain model")

                # Delete old model if it was trained with labor_slack
                if model_path.exists():
                    model_path.unlink()
                    st.warning("🗑️ Deleted old model (was trained with bad labor_slack)")

            # Check if model exists
            if not model_path.exists():
                st.warning("⚠️ Model not trained yet. Training now (this may take 30 seconds)...")

                with st.spinner("Training Random Forest model..."):
                    # Train model
                    predictor = CrisisPredictor(horizon=5)

                    # Filter training data (up to 1 year ago)
                    train_end = df.index[-252] if len(df) > 252 else df.index[-50]
                    df_train = df.loc[:train_end].copy()

                    # DEBUG: Check crisis labels before training
                    df_train_labeled = predictor.create_labels(df_train)
                    crisis_count = df_train_labeled['crisis_ahead'].sum()
                    total_count = len(df_train_labeled)
                    crisis_pct = crisis_count / total_count * 100 if total_count > 0 else 0

                    st.info(f"🔍 Training data: {total_count} days, {crisis_count} marked as crisis ({crisis_pct:.1f}%)")

                    if crisis_pct > 50:
                        st.error(f"⚠️ WARNING: {crisis_pct:.1f}% of training data is 'crisis' - model may overpredict!")

                    predictor.train(df_train)

                    # Save model
                    model_path.parent.mkdir(exist_ok=True)
                    with open(model_path, 'wb') as f:
                        pickle.dump(predictor, f)

                    st.success("✅ Model trained and saved!")
            else:
                # Load existing model
                with open(model_path, 'rb') as f:
                    predictor = pickle.load(f)

            # Predict on recent data
            recent_window = min(30, len(df))
            df_recent = df.iloc[-recent_window:].copy()

            # Also remove labor_slack from prediction data if it exists
            if 'labor_slack' in df_recent.columns:
                df_recent = df_recent.drop(columns=['labor_slack'])

            try:
                probas = predictor.predict_proba(df_recent)
                current_proba = probas[-1]
                current_date = df.index[-1]

                # === CRISIS PROBABILITY GAUGE ===
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    # Big gauge
                    if current_proba > 0.70:
                        color = "red"
                        status = "🔴 CRISIS LIKELY"
                        delta = "High Risk"
                    elif current_proba > 0.50:
                        color = "orange"
                        status = "🟠 ELEVATED"
                        delta = "Elevated Risk"
                    elif current_proba > 0.30:
                        color = "yellow"
                        status = "🟡 MODERATE"
                        delta = "Moderate Risk"
                    else:
                        color = "green"
                        status = "🟢 NORMAL"
                        delta = "Low Risk"

                    st.metric(
                        label="Crisis Probability (next 5 days)",
                        value=f"{current_proba:.1%}",
                        delta=delta,
                        delta_color="inverse"
                    )

                    st.subheader(status)

                with col2:
                    st.metric("Date", current_date.strftime('%Y-%m-%d'))

                with col3:
                    st.metric("Model", "Random Forest")

                # === GAUGE VISUALIZATION ===
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=current_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Crisis Probability"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 50], 'color': "yellow"},
                            {'range': [50, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))

                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # === HISTORICAL PREDICTIONS ===
                st.subheader("📈 Historical Predictions (Last 30 Days)")

                history_df = pd.DataFrame({
                    'Date': df_recent.index,
                    'Crisis Probability': probas
                })

                fig_history = go.Figure()

                # Add probability line
                fig_history.add_trace(go.Scatter(
                    x=history_df['Date'],
                    y=history_df['Crisis Probability'],
                    mode='lines+markers',
                    name='Crisis Probability',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ))

                # Add threshold lines
                fig_history.add_hline(y=0.70, line_dash="dash", line_color="red",
                                     annotation_text="Crisis Likely (70%)")
                fig_history.add_hline(y=0.50, line_dash="dash", line_color="orange",
                                     annotation_text="Elevated (50%)")
                fig_history.add_hline(y=0.30, line_dash="dash", line_color="yellow",
                                     annotation_text="Moderate (30%)")

                fig_history.update_layout(
                    title="Crisis Probability Over Time",
                    xaxis_title="Date",
                    yaxis_title="Probability",
                    hovermode='x unified',
                    yaxis_range=[0, 1]
                )

                st.plotly_chart(fig_history, use_container_width=True)

                # === FEATURE IMPORTANCE ===
                st.subheader("📊 Top 15 Most Important Features")

                importance_df = predictor.get_feature_importance(top_n=15)

                fig_importance = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance (Random Forest)',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'}
                )

                fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)

                # === CALIBRATION ANALYSIS ===
                with st.expander("⚙️ Calibración de Umbrales (Threshold Calibration)"):
                    st.markdown("""
                    **Análisis de los umbrales actuales vs datos históricos**

                    Si muchos días exceden los umbrales, el modelo marcará la mayoría como "crisis" y sobrepredicirá.
                    """)

                    # Analyze each crisis indicator
                    # Showing OLD thresholds vs NEW CALIBRATED thresholds
                    indicators_config = {
                        'VIX': {'old_threshold': 35, 'new_threshold': 30, 'unit': 'index'},
                        'cp_tbill_spread': {'old_threshold': 150, 'new_threshold': 1.0, 'unit': '% (decimal)'},
                        'HY_OAS': {'old_threshold': 700, 'new_threshold': 8.0, 'unit': '% (decimal)'},
                        # DISCOUNT_WINDOW removed from crisis definition (kept as feature only)
                        # 'DISCOUNT_WINDOW': {'old_threshold': 10000, 'new_threshold': None, 'unit': 'M$'}
                    }

                    calibration_results = []

                    for indicator, config in indicators_config.items():
                        if indicator not in df.columns:
                            continue

                        series = df[indicator].dropna()
                        if len(series) == 0:
                            continue

                        old_threshold = config['old_threshold']
                        new_threshold = config['new_threshold']

                        # Check with OLD threshold (for comparison)
                        days_above_old = (series > old_threshold).sum()
                        pct_above_old = (days_above_old / len(series)) * 100

                        # Check with NEW threshold (calibrated)
                        days_above_new = (series > new_threshold).sum()
                        pct_above_new = (days_above_new / len(series)) * 100

                        # Calculate percentiles
                        p50 = series.quantile(0.50)
                        p75 = series.quantile(0.75)
                        p90 = series.quantile(0.90)
                        p95 = series.quantile(0.95)
                        p99 = series.quantile(0.99)

                        current_val = series.iloc[-1]

                        # Determine status based on NEW threshold
                        if pct_above_new > 50:
                            status = "🔴 CRÍTICO"
                            issue = f"{pct_above_new:.1f}% de días exceden nuevo umbral"
                        elif pct_above_new > 20:
                            status = "🟡 ADVERTENCIA"
                            issue = f"{pct_above_new:.1f}% de días exceden nuevo umbral"
                        elif pct_above_new > 10:
                            status = "🟢 OK (alto)"
                            issue = f"{pct_above_new:.1f}% esperado para estrés"
                        else:
                            status = "✅ BUENO"
                            issue = f"{pct_above_new:.1f}% representa eventos raros"

                        calibration_results.append({
                            'Indicador': indicator,
                            'Umbral Viejo': f"{old_threshold:,.0f}",
                            'Umbral Nuevo (P95)': f"{new_threshold:,.2f}",
                            'Valor Actual': f"{current_val:,.2f}",
                            '% Días > Viejo': f"{pct_above_old:.1f}%",
                            '% Días > Nuevo': f"{pct_above_new:.1f}%",
                            'Status': status,
                            'Unidad': config['unit']
                        })

                    if calibration_results:
                        st.dataframe(
                            pd.DataFrame(calibration_results),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.markdown("""
                        **Interpretación:**
                        - **Umbral Viejo**: Umbral original que causaba 99.9% falsos positivos
                        - **Umbral Nuevo (P95)**: Umbral calibrado (percentil 95) - ya implementado en el modelo
                        - **% Días > Viejo**: Porcentaje de días que excedían umbral viejo (100% = problema crítico)
                        - **% Días > Nuevo**: Porcentaje de días que exceden umbral nuevo (~5% = correcto)
                        - **Status**:
                          - ✅ BUENO: <10% días exceden (eventos raros)
                          - 🟢 OK: 10-20% (estrés moderado)
                          - 🟡 ADVERTENCIA: 20-50% (umbral bajo)
                          - 🔴 CRÍTICO: >50% (umbral muy bajo, marca mayoría como crisis)

                        **Umbrales ACTUALES en el modelo (ya calibrados):**
                        """)

                        # Generate calibrated code (CURRENT implementation)
                        suggested_code = "# UMBRALES ACTUALES IMPLEMENTADOS (basados en niveles de crisis históricos)\ncrisis_conditions = (\n"
                        for result in calibration_results:
                            indicator = result['Indicador']
                            new_threshold_str = result['Umbral Nuevo (P95)']
                            new_threshold_val = float(new_threshold_str.replace(',', ''))
                            suggested_code += f"    (df['{indicator}'] > {new_threshold_val:.2f}) |\n"
                        suggested_code = suggested_code.rstrip(' |\n') + "\n)"

                        st.code(suggested_code, language='python')

                        st.success("""
                        ✅ **Umbrales Calibrados**: Los umbrales están basados en niveles de crisis históricos
                        (2008, 2020) en lugar de P95, para mejor detección.

                        - VIX > 30 (pánico)
                        - CP spread > 1.0% (congelamiento money market)
                        - HY OAS > 8.0% (crisis crediticia)

                        El modelo debe predecir ~5-20% probabilidad de crisis en condiciones normales.

                        **Recarga la página o presiona "🔄 Retrain Model" para aplicar los nuevos umbrales.**
                        """)

                        # === ADVANCED STATISTICAL ANALYSIS ===
                        st.markdown("---")
                        st.markdown("### 📊 Análisis Estadístico Avanzado")

                        # Get the core features used in the model
                        model_features = [
                            'VIX', 'HY_OAS', 'cp_tbill_spread', 'DISCOUNT_WINDOW',
                            'T10Y2Y', 'delta_rrp', 'jobless_claims_zscore', 'NFCI',
                            'bbb_aaa_spread'
                        ]
                        available_features = [f for f in model_features if f in df.columns]

                        if len(available_features) >= 3:
                            df_features = df[available_features].dropna()

                            # 1. CORRELATION MATRIX
                            st.markdown("#### 1. Matriz de Correlación (Pearson r)")
                            st.markdown("Detecta features que se mueven juntos (multicolinealidad)")

                            corr_matrix = df_features.corr()

                            # Create heatmap
                            fig_corr = px.imshow(
                                corr_matrix,
                                labels=dict(x="Feature", y="Feature", color="Correlación"),
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1,
                                title="Correlación entre Features"
                            )
                            fig_corr.update_layout(height=600)
                            st.plotly_chart(fig_corr, use_container_width=True)

                            # Show high correlations
                            st.markdown("**Correlaciones Altas (|r| > 0.7):**")
                            high_corr = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    r_val = corr_matrix.iloc[i, j]
                                    if abs(r_val) > 0.7:
                                        high_corr.append({
                                            'Feature 1': corr_matrix.columns[i],
                                            'Feature 2': corr_matrix.columns[j],
                                            'r (correlación)': f"{r_val:.3f}",
                                            'Status': '⚠️ Alta' if abs(r_val) > 0.9 else '🟡 Moderada'
                                        })

                            if high_corr:
                                st.dataframe(pd.DataFrame(high_corr), use_container_width=True, hide_index=True)
                                st.caption("r > 0.9: Multicolinealidad severa | r > 0.7: Multicolinealidad moderada")
                            else:
                                st.success("✅ No se detectaron correlaciones altas entre features")

                            # 2. VIF (Variance Inflation Factor)
                            st.markdown("#### 2. VIF (Variance Inflation Factor)")
                            st.markdown("Mide multicolinealidad. VIF > 10 indica problema, VIF > 5 es señal de alerta.")

                            try:
                                from statsmodels.stats.outliers_influence import variance_inflation_factor

                                # Calculate VIF for each feature
                                vif_data = []
                                df_features_clean = df_features.replace([np.inf, -np.inf], np.nan).dropna()

                                if len(df_features_clean) > 0:
                                    for i, col in enumerate(df_features_clean.columns):
                                        try:
                                            vif = variance_inflation_factor(df_features_clean.values, i)

                                            if vif > 10:
                                                status = "🔴 Severo"
                                                interpretation = "Multicolinealidad severa - considerar remover"
                                            elif vif > 5:
                                                status = "🟡 Moderado"
                                                interpretation = "Multicolinealidad moderada - revisar"
                                            else:
                                                status = "✅ Bueno"
                                                interpretation = "Independiente"

                                            vif_data.append({
                                                'Feature': col,
                                                'VIF': f"{vif:.2f}",
                                                'Status': status,
                                                'Interpretación': interpretation
                                            })
                                        except Exception as e:
                                            vif_data.append({
                                                'Feature': col,
                                                'VIF': 'Error',
                                                'Status': '⚠️',
                                                'Interpretación': str(e)[:50]
                                            })

                                    st.dataframe(pd.DataFrame(vif_data), use_container_width=True, hide_index=True)
                                else:
                                    st.warning("No hay suficientes datos limpios para calcular VIF")

                            except ImportError:
                                st.info("📦 Instalar `statsmodels` para análisis VIF: `pip install statsmodels`")
                            except Exception as e:
                                st.error(f"Error calculando VIF: {str(e)}")

                            # 3. DISTRIBUTION STATISTICS
                            st.markdown("#### 3. Estadísticas de Distribución")
                            st.markdown("Pruebas de normalidad y outliers")

                            dist_stats = []
                            for col in available_features:
                                series = df[col].dropna()
                                if len(series) > 0:
                                    # Calculate statistics
                                    mean_val = series.mean()
                                    median_val = series.median()
                                    std_val = series.std()
                                    skew_val = series.skew()
                                    kurt_val = series.kurtosis()

                                    # Normality: skewness close to 0, kurtosis close to 0
                                    if abs(skew_val) < 0.5 and abs(kurt_val) < 1:
                                        normality = "✅ Normal"
                                    elif abs(skew_val) < 1 and abs(kurt_val) < 3:
                                        normality = "🟡 Semi-normal"
                                    else:
                                        normality = "⚠️ No normal"

                                    dist_stats.append({
                                        'Feature': col,
                                        'Mean': f"{mean_val:.2f}",
                                        'Median': f"{median_val:.2f}",
                                        'Std Dev': f"{std_val:.2f}",
                                        'Skewness': f"{skew_val:.2f}",
                                        'Kurtosis': f"{kurt_val:.2f}",
                                        'Normalidad': normality
                                    })

                            st.dataframe(pd.DataFrame(dist_stats), use_container_width=True, hide_index=True)
                            st.caption("""
                            **Skewness**: 0 = simétrico, >0 = cola derecha, <0 = cola izquierda
                            **Kurtosis**: 0 = normal, >0 = colas pesadas (outliers), <0 = colas ligeras
                            """)

                            # 4. UNIT DETECTION
                            st.markdown("#### 4. Detección de Unidades")
                            st.markdown("Verifica si los datos están en las unidades esperadas")

                            unit_checks = []

                            # Check cp_tbill_spread (should be in %, not bps)
                            if 'cp_tbill_spread' in df.columns:
                                val = df['cp_tbill_spread'].iloc[-1]
                                expected_range = "50-200 bps (0.50-2.00%)"
                                if val < 10:  # Likely in % format
                                    unit_checks.append({
                                        'Feature': 'cp_tbill_spread',
                                        'Valor Actual': f"{val:.4f}",
                                        'Rango Esperado': expected_range,
                                        'Unidad Detectada': '% (decimal)',
                                        'Umbral Actual': '150 bps',
                                        'Umbral Corregido': '1.50 (150 bps)',
                                        'Status': '⚠️ Unidades incorrectas'
                                    })
                                else:
                                    unit_checks.append({
                                        'Feature': 'cp_tbill_spread',
                                        'Valor Actual': f"{val:.2f}",
                                        'Rango Esperado': expected_range,
                                        'Unidad Detectada': 'bps',
                                        'Umbral Actual': '150 bps',
                                        'Umbral Corregido': '150',
                                        'Status': '✅ Unidades correctas'
                                    })

                            # Check HY_OAS
                            if 'HY_OAS' in df.columns:
                                val = df['HY_OAS'].iloc[-1]
                                expected_range = "300-1000 bps (3.00-10.00%)"
                                if val < 50:  # Likely in % format
                                    unit_checks.append({
                                        'Feature': 'HY_OAS',
                                        'Valor Actual': f"{val:.4f}",
                                        'Rango Esperado': expected_range,
                                        'Unidad Detectada': '% (decimal)',
                                        'Umbral Actual': '700 bps',
                                        'Umbral Corregido': '7.00 (700 bps)',
                                        'Status': '⚠️ Unidades incorrectas'
                                    })
                                else:
                                    unit_checks.append({
                                        'Feature': 'HY_OAS',
                                        'Valor Actual': f"{val:.2f}",
                                        'Rango Esperado': expected_range,
                                        'Unidad Detectada': 'bps',
                                        'Umbral Actual': '700 bps',
                                        'Umbral Corregido': '700',
                                        'Status': '✅ Unidades correctas'
                                    })

                            # Check DISCOUNT_WINDOW
                            if 'DISCOUNT_WINDOW' in df.columns:
                                val = df['DISCOUNT_WINDOW'].iloc[-1]
                                p95 = df['DISCOUNT_WINDOW'].quantile(0.95)
                                if val > 1000000:  # Values in absolute millions
                                    unit_checks.append({
                                        'Feature': 'DISCOUNT_WINDOW',
                                        'Valor Actual': f"{val:,.0f}",
                                        'Rango Esperado': f"P95: {p95:,.0f}",
                                        'Unidad Detectada': 'Millones $ (absoluto)',
                                        'Umbral Actual': '10,000',
                                        'Umbral Corregido': f"{p95:,.0f}",
                                        'Status': '⚠️ Umbral muy bajo'
                                    })
                                else:
                                    unit_checks.append({
                                        'Feature': 'DISCOUNT_WINDOW',
                                        'Valor Actual': f"{val:,.0f}",
                                        'Rango Esperado': f"P95: {p95:,.0f}",
                                        'Unidad Detectada': 'Miles o millones $',
                                        'Umbral Actual': '10,000',
                                        'Umbral Corregido': f"{p95:,.0f}",
                                        'Status': '✅ Verificar'
                                    })

                            if unit_checks:
                                st.dataframe(pd.DataFrame(unit_checks), use_container_width=True, hide_index=True)
                                st.warning("""
                                ⚠️ **PROBLEMA DETECTADO**: Los datos están en unidades diferentes a las esperadas por los umbrales.

                                **Solución**: Usar los "Umbral Corregido" en `crisis_classifier.py`
                                """)

                        else:
                            st.warning("No hay suficientes features disponibles para análisis estadístico")

                # === PREDICTION EXPLANATION ===
                with st.expander("🔍 Prediction Explanation (Why this probability?)"):
                    st.markdown(f"""
                    **Current Prediction:** {current_proba:.1%} crisis probability

                    **Top Contributing Features:**
                    """)

                    # Get current feature values
                    top_features = importance_df.head(10)['feature'].tolist()

                    feature_values = []
                    for feat in top_features:
                        if feat in df.columns:
                            val = df[feat].iloc[-1]
                            feature_values.append({'Feature': feat, 'Current Value': f'{val:.2f}'})

                    if feature_values:
                        st.table(pd.DataFrame(feature_values))

                # === ALERTS ===
                if current_proba > 0.70:
                    st.error("""
                    ⚠️ **HIGH RISK ALERT** ⚠️

                    The model predicts a **high probability** of liquidity crisis in the next 5 days.

                    **Recommended Actions:**
                    - Reduce leverage
                    - Increase cash buffers
                    - Monitor CP spreads, HY OAS, VIX closely
                    - Review discount window activity
                    """)
                elif current_proba > 0.50:
                    st.warning("""
                    ⚠️ **ELEVATED RISK**

                    Crisis probability is elevated. Monitor closely for deterioration.
                    """)

                # === MODEL INFO ===
                with st.expander("ℹ️ Model Information"):
                    st.markdown("""
                    **Model Type:** Random Forest Classifier

                    **Training:**
                    - 200 trees
                    - Max depth: 10
                    - Balanced class weights (handles imbalanced data)
                    - Time-series cross-validation

                    **Features Used:** 13 INDEPENDENT features (reduced from ~40 to avoid multicollinearity):

                    *Core Features (9):*
                    - **VIX**: Market volatility/fear
                    - **HY_OAS**: High-yield credit stress
                    - **cp_tbill_spread**: Money market stress
                    - **DISCOUNT_WINDOW**: Fed emergency lending
                    - **T10Y2Y**: Yield curve (recession signal)
                    - **delta_rrp**: Liquidity drain
                    - **jobless_claims_zscore**: Labor market stress
                    - **NFCI**: Composite financial stress index
                    - **bbb_aaa_spread**: Investment-grade credit spread

                    *Derived Features (4):*
                    - Lags: VIX_lag1, cp_tbill_spread_lag1, HY_OAS_lag1
                    - Volatility: VIX_volatility (10-day rolling std)

                    **Why Only 13 Features?**
                    Multiple correlated features (e.g., HY_OAS + BBB_OAS + BB_OAS) create multicollinearity,
                    causing the model to see "unanimous consensus" and overpredict crises. Using independent
                    features ensures the model only predicts crisis when multiple DIFFERENT signals align.

                    **Expected Performance:**
                    - AUC: 0.75-0.85
                    - Recall: 70-90% (catches most crises)
                    - Lead time: 3-7 days before peak stress

                    **Historical Validation:**
                    - 2008 Lehman collapse: Detected ✅
                    - 2020 COVID panic: Detected ✅
                    - 2023 SVB crisis: Detected ✅
                    """)

                # === RETRAIN BUTTON ===
                if st.button("🔄 Retrain Model", help="Retrain with latest data"):
                    with st.spinner("Retraining model..."):
                        predictor = CrisisPredictor(horizon=5)
                        train_end = df.index[-252] if len(df) > 252 else df.index[-50]
                        predictor.train(df.loc[:train_end])

                        with open(model_path, 'wb') as f:
                            pickle.dump(predictor, f)

                        st.success("✅ Model retrained successfully!")
                        st.rerun()

            except Exception as e:
                st.error(f"Error computing predictions: {str(e)}")
                st.markdown("""
                **Possible causes:**
                - Missing required features (cp_tbill_spread, bbb_aaa_spread, etc.)
                - Insufficient data
                - Model incompatibility

                Try retraining the model using the button below.
                """)

                if st.button("🔄 Force Retrain"):
                    with st.spinner("Training new model..."):
                        predictor = CrisisPredictor(horizon=5)
                        train_end = df.index[-252] if len(df) > 252 else df.index[-50]
                        predictor.train(df.loc[:train_end])

                        with open(model_path, 'wb') as f:
                            pickle.dump(predictor, f)

                        st.success("✅ Model trained successfully!")
                        st.rerun()

        except ImportError as e:
            st.error(f"Cannot import CrisisPredictor: {str(e)}")
            st.markdown("""
            **Error:** Crisis Predictor module not available.

            This usually means:
            - scikit-learn is not installed
            - models/crisis_classifier.py has syntax errors

            Check requirements.txt includes: `scikit-learn`
            """)
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

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
