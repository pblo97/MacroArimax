"""
Macro Dashboard - Priority 1 Crisis Indicators

Implements the 4 priority crisis indicators from academic literature:
1. FX Cross-Currency Basis (Du et al. 2018) - GAP #1
2. Primary Dealer Leverage (Adrian-Shin 2010) - GAP #2
3. Commercial Paper Spreads (Kacperczyk-Schnabl 2010)
4. MOVE Index (Bond Market Volatility)

Plus macro context: yield curve, inflation, Fed balance sheet, activity
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_crisis_composite_gauge(crisis_score):
    """Create a gauge chart for the Crisis Composite Score (0-4)."""

    # Determine color and status
    if crisis_score >= 3:
        color = "red"
        status = "EXTREME STRESS"
    elif crisis_score >= 2:
        color = "orange"
        status = "ELEVATED STRESS"
    elif crisis_score >= 1:
        color = "yellow"
        status = "MODERATE STRESS"
    else:
        color = "green"
        status = "NORMAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=crisis_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Crisis Composite</b><br><sub>{status}</sub>"},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [1, 2], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [2, 3], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [3, 4], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 3
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'size': 14}
    )

    return fig


def create_priority1_panel(df):
    """Create Priority 1 Crisis Indicators panel.

    Displays the 4 critical early warning indicators from academic literature.
    """

    st.subheader("🚨 Priority 1: Crisis Detection Indicators")
    st.caption("Based on Du et al. (2018), Adrian-Shin (2010), Kacperczyk-Schnabl (2010)")

    # Calculate latest values
    latest = df.iloc[-1] if len(df) > 0 else None

    if latest is None:
        st.warning("No data available for crisis indicators")
        return

    # ========== DEBUG PANEL ==========
    with st.expander("🔍 DEBUG: Data Availability", expanded=False):
        st.write("**DataFrame shape:**", df.shape)
        st.write("**Last date:**", df.index[-1])

        # Show ALL columns available
        st.write("**All columns in DataFrame:**")
        all_cols = sorted(df.columns.tolist())
        st.write(f"Total columns: {len(all_cols)}")
        st.code(", ".join(all_cols))

        # Check base series
        st.write("**Base Series Available:**")
        base_series = {
            'EUR3MTD156N': 'optional - needed for fx_basis_proxy',
            'TB3MS': 'required',
            'CP_FINANCIAL_3M': 'required',
            'MOVE': 'optional - bond volatility',
            'VIX': 'required',
            'HY_OAS': 'required'
        }
        for series, note in base_series.items():
            if series in df.columns:
                last_val = df[series].iloc[-1]
                non_null_count = df[series].notna().sum()
                st.write(f"✅ {series}: {last_val:.2f} (non-null: {non_null_count}/{len(df)}) - {note}")
            else:
                st.write(f"❌ {series}: NOT IN DATAFRAME - {note}")

        # Check derived features
        st.write("**Derived Features:**")
        derived = ['fx_basis_proxy', 'cp_tbill_spread', 'crisis_composite']
        for feature in derived:
            if feature in df.columns:
                last_val = df[feature].iloc[-1]
                non_null_count = df[feature].notna().sum()
                st.write(f"✅ {feature}: {last_val:.2f} (non-null: {non_null_count}/{len(df)})")
            else:
                st.write(f"❌ {feature}: NOT IN DATAFRAME")

        # Show last 5 rows of key columns
        st.write("**Last 5 values of key series:**")
        cols_to_show = [c for c in ['EUR3MTD156N', 'TB3MS', 'fx_basis_proxy', 'CP_FINANCIAL_3M', 'cp_tbill_spread', 'MOVE', 'VIX'] if c in df.columns]
        if cols_to_show:
            st.dataframe(df[cols_to_show].tail(5))

        # Check what's needed for derived features
        st.write("**Derived Feature Dependencies:**")
        st.write("- fx_basis_proxy = EUR3MTD156N - TB3MS")
        if 'EUR3MTD156N' in df.columns and 'TB3MS' in df.columns:
            st.write("  ✅ Both components available")
            eur_val = df['EUR3MTD156N'].iloc[-1]
            tb3_val = df['TB3MS'].iloc[-1]
            manual_calc = eur_val - tb3_val
            st.write(f"  Manual calculation: {eur_val:.4f} - {tb3_val:.4f} = {manual_calc:.4f}")
        else:
            st.write("  ❌ Missing EUR3MTD156N or TB3MS")

        st.write("- cp_tbill_spread = CP_FINANCIAL_3M - TB3MS")
        if 'CP_FINANCIAL_3M' in df.columns and 'TB3MS' in df.columns:
            st.write("  ✅ Both components available")
            cp_val = df['CP_FINANCIAL_3M'].iloc[-1]
            tb3_val = df['TB3MS'].iloc[-1]
            manual_calc = cp_val - tb3_val
            st.write(f"  Manual calculation: {cp_val:.4f} - {tb3_val:.4f} = {manual_calc:.4f}")
        else:
            st.write("  ❌ Missing CP_FINANCIAL_3M or TB3MS")
    # ========== END DEBUG ==========

    # Create 4 columns for the 4 indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # FX Cross-Currency Basis Proxy
        fx_basis = latest.get('fx_basis_proxy', np.nan)

        if not np.isnan(fx_basis):
            delta_fx = df['fx_basis_proxy'].diff().iloc[-1] if 'fx_basis_proxy' in df.columns else 0
            st.metric(
                label="FX Basis Proxy (bp)",
                value=f"{fx_basis:.1f}",
                delta=f"{delta_fx:.1f} bp",
                delta_color="inverse",  # Negative is bad (USD shortage)
                help="EURIBOR-TB3M spread. Negative = USD shortage (Du et al. 2018). Critical below -50bp"
            )

            # Alert if critical
            if fx_basis < -50:
                st.error("⚠️ CRITICAL: USD shortage detected!")
            elif fx_basis < -20:
                st.warning("⚠️ Elevated USD funding stress")
        else:
            st.metric(
                label="FX Basis Proxy (bp)",
                value="N/A",
                help="Requires EUR3MTD156N data (not available)"
            )
            st.caption("⚠️ EUR3MTD156N required")

    with col2:
        # Commercial Paper Spread
        cp_spread = latest.get('cp_tbill_spread', np.nan)

        if not np.isnan(cp_spread):
            delta_cp = df['cp_tbill_spread'].diff().iloc[-1] if 'cp_tbill_spread' in df.columns else 0
            st.metric(
                label="CP-TBill Spread (bp)",
                value=f"{cp_spread:.1f}",
                delta=f"{delta_cp:.1f} bp",
                delta_color="inverse",
                help="Financial CP - Treasury Bill 3M. Crisis signal >100bp (Kacperczyk-Schnabl 2010)"
            )

            if cp_spread > 100:
                st.error("⚠️ CRITICAL: Funding crisis!")
            elif cp_spread > 50:
                st.warning("⚠️ Elevated funding stress")
        else:
            st.metric(
                label="CP-TBill Spread (bp)",
                value="N/A",
                help="Requires CP_FINANCIAL_3M and TB3MS"
            )
            st.caption("⚠️ Data missing")

    with col3:
        # MOVE Index (Bond Volatility)
        move_index = latest.get('MOVE', np.nan)

        if not np.isnan(move_index):
            delta_move = df['MOVE'].diff().iloc[-1]
            st.metric(
                label="MOVE Index",
                value=f"{move_index:.1f}",
                delta=f"{delta_move:.1f}",
                delta_color="inverse",
                help="Bond market volatility. Crisis signal >120. Normal range: 50-80"
            )

            if move_index > 150:
                st.error("⚠️ CRITICAL: Bond market panic!")
            elif move_index > 100:
                st.warning("⚠️ Elevated bond volatility")
        else:
            st.metric(
                label="MOVE Index",
                value="N/A",
                help="MOVE index not available (optional series)"
            )
            st.caption("⚠️ Optional series")

    with col4:
        # Primary Dealer Leverage (GAP #2 - using VIX as proxy)
        vix = latest.get('VIX', np.nan)

        if not np.isnan(vix):
            delta_vix = df['VIX'].diff().iloc[-1]
            st.metric(
                label="VIX (PD Leverage Proxy)",
                value=f"{vix:.1f}",
                delta=f"{delta_vix:.1f}",
                delta_color="inverse",
                help="VIX as proxy for dealer stress (GAP #2: True PD leverage not available in FRED). Crisis signal >40"
            )

            if vix > 40:
                st.error("⚠️ CRITICAL: Extreme volatility!")
            elif vix > 30:
                st.warning("⚠️ Elevated volatility")
        else:
            st.metric(
                label="VIX (PD Leverage Proxy)",
                value="N/A",
                help="VIX data not available"
            )
            st.caption("⚠️ Data missing")

    st.markdown("---")

    # Crisis Composite Score
    st.subheader("Crisis Composite Score (Adrian et al. 2019)")

    if 'crisis_composite' in df.columns:
        crisis_score = latest.get('crisis_composite', 0)

        # Show current breakdown
        vix_val = latest.get('VIX', 0)
        hy_oas_val = latest.get('HY_OAS', 0)
        cp_spread_val = latest.get('cp_tbill_spread', 0)
        move_val = latest.get('MOVE', np.nan)

        col_gauge, col_chart = st.columns([1, 2])

        with col_gauge:
            fig_gauge = create_crisis_composite_gauge(crisis_score)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Show current score breakdown
            st.markdown(f"**Current Score: {crisis_score:.0f} / 4**")
            st.caption(f"""
            **Active Indicators:**
            - VIX = {vix_val:.1f} {'✅ (< 30)' if vix_val < 30 else '⚠️ (> 30) +1'}
            - HY OAS = {hy_oas_val:.0f}bp {'✅ (< 600)' if hy_oas_val < 600 else '⚠️ (> 600) +1'}
            - CP Spread = {cp_spread_val:.1f}bp {'✅ (< 100)' if cp_spread_val < 100 else '⚠️ (> 100) +1'}
            - MOVE = {'N/A (not counted)' if np.isnan(move_val) else f'{move_val:.0f} {"✅ (< 100)" if move_val < 100 else "⚠️ (> 100) +1"}'}

            **Thresholds:**
            0-1 = Normal | 1-2 = Moderate | 2-3 = Elevated | 3-4 = Crisis
            """)

        with col_chart:
            # Time series of crisis composite
            fig_ts = go.Figure()

            # Get last 6 months of data
            df_recent = df.tail(126)  # ~6 months

            fig_ts.add_trace(go.Scatter(
                x=df_recent.index,
                y=df_recent['crisis_composite'],
                mode='lines+markers',
                name='Crisis Score',
                line=dict(color='darkred', width=2),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(220, 20, 60, 0.2)'
            ))

            # Add threshold lines
            fig_ts.add_hline(y=3, line_dash="dash", line_color="red",
                           annotation_text="Crisis Threshold", annotation_position="right")
            fig_ts.add_hline(y=2, line_dash="dash", line_color="orange",
                           annotation_text="Elevated Stress", annotation_position="right")
            fig_ts.add_hline(y=1, line_dash="dash", line_color="yellow",
                           annotation_text="Moderate Stress", annotation_position="right")

            fig_ts.update_layout(
                title="Crisis Composite - Last 6 Months",
                xaxis_title="Date",
                yaxis_title="Score (0-4)",
                height=300,
                hovermode='x unified',
                yaxis=dict(range=[-0.2, 4.2])
            )

            st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.warning("⚠️ Crisis composite indicator not available. Check that VIX and HY_OAS are present in data.")

    # Time series of Priority 1 indicators
    st.subheader("Historical Trends - Priority 1 Indicators")

    # Get last 252 days (~1 year)
    df_recent = df.tail(252)

    # Check which indicators are available
    available_indicators = []
    if 'fx_basis_proxy' in df.columns:
        available_indicators.append('FX Basis')
    if 'cp_tbill_spread' in df.columns:
        available_indicators.append('CP Spread')
    if 'MOVE' in df.columns:
        available_indicators.append('MOVE')
    if 'VIX' in df.columns:
        available_indicators.append('VIX')

    if len(available_indicators) >= 2:
        # Create 2x2 grid of charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("FX Basis Proxy (EURIBOR-TB3M)", "Commercial Paper Spread",
                           "MOVE Index (Bond Volatility)", "VIX (Equity Volatility)"),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # FX Basis
        if 'fx_basis_proxy' in df.columns:
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['fx_basis_proxy'],
                          mode='lines', name='FX Basis', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            fig.add_hline(y=-50, line_dash="dash", line_color="red", row=1, col=1)
            fig.update_yaxes(title_text="Basis Points", row=1, col=1)
        else:
            # Add placeholder text
            fig.add_annotation(
                text="Data not available<br>(EUR3MTD156N required)",
                xref="x", yref="y",
                x=0.5, y=0.5,
                xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=12, color="gray"),
                row=1, col=1
            )

        # CP Spread
        if 'cp_tbill_spread' in df.columns:
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['cp_tbill_spread'],
                          mode='lines', name='CP Spread', line=dict(color='purple', width=2)),
                row=1, col=2
            )
            fig.add_hline(y=100, line_dash="dash", line_color="red", row=1, col=2)
            fig.update_yaxes(title_text="Basis Points", row=1, col=2)

        # MOVE Index
        if 'MOVE' in df.columns:
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['MOVE'],
                          mode='lines', name='MOVE', line=dict(color='orange', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=100, line_dash="dash", line_color="red", row=2, col=1)
            fig.update_yaxes(title_text="Index Level", row=2, col=1)
        else:
            # Add placeholder text
            fig.add_annotation(
                text="Data not available<br>(MOVE index required)",
                xref="x2", yref="y2",
                x=0.5, y=0.5,
                xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=12, color="gray"),
                row=2, col=1
            )

        # VIX
        if 'VIX' in df.columns:
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['VIX'],
                          mode='lines', name='VIX', line=dict(color='darkred', width=2)),
                row=2, col=2
            )
            fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=2)
            fig.update_yaxes(title_text="Index Level", row=2, col=2)

        fig.update_layout(
            height=600,
            showlegend=False,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show note about missing indicators
        missing = []
        if 'fx_basis_proxy' not in df.columns:
            missing.append("FX Basis Proxy (EUR3MTD156N required)")
        if 'MOVE' not in df.columns:
            missing.append("MOVE Index")

        if missing:
            st.caption(f"⚠️ Missing optional indicators: {', '.join(missing)}")
    else:
        st.warning("⚠️ Insufficient data to display historical trends. Need at least 2 of: FX Basis, CP Spread, MOVE, VIX")


def create_macro_context_panel(df):
    """Create Macro Context panel: yield curve, inflation, Fed BS."""

    st.subheader("📊 Macro Context: Rates, Inflation & Policy")

    latest = df.iloc[-1] if len(df) > 0 else None

    if latest is None:
        st.warning("No data available for macro context")
        return

    # Top row: Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        curve = latest.get('T10Y2Y', np.nan)
        st.metric(
            label="2Y-10Y Spread (bp)",
            value=f"{curve:.1f}" if not np.isnan(curve) else "N/A",
            delta=f"{df['T10Y2Y'].diff().iloc[-1]:.1f} bp" if 'T10Y2Y' in df.columns else "N/A",
            help="Yield curve slope. Negative = inverted (recession signal)"
        )
        if curve < 0:
            st.warning("⚠️ Inverted yield curve!")

    with col2:
        core_pce = latest.get('core_pce_yoy', np.nan)
        st.metric(
            label="Core PCE YoY (%)",
            value=f"{core_pce:.2f}%" if not np.isnan(core_pce) else "N/A",
            delta=f"{df['core_pce_yoy'].diff().iloc[-1]:.2f}pp" if 'core_pce_yoy' in df.columns else "N/A",
            help="Fed's preferred inflation measure. Target: 2%"
        )
        if core_pce > 3:
            st.warning("⚠️ Above Fed target!")

    with col3:
        real_rate = latest.get('real_rate_5y', np.nan)
        st.metric(
            label="5Y Real Rate (%)",
            value=f"{real_rate:.2f}%" if not np.isnan(real_rate) else "N/A",
            delta=f"{df['real_rate_5y'].diff().iloc[-1]:.2f}pp" if 'real_rate_5y' in df.columns else "N/A",
            help="5Y nominal yield minus breakeven inflation"
        )

    with col4:
        fed_bs = latest.get('WALCL', np.nan) / 1000  # Convert to trillions
        st.metric(
            label="Fed Balance Sheet ($T)",
            value=f"${fed_bs:.2f}T" if not np.isnan(fed_bs) else "N/A",
            delta=f"{df['WALCL'].diff().iloc[-1] / 1000:.2f}T" if 'WALCL' in df.columns else "N/A",
            delta_color="off",
            help="Federal Reserve total assets"
        )

    st.markdown("---")

    # Charts
    tab1, tab2, tab3 = st.tabs(["📈 Yield Curve & Spreads", "💵 Inflation Dynamics", "🏦 Fed Balance Sheet"])

    with tab1:
        # Yield curve evolution
        df_recent = df.tail(252)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Yield Curve Spread (2Y-10Y)", "Real Rates"),
            vertical_spacing=0.15
        )

        if 'T10Y2Y' in df.columns:
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['T10Y2Y'],
                          mode='lines', name='2Y-10Y Spread',
                          line=dict(color='darkblue', width=2),
                          fill='tozeroy'),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="solid", line_color="red", row=1, col=1)
            fig.update_yaxes(title_text="Basis Points", row=1, col=1)

        if 'real_rate_5y' in df.columns and 'real_rate_10y' in df.columns:
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['real_rate_5y'],
                          mode='lines', name='5Y Real Rate',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['real_rate_10y'],
                          mode='lines', name='10Y Real Rate',
                          line=dict(color='darkgreen', width=2)),
                row=2, col=1
            )
            fig.update_yaxes(title_text="Percent", row=2, col=1)

        fig.update_layout(height=600, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Inflation dynamics
        df_recent = df.tail(252)

        fig = go.Figure()

        if 'cpi_yoy' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index, y=df_recent['cpi_yoy'],
                mode='lines', name='CPI YoY',
                line=dict(color='red', width=2)
            ))

        if 'core_cpi_yoy' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index, y=df_recent['core_cpi_yoy'],
                mode='lines', name='Core CPI YoY',
                line=dict(color='orange', width=2)
            ))

        if 'pce_yoy' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index, y=df_recent['pce_yoy'],
                mode='lines', name='PCE YoY',
                line=dict(color='blue', width=2)
            ))

        if 'core_pce_yoy' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index, y=df_recent['core_pce_yoy'],
                mode='lines', name='Core PCE YoY',
                line=dict(color='darkblue', width=2, dash='dash')
            ))

        # Add 2% Fed target line
        fig.add_hline(y=2, line_dash="dash", line_color="green",
                     annotation_text="Fed Target: 2%", annotation_position="right")

        fig.update_layout(
            title="Inflation Measures - Year-over-Year % Change",
            xaxis_title="Date",
            yaxis_title="Percent (%)",
            height=500,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("**Note**: Core PCE (dark blue dashed) is the Fed's preferred inflation measure")

    with tab3:
        # Fed Balance Sheet
        df_recent = df.tail(252)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Fed Total Assets (WALCL)", "QT Rate (Weekly Change)"),
            vertical_spacing=0.15
        )

        if 'WALCL' in df.columns:
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['WALCL'] / 1000,
                          mode='lines', name='Fed BS',
                          line=dict(color='purple', width=2),
                          fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.2)'),
                row=1, col=1
            )
            fig.update_yaxes(title_text="Trillions ($)", row=1, col=1)

            # QT rate (weekly change)
            qt_rate = df_recent['WALCL'].diff(periods=5) / 1000  # 5-day change
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=qt_rate,
                          mode='lines', name='QT Rate',
                          line=dict(color='darkred', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)
            fig.update_yaxes(title_text="Weekly Change ($T)", row=2, col=1)

        fig.update_layout(height=600, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        st.caption("**QT Rate**: Negative values indicate Quantitative Tightening (balance sheet reduction)")


def create_credit_stress_panel(df):
    """Create Credit & Market Stress panel."""

    st.subheader("💳 Credit & Market Stress")

    latest = df.iloc[-1] if len(df) > 0 else None

    if latest is None:
        st.warning("No data available for credit stress")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        hy_oas = latest.get('HY_OAS', np.nan)
        st.metric(
            label="HY OAS (bp)",
            value=f"{hy_oas:.0f}" if not np.isnan(hy_oas) else "N/A",
            delta=f"{df['HY_OAS'].diff().iloc[-1]:.0f} bp" if 'HY_OAS' in df.columns else "N/A",
            delta_color="inverse",
            help="High Yield Option-Adjusted Spread. Crisis signal >800bp"
        )
        if hy_oas > 800:
            st.error("⚠️ CRITICAL: Credit crisis!")
        elif hy_oas > 600:
            st.warning("⚠️ Elevated credit stress")

    with col2:
        ig_oas = latest.get('CORP_AAA_OAS', np.nan)
        st.metric(
            label="IG AAA OAS (bp)",
            value=f"{ig_oas:.0f}" if not np.isnan(ig_oas) else "N/A",
            delta=f"{df['CORP_AAA_OAS'].diff().iloc[-1]:.0f} bp" if 'CORP_AAA_OAS' in df.columns else "N/A",
            delta_color="inverse",
            help="Investment Grade AAA spread"
        )

    with col3:
        bbb_aaa = latest.get('bbb_aaa_spread', np.nan)
        st.metric(
            label="BBB-AAA Spread (bp)",
            value=f"{bbb_aaa:.0f}" if not np.isnan(bbb_aaa) else "N/A",
            delta=f"{df['bbb_aaa_spread'].diff().iloc[-1]:.0f} bp" if 'bbb_aaa_spread' in df.columns else "N/A",
            delta_color="inverse",
            help="Credit quality spread within IG"
        )

    with col4:
        vix = latest.get('VIX', np.nan)
        st.metric(
            label="VIX",
            value=f"{vix:.1f}" if not np.isnan(vix) else "N/A",
            delta=f"{df['VIX'].diff().iloc[-1]:.1f}" if 'VIX' in df.columns else "N/A",
            delta_color="inverse",
            help="CBOE Volatility Index"
        )

    st.markdown("---")

    # Credit spreads chart
    df_recent = df.tail(252)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("High Yield vs Investment Grade OAS", "Credit Cascade (Full Spectrum)"),
        vertical_spacing=0.15
    )

    if 'HY_OAS' in df.columns:
        fig.add_trace(
            go.Scatter(x=df_recent.index, y=df_recent['HY_OAS'],
                      mode='lines', name='HY OAS',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )

    if 'CORP_AAA_OAS' in df.columns:
        fig.add_trace(
            go.Scatter(x=df_recent.index, y=df_recent['CORP_AAA_OAS'],
                      mode='lines', name='AAA OAS',
                      line=dict(color='green', width=2)),
            row=1, col=1
        )

    if 'CORP_BBB_OAS' in df.columns:
        fig.add_trace(
            go.Scatter(x=df_recent.index, y=df_recent['CORP_BBB_OAS'],
                      mode='lines', name='BBB OAS',
                      line=dict(color='orange', width=2)),
            row=1, col=1
        )

    fig.update_yaxes(title_text="Basis Points", row=1, col=1)

    # Credit cascade
    if 'credit_cascade' in df.columns:
        fig.add_trace(
            go.Scatter(x=df_recent.index, y=df_recent['credit_cascade'],
                      mode='lines', name='CCC-AAA Spread',
                      line=dict(color='darkred', width=2),
                      fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.2)'),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Basis Points", row=2, col=1)

    fig.update_layout(height=600, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def render_macro_dashboard(df):
    """Main function to render the complete Macro Dashboard.

    Args:
        df: DataFrame with all macro indicators
    """

    st.header("🌍 Macro Dashboard - Crisis Detection & Context")
    st.caption("Priority 1 indicators based on Du et al. (2018), Adrian-Shin (2010), Kacperczyk-Schnabl (2010)")

    # Add note about GAP #2
    with st.expander("ℹ️ Data Availability Notes"):
        st.markdown("""
        **Priority 1 Indicators Status:**

        1. ✅ **FX Cross-Currency Basis**: Proxy available (EURIBOR-TB3M). True CIP basis requires FX forward data not in FRED.
        2. ⚠️ **Primary Dealer Leverage**: GAP #2 - True broker-dealer leverage not available in FRED. Using VIX as proxy.
        3. ✅ **Commercial Paper Spreads**: Full data available (Financial + Nonfinancial CP)
        4. ✅ **MOVE Index**: Bond market volatility available

        **Academic References:**
        - Du, Tepper & Verdelhan (2018): "Deviations from Covered Interest Parity" - FX basis as #1 crisis signal
        - Adrian & Shin (2010): "Liquidity and Leverage" - Dealer leverage amplification channel
        - Kacperczyk & Schnabl (2010): "When Safe Proved Risky" - CP crisis indicators
        - Adrian, Boyarchenko & Giannone (2019): "Vulnerable Growth" - Fed's Growth-at-Risk framework
        """)

    # Create tabs for different panels
    panel1, panel2, panel3 = st.tabs([
        "🚨 Crisis Detection (Priority 1)",
        "📊 Macro Context",
        "💳 Credit & Market Stress"
    ])

    with panel1:
        create_priority1_panel(df)

    with panel2:
        create_macro_context_panel(df)

    with panel3:
        create_credit_stress_panel(df)
