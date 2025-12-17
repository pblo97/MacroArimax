"""
QRA Analysis - Quarterly Refunding Announcement Intelligence

Analyzes US Treasury debt issuance patterns and composition to understand:
1. Treasury Bills vs Long-term Bonds issuance trends
2. Weighted Average Maturity (WAM) of debt
3. QRA signals for liquidity and market impact

Key Indicators:
- Bills Share: % of debt in T-Bills (< 1 year maturity)
- WAM: Weighted Average Maturity of outstanding debt
- Issuance Velocity: Rate of change in debt composition
- Liquidity Signal: Impact on money market conditions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta


def calculate_qra_metrics(df):
    """
    Calculate QRA intelligence metrics from Treasury data.

    Returns dict with:
    - bills_share: % of debt in T-Bills
    - wam_proxy: Weighted Average Maturity proxy
    - issuance_velocity: Rate of change in composition
    - liquidity_signal: Impact signal (-1 to 1)
    """
    metrics = {}

    # Calculate Bills Share if we have total debt and bills data
    if 'GFDEBTN' in df.columns and 'FDHBFIN' in df.columns:
        # GFDEBTN = Total Public Debt Outstanding (billions)
        # We'll use short-term debt proxies
        total_debt = df['GFDEBTN']

        # Bills share proxy: Use TB3MS vs longer rates as signal
        # Higher short-term issuance = higher bills share
        if 'TB3MS' in df.columns and 'DGS10' in df.columns:
            # When TB3MS rises faster than DGS10, suggests more bill issuance
            short_long_ratio = df['TB3MS'] / (df['DGS10'] + 0.01)  # Avoid div by zero
            metrics['bills_share_proxy'] = short_long_ratio * 100

    # WAM Proxy: Use yield curve slope as inverse proxy
    # Steeper curve = longer WAM (Treasury issuing more long-term)
    # Flatter/inverted = shorter WAM (Treasury issuing more bills)
    if 'T10Y2Y' in df.columns:
        curve_slope = df['T10Y2Y']
        # Normalize to 0-100 scale, where higher = longer maturity
        wam_proxy = 50 + (curve_slope * 2)  # Rough normalization
        metrics['wam_proxy'] = wam_proxy.clip(0, 100)

    # Issuance Velocity: Rate of change in debt composition
    if 'bills_share_proxy' in metrics:
        velocity = metrics['bills_share_proxy'].diff(periods=20)  # 20-day change
        metrics['issuance_velocity'] = velocity

    # Liquidity Signal:
    # Positive = Bills issuance (adds liquidity)
    # Negative = Bond issuance (drains liquidity from bills)
    if 'issuance_velocity' in metrics:
        velocity = metrics['issuance_velocity']
        # Normalize to -1 to 1
        signal = velocity / (velocity.abs().rolling(63).mean() + 0.01)
        metrics['liquidity_signal'] = signal.clip(-1, 1)

    return metrics


def create_qra_gauge(value, title, threshold_low, threshold_high):
    """Create a gauge chart for QRA metrics."""

    # Determine color and status
    if value >= threshold_high:
        color = "red"
        status = "HIGH"
    elif value <= threshold_low:
        color = "green"
        status = "LOW"
    else:
        color = "yellow"
        status = "MODERATE"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{title}</b><br><sub>{status}</sub>"},
        number={'font': {'size': 32}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold_low], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [threshold_low, threshold_high], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [threshold_high, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'size': 12}
    )

    return fig


def render_qra_analysis(df):
    """
    Main function to render QRA Analysis tab.

    Args:
        df: DataFrame with macro indicators including Treasury data
    """

    st.header("📊 QRA Analysis - Treasury Debt Issuance Intelligence")
    st.caption("Analyzing US Treasury Quarterly Refunding Announcements and debt composition trends")

    # Info expander
    with st.expander("ℹ️ What is QRA Analysis?"):
        st.markdown("""
        **Quarterly Refunding Announcement (QRA)** is the US Treasury's quarterly announcement of its
        borrowing plans and debt composition strategy.

        **Why it matters:**
        - **Bills vs Bonds**: Short-term bills add liquidity, long-term bonds drain it from money markets
        - **WAM (Weighted Average Maturity)**: Indicates Treasury's financing strategy and refinancing risk
        - **Market Impact**: Large bill issuance can pressure money market rates and RRP usage
        - **Liquidity Signal**: Changes in composition affect financial conditions

        **Key Metrics:**
        1. **Bills Share**: Higher = more short-term financing (typically expansionary)
        2. **WAM**: Higher = longer maturity profile (reduces refinancing risk)
        3. **Issuance Velocity**: Rate of change in debt composition
        4. **Liquidity Signal**: Net impact on money market liquidity

        **Academic References:**
        - Greenwood, Hanson & Stein (2015): "A Gap-Filling Theory of Corporate Debt Maturity Choice"
        - Krishnamurthy & Vissing-Jorgensen (2012): "The Aggregate Demand for Treasury Debt"
        - Sunderam (2015): "Money Creation and the Shadow Banking System"
        """)

    if df is None or len(df) == 0:
        st.warning("No data available for QRA analysis")
        return

    # Calculate QRA metrics
    with st.spinner("Calculating QRA metrics..."):
        qra_metrics = calculate_qra_metrics(df)

    if not qra_metrics:
        st.error("Unable to calculate QRA metrics. Required data series may be missing.")
        st.info("Required series: GFDEBTN, TB3MS, DGS10, T10Y2Y")
        return

    # Latest values
    latest_idx = df.index[-1]

    st.subheader("🎯 Current QRA Signals")

    # Top row: Key gauges
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'bills_share_proxy' in qra_metrics:
            bills_share = qra_metrics['bills_share_proxy'].iloc[-1]
            fig_bills = create_qra_gauge(bills_share, "Bills Issuance Intensity", 30, 70)
            st.plotly_chart(fig_bills, use_container_width=True)
            st.caption("Higher = More T-Bill issuance (adds liquidity)")
        else:
            st.warning("Bills share data not available")

    with col2:
        if 'wam_proxy' in qra_metrics:
            wam = qra_metrics['wam_proxy'].iloc[-1]
            fig_wam = create_qra_gauge(wam, "WAM Proxy", 40, 60)
            st.plotly_chart(fig_wam, use_container_width=True)
            st.caption("Higher = Longer maturity profile")
        else:
            st.warning("WAM proxy not available")

    with col3:
        if 'liquidity_signal' in qra_metrics:
            liq_signal = qra_metrics['liquidity_signal'].iloc[-1]
            # Convert -1 to 1 scale to 0 to 100 for gauge
            liq_gauge_value = (liq_signal + 1) * 50
            fig_liq = create_qra_gauge(liq_gauge_value, "Liquidity Impact", 30, 70)
            st.plotly_chart(fig_liq, use_container_width=True)

            if liq_signal > 0.3:
                st.success("✅ Bills issuance adding liquidity")
            elif liq_signal < -0.3:
                st.error("⚠️ Bond issuance draining bill market")
            else:
                st.info("📊 Neutral issuance mix")
        else:
            st.warning("Liquidity signal not available")

    st.markdown("---")

    # Detailed metrics
    st.subheader("📈 Detailed Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'GFDEBTN' in df.columns:
            total_debt = df['GFDEBTN'].iloc[-1]
            debt_change = df['GFDEBTN'].diff().iloc[-1]
            st.metric(
                "Total Public Debt",
                f"${total_debt:,.0f}B",
                f"{debt_change:+,.0f}B",
                help="Total US Public Debt Outstanding"
            )
        else:
            st.metric("Total Public Debt", "N/A")

    with col2:
        if 'TB3MS' in df.columns:
            tb3m = df['TB3MS'].iloc[-1]
            tb3m_change = df['TB3MS'].diff().iloc[-1]
            st.metric(
                "3M T-Bill Rate",
                f"{tb3m:.2f}%",
                f"{tb3m_change:+.2f}pp",
                delta_color="off",
                help="3-Month Treasury Bill Rate"
            )
        else:
            st.metric("3M T-Bill Rate", "N/A")

    with col3:
        if 'T10Y2Y' in df.columns:
            curve = df['T10Y2Y'].iloc[-1]
            curve_change = df['T10Y2Y'].diff().iloc[-1]
            st.metric(
                "2Y-10Y Curve",
                f"{curve:.0f}bp",
                f"{curve_change:+.0f}bp",
                delta_color="off",
                help="Yield curve slope (proxy for maturity preference)"
            )

            if curve > 50:
                st.caption("✅ Normal curve - mixed issuance")
            elif curve < 0:
                st.caption("⚠️ Inverted - likely more bills")
        else:
            st.metric("2Y-10Y Curve", "N/A")

    with col4:
        if 'issuance_velocity' in qra_metrics:
            velocity = qra_metrics['issuance_velocity'].iloc[-1]
            st.metric(
                "Issuance Velocity",
                f"{velocity:.1f}",
                help="20-day change in bills intensity"
            )

            if abs(velocity) > 5:
                st.caption("⚠️ Rapid composition shift")
            else:
                st.caption("📊 Stable issuance mix")
        else:
            st.metric("Issuance Velocity", "N/A")

    st.markdown("---")

    # Charts
    st.subheader("📊 Historical Trends")

    tab1, tab2, tab3 = st.tabs([
        "📈 Bills vs Bonds Intensity",
        "⏱️ Maturity Profile (WAM)",
        "💧 Liquidity Impact"
    ])

    with tab1:
        # Bills intensity over time
        if 'bills_share_proxy' in qra_metrics:
            df_recent = df.tail(252)  # Last year
            bills_series = qra_metrics['bills_share_proxy'].tail(252)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_recent.index,
                y=bills_series,
                mode='lines',
                name='Bills Intensity',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 255, 0.2)'
            ))

            # Add threshold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                         annotation_text="High Bills Issuance", annotation_position="right")
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         annotation_text="Low Bills Issuance", annotation_position="right")

            fig.update_layout(
                title="T-Bills Issuance Intensity - Last 12 Months",
                xaxis_title="Date",
                yaxis_title="Intensity Index",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Interpretation
            current_bills = bills_series.iloc[-1]
            avg_bills = bills_series.mean()

            st.markdown("**Interpretation:**")
            if current_bills > avg_bills + 10:
                st.info(f"📊 Current bills intensity ({current_bills:.1f}) is **above average** ({avg_bills:.1f}). "
                       "Treasury is issuing more short-term debt, which typically adds liquidity to money markets.")
            elif current_bills < avg_bills - 10:
                st.info(f"📊 Current bills intensity ({current_bills:.1f}) is **below average** ({avg_bills:.1f}). "
                       "Treasury is issuing more long-term debt, which may drain liquidity from money markets.")
            else:
                st.info(f"📊 Current bills intensity ({current_bills:.1f}) is **near average** ({avg_bills:.1f}). "
                       "Balanced issuance mix.")
        else:
            st.warning("Bills intensity data not available")

    with tab2:
        # WAM proxy over time
        if 'wam_proxy' in qra_metrics:
            df_recent = df.tail(252)
            wam_series = qra_metrics['wam_proxy'].tail(252)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_recent.index,
                y=wam_series,
                mode='lines',
                name='WAM Proxy',
                line=dict(color='green', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 200, 0, 0.2)'
            ))

            # Add reference line
            fig.add_hline(y=50, line_dash="solid", line_color="gray",
                         annotation_text="Neutral", annotation_position="right")

            fig.update_layout(
                title="Weighted Average Maturity Proxy - Last 12 Months",
                xaxis_title="Date",
                yaxis_title="WAM Proxy (0-100)",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show correlation with curve
            if 'T10Y2Y' in df.columns:
                st.caption("**Note**: WAM proxy based on yield curve slope. Steeper curve = longer maturity issuance.")

                current_wam = wam_series.iloc[-1]
                if current_wam > 60:
                    st.success("✅ Treasury extending maturity profile (issuing more long-term debt)")
                elif current_wam < 40:
                    st.warning("⚠️ Treasury shortening maturity profile (issuing more bills)")
                else:
                    st.info("📊 Balanced maturity profile")
        else:
            st.warning("WAM proxy data not available")

    with tab3:
        # Liquidity impact over time
        if 'liquidity_signal' in qra_metrics:
            df_recent = df.tail(252)
            liq_series = qra_metrics['liquidity_signal'].tail(252)

            fig = go.Figure()

            # Color based on positive/negative
            colors = ['green' if x > 0 else 'red' for x in liq_series]

            fig.add_trace(go.Bar(
                x=df_recent.index,
                y=liq_series,
                name='Liquidity Signal',
                marker_color=colors,
                opacity=0.7
            ))

            fig.add_hline(y=0, line_dash="solid", line_color="black")
            fig.add_hline(y=0.3, line_dash="dash", line_color="green",
                         annotation_text="Adding Liquidity", annotation_position="right")
            fig.add_hline(y=-0.3, line_dash="dash", line_color="red",
                         annotation_text="Draining Liquidity", annotation_position="right")

            fig.update_layout(
                title="QRA Liquidity Impact Signal - Last 12 Months",
                xaxis_title="Date",
                yaxis_title="Signal (-1 to +1)",
                height=400,
                hovermode='x unified',
                yaxis=dict(range=[-1.2, 1.2])
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            positive_pct = (liq_series > 0.3).sum() / len(liq_series) * 100
            negative_pct = (liq_series < -0.3).sum() / len(liq_series) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Liquidity Adding Days", f"{positive_pct:.1f}%",
                         help="Days with strong bills issuance signal")
            with col2:
                st.metric("Liquidity Draining Days", f"{negative_pct:.1f}%",
                         help="Days with strong bonds issuance signal")
            with col3:
                neutral_pct = 100 - positive_pct - negative_pct
                st.metric("Neutral Days", f"{neutral_pct:.1f}%",
                         help="Days with balanced issuance")
        else:
            st.warning("Liquidity signal data not available")

    st.markdown("---")

    # Treasury debt composition table
    st.subheader("📋 Treasury Debt Composition Summary")

    if 'GFDEBTN' in df.columns and 'TB3MS' in df.columns and 'DGS10' in df.columns:
        latest = df.iloc[-1]

        summary_data = {
            'Metric': [
                'Total Public Debt',
                '3M T-Bill Rate',
                '10Y Treasury Rate',
                '2Y-10Y Spread',
                'Bills Intensity',
                'WAM Proxy',
                'Liquidity Signal'
            ],
            'Current Value': [
                f"${latest.get('GFDEBTN', 0):,.0f}B",
                f"{latest.get('TB3MS', 0):.2f}%",
                f"{latest.get('DGS10', 0):.2f}%",
                f"{latest.get('T10Y2Y', 0):.0f}bp",
                f"{qra_metrics.get('bills_share_proxy', pd.Series([0])).iloc[-1]:.1f}" if 'bills_share_proxy' in qra_metrics else "N/A",
                f"{qra_metrics.get('wam_proxy', pd.Series([0])).iloc[-1]:.1f}" if 'wam_proxy' in qra_metrics else "N/A",
                f"{qra_metrics.get('liquidity_signal', pd.Series([0])).iloc[-1]:.2f}" if 'liquidity_signal' in qra_metrics else "N/A"
            ],
            'Signal': [
                "→",
                "↑" if df['TB3MS'].diff().iloc[-1] > 0 else "↓",
                "↑" if df['DGS10'].diff().iloc[-1] > 0 else "↓",
                "↑" if df['T10Y2Y'].diff().iloc[-1] > 0 else "↓",
                "↑" if 'bills_share_proxy' in qra_metrics and qra_metrics['bills_share_proxy'].diff().iloc[-1] > 0 else "↓",
                "↑" if 'wam_proxy' in qra_metrics and qra_metrics['wam_proxy'].diff().iloc[-1] > 0 else "↓",
                "↑" if 'liquidity_signal' in qra_metrics and qra_metrics['liquidity_signal'].diff().iloc[-1] > 0 else "↓"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Insufficient data for composition summary")

    # Market implications
    st.markdown("---")
    st.subheader("💡 Market Implications")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**If Treasury issues more Bills (short-term):**")
        st.markdown("""
        - ✅ Adds liquidity to money markets
        - ✅ Can support RRP drainage
        - ⚠️ May pressure T-bill rates lower
        - ⚠️ Increases refinancing frequency
        - 📊 Typically accommodative for risk assets
        """)

    with col2:
        st.markdown("**If Treasury issues more Bonds (long-term):**")
        st.markdown("""
        - ⚠️ Drains liquidity from money markets
        - ⚠️ Can pressure T-bill rates higher
        - ✅ Extends maturity profile (reduces risk)
        - ✅ Reduces refinancing frequency
        - 📊 Can be restrictive for risk assets
        """)

    # Data source note
    st.caption("""
    **Data Sources**: FRED (Federal Reserve Economic Data)
    - GFDEBTN: Total Public Debt Outstanding
    - TB3MS: 3-Month Treasury Bill Rate
    - DGS10: 10-Year Treasury Constant Maturity Rate
    - T10Y2Y: 10-Year minus 2-Year Treasury Spread

    **Note**: This analysis uses proxy metrics based on market rates and yield curve dynamics.
    For official QRA announcements, visit: https://home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding
    """)
