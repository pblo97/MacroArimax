"""
QRA Analysis - Quarterly Refunding Announcement Intelligence

Analyzes US Treasury debt issuance patterns and composition to understand:
1. Treasury Bills vs Long-term Bonds issuance trends
2. Weighted Average Maturity (WAM) of debt
3. QRA signals for liquidity and market impact

Key Indicators:
- Bills Intensity: Proxy based on short/long rate ratios
- WAM: Weighted Average Maturity proxy from curve slope
- Issuance Velocity: Rate of change in debt outstanding
- Liquidity Signal: Net impact on money market conditions
- TGA Analysis: Treasury cash management patterns
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta


def create_qra_gauge(value, title, threshold_low, threshold_high, value_range=(0, 100)):
    """Create a gauge chart for QRA metrics."""

    # Handle NaN values
    if pd.isna(value):
        value = 50  # Neutral position for display
        status = "NO DATA"
        color = "gray"
    else:
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
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [value_range[0], value_range[1]], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [value_range[0], threshold_low], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [threshold_low, threshold_high], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [threshold_high, value_range[1]], 'color': 'rgba(255, 0, 0, 0.3)'}
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
        - **TGA Management**: Treasury cash buffer impacts reserve balances

        **Key Metrics:**
        1. **Bills Intensity**: Higher = more short-term financing (typically expansionary)
        2. **WAM Proxy**: Higher = longer maturity profile (reduces refinancing risk)
        3. **Debt Velocity**: Rate of increase in total debt outstanding
        4. **TGA Ratio**: Treasury cash buffer as % of total debt
        5. **Fed Holdings**: % of debt held by Federal Reserve (QE impact)

        **Academic References:**
        - Greenwood, Hanson & Stein (2015): "A Gap-Filling Theory of Corporate Debt Maturity Choice"
        - Krishnamurthy & Vissing-Jorgensen (2012): "The Aggregate Demand for Treasury Debt"
        - Sunderam (2015): "Money Creation and the Shadow Banking System"
        """)

    if df is None or len(df) == 0:
        st.warning("No data available for QRA analysis")
        return

    # Check what data is available
    has_debt_data = 'GFDEBTN' in df.columns
    has_rate_data = 'TB3MS' in df.columns and 'DGS10' in df.columns
    has_curve_data = 'T10Y2Y' in df.columns
    has_tga_data = 'TGA' in df.columns

    # Check for derived features
    has_bills_intensity = 'bills_intensity_proxy' in df.columns
    has_wam_proxy = 'wam_curve_proxy' in df.columns
    has_debt_velocity = 'debt_issuance_velocity' in df.columns
    has_fed_holdings = 'fed_holdings_pct' in df.columns

    # Show data availability status
    with st.expander("🔍 Data Availability Status", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Core Series:**")
            st.write(f"{'✅' if has_debt_data else '❌'} Total Public Debt (GFDEBTN)")
            st.write(f"{'✅' if has_rate_data else '❌'} Treasury Rates (TB3MS, DGS10)")
            st.write(f"{'✅' if has_curve_data else '❌'} Yield Curve (T10Y2Y)")
            st.write(f"{'✅' if has_tga_data else '❌'} Treasury General Account (TGA)")

        with col2:
            st.markdown("**Derived Metrics:**")
            st.write(f"{'✅' if has_bills_intensity else '❌'} Bills Intensity Proxy")
            st.write(f"{'✅' if has_wam_proxy else '❌'} WAM Curve Proxy")
            st.write(f"{'✅' if has_debt_velocity else '❌'} Debt Issuance Velocity")
            st.write(f"{'✅' if has_fed_holdings else '❌'} Fed Holdings %")

    if not has_rate_data:
        st.error("❌ Missing required Treasury rate data (TB3MS, DGS10). Cannot perform QRA analysis.")
        st.info("Please ensure FRED API is properly configured and Treasury rate series are available.")
        return

    # Get latest values
    latest = df.iloc[-1]
    latest_idx = df.index[-1]

    st.subheader("🎯 Current QRA Signals")
    st.caption(f"Data as of: {latest_idx.strftime('%Y-%m-%d')}")

    # Top row: Key gauges
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if has_bills_intensity:
            bills_intensity = latest.get('bills_intensity_proxy', np.nan)
            if not pd.isna(bills_intensity):
                fig_bills = create_qra_gauge(bills_intensity, "Bills Intensity", 30, 50)
                st.plotly_chart(fig_bills, use_container_width=True)

                # Interpretation
                if bills_intensity > 50:
                    st.warning("⚠️ High short-term rate pressure")
                elif bills_intensity < 30:
                    st.success("✅ Low short-term rate environment")
                else:
                    st.info("📊 Normal rate environment")
            else:
                st.warning("Bills intensity data unavailable")
        else:
            st.warning("Bills intensity metric not calculated")
            st.caption("Requires TB3MS and DGS10")

    with col2:
        if has_wam_proxy:
            wam = latest.get('wam_curve_proxy', np.nan)
            if not pd.isna(wam):
                fig_wam = create_qra_gauge(wam, "WAM Proxy", 40, 60)
                st.plotly_chart(fig_wam, use_container_width=True)

                if wam > 60:
                    st.success("✅ Extended maturity profile")
                elif wam < 40:
                    st.warning("⚠️ Shortened maturity profile")
                else:
                    st.info("📊 Balanced maturity")
            else:
                st.warning("WAM proxy unavailable")
        else:
            st.warning("WAM proxy not calculated")
            st.caption("Requires T10Y2Y curve data")

    with col3:
        if has_debt_velocity:
            velocity = latest.get('debt_issuance_velocity', np.nan)
            if not pd.isna(velocity):
                # Convert to 0-100 scale for gauge (velocity is in %)
                velocity_gauge = 50 + (velocity * 10)  # Scale to make visible
                velocity_gauge = np.clip(velocity_gauge, 0, 100)

                fig_vel = create_qra_gauge(velocity_gauge, "Issuance Velocity", 45, 55)
                st.plotly_chart(fig_vel, use_container_width=True)
                st.metric("Actual Velocity", f"{velocity:.2f}%",
                         help="20-day rate of change in total debt")
            else:
                st.warning("Debt velocity unavailable")
        else:
            st.warning("Debt velocity not calculated")
            st.caption("Requires GFDEBTN")

    with col4:
        if has_fed_holdings:
            fed_pct = latest.get('fed_holdings_pct', np.nan)
            if not pd.isna(fed_pct):
                fig_fed = create_qra_gauge(fed_pct, "Fed Holdings %", 10, 25, value_range=(0, 40))
                st.plotly_chart(fig_fed, use_container_width=True)

                if fed_pct > 25:
                    st.info("📊 High Fed ownership (QE legacy)")
                elif fed_pct < 10:
                    st.info("📊 Low Fed ownership (QT active)")
            else:
                st.warning("Fed holdings data unavailable")
        else:
            st.warning("Fed holdings not calculated")
            st.caption("Requires FDHBFRBN and GFDEBTN")

    st.markdown("---")

    # Detailed metrics
    st.subheader("📈 Detailed Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if has_debt_data:
            total_debt = latest.get('GFDEBTN', np.nan)
            if not pd.isna(total_debt):
                debt_change = df['GFDEBTN'].diff().iloc[-1]
                st.metric(
                    "Total Public Debt",
                    f"${total_debt:,.0f}B",
                    f"{debt_change:+,.0f}B",
                    help="Total US Public Debt Outstanding (GFDEBTN)"
                )
            else:
                st.metric("Total Public Debt", "N/A")
        else:
            st.metric("Total Public Debt", "Loading...")
            st.caption("⏳ GFDEBTN being fetched")

    with col2:
        if has_tga_data:
            tga = latest.get('TGA', np.nan)
            if not pd.isna(tga):
                tga_change = df['TGA'].diff().iloc[-1]
                st.metric(
                    "Treasury General Account",
                    f"${tga:,.0f}B",
                    f"{tga_change:+,.0f}B",
                    help="Treasury's cash balance at the Fed"
                )

                # TGA-Debt ratio if available
                if has_debt_data and 'tga_debt_ratio' in df.columns:
                    tga_ratio = latest.get('tga_debt_ratio', np.nan)
                    if not pd.isna(tga_ratio):
                        st.caption(f"TGA/Debt: {tga_ratio*100:.2f}%")
            else:
                st.metric("TGA", "N/A")
        else:
            st.metric("TGA", "N/A")

    with col3:
        tb3m = latest.get('TB3MS', np.nan)
        if not pd.isna(tb3m):
            tb3m_change = df['TB3MS'].diff().iloc[-1]
            st.metric(
                "3M T-Bill Rate",
                f"{tb3m:.2f}%",
                f"{tb3m_change:+.2f}pp",
                delta_color="off",
                help="3-Month Treasury Bill Secondary Market Rate"
            )
        else:
            st.metric("3M T-Bill Rate", "N/A")

    with col4:
        if has_curve_data:
            curve = latest.get('T10Y2Y', np.nan)
            if not pd.isna(curve):
                curve_change = df['T10Y2Y'].diff().iloc[-1]
                st.metric(
                    "2Y-10Y Curve",
                    f"{curve:.0f}bp",
                    f"{curve_change:+.0f}bp",
                    delta_color="off",
                    help="Yield curve slope (maturity preference indicator)"
                )

                if curve > 50:
                    st.caption("✅ Normal curve")
                elif curve < 0:
                    st.caption("⚠️ Inverted curve")
                else:
                    st.caption("📊 Flat curve")
            else:
                st.metric("2Y-10Y Curve", "N/A")
        else:
            st.metric("2Y-10Y Curve", "N/A")

    st.markdown("---")

    # Charts
    st.subheader("📊 Historical Trends - Last 12 Months")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Bills Intensity & WAM",
        "💰 Debt & TGA Dynamics",
        "📉 Treasury Rates",
        "🔄 Issuance Velocity"
    ])

    df_recent = df.tail(252)  # Last year

    with tab1:
        # Bills intensity and WAM proxy
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Bills Intensity Proxy (TB3MS/DGS10 Ratio)",
                           "Weighted Average Maturity Proxy"),
            vertical_spacing=0.15
        )

        if has_bills_intensity:
            bills_series = df_recent['bills_intensity_proxy']
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=bills_series,
                          mode='lines', name='Bills Intensity',
                          line=dict(color='blue', width=2),
                          fill='tozeroy', fillcolor='rgba(0, 100, 255, 0.2)'),
                row=1, col=1
            )
            fig.add_hline(y=50, line_dash="dash", line_color="orange", row=1, col=1,
                         annotation_text="High Threshold", annotation_position="right")
            fig.update_yaxes(title_text="Ratio Index", row=1, col=1)
        else:
            fig.add_annotation(text="Data not available", xref="x", yref="y",
                             x=0.5, y=0.5, showarrow=False, row=1, col=1)

        if has_wam_proxy:
            wam_series = df_recent['wam_curve_proxy']
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=wam_series,
                          mode='lines', name='WAM Proxy',
                          line=dict(color='green', width=2),
                          fill='tozeroy', fillcolor='rgba(0, 200, 0, 0.2)'),
                row=2, col=1
            )
            fig.add_hline(y=50, line_dash="solid", line_color="gray", row=2, col=1,
                         annotation_text="Neutral", annotation_position="right")
            fig.update_yaxes(title_text="WAM Index (0-100)", row=2, col=1)
        else:
            fig.add_annotation(text="Data not available", xref="x2", yref="y2",
                             x=0.5, y=0.5, showarrow=False, row=2, col=1)

        fig.update_layout(height=600, hovermode='x unified', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        if has_bills_intensity and has_wam_proxy:
            current_bills = df_recent['bills_intensity_proxy'].iloc[-1]
            current_wam = df_recent['wam_curve_proxy'].iloc[-1]
            avg_bills = df_recent['bills_intensity_proxy'].mean()
            avg_wam = df_recent['wam_curve_proxy'].mean()

            st.markdown("**Current vs Average:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bills Intensity", f"{current_bills:.1f}",
                         f"{current_bills - avg_bills:+.1f} vs avg")
            with col2:
                st.metric("WAM Proxy", f"{current_wam:.1f}",
                         f"{current_wam - avg_wam:+.1f} vs avg")

    with tab2:
        # Debt and TGA dynamics
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Total Public Debt Outstanding", "Treasury General Account"),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )

        if has_debt_data:
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['GFDEBTN'],
                          mode='lines', name='Total Debt',
                          line=dict(color='darkred', width=2),
                          fill='tozeroy', fillcolor='rgba(200, 0, 0, 0.2)'),
                row=1, col=1
            )
            fig.update_yaxes(title_text="Billions $", row=1, col=1)
        else:
            fig.add_annotation(text="GFDEBTN data being loaded from FRED",
                             xref="x", yref="y", x=0.5, y=0.5,
                             showarrow=False, font=dict(size=14, color="orange"),
                             row=1, col=1)

        if has_tga_data:
            # TGA level
            fig.add_trace(
                go.Scatter(x=df_recent.index, y=df_recent['TGA'],
                          mode='lines', name='TGA Level',
                          line=dict(color='purple', width=2)),
                row=2, col=1, secondary_y=False
            )

            # TGA changes (weekly)
            if 'delta_tga' in df.columns:
                tga_changes = df_recent['delta_tga']
                fig.add_trace(
                    go.Bar(x=df_recent.index, y=tga_changes,
                          name='TGA Change', marker_color='lightblue',
                          opacity=0.5),
                    row=2, col=1, secondary_y=True
                )
                fig.update_yaxes(title_text="Weekly Change ($B)", row=2, col=1, secondary_y=True)

            fig.update_yaxes(title_text="TGA Level ($B)", row=2, col=1, secondary_y=False)
        else:
            fig.add_annotation(text="TGA data not available",
                             xref="x2", yref="y2", x=0.5, y=0.5,
                             showarrow=False, row=2, col=1)

        fig.update_layout(height=600, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        **TGA Impact**: When TGA increases (Treasury builds cash), it drains reserves from the banking system.
        When TGA decreases (Treasury spends), it adds reserves.
        """)

    with tab3:
        # Treasury rates comparison
        fig = go.Figure()

        if 'TB3MS' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index, y=df_recent['TB3MS'],
                mode='lines', name='3M T-Bill',
                line=dict(color='blue', width=2)
            ))

        if 'YIELD_2Y' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index, y=df_recent['YIELD_2Y'],
                mode='lines', name='2Y Treasury',
                line=dict(color='green', width=2)
            ))

        if 'DGS10' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index, y=df_recent['DGS10'],
                mode='lines', name='10Y Treasury',
                line=dict(color='red', width=2)
            ))

        if 'YIELD_30Y' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index, y=df_recent['YIELD_30Y'],
                mode='lines', name='30Y Treasury',
                line=dict(color='darkred', width=2)
            ))

        fig.update_layout(
            title="Treasury Yield Curve - All Maturities",
            xaxis_title="Date",
            yaxis_title="Yield (%)",
            height=500,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99)
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Issuance velocity
        if has_debt_velocity:
            velocity_series = df_recent['debt_issuance_velocity']

            fig = go.Figure()

            # Color bars based on positive/negative
            colors = ['green' if x > 0 else 'red' for x in velocity_series]

            fig.add_trace(go.Bar(
                x=df_recent.index,
                y=velocity_series,
                name='Debt Issuance Velocity',
                marker_color=colors,
                opacity=0.7
            ))

            fig.add_hline(y=0, line_dash="solid", line_color="black")

            fig.update_layout(
                title="Debt Issuance Velocity (20-day % change)",
                xaxis_title="Date",
                yaxis_title="Velocity (%)",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            positive_pct = (velocity_series > 0.5).sum() / len(velocity_series) * 100
            negative_pct = (velocity_series < -0.5).sum() / len(velocity_series) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accelerating Issuance", f"{positive_pct:.1f}%",
                         help="Days with >0.5% velocity")
            with col2:
                st.metric("Decelerating Issuance", f"{negative_pct:.1f}%",
                         help="Days with <-0.5% velocity")
            with col3:
                neutral_pct = 100 - positive_pct - negative_pct
                st.metric("Stable Issuance", f"{neutral_pct:.1f}%",
                         help="Days with normal velocity")
        else:
            st.warning("Debt issuance velocity metric not available")
            st.info("This metric requires GFDEBTN (Total Public Debt) series from FRED")

    st.markdown("---")

    # Summary table
    st.subheader("📋 Treasury Debt Summary")

    if has_rate_data:
        summary_data = {
            'Metric': [],
            'Current Value': [],
            'Trend': []
        }

        # Add available metrics
        if has_debt_data:
            summary_data['Metric'].append('Total Public Debt')
            summary_data['Current Value'].append(f"${latest.get('GFDEBTN', 0):,.0f}B")
            summary_data['Trend'].append("↑" if df['GFDEBTN'].diff().iloc[-1] > 0 else "↓")

        if has_tga_data:
            summary_data['Metric'].append('Treasury General Account')
            summary_data['Current Value'].append(f"${latest.get('TGA', 0):,.0f}B")
            summary_data['Trend'].append("↑" if df['TGA'].diff().iloc[-1] > 0 else "↓")

        summary_data['Metric'].extend(['3M T-Bill Rate', '10Y Treasury Rate', '2Y-10Y Spread'])
        summary_data['Current Value'].extend([
            f"{latest.get('TB3MS', 0):.2f}%",
            f"{latest.get('DGS10', 0):.2f}%",
            f"{latest.get('T10Y2Y', 0):.0f}bp"
        ])
        summary_data['Trend'].extend([
            "↑" if df['TB3MS'].diff().iloc[-1] > 0 else "↓",
            "↑" if df['DGS10'].diff().iloc[-1] > 0 else "↓",
            "↑" if df['T10Y2Y'].diff().iloc[-1] > 0 else "↓"
        ])

        if has_bills_intensity:
            summary_data['Metric'].append('Bills Intensity Proxy')
            summary_data['Current Value'].append(f"{latest.get('bills_intensity_proxy', 0):.1f}")
            summary_data['Trend'].append("↑" if df['bills_intensity_proxy'].diff().iloc[-1] > 0 else "↓")

        if has_wam_proxy:
            summary_data['Metric'].append('WAM Proxy')
            summary_data['Current Value'].append(f"{latest.get('wam_curve_proxy', 0):.1f}")
            summary_data['Trend'].append("↑" if df['wam_curve_proxy'].diff().iloc[-1] > 0 else "↓")

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Insufficient data for summary table")

    # Market implications
    st.markdown("---")
    st.subheader("💡 Market Implications")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**More Bills Issuance (short-term):**")
        st.markdown("""
        - ✅ Adds collateral to money markets
        - ✅ Can support RRP drainage
        - ✅ Increases T-bill supply (may lower yields)
        - ⚠️ Increases refinancing frequency & rollover risk
        - 📊 Generally accommodative for risk assets
        - 💰 Lower WAM = higher refinancing risk
        """)

    with col2:
        st.markdown("**More Bonds Issuance (long-term):**")
        st.markdown("""
        - ⚠️ Drains collateral from money markets
        - ⚠️ Can pressure T-bill yields higher
        - ⚠️ May increase term premium
        - ✅ Extends maturity profile (reduces rollover risk)
        - ✅ Reduces refinancing frequency
        - 📊 Can be restrictive for duration-sensitive assets
        - 💪 Higher WAM = more stable debt structure
        """)

    # Data sources note
    st.markdown("---")
    st.caption("""
    **Data Sources**: Federal Reserve Economic Data (FRED)

    **Core Series:**
    - GFDEBTN: Total Public Debt Outstanding
    - FDHBFRBN: Federal Debt Held by Federal Reserve Banks
    - FYGFDPUN: Federal Debt Held by the Public
    - TGA (WTREGEN): Treasury General Account
    - TB3MS: 3-Month Treasury Bill Rate
    - DGS10: 10-Year Treasury Constant Maturity Rate
    - T10Y2Y: 10-Year minus 2-Year Treasury Spread

    **Derived Metrics** (calculated automatically):
    - Bills Intensity Proxy = (TB3MS / DGS10) × 100
    - WAM Curve Proxy = 50 + (T10Y2Y × 2)
    - Debt Issuance Velocity = 20-day % change in GFDEBTN
    - Fed Holdings % = (FDHBFRBN / GFDEBTN) × 100

    **Official QRA**: For official Treasury Quarterly Refunding Announcements, visit:
    https://home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding

    **Note**: This analysis uses proxy metrics based on market rates, yield curve dynamics, and debt outstanding data.
    Actual Treasury issuance composition may differ from these market-based indicators.
    """)
