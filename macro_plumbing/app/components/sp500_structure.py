"""
S&P 500 Market Structure Analysis
==================================

Academic framework for detecting regime changes through technical structure.

Based on:
- Lo, Mamaysky & Wang (2000): "Foundations of Technical Analysis" - Journal of Finance
- Neely, Weller & Ulrich (2009): "Adaptive Markets Hypothesis" - Fed St. Louis
- Osler (2000): "Support for Resistance" - Federal Reserve Bank of New York

Implements:
1. Market Structure Detection (HH, HL, LH, LL)
2. Swing High/Low Identification
3. Support/Resistance Levels
4. Break of Structure (BOS) alerts
5. Change of Character (CHoCH) detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from typing import Tuple, List, Dict


def detect_swing_points(series: pd.Series, order: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Detect swing highs and lows using local extrema.

    Parameters
    ----------
    series : pd.Series
        Price series
    order : int
        Number of points on each side to use for comparison

    Returns
    -------
    swing_highs, swing_lows : Tuple[pd.Series, pd.Series]
        Boolean series indicating swing points
    """
    # Find local maxima (swing highs)
    highs_idx = argrelextrema(series.values, np.greater, order=order)[0]
    swing_highs = pd.Series(False, index=series.index)
    swing_highs.iloc[highs_idx] = True

    # Find local minima (swing lows)
    lows_idx = argrelextrema(series.values, np.less, order=order)[0]
    swing_lows = pd.Series(False, index=series.index)
    swing_lows.iloc[lows_idx] = True

    return swing_highs, swing_lows


def identify_market_structure(df: pd.DataFrame, price_col: str = 'SP500') -> Dict:
    """
    Identify current market structure (HH/HL vs LH/LL).

    Based on Lo et al. (2000) pattern recognition framework.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    price_col : str
        Column name for price

    Returns
    -------
    dict
        Market structure analysis
    """
    if price_col not in df.columns:
        return {
            'structure': 'N/A',
            'trend': 'Unknown',
            'last_swing_high': None,
            'last_swing_low': None,
            'prev_swing_high': None,
            'prev_swing_low': None
        }

    prices = df[price_col].dropna()

    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(prices, order=10)

    # Get last 2 swing highs and lows
    high_dates = swing_highs[swing_highs].index[-2:] if swing_highs.sum() >= 2 else []
    low_dates = swing_lows[swing_lows].index[-2:] if swing_lows.sum() >= 2 else []

    if len(high_dates) < 2 or len(low_dates) < 2:
        return {
            'structure': 'Insufficient data',
            'trend': 'Unknown',
            'last_swing_high': None,
            'last_swing_low': None,
            'prev_swing_high': None,
            'prev_swing_low': None
        }

    # Extract values
    last_high = prices.loc[high_dates[-1]]
    prev_high = prices.loc[high_dates[-2]]
    last_low = prices.loc[low_dates[-1]]
    prev_low = prices.loc[low_dates[-2]]

    # Determine structure
    higher_highs = last_high > prev_high
    higher_lows = last_low > prev_low
    lower_highs = last_high < prev_high
    lower_lows = last_low < prev_low

    if higher_highs and higher_lows:
        structure = "HH + HL"
        trend = "Bullish (Uptrend)"
    elif lower_highs and lower_lows:
        structure = "LH + LL"
        trend = "Bearish (Downtrend)"
    elif higher_highs and lower_lows:
        structure = "HH + LL"
        trend = "Expansion/Volatility"
    elif lower_highs and higher_lows:
        structure = "LH + HL"
        trend = "Contraction/Consolidation"
    else:
        structure = "Mixed"
        trend = "Unclear"

    return {
        'structure': structure,
        'trend': trend,
        'last_swing_high': last_high,
        'last_swing_low': last_low,
        'prev_swing_high': prev_high,
        'prev_swing_low': prev_low,
        'last_high_date': high_dates[-1],
        'last_low_date': low_dates[-1],
        'higher_highs': higher_highs,
        'higher_lows': higher_lows
    }


def calculate_support_resistance(df: pd.DataFrame, price_col: str = 'SP500',
                                 window: int = 252, n_levels: int = 5) -> Dict:
    """
    Calculate key support and resistance levels.

    Based on Osler (2000): levels act as self-fulfilling prophecies.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    price_col : str
        Column name
    window : int
        Lookback window
    n_levels : int
        Number of levels to identify

    Returns
    -------
    dict
        Support and resistance levels
    """
    if price_col not in df.columns:
        return {'support': [], 'resistance': []}

    recent = df[price_col].tail(window).dropna()

    if len(recent) == 0:
        return {'support': [], 'resistance': []}

    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(recent, order=5)

    # Get swing high/low values
    high_levels = recent[swing_highs].values
    low_levels = recent[swing_lows].values

    # Cluster levels (combine similar levels within 0.5%)
    def cluster_levels(levels, tolerance=0.005):
        if len(levels) == 0:
            return []

        sorted_levels = np.sort(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) < tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]

        clusters.append(np.mean(current_cluster))
        return clusters

    resistance_levels = cluster_levels(high_levels)
    support_levels = cluster_levels(low_levels)

    # Sort and take top N by proximity to current price
    current_price = recent.iloc[-1]

    # Resistance: levels above current price
    resistance = [r for r in resistance_levels if r > current_price]
    resistance.sort()
    resistance = resistance[:n_levels]

    # Support: levels below current price
    support = [s for s in support_levels if s < current_price]
    support.sort(reverse=True)
    support = support[:n_levels]

    return {
        'support': support,
        'resistance': resistance,
        'current_price': current_price
    }


def detect_break_of_structure(df: pd.DataFrame, structure: Dict,
                               price_col: str = 'SP500') -> Dict:
    """
    Detect if there's been a recent Break of Structure (BOS).

    BOS = price breaks the last significant swing point in the direction of trend.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    structure : dict
        Market structure from identify_market_structure()
    price_col : str
        Price column

    Returns
    -------
    dict
        BOS detection results
    """
    if price_col not in df.columns or structure['structure'] == 'N/A':
        return {
            'bos_detected': False,
            'bos_type': None,
            'description': 'Insufficient data'
        }

    current_price = df[price_col].iloc[-1]

    # Check for bullish BOS (price breaks above last swing high)
    if structure.get('last_swing_high') is not None:
        if current_price > structure['last_swing_high']:
            return {
                'bos_detected': True,
                'bos_type': 'Bullish',
                'level': structure['last_swing_high'],
                'description': f'Price broke above {structure["last_swing_high"]:.2f}'
            }

    # Check for bearish BOS (price breaks below last swing low)
    if structure.get('last_swing_low') is not None:
        if current_price < structure['last_swing_low']:
            return {
                'bos_detected': True,
                'bos_type': 'Bearish',
                'level': structure['last_swing_low'],
                'description': f'Price broke below {structure["last_swing_low"]:.2f}'
            }

    return {
        'bos_detected': False,
        'bos_type': None,
        'description': 'No recent BOS'
    }


def create_structure_chart(df: pd.DataFrame, price_col: str = 'SP500') -> go.Figure:
    """
    Create interactive chart showing market structure.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with structure annotations
    price_col : str
        Price column name

    Returns
    -------
    go.Figure
        Plotly figure
    """
    if price_col not in df.columns:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="S&P 500 data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    # Get last 252 days (~1 year)
    df_recent = df.tail(252).copy()
    prices = df_recent[price_col].dropna()

    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(prices, order=10)

    # Create figure
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices.values,
        mode='lines',
        name='S&P 500',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    # Mark swing highs
    high_points = prices[swing_highs]
    if len(high_points) > 0:
        fig.add_trace(go.Scatter(
            x=high_points.index,
            y=high_points.values,
            mode='markers',
            name='Swing High',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            hovertemplate='High: %{y:.2f}<extra></extra>'
        ))

    # Mark swing lows
    low_points = prices[swing_lows]
    if len(low_points) > 0:
        fig.add_trace(go.Scatter(
            x=low_points.index,
            y=low_points.values,
            mode='markers',
            name='Swing Low',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            hovertemplate='Low: %{y:.2f}<extra></extra>'
        ))

    # Add moving averages
    if len(prices) >= 50:
        ma_50 = prices.rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=ma_50.index,
            y=ma_50.values,
            mode='lines',
            name='MA 50',
            line=dict(color='orange', width=1, dash='dash'),
            opacity=0.7
        ))

    if len(prices) >= 200:
        ma_200 = prices.rolling(200).mean()
        fig.add_trace(go.Scatter(
            x=ma_200.index,
            y=ma_200.values,
            mode='lines',
            name='MA 200',
            line=dict(color='purple', width=1, dash='dash'),
            opacity=0.7
        ))

    fig.update_layout(
        title="S&P 500 Market Structure - Last 12 Months",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


def render_sp500_structure(df: pd.DataFrame):
    """
    Main function to render S&P 500 Market Structure dashboard.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with S&P 500 data
    """
    st.header("📈 S&P 500 Market Structure & Regime Detection")
    st.caption("Academic framework: Lo et al. (2000), Neely et al. (2009), Osler (2000)")

    # Check if SP500 data exists
    if 'SP500' not in df.columns:
        st.error("S&P 500 data not available in dataset")
        st.info("This analysis requires SP500 series from FRED")
        return

    # Add methodology note
    with st.expander("ℹ️ Methodology & Academic Foundation"):
        st.markdown("""
        **Academic Framework:**

        1. **Lo, Mamaysky & Wang (2000)** - Journal of Finance
           - "Foundations of Technical Analysis: Computational Algorithms..."
           - Technical patterns have statistically significant predictive power
           - Market structure changes indicate regime shifts

        2. **Neely, Weller & Ulrich (2009)** - Federal Reserve St. Louis
           - "The Adaptive Markets Hypothesis"
           - Technical rules identify regime changes before they're obvious
           - Patterns work until too many traders use them

        3. **Osler (2000)** - Federal Reserve Bank of New York
           - "Support for Resistance"
           - Support/resistance levels predict trend interruptions
           - Self-fulfilling prophecies through stop-loss clustering

        **What We Detect:**
        - **Higher Highs (HH) + Higher Lows (HL)**: Bullish regime
        - **Lower Highs (LH) + Lower Lows (LL)**: Bearish regime
        - **Break of Structure (BOS)**: Regime change signal
        - **Support/Resistance**: Key psychological levels
        """)

    # Analyze market structure
    structure = identify_market_structure(df, 'SP500')
    levels = calculate_support_resistance(df, 'SP500')
    bos = detect_break_of_structure(df, structure, 'SP500')

    # Section 1: Current Market Structure
    st.subheader("🔍 Current Market Structure")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        trend_emoji = "📈" if "Bullish" in structure['trend'] else "📉" if "Bearish" in structure['trend'] else "➡️"
        st.metric(
            label="Trend",
            value=structure['trend'],
            help="Based on swing high/low analysis"
        )

    with col2:
        st.metric(
            label="Structure",
            value=structure['structure'],
            help="HH/HL = Bullish, LH/LL = Bearish"
        )

    with col3:
        current_price = df['SP500'].iloc[-1] if len(df) > 0 else 0
        st.metric(
            label="Current Price",
            value=f"{current_price:.2f}",
            delta=f"{df['SP500'].pct_change().iloc[-1]*100:.2f}%" if len(df) > 1 else None
        )

    with col4:
        if bos['bos_detected']:
            st.metric(
                label="Break of Structure",
                value=bos['bos_type'],
                help=bos['description']
            )
            if bos['bos_type'] == 'Bullish':
                st.success("⚠️ Bullish BOS detected!")
            else:
                st.error("⚠️ Bearish BOS detected!")
        else:
            st.metric(
                label="Break of Structure",
                value="None",
                help="No recent BOS"
            )

    # Section 2: Support & Resistance Levels
    st.markdown("---")
    st.subheader("📊 Key Support & Resistance Levels")

    col_levels1, col_levels2 = st.columns(2)

    with col_levels1:
        st.markdown("**Resistance (Above)**")
        if levels['resistance']:
            for i, level in enumerate(levels['resistance'], 1):
                distance = ((level - levels['current_price']) / levels['current_price']) * 100
                st.write(f"R{i}: **{level:.2f}** (+{distance:.2f}%)")
        else:
            st.caption("No resistance levels identified")

    with col_levels2:
        st.markdown("**Support (Below)**")
        if levels['support']:
            for i, level in enumerate(levels['support'], 1):
                distance = ((levels['current_price'] - level) / levels['current_price']) * 100
                st.write(f"S{i}: **{level:.2f}** (-{distance:.2f}%)")
        else:
            st.caption("No support levels identified")

    # Section 3: Price Chart with Structure
    st.markdown("---")
    st.subheader("📈 Market Structure Chart")

    fig = create_structure_chart(df, 'SP500')

    # Add support/resistance lines
    if levels['resistance']:
        for i, level in enumerate(levels['resistance'][:3], 1):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text=f"R{i}",
                annotation_position="right"
            )

    if levels['support']:
        for i, level in enumerate(levels['support'][:3], 1):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="green",
                opacity=0.5,
                annotation_text=f"S{i}",
                annotation_position="right"
            )

    st.plotly_chart(fig, use_container_width=True)

    # Section 4: Swing Points Detail
    st.markdown("---")
    st.subheader("🎯 Recent Swing Points")

    if structure['last_swing_high'] is not None:
        col_swing1, col_swing2 = st.columns(2)

        with col_swing1:
            st.markdown("**Last Swing High**")
            st.write(f"Price: **{structure['last_swing_high']:.2f}**")
            if 'last_high_date' in structure:
                st.write(f"Date: {structure['last_high_date'].strftime('%Y-%m-%d')}")

            st.markdown("**Previous Swing High**")
            st.write(f"Price: **{structure['prev_swing_high']:.2f}**")

            if structure.get('higher_highs'):
                st.success("✅ Higher High (bullish)")
            else:
                st.warning("⚠️ Lower High (bearish)")

        with col_swing2:
            st.markdown("**Last Swing Low**")
            st.write(f"Price: **{structure['last_swing_low']:.2f}**")
            if 'last_low_date' in structure:
                st.write(f"Date: {structure['last_low_date'].strftime('%Y-%m-%d')}")

            st.markdown("**Previous Swing Low**")
            st.write(f"Price: **{structure['prev_swing_low']:.2f}**")

            if structure.get('higher_lows'):
                st.success("✅ Higher Low (bullish)")
            else:
                st.warning("⚠️ Lower Low (bearish)")
    else:
        st.info("Insufficient data to identify swing points")
