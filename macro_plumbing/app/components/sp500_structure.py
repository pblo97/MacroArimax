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


def calculate_trend_strength(df: pd.DataFrame, price_col: str = 'SP500', window: int = 14) -> Dict:
    """
    Calculate trend strength metrics (ADX-like measure).

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    price_col : str
        Price column
    window : int
        Calculation window

    Returns
    -------
    dict
        Trend strength metrics
    """
    if price_col not in df.columns or len(df) < window * 2:
        return {
            'strength': 0,
            'direction_score': 0,
            'consistency': 0,
            'slope': 0
        }

    prices = df[price_col].tail(252).dropna()

    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(prices, order=10)
    high_points = prices[swing_highs]
    low_points = prices[swing_lows]

    # Calculate slope of swing highs and lows
    if len(high_points) >= 2:
        high_slope = (high_points.iloc[-1] - high_points.iloc[0]) / len(high_points)
    else:
        high_slope = 0

    if len(low_points) >= 2:
        low_slope = (low_points.iloc[-1] - low_points.iloc[0]) / len(low_points)
    else:
        low_slope = 0

    # Direction score: both slopes pointing same direction
    if high_slope > 0 and low_slope > 0:
        direction_score = 100  # Strong bullish
        is_aligned = True
    elif high_slope < 0 and low_slope < 0:
        direction_score = -100  # Strong bearish
        is_aligned = True
    else:
        # Mixed signals (HH+LL or LH+HL) = conflicting directions
        # Calculate net bias but recognize the conflict
        max_slope = max(abs(high_slope), abs(low_slope))
        if max_slope > 0:
            # Net direction as percentage of dominant slope
            net_slope = (high_slope + low_slope) / 2
            direction_score = (net_slope / max_slope) * 100
        else:
            direction_score = 0
        is_aligned = False

    # Consistency: how many swing points follow the trend
    total_swings = len(high_points) + len(low_points)
    if total_swings > 0:
        consistency = abs(direction_score) / 100 * 100
    else:
        consistency = 0

    # Overall strength (0-100)
    strength = min(abs(direction_score), 100)

    # CRITICAL: Cap strength for mixed structure (expansion/contraction)
    # HH+LL or LH+HL indicates indecision, max strength should be 60
    if not is_aligned:
        strength = min(strength, 60)

    # Average slope
    avg_slope = (high_slope + low_slope) / 2

    return {
        'strength': strength,
        'direction_score': direction_score,
        'consistency': consistency,
        'slope': avg_slope,
        'high_slope': high_slope,
        'low_slope': low_slope
    }


def calculate_risk_reward(current_price: float, resistance_levels: List[float],
                         support_levels: List[float]) -> List[Dict]:
    """
    Calculate risk/reward ratios for potential trades.

    Parameters
    ----------
    current_price : float
        Current market price
    resistance_levels : list
        List of resistance levels
    support_levels : list
        List of support levels

    Returns
    -------
    list of dict
        R:R analysis for each target
    """
    rr_analysis = []

    if not support_levels:
        return rr_analysis

    # Suggested stop: nearest support
    stop_loss = support_levels[0] if support_levels else current_price * 0.98
    risk = current_price - stop_loss

    if risk <= 0:
        return rr_analysis

    # Calculate R:R for each resistance target
    for i, target in enumerate(resistance_levels, 1):
        reward = target - current_price
        if reward > 0:
            rr_ratio = reward / risk
            rr_analysis.append({
                'target_level': f'R{i}',
                'target_price': target,
                'reward_pct': (reward / current_price) * 100,
                'risk_pct': (risk / current_price) * 100,
                'rr_ratio': rr_ratio,
                'stop_loss': stop_loss
            })

    return rr_analysis


def detect_change_of_character(df: pd.DataFrame, structure: Dict,
                                price_col: str = 'SP500') -> Dict:
    """
    Detect Change of Character (CHoCH) - early warning before full BOS.

    CHoCH = swing points showing weakness before full structure break.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    structure : dict
        Current market structure
    price_col : str
        Price column

    Returns
    -------
    dict
        CHoCH detection results
    """
    if price_col not in df.columns or structure['structure'] == 'N/A':
        return {
            'choch_detected': False,
            'warning_level': 'None',
            'description': 'Insufficient data'
        }

    prices = df[price_col].tail(100).dropna()

    if len(prices) < 50:
        return {
            'choch_detected': False,
            'warning_level': 'None',
            'description': 'Insufficient data'
        }

    # Detect swing points on shorter timeframe
    swing_highs, swing_lows = detect_swing_points(prices, order=3)

    high_points = prices[swing_highs]
    low_points = prices[swing_lows]

    choch_detected = False
    warning_level = 'None'
    description = 'No CHoCH detected'

    # For bullish structure (HH + HL), watch for failing highs
    if structure['trend'] == 'Bullish (Uptrend)':
        if len(high_points) >= 3:
            # Check if recent highs are weakening
            if high_points.iloc[-1] < high_points.iloc[-2]:
                choch_detected = True
                warning_level = 'Medium'
                description = 'Recent swing high failed to make new high - potential CHoCH'

                # Check if it's getting worse
                if len(high_points) >= 3 and high_points.iloc[-1] < high_points.iloc[-3]:
                    warning_level = 'High'
                    description = 'Multiple swing highs weakening - CHoCH warning'

    # For bearish structure (LH + LL), watch for failing lows
    elif structure['trend'] == 'Bearish (Downtrend)':
        if len(low_points) >= 3:
            # Check if recent lows are weakening
            if low_points.iloc[-1] > low_points.iloc[-2]:
                choch_detected = True
                warning_level = 'Medium'
                description = 'Recent swing low failed to make new low - potential CHoCH'

                if len(low_points) >= 3 and low_points.iloc[-1] > low_points.iloc[-3]:
                    warning_level = 'High'
                    description = 'Multiple swing lows weakening - CHoCH warning'

    return {
        'choch_detected': choch_detected,
        'warning_level': warning_level,
        'description': description
    }


def calculate_proximity_alerts(current_price: float, resistance_levels: List[float],
                               support_levels: List[float], threshold_pct: float = 0.5) -> Dict:
    """
    Calculate proximity alerts for nearby key levels.

    Parameters
    ----------
    current_price : float
        Current price
    resistance_levels : list
        Resistance levels
    support_levels : list
        Support levels
    threshold_pct : float
        Alert threshold in percentage

    Returns
    -------
    dict
        Proximity alerts
    """
    alerts = []

    # Check resistance levels
    for i, level in enumerate(resistance_levels, 1):
        distance_pct = abs((level - current_price) / current_price * 100)
        if distance_pct <= threshold_pct:
            alerts.append({
                'level': f'R{i}',
                'price': level,
                'distance_pct': distance_pct,
                'type': 'resistance',
                'urgency': 'HIGH' if distance_pct < 0.25 else 'MEDIUM'
            })

    # Check support levels
    for i, level in enumerate(support_levels, 1):
        distance_pct = abs((current_price - level) / current_price * 100)
        if distance_pct <= threshold_pct:
            alerts.append({
                'level': f'S{i}',
                'price': level,
                'distance_pct': distance_pct,
                'type': 'support',
                'urgency': 'HIGH' if distance_pct < 0.25 else 'MEDIUM'
            })

    # Determine market position
    if alerts:
        if all(a['type'] == 'resistance' for a in alerts):
            position = 'Near resistance - critical decision zone'
        elif all(a['type'] == 'support' for a in alerts):
            position = 'Near support - critical decision zone'
        else:
            position = 'Within tight range - consolidation'
    else:
        position = 'No nearby key levels - clear to trend'

    return {
        'alerts': alerts,
        'position': position,
        'has_alerts': len(alerts) > 0
    }


def calculate_historical_stats(df: pd.DataFrame, price_col: str = 'SP500',
                               lookback_days: int = 252) -> Dict:
    """
    Calculate historical statistics for current structure type.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    price_col : str
        Price column
    lookback_days : int
        Analysis window

    Returns
    -------
    dict
        Historical statistics
    """
    if price_col not in df.columns:
        return {}

    prices = df[price_col].tail(lookback_days).dropna()

    if len(prices) < 50:
        return {}

    current_price = prices.iloc[-1]

    # Calculate statistics
    ath = prices.max()
    atl = prices.min()
    drawdown = (current_price - ath) / ath * 100
    recovery = (current_price - atl) / atl * 100

    # Volatility
    returns = prices.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized

    # Trend metrics
    days_from_ath = (prices.index[-1] - prices.idxmax()).days
    days_from_atl = (prices.index[-1] - prices.idxmin()).days

    # Average move statistics
    avg_daily_move = abs(returns.mean()) * 100
    max_daily_gain = returns.max() * 100
    max_daily_loss = returns.min() * 100

    return {
        'ath': ath,
        'atl': atl,
        'drawdown_from_ath': drawdown,
        'recovery_from_atl': recovery,
        'days_from_ath': days_from_ath,
        'days_from_atl': days_from_atl,
        'annualized_volatility': volatility,
        'avg_daily_move': avg_daily_move,
        'max_daily_gain': max_daily_gain,
        'max_daily_loss': max_daily_loss
    }


def detect_liquidity_zones(df: pd.DataFrame, price_col: str = 'SP500') -> Dict:
    """
    Detect potential liquidity zones (stop loss clusters).

    Based on Osler (2000): stops cluster around round numbers and S/R levels.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    price_col : str
        Price column

    Returns
    -------
    dict
        Liquidity zones
    """
    if price_col not in df.columns:
        return {'zones': []}

    prices = df[price_col].tail(252).dropna()
    current_price = prices.iloc[-1]

    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(prices, order=5)

    high_levels = prices[swing_highs].values
    low_levels = prices[swing_lows].values

    liquidity_zones = []

    # Above current price: long stop losses likely above swing highs
    above_levels = [h for h in high_levels if h > current_price]
    for level in above_levels[:5]:
        liquidity_zones.append({
            'price': level,
            'type': 'Long Stops',
            'description': f'Stop cluster above {level:.2f}',
            'direction': 'above'
        })

    # Below current price: short stop losses likely below swing lows
    below_levels = [l for l in low_levels if l < current_price]
    for level in sorted(below_levels, reverse=True)[:5]:
        liquidity_zones.append({
            'price': level,
            'type': 'Short Stops',
            'description': f'Stop cluster below {level:.2f}',
            'direction': 'below'
        })

    return {'zones': liquidity_zones}


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


def get_macro_context(df: pd.DataFrame) -> Dict:
    """
    Extract macro context from the dataframe.

    Integrates with macro dashboard crisis indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with macro indicators

    Returns
    -------
    dict
        Macro context summary
    """
    latest = df.iloc[-1] if len(df) > 0 else pd.Series()

    context = {
        'crisis_score': latest.get('crisis_composite', np.nan),
        'vix': latest.get('VIX', np.nan),
        'hy_oas': latest.get('HY_OAS', np.nan),
        'liquidity_score': latest.get('stress_score', np.nan),
        'liquidity_regime': 'Unknown'
    }

    # Determine liquidity regime
    # Note: stress_score works inversely - higher values = more stress = less liquidity
    if not np.isnan(context['liquidity_score']):
        if context['liquidity_score'] < 0.3:
            context['liquidity_regime'] = 'Ample Liquidity'
        elif context['liquidity_score'] < 0.5:
            context['liquidity_regime'] = 'Moderate Liquidity'
        elif context['liquidity_score'] < 0.7:
            context['liquidity_regime'] = 'Neutral'
        elif context['liquidity_score'] < 1.0:
            context['liquidity_regime'] = 'Tight Liquidity'
        else:
            context['liquidity_regime'] = 'Severe Stress'

    # Volatility regime
    context['volatility_regime'] = 'Unknown'
    if not np.isnan(context['vix']):
        if context['vix'] < 15:
            context['volatility_regime'] = 'Low (Complacent)'
        elif context['vix'] < 20:
            context['volatility_regime'] = 'Normal'
        elif context['vix'] < 30:
            context['volatility_regime'] = 'Elevated'
        else:
            context['volatility_regime'] = 'High (Stress)'

    return context


def analyze_multi_timeframe(df: pd.DataFrame, price_col: str = 'SP500') -> Dict:
    """
    Analyze market structure across multiple timeframes.

    Parameters
    ----------
    df : pd.DataFrame
        Daily price data
    price_col : str
        Price column

    Returns
    -------
    dict
        Multi-timeframe analysis
    """
    if price_col not in df.columns:
        return {
            'daily': {'trend': 'Unknown', 'structure': 'N/A'},
            'weekly': {'trend': 'Unknown', 'structure': 'N/A'},
            'monthly': {'trend': 'Unknown', 'structure': 'N/A'},
            'alignment': 'Unknown'
        }

    # Daily structure (already have this)
    daily_structure = identify_market_structure(df, price_col)

    # Resample to weekly
    df_weekly = df.copy()
    df_weekly[price_col + '_weekly'] = df[price_col].resample('W').last()
    df_weekly = df_weekly.dropna(subset=[price_col + '_weekly'])

    weekly_structure = identify_market_structure(df_weekly, price_col + '_weekly') if len(df_weekly) > 50 else {'trend': 'Insufficient data', 'structure': 'N/A'}

    # Resample to monthly
    df_monthly = df.copy()
    df_monthly[price_col + '_monthly'] = df[price_col].resample('M').last()
    df_monthly = df_monthly.dropna(subset=[price_col + '_monthly'])

    monthly_structure = identify_market_structure(df_monthly, price_col + '_monthly') if len(df_monthly) > 24 else {'trend': 'Insufficient data', 'structure': 'N/A'}

    # Calculate alignment
    bullish_count = 0
    bearish_count = 0

    if 'Bullish' in daily_structure['trend']:
        bullish_count += 1
    elif 'Bearish' in daily_structure['trend']:
        bearish_count += 1

    if 'Bullish' in weekly_structure['trend']:
        bullish_count += 1
    elif 'Bearish' in weekly_structure['trend']:
        bearish_count += 1

    if 'Bullish' in monthly_structure['trend']:
        bullish_count += 1
    elif 'Bearish' in monthly_structure['trend']:
        bearish_count += 1

    if bullish_count >= 2:
        alignment = f'{bullish_count}/3 Bullish - Strong Alignment'
    elif bearish_count >= 2:
        alignment = f'{bearish_count}/3 Bearish - Strong Alignment'
    else:
        alignment = 'Mixed - No clear alignment'

    return {
        'daily': {
            'trend': daily_structure['trend'],
            'structure': daily_structure['structure']
        },
        'weekly': {
            'trend': weekly_structure['trend'],
            'structure': weekly_structure['structure']
        },
        'monthly': {
            'trend': monthly_structure['trend'],
            'structure': monthly_structure['structure']
        },
        'alignment': alignment,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count
    }


def calculate_fibonacci_levels(df: pd.DataFrame, structure: Dict,
                               price_col: str = 'SP500') -> Dict:
    """
    Calculate Fibonacci retracement and extension levels.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    structure : dict
        Market structure
    price_col : str
        Price column

    Returns
    -------
    dict
        Fibonacci levels
    """
    if price_col not in df.columns or structure['structure'] == 'N/A':
        return {'retracements': [], 'extensions': []}

    prices = df[price_col].tail(252).dropna()

    if len(prices) < 2:
        return {'retracements': [], 'extensions': []}

    # Get swing high and low for calculation
    swing_high = structure.get('last_swing_high', prices.max())
    swing_low = structure.get('last_swing_low', prices.min())

    if swing_high is None or swing_low is None:
        return {'retracements': [], 'extensions': []}

    # Calculate range
    price_range = swing_high - swing_low

    # Fibonacci retracement levels (from swing high to swing low)
    fib_ratios_retrace = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    retracements = []

    for ratio in fib_ratios_retrace:
        level = swing_high - (price_range * ratio)
        retracements.append({
            'ratio': ratio,
            'price': level,
            'label': f'{ratio:.3f}'
        })

    # Fibonacci extension levels (beyond swing high)
    fib_ratios_ext = [1.272, 1.414, 1.618, 2.0, 2.618]
    extensions = []

    for ratio in fib_ratios_ext:
        level = swing_high + (price_range * (ratio - 1))
        extensions.append({
            'ratio': ratio,
            'price': level,
            'label': f'{ratio:.3f}'
        })

    return {
        'retracements': retracements,
        'extensions': extensions,
        'swing_high': swing_high,
        'swing_low': swing_low,
        'range': price_range
    }


def calculate_performance_metrics(df: pd.DataFrame, price_col: str = 'SP500',
                                  lookback_days: int = 252) -> Dict:
    """
    Calculate performance metrics for BOS detection system.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    price_col : str
        Price column
    lookback_days : int
        Analysis window

    Returns
    -------
    dict
        Performance statistics
    """
    if price_col not in df.columns:
        return {}

    # Get historical data
    prices = df[price_col].tail(lookback_days).dropna()

    if len(prices) < 100:
        return {}

    # Detect all historical BOS events
    bos_events = []

    # Rolling window to detect structure changes
    for i in range(50, len(prices), 10):  # Check every 10 days
        window_df = df.iloc[:i]
        structure = identify_market_structure(window_df, price_col)
        bos = detect_break_of_structure(window_df, structure, price_col)

        if bos['bos_detected']:
            bos_events.append({
                'date': prices.index[i],
                'price': prices.iloc[i],
                'type': bos['bos_type'],
                'level': bos.get('level', 0)
            })

    if len(bos_events) == 0:
        return {
            'bos_count': 0,
            'avg_move_after_bos': 0,
            'success_rate': 0,
            'avg_days_to_reversal': 0
        }

    # Calculate metrics for BOS events
    moves_after_bos = []
    days_to_reversal = []

    for event in bos_events:
        event_idx = prices.index.get_loc(event['date'])

        # Look at price movement 20 days after BOS
        if event_idx + 20 < len(prices):
            future_prices = prices.iloc[event_idx:event_idx+20]
            move_pct = ((future_prices.iloc[-1] - event['price']) / event['price']) * 100
            moves_after_bos.append(move_pct)

            # Find days to reversal (price crosses back through BOS level)
            if event['type'] == 'Bullish':
                reversal_idx = future_prices[future_prices < event['level']].index
            else:
                reversal_idx = future_prices[future_prices > event['level']].index

            if len(reversal_idx) > 0:
                days = (reversal_idx[0] - event['date']).days
                days_to_reversal.append(days)

    # Calculate success rate (BOS followed by continuation)
    successful_bos = sum(1 for move in moves_after_bos if abs(move) > 1)  # At least 1% move
    success_rate = (successful_bos / len(moves_after_bos) * 100) if moves_after_bos else 0

    return {
        'bos_count': len(bos_events),
        'avg_move_after_bos': np.mean(moves_after_bos) if moves_after_bos else 0,
        'success_rate': success_rate,
        'avg_days_to_reversal': np.mean(days_to_reversal) if days_to_reversal else 0,
        'recent_bos_events': bos_events[-5:]  # Last 5 events
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

    # Calculate all metrics
    structure = identify_market_structure(df, 'SP500')
    levels = calculate_support_resistance(df, 'SP500')
    bos = detect_break_of_structure(df, structure, 'SP500')
    trend_strength = calculate_trend_strength(df, 'SP500')
    rr_analysis = calculate_risk_reward(levels.get('current_price', 0), levels.get('resistance', []), levels.get('support', []))
    choch = detect_change_of_character(df, structure, 'SP500')
    proximity = calculate_proximity_alerts(levels.get('current_price', 0), levels.get('resistance', []), levels.get('support', []))
    hist_stats = calculate_historical_stats(df, 'SP500')
    liq_zones = detect_liquidity_zones(df, 'SP500')
    macro_ctx = get_macro_context(df)
    mtf_analysis = analyze_multi_timeframe(df, 'SP500')
    fib_levels = calculate_fibonacci_levels(df, structure, 'SP500')
    performance = calculate_performance_metrics(df, 'SP500')

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
        - **Change of Character (CHoCH)**: Early warning before BOS
        - **Support/Resistance**: Key psychological levels
        - **Liquidity Zones**: Stop loss clusters
        """)

    # Section 1: Current Market Structure & Macro Context
    st.subheader("🔍 Current Market Structure")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        current_price = df['SP500'].iloc[-1] if len(df) > 0 else 0
        st.metric(
            label="Current Price",
            value=f"{current_price:.2f}",
            delta=f"{df['SP500'].pct_change().iloc[-1]*100:.2f}%" if len(df) > 1 else None
        )

    with col2:
        st.metric(
            label="Trend",
            value=structure['trend'].split('(')[0].strip(),
            help=f"Structure: {structure['structure']}"
        )

    with col3:
        strength_pct = trend_strength.get('strength', 0)
        st.metric(
            label="Trend Strength",
            value=f"{strength_pct:.0f}/100",
            help=f"Direction score: {trend_strength.get('direction_score', 0):.0f}"
        )

    with col4:
        if bos['bos_detected']:
            st.metric(
                label="Break of Structure",
                value=bos['bos_type'],
                help=bos['description']
            )
        else:
            st.metric(
                label="BOS Status",
                value="None",
                help="No recent break"
            )

    with col5:
        if choch['choch_detected']:
            st.metric(
                label="CHoCH Warning",
                value=choch['warning_level'],
                help=choch['description']
            )
        else:
            st.metric(
                label="CHoCH Status",
                value="None",
                help="No early warning"
            )

    # Macro Context Bar
    if not np.isnan(macro_ctx.get('crisis_score', np.nan)):
        st.markdown("---")
        st.markdown("**📊 Macro Context Overlay**")
        col_macro1, col_macro2, col_macro3, col_macro4 = st.columns(4)

        with col_macro1:
            crisis_score = macro_ctx.get('crisis_score', 0)
            st.metric(
                "Crisis Score",
                f"{crisis_score:.0f}/4",
                help="From Macro Dashboard"
            )

        with col_macro2:
            vix = macro_ctx.get('vix', np.nan)
            if not np.isnan(vix):
                st.metric("VIX", f"{vix:.1f}", help=macro_ctx.get('volatility_regime', 'Unknown'))

        with col_macro3:
            liq_score = macro_ctx.get('liquidity_score', np.nan)
            if not np.isnan(liq_score):
                st.metric("Liquidity", f"{liq_score:.1f}", help=macro_ctx.get('liquidity_regime', 'Unknown'))

        with col_macro4:
            st.caption(f"**Regime:** {macro_ctx.get('volatility_regime', 'Unknown')}")
            st.caption(f"**Liquidity:** {macro_ctx.get('liquidity_regime', 'Unknown')}")

    # Proximity Alerts
    if proximity.get('has_alerts'):
        st.warning(f"⚠️ **Price Position:** {proximity['position']}")
        for alert in proximity['alerts'][:3]:
            st.caption(f"• {alert['level']} ({alert['price']:.2f}) - {alert['distance_pct']:.2f}% away [{alert['urgency']}]")

    # Section 2: Multi-Timeframe Analysis
    st.markdown("---")
    st.subheader("🔄 Multi-Timeframe Confirmation")

    col_mtf1, col_mtf2, col_mtf3, col_mtf4 = st.columns(4)

    with col_mtf1:
        daily_emoji = "📈" if "Bullish" in mtf_analysis['daily']['trend'] else "📉" if "Bearish" in mtf_analysis['daily']['trend'] else "➡️"
        st.markdown(f"**{daily_emoji} Daily**")
        st.caption(mtf_analysis['daily']['trend'])
        st.caption(f"*{mtf_analysis['daily']['structure']}*")

    with col_mtf2:
        weekly_emoji = "📈" if "Bullish" in mtf_analysis['weekly']['trend'] else "📉" if "Bearish" in mtf_analysis['weekly']['trend'] else "➡️"
        st.markdown(f"**{weekly_emoji} Weekly**")
        st.caption(mtf_analysis['weekly']['trend'])
        st.caption(f"*{mtf_analysis['weekly']['structure']}*")

    with col_mtf3:
        monthly_emoji = "📈" if "Bullish" in mtf_analysis['monthly']['trend'] else "📉" if "Bearish" in mtf_analysis['monthly']['trend'] else "➡️"
        st.markdown(f"**{monthly_emoji} Monthly**")
        st.caption(mtf_analysis['monthly']['trend'])
        st.caption(f"*{mtf_analysis['monthly']['structure']}*")

    with col_mtf4:
        st.markdown("**Confluence**")
        st.caption(mtf_analysis['alignment'])
        if mtf_analysis.get('bullish_count', 0) >= 2:
            st.success("✅ Strong bullish alignment")
        elif mtf_analysis.get('bearish_count', 0) >= 2:
            st.error("⚠️ Strong bearish alignment")

    # Section 3: Risk/Reward Analysis
    st.markdown("---")
    st.subheader("💰 Risk/Reward Analysis")

    if rr_analysis:
        st.markdown(f"**Suggested Stop Loss:** {rr_analysis[0]['stop_loss']:.2f} (based on nearest support)")

        for rr in rr_analysis[:3]:
            col_rr1, col_rr2, col_rr3, col_rr4 = st.columns([1, 1, 1, 1])

            with col_rr1:
                st.write(f"**Target {rr['target_level']}:**")

            with col_rr2:
                st.write(f"{rr['target_price']:.2f} (+{rr['reward_pct']:.2f}%)")

            with col_rr3:
                st.write(f"Risk: -{rr['risk_pct']:.2f}%")

            with col_rr4:
                rr_ratio = rr['rr_ratio']
                if rr_ratio >= 2:
                    st.success(f"R:R = {rr_ratio:.2f}:1 ✅")
                elif rr_ratio >= 1:
                    st.warning(f"R:R = {rr_ratio:.2f}:1 ⚠️")
                else:
                    st.error(f"R:R = {rr_ratio:.2f}:1 ❌")
    else:
        st.caption("Insufficient support levels for R:R calculation")

    # Section 4: Support & Resistance Levels
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

    # Section 5: Fibonacci Levels
    if fib_levels.get('retracements'):
        st.markdown("---")
        st.subheader("📐 Fibonacci Levels")

        current_price = levels.get('current_price', 0)

        st.caption(f"Based on: High {fib_levels['swing_high']:.2f} → Low {fib_levels['swing_low']:.2f}")

        col_fib1, col_fib2 = st.columns(2)

        with col_fib1:
            st.markdown("**Retracements (Support if pullback)**")
            for fib in fib_levels['retracements'][1:5]:  # Show 0.236, 0.382, 0.5, 0.618
                price = fib['price']
                # Mark if price is near this level
                if abs(price - current_price) / current_price < 0.01:  # Within 1%
                    st.caption(f"→ {fib['label']}: **{price:.2f}** ← CURRENT AREA")
                else:
                    st.caption(f"{fib['label']}: {price:.2f}")

        with col_fib2:
            st.markdown("**Extensions (Upside targets)**")
            for fib in fib_levels['extensions'][:3]:  # Show 1.272, 1.414, 1.618
                st.caption(f"{fib['label']}: {fib['price']:.2f}")

    # Section 6: Liquidity Zones
    if liq_zones.get('zones'):
        st.markdown("---")
        st.subheader("🎯 Liquidity Zones (Stop Loss Clusters)")

        col_liq1, col_liq2 = st.columns(2)

        above_zones = [z for z in liq_zones['zones'] if z['direction'] == 'above'][:3]
        below_zones = [z for z in liq_zones['zones'] if z['direction'] == 'below'][:3]

        with col_liq1:
            st.markdown("**Above (Long Stops)**")
            for zone in above_zones:
                st.caption(f"• {zone['price']:.2f} - {zone['type']}")

        with col_liq2:
            st.markdown("**Below (Short Stops)**")
            for zone in below_zones:
                st.caption(f"• {zone['price']:.2f} - {zone['type']}")

    # Section 7: Historical Statistics
    if hist_stats:
        st.markdown("---")
        st.subheader("📈 Historical Statistics (Last 12 Months)")

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        with col_stat1:
            st.metric("ATH", f"{hist_stats['ath']:.2f}")
            st.caption(f"{hist_stats['days_from_ath']} days ago")

        with col_stat2:
            st.metric("Drawdown", f"{hist_stats['drawdown_from_ath']:.2f}%")
            st.caption("From ATH")

        with col_stat3:
            st.metric("Volatility", f"{hist_stats['annualized_volatility']:.1f}%")
            st.caption("Annualized")

        with col_stat4:
            st.metric("Avg Daily Move", f"{hist_stats['avg_daily_move']:.2f}%")
            st.caption(f"Max: {hist_stats['max_daily_gain']:.2f}%")

    # Section 8: Performance Metrics
    if performance and performance.get('bos_count', 0) > 0:
        st.markdown("---")
        st.subheader("📊 BOS Detection Performance (Last 12 Months)")

        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)

        with col_perf1:
            st.metric("BOS Detected", f"{performance['bos_count']}")

        with col_perf2:
            st.metric("Success Rate", f"{performance['success_rate']:.0f}%")
            st.caption("≥1% move after BOS")

        with col_perf3:
            st.metric("Avg Move Post-BOS", f"{performance['avg_move_after_bos']:.2f}%")

        with col_perf4:
            st.metric("Avg Days to Reversal", f"{performance['avg_days_to_reversal']:.0f}")

    # Section 9: Price Chart with Structure
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

    # Add Fibonacci levels (just 0.382 and 0.618)
    if fib_levels.get('retracements'):
        for fib in fib_levels['retracements']:
            if fib['ratio'] in [0.382, 0.618]:
                fig.add_hline(
                    y=fib['price'],
                    line_dash="dot",
                    line_color="purple",
                    opacity=0.3,
                    annotation_text=f"Fib {fib['label']}",
                    annotation_position="left"
                )

    st.plotly_chart(fig, use_container_width=True)

    # Section 10: Swing Points Detail
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
