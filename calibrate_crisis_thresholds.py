#!/usr/bin/env python3
"""
Calibrate Crisis Detection Thresholds

Analyzes historical data to determine appropriate thresholds for crisis detection.
Shows distribution of key indicators and recommends calibrated thresholds.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from macro_plumbing.data.fred_client import FREDClient


def analyze_crisis_indicators(df):
    """
    Analyze distribution of crisis indicators and suggest thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data with all features

    Returns
    -------
    dict
        Analysis results with current values, stats, and suggested thresholds
    """
    indicators = {
        'VIX': {
            'current_threshold': 35,
            'description': 'Volatility Index (market fear)',
            'unit': 'index points'
        },
        'cp_tbill_spread': {
            'current_threshold': 150,
            'description': 'Commercial Paper - T-Bill Spread',
            'unit': 'basis points (expected)'
        },
        'HY_OAS': {
            'current_threshold': 700,
            'description': 'High Yield Option-Adjusted Spread',
            'unit': 'basis points (expected)'
        },
        'DISCOUNT_WINDOW': {
            'current_threshold': 10000,
            'description': 'Fed Discount Window Borrowing',
            'unit': 'millions $ (expected)'
        }
    }

    results = {}

    print("\n" + "="*80)
    print("CRISIS INDICATOR CALIBRATION ANALYSIS")
    print("="*80)

    for indicator, info in indicators.items():
        if indicator not in df.columns:
            print(f"\n⚠️  {indicator} not found in data")
            continue

        series = df[indicator].dropna()

        if len(series) == 0:
            print(f"\n⚠️  {indicator} has no data")
            continue

        # Calculate statistics
        stats = {
            'count': len(series),
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'p50': series.quantile(0.50),
            'p75': series.quantile(0.75),
            'p90': series.quantile(0.90),
            'p95': series.quantile(0.95),
            'p99': series.quantile(0.99),
            'current_value': series.iloc[-1],
            'current_threshold': info['current_threshold']
        }

        # Check how many days exceed current threshold
        days_above_threshold = (series > info['current_threshold']).sum()
        pct_above_threshold = (days_above_threshold / len(series)) * 100

        stats['days_above_threshold'] = days_above_threshold
        stats['pct_above_threshold'] = pct_above_threshold

        # Suggest calibrated thresholds based on percentiles
        # Crisis = top 5% of historical distribution
        # Stress = top 10%
        # Elevated = top 25%
        stats['suggested_crisis_threshold'] = series.quantile(0.95)
        stats['suggested_stress_threshold'] = series.quantile(0.90)
        stats['suggested_elevated_threshold'] = series.quantile(0.75)

        results[indicator] = stats

        # Print analysis
        print(f"\n{indicator} - {info['description']}")
        print(f"  Unit: {info['unit']}")
        print(f"  Data points: {stats['count']:,}")
        print(f"\n  Distribution:")
        print(f"    Min:     {stats['min']:>12.2f}")
        print(f"    25th %:  {series.quantile(0.25):>12.2f}")
        print(f"    Median:  {stats['median']:>12.2f}")
        print(f"    75th %:  {stats['p75']:>12.2f}")
        print(f"    90th %:  {stats['p90']:>12.2f}")
        print(f"    95th %:  {stats['p95']:>12.2f}")
        print(f"    99th %:  {stats['p99']:>12.2f}")
        print(f"    Max:     {stats['max']:>12.2f}")
        print(f"\n  Current Threshold: {info['current_threshold']:>12.2f}")
        print(f"    Days above:  {days_above_threshold:,} ({pct_above_threshold:.1f}%)")

        if pct_above_threshold > 20:
            print(f"    ⚠️  PROBLEM: {pct_above_threshold:.1f}% of days exceed threshold!")
            print(f"    This will mark most days as 'crisis'")

        print(f"\n  Current Value: {stats['current_value']:>12.2f}")

        print(f"\n  Suggested Thresholds (percentile-based):")
        print(f"    Elevated (75th %): {stats['suggested_elevated_threshold']:>12.2f}")
        print(f"    Stress   (90th %): {stats['suggested_stress_threshold']:>12.2f}")
        print(f"    Crisis   (95th %): {stats['suggested_crisis_threshold']:>12.2f}")

    return results


def analyze_crisis_definition(df):
    """
    Analyze current crisis definition to see why so many days are marked as crisis.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data
    """
    print("\n" + "="*80)
    print("CURRENT CRISIS DEFINITION ANALYSIS")
    print("="*80)

    # Current thresholds
    vix_threshold = 35
    cp_threshold = 150
    hy_threshold = 700
    dw_threshold = 10000

    # Check each condition
    conditions = {}

    if 'VIX' in df.columns:
        conditions['VIX > 35'] = df['VIX'] > vix_threshold

    if 'cp_tbill_spread' in df.columns:
        conditions['cp_tbill_spread > 150'] = df['cp_tbill_spread'] > cp_threshold

    if 'HY_OAS' in df.columns:
        conditions['HY_OAS > 700'] = df['HY_OAS'] > hy_threshold

    if 'DISCOUNT_WINDOW' in df.columns:
        conditions['DISCOUNT_WINDOW > 10000'] = df['DISCOUNT_WINDOW'] > dw_threshold

    # Combine with OR (any condition triggers crisis)
    crisis_flags = pd.DataFrame(conditions)
    any_crisis = crisis_flags.any(axis=1)

    total_days = len(df)
    crisis_days = any_crisis.sum()
    crisis_pct = (crisis_days / total_days) * 100

    print(f"\nTotal days analyzed: {total_days:,}")
    print(f"Days marked as crisis (ANY condition): {crisis_days:,} ({crisis_pct:.1f}%)")

    print(f"\nBreakdown by condition:")
    for condition_name, condition_series in conditions.items():
        days_true = condition_series.sum()
        pct_true = (days_true / total_days) * 100
        print(f"  {condition_name:<30} {days_true:>6,} days ({pct_true:>5.1f}%)")

    print(f"\n{'Condition':<30} {'Status':<15} {'Issue'}")
    print("-" * 80)
    for condition_name, condition_series in conditions.items():
        days_true = condition_series.sum()
        pct_true = (days_true / total_days) * 100

        if pct_true > 50:
            status = "🔴 CRITICAL"
            issue = f"{pct_true:.1f}% of days exceed threshold!"
        elif pct_true > 20:
            status = "🟡 WARNING"
            issue = f"{pct_true:.1f}% of days exceed threshold"
        elif pct_true > 10:
            status = "🟢 OK (high)"
            issue = f"{pct_true:.1f}% is expected for stress indicator"
        else:
            status = "✅ GOOD"
            issue = f"{pct_true:.1f}% represents rare events"

        print(f"{condition_name:<30} {status:<15} {issue}")


def suggest_calibrated_thresholds(results):
    """
    Suggest calibrated thresholds based on analysis.

    Parameters
    ----------
    results : dict
        Analysis results from analyze_crisis_indicators
    """
    print("\n" + "="*80)
    print("RECOMMENDED CRISIS DEFINITION")
    print("="*80)

    print("\nBased on historical distribution (95th percentile for crisis):")
    print("\n```python")
    print("# Calibrated Crisis Definition")
    print("crisis_conditions = (")

    conditions = []
    for indicator, stats in results.items():
        threshold = stats['suggested_crisis_threshold']
        conditions.append(f"    (df['{indicator}'] > {threshold:.2f})")

    print(" |\n".join(conditions))
    print(")")
    print("```")

    print("\nExpected impact:")
    print("  - ~5% of days will be marked as 'crisis' (historical 95th percentile)")
    print("  - Model will learn what true crisis looks like")
    print("  - Crisis probability should be ~5-20% in normal conditions")
    print("  - Crisis probability should spike to >70% during actual crises (2008, 2020)")


def main():
    """Main execution."""
    print("\n🔍 Loading data from FRED...")

    # Get API key
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        print("Error: FRED_API_KEY environment variable not set")
        return 1

    # Load data
    client = FREDClient(api_key=api_key)

    # Try to load from cache first
    cache_path = Path("data/raw_series.csv")
    if cache_path.exists():
        print(f"  Loading from cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print("  Fetching from FRED API...")
        df = client.fetch_all_series()
        df.to_csv(cache_path)

    # Compute derived features
    print("  Computing derived features...")
    df = client.compute_derived_features(df)

    print(f"✅ Loaded {len(df):,} days of data with {len(df.columns)} features")

    # Run analysis
    results = analyze_crisis_indicators(df)
    analyze_crisis_definition(df)
    suggest_calibrated_thresholds(results)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Update crisis_classifier.py with calibrated thresholds")
    print("2. Delete old model.pkl to force retraining")
    print("3. Verify training data shows ~5-10% crisis rate (not 99%)")
    print("4. Check crisis probability drops to ~5-20% in normal conditions")
    print("\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
