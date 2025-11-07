#!/usr/bin/env python3
"""
Verify Crisis Thresholds

Quick script to verify that crisis thresholds are working correctly.
Shows what % of historical days would be marked as crisis.

Expected: ~5-15% of days during normal times, higher during 2008/2020.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

def verify_thresholds():
    """Check if crisis thresholds produce reasonable labeling."""

    print("="*80)
    print("CRISIS THRESHOLD VERIFICATION")
    print("="*80)
    print()

    # Try to load cached data
    cache_file = Path('.fred_cache/full_data.pkl')

    if not cache_file.exists():
        print("❌ No cached data found at .fred_cache/full_data.pkl")
        print()
        print("Please run one of these first to fetch data:")
        print("  1. python train_crisis_model.py")
        print("  2. python -c \"from macro_plumbing.data import FREDClient; ...")
        print()
        return False

    print(f"✅ Loading cached data from {cache_file}...")
    df = pd.read_pickle(cache_file)

    print(f"   Loaded {len(df):,} days ({df.index[0]} to {df.index[-1]})")
    print()

    # Current thresholds (from crisis_classifier.py)
    thresholds = {
        'VIX': 30,
        'cp_tbill_spread': 1.0,
        'HY_OAS': 8.0,
        'DISCOUNT_WINDOW': 10000
    }

    print("CURRENT CRISIS THRESHOLDS:")
    print("-"*80)
    for indicator, threshold in thresholds.items():
        if indicator == 'DISCOUNT_WINDOW':
            print(f"  {indicator:<20} > {threshold:>12,.0f}  (millions USD = ${threshold/1000:.1f}B)")
        elif indicator in ['cp_tbill_spread', 'HY_OAS']:
            print(f"  {indicator:<20} > {threshold:>12.2f}%  ({threshold*100:.0f} basis points)")
        else:
            print(f"  {indicator:<20} > {threshold:>12.2f}")

    print()
    print("="*80)
    print("INDICATOR ANALYSIS")
    print("="*80)

    crisis_conditions = {}

    for indicator, threshold in thresholds.items():
        print(f"\n{indicator}")
        print("-"*80)

        if indicator not in df.columns:
            print(f"  ⚠️  NOT FOUND in data")
            continue

        series = df[indicator].dropna()

        if len(series) == 0:
            print(f"  ⚠️  No data available")
            continue

        # Statistics
        print(f"  Data points:     {len(series):>10,}")
        print(f"  Min:             {series.min():>10.2f}")
        print(f"  25th percentile: {series.quantile(0.25):>10.2f}")
        print(f"  Median:          {series.median():>10.2f}")
        print(f"  75th percentile: {series.quantile(0.75):>10.2f}")
        print(f"  95th percentile: {series.quantile(0.95):>10.2f}")
        print(f"  Max:             {series.max():>10.2f}")
        print(f"  Current (latest):{series.iloc[-1]:>10.2f}")

        # Days above threshold
        days_above = (series > threshold).sum()
        pct_above = (days_above / len(series)) * 100

        print(f"\n  Threshold:       {threshold:>10.2f}")
        print(f"  Days above:      {days_above:>10,}  ({pct_above:.1f}%)")

        # Status
        if pct_above > 50:
            print(f"  Status:          🔴 PROBLEM - threshold too low!")
        elif pct_above > 25:
            print(f"  Status:          🟡 WARNING - threshold may be low")
        elif pct_above > 3:
            print(f"  Status:          ✅ GOOD - captures stress periods")
        else:
            print(f"  Status:          🟢 OK - rare events only")

        # Store condition
        crisis_conditions[indicator] = series > threshold

    # Combined crisis definition (ANY condition)
    print()
    print("="*80)
    print("COMBINED CRISIS LABELING")
    print("="*80)
    print()

    if crisis_conditions:
        # Align all series to same index
        aligned_conditions = pd.DataFrame(crisis_conditions)
        any_crisis = aligned_conditions.any(axis=1)

        total_days = len(any_crisis)
        crisis_days = any_crisis.sum()
        crisis_pct = (crisis_days / total_days) * 100

        print(f"Total days:                 {total_days:>10,}")
        print(f"Days marked as 'crisis':    {crisis_days:>10,}  ({crisis_pct:.1f}%)")
        print(f"Days marked as 'normal':    {total_days - crisis_days:>10,}  ({100 - crisis_pct:.1f}%)")
        print()

        # Breakdown by condition
        print("Breakdown by condition:")
        for indicator, condition in aligned_conditions.items():
            days_true = condition.sum()
            pct_true = (days_true / total_days) * 100
            print(f"  {indicator:<20} {days_true:>10,} days ({pct_true:>5.1f}%)")

        print()

        # Final assessment
        if crisis_pct > 50:
            print("🔴 CRITICAL: >50% marked as crisis - thresholds too low!")
            print("   Recommendation: Increase thresholds")
            return False
        elif crisis_pct > 25:
            print("🟡 WARNING: >25% marked as crisis - thresholds may be too low")
            print("   Recommendation: Review and possibly increase thresholds")
            return False
        elif crisis_pct > 5:
            print("✅ EXCELLENT: 5-25% marked as crisis")
            print("   This is appropriate for a crisis detection system")
            print("   Model should learn to distinguish crisis from normal conditions")
            return True
        else:
            print("🟢 GOOD: <5% marked as crisis")
            print("   Thresholds capture only severe stress events")
            print("   Consider if you want to capture more moderate stress periods")
            return True
    else:
        print("❌ No crisis conditions could be evaluated")
        return False


if __name__ == "__main__":
    success = verify_thresholds()
    sys.exit(0 if success else 1)
