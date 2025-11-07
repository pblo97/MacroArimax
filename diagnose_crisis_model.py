#!/usr/bin/env python3
"""
Diagnostic script to understand why Crisis Predictor shows 99% probability.

Checks:
1. How many days are labeled as "crisis" in training data
2. Which features have extreme values
3. Crisis definition thresholds vs actual data ranges
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from macro_plumbing.data.fred_client import FREDClient
import os

# Check API key
api_key = os.getenv('FRED_API_KEY')
if not api_key:
    print("ERROR: Set FRED_API_KEY environment variable")
    sys.exit(1)

print("="*80)
print("CRISIS PREDICTOR DIAGNOSTIC")
print("="*80)
print()

# Load data
print("Loading data...")
fred = FREDClient(api_key=api_key)
df = fred.fetch_all(start_date='2010-01-01')
df = fred.compute_derived_features(df)

# Remove labor_slack if exists
if 'labor_slack' in df.columns:
    df = df.drop(columns=['labor_slack'])
    print("✅ Removed labor_slack")

print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print()

# Check crisis definition
print("="*80)
print("CRISIS DEFINITION CHECK")
print("="*80)
print()

print("Crisis conditions (ANY of these = crisis):")
print("  1. VIX > 35")
print("  2. CP spread > 150bp")
print("  3. HY OAS > 700bp")
print("  4. Discount Window > $10B")
print()

# Check actual data ranges
features_to_check = {
    'VIX': (35, 'VIX > 35'),
    'cp_tbill_spread': (150, 'CP spread > 150bp'),
    'HY_OAS': (700, 'HY OAS > 700bp'),
    'DISCOUNT_WINDOW': (10000, 'Discount Window > $10B')
}

print("ACTUAL DATA RANGES:")
print()

for feat, (threshold, description) in features_to_check.items():
    if feat in df.columns:
        series = df[feat].dropna()

        if len(series) == 0:
            print(f"❌ {feat}: NO DATA")
            continue

        min_val = series.min()
        max_val = series.max()
        mean_val = series.mean()
        median_val = series.median()
        current_val = series.iloc[-1]

        # Count how many days exceed threshold
        exceeds = (series > threshold).sum()
        pct_exceeds = exceeds / len(series) * 100

        print(f"{feat}:")
        print(f"  Range: {min_val:.2f} - {max_val:.2f}")
        print(f"  Mean: {mean_val:.2f}, Median: {median_val:.2f}")
        print(f"  Current: {current_val:.2f}")
        print(f"  Threshold: {threshold}")
        print(f"  Days exceeding threshold: {exceeds} ({pct_exceeds:.1f}%)")

        if pct_exceeds > 50:
            print(f"  ⚠️  WARNING: >50% of days exceed threshold!")
        elif pct_exceeds > 20:
            print(f"  ⚠️  WARNING: >20% of days exceed threshold")

        print()
    else:
        print(f"❌ {feat}: NOT IN DATAFRAME")
        print()

# Create crisis labels using same logic as model
print("="*80)
print("CRISIS LABELS ANALYSIS")
print("="*80)
print()

crisis_conditions = (
    (df.get('VIX', pd.Series(0, index=df.index)) > 35) |
    (df.get('cp_tbill_spread', pd.Series(0, index=df.index)) > 150) |
    (df.get('HY_OAS', pd.Series(0, index=df.index)) > 700) |
    (df.get('DISCOUNT_WINDOW', pd.Series(0, index=df.index)) > 10000)
)

# Look ahead 5 days (same as model)
crisis_ahead = crisis_conditions.rolling(window=5, min_periods=1).max().shift(-5).fillna(0).astype(int)

crisis_count = crisis_ahead.sum()
total_count = len(crisis_ahead)
crisis_pct = crisis_count / total_count * 100

print(f"Total days: {total_count}")
print(f"Days labeled as 'crisis': {crisis_count} ({crisis_pct:.1f}%)")
print(f"Days labeled as 'normal': {total_count - crisis_count} ({100-crisis_pct:.1f}%)")
print()

if crisis_pct > 50:
    print("🚨 PROBLEM FOUND: >50% of days labeled as 'crisis'!")
    print("   This causes model to predict crisis almost always")
    print()
elif crisis_pct > 30:
    print("⚠️  WARNING: >30% of days labeled as 'crisis'")
    print("   This may cause high false positive rate")
    print()
else:
    print("✅ Crisis rate looks reasonable (<30%)")
    print()

# Show when crises occurred
print("="*80)
print("CRISIS PERIODS (sample)")
print("="*80)
print()

crisis_dates = df.index[crisis_conditions]
if len(crisis_dates) > 0:
    print(f"Total crisis days: {len(crisis_dates)}")
    print()
    print("First 10 crisis dates:")
    for date in crisis_dates[:10]:
        vix = df.loc[date, 'VIX'] if 'VIX' in df.columns else np.nan
        hy_oas = df.loc[date, 'HY_OAS'] if 'HY_OAS' in df.columns else np.nan
        cp_spread = df.loc[date, 'cp_tbill_spread'] if 'cp_tbill_spread' in df.columns else np.nan
        dw = df.loc[date, 'DISCOUNT_WINDOW'] if 'DISCOUNT_WINDOW' in df.columns else np.nan

        print(f"{date.strftime('%Y-%m-%d')}: VIX={vix:.1f}, HY_OAS={hy_oas:.1f}, CP={cp_spread:.1f}, DW={dw:.0f}")

    print()
    print("Last 10 crisis dates:")
    for date in crisis_dates[-10:]:
        vix = df.loc[date, 'VIX'] if 'VIX' in df.columns else np.nan
        hy_oas = df.loc[date, 'HY_OAS'] if 'HY_OAS' in df.columns else np.nan
        cp_spread = df.loc[date, 'cp_tbill_spread'] if 'cp_tbill_spread' in df.columns else np.nan
        dw = df.loc[date, 'DISCOUNT_WINDOW'] if 'DISCOUNT_WINDOW' in df.columns else np.nan

        print(f"{date.strftime('%Y-%m-%d')}: VIX={vix:.1f}, HY_OAS={hy_oas:.1f}, CP={cp_spread:.1f}, DW={dw:.0f}")
else:
    print("❌ No crisis days found!")
    print("   This means thresholds are too high")

print()
print("="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
