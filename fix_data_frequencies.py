#!/usr/bin/env python3
"""
Fix Mixed Data Frequencies & Normalization

Problems:
1. Mixed frequencies (daily, weekly, monthly, quarterly)
2. Incorrect normalization (z-scores on wrong windows)
3. Look-ahead bias (using future data)

Solutions:
1. Resample all to daily frequency (forward fill)
2. Apply rolling z-scores (avoid look-ahead)
3. Handle missing data properly

Author: MacroArimax
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def check_data_frequency(df):
    """
    Analyze data frequency for each column.

    Returns dict with frequency analysis.
    """
    print("="*80)
    print("DATA FREQUENCY ANALYSIS")
    print("="*80)

    freq_analysis = []

    for col in df.columns:
        series = df[col].dropna()

        if len(series) == 0:
            continue

        # Calculate typical gap between observations
        gaps = series.index.to_series().diff().dropna()
        median_gap = gaps.median()

        # Infer frequency
        if median_gap <= pd.Timedelta(days=1):
            freq = "Daily"
        elif median_gap <= pd.Timedelta(days=7):
            freq = "Weekly"
        elif median_gap <= pd.Timedelta(days=31):
            freq = "Monthly"
        elif median_gap <= pd.Timedelta(days=92):
            freq = "Quarterly"
        else:
            freq = "Unknown"

        pct_missing = (1 - len(series) / len(df)) * 100

        freq_analysis.append({
            'Column': col,
            'Frequency': freq,
            'Non-null': len(series),
            'Missing %': pct_missing,
            'Median Gap': median_gap.days
        })

    freq_df = pd.DataFrame(freq_analysis)
    freq_df = freq_df.sort_values('Missing %', ascending=False)

    print(f"\nTotal columns: {len(df.columns)}")
    print(f"\nFrequency distribution:")
    print(freq_df.groupby('Frequency').size())

    print(f"\n\nColumns with >50% missing data:")
    high_missing = freq_df[freq_df['Missing %'] > 50]
    if len(high_missing) > 0:
        print(high_missing[['Column', 'Frequency', 'Missing %']].to_string(index=False))
        print(f"\n⚠️ Consider removing {len(high_missing)} columns with >50% missing")
    else:
        print("✅ No columns with >50% missing")

    print(f"\n\nColumns by frequency:")
    for freq in ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Unknown']:
        cols = freq_df[freq_df['Frequency'] == freq]['Column'].tolist()
        if cols:
            print(f"\n{freq} ({len(cols)}):")
            for col in cols[:10]:  # Show first 10
                pct = freq_df[freq_df['Column'] == col]['Missing %'].values[0]
                print(f"  - {col:<30} ({pct:.1f}% missing)")
            if len(cols) > 10:
                print(f"  ... and {len(cols) - 10} more")

    return freq_df


def resample_to_daily(df):
    """
    Resample all columns to daily frequency.

    Method: Forward fill (carry last observation forward)
    This is standard for financial/economic data.
    """
    print("\n" + "="*80)
    print("RESAMPLING TO DAILY FREQUENCY")
    print("="*80)

    # Already daily indexed
    if not isinstance(df.index, pd.DatetimeIndex):
        print("⚠️ Index is not DatetimeIndex, converting...")
        df.index = pd.to_datetime(df.index)

    # Create complete daily range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

    # Reindex and forward fill
    df_daily = df.reindex(full_range)
    df_daily = df_daily.fillna(method='ffill', limit=5)  # Forward fill max 5 days

    print(f"Original shape: {df.shape}")
    print(f"Daily shape: {df_daily.shape}")

    # Show improvement in data coverage
    original_missing = df.isna().sum().sum()
    daily_missing = df_daily.isna().sum().sum()

    print(f"\nMissing values:")
    print(f"  Before: {original_missing:,} ({original_missing / df.size * 100:.1f}%)")
    print(f"  After:  {daily_missing:,} ({daily_missing / df_daily.size * 100:.1f}%)")

    return df_daily


def apply_rolling_normalization(df, features, window=252):
    """
    Apply rolling z-score normalization.

    IMPORTANT: This avoids look-ahead bias by using only past data.

    Parameters
    ----------
    window : int
        Rolling window in days (252 = 1 year of trading days)
    """
    print("\n" + "="*80)
    print(f"ROLLING NORMALIZATION (window={window} days)")
    print("="*80)

    df_normalized = df.copy()

    for col in features:
        if col not in df.columns:
            continue

        # Calculate rolling mean and std
        rolling_mean = df[col].rolling(window=window, min_periods=max(30, window//4)).mean()
        rolling_std = df[col].rolling(window=window, min_periods=max(30, window//4)).std()

        # Z-score
        df_normalized[f'{col}_zscore'] = (df[col] - rolling_mean) / rolling_std

        # Replace NaN with 0 (for initial period)
        df_normalized[f'{col}_zscore'] = df_normalized[f'{col}_zscore'].fillna(0)

    print(f"✅ Created {len(features)} z-score normalized features")

    return df_normalized


def remove_high_missing_columns(df, threshold=0.5):
    """
    Remove columns with >threshold missing data.

    Default: Remove if >50% missing
    """
    missing_pct = df.isna().sum() / len(df)
    high_missing = missing_pct[missing_pct > threshold].index.tolist()

    if high_missing:
        print(f"\n🗑️ Removing {len(high_missing)} columns with >{threshold*100:.0f}% missing:")
        for col in high_missing:
            pct = missing_pct[col] * 100
            print(f"  - {col:<30} ({pct:.1f}% missing)")

        df = df.drop(columns=high_missing)

    return df


def main():
    """Main execution."""

    # Load data
    cache_file = Path('.fred_cache/full_data.pkl')

    if not cache_file.exists():
        print("❌ No cached data found. Please run train_crisis_model.py first.")
        return 1

    print("📂 Loading data...")
    df = pd.read_pickle(cache_file)
    print(f"   Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Step 1: Analyze frequencies
    freq_df = check_data_frequency(df)

    # Step 2: Remove high missing columns
    print("\n" + "="*80)
    print("CLEANING: REMOVE HIGH MISSING COLUMNS")
    print("="*80)
    df_clean = remove_high_missing_columns(df, threshold=0.5)
    print(f"\nCleaned shape: {df_clean.shape}")

    # Step 3: Resample to daily
    df_daily = resample_to_daily(df_clean)

    # Step 4: Apply normalization (optional - show example)
    print("\n" + "="*80)
    print("NORMALIZATION EXAMPLE")
    print("="*80)
    print("\nFor features like VIX, HY_OAS, etc., you can apply rolling z-scores:")
    print("  z = (x - rolling_mean) / rolling_std")
    print("\nThis should be done in prepare_features() in CrisisPredictor")
    print("to ensure no look-ahead bias.")

    # Save cleaned data
    output_file = Path('.fred_cache/full_data_daily.pkl')
    df_daily.to_pickle(output_file)
    print(f"\n✅ Saved cleaned daily data to: {output_file}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n1. Use the cleaned daily data (full_data_daily.pkl)")
    print("2. Apply rolling normalization in prepare_features()")
    print("3. Verify no look-ahead bias (use only past data)")
    print("4. Test model with cleaned data")
    print("\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
