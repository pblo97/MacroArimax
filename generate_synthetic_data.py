#!/usr/bin/env python3
"""
Generate synthetic crisis data for model benchmarking.

Creates realistic financial stress data for testing models when
real FRED data is not available.

Author: MacroArimax
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_crisis_data(n_samples=3713, crisis_rate=0.10):
    """
    Generate synthetic financial data with realistic crisis patterns.

    Parameters
    ----------
    n_samples : int
        Number of days to generate
    crisis_rate : float
        Target crisis rate (~10%)

    Returns
    -------
    pd.DataFrame
        Synthetic data with 5 features and crisis labels
    """
    np.random.seed(42)

    # Date range
    dates = pd.date_range(end='2025-11-07', periods=n_samples, freq='D')

    # Initialize features
    data = pd.DataFrame(index=dates)

    # Generate features with realistic distributions
    # Normal periods vs crisis periods

    # Create crisis periods (random clusters)
    n_crisis_days = int(n_samples * crisis_rate)
    crisis_periods = np.zeros(n_samples)

    # Create 5-10 crisis clusters
    n_clusters = np.random.randint(5, 11)
    cluster_starts = np.random.choice(n_samples - 50, n_clusters, replace=False)

    for start in cluster_starts:
        length = np.random.randint(10, 50)  # Crisis lasts 10-50 days
        crisis_periods[start:start+length] = 1

    # Cap at target crisis rate
    crisis_indices = np.where(crisis_periods == 1)[0]
    if len(crisis_indices) > n_crisis_days:
        excess = len(crisis_indices) - n_crisis_days
        remove_indices = np.random.choice(crisis_indices, excess, replace=False)
        crisis_periods[remove_indices] = 0

    # 1. VIX (volatility)
    # Normal: 10-20, Crisis: 30-80
    vix_normal = np.random.normal(15, 3, n_samples)
    vix_crisis = np.random.normal(40, 15, n_samples)
    data['VIX'] = np.where(crisis_periods, vix_crisis, vix_normal)
    data['VIX'] = data['VIX'].clip(9, 85)

    # 2. HY_OAS (credit spread) - in percentage
    # Normal: 3-5%, Crisis: 8-15%
    hy_normal = np.random.normal(4, 0.5, n_samples)
    hy_crisis = np.random.normal(10, 2, n_samples)
    data['HY_OAS'] = np.where(crisis_periods, hy_crisis, hy_normal)
    data['HY_OAS'] = data['HY_OAS'].clip(2.5, 20)

    # 3. cp_tbill_spread (money market) - in percentage
    # Normal: 0.05-0.30%, Crisis: 1.0-3.0%
    cp_normal = np.random.normal(0.15, 0.08, n_samples)
    cp_crisis = np.random.normal(1.5, 0.5, n_samples)
    data['cp_tbill_spread'] = np.where(crisis_periods, cp_crisis, cp_normal)
    data['cp_tbill_spread'] = data['cp_tbill_spread'].clip(0, 5)

    # 4. T10Y2Y (yield curve)
    # Normal: 0.5-2.0, Crisis: -0.5 to 0.5 (inverted)
    t10y2y_normal = np.random.normal(1.0, 0.5, n_samples)
    t10y2y_crisis = np.random.normal(0, 0.5, n_samples)
    data['T10Y2Y'] = np.where(crisis_periods, t10y2y_crisis, t10y2y_normal)
    data['T10Y2Y'] = data['T10Y2Y'].clip(-2, 3)

    # 5. NFCI (financial conditions)
    # Normal: -0.5 to 0, Crisis: 0 to 2
    nfci_normal = np.random.normal(-0.3, 0.2, n_samples)
    nfci_crisis = np.random.normal(0.5, 0.5, n_samples)
    data['NFCI'] = np.where(crisis_periods, nfci_crisis, nfci_normal)
    data['NFCI'] = data['NFCI'].clip(-1.5, 3)

    # Add some autocorrelation (smooth time series)
    for col in data.columns:
        data[col] = data[col].rolling(5, min_periods=1).mean()

    # Add labels
    data['crisis_ahead'] = crisis_periods.astype(int)

    print(f"Generated {n_samples} days of synthetic data")
    print(f"Crisis days: {data['crisis_ahead'].sum()} ({data['crisis_ahead'].mean():.1%})")
    print(f"\nFeature ranges:")
    for col in ['VIX', 'HY_OAS', 'cp_tbill_spread', 'T10Y2Y', 'NFCI']:
        print(f"  {col:<20} {data[col].min():.2f} to {data[col].max():.2f}")

    return data


def main():
    """Generate and save synthetic data."""
    print("="*60)
    print("SYNTHETIC CRISIS DATA GENERATOR")
    print("="*60)
    print()

    # Generate data
    df = generate_synthetic_crisis_data(n_samples=3713, crisis_rate=0.087)

    # Save
    cache_dir = Path('.fred_cache')
    cache_dir.mkdir(exist_ok=True)

    output_file = cache_dir / 'full_data.pkl'
    df.to_pickle(output_file)

    print(f"\n✅ Saved to: {output_file}")
    print()
    print("Now you can run:")
    print("  python model_comparison_benchmark.py")
    print()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
