"""
Check current VIF scores for the 5 features used in Logistic Regression model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Try to load cached data
    cache_file = Path("data/fred_cache.pkl")

    if cache_file.exists():
        print("Loading cached FRED data...")
        df = pd.read_pickle(cache_file)
        print(f"Loaded {len(df)} observations")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    else:
        print("No cached data found. Generating synthetic data...")
        # Create synthetic data for testing
        np.random.seed(42)
        dates = pd.date_range('2015-01-01', '2025-01-01', freq='D')
        n = len(dates)

        df = pd.DataFrame({
            'VIX': np.random.gamma(3, 5, n) + 10,
            'HY_OAS': np.random.gamma(2, 2, n) + 2,
            'cp_tbill_spread': np.random.gamma(1.5, 0.2, n),
            'T10Y2Y': np.random.normal(1.0, 0.8, n),
            'NFCI': np.random.normal(-0.2, 0.5, n)
        }, index=dates)

    # The 5 features currently in the model
    features = ['VIX', 'HY_OAS', 'cp_tbill_spread', 'T10Y2Y', 'NFCI']

    # Check which features are available
    available_features = [f for f in features if f in df.columns]

    if len(available_features) < 2:
        print(f"\nERROR: Only {len(available_features)} features found in data")
        print(f"Available: {available_features}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"VIF ANALYSIS - CHECKING MULTICOLLINEARITY")
    print(f"{'='*70}")
    print(f"\nFeatures being analyzed: {', '.join(available_features)}")
    print(f"Observations: {len(df):,}")

    # Prepare data
    df_features = df[available_features].replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Observations after dropping NaN/Inf: {len(df_features):,}")

    if len(df_features) < 100:
        print("\n⚠️  WARNING: Very few observations available for VIF calculation")

    # Calculate VIF for each feature
    print(f"\n{'='*70}")
    print(f"{'Feature':<20} {'VIF':>10} {'Status':>15} {'Interpretation'}")
    print(f"{'='*70}")

    vif_results = []

    for i, col in enumerate(df_features.columns):
        try:
            vif = variance_inflation_factor(df_features.values, i)

            if vif > 10:
                status = "🔴 SEVERE"
                interpretation = "REMOVE - severe multicollinearity"
            elif vif > 5:
                status = "🟡 MODERATE"
                interpretation = "WARNING - consider removing"
            else:
                status = "✅ GOOD"
                interpretation = "Independent"

            vif_results.append({
                'feature': col,
                'vif': vif,
                'status': status,
                'interpretation': interpretation
            })

            print(f"{col:<20} {vif:>10.2f} {status:>15} {interpretation}")

        except Exception as e:
            print(f"{col:<20} {'ERROR':>10} {'⚠️':>15} {str(e)[:40]}")

    print(f"{'='*70}")

    # Summary
    vif_df = pd.DataFrame(vif_results)

    severe = vif_df[vif_df['vif'] > 10]
    moderate = vif_df[(vif_df['vif'] > 5) & (vif_df['vif'] <= 10)]
    good = vif_df[vif_df['vif'] <= 5]

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Good (VIF ≤ 5):     {len(good)} features")
    print(f"🟡 Moderate (5 < VIF ≤ 10): {len(moderate)} features")
    print(f"🔴 Severe (VIF > 10):  {len(severe)} features")

    if len(severe) > 0:
        print(f"\n⚠️  SEVERE MULTICOLLINEARITY DETECTED")
        print(f"\nFeatures to REMOVE:")
        for idx, row in severe.iterrows():
            print(f"  - {row['feature']} (VIF={row['vif']:.2f})")
        print(f"\nRecommended features to keep:")
        for idx, row in good.iterrows():
            print(f"  ✅ {row['feature']} (VIF={row['vif']:.2f})")
        if len(moderate) > 0:
            print(f"\nModerate multicollinearity (borderline):")
            for idx, row in moderate.iterrows():
                print(f"  🟡 {row['feature']} (VIF={row['vif']:.2f})")
    else:
        print(f"\n✅ NO SEVERE MULTICOLLINEARITY DETECTED")
        if len(moderate) > 0:
            print(f"\n🟡 Moderate multicollinearity in:")
            for idx, row in moderate.iterrows():
                print(f"  - {row['feature']} (VIF={row['vif']:.2f})")
        else:
            print(f"\n✅ All features are independent (VIF ≤ 5)")

    # Correlation matrix
    print(f"\n{'='*70}")
    print(f"CORRELATION MATRIX (Pearson)")
    print(f"{'='*70}")
    corr_matrix = df_features.corr()
    print(corr_matrix.round(3))

    print(f"\n{'='*70}")
    print(f"HIGH CORRELATIONS (|r| > 0.7)")
    print(f"{'='*70}")
    high_corr_found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r_val = corr_matrix.iloc[i, j]
            if abs(r_val) > 0.7:
                high_corr_found = True
                status = "⚠️ SEVERE" if abs(r_val) > 0.9 else "🟡 MODERATE"
                print(f"{corr_matrix.columns[i]:20s} ↔ {corr_matrix.columns[j]:20s} r={r_val:>6.3f} {status}")

    if not high_corr_found:
        print("✅ No high correlations detected (all |r| ≤ 0.7)")

except ImportError as e:
    print(f"ERROR: {e}")
    print("\nPlease install statsmodels:")
    print("  pip install statsmodels")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
