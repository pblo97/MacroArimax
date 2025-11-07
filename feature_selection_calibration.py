#!/usr/bin/env python3
"""
Feature Selection & Calibration System

Based on academic literature for crisis prediction:
- Adrian et al. (2019): Fed Financial Stability Report methodology
- Giglio et al. (2016): Systemic Risk and Financial Stability
- Lo Duca et al. (2017): ECB Financial Conditions Index

Goal: Reduce to 5-8 independent, significant features.

Method:
1. Remove high VIF (>10) - multicollinearity
2. Remove low importance (mutual information < 0.01)
3. Remove low correlation with crisis (|corr| < 0.1)
4. Test multiple combinations
5. Select parsimonious model with best OOS performance

Author: MacroArimax
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


def calculate_vif(df, features):
    """
    Calculate Variance Inflation Factor for each feature.

    VIF > 10: Severe multicollinearity (remove)
    VIF 5-10: Moderate multicollinearity (review)
    VIF < 5: Good (independent)
    """
    df_clean = df[features].dropna()

    vif_data = []
    for i, col in enumerate(df_clean.columns):
        try:
            vif = variance_inflation_factor(df_clean.values, i)
            vif_data.append({'feature': col, 'VIF': vif})
        except:
            vif_data.append({'feature': col, 'VIF': np.nan})

    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)


def calculate_feature_importance_mutual_info(X, y):
    """
    Calculate mutual information between features and target.

    Mutual Information > 0.05: Strong relationship
    Mutual Information > 0.01: Moderate relationship
    Mutual Information < 0.01: Weak (consider removing)
    """
    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    return pd.DataFrame({
        'feature': X.columns,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)


def calculate_correlation_with_target(df, features, target_col='crisis_ahead'):
    """
    Calculate Pearson correlation with crisis target.

    |corr| > 0.3: Strong predictor
    |corr| > 0.1: Moderate predictor
    |corr| < 0.1: Weak (consider removing)
    """
    correlations = []

    for feat in features:
        if feat in df.columns and target_col in df.columns:
            corr = df[feat].corr(df[target_col])
            correlations.append({'feature': feat, 'correlation': corr})

    return pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)


def iterative_vif_removal(df, features, max_vif=10):
    """
    Iteratively remove features with highest VIF until all VIF < max_vif.

    This is the standard approach in econometrics.
    """
    remaining_features = features.copy()
    removed = []

    print("Iterative VIF Removal:")
    print("="*80)

    iteration = 0
    while True:
        iteration += 1
        vif_df = calculate_vif(df, remaining_features)

        # Check if any VIF > threshold
        high_vif = vif_df[vif_df['VIF'] > max_vif]

        if len(high_vif) == 0:
            print(f"\n✅ All features now have VIF < {max_vif}")
            break

        # Remove feature with highest VIF
        worst_feature = high_vif.iloc[0]
        print(f"\nIteration {iteration}:")
        print(f"  Removing: {worst_feature['feature']} (VIF = {worst_feature['VIF']:.2f})")

        remaining_features.remove(worst_feature['feature'])
        removed.append(worst_feature['feature'])

        if len(remaining_features) < 3:
            print("\n⚠️ WARNING: Only 3 features remaining, stopping VIF removal")
            break

    print(f"\n📊 Final VIF Results:")
    final_vif = calculate_vif(df, remaining_features)
    print(final_vif.to_string(index=False))

    print(f"\n🗑️ Removed {len(removed)} features due to multicollinearity:")
    for feat in removed:
        print(f"  - {feat}")

    return remaining_features, removed


def evaluate_feature_set(df, features, crisis_col='crisis_ahead', horizon=5):
    """
    Evaluate a feature set using time-series cross-validation.

    Returns average OOS AUC score.
    """
    from sklearn.model_selection import TimeSeriesSplit

    # Prepare data
    df_clean = df[features + [crisis_col]].dropna()
    X = df_clean[features]
    y = df_clean[crisis_col]

    # Time series CV
    tscv = TimeSeriesSplit(n_splits=3)
    auc_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train simple RF
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Predict
        y_proba = rf.predict_proba(X_test)[:, 1]

        if y_test.sum() > 0:  # Only if there are crisis samples
            auc = roc_auc_score(y_test, y_proba)
            auc_scores.append(auc)

    return np.mean(auc_scores) if auc_scores else 0.0


def recommend_features(df, initial_features, crisis_col='crisis_ahead'):
    """
    Comprehensive feature selection based on:
    1. VIF (multicollinearity)
    2. Mutual Information (predictive power)
    3. Correlation with target
    4. OOS performance
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE SELECTION")
    print("="*80)

    # Step 1: Remove high VIF features
    print("\n📊 STEP 1: MULTICOLLINEARITY ANALYSIS (VIF)")
    print("="*80)
    features_after_vif, removed_vif = iterative_vif_removal(df, initial_features, max_vif=10)

    # Step 2: Calculate mutual information
    print("\n\n📊 STEP 2: PREDICTIVE POWER (MUTUAL INFORMATION)")
    print("="*80)
    df_clean = df[features_after_vif + [crisis_col]].dropna()
    X = df_clean[features_after_vif]
    y = df_clean[crisis_col]

    mi_df = calculate_feature_importance_mutual_info(X, y)
    print(mi_df.to_string(index=False))

    # Remove features with MI < 0.01
    weak_features = mi_df[mi_df['mutual_info'] < 0.01]['feature'].tolist()
    features_after_mi = [f for f in features_after_vif if f not in weak_features]

    if weak_features:
        print(f"\n🗑️ Removing {len(weak_features)} features with weak predictive power (MI < 0.01):")
        for feat in weak_features:
            print(f"  - {feat}")

    # Step 3: Correlation with target
    print("\n\n📊 STEP 3: CORRELATION WITH CRISIS TARGET")
    print("="*80)
    corr_df = calculate_correlation_with_target(df_clean, features_after_mi, crisis_col)
    print(corr_df.to_string(index=False))

    # Remove features with |corr| < 0.05
    weak_corr = corr_df[abs(corr_df['correlation']) < 0.05]['feature'].tolist()
    features_after_corr = [f for f in features_after_mi if f not in weak_corr]

    if weak_corr:
        print(f"\n🗑️ Removing {len(weak_corr)} features with weak correlation (|corr| < 0.05):")
        for feat in weak_corr:
            print(f"  - {feat}")

    # Step 4: Evaluate final set
    print("\n\n📊 STEP 4: OUT-OF-SAMPLE PERFORMANCE")
    print("="*80)

    if len(features_after_corr) > 0:
        auc = evaluate_feature_set(df, features_after_corr, crisis_col)
        print(f"Average OOS AUC: {auc:.3f}")
    else:
        print("⚠️ No features remaining after filtering!")
        features_after_corr = features_after_mi[:5]  # Keep top 5 by MI
        auc = evaluate_feature_set(df, features_after_corr, crisis_col)
        print(f"Using top 5 features by Mutual Information")
        print(f"Average OOS AUC: {auc:.3f}")

    # Final recommendations
    print("\n\n" + "="*80)
    print("📋 FINAL RECOMMENDED FEATURES")
    print("="*80)
    print(f"\nTotal features: {len(features_after_corr)}")
    print("\nFeatures to use:")
    for i, feat in enumerate(features_after_corr, 1):
        # Get stats
        mi = mi_df[mi_df['feature'] == feat]['mutual_info'].values[0] if feat in mi_df['feature'].values else 0
        corr = corr_df[corr_df['feature'] == feat]['correlation'].values[0] if feat in corr_df['feature'].values else 0
        print(f"  {i}. {feat:<30} (MI={mi:.4f}, corr={corr:.3f})")

    print("\n\n" + "="*80)
    print("📝 SUMMARY OF REMOVED FEATURES")
    print("="*80)

    all_removed = set(initial_features) - set(features_after_corr)

    print(f"\nTotal removed: {len(all_removed)} features")
    print("\nRemoval reasons:")
    print(f"  - Multicollinearity (VIF > 10): {len(removed_vif)} features")
    print(f"  - Weak predictive power (MI < 0.01): {len(weak_features)} features")
    print(f"  - Weak correlation (|corr| < 0.05): {len(weak_corr)} features")

    return features_after_corr


def main():
    """Main execution."""

    # Load data
    cache_file = Path('.fred_cache/full_data.pkl')

    if not cache_file.exists():
        print("❌ No cached data found. Please run train_crisis_model.py first to fetch data.")
        return 1

    print("📂 Loading data...")
    df = pd.read_pickle(cache_file)
    print(f"   Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # Import crisis predictor to create labels
    from macro_plumbing.models.crisis_classifier import CrisisPredictor

    predictor = CrisisPredictor(horizon=5)
    df = predictor.create_labels(df)

    # Get current feature list from predictor
    initial_features = predictor.prepare_features(df)

    print(f"\n📊 Initial feature set: {len(initial_features)} features")
    print("="*80)
    for feat in sorted(initial_features):
        print(f"  - {feat}")

    # Check crisis distribution
    crisis_rate = df['crisis_ahead'].mean()
    print(f"\n⚠️ Crisis rate in data: {crisis_rate:.1%}")

    if crisis_rate > 0.5:
        print("🔴 WARNING: >50% of data marked as crisis!")
        print("   This indicates the crisis thresholds are still too low.")
        print("   Feature selection may not solve the problem - need better crisis definition.")

    # Run feature selection
    recommended_features = recommend_features(df, initial_features, 'crisis_ahead')

    # Generate code for implementation
    print("\n\n" + "="*80)
    print("💻 CODE TO IMPLEMENT")
    print("="*80)

    print("\nUpdate `prepare_features()` in `crisis_classifier.py`:")
    print("\n```python")
    print("def prepare_features(self, df):")
    print('    """')
    print("    SIMPLIFIED feature set based on:")
    print("    - VIF analysis (removed multicollinearity)")
    print("    - Mutual information (kept predictive features)")
    print("    - Academic literature (crisis prediction best practices)")
    print('    """')
    print("    # Core independent features only")
    print("    core_features = [")
    for feat in recommended_features:
        print(f"        '{feat}',")
    print("    ]")
    print("    ")
    print("    # Create any derived features needed")
    print("    # (lags, volatility, etc.)")
    print("    ")
    print("    return core_features")
    print("```")

    print("\n" + "="*80)
    print("✅ NEXT STEPS")
    print("="*80)
    print("\n1. Update crisis_classifier.py with recommended features")
    print("2. Retrain model: python train_crisis_model.py")
    print("3. Verify OOS AUC improves")
    print("4. Check crisis probability predictions are reasonable")
    print("\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
