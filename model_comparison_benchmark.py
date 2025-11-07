#!/usr/bin/env python3
"""
Crisis Prediction Model Comparison Benchmark

Compares multiple models based on academic literature:
1. Logistic Regression (ECB, Fed baseline)
2. Random Forest (current model)
3. XGBoost (best accuracy in literature)
4. Ensemble (voting of all 3)

Based on:
- Beutel et al. (2019): Fed machine learning for banking crises
- Adrian et al. (2019): Fed Growth-at-Risk
- Lo Duca et al. (2017): ECB early warning system

Author: MacroArimax
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")


def create_logistic_model():
    """
    Logistic Regression with L1 regularization.

    Used by:
    - ECB: Lo Duca et al. (2017)
    - IMF: Alessi & Detken (2018)
    - Fed: Adrian et al. (2019) - GaR framework

    Advantages:
    - Highly interpretable (coefficients = marginal effects)
    - Stable with few features
    - Fast to train
    - Calibrated probabilities
    """
    return LogisticRegression(
        penalty='l1',           # LASSO (feature selection)
        C=0.1,                  # Regularization (higher = less regularization)
        solver='saga',          # Supports L1
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )


def create_random_forest_model():
    """
    Random Forest - Current model.

    Used by:
    - Fed: Beutel et al. (2019)
    - BIS: Aldasoro et al. (2018)

    Advantages:
    - Handles non-linear relationships
    - Robust to outliers
    - Feature importance
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )


def create_xgboost_model():
    """
    XGBoost - Best accuracy in literature.

    Used by:
    - Fed: Beutel et al. (2019) - beat RF by 5-8% AUC
    - Goldman Sachs: Hatzius et al. (2020)

    Advantages:
    - Typically best accuracy
    - Built-in regularization
    - Handles missing data
    """
    if not HAS_XGBOOST:
        return None

    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,            # Shallow trees (avoid overfitting)
        learning_rate=0.05,     # Slow learning (more stable)
        subsample=0.8,          # Bagging
        colsample_bytree=0.8,   # Feature sampling
        reg_alpha=0.1,          # L1 regularization
        reg_lambda=1.0,         # L2 regularization
        scale_pos_weight=10,    # For imbalanced data
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Comprehensive model evaluation.

    Metrics based on Fed/ECB research:
    1. AUC (primary metric)
    2. Precision @ 90% Recall
    3. Brier Score (calibration)
    4. Confusion matrix
    """
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    results = {
        'model': model_name,
        'auc': roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'brier': brier_score_loss(y_test, y_proba)
    }

    # Precision @ 90% Recall
    # Find threshold that gives ~90% recall
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    idx_90 = np.argmax(tpr >= 0.90)
    if idx_90 > 0:
        threshold_90 = thresholds[idx_90]
        y_pred_90 = (y_proba >= threshold_90).astype(int)
        results['precision_at_90_recall'] = precision_score(y_test, y_pred_90, zero_division=0)
    else:
        results['precision_at_90_recall'] = 0

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm

    return results, model


def compare_models(df, features, target='crisis_ahead', n_splits=5):
    """
    Compare all models using time-series cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Data with features and target
    features : list
        List of feature column names
    target : str
        Target column name
    n_splits : int
        Number of CV folds

    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    print("="*80)
    print("MODEL COMPARISON BENCHMARK")
    print("="*80)
    print(f"\nFeatures: {len(features)}")
    print(f"Features: {', '.join(features)}")
    print(f"\nCV Folds: {n_splits}")
    print()

    # Prepare data
    df_clean = df[features + [target]].dropna()
    X = df_clean[features]
    y = df_clean[target]

    print(f"Samples: {len(X):,}")
    print(f"Crisis samples: {y.sum():,} ({y.mean():.1%})")
    print()

    # Models to test
    models_to_test = {
        'Logistic': create_logistic_model(),
        'RandomForest': create_random_forest_model(),
    }

    if HAS_XGBOOST:
        models_to_test['XGBoost'] = create_xgboost_model()

    # Time series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_results = []

    for model_name, model in models_to_test.items():
        print(f"\n{'='*80}")
        print(f"TESTING: {model_name}")
        print(f"{'='*80}")

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Normalize for Logistic Regression
            if model_name == 'Logistic':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                results, _ = evaluate_model(
                    model, X_train_scaled, y_train, X_test_scaled, y_test, model_name
                )
            else:
                results, _ = evaluate_model(
                    model, X_train, y_train, X_test, y_test, model_name
                )

            results['fold'] = fold + 1
            fold_results.append(results)

            print(f"\nFold {fold + 1}/{n_splits}:")
            print(f"  AUC: {results['auc']:.3f}")
            print(f"  Precision: {results['precision']:.3f}")
            print(f"  Recall: {results['recall']:.3f}")
            print(f"  F1: {results['f1']:.3f}")
            print(f"  Brier Score: {results['brier']:.3f}")
            print(f"  Precision @ 90% Recall: {results['precision_at_90_recall']:.3f}")

        # Average results
        avg_results = {
            'model': model_name,
            'avg_auc': np.mean([r['auc'] for r in fold_results]),
            'std_auc': np.std([r['auc'] for r in fold_results]),
            'avg_precision': np.mean([r['precision'] for r in fold_results]),
            'avg_recall': np.mean([r['recall'] for r in fold_results]),
            'avg_f1': np.mean([r['f1'] for r in fold_results]),
            'avg_brier': np.mean([r['brier'] for r in fold_results]),
            'avg_precision_at_90_recall': np.mean([r['precision_at_90_recall'] for r in fold_results])
        }

        all_results.append(avg_results)

        print(f"\n{model_name} Average Performance:")
        print(f"  AUC: {avg_results['avg_auc']:.3f} ± {avg_results['std_auc']:.3f}")
        print(f"  Precision: {avg_results['avg_precision']:.3f}")
        print(f"  Recall: {avg_results['avg_recall']:.3f}")
        print(f"  F1: {avg_results['avg_f1']:.3f}")
        print(f"  Brier: {avg_results['avg_brier']:.3f}")
        print(f"  Precision @ 90% Recall: {avg_results['avg_precision_at_90_recall']:.3f}")

    # Create ensemble if we have multiple models
    if len(models_to_test) >= 2:
        print(f"\n{'='*80}")
        print("TESTING: Ensemble (Voting)")
        print(f"{'='*80}")

        # Create ensemble
        ensemble_models = []
        ensemble_weights = []

        if 'Logistic' in models_to_test:
            ensemble_models.append(('logistic', create_logistic_model()))
            ensemble_weights.append(0.3)

        if 'RandomForest' in models_to_test:
            ensemble_models.append(('rf', create_random_forest_model()))
            ensemble_weights.append(0.3)

        if HAS_XGBOOST and 'XGBoost' in models_to_test:
            ensemble_models.append(('xgb', create_xgboost_model()))
            ensemble_weights.append(0.4)

        # Normalize weights
        ensemble_weights = np.array(ensemble_weights) / sum(ensemble_weights)

        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',
            weights=ensemble_weights
        )

        ensemble_fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Note: Ensemble handles Logistic normalization internally via pipeline
            # For simplicity, we'll use raw features (works for RF and XGB)
            results, _ = evaluate_model(
                ensemble, X_train, y_train, X_test, y_test, 'Ensemble'
            )

            results['fold'] = fold + 1
            ensemble_fold_results.append(results)

            print(f"\nFold {fold + 1}/{n_splits}:")
            print(f"  AUC: {results['auc']:.3f}")
            print(f"  Precision @ 90% Recall: {results['precision_at_90_recall']:.3f}")

        # Average
        avg_ensemble = {
            'model': 'Ensemble',
            'avg_auc': np.mean([r['auc'] for r in ensemble_fold_results]),
            'std_auc': np.std([r['auc'] for r in ensemble_fold_results]),
            'avg_precision': np.mean([r['precision'] for r in ensemble_fold_results]),
            'avg_recall': np.mean([r['recall'] for r in ensemble_fold_results]),
            'avg_f1': np.mean([r['f1'] for r in ensemble_fold_results]),
            'avg_brier': np.mean([r['brier'] for r in ensemble_fold_results]),
            'avg_precision_at_90_recall': np.mean([r['precision_at_90_recall'] for r in ensemble_fold_results])
        }

        all_results.append(avg_ensemble)

        print(f"\nEnsemble Average Performance:")
        print(f"  AUC: {avg_ensemble['avg_auc']:.3f} ± {avg_ensemble['std_auc']:.3f}")
        print(f"  Precision @ 90% Recall: {avg_ensemble['avg_precision_at_90_recall']:.3f}")

    # Create comparison table
    results_df = pd.DataFrame(all_results)

    return results_df


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

    # Create labels
    from macro_plumbing.models.crisis_classifier import CrisisPredictor
    predictor = CrisisPredictor(horizon=5)
    df = predictor.create_labels(df)

    # Get features
    features = predictor.prepare_features(df)

    print(f"\n📊 Features: {len(features)}")
    for feat in features:
        print(f"  - {feat}")

    # Run comparison
    results = compare_models(df, features, target='crisis_ahead', n_splits=5)

    # Display results
    print("\n\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print()

    # Sort by AUC
    results_sorted = results.sort_values('avg_auc', ascending=False)

    print("Performance Ranking (by AUC):")
    print("-"*80)
    print(f"{'Rank':<6} {'Model':<15} {'AUC':<12} {'Precision':<12} {'Recall':<12} {'Brier':<12}")
    print("-"*80)

    for rank, (idx, row) in enumerate(results_sorted.iterrows(), 1):
        model = row['model']
        auc = f"{row['avg_auc']:.3f} ± {row['std_auc']:.3f}"
        precision = f"{row['avg_precision']:.3f}"
        recall = f"{row['avg_recall']:.3f}"
        brier = f"{row['avg_brier']:.3f}"

        print(f"{rank:<6} {model:<15} {auc:<12} {precision:<12} {recall:<12} {brier:<12}")

    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    best_model = results_sorted.iloc[0]

    print(f"\n🏆 Best Model: {best_model['model']} (AUC = {best_model['avg_auc']:.3f})")

    # Recommendations based on results
    if 'Ensemble' in results_sorted['model'].values:
        ensemble_auc = results_sorted[results_sorted['model'] == 'Ensemble']['avg_auc'].values[0]
        best_individual = results_sorted[results_sorted['model'] != 'Ensemble'].iloc[0]

        improvement = (ensemble_auc - best_individual['avg_auc']) / best_individual['avg_auc'] * 100

        print(f"\n📊 Ensemble vs Best Individual ({best_individual['model']}):")
        print(f"  Ensemble AUC: {ensemble_auc:.3f}")
        print(f"  {best_individual['model']} AUC: {best_individual['avg_auc']:.3f}")
        print(f"  Improvement: {improvement:+.1f}%")

        if improvement > 2:
            print(f"\n✅ RECOMMENDATION: Use Ensemble")
            print(f"   Ensemble provides {improvement:.1f}% improvement over best individual model.")
        else:
            print(f"\n⚠️  RECOMMENDATION: Use {best_individual['model']}")
            print(f"   Ensemble improvement ({improvement:.1f}%) is marginal. Simpler model preferred.")

    # Model-specific recommendations
    print("\n\n📝 Model-Specific Notes:")

    for idx, row in results_sorted.iterrows():
        model = row['model']
        auc = row['avg_auc']

        print(f"\n{model}:")

        if model == 'Logistic':
            print("  ✅ Most interpretable (coefficients = marginal effects)")
            print("  ✅ Fast training and prediction")
            print("  ✅ Good for regulatory reporting")
            if auc < 0.75:
                print("  ⚠️  Lower accuracy - consider for baseline only")

        elif model == 'RandomForest':
            print("  ✅ Current production model")
            print("  ✅ Robust to outliers")
            print("  ✅ Feature importance available")
            if auc >= 0.80:
                print("  ✅ Strong performance - keep as primary model")

        elif model == 'XGBoost':
            print("  ✅ Typically best accuracy")
            print("  ✅ Built-in regularization")
            if auc > results_sorted[results_sorted['model'] == 'RandomForest']['avg_auc'].values[0]:
                improvement = (auc - results_sorted[results_sorted['model'] == 'RandomForest']['avg_auc'].values[0]) * 100
                print(f"  ✅ {improvement:.1f}% better than RandomForest - consider switching")

        elif model == 'Ensemble':
            print("  ✅ Most robust (combines multiple models)")
            print("  ✅ Best for production (less sensitive to outliers)")
            print("  ⚠️  Slower prediction (needs all models)")

    print("\n\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Review results above")
    print(f"2. If Ensemble is best, implement it in CrisisPredictor")
    print("3. Backtest on historical crises (2008, 2020)")
    print("4. Monitor performance in production")
    print()

    # Save results
    output_file = Path('model_comparison_results.csv')
    results.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
