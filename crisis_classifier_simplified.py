"""
Crisis Classifier - SIMPLIFIED VERSION

Based on academic research:
- Adrian et al. (2019): Fed Financial Stability Monitoring
- Giglio et al. (2016): Systemic Risk and the Macroeconomy
- Hatzius et al. (2010): Financial Conditions Indexes
- Lo Duca et al. (2017): ECB Financial Conditions Index

Key principle: Use 5-7 INDEPENDENT indicators that capture different
aspects of financial stress.

Features selected based on:
1. Academic literature consensus
2. Low multicollinearity (VIF < 5)
3. High predictive power (mutual information)
4. Available in real-time

Author: MacroArimax
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class SimplifiedCrisisPredictor:
    """
    Simplified crisis predictor using only 5-7 core independent features.

    Features (based on academic literature):
    1. VIX - Market volatility (equity market stress)
    2. HY_OAS - High yield credit spread (credit market stress)
    3. cp_tbill_spread - Money market spread (funding stress)
    4. T10Y2Y - Term spread (recession/inversion signal)
    5. NFCI - Financial conditions composite (Fed research)

    Optional:
    6. delta_rrp - Change in Fed RRP (liquidity drain)
    7. jobless_claims_zscore - Labor market stress

    These 5-7 features are:
    - Theoretically motivated (different stress dimensions)
    - Empirically validated (crisis literature)
    - Low correlation (VIF < 5)
    - Available daily/weekly

    Parameters
    ----------
    horizon : int
        Days ahead to predict (default 5)
    features : list, optional
        Custom feature list (default uses recommended 5)
    """

    def __init__(
        self,
        horizon=5,
        features=None,
        n_estimators=100,
        max_depth=6,
        random_state=42
    ):
        self.horizon = horizon
        self.random_state = random_state

        # Default: 5 core features (minimal set)
        if features is None:
            self.features = [
                'VIX',                  # Volatility
                'HY_OAS',              # Credit risk
                'cp_tbill_spread',     # Money market stress
                'T10Y2Y',              # Term structure
                'NFCI'                 # Composite financial conditions
            ]
        else:
            self.features = features

        # Simpler model to avoid overfitting
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

        self.feature_importance_ = None
        self.is_trained = False

    def create_labels(self, df):
        """
        Create crisis labels using market-based thresholds.

        Crisis = ANY of:
        - VIX > 30 (panic)
        - HY OAS > 8.0% (credit crisis)
        - CP spread > 1.0% (money market freeze)

        These thresholds are based on historical crises (2008, 2020).
        """
        df = df.copy()

        # Crisis conditions
        crisis_conditions = (
            (df['VIX'] > 30) |
            (df.get('HY_OAS', pd.Series(0, index=df.index)) > 8.0) |
            (df.get('cp_tbill_spread', pd.Series(0, index=df.index)) > 1.0)
        )

        # Look ahead
        df['crisis_ahead'] = crisis_conditions.rolling(
            window=self.horizon,
            min_periods=1
        ).max().shift(-self.horizon).fillna(0).astype(int)

        return df

    def prepare_features(self, df):
        """
        Prepare features - NO DERIVED FEATURES.

        Use raw features only to minimize multicollinearity.
        Normalization (if needed) should be done before calling this.
        """
        # Just verify features exist
        available = [f for f in self.features if f in df.columns]

        if len(available) < len(self.features):
            missing = set(self.features) - set(available)
            print(f"⚠️ Missing features: {missing}")
            print(f"Using {len(available)}/{len(self.features)} features")

        return available

    def train(self, df):
        """Train model on historical data."""
        print(f"Training Simplified Crisis Predictor...")
        print(f"Horizon: {self.horizon} days")
        print(f"Features: {len(self.features)}")
        print("="*60)

        # Create labels
        df = self.create_labels(df)

        # Get available features
        available_features = self.prepare_features(df)

        if len(available_features) == 0:
            raise ValueError("No features available!")

        # Clean data
        X = df[available_features].dropna()
        y = df.loc[X.index, 'crisis_ahead']

        print(f"\nFeatures used: {', '.join(available_features)}")
        print(f"\nTraining samples: {len(X):,}")
        print(f"Crisis samples: {y.sum():,} ({y.mean():.1%})")
        print(f"Normal samples: {(~y.astype(bool)).sum():,} ({(~y.astype(bool)).mean():.1%})")

        # Check for reasonable crisis rate
        if y.mean() > 0.5:
            print("\n⚠️ WARNING: >50% marked as crisis!")
            print("   Crisis thresholds may be too low.")
        elif y.mean() < 0.01:
            print("\n⚠️ WARNING: <1% marked as crisis!")
            print("   Crisis thresholds may be too high.")
        else:
            print("\n✅ Crisis rate looks reasonable (1-50%)")

        # Train
        print("\nTraining Random Forest...")
        self.model.fit(X, y)
        self.is_trained = True
        self.features = available_features  # Update to actual features used

        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        for idx, row in self.feature_importance_.iterrows():
            print(f"{row['feature']:30s} {row['importance']:.4f}")

        # In-sample metrics
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        print("\n" + "="*60)
        print("IN-SAMPLE PERFORMANCE")
        print("="*60)
        print(f"AUC: {roc_auc_score(y, y_proba):.3f}")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(f"                Predicted")
        print(f"              Normal  Crisis")
        print(f"Actual Normal {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Crisis {cm[1,0]:6d}  {cm[1,1]:6d}")

        return self

    def predict_proba(self, df):
        """Predict crisis probability."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Get features
        X = df[self.features]

        # Handle missing
        X = X.fillna(0)

        proba = self.model.predict_proba(X)[:, 1]
        return proba

    def predict(self, df, threshold=0.5):
        """Predict crisis (binary)."""
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)

    def backtest(self, df, n_splits=5):
        """Time-series cross-validation."""
        print("\n" + "="*60)
        print(f"BACKTESTING - {n_splits} FOLDS")
        print("="*60)

        # Prepare
        df = self.create_labels(df)
        X = df[self.features].dropna()
        y = df.loc[X.index, 'crisis_ahead']

        # CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(f"\nFold {fold + 1}/{n_splits}")
            print(f"Train: {X_train.index[0]} to {X_train.index[-1]}")
            print(f"Test:  {X_test.index[0]} to {X_test.index[-1]}")

            # Train
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=30,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Predict
            y_proba = model.predict_proba(X_test)[:, 1]

            if y_test.sum() > 0:
                auc = roc_auc_score(y_test, y_proba)
                print(f"AUC: {auc:.3f}")
                results.append(auc)
            else:
                print("No crisis samples in test set")

        # Summary
        avg_auc = np.mean(results) if results else 0.0
        print(f"\n{'='*60}")
        print(f"AVERAGE AUC: {avg_auc:.3f}")

        if avg_auc > 0.75:
            print("✅ GOOD - Model has useful predictive power")
        elif avg_auc > 0.60:
            print("⚠️ MODERATE - Model has some predictive value")
        else:
            print("❌ POOR - Model may not be reliable")

        return results

    def explain_prediction(self, df, date=None):
        """Explain current prediction."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        if date is None:
            date = df.index[-1]

        print(f"\n{'='*60}")
        print(f"PREDICTION EXPLANATION - {date}")
        print(f"{'='*60}")

        # Get values
        values = df.loc[date, self.features]
        proba = self.predict_proba(df.loc[[date]])[0]

        print(f"\nCrisis Probability: {proba:.1%}\n")

        print(f"{'Feature':<30} {'Value':>10} {'Importance':>12}")
        print("-"*54)

        for idx, row in self.feature_importance_.iterrows():
            feat = row['feature']
            imp = row['importance']
            val = values[feat]
            print(f"{feat:<30} {val:>10.2f} {imp:>12.4f}")


# Example usage
if __name__ == "__main__":
    print("Simplified Crisis Predictor")
    print("="*60)
    print("\nUsage:")
    print("  from crisis_classifier_simplified import SimplifiedCrisisPredictor")
    print("  ")
    print("  predictor = SimplifiedCrisisPredictor()")
    print("  predictor.train(df)")
    print("  proba = predictor.predict_proba(df)")
    print("\nFeatures used:")
    print("  1. VIX - Volatility")
    print("  2. HY_OAS - Credit spread")
    print("  3. cp_tbill_spread - Money market spread")
    print("  4. T10Y2Y - Term spread")
    print("  5. NFCI - Financial conditions")
