"""
Crisis Predictor - Ensemble Version

Combines multiple models based on academic literature:
- Logistic Regression (interpretability)
- Random Forest (robustness)
- XGBoost (accuracy)

Based on Fed/ECB research showing ensemble > individual models.

Usage:
    from crisis_predictor_ensemble import EnsembleCrisisPredictor

    predictor = EnsembleCrisisPredictor()
    predictor.train(df)
    proba = predictor.predict_proba(df)

Author: MacroArimax
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Try XGBoost (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class EnsembleCrisisPredictor:
    """
    Ensemble crisis predictor combining 3 models.

    Models:
    1. Logistic Regression (30%) - interpretability
    2. Random Forest (30%) - robustness
    3. XGBoost (40%) - accuracy (if available)

    If XGBoost not available, uses Logistic (50%) + RF (50%).
    """

    def __init__(self, horizon=5, use_xgboost=True):
        """
        Initialize ensemble predictor.

        Parameters
        ----------
        horizon : int
            Days ahead to predict
        use_xgboost : bool
            Whether to use XGBoost (if available)
        """
        self.horizon = horizon
        self.use_xgboost = use_xgboost and HAS_XGBOOST

        # Features (same as simplified model)
        self.features = [
            'VIX',
            'HY_OAS',
            'cp_tbill_spread',
            'T10Y2Y',
            'NFCI'
        ]

        # Scaler for Logistic Regression
        self.scaler = StandardScaler()

        # Models
        self.logistic_model = None
        self.rf_model = None
        self.xgb_model = None
        self.ensemble = None

        self.is_trained = False
        self.feature_importance_ = None

    def create_labels(self, df):
        """
        Create crisis labels.

        Same as SimplifiedCrisisPredictor.
        """
        df = df.copy()

        crisis_conditions = (
            (df['VIX'] > 30) |
            (df.get('HY_OAS', pd.Series(0, index=df.index)) > 8.0) |
            (df.get('cp_tbill_spread', pd.Series(0, index=df.index)) > 1.0)
        )

        df['crisis_ahead'] = crisis_conditions.rolling(
            window=self.horizon,
            min_periods=1
        ).max().shift(-self.horizon).fillna(0).astype(int)

        return df

    def prepare_features(self, df):
        """Get available features."""
        available = [f for f in self.features if f in df.columns]

        if len(available) < len(self.features):
            missing = set(self.features) - set(available)
            print(f"⚠️ Missing features: {missing}")

        return available

    def train(self, df):
        """
        Train ensemble model.

        Parameters
        ----------
        df : pd.DataFrame
            Historical data with features
        """
        print("="*60)
        print("ENSEMBLE CRISIS PREDICTOR - TRAINING")
        print("="*60)
        print(f"\nHorizon: {self.horizon} days")
        print(f"Features: {len(self.features)}")

        if self.use_xgboost:
            print("Models: Logistic + RandomForest + XGBoost")
        else:
            print("Models: Logistic + RandomForest")
            if not HAS_XGBOOST:
                print("  (XGBoost not available - install with: pip install xgboost)")

        # Create labels
        df = self.create_labels(df)

        # Get features
        available_features = self.prepare_features(df)
        self.features = available_features  # Update to actual available

        # Clean data
        X = df[self.features].dropna()
        y = df.loc[X.index, 'crisis_ahead']

        print(f"\nTraining samples: {len(X):,}")
        print(f"Crisis samples: {y.sum():,} ({y.mean():.1%})")
        print(f"Normal samples: {(~y.astype(bool)).sum():,} ({(~y.astype(bool)).mean():.1%})")

        # 1. LOGISTIC REGRESSION
        print("\n" + "-"*60)
        print("Training Logistic Regression...")

        X_scaled = self.scaler.fit_transform(X)

        self.logistic_model = LogisticRegression(
            penalty='l1',
            C=0.1,
            solver='saga',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.logistic_model.fit(X_scaled, y)

        # Get coefficients for interpretation
        coef_df = pd.DataFrame({
            'feature': self.features,
            'coefficient': self.logistic_model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)

        print("\nLogistic Regression Coefficients:")
        for idx, row in coef_df.iterrows():
            sign = "+" if row['coefficient'] > 0 else ""
            print(f"  {row['feature']:<20} {sign}{row['coefficient']:.4f}")

        # 2. RANDOM FOREST
        print("\n" + "-"*60)
        print("Training Random Forest...")

        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)

        rf_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nRandom Forest Feature Importance:")
        for idx, row in rf_importance.iterrows():
            print(f"  {row['feature']:<20} {row['importance']:.4f}")

        # 3. XGBOOST (if available)
        if self.use_xgboost:
            print("\n" + "-"*60)
            print("Training XGBoost...")

            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=10,
                eval_metric='auc',
                random_state=42,
                n_jobs=-1
            )
            self.xgb_model.fit(X, y)

            xgb_importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nXGBoost Feature Importance:")
            for idx, row in xgb_importance.iterrows():
                print(f"  {row['feature']:<20} {row['importance']:.4f}")

        # 4. CREATE ENSEMBLE
        print("\n" + "-"*60)
        print("Creating Ensemble...")

        # Note: We can't use VotingClassifier with Logistic (needs scaling)
        # So we'll manually combine predictions
        self.is_trained = True

        # Store average importance for reporting
        if self.use_xgboost:
            self.feature_importance_ = pd.DataFrame({
                'feature': self.features,
                'importance': (
                    0.3 * rf_importance.set_index('feature')['importance'] +
                    0.4 * xgb_importance.set_index('feature')['importance']
                ) / 0.7  # Normalize (Logistic doesn't have importance)
            }).reset_index().sort_values('importance', ascending=False)
        else:
            self.feature_importance_ = rf_importance

        print("\n✅ Ensemble trained successfully!")
        print("="*60)

        return self

    def predict_proba(self, df):
        """
        Predict crisis probability using ensemble.

        Combines:
        - Logistic: 30% (or 50% if no XGBoost)
        - RandomForest: 30% (or 50% if no XGBoost)
        - XGBoost: 40% (if available)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare features
        X = df[self.features].fillna(0)

        # Get predictions from each model
        X_scaled = self.scaler.transform(X)
        logistic_proba = self.logistic_model.predict_proba(X_scaled)[:, 1]
        rf_proba = self.rf_model.predict_proba(X)[:, 1]

        if self.use_xgboost:
            xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
            # Weighted average
            ensemble_proba = (
                0.3 * logistic_proba +
                0.3 * rf_proba +
                0.4 * xgb_proba
            )
        else:
            # Simple average
            ensemble_proba = (
                0.5 * logistic_proba +
                0.5 * rf_proba
            )

        return ensemble_proba

    def predict(self, df, threshold=0.5):
        """Predict crisis (binary)."""
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)

    def get_individual_predictions(self, df):
        """
        Get predictions from each individual model.

        Useful for debugging and understanding ensemble behavior.
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        X = df[self.features].fillna(0)
        X_scaled = self.scaler.transform(X)

        predictions = {
            'logistic': self.logistic_model.predict_proba(X_scaled)[:, 1],
            'random_forest': self.rf_model.predict_proba(X)[:, 1]
        }

        if self.use_xgboost:
            predictions['xgboost'] = self.xgb_model.predict_proba(X)[:, 1]

        predictions['ensemble'] = self.predict_proba(df)

        return pd.DataFrame(predictions, index=df.index)

    def explain_prediction(self, df, date=None):
        """Explain prediction by showing each model's contribution."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        if date is None:
            date = df.index[-1]

        # Get predictions
        preds = self.get_individual_predictions(df.loc[[date]])

        print(f"\n{'='*60}")
        print(f"ENSEMBLE PREDICTION EXPLANATION - {date}")
        print(f"{'='*60}")
        print()

        print("Individual Model Predictions:")
        print(f"  Logistic Regression: {preds['logistic'].values[0]:.1%} (weight: 30%)")
        print(f"  Random Forest:       {preds['random_forest'].values[0]:.1%} (weight: 30%)")

        if self.use_xgboost:
            print(f"  XGBoost:            {preds['xgboost'].values[0]:.1%} (weight: 40%)")

        print(f"\n  Ensemble (weighted): {preds['ensemble'].values[0]:.1%}")

        # Show feature values
        print(f"\nFeature Values:")
        for feat in self.features:
            val = df.loc[date, feat]
            print(f"  {feat:<20} {val:>10.2f}")


# Example usage
if __name__ == "__main__":
    print("Ensemble Crisis Predictor")
    print("="*60)
    print()
    print("Usage:")
    print("  from crisis_predictor_ensemble import EnsembleCrisisPredictor")
    print()
    print("  predictor = EnsembleCrisisPredictor()")
    print("  predictor.train(df)")
    print("  proba = predictor.predict_proba(df)")
    print()
    print("Models combined:")
    print("  - Logistic Regression (30%) - interpretability")
    print("  - Random Forest (30%) - robustness")
    print("  - XGBoost (40%) - accuracy")
