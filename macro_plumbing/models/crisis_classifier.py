"""
Crisis Classifier - Random Forest Model for Liquidity Stress Prediction

Predicts probability of liquidity crisis in next N days.
NOT predicting prices - predicting crisis probability.

Usage:
    from macro_plumbing.models import CrisisPredictor

    # Train
    predictor = CrisisPredictor(horizon=5)
    predictor.train(df)

    # Predict
    proba = predictor.predict_proba(df)
    print(f"Crisis probability: {proba[-1]:.1%}")

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
    precision_recall_curve,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class CrisisPredictor:
    """
    Random Forest classifier for predicting liquidity crises.

    Predicts probability of crisis in next N days using:
    - Spreads (CP, credit, repo)
    - Volatility (VIX)
    - Fed facilities (Discount Window)
    - Liquidity metrics (RRP, TGA, reserves)
    - Real economy (jobless claims, credit)

    Parameters
    ----------
    horizon : int
        Days ahead to predict (default 5)
    n_estimators : int
        Number of trees in forest (default 200)
    max_depth : int
        Maximum tree depth (default 10)
    class_weight : str
        How to handle imbalanced data (default 'balanced')
    random_state : int
        Random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        horizon=5,
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ):
        self.horizon = horizon
        self.random_state = random_state

        # Simpler model for 5 features (avoid overfitting)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=30,    # Increased from 20 to avoid overfitting
            min_samples_leaf=15,     # Increased from 10 to avoid overfitting
            max_features='sqrt',     # For 5 features, this is ~2 features per split
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )

        self.features = None
        self.feature_importance_ = None
        self.is_trained = False

    def create_labels(self, df):
        """
        Create binary crisis labels.

        Crisis definition (ANY of):
        - VIX > 30 (panic level - volatility spike)
        - CP spread > 1.0% (severe money market stress - 100+ bps)
        - HY OAS > 8.0% (credit market crisis - 800+ bps)

        CALIBRATED thresholds based on financial market norms and historical crises.
        Expected: ~5-15% of days marked as crisis during stress periods (2008, 2020).

        Note: DISCOUNT_WINDOW removed due to unclear data units causing false positives.
              Model will use it as a feature but not in the crisis label definition.

        Parameters
        ----------
        df : pd.DataFrame
            Data with FRED features

        Returns
        -------
        pd.DataFrame
            DataFrame with 'crisis_ahead' column added
        """
        df = df.copy()

        # Define crisis conditions (CALIBRATED to financial market stress levels)
        # Thresholds based on well-established market stress indicators:
        # - VIX: 30+ indicates fear/panic (normal: 10-20, elevated: 20-30, crisis: 30+)
        # - cp_tbill_spread: 1.0%+ indicates money market freeze (normal: 10-30 bps, crisis: 100+ bps)
        # - HY_OAS: 8.0%+ indicates credit crisis (normal: 4-5%, elevated: 6-7%, crisis: 8%+)
        #
        # Note: Data is in decimal format (e.g., 1.0 = 1%, not 100 bps)
        crisis_conditions = (
            (df['VIX'] > 30) |
            (df.get('cp_tbill_spread', pd.Series(0, index=df.index)) > 1.0) |
            (df.get('HY_OAS', pd.Series(0, index=df.index)) > 8.0)
        )

        # Look ahead N days (is there a crisis in next N days?)
        df['crisis_ahead'] = crisis_conditions.rolling(
            window=self.horizon,
            min_periods=1
        ).max().shift(-self.horizon).fillna(0).astype(int)

        return df

    def prepare_features(self, df):
        """
        Prepare feature set for model.

        ULTRA-SIMPLIFIED to eliminate multicollinearity:
        - Only 5 core INDEPENDENT features (VIF < 5)
        - NO lags (cause multicollinearity)
        - NO derived features (cause multicollinearity)
        - Based on academic literature for crisis prediction

        Features selected:
        1. VIX - Market volatility (equity stress)
        2. HY_OAS - Credit spread (corporate stress)
        3. cp_tbill_spread - Money market spread (funding stress)
        4. T10Y2Y - Term spread (recession signal)
        5. NFCI - Composite financial conditions (Fed index)

        These 5 features:
        - Cover different dimensions of financial stress
        - Have low correlation (VIF analysis)
        - Are available in real-time
        - Are validated in academic literature

        Parameters
        ----------
        df : pd.DataFrame
            Data with FRED features

        Returns
        -------
        list
            List of feature column names
        """
        # MINIMAL FEATURE SET - 5 independent features only
        core_features = []

        # 1. Volatility (market fear)
        if 'VIX' in df.columns:
            core_features.append('VIX')

        # 2. Credit stress (HY OAS only - remove bbb_aaa_spread due to multicollinearity)
        if 'HY_OAS' in df.columns:
            core_features.append('HY_OAS')

        # 3. Money market stress (CP spread - independent, VIF=2.43)
        if 'cp_tbill_spread' in df.columns:
            core_features.append('cp_tbill_spread')

        # 4. Term structure (recession signal - independent, VIF=2.60)
        if 'T10Y2Y' in df.columns:
            core_features.append('T10Y2Y')

        # 5. Composite stress index (NFCI - moderate VIF=8.37 but valuable composite)
        if 'NFCI' in df.columns:
            core_features.append('NFCI')

        # REMOVED due to severe multicollinearity (VIF > 10):
        # - DISCOUNT_WINDOW (VIF=15.63, unclear data units)
        # - bbb_aaa_spread (VIF=152.82, redundant with HY_OAS)
        # - VIX_lag1, HY_OAS_lag1 (causes multicollinearity)
        # - VIX_volatility (causes multicollinearity)
        # - delta_rrp (not consistently significant)
        # - jobless_claims_zscore (labor, different frequency)

        return core_features

    def train(self, df):
        """
        Train Random Forest model on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Historical data with FRED features

        Returns
        -------
        self
            Trained model instance
        """
        print(f"Training Crisis Predictor (horizon={self.horizon} days)...")
        print("="*60)

        # Create labels
        df = self.create_labels(df)

        # Prepare features
        self.features = self.prepare_features(df)

        print(f"Features selected: {len(self.features)}")
        print(f"Available features: {', '.join(self.features[:10])}...")

        # Remove rows with NaN (from lags/rolling)
        X = df[self.features].dropna()
        y = df.loc[X.index, 'crisis_ahead']

        print(f"\nTraining samples: {len(X)}")
        print(f"Crisis samples: {y.sum()} ({y.mean():.1%})")
        print(f"Normal samples: {(~y.astype(bool)).sum()} ({(~y.astype(bool)).mean():.1%})")

        # Train model
        print("\nTraining Random Forest...")
        self.model.fit(X, y)
        self.is_trained = True

        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n" + "="*60)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("="*60)
        for idx, row in self.feature_importance_.head(10).iterrows():
            print(f"{row['feature']:30s} {row['importance']:.4f}")

        # In-sample performance (just for reference)
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        print("\n" + "="*60)
        print("IN-SAMPLE PERFORMANCE (Reference Only)")
        print("="*60)
        print(f"AUC: {roc_auc_score(y, y_proba):.3f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['No Crisis', 'Crisis']))

        print("\n⚠️  Use backtest() for out-of-sample performance ⚠️")
        print("="*60)

        return self

    def predict_proba(self, df):
        """
        Predict crisis probability.

        Parameters
        ----------
        df : pd.DataFrame
            Data with FRED features

        Returns
        -------
        np.ndarray
            Crisis probabilities (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Prepare features (creates lag/volatility columns if they don't exist)
        # This is necessary because prepare_features() modifies df in-place
        available_features = self.prepare_features(df)

        # Check which features are actually available
        missing_features = [f for f in self.features if f not in df.columns]

        if missing_features:
            # Use only available features
            usable_features = [f for f in self.features if f in df.columns]

            if len(usable_features) < 5:
                raise ValueError(
                    f"Too many missing features ({len(missing_features)}/{len(self.features)}). "
                    f"Missing: {missing_features[:10]}... "
                    "Retrain the model with current data."
                )

            # Fill missing features with zeros
            for feat in missing_features:
                df[feat] = 0.0

        # Now extract features
        X = df[self.features]
        proba = self.model.predict_proba(X)[:, 1]

        return proba

    def predict(self, df, threshold=0.5):
        """
        Predict crisis (binary).

        Parameters
        ----------
        df : pd.DataFrame
            Data with FRED features
        threshold : float
            Probability threshold for classification (default 0.5)

        Returns
        -------
        np.ndarray
            Binary predictions (0 = no crisis, 1 = crisis)
        """
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)

    def backtest(self, df, n_splits=5):
        """
        Time-series cross-validation backtesting.

        CRITICAL: Uses walk-forward validation (no future data leakage).

        Parameters
        ----------
        df : pd.DataFrame
            Data with FRED features
        n_splits : int
            Number of time-series folds (default 5)

        Returns
        -------
        list
            Backtest results for each fold
        """
        print("\n" + "="*60)
        print(f"BACKTESTING - {n_splits}-FOLD TIME SERIES CROSS-VALIDATION")
        print("="*60)

        # Create labels and features
        df = self.create_labels(df)
        features = self.prepare_features(df)

        X = df[features].dropna()
        y = df.loc[X.index, 'crisis_ahead']

        # Time series split (preserves temporal order)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{n_splits}")
            print(f"{'='*60}")

            # Train on past, test on future
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(f"Train: {X_train.index[0]} to {X_train.index[-1]} ({len(X_train)} samples)")
            print(f"Test:  {X_test.index[0]} to {X_test.index[-1]} ({len(X_test)} samples)")
            print(f"Test crisis rate: {y_test.mean():.1%}")

            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc = roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0

            print(f"\nAUC: {auc:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Crisis', 'Crisis']))

            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(f"                Predicted")
            print(f"              No Crisis  Crisis")
            print(f"Actual  No     {cm[0,0]:6d}  {cm[0,1]:6d}")
            print(f"        Yes    {cm[1,0]:6d}  {cm[1,1]:6d}")

            results.append({
                'fold': fold + 1,
                'auc': auc,
                'train_start': X_train.index[0],
                'train_end': X_train.index[-1],
                'test_start': X_test.index[0],
                'test_end': X_test.index[-1],
                'test_crisis_rate': y_test.mean(),
                'predictions': y_proba,
                'actual': y_test.values
            })

        # Overall performance
        avg_auc = np.mean([r['auc'] for r in results])

        print(f"\n{'='*60}")
        print(f"OVERALL BACKTEST PERFORMANCE")
        print(f"{'='*60}")
        print(f"Average AUC: {avg_auc:.3f}")

        # Format AUC scores
        auc_scores = [f"{r['auc']:.3f}" for r in results]
        print(f"AUC by fold: {auc_scores}")

        if avg_auc > 0.80:
            print("✅ EXCELLENT - Model has strong predictive power")
        elif avg_auc > 0.70:
            print("✅ GOOD - Model is useful for crisis detection")
        elif avg_auc > 0.60:
            print("⚠️  MODERATE - Model has some predictive value")
        else:
            print("❌ POOR - Model may not be reliable")

        return results

    def get_feature_importance(self, top_n=20):
        """
        Get feature importance ranking.

        Parameters
        ----------
        top_n : int
            Number of top features to return (default 20)

        Returns
        -------
        pd.DataFrame
            Feature importance sorted by importance
        """
        if self.feature_importance_ is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.feature_importance_.head(top_n)

    def explain_prediction(self, df, date=None):
        """
        Explain why model predicts crisis.

        Shows which features are contributing most to prediction.

        Parameters
        ----------
        df : pd.DataFrame
            Data with FRED features
        date : str or pd.Timestamp, optional
            Specific date to explain (default: last date)

        Returns
        -------
        pd.DataFrame
            Feature contributions to prediction
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if date is None:
            date = df.index[-1]

        # Get feature values for this date
        feature_values = df.loc[date, self.features]

        # Get prediction
        proba = self.predict_proba(df.loc[[date]])[0]

        print(f"\n{'='*60}")
        print(f"PREDICTION EXPLANATION - {date}")
        print(f"{'='*60}")
        print(f"Crisis Probability: {proba:.1%}")

        # Show top contributing features (by importance * value)
        importance_df = self.feature_importance_.copy()
        importance_df['value'] = importance_df['feature'].map(feature_values)
        importance_df['contribution'] = importance_df['importance'] * importance_df['value'].abs()
        importance_df = importance_df.sort_values('contribution', ascending=False)

        print("\nTop 10 Contributing Features:")
        print(f"{'Feature':<30} {'Value':>10} {'Importance':>12} {'Contribution':>12}")
        print("-"*66)

        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:<30} {row['value']:>10.2f} {row['importance']:>12.4f} {row['contribution']:>12.4f}")

        return importance_df.head(10)


# Example usage
if __name__ == "__main__":
    # This would normally load your actual data
    print("CrisisPredictor - Example Usage")
    print("="*60)
    print()
    print("from macro_plumbing.models import CrisisPredictor")
    print("from macro_plumbing.data import FREDClient")
    print()
    print("# Load data")
    print("fred = FREDClient(api_key=os.getenv('FRED_API_KEY'))")
    print("df = fred.fetch_all(start_date='2015-01-01')")
    print("df = fred.compute_derived_features(df)")
    print()
    print("# Train model")
    print("predictor = CrisisPredictor(horizon=5)")
    print("predictor.train(df.loc[:'2022-12-31'])")
    print()
    print("# Backtest")
    print("results = predictor.backtest(df)")
    print()
    print("# Predict current probability")
    print("current_proba = predictor.predict_proba(df.iloc[[-1]])[0]")
    print(f"print(f'Crisis probability (next 5 days): {current_proba:.1%}')")
    print()
    print("# Feature importance")
    print("importance = predictor.get_feature_importance(top_n=10)")
    print("print(importance)")
