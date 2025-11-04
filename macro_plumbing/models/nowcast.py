"""
nowcast.py
Nowcasting models (Logit, Quantile Regression) for risk-off probability.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import statsmodels.api as sm
import statsmodels.formula.api as smf


class RiskOffNowcaster:
    """Nowcast probability of risk-off events."""

    def __init__(self, calibrate: bool = True):
        """
        Initialize nowcaster.

        Parameters
        ----------
        calibrate : bool
            Whether to calibrate probabilities (Platt scaling)
        """
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.calibrate = calibrate
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "RiskOffNowcaster":
        """
        Fit model.

        Parameters
        ----------
        X : pd.DataFrame
            Features (liquidity indicators)
        y : pd.Series
            Binary target (1 = risk-off, 0 = calm)

        Returns
        -------
        self
        """
        # Prepare data
        data = pd.concat([y, X], axis=1).dropna()
        y_data = data[y.name]
        X_data = data.drop(columns=y.name)

        # Scale
        X_scaled = self.scaler.fit_transform(X_data)

        # Fit
        self.model.fit(X_scaled, y_data)

        # Calibrate
        if self.calibrate:
            self.calibrated_model = CalibratedClassifierCV(self.model, method="sigmoid", cv=5)
            self.calibrated_model.fit(X_scaled, y_data)

        self.fitted = True
        self.feature_names = X_data.columns.tolist()

        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict probability of risk-off.

        Returns
        -------
        pd.Series
            Probability (0-1)
        """
        if not self.fitted:
            raise ValueError("Model not fitted")

        X_clean = X[self.feature_names].dropna()
        X_scaled = self.scaler.transform(X_clean)

        if self.calibrate and self.calibrated_model is not None:
            probs = self.calibrated_model.predict_proba(X_scaled)[:, 1]
        else:
            probs = self.model.predict_proba(X_scaled)[:, 1]

        return pd.Series(probs, index=X_clean.index, name="prob_risk_off")

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances (coefficients)."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        coefs = self.model.coef_[0]
        return pd.Series(coefs, index=self.feature_names, name="importance").sort_values(ascending=False)


class QuantileNowcaster:
    """Quantile regression for market outcome forecasting."""

    def __init__(self, quantile: float = 0.10):
        """
        Initialize quantile regressor.

        Parameters
        ----------
        quantile : float
            Target quantile (e.g., 0.10 for lower tail)
        """
        self.quantile = quantile
        self.model = None
        self.result = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "QuantileNowcaster":
        """
        Fit quantile regression.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target (e.g., forward returns)

        Returns
        -------
        self
        """
        data = pd.concat([y, X], axis=1).dropna()
        y_data = data[y.name]
        X_data = sm.add_constant(data.drop(columns=y.name))

        # Fit quantile regression
        self.model = sm.QuantReg(y_data, X_data)
        self.result = self.model.fit(q=self.quantile)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict quantile values."""
        if self.result is None:
            raise ValueError("Model not fitted")

        X_const = sm.add_constant(X, has_constant="add")
        predictions = self.result.predict(X_const)

        return pd.Series(predictions, index=X.index, name=f"quantile_{self.quantile}")


def create_risk_off_target(
    returns: pd.Series,
    threshold: float = -0.02,
    horizon: int = 5,
) -> pd.Series:
    """
    Create binary risk-off target.

    Parameters
    ----------
    returns : pd.Series
        Daily returns
    threshold : float
        Return threshold for risk-off (e.g., -2%)
    horizon : int
        Forward looking horizon (days)

    Returns
    -------
    pd.Series
        Binary target (1 = risk-off ahead, 0 = calm)
    """
    # Forward returns
    fwd_returns = returns.rolling(horizon).sum().shift(-horizon)

    # Binary target
    target = (fwd_returns < threshold).astype(int)
    target.name = "risk_off_target"

    return target


def nowcast_risk_off(
    X: pd.DataFrame,
    returns: pd.Series,
    threshold: float = -0.02,
    horizon: int = 5,
    test_size: float = 0.3,
) -> Tuple[pd.Series, RiskOffNowcaster]:
    """
    Convenience function for risk-off nowcasting.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    returns : pd.Series
        Market returns
    threshold : float
        Risk-off threshold
    horizon : int
        Forecast horizon
    test_size : float
        Test set proportion

    Returns
    -------
    tuple
        (probabilities, model)
    """
    # Create target
    y = create_risk_off_target(returns, threshold, horizon)

    # Train/test split
    data = pd.concat([y, X], axis=1).dropna()
    n = len(data)
    split = int(n * (1 - test_size))

    train = data.iloc[:split]
    test = data.iloc[split:]

    # Fit model
    model = RiskOffNowcaster()
    model.fit(train.drop(columns=y.name), train[y.name])

    # Predict on all data
    probs = model.predict_proba(X)

    return probs, model


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Features
    X = pd.DataFrame({
        "liquidity": np.cumsum(np.random.randn(len(dates))) * 0.1,
        "spread": np.abs(np.random.randn(len(dates))) * 2,
        "volatility": np.abs(np.random.randn(len(dates))) * 5,
    }, index=dates)

    # Returns (correlated with liquidity)
    returns = -X["liquidity"] * 0.01 + np.random.randn(len(dates)) * 0.015
    returns = pd.Series(returns, index=dates, name="returns")

    # Nowcast
    print("Nowcasting risk-off...")
    probs, model = nowcast_risk_off(X, returns, horizon=5)

    print(f"\nRisk-off probability (last 5):")
    print(probs.tail())

    print(f"\nFeature importance:")
    print(model.get_feature_importance())
