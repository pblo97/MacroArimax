"""
anomalies.py
Anomaly detection using IsolationForest for multivariate outliers.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Optional


class LiquidityAnomalyDetector:
    """Detect anomalies in liquidity plumbing indicators."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize detector.

        Parameters
        ----------
        contamination : float
            Expected proportion of outliers
        n_estimators : int
            Number of trees
        random_state : int
            Random seed
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.fitted = False

    def fit(self, X: pd.DataFrame) -> "LiquidityAnomalyDetector":
        """Fit detector on normal data."""
        self.model.fit(X.dropna())
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict anomalies.

        Returns
        -------
        pd.Series
            1 = normal, -1 = anomaly
        """
        if not self.fitted:
            self.fit(X)

        X_clean = X.dropna()
        predictions = self.model.predict(X_clean)
        result = pd.Series(predictions, index=X_clean.index, name="anomaly")
        return result

    def anomaly_score(self, X: pd.DataFrame) -> pd.Series:
        """
        Get anomaly scores (lower = more anomalous).

        Returns
        -------
        pd.Series
            Anomaly scores
        """
        if not self.fitted:
            self.fit(X)

        X_clean = X.dropna()
        scores = self.model.score_samples(X_clean)
        result = pd.Series(scores, index=X_clean.index, name="anomaly_score")
        return result

    def get_top_anomalies(self, X: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top N most anomalous observations."""
        scores = self.anomaly_score(X)
        top_indices = scores.nsmallest(n).index
        return X.loc[top_indices]


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.Series:
    """
    Convenience function to detect anomalies.

    Parameters
    ----------
    df : pd.DataFrame
        Input features
    contamination : float
        Expected anomaly rate

    Returns
    -------
    pd.Series
        Binary flags (1 = anomaly, 0 = normal)
    """
    detector = LiquidityAnomalyDetector(contamination=contamination)
    predictions = detector.predict(df)
    # Convert to binary (1 = anomaly, 0 = normal)
    anomalies = (predictions == -1).astype(int)
    anomalies.name = "is_anomaly"
    return anomalies


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Normal data
    df = pd.DataFrame({
        "feature1": np.random.randn(len(dates)),
        "feature2": np.random.randn(len(dates)),
        "feature3": np.random.randn(len(dates)),
    }, index=dates)

    # Inject anomalies
    anomaly_dates = dates[::200]  # Every 200 days
    for date in anomaly_dates:
        df.loc[date, :] = np.random.randn(3) * 5  # Extreme values

    # Detect
    detector = LiquidityAnomalyDetector(contamination=0.05)
    anomalies = detector.predict(df)
    scores = detector.anomaly_score(df)

    print(f"Detected {(anomalies == -1).sum()} anomalies")
    print("\nTop 5 anomalies:")
    print(detector.get_top_anomalies(df, n=5))
