"""
fusion.py
Signal fusion with Bayesian Model Averaging and calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit  # Logistic sigmoid


class SignalFusion:
    """Fuse multiple signals into a single stress probability."""

    def __init__(
        self,
        method: str = "weighted_average",
        calibration: str = "isotonic",
    ):
        """
        Initialize fusion engine.

        Parameters
        ----------
        method : str
            'weighted_average', 'bma' (Bayesian Model Averaging), 'median'
        calibration : str
            'platt', 'isotonic', None
        """
        self.method = method
        self.calibration = calibration
        self.weights = {}
        self.calibrator = None

    def fit(
        self,
        signals: pd.DataFrame,
        target: Optional[pd.Series] = None,
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> "SignalFusion":
        """
        Fit fusion model.

        Parameters
        ----------
        signals : pd.DataFrame
            Input signals (each column is a signal)
        target : pd.Series, optional
            Ground truth for weight optimization
        initial_weights : dict, optional
            Initial weights for signals

        Returns
        -------
        self
        """
        # Initialize weights
        if initial_weights is not None:
            self.weights = initial_weights
        else:
            # Equal weights
            self.weights = {col: 1.0 / len(signals.columns) for col in signals.columns}

        # Optimize weights if target provided
        if target is not None:
            self._optimize_weights(signals, target)

        return self

    def _optimize_weights(self, signals: pd.DataFrame, target: pd.Series):
        """Optimize weights based on target."""
        from sklearn.linear_model import Ridge

        # Prepare data
        data = pd.concat([target, signals], axis=1).dropna()
        y = data[target.name]
        X = data.drop(columns=target.name)

        # Use Ridge regression to get weights
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)

        # Normalize weights to sum to 1
        weights_raw = np.abs(ridge.coef_)
        weights_norm = weights_raw / weights_raw.sum()

        self.weights = {col: w for col, w in zip(X.columns, weights_norm)}

    def fuse(self, signals: pd.DataFrame) -> pd.Series:
        """
        Fuse signals into single score.

        Parameters
        ----------
        signals : pd.DataFrame
            Input signals

        Returns
        -------
        pd.Series
            Fused score (0-1 scale if calibrated)
        """
        if self.method == "weighted_average":
            # Weighted average
            fused = pd.Series(0.0, index=signals.index)
            for col in signals.columns:
                if col in self.weights:
                    fused += signals[col] * self.weights[col]

        elif self.method == "median":
            # Robust median
            fused = signals.median(axis=1)

        elif self.method == "bma":
            # Bayesian Model Averaging (simplified)
            fused = signals.mean(axis=1)  # Placeholder for full BMA

        else:
            raise ValueError(f"Unknown method: {self.method}")

        fused.name = "fused_score"
        return fused

    def calibrate(
        self,
        scores: pd.Series,
        target: pd.Series,
    ) -> pd.Series:
        """
        Calibrate scores to probabilities.

        Parameters
        ----------
        scores : pd.Series
            Raw scores
        target : pd.Series
            Binary target

        Returns
        -------
        pd.Series
            Calibrated probabilities
        """
        # Prepare data
        data = pd.concat([target, scores], axis=1).dropna()
        y = data[target.name].values
        X = data[scores.name].values.reshape(-1, 1)

        if self.calibration == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(X.ravel(), y)
            calibrated = self.calibrator.predict(X.ravel())

        elif self.calibration == "platt":
            # Platt scaling: logistic regression on scores
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression()
            lr.fit(X, y)
            calibrated = lr.predict_proba(X)[:, 1]
            self.calibrator = lr

        else:
            # No calibration
            calibrated = X.ravel()

        return pd.Series(calibrated, index=data.index, name="prob_stress_calibrated")

    def transform(self, signals: pd.DataFrame) -> pd.Series:
        """Apply fitted calibrator to new scores."""
        scores = self.fuse(signals)

        if self.calibrator is not None:
            X = scores.values.reshape(-1, 1)
            if self.calibration == "isotonic":
                calibrated = self.calibrator.predict(X.ravel())
            elif self.calibration == "platt":
                calibrated = self.calibrator.predict_proba(X)[:, 1]
            else:
                calibrated = X.ravel()

            return pd.Series(calibrated, index=scores.index, name="prob_stress")
        else:
            return scores


def create_signal_ensemble(
    signal_dict: Dict[str, pd.Series],
    target: pd.Series,
    method: str = "weighted_average",
    calibration: str = "isotonic",
) -> Tuple[pd.Series, SignalFusion]:
    """
    Create ensemble from multiple signals.

    Parameters
    ----------
    signal_dict : dict
        Dictionary of {name: signal}
    target : pd.Series
        Binary target for optimization
    method : str
        Fusion method
    calibration : str
        Calibration method

    Returns
    -------
    tuple
        (fused_probabilities, model)
    """
    # Combine signals
    signals = pd.DataFrame(signal_dict)

    # Fit fusion
    fusion = SignalFusion(method=method, calibration=calibration)
    fusion.fit(signals, target)

    # Fuse
    scores = fusion.fuse(signals)

    # Calibrate
    probs = fusion.calibrate(scores, target)

    return probs, fusion


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Multiple signals
    signal1 = pd.Series(np.random.randn(len(dates)), index=dates, name="signal1")
    signal2 = pd.Series(np.random.randn(len(dates)) * 0.8, index=dates, name="signal2")
    signal3 = pd.Series(np.random.randn(len(dates)) * 1.2, index=dates, name="signal3")

    signals = pd.DataFrame({"s1": signal1, "s2": signal2, "s3": signal3})

    # Target (correlated with signals)
    target = ((signal1 + signal2 + signal3) / 3 > 0.5).astype(int)
    target.name = "target"

    # Fuse
    fusion = SignalFusion(method="weighted_average", calibration="isotonic")
    fusion.fit(signals, target)

    print("Weights:", fusion.weights)

    # Get fused score
    fused = fusion.fuse(signals)
    print(f"\nFused score (last 5):")
    print(fused.tail())

    # Calibrate
    calibrated = fusion.calibrate(fused, target)
    print(f"\nCalibrated probabilities (last 5):")
    print(calibrated.tail())
