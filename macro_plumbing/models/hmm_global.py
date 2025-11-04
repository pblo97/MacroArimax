"""
hmm_global.py
Hidden Markov Model / Markov Switching for global regime detection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.preprocessing import StandardScaler


class LiquidityHMM:
    """HMM for liquidity regime detection (calm vs stressed)."""

    def __init__(
        self,
        n_regimes: int = 2,
        switching_variance: bool = True,
    ):
        """
        Initialize HMM.

        Parameters
        ----------
        n_regimes : int
            Number of regimes (typically 2: calm/stressed)
        switching_variance : bool
            Allow variance to switch between regimes
        """
        self.n_regimes = n_regimes
        self.switching_variance = switching_variance
        self.model = None
        self.result = None
        self.scaler = StandardScaler()

    def fit(
        self,
        y: pd.Series,
        X: Optional[pd.DataFrame] = None,
    ) -> "LiquidityHMM":
        """
        Fit HMM to data.

        Parameters
        ----------
        y : pd.Series
            Dependent variable (e.g., equity returns)
        X : pd.DataFrame, optional
            Exogenous variables (e.g., liquidity factor)

        Returns
        -------
        self
        """
        # Prepare data
        if X is not None:
            data = pd.concat([y, X], axis=1).dropna()
            y_data = data[y.name]
            X_data = sm.add_constant(data.drop(columns=y.name))

            # Standardize
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_data),
                index=X_data.index,
                columns=X_data.columns,
            )
        else:
            y_data = y.dropna()
            X_scaled = None

        # Fit Markov Switching model
        try:
            if X_scaled is not None:
                self.model = MarkovRegression(
                    y_data,
                    k_regimes=self.n_regimes,
                    exog=X_scaled,
                    switching_variance=self.switching_variance,
                )
            else:
                self.model = MarkovRegression(
                    y_data,
                    k_regimes=self.n_regimes,
                    switching_variance=self.switching_variance,
                )

            self.result = self.model.fit(method="bfgs", maxiter=1000, disp=False)
        except Exception as e:
            print(f"HMM fitting failed: {e}")
            self.result = None

        return self

    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Get smoothed regime probabilities.

        Returns
        -------
        pd.DataFrame
            Probabilities for each regime
        """
        if self.result is None:
            return pd.DataFrame()

        probs = self.result.smoothed_marginal_probabilities
        probs.columns = [f"prob_regime_{i}" for i in range(self.n_regimes)]
        return probs

    def get_stress_probability(self) -> pd.Series:
        """
        Get probability of stress regime.

        Assumes regime 0 = calm, regime 1 = stressed.
        (Or identify by which has higher volatility)

        Returns
        -------
        pd.Series
            Probability of stressed regime
        """
        if self.result is None:
            return pd.Series(dtype=float)

        probs = self.get_regime_probabilities()

        # Identify stress regime (higher variance)
        try:
            # Extract regime variances from params
            sigmas = [self.result.params[f"sigma2.{i}"] for i in range(self.n_regimes)]
            stress_regime = np.argmax(sigmas)
        except:
            # Fallback: assume regime 1 is stress
            stress_regime = 1

        prob_stress = probs[f"prob_regime_{stress_regime}"]
        prob_stress.name = "prob_stress"
        return prob_stress

    def get_most_likely_regime(self) -> pd.Series:
        """Get most likely regime at each time point."""
        probs = self.get_regime_probabilities()
        if probs.empty:
            return pd.Series(dtype=int)

        regime = probs.idxmax(axis=1).str.extract(r"(\d+)").astype(int).squeeze()
        regime.name = "regime"
        return regime


def fit_liquidity_hmm(
    y: pd.Series,
    X: Optional[pd.DataFrame] = None,
    n_regimes: int = 2,
) -> Tuple[pd.Series, LiquidityHMM]:
    """
    Convenience function to fit HMM and get stress probability.

    Parameters
    ----------
    y : pd.Series
        Target variable
    X : pd.DataFrame, optional
        Exogenous predictors
    n_regimes : int
        Number of regimes

    Returns
    -------
    tuple
        (prob_stress, model)
    """
    model = LiquidityHMM(n_regimes=n_regimes)
    model.fit(y, X)
    prob_stress = model.get_stress_probability()
    return prob_stress, model


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Simulate regime-switching returns
    n = len(dates)
    regimes = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    returns = np.where(regimes == 0, np.random.randn(n) * 0.01, np.random.randn(n) * 0.03)
    y = pd.Series(returns, index=dates, name="returns")

    # Predictor
    X = pd.DataFrame({"liquidity": np.cumsum(np.random.randn(n)) * 0.1}, index=dates)

    # Fit HMM
    print("Fitting HMM...")
    prob_stress, model = fit_liquidity_hmm(y, X)

    print(f"\nStress probability (last 5):")
    print(prob_stress.tail())

    print(f"\nRegimes (last 10):")
    print(model.get_most_likely_regime().tail(10))
