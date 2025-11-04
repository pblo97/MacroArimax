"""
dfm_kalman.py
Dynamic Factor Model with Kalman Filter for liquidity factor estimation.

Model:
  y_t = Λ * f_t + ε_t    (observation equation)
  f_t = Φ * f_{t-1} + η_t  (state equation)

where:
  - y_t: observed indicators (N x 1)
  - f_t: latent liquidity factor (1 x 1)
  - Λ: factor loadings (N x 1)
  - Φ: autoregressive coefficient (1 x 1)
  - ε_t ~ N(0, R): observation noise
  - η_t ~ N(0, Q): state noise

The Kalman filter provides optimal estimation and smoothing of the latent factor.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor


class LiquidityDFM:
    """
    Dynamic Factor Model for estimating latent liquidity factor.
    """

    def __init__(
        self,
        n_factors: int = 1,
        factor_order: int = 1,
        error_order: int = 0,
    ):
        """
        Initialize DFM.

        Parameters
        ----------
        n_factors : int
            Number of latent factors (usually 1 for liquidity)
        factor_order : int
            AR order for factor dynamics
        error_order : int
            MA order for idiosyncratic errors
        """
        self.n_factors = n_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.model = None
        self.result = None
        self.scaler = StandardScaler()

    def fit(
        self,
        df: pd.DataFrame,
        scale: bool = True,
    ) -> "LiquidityDFM":
        """
        Fit the DFM to data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data (T x N) with indicators as columns
        scale : bool
            Whether to standardize data

        Returns
        -------
        self
        """
        # Prepare data
        data = df.dropna()

        if scale:
            data_scaled = pd.DataFrame(
                self.scaler.fit_transform(data),
                index=data.index,
                columns=data.columns,
            )
        else:
            data_scaled = data

        # Fit DFM
        self.model = DynamicFactor(
            data_scaled,
            k_factors=self.n_factors,
            factor_order=self.factor_order,
            error_order=self.error_order,
        )

        try:
            self.result = self.model.fit(disp=False, maxiter=1000)
        except Exception as e:
            print(f"DFM fitting failed: {e}")
            # Fallback to simple PCA
            self.result = None

        return self

    def get_factor(
        self,
        smoothed: bool = True,
    ) -> pd.Series:
        """
        Extract latent factor.

        Parameters
        ----------
        smoothed : bool
            If True, use smoothed estimates (uses all data)
            If False, use filtered estimates (real-time)

        Returns
        -------
        pd.Series
            Latent factor time series
        """
        if self.result is None:
            return pd.Series(dtype=float)

        if smoothed:
            factor = pd.Series(
                self.result.factors.smoothed[0, :],
                index=self.result.data.dates,
                name="liquidity_factor_smoothed",
            )
        else:
            factor = pd.Series(
                self.result.factors.filtered[0, :],
                index=self.result.data.dates,
                name="liquidity_factor_filtered",
            )

        return factor

    def get_factor_zscore(
        self,
        window: Optional[int] = None,
        smoothed: bool = True,
    ) -> pd.Series:
        """
        Get z-score of latent factor.

        Parameters
        ----------
        window : int, optional
            Rolling window for z-score (if None, use full sample)
        smoothed : bool
            Use smoothed or filtered factor

        Returns
        -------
        pd.Series
            Z-scored factor
        """
        factor = self.get_factor(smoothed=smoothed)

        if window is None:
            z = (factor - factor.mean()) / factor.std()
        else:
            roll = factor.rolling(window, min_periods=window)
            z = (factor - roll.mean()) / roll.std()

        z.name = "liquidity_factor_z"
        return z

    def get_loadings(self) -> pd.Series:
        """
        Get factor loadings (Λ).

        Returns
        -------
        pd.Series
            Factor loadings for each indicator
        """
        if self.result is None:
            return pd.Series(dtype=float)

        loadings = pd.Series(
            self.result.params[: self.result.k_endog],
            index=self.result.data.ynames,
            name="factor_loadings",
        )
        return loadings

    def forecast_factor(
        self,
        steps: int = 10,
    ) -> pd.Series:
        """
        Forecast latent factor.

        Parameters
        ----------
        steps : int
            Number of steps ahead

        Returns
        -------
        pd.Series
            Forecasted factor
        """
        if self.result is None:
            return pd.Series(dtype=float)

        forecast = self.result.forecast(steps=steps)
        # Extract factor forecast (not observations)
        # This is simplified; actual implementation depends on statsmodels version

        return pd.Series(dtype=float)  # Placeholder


class SimpleKalmanFilter:
    """
    Simplified Kalman filter for univariate AR(1) factor.

    More efficient than full DFM when you have a pre-computed composite.
    """

    def __init__(
        self,
        transition_coeff: float = 0.95,
        process_var: float = 0.1,
        measurement_var: float = 1.0,
    ):
        """
        Initialize Kalman filter.

        Parameters
        ----------
        transition_coeff : float
            Φ in state equation (AR coefficient)
        process_var : float
            Q: process noise variance
        measurement_var : float
            R: measurement noise variance
        """
        self.phi = transition_coeff
        self.Q = process_var
        self.R = measurement_var

        # State estimates
        self.x_filtered = []
        self.P_filtered = []
        self.x_smoothed = []

    def filter(
        self,
        observations: pd.Series,
    ) -> pd.Series:
        """
        Apply Kalman filter to observations.

        Parameters
        ----------
        observations : pd.Series
            Observed signal

        Returns
        -------
        pd.Series
            Filtered (real-time) estimates
        """
        y = observations.values
        T = len(y)

        # Initialize
        x = np.zeros(T)  # State estimates
        P = np.zeros(T)  # State covariances

        x[0] = y[0]
        P[0] = 1.0

        # Forward pass (filtering)
        for t in range(1, T):
            # Prediction
            x_pred = self.phi * x[t - 1]
            P_pred = self.phi**2 * P[t - 1] + self.Q

            # Update (if observation available)
            if not np.isnan(y[t]):
                # Kalman gain
                K = P_pred / (P_pred + self.R)

                # Update
                x[t] = x_pred + K * (y[t] - x_pred)
                P[t] = (1 - K) * P_pred
            else:
                # No observation
                x[t] = x_pred
                P[t] = P_pred

        self.x_filtered = x
        self.P_filtered = P

        return pd.Series(x, index=observations.index, name="kalman_filtered")

    def smooth(
        self,
        observations: pd.Series,
    ) -> pd.Series:
        """
        Apply Kalman smoother (uses all data, backward pass).

        Parameters
        ----------
        observations : pd.Series
            Observed signal

        Returns
        -------
        pd.Series
            Smoothed estimates
        """
        # First run filter
        if len(self.x_filtered) == 0:
            self.filter(observations)

        y = observations.values
        T = len(y)

        x_smooth = self.x_filtered.copy()
        P_smooth = self.P_filtered.copy()

        # Backward pass (smoothing)
        for t in range(T - 2, -1, -1):
            # Smoother gain
            P_pred = self.phi**2 * P_smooth[t] + self.Q
            J = P_smooth[t] * self.phi / P_pred

            # Smooth
            x_smooth[t] = x_smooth[t] + J * (x_smooth[t + 1] - self.phi * x_smooth[t])
            P_smooth[t] = P_smooth[t] + J**2 * (P_smooth[t + 1] - P_pred)

        self.x_smoothed = x_smooth

        return pd.Series(x_smooth, index=observations.index, name="kalman_smoothed")


def fit_dfm_liquidity(
    df: pd.DataFrame,
    n_factors: int = 1,
    scale: bool = True,
) -> Tuple[pd.Series, pd.Series, LiquidityDFM]:
    """
    Convenience function to fit DFM and extract factors.

    Parameters
    ----------
    df : pd.DataFrame
        Input indicators
    n_factors : int
        Number of factors
    scale : bool
        Standardize inputs

    Returns
    -------
    tuple
        (filtered_factor, smoothed_factor, model)
    """
    model = LiquidityDFM(n_factors=n_factors)
    model.fit(df, scale=scale)

    filtered = model.get_factor(smoothed=False)
    smoothed = model.get_factor(smoothed=True)

    return filtered, smoothed, model


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Simulate indicators driven by latent factor
    true_factor = np.cumsum(np.random.randn(len(dates))) * 0.1
    n_indicators = 5

    data = {}
    for i in range(n_indicators):
        loading = np.random.uniform(0.5, 1.5)
        noise = np.random.randn(len(dates)) * 0.5
        data[f"indicator_{i}"] = true_factor * loading + noise

    df = pd.DataFrame(data, index=dates)

    # Fit DFM
    print("Fitting DFM...")
    filtered, smoothed, model = fit_dfm_liquidity(df)

    print("\nFiltered factor (last 5):")
    print(filtered.tail())

    print("\nSmoothed factor (last 5):")
    print(smoothed.tail())

    # Simple Kalman filter on a composite
    composite = df.mean(axis=1)
    kf = SimpleKalmanFilter()
    smoothed_kf = kf.smooth(composite)

    print("\nKalman smoothed composite (last 5):")
    print(smoothed_kf.tail())