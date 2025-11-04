"""
cusum_ewma.py
CUSUM and EWMA control charts for detecting regime shifts in liquidity.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class CUSUM:
    """CUSUM (Cumulative Sum) detector for mean shifts."""

    def __init__(self, target_mean: float = 0.0, k: float = 0.5, h: float = 4.0):
        """
        Initialize CUSUM.

        Parameters
        ----------
        target_mean : float
            Target/expected mean
        k : float
            Slack parameter (typically 0.5 * sigma)
        h : float
            Threshold for alarm (typically 4-5 * sigma)
        """
        self.target = target_mean
        self.k = k
        self.h = h

    def detect(self, x: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Run CUSUM on series.

        Returns
        -------
        tuple
            (cusum_positive, cusum_negative) - upper and lower CUSUM statistics
        """
        x_arr = x.values
        n = len(x_arr)

        S_pos = np.zeros(n)
        S_neg = np.zeros(n)

        for t in range(1, n):
            S_pos[t] = max(0, S_pos[t - 1] + (x_arr[t] - self.target - self.k))
            S_neg[t] = min(0, S_neg[t - 1] + (x_arr[t] - self.target + self.k))

        return (
            pd.Series(S_pos, index=x.index, name="cusum_pos"),
            pd.Series(np.abs(S_neg), index=x.index, name="cusum_neg"),
        )

    def get_signals(self, x: pd.Series) -> pd.Series:
        """Get binary alarm signals (1 = alarm, 0 = ok)."""
        S_pos, S_neg = self.detect(x)
        alarm = ((S_pos > self.h) | (S_neg > self.h)).astype(int)
        alarm.name = "cusum_alarm"
        return alarm


class EWMA:
    """EWMA (Exponentially Weighted Moving Average) control chart."""

    def __init__(self, lambda_: float = 0.2, L: float = 3.0):
        """
        Initialize EWMA.

        Parameters
        ----------
        lambda_ : float
            Smoothing parameter (0 < λ ≤ 1)
            Smaller values = more smoothing
        L : float
            Control limit multiplier (typically 2.7-3.0)
        """
        self.lambda_ = lambda_
        self.L = L

    def detect(self, x: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Run EWMA.

        Returns
        -------
        tuple
            (ewma, upper_limit, lower_limit)
        """
        ewma = x.ewm(alpha=self.lambda_, adjust=False).mean()

        # Asymptotic variance
        sigma = x.std()
        var_ewma = sigma**2 * (self.lambda_ / (2 - self.lambda_))
        std_ewma = np.sqrt(var_ewma)

        mean = x.mean()
        ucl = mean + self.L * std_ewma
        lcl = mean - self.L * std_ewma

        return (
            ewma,
            pd.Series(ucl, index=x.index, name="ewma_ucl"),
            pd.Series(lcl, index=x.index, name="ewma_lcl"),
        )

    def get_signals(self, x: pd.Series) -> pd.Series:
        """Get binary alarm signals."""
        ewma, ucl, lcl = self.detect(x)
        alarm = ((ewma > ucl) | (ewma < lcl)).astype(int)
        alarm.name = "ewma_alarm"
        return alarm


def detect_stress_cusum(spread: pd.Series, k: float = 0.5, h: float = 4.0) -> pd.Series:
    """Convenience function for CUSUM stress detection."""
    cusum = CUSUM(target_mean=spread.mean(), k=k, h=h)
    return cusum.get_signals(spread)


def detect_stress_ewma(spread: pd.Series, lambda_: float = 0.2, L: float = 3.0) -> pd.Series:
    """Convenience function for EWMA stress detection."""
    ewma = EWMA(lambda_=lambda_, L=L)
    return ewma.get_signals(spread)


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Normal regime then mean shift
    x1 = np.random.randn(len(dates) // 2)
    x2 = np.random.randn(len(dates) // 2) + 2.0  # Mean shift
    x = np.concatenate([x1, x2])
    series = pd.Series(x, index=dates)

    # CUSUM
    cusum = CUSUM(k=0.5, h=3.0)
    S_pos, S_neg = cusum.detect(series)
    alarms = cusum.get_signals(series)
    print(f"CUSUM alarms: {alarms.sum()} out of {len(alarms)}")

    # EWMA
    ewma_det = EWMA(lambda_=0.2, L=3.0)
    ewma, ucl, lcl = ewma_det.detect(series)
    ewma_alarms = ewma_det.get_signals(series)
    print(f"EWMA alarms: {ewma_alarms.sum()} out of {len(ewma_alarms)}")
