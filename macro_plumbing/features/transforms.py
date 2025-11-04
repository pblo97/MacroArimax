"""
transforms.py
Feature transformations for liquidity stress detection:
- Z-scores (rolling)
- Rate of change
- ATR-normalized moves
- Quarter-end boost flags
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


def zscore_rolling(
    s: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Calculate rolling z-score.

    Parameters
    ----------
    s : pd.Series
        Input series
    window : int
        Rolling window size
    min_periods : int, optional
        Minimum observations required

    Returns
    -------
    pd.Series
        Z-score series
    """
    if min_periods is None:
        min_periods = window

    roll = s.rolling(window, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std()

    # Avoid division by zero
    z = (s - mean) / std.replace(0, np.nan)
    z.name = f"{s.name}_z{window}"
    return z


def rate_of_change(
    s: pd.Series,
    periods: int = 1,
    method: str = "pct",
) -> pd.Series:
    """
    Calculate rate of change.

    Parameters
    ----------
    s : pd.Series
        Input series
    periods : int
        Number of periods for change
    method : str
        'pct' for percentage change, 'diff' for absolute difference, 'log' for log returns

    Returns
    -------
    pd.Series
        Rate of change series
    """
    if method == "pct":
        roc = s.pct_change(periods)
    elif method == "diff":
        roc = s.diff(periods)
    elif method == "log":
        roc = np.log(s / s.shift(periods))
    else:
        raise ValueError(f"Unknown method: {method}")

    roc.name = f"{s.name}_roc{periods}"
    return roc


def atr_normalized_move(
    s: pd.Series,
    atr_window: int = 14,
    method: str = "simple",
) -> pd.Series:
    """
    Normalize price moves by Average True Range (ATR).

    Parameters
    ----------
    s : pd.Series
        Input series (typically a price)
    atr_window : int
        Window for ATR calculation
    method : str
        'simple' for rolling std, 'true_range' for actual TR-based ATR

    Returns
    -------
    pd.Series
        ATR-normalized move
    """
    change = s.diff()

    if method == "simple":
        # Use rolling std as proxy for ATR
        atr = change.rolling(atr_window, min_periods=atr_window).std()
    elif method == "true_range":
        # True Range calculation (simplified for single series)
        high = s.rolling(2).max()
        low = s.rolling(2).min()
        tr = high - low
        atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Avoid division by zero
    normalized = change / atr.replace(0, np.nan)
    normalized.name = f"{s.name}_atr_norm"
    return normalized


def quarter_end_boost(
    index: pd.DatetimeIndex,
    window_days: int = 5,
) -> pd.Series:
    """
    Create quarter-end boost flag (1 if within window_days of quarter end).

    Parameters
    ----------
    index : pd.DatetimeIndex
        Date index
    window_days : int
        Number of days around quarter end to flag

    Returns
    -------
    pd.Series
        Binary flag series
    """
    df = pd.DataFrame(index=index)
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_qend_month"] = df["month"].isin([3, 6, 9, 12])

    # Days until end of month
    df["days_to_month_end"] = (df.index + pd.offsets.MonthEnd(0) - df.index).days

    # Flag: within window and in quarter-end month
    boost = (df["is_qend_month"]) & (df["days_to_month_end"] <= window_days)
    boost = boost.astype(int)
    boost.name = "quarter_end_boost"
    return boost


def month_end_boost(
    index: pd.DatetimeIndex,
    window_days: int = 3,
) -> pd.Series:
    """
    Create month-end boost flag (1 if within window_days of month end).

    Parameters
    ----------
    index : pd.DatetimeIndex
        Date index
    window_days : int
        Number of days around month end to flag

    Returns
    -------
    pd.Series
        Binary flag series
    """
    df = pd.DataFrame(index=index)
    df["days_to_month_end"] = (df.index + pd.offsets.MonthEnd(0) - df.index).days

    boost = (df["days_to_month_end"] <= window_days).astype(int)
    boost.name = "month_end_boost"
    return boost


def ewma_smoother(
    s: pd.Series,
    span: int,
    adjust: bool = False,
) -> pd.Series:
    """
    Exponentially weighted moving average.

    Parameters
    ----------
    s : pd.Series
        Input series
    span : int
        Span for EWMA (roughly equivalent to window)
    adjust : bool
        Adjust for bias

    Returns
    -------
    pd.Series
        Smoothed series
    """
    smoothed = s.ewm(span=span, adjust=adjust).mean()
    smoothed.name = f"{s.name}_ewma{span}"
    return smoothed


def percentile_rank(
    s: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate rolling percentile rank (0-100).

    Parameters
    ----------
    s : pd.Series
        Input series
    window : int
        Rolling window size

    Returns
    -------
    pd.Series
        Percentile rank series (0-100)
    """
    def _pct_rank(x):
        if len(x) < 2:
            return np.nan
        return (x < x.iloc[-1]).sum() / len(x) * 100

    pct = s.rolling(window).apply(_pct_rank, raw=False)
    pct.name = f"{s.name}_pctrank{window}"
    return pct


def vol_adjusted_zscore(
    s: pd.Series,
    window: int,
    vol_window: Optional[int] = None,
) -> pd.Series:
    """
    Calculate z-score adjusted for recent volatility.

    Uses exponential weighting for recent volatility adjustment.

    Parameters
    ----------
    s : pd.Series
        Input series
    window : int
        Window for mean calculation
    vol_window : int, optional
        Window for volatility (defaults to window)

    Returns
    -------
    pd.Series
        Volatility-adjusted z-score
    """
    if vol_window is None:
        vol_window = window

    # Rolling mean
    mean = s.rolling(window, min_periods=window).mean()

    # EWMA of volatility (more reactive to recent changes)
    vol = s.rolling(vol_window, min_periods=vol_window).std()
    vol_ewma = vol.ewm(span=vol_window // 2).mean()

    # Z-score
    z = (s - mean) / vol_ewma.replace(0, np.nan)
    z.name = f"{s.name}_vol_adj_z"
    return z


def create_feature_matrix(
    df: pd.DataFrame,
    features_config: dict,
) -> pd.DataFrame:
    """
    Create feature matrix from raw data based on configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data
    features_config : dict
        Configuration dict with feature definitions

    Returns
    -------
    pd.DataFrame
        Feature matrix
    """
    feature_df = df.copy()

    # Example processing (customize based on config)
    for col in df.columns:
        if col in features_config:
            cfg = features_config[col]

            # Z-scores
            if cfg.get("zscore"):
                window = cfg.get("zscore_window", 52)
                feature_df[f"{col}_z"] = zscore_rolling(df[col], window)

            # Rate of change
            if cfg.get("roc"):
                periods = cfg.get("roc_periods", [1, 5, 10])
                if not isinstance(periods, list):
                    periods = [periods]
                for p in periods:
                    feature_df[f"{col}_roc{p}"] = rate_of_change(df[col], p)

            # ATR normalization
            if cfg.get("atr_norm"):
                feature_df[f"{col}_atr"] = atr_normalized_move(df[col])

    return feature_df


# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame({
        "price": np.cumsum(np.random.randn(len(dates))) + 100,
        "volume": np.abs(np.random.randn(len(dates))) * 1000,
    }, index=dates)

    # Z-score
    df["price_z"] = zscore_rolling(df["price"], window=60)

    # ROC
    df["price_roc5"] = rate_of_change(df["price"], periods=5)

    # ATR normalized
    df["price_atr"] = atr_normalized_move(df["price"], atr_window=14)

    # Quarter-end boost
    df["qe_boost"] = quarter_end_boost(df.index, window_days=5)

    print(df.tail(10))
