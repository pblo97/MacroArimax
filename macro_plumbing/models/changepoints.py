"""
changepoints.py
Structural break detection using ruptures library (Pelt, Binary Segmentation).
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import ruptures as rpt


def detect_changepoints(
    x: pd.Series,
    model: str = "rbf",
    method: str = "pelt",
    penalty: float = 10.0,
    min_size: int = 30,
) -> List[int]:
    """
    Detect structural breaks in time series.

    Parameters
    ----------
    x : pd.Series
        Input series
    model : str
        Cost function: 'l1', 'l2', 'rbf', 'normal', etc.
    method : str
        Detection method: 'pelt', 'binseg', 'bottomup', 'window'
    penalty : float
        Penalty value (higher = fewer breaks)
    min_size : int
        Minimum segment size

    Returns
    -------
    list
        Change point indices
    """
    signal = x.dropna().values.reshape(-1, 1)

    # Select algorithm
    if method == "pelt":
        algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
        breakpoints = algo.predict(pen=penalty)
    elif method == "binseg":
        algo = rpt.Binseg(model=model, min_size=min_size).fit(signal)
        n_breaks = int(len(signal) / (min_size * 3))  # Auto-determine
        breakpoints = algo.predict(n_bkps=max(1, n_breaks))
    elif method == "window":
        algo = rpt.Window(width=min_size, model=model).fit(signal)
        breakpoints = algo.predict(pen=penalty)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ruptures returns indices in the signal, convert to original index
    return breakpoints[:-1]  # Remove last point (end of series)


def changepoint_flags(
    x: pd.Series,
    method: str = "pelt",
    penalty: float = 10.0,
    window: int = 5,
) -> pd.Series:
    """
    Create binary flags for change points (with window around them).

    Parameters
    ----------
    x : pd.Series
        Input series
    method : str
        Detection method
    penalty : float
        Penalty
    window : int
        Flag window around change point

    Returns
    -------
    pd.Series
        Binary flags (1 = change point region, 0 = stable)
    """
    x_clean = x.dropna()
    bkps = detect_changepoints(x_clean, method=method, penalty=penalty)

    flags = pd.Series(0, index=x.index)

    for bkp in bkps:
        # Get actual date
        if bkp < len(x_clean):
            date = x_clean.index[bkp]
            # Flag window around it
            start = max(0, bkp - window)
            end = min(len(x_clean), bkp + window)
            for i in range(start, end):
                if i < len(x_clean):
                    flags[x_clean.index[i]] = 1

    flags.name = "changepoint_flag"
    return flags


def segment_statistics(
    x: pd.Series,
    breakpoints: List[int],
) -> pd.DataFrame:
    """
    Calculate statistics for each segment between breakpoints.

    Parameters
    ----------
    x : pd.Series
        Input series
    breakpoints : list
        Change point indices

    Returns
    -------
    pd.DataFrame
        Segment statistics
    """
    x_clean = x.dropna()
    segments = []

    start = 0
    for bkp in breakpoints + [len(x_clean)]:
        segment = x_clean.iloc[start:bkp]
        if len(segment) > 0:
            segments.append(
                {
                    "start": start,
                    "end": bkp,
                    "start_date": x_clean.index[start],
                    "end_date": x_clean.index[bkp - 1] if bkp > 0 else x_clean.index[0],
                    "length": len(segment),
                    "mean": segment.mean(),
                    "std": segment.std(),
                    "min": segment.min(),
                    "max": segment.max(),
                }
            )
        start = bkp

    return pd.DataFrame(segments)


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Create series with structural breaks
    seg1 = np.random.randn(400) + 0
    seg2 = np.random.randn(400) + 3  # Break
    seg3 = np.random.randn(len(dates) - 800) - 2  # Another break
    signal = np.concatenate([seg1, seg2, seg3])
    series = pd.Series(signal, index=dates)

    # Detect breakpoints
    print("Detecting change points...")
    bkps = detect_changepoints(series, method="pelt", penalty=5.0)
    print(f"Found {len(bkps)} change points at indices: {bkps}")

    # Flags
    flags = changepoint_flags(series, method="pelt", penalty=5.0)
    print(f"Change point flags sum: {flags.sum()}")

    # Segment stats
    stats = segment_statistics(series, bkps)
    print("\nSegment statistics:")
    print(stats)
