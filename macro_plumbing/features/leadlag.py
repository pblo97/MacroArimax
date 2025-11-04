"""
leadlag.py
Lead-lag analysis for identifying leading indicators of liquidity stress.

Cross-correlation scanner to determine optimal lags for each signal.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats


def cross_correlation(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 10,
) -> pd.Series:
    """
    Calculate cross-correlation between two series at various lags.

    Parameters
    ----------
    x : pd.Series
        Leading series (potentially)
    y : pd.Series
        Lagging series (target)
    max_lag : int
        Maximum lag to test

    Returns
    -------
    pd.Series
        Cross-correlation at each lag (-max_lag to +max_lag)
    """
    # Align series
    df = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()

    if len(df) < max_lag + 10:
        return pd.Series(dtype=float)

    lags = range(-max_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        if lag < 0:
            # x leads y (shift y backward / x forward)
            corr = df["x"].iloc[:lag].corr(df["y"].iloc[-lag:])
        elif lag > 0:
            # y leads x (shift x backward / y forward)
            corr = df["x"].iloc[lag:].corr(df["y"].iloc[:-lag])
        else:
            # No lag
            corr = df["x"].corr(df["y"])

        corrs.append(corr)

    result = pd.Series(corrs, index=lags)
    result.name = f"ccf_{x.name}_vs_{y.name}"
    return result


def find_optimal_lag(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 10,
    direction: str = "both",
) -> Tuple[int, float]:
    """
    Find optimal lag with maximum absolute correlation.

    Parameters
    ----------
    x : pd.Series
        Predictor series
    y : pd.Series
        Target series
    max_lag : int
        Maximum lag to test
    direction : str
        'positive' (x leads y), 'negative' (y leads x), 'both'

    Returns
    -------
    tuple
        (optimal_lag, max_correlation)
        - Positive lag: x leads y by N periods
        - Negative lag: y leads x by N periods
    """
    ccf = cross_correlation(x, y, max_lag)

    if ccf.empty:
        return (0, np.nan)

    # Filter by direction
    if direction == "positive":
        ccf = ccf[ccf.index >= 0]
    elif direction == "negative":
        ccf = ccf[ccf.index <= 0]

    # Find max absolute correlation
    abs_ccf = ccf.abs()
    optimal_lag = abs_ccf.idxmax()
    max_corr = ccf.loc[optimal_lag]

    return (int(optimal_lag), float(max_corr))


def leadlag_scanner(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: Optional[List[str]] = None,
    max_lag: int = 10,
    min_corr: float = 0.1,
) -> pd.DataFrame:
    """
    Scan all predictors for lead-lag relationships with target.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    target_col : str
        Target variable
    predictor_cols : list, optional
        List of predictor columns (if None, use all except target)
    max_lag : int
        Maximum lag to test
    min_corr : float
        Minimum correlation threshold for reporting

    Returns
    -------
    pd.DataFrame
        Summary of lead-lag relationships, sorted by abs(correlation)
    """
    if predictor_cols is None:
        predictor_cols = [c for c in df.columns if c != target_col]

    results = []

    for col in predictor_cols:
        if col not in df.columns:
            continue

        # Find optimal lag
        lag, corr = find_optimal_lag(df[col], df[target_col], max_lag, direction="both")

        # Interpret
        if lag > 0:
            interpretation = f"{col} leads {target_col} by {lag} periods"
        elif lag < 0:
            interpretation = f"{target_col} leads {col} by {-lag} periods"
        else:
            interpretation = "Contemporaneous"

        results.append({
            "predictor": col,
            "target": target_col,
            "optimal_lag": lag,
            "correlation": corr,
            "abs_correlation": abs(corr),
            "interpretation": interpretation,
        })

    result_df = pd.DataFrame(results)

    # Filter and sort
    result_df = result_df[result_df["abs_correlation"] >= min_corr]
    result_df = result_df.sort_values("abs_correlation", ascending=False)

    return result_df


def granger_causality_test(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 5,
) -> Dict[str, float]:
    """
    Simple Granger causality test: does x Granger-cause y?

    Uses F-test to compare:
    - Restricted model: y ~ lags(y)
    - Unrestricted model: y ~ lags(y) + lags(x)

    Parameters
    ----------
    x : pd.Series
        Potential cause
    y : pd.Series
        Effect
    max_lag : int
        Number of lags

    Returns
    -------
    dict
        {'f_stat': ..., 'p_value': ..., 'causality': bool}
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # Prepare data
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()

    if len(df) < max_lag + 20:
        return {"f_stat": np.nan, "p_value": np.nan, "causality": False}

    try:
        # Run test
        result = grangercausalitytests(df[["y", "x"]], max_lag, verbose=False)

        # Extract best lag result
        best_lag = max_lag
        f_stat = result[best_lag][0]["ssr_ftest"][0]
        p_value = result[best_lag][0]["ssr_ftest"][1]

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "causality": p_value < 0.05,
            "lag": best_lag,
        }
    except Exception as e:
        return {"f_stat": np.nan, "p_value": np.nan, "causality": False, "error": str(e)}


def rolling_leadlag(
    x: pd.Series,
    y: pd.Series,
    window: int = 252,
    lag: int = 1,
) -> pd.Series:
    """
    Calculate rolling correlation with fixed lag.

    Useful for monitoring stability of lead-lag relationship.

    Parameters
    ----------
    x : pd.Series
        Predictor
    y : pd.Series
        Target
    window : int
        Rolling window
    lag : int
        Fixed lag (positive if x leads y)

    Returns
    -------
    pd.Series
        Rolling correlation
    """
    if lag > 0:
        # x leads y
        x_shifted = x.shift(lag)
    elif lag < 0:
        # y leads x
        x_shifted = x.shift(lag)
    else:
        x_shifted = x

    rolling_corr = x_shifted.rolling(window).corr(y)
    rolling_corr.name = f"rolling_corr_lag{lag}"
    return rolling_corr


def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """
    Create lagged versions of specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    columns : list
        Columns to lag
    lags : list
        List of lag values

    Returns
    -------
    pd.DataFrame
        DataFrame with original + lagged columns
    """
    result = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for lag in lags:
            if lag == 0:
                continue
            result[f"{col}_L{lag}"] = df[col].shift(lag)

    return result


def information_coefficient(
    predictions: pd.Series,
    actuals: pd.Series,
) -> float:
    """
    Calculate Information Coefficient (Spearman rank correlation).

    Commonly used in quantitative finance to measure predictive power.

    Parameters
    ----------
    predictions : pd.Series
        Predicted values
    actuals : pd.Series
        Actual values

    Returns
    -------
    float
        Spearman correlation
    """
    df = pd.concat([predictions, actuals], axis=1).dropna()
    if len(df) < 10:
        return np.nan

    ic, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return ic


def leadlag_heatmap_data(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    Generate data for lead-lag heatmap visualization.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    target_col : str
        Target variable
    predictor_cols : list
        Predictor variables
    max_lag : int
        Maximum lag

    Returns
    -------
    pd.DataFrame
        Correlation matrix (predictors x lags)
    """
    lags = range(-max_lag, max_lag + 1)
    heatmap_data = []

    for pred in predictor_cols:
        if pred not in df.columns:
            continue

        ccf = cross_correlation(df[pred], df[target_col], max_lag)
        heatmap_data.append(ccf)

    result = pd.DataFrame(heatmap_data, index=predictor_cols)
    result.columns.name = "Lag"
    result.index.name = "Predictor"

    return result


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Create synthetic series where x1 leads target by 3 days
    target = np.cumsum(np.random.randn(len(dates)))
    x1 = np.roll(target, 3) + np.random.randn(len(dates)) * 0.5  # x1 leads by 3
    x2 = np.cumsum(np.random.randn(len(dates)))  # Independent

    df = pd.DataFrame({
        "target": target,
        "x1": x1,
        "x2": x2,
    }, index=dates)

    # Lead-lag scanner
    print("Lead-Lag Scanner Results:")
    leadlag_results = leadlag_scanner(df, "target", max_lag=10)
    print(leadlag_results)

    # Cross-correlation for x1
    print("\nCross-correlation (x1 vs target):")
    ccf = cross_correlation(df["x1"], df["target"], max_lag=10)
    print(ccf)

    # Optimal lag
    lag, corr = find_optimal_lag(df["x1"], df["target"], max_lag=10)
    print(f"\nOptimal lag: {lag}, Correlation: {corr:.3f}")
