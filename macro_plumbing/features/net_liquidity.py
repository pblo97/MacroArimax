"""
net_liquidity.py
Net Liquidity calculation and analysis (Yardeni-style and variants).

Net Liquidity = Reserves - TGA - ON RRP
Reflects actual liquidity available to the financial system.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


def compute_net_liquidity(
    reserves: pd.Series,
    tga: pd.Series,
    rrp: pd.Series,
) -> pd.Series:
    """
    Compute Net Liquidity (Yardeni formula).

    NL = Reserves - TGA - ON RRP

    Parameters
    ----------
    reserves : pd.Series
        Total reserves of depository institutions
    tga : pd.Series
        Treasury General Account balance
    rrp : pd.Series
        Overnight Reverse Repo volume

    Returns
    -------
    pd.Series
        Net Liquidity series
    """
    nl = reserves - tga - rrp
    nl.name = "net_liquidity"
    return nl


def compute_net_liquidity_components(
    df: pd.DataFrame,
    reserves_col: str = "RESERVES",
    tga_col: str = "TGA",
    rrp_col: str = "RRP",
) -> pd.DataFrame:
    """
    Compute Net Liquidity and component contributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with required columns
    reserves_col : str
        Column name for reserves
    tga_col : str
        Column name for TGA
    rrp_col : str
        Column name for RRP

    Returns
    -------
    pd.DataFrame
        DataFrame with NL and components
    """
    result = pd.DataFrame(index=df.index)

    # Check if columns exist, try alternatives if not
    def find_column(df, primary_name, alternatives=None):
        """Find column by primary name or alternatives."""
        if primary_name in df.columns:
            return primary_name
        if alternatives:
            for alt in alternatives:
                if alt in df.columns:
                    return alt
        return None

    # Try to find the columns with alternatives
    reserves_found = find_column(df, reserves_col, ["WRESBAL", "BANK_RESERVES_WEEKLY"])
    tga_found = find_column(df, tga_col, ["WTREGEN"])
    rrp_found = find_column(df, rrp_col, ["RRPONTSYD", "ON_RRP"])

    # Check if we have all required columns
    missing_cols = []
    if not reserves_found:
        missing_cols.append(f"{reserves_col} (tried: WRESBAL, BANK_RESERVES_WEEKLY)")
    if not tga_found:
        missing_cols.append(f"{tga_col} (tried: WTREGEN)")
    if not rrp_found:
        missing_cols.append(f"{rrp_col} (tried: RRPONTSYD, ON_RRP)")

    if missing_cols:
        # Return empty dataframe with expected columns but all NaN
        print(f"WARNING: Net Liquidity calculation - missing columns: {', '.join(missing_cols)}")
        print(f"Available columns: {list(df.columns)[:20]}...")  # Show first 20 columns
        result["reserves"] = np.nan
        result["tga"] = np.nan
        result["rrp"] = np.nan
        result["net_liquidity"] = np.nan
        result["delta_reserves"] = np.nan
        result["delta_tga"] = np.nan
        result["delta_rrp"] = np.nan
        result["delta_net_liquidity"] = np.nan
        return result

    # Raw components
    result["reserves"] = df[reserves_found]
    result["tga"] = df[tga_found]
    result["rrp"] = df[rrp_found]

    # Net Liquidity
    result["net_liquidity"] = compute_net_liquidity(
        df[reserves_found],
        df[tga_found],
        df[rrp_found],
    )

    # Changes
    result["delta_reserves"] = result["reserves"].diff()
    result["delta_tga"] = result["tga"].diff()
    result["delta_rrp"] = result["rrp"].diff()
    result["delta_net_liquidity"] = result["net_liquidity"].diff()

    # Percentage changes
    result["pct_reserves"] = result["reserves"].pct_change()
    result["pct_tga"] = result["tga"].pct_change()
    result["pct_rrp"] = result["rrp"].pct_change()
    result["pct_net_liquidity"] = result["net_liquidity"].pct_change()

    return result


def net_liquidity_stress_score(
    nl: pd.Series,
    window: int = 252,
    percentile_low: float = 0.20,
    percentile_high: float = 0.80,
) -> pd.Series:
    """
    Calculate stress score based on Net Liquidity percentile rank.

    Score:
    - 0.0 = very loose (above 80th percentile)
    - 0.5 = neutral (between 20th and 80th)
    - 1.0 = stressed (below 20th percentile)

    Parameters
    ----------
    nl : pd.Series
        Net Liquidity series
    window : int
        Rolling window for percentile calculation
    percentile_low : float
        Lower threshold (0-1)
    percentile_high : float
        Upper threshold (0-1)

    Returns
    -------
    pd.Series
        Stress score (0-1)
    """
    # Rolling percentile rank
    def _pct_rank(x):
        if len(x) < 2:
            return np.nan
        return (x < x.iloc[-1]).sum() / len(x)

    pct_rank = nl.rolling(window).apply(_pct_rank, raw=False)

    # Map to stress score
    stress = pd.Series(0.5, index=nl.index)  # Neutral default
    stress[pct_rank < percentile_low] = 1.0  # Stressed
    stress[pct_rank > percentile_high] = 0.0  # Loose

    # Smooth transition
    mid_mask = (pct_rank >= percentile_low) & (pct_rank <= percentile_high)
    stress[mid_mask] = 1.0 - (pct_rank[mid_mask] - percentile_low) / (percentile_high - percentile_low)

    stress.name = "nl_stress_score"
    return stress


def net_liquidity_regime(
    nl: pd.Series,
    window: int = 252,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Classify Net Liquidity into regimes based on percentile.

    Regimes:
    - 'crisis': < 10th percentile
    - 'tight': 10th-30th percentile
    - 'neutral': 30th-70th percentile
    - 'loose': 70th-90th percentile
    - 'flood': > 90th percentile

    Parameters
    ----------
    nl : pd.Series
        Net Liquidity series
    window : int
        Rolling window for percentile
    thresholds : dict, optional
        Custom percentile thresholds

    Returns
    -------
    pd.Series
        Regime labels
    """
    if thresholds is None:
        thresholds = {
            "crisis": 0.10,
            "tight": 0.30,
            "neutral": 0.70,
            "loose": 0.90,
        }

    # Rolling percentile rank
    def _pct_rank(x):
        if len(x) < 2:
            return np.nan
        return (x < x.iloc[-1]).sum() / len(x)

    pct_rank = nl.rolling(window).apply(_pct_rank, raw=False)

    # Classify
    regime = pd.Series("neutral", index=nl.index)
    regime[pct_rank < thresholds["crisis"]] = "crisis"
    regime[(pct_rank >= thresholds["crisis"]) & (pct_rank < thresholds["tight"])] = "tight"
    regime[(pct_rank >= thresholds["tight"]) & (pct_rank < thresholds["neutral"])] = "neutral"
    regime[(pct_rank >= thresholds["neutral"]) & (pct_rank < thresholds["loose"])] = "loose"
    regime[pct_rank >= thresholds["loose"]] = "flood"

    regime.name = "nl_regime"
    return regime


def decompose_nl_change(
    df: pd.DataFrame,
    date: pd.Timestamp,
    reserves_col: str = "RESERVES",
    tga_col: str = "TGA",
    rrp_col: str = "RRP",
    periods: int = 1,
) -> Dict[str, float]:
    """
    Decompose Net Liquidity change into component contributions.

    Parameters
    ----------
    df : pd.DataFrame
        Data with reserves, TGA, RRP
    date : pd.Timestamp
        Date for decomposition
    reserves_col : str
        Reserves column
    tga_col : str
        TGA column
    rrp_col : str
        RRP column
    periods : int
        Number of periods for change

    Returns
    -------
    dict
        Component contributions
    """
    idx = df.index.get_loc(date)
    prev_idx = idx - periods

    if prev_idx < 0:
        return {
            "delta_reserves": np.nan,
            "delta_tga": np.nan,
            "delta_rrp": np.nan,
            "delta_nl": np.nan,
        }

    prev_date = df.index[prev_idx]

    delta_reserves = df.loc[date, reserves_col] - df.loc[prev_date, reserves_col]
    delta_tga = df.loc[date, tga_col] - df.loc[prev_date, tga_col]
    delta_rrp = df.loc[date, rrp_col] - df.loc[prev_date, rrp_col]

    # NL = Reserves - TGA - RRP
    # ΔNL = ΔReserves - ΔTGA - ΔRRP
    delta_nl = delta_reserves - delta_tga - delta_rrp

    return {
        "delta_reserves": delta_reserves,
        "delta_tga": -delta_tga,  # Negative because it's subtracted
        "delta_rrp": -delta_rrp,  # Negative because it's subtracted
        "delta_nl": delta_nl,
        "pct_reserves": (delta_reserves / delta_nl * 100) if delta_nl != 0 else np.nan,
        "pct_tga": (-delta_tga / delta_nl * 100) if delta_nl != 0 else np.nan,
        "pct_rrp": (-delta_rrp / delta_nl * 100) if delta_nl != 0 else np.nan,
    }


def net_liquidity_velocity(
    nl: pd.Series,
    market_indicator: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Estimate 'velocity' of Net Liquidity impact on markets.

    Velocity = correlation(ΔNL, ΔMarket) over rolling window.

    Parameters
    ----------
    nl : pd.Series
        Net Liquidity series
    market_indicator : pd.Series
        Market indicator (e.g., SPX)
    window : int
        Rolling window

    Returns
    -------
    pd.Series
        Velocity (correlation)
    """
    delta_nl = nl.pct_change()
    delta_market = market_indicator.pct_change()

    velocity = delta_nl.rolling(window).corr(delta_market)
    velocity.name = "nl_velocity"
    return velocity


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    np.random.seed(42)

    df = pd.DataFrame({
        "RESERVES": 3000 + np.cumsum(np.random.randn(len(dates)) * 10),
        "TGA": 500 + np.cumsum(np.random.randn(len(dates)) * 5),
        "RRP": 1000 + np.cumsum(np.random.randn(len(dates)) * 8),
    }, index=dates)

    # Compute NL components
    nl_df = compute_net_liquidity_components(df)
    print("Net Liquidity Components:")
    print(nl_df.tail())

    # Stress score
    stress = net_liquidity_stress_score(nl_df["net_liquidity"])
    print(f"\nCurrent stress score: {stress.iloc[-1]:.2f}")

    # Regime
    regime = net_liquidity_regime(nl_df["net_liquidity"])
    print(f"Current regime: {regime.iloc[-1]}")

    # Decomposition
    decomp = decompose_nl_change(df, dates[-1], periods=5)
    print(f"\n5-day decomposition:")
    for k, v in decomp.items():
        print(f"  {k}: {v:.2f}")
