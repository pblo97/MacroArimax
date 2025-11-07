"""
fred_client.py
FRED data fetcher with incremental cache for liquidity stress detection.
"""

import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import yaml
from fredapi import Fred


class FREDClient:
    """
    FRED data client with intelligent caching and incremental updates.
    """

    def __init__(
        self,
        api_key: str,
        config_path: Optional[str] = None,
        cache_dir: str = ".fred_cache",
        cache_days: int = 30,
    ):
        """
        Initialize FRED client.

        Parameters
        ----------
        api_key : str
            FRED API key
        config_path : str, optional
            Path to series_map.yaml configuration file
        cache_dir : str
            Directory for caching data
        cache_days : int
            Days to keep cache valid
        """
        self.fred = Fred(api_key=api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_days = cache_days

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "series_map.yaml"
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self._series_map = self._build_series_map()

    def _build_series_map(self) -> Dict[str, Dict]:
        """Build flat series map from configuration."""
        series_map = {}
        for category in ["core_plumbing", "stress_indicators", "reference_rates", "market_indicators"]:
            if category in self.config:
                for name, info in self.config[category].items():
                    series_map[name] = info
        return series_map

    def _get_cache_path(self, series_name: str) -> Path:
        """Get cache file path for a series."""
        return self.cache_dir / f"{series_name}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid based on modification time."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return (datetime.now() - mtime).days < self.cache_days

    def _load_from_cache(self, series_name: str) -> Optional[pd.Series]:
        """Load series from cache if valid."""
        cache_path = self._get_cache_path(series_name)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, series_name: str, data: pd.Series):
        """Save series to cache."""
        cache_path = self._get_cache_path(series_name)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def fetch_series(
        self,
        series_name: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Fetch a single series from FRED with incremental caching.

        Parameters
        ----------
        series_name : str
            Name of series from configuration
        start_date : str or pd.Timestamp, optional
            Start date for data
        use_cache : bool
            Whether to use cache

        Returns
        -------
        pd.Series
            Time series data
        """
        if series_name not in self._series_map:
            raise ValueError(f"Unknown series: {series_name}")

        info = self._series_map[series_name]
        fred_code = info["code"]

        # Try cache first
        cached_data = None
        if use_cache:
            cached_data = self._load_from_cache(series_name)

        if cached_data is not None:
            # Incremental fetch: get new data since cache
            last_date = cached_data.index[-1]
            try:
                # Fetch last N days to ensure overlap
                fetch_start = last_date - timedelta(days=5)
                new_data = self.fred.get_series(fred_code, observation_start=fetch_start)
                # Merge with cache
                combined = pd.concat([cached_data, new_data])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                # Save updated cache
                self._save_to_cache(series_name, combined)
                return combined
            except Exception as e:
                # If incremental fails, return cached data
                print(f"Warning: Incremental fetch failed for {series_name}: {e}")
                return cached_data
        else:
            # Full fetch
            try:
                data = self.fred.get_series(
                    fred_code,
                    observation_start=start_date if start_date else self.config["processing"]["default_start_date"],
                )
                data.name = series_name
                # Save to cache
                if use_cache:
                    self._save_to_cache(series_name, data)
                return data
            except Exception as e:
                print(f"Error fetching {series_name}: {e}")
                return pd.Series(dtype=float, name=series_name)

    def fetch_all(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        use_cache: bool = True,
        include_optional: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch all configured series.

        Parameters
        ----------
        start_date : str or pd.Timestamp, optional
            Start date for data
        use_cache : bool
            Whether to use cache
        include_optional : bool
            Whether to include optional series

        Returns
        -------
        pd.DataFrame
            DataFrame with all series
        """
        all_series = {}

        for series_name, info in self._series_map.items():
            # Skip optional series if not requested
            if info.get("optional", False) and not include_optional:
                continue

            print(f"Fetching {series_name}...")
            series = self.fetch_series(series_name, start_date, use_cache)
            if not series.empty:
                all_series[series_name] = series

        # Combine into DataFrame
        df = pd.DataFrame(all_series)

        # Ensure we have a DatetimeIndex (handle case where all fetches failed)
        if len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
            # Create empty DataFrame with DatetimeIndex
            df = pd.DataFrame(index=pd.date_range(start=start_date or "2015-01-01", periods=0))

        df.index.name = "date"

        # Resample to daily and forward fill (only if we have data)
        if len(df) > 0:
            df = df.resample("D").last().ffill()

        return df

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features (spreads, net liquidity, etc.).

        Parameters
        ----------
        df : pd.DataFrame
            Raw data from FRED

        Returns
        -------
        pd.DataFrame
            DataFrame with derived features added
        """
        df = df.copy()

        # === ORIGINAL SPREADS ===
        if "SOFR" in df.columns and "EFFR" in df.columns:
            df["sofr_effr_spread"] = df["SOFR"] - df["EFFR"]

        if "OBFR" in df.columns and "SOFR" in df.columns:
            df["obfr_sofr_spread"] = df["OBFR"] - df["SOFR"]

        if "TGCR" in df.columns and "SOFR" in df.columns:
            df["tgcr_sofr_spread"] = df["TGCR"] - df["SOFR"]

        # === TIER 1: COMMERCIAL PAPER SPREADS ===
        if "CP_FINANCIAL_3M" in df.columns and "TB3MS" in df.columns:
            df["cp_tbill_spread"] = df["CP_FINANCIAL_3M"] - df["TB3MS"]

        if "CP_NONFINANCIAL_3M" in df.columns and "TB3MS" in df.columns:
            df["cp_nonfinancial_spread"] = df["CP_NONFINANCIAL_3M"] - df["TB3MS"]

        # === TIER 1: DISCOUNT WINDOW ALARM ===
        if "DISCOUNT_WINDOW" in df.columns:
            df["discount_window_alarm"] = (df["DISCOUNT_WINDOW"] > 5000).astype(int)

        # === TIER 1: REPO SPREADS ===
        if "BGCR" in df.columns and "SOFR" in df.columns:
            df["bgcr_sofr_spread"] = df["BGCR"] - df["SOFR"]

        if "SOFR_90D_AVG" in df.columns and "SOFR" in df.columns:
            df["sofr_term_premium"] = df["SOFR_90D_AVG"] - df["SOFR"]

        # === TIER 1: CREDIT SPREADS ===
        if "CORP_BBB_OAS" in df.columns and "CORP_AAA_OAS" in df.columns:
            df["bbb_aaa_spread"] = df["CORP_BBB_OAS"] - df["CORP_AAA_OAS"]

        # === TIER 1: DOLLAR STRENGTH ===
        if "DOLLAR_INDEX" in df.columns:
            df["dollar_strength_zscore"] = zscore_rolling(df["DOLLAR_INDEX"], window=252)

        # === NET LIQUIDITY ===
        if all(col in df.columns for col in ["RESERVES", "TGA", "RRP"]):
            df["net_liquidity"] = df["RESERVES"] - df["TGA"] - df["RRP"]

        # === DELTAS ===
        for col in ["RRP", "TGA", "RESERVES", "net_liquidity"]:
            if col in df.columns:
                df[f"delta_{col.lower()}"] = df[col].diff()

        # === TIER 2: CREDIT CURVE SPREADS ===
        if "CORP_BB_OAS" in df.columns and "CORP_BBB_OAS" in df.columns:
            df["bb_bbb_spread"] = df["CORP_BB_OAS"] - df["CORP_BBB_OAS"]

        if "CORP_CCC_OAS" in df.columns and "CORP_BB_OAS" in df.columns:
            df["ccc_bb_spread"] = df["CORP_CCC_OAS"] - df["CORP_BB_OAS"]

        if "CORP_CCC_OAS" in df.columns and "CORP_AAA_OAS" in df.columns:
            df["credit_cascade"] = df["CORP_CCC_OAS"] - df["CORP_AAA_OAS"]

        # === TIER 2: REAL RATES ===
        if "YIELD_5Y" in df.columns and "BREAKEVEN_5Y" in df.columns:
            df["real_rate_5y"] = df["YIELD_5Y"] - df["BREAKEVEN_5Y"]

        if "DGS10" in df.columns and "BREAKEVEN_10Y" in df.columns:
            df["real_rate_10y"] = df["DGS10"] - df["BREAKEVEN_10Y"]

        if "BREAKEVEN_10Y" in df.columns and "BREAKEVEN_5Y" in df.columns:
            df["breakeven_slope"] = df["BREAKEVEN_10Y"] - df["BREAKEVEN_5Y"]

        # === TIER 2: TERM STRUCTURE ===
        if "YIELD_30Y" in df.columns and "DGS10" in df.columns:
            df["term_spread_10y30y"] = df["YIELD_30Y"] - df["DGS10"]

        if "DGS10" in df.columns and "YIELD_5Y" in df.columns:
            df["term_spread_5y10y"] = df["DGS10"] - df["YIELD_5Y"]

        if "YIELD_5Y" in df.columns and "YIELD_2Y" in df.columns:
            df["term_spread_2y5y"] = df["YIELD_5Y"] - df["YIELD_2Y"]

        # === TIER 2: BANK CREDIT DELTAS ===
        if "TOTAL_BANK_CREDIT" in df.columns:
            df["delta_bank_credit"] = df["TOTAL_BANK_CREDIT"].diff()

        if "CI_LOANS" in df.columns:
            df["delta_ci_loans"] = df["CI_LOANS"].diff()

        # === TIER 2: LABOR MARKET STRESS ===
        if "JOBLESS_CLAIMS" in df.columns:
            df["jobless_claims_zscore"] = zscore_rolling(df["JOBLESS_CLAIMS"], window=52)

        # === TIER 3: SAFE HAVEN & VOLATILITY ===
        if "GOLD_PRICE" in df.columns:
            df["gold_zscore"] = zscore_rolling(df["GOLD_PRICE"], window=252)

        if "WTI_OIL" in df.columns:
            df["oil_zscore"] = zscore_rolling(df["WTI_OIL"], window=252)

        if "VIX" in df.columns:
            df["vix_alarm"] = (df["VIX"] > 30).astype(int)

        # === TIER 3: INTERNATIONAL SPREADS ===
        if "EURIBOR_3M" in df.columns and "SOFR" in df.columns:
            df["euribor_ois_proxy"] = df["EURIBOR_3M"] - df["SOFR"]

        if "DGS10" in df.columns and "JAPAN_10Y" in df.columns:
            df["us_japan_spread"] = df["DGS10"] - df["JAPAN_10Y"]

        # === TIER 3: LABOR MARKET DETAIL ===
        if "CONTINUED_CLAIMS" in df.columns:
            df["continued_claims_zscore"] = zscore_rolling(df["CONTINUED_CLAIMS"], window=52)

        if all(col in df.columns for col in ["UNEMPLOYMENT_RATE", "PART_TIME_ECONOMIC", "LABOR_PARTICIPATION"]):
            # U-6 style underemployment proxy
            df["labor_slack"] = df["UNEMPLOYMENT_RATE"] + (df["PART_TIME_ECONOMIC"] / df["LABOR_PARTICIPATION"] * 100)

        # === TIER 3: HOUSING MARKET ===
        if "MORTGAGE_30Y" in df.columns and "DGS10" in df.columns:
            df["mortgage_spread"] = df["MORTGAGE_30Y"] - df["DGS10"]

        if "HOUSING_STARTS" in df.columns:
            df["housing_momentum"] = df["HOUSING_STARTS"].diff()

        # === TIER 3: CONSUMER CREDIT ===
        if "CONSUMER_CREDIT" in df.columns:
            df["consumer_credit_growth"] = df["CONSUMER_CREDIT"].pct_change(periods=12) * 100  # YoY %

        if "CREDIT_CARD_DELINQ" in df.columns and "AUTO_LOAN_DELINQ" in df.columns:
            df["delinquency_index"] = (df["CREDIT_CARD_DELINQ"] + df["AUTO_LOAN_DELINQ"]) / 2

        # === TIER 3: REAL ECONOMY ===
        if "RETAIL_SALES" in df.columns:
            df["retail_sales_momentum"] = df["RETAIL_SALES"].pct_change(periods=3) * 100  # 3-month %

        if "CAPACITY_UTILIZATION" in df.columns:
            df["capacity_gap"] = 85 - df["CAPACITY_UTILIZATION"]  # Slack (85% baseline)

        # === CALENDAR FLAGS ===
        df["month_end"] = df.index.is_month_end
        df["quarter_end"] = df.index.is_quarter_end
        df["year_end"] = (df.index.month == 12) & (df.index.day == 31)

        return df

    def clear_cache(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print(f"Cache cleared: {self.cache_dir}")


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    """
    Winsorize series at specified percentile.

    Parameters
    ----------
    s : pd.Series
        Input series
    p : float
        Percentile (0.01 = 1%)

    Returns
    -------
    pd.Series
        Winsorized series
    """
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def zscore_rolling(s: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute rolling z-score for a series.

    Parameters
    ----------
    s : pd.Series
        Input series
    window : int
        Rolling window size (default: 252 = 1 year of trading days)

    Returns
    -------
    pd.Series
        Rolling z-score of series
    """
    if s.dropna().empty or len(s) < window:
        return pd.Series(0, index=s.index, name=f"{s.name}_zscore")

    rolling_mean = s.rolling(window=window, min_periods=window//2).mean()
    rolling_std = s.rolling(window=window, min_periods=window//2).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, 1)

    zscore = (s - rolling_mean) / rolling_std
    zscore.name = f"{s.name}_zscore"

    return zscore


# Example usage
if __name__ == "__main__":
    import os

    # Example: fetch data
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        print("Set FRED_API_KEY environment variable")
    else:
        client = FREDClient(api_key=api_key)
        df = client.fetch_all(start_date="2020-01-01")
        df = client.compute_derived_features(df)
        print(df.tail())
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
