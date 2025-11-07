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

        # Spreads
        if "SOFR" in df.columns and "EFFR" in df.columns:
            df["sofr_effr_spread"] = df["SOFR"] - df["EFFR"]

        if "OBFR" in df.columns and "SOFR" in df.columns:
            df["obfr_sofr_spread"] = df["OBFR"] - df["SOFR"]

        if "TGCR" in df.columns and "SOFR" in df.columns:
            df["tgcr_sofr_spread"] = df["TGCR"] - df["SOFR"]

        # Net Liquidity
        if all(col in df.columns for col in ["RESERVES", "TGA", "RRP"]):
            df["net_liquidity"] = df["RESERVES"] - df["TGA"] - df["RRP"]

        # Deltas
        for col in ["RRP", "TGA", "RESERVES", "net_liquidity"]:
            if col in df.columns:
                df[f"delta_{col.lower()}"] = df[col].diff()

        # Calendar flags
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
