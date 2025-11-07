"""
Master Data Scraper & Orchestrator
===================================

Central coordinator for all data sources:
- FRED API (existing)
- FRBNY scrapers (dealer leverage, repo, SOMA)
- ECB/BIS scrapers (FX basis, credit gaps)
- ICI scrapers (MMF flows)
- Calculated proxies (VRP, convenience yield)

Features:
- Parallel execution across all sources
- Intelligent fallbacks
- Unified caching
- Data validation
- Automatic retry
- Progress reporting

Author: MacroArimax Enhancement
Date: 2025-11-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import warnings

from .scraping_infrastructure import (
    ParallelScraper,
    cache,
    logger,
    DataValidator
)

# Import all scrapers
from .fred_client import FREDClient
from .frbny_scrapers import (
    fetch_dealer_leverage,
    fetch_triparty_repo,
    fetch_soma_holdings,
    fetch_reference_rates_with_dispersion
)
from .international_scrapers import (
    fetch_ecb_fx_basis,
    fetch_estr,
    fetch_bis_credit_gaps,
    fetch_ici_mmf_flows,
    compute_variance_risk_premium,
    compute_convenience_yield
)

warnings.filterwarnings('ignore')


class MasterDataFetcher:
    """
    Central orchestrator for all data fetching.

    Usage:
        fetcher = MasterDataFetcher()
        data = fetcher.fetch_all()
        df = fetcher.to_dataframe(data)
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        use_cache: bool = True,
        parallel: bool = True,
        max_workers: int = 6
    ):
        """
        Parameters
        ----------
        fred_api_key : Optional[str]
            FRED API key (if None, reads from secrets)
        use_cache : bool
            Whether to use cached data
        parallel : bool
            Whether to fetch sources in parallel
        max_workers : int
            Number of parallel workers
        """
        self.fred_api_key = fred_api_key
        self.use_cache = use_cache
        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize FRED client
        self.fred = FREDClient(api_key=fred_api_key)

        # Initialize parallel scraper
        if parallel:
            self.scraper = ParallelScraper(
                max_workers=max_workers,
                cache_ttl_hours=24
            )

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all data sources.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with all fetched data:
            {
                'fred_core': DataFrame with FRED series,
                'dealer_leverage': DataFrame,
                'triparty_repo': DataFrame,
                'soma_holdings': DataFrame,
                'reference_rates': DataFrame,
                'ecb_fx_basis': DataFrame,
                'estr': DataFrame,
                'bis_credit_gaps': DataFrame,
                'ici_mmf_flows': DataFrame,
                'vrp': DataFrame,
                'convenience_yield': DataFrame
            }
        """
        logger.info("="*80)
        logger.info("MASTER DATA FETCH INITIATED")
        logger.info("="*80)

        results = {}

        # 1. Fetch FRED data (core series)
        logger.info("\n[1/3] Fetching FRED core series...")
        results['fred_core'] = self._fetch_fred_core()

        # 2. Fetch scraped data (FRBNY, ECB, BIS, ICI)
        if self.parallel:
            logger.info("\n[2/3] Fetching scraped data (parallel)...")
            results.update(self._fetch_scraped_parallel())
        else:
            logger.info("\n[2/3] Fetching scraped data (sequential)...")
            results.update(self._fetch_scraped_sequential())

        # 3. Compute derived features
        logger.info("\n[3/3] Computing derived features...")
        results.update(self._compute_derived(results))

        # Log summary
        logger.info("\n" + "="*80)
        logger.info("FETCH SUMMARY")
        logger.info("="*80)
        for name, df in results.items():
            if df is not None:
                logger.info(f"  ✅ {name:30s} {len(df):6d} rows  (Latest: {df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'})")
            else:
                logger.info(f"  ❌ {name:30s} FAILED")

        return results

    def _fetch_fred_core(self) -> pd.DataFrame:
        """Fetch core FRED series."""
        try:
            df = self.fred.fetch_all()
            if df is not None and len(df) > 0:
                logger.info(f"  ✅ FRED: {len(df)} rows")
                return df
            else:
                logger.warning("  ⚠️  FRED returned empty data")
                return None
        except Exception as e:
            logger.error(f"  ❌ FRED failed: {e}")
            return None

    def _fetch_scraped_parallel(self) -> Dict[str, pd.DataFrame]:
        """Fetch scraped data in parallel."""
        # Pass fred_api_key to functions that need it for proxies
        api_key = self.fred_api_key

        tasks = [
            ('dealer_leverage', lambda: fetch_dealer_leverage(fred_api_key=api_key)),
            ('triparty_repo', fetch_triparty_repo),
            ('soma_holdings', fetch_soma_holdings),
            ('reference_rates', fetch_reference_rates_with_dispersion),
            ('ecb_fx_basis', lambda: fetch_ecb_fx_basis(fred_api_key=api_key)),
            ('estr', fetch_estr),
            ('bis_credit_gaps', fetch_bis_credit_gaps),
            ('ici_mmf_flows', lambda: fetch_ici_mmf_flows(fred_api_key=api_key)),
            ('vrp', lambda: compute_variance_risk_premium(fred_api_key=api_key)),
            ('convenience_yield', lambda: compute_convenience_yield(fred_api_key=api_key))
        ]

        return self.scraper.scrape_all(tasks, use_cache=self.use_cache)

    def _fetch_scraped_sequential(self) -> Dict[str, pd.DataFrame]:
        """Fetch scraped data sequentially."""
        results = {}

        # Get API key for proxy functions
        api_key = self.fred_api_key

        fetchers = {
            'dealer_leverage': lambda: fetch_dealer_leverage(fred_api_key=api_key),
            'triparty_repo': fetch_triparty_repo,
            'soma_holdings': fetch_soma_holdings,
            'reference_rates': fetch_reference_rates_with_dispersion,
            'ecb_fx_basis': lambda: fetch_ecb_fx_basis(fred_api_key=api_key),
            'estr': fetch_estr,
            'bis_credit_gaps': fetch_bis_credit_gaps,
            'ici_mmf_flows': lambda: fetch_ici_mmf_flows(fred_api_key=api_key),
            'vrp': lambda: compute_variance_risk_premium(fred_api_key=api_key),
            'convenience_yield': lambda: compute_convenience_yield(fred_api_key=api_key)
        }

        for name, func in fetchers.items():
            try:
                logger.info(f"  Fetching {name}...")
                results[name] = func()
            except Exception as e:
                logger.error(f"  ❌ {name} failed: {e}")
                results[name] = None

        return results

    def _compute_derived(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Compute derived features from fetched data."""
        derived = {}

        try:
            fred_core = data.get('fred_core')
            if fred_core is None:
                return derived

            # Spreads
            if 'SOFR' in fred_core.columns and 'EFFR' in fred_core.columns:
                derived['sofr_effr_spread'] = pd.DataFrame({
                    'spread': fred_core['SOFR'] - fred_core['EFFR']
                })

            # IOR spreads (if IOR was fetched)
            if 'IORB' in fred_core.columns:
                if 'EFFR' in fred_core.columns:
                    derived['effr_ior_spread'] = pd.DataFrame({
                        'spread': fred_core['EFFR'] - fred_core['IORB']
                    })

                if 'RRPONTSYD' in fred_core.columns:
                    derived['rrp_ior_spread'] = pd.DataFrame({
                        'spread': fred_core['RRPONTSYD'] - fred_core['IORB']
                    })

            # Commercial Paper spread
            if 'CPF3M' in fred_core.columns and 'SOFR' in fred_core.columns:
                derived['cp_financial_spread'] = pd.DataFrame({
                    'spread': fred_core['CPF3M'] - fred_core['SOFR']
                })

            logger.info(f"  ✅ Computed {len(derived)} derived features")

        except Exception as e:
            logger.error(f"  ❌ Derived computation failed: {e}")

        return derived

    def to_dataframe(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all data sources into single DataFrame.

        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Output from fetch_all()

        Returns
        -------
        pd.DataFrame
            Unified DataFrame with all series
        """
        logger.info("Merging all data sources...")

        df_merged = None

        for name, df in data.items():
            if df is None:
                continue

            if df_merged is None:
                df_merged = df
            else:
                # Join on date index
                df_merged = df_merged.join(df, how='outer', rsuffix=f'_{name}')

        if df_merged is not None:
            # Forward fill missing data (max 5 days)
            df_merged = df_merged.fillna(method='ffill', limit=5)

            logger.info(f"✅ Merged data: {len(df_merged)} rows × {len(df_merged.columns)} columns")

        return df_merged

    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get summary statistics for all fetched data.

        Returns
        -------
        pd.DataFrame
            Summary with columns: ['Source', 'Rows', 'Start_Date', 'End_Date', 'Missing_Pct', 'Status']
        """
        summary = []

        for name, df in data.items():
            if df is None:
                summary.append({
                    'Source': name,
                    'Rows': 0,
                    'Start_Date': None,
                    'End_Date': None,
                    'Missing_Pct': 100.0,
                    'Status': '❌ Failed'
                })
            else:
                missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100

                summary.append({
                    'Source': name,
                    'Rows': len(df),
                    'Start_Date': df.index.min(),
                    'End_Date': df.index.max(),
                    'Missing_Pct': missing_pct,
                    'Status': '✅ OK' if missing_pct < 20 else '⚠️  High Missing'
                })

        return pd.DataFrame(summary)

    def close(self):
        """Close connections."""
        if hasattr(self, 'scraper'):
            self.scraper.close()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_fetch(
    fred_api_key: Optional[str] = None,
    use_cache: bool = True,
    parallel: bool = True
) -> pd.DataFrame:
    """
    Quick fetch and merge all data.

    Returns
    -------
    pd.DataFrame
        Unified DataFrame with all series (including derived features)
    """
    fetcher = MasterDataFetcher(
        fred_api_key=fred_api_key,
        use_cache=use_cache,
        parallel=parallel
    )

    data = fetcher.fetch_all()
    df = fetcher.to_dataframe(data)

    # Compute derived features (spreads, net_liquidity, deltas, calendar flags)
    df = fetcher.fred.compute_derived_features(df)

    fetcher.close()

    return df


def fetch_with_summary(
    fred_api_key: Optional[str] = None,
    use_cache: bool = True
) -> tuple:
    """
    Fetch data and return with summary.

    Returns
    -------
    tuple
        (data_dict, summary_dataframe)
    """
    fetcher = MasterDataFetcher(fred_api_key=fred_api_key, use_cache=use_cache)

    data = fetcher.fetch_all()
    summary = fetcher.get_data_summary(data)
    fetcher.close()

    return data, summary


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MASTER DATA FETCHER TEST")
    print("="*80)

    # Test quick fetch
    print("\nTesting quick_fetch()...")
    df = quick_fetch(use_cache=True, parallel=True)

    if df is not None:
        print(f"\n✅ Success: {len(df)} rows × {len(df.columns)} columns")
        print("\nFirst 5 columns:")
        print(df.iloc[:5, :5])
        print("\nLast 5 rows:")
        print(df.tail())
    else:
        print("\n❌ Failed")

    # Test with summary
    print("\n" + "="*80)
    print("Testing fetch_with_summary()...")
    data, summary = fetch_with_summary(use_cache=True)

    print("\nData Summary:")
    print(summary.to_string(index=False))

    print("\n✅ Master Data Fetcher test complete!")
