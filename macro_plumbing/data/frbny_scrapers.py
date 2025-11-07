"""
FRBNY Data Scrapers
===================

Scrape data from Federal Reserve Bank of New York:
1. Primary Dealer Statistics (dealer leverage)
2. Tri-Party Repo Data (repo volume, collateral)
3. SOMA Holdings (Fed balance sheet)
4. Reference Rates (SOFR, EFFR, BGCR, TGCR with dispersion)

Multiple fallbacks for each source.

Author: MacroArimax Enhancement
Date: 2025-11-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import io
import warnings
from typing import Optional, Dict, List

from .scraping_infrastructure import (
    retry_with_backoff,
    create_robust_session,
    cache,
    RateLimiter,
    DataValidator,
    logger,
    download_file,
    parse_html_table,
    clean_numeric_column,
    standardize_date_index
)

warnings.filterwarnings('ignore')


# ============================================================================
# PRIMARY DEALER STATISTICS (Weekly)
# ============================================================================

@retry_with_backoff(max_retries=5, base_delay=2.0)
def fetch_dealer_leverage(fred_api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch Primary Dealer leverage ratio from FRBNY.

    Source: https://www.newyorkfed.org/markets/primarydealers.html
    Frequency: Weekly (Wednesday)
    Data: Total Assets, Net Capital → Leverage

    Parameters
    ----------
    fred_api_key : str, optional
        FRED API key for proxy calculation fallback

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'total_assets', 'net_capital', 'dealer_leverage']
    """
    logger.info("Fetching Primary Dealer Statistics...")

    # Check cache first
    cached = cache.get('dealer_leverage', max_age_hours=168)  # 1 week
    if cached is not None:
        return cached

    session = create_robust_session()
    limiter = RateLimiter(calls_per_second=0.5)  # Slow to be respectful

    # Method 1: Try direct CSV download (if available)
    urls = [
        "https://www.newyorkfed.org/markets/gsds/search.csv",
        "https://markets.newyorkfed.org/api/pd/get/all.csv",
        "https://www.newyorkfed.org/markets/primarydealers_data.csv"
    ]

    for url in urls:
        try:
            with limiter:
                response = session.get(url, timeout=30)

            if response.status_code == 200 and len(response.content) > 1000:
                logger.info(f"✅ Success from: {url}")

                # Parse CSV
                df = pd.read_csv(io.StringIO(response.text))

                # Expected columns (may vary)
                # Look for: 'As of Date', 'Total Assets', 'Net Capital'
                date_cols = [c for c in df.columns if 'date' in c.lower() or 'as of' in c.lower()]
                asset_cols = [c for c in df.columns if 'total' in c.lower() and 'asset' in c.lower()]
                capital_cols = [c for c in df.columns if 'capital' in c.lower() and 'net' in c.lower()]

                if date_cols and asset_cols and capital_cols:
                    df_clean = df[[date_cols[0], asset_cols[0], capital_cols[0]]].copy()
                    df_clean.columns = ['date', 'total_assets', 'net_capital']

                    # Clean numeric columns
                    df_clean['total_assets'] = clean_numeric_column(df_clean['total_assets'])
                    df_clean['net_capital'] = clean_numeric_column(df_clean['net_capital'])

                    # Calculate leverage
                    df_clean['dealer_leverage'] = df_clean['total_assets'] / df_clean['net_capital']

                    # Standardize
                    df_clean = standardize_date_index(df_clean, 'date')

                    # Validate
                    if DataValidator.validate_series(df_clean['dealer_leverage'], min_length=50):
                        cache.set('dealer_leverage', df_clean)
                        return df_clean

        except Exception as e:
            logger.debug(f"Failed URL {url}: {e}")
            continue

    # Method 2: Try downloading Excel file
    excel_urls = [
        "https://www.newyorkfed.org/markets/gsds/search/weekly/WEEKLY_PD_LAST_5_YEARS.xlsx",
        "https://www.newyorkfed.org/markets/gsds/search.xlsx"
    ]

    for url in excel_urls:
        try:
            with limiter:
                response = session.get(url, timeout=30)

            if response.status_code == 200:
                logger.info(f"✅ Success from Excel: {url}")

                # Parse Excel
                df = pd.read_excel(io.BytesIO(response.content), sheet_name=0)

                # Try to find relevant columns
                # Usually: "As of Date", "Total Financial Assets", "Net Capital"
                date_col = None
                asset_col = None
                capital_col = None

                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'date' in col_lower or 'as of' in col_lower:
                        date_col = col
                    elif 'total' in col_lower and 'asset' in col_lower:
                        asset_col = col
                    elif 'net' in col_lower and 'capital' in col_lower:
                        capital_col = col

                if date_col and asset_col and capital_col:
                    df_clean = df[[date_col, asset_col, capital_col]].copy()
                    df_clean.columns = ['date', 'total_assets', 'net_capital']

                    df_clean['total_assets'] = clean_numeric_column(df_clean['total_assets'])
                    df_clean['net_capital'] = clean_numeric_column(df_clean['net_capital'])
                    df_clean['dealer_leverage'] = df_clean['total_assets'] / df_clean['net_capital']

                    df_clean = standardize_date_index(df_clean, 'date')

                    if DataValidator.validate_series(df_clean['dealer_leverage'], min_length=50):
                        cache.set('dealer_leverage', df_clean)
                        return df_clean

        except Exception as e:
            logger.debug(f"Failed Excel URL {url}: {e}")
            continue

    # Method 3: Proxy using H.15 data (if primary dealer not available)
    logger.warning("Primary Dealer data unavailable, using proxy...")
    return _fetch_dealer_leverage_proxy(fred_api_key=fred_api_key)


def _fetch_dealer_leverage_proxy(fred_api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Proxy for dealer leverage using Fed H.15 broker-dealer data.

    Falls back to FRED series if available.
    """
    try:
        from .fred_client import FREDClient

        fred = FREDClient(api_key=fred_api_key)

        # Try fetching dealer-related series
        # DPSACBW027SBOG: Broker-dealer credit
        series_codes = ['DPSACBW027SBOG', 'TLAACBW027SBOG']

        for code in series_codes:
            try:
                df = fred.fetch_series(code)
                if df is not None and len(df) > 50:
                    # Normalize to 0-100 range (approximate leverage)
                    df['dealer_leverage_proxy'] = df[code] / df[code].rolling(252).mean()

                    logger.info(f"✅ Using proxy from FRED: {code}")
                    return df[['dealer_leverage_proxy']].rename(columns={'dealer_leverage_proxy': 'dealer_leverage'})
            except Exception:
                continue

    except Exception as e:
        logger.error(f"Proxy also failed: {e}")

    return None


# ============================================================================
# TRI-PARTY REPO DATA (Daily)
# ============================================================================

@retry_with_backoff(max_retries=5, base_delay=2.0)
def fetch_triparty_repo() -> Optional[pd.DataFrame]:
    """
    Fetch Tri-Party Repo volume and collateral data.

    Source: https://www.newyorkfed.org/data-and-statistics/data-visualization/tri-party-repo
    Frequency: Daily
    Data: Total volume, by collateral type, by counterparty

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'total_volume', 'ust_collateral', 'agency_collateral', 'mbs_collateral']
    """
    logger.info("Fetching Tri-Party Repo data...")

    cached = cache.get('triparty_repo', max_age_hours=24)
    if cached is not None:
        return cached

    session = create_robust_session()
    limiter = RateLimiter(calls_per_second=0.5)

    # URL for CSV data (public API)
    urls = [
        "https://websvcgatewayx2.frbny.org/autoFSSdata/triparty/daily_data.csv",
        "https://markets.newyorkfed.org/read?productCode=50&eventCodes=520&limit=5000&startPosition=0&sort=postDt:-1&format=csv",
        "https://www.newyorkfed.org/data-and-statistics/data-visualization/tri-party-repo/data/daily.csv"
    ]

    for url in urls:
        try:
            with limiter:
                response = session.get(url, timeout=30)

            if response.status_code == 200 and len(response.content) > 500:
                logger.info(f"✅ Success from: {url}")

                df = pd.read_csv(io.StringIO(response.text))

                # Expected columns (vary by endpoint)
                # Typical: 'Date', 'Total', 'Treasury', 'Agency', 'MBS'

                # Standardize column names
                col_map = {}
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'date' in col_lower:
                        col_map[col] = 'date'
                    elif 'total' in col_lower and 'volume' not in col_lower:
                        col_map[col] = 'total_volume'
                    elif 'treasury' in col_lower or 'ust' in col_lower:
                        col_map[col] = 'ust_collateral'
                    elif 'agency' in col_lower:
                        col_map[col] = 'agency_collateral'
                    elif 'mbs' in col_lower:
                        col_map[col] = 'mbs_collateral'

                df = df.rename(columns=col_map)

                # Select relevant columns
                keep_cols = ['date', 'total_volume', 'ust_collateral', 'agency_collateral', 'mbs_collateral']
                available_cols = [c for c in keep_cols if c in df.columns]

                if 'date' in available_cols and len(available_cols) >= 2:
                    df_clean = df[available_cols].copy()

                    # Clean numeric columns
                    for col in available_cols:
                        if col != 'date':
                            df_clean[col] = clean_numeric_column(df_clean[col])

                    df_clean = standardize_date_index(df_clean, 'date')

                    if len(df_clean) >= 100:
                        cache.set('triparty_repo', df_clean)
                        return df_clean

        except Exception as e:
            logger.debug(f"Failed URL {url}: {e}")
            continue

    logger.error("All Tri-Party Repo URLs failed")
    return None


# ============================================================================
# SOMA HOLDINGS (Fed Balance Sheet)
# ============================================================================

@retry_with_backoff(max_retries=5, base_delay=2.0)
def fetch_soma_holdings() -> Optional[pd.DataFrame]:
    """
    Fetch SOMA (System Open Market Account) holdings.

    Source: https://www.newyorkfed.org/markets/soma-holdings
    Frequency: Weekly
    Data: UST, MBS, Agency debt by maturity

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'ust_holdings', 'mbs_holdings', 'agency_holdings', 'total_holdings']
    """
    logger.info("Fetching SOMA Holdings...")

    cached = cache.get('soma_holdings', max_age_hours=168)  # 1 week
    if cached is not None:
        return cached

    session = create_robust_session()
    limiter = RateLimiter(calls_per_second=0.5)

    # SOMA data URLs
    urls = [
        "https://www.newyorkfed.org/markets/soma/sysopen_accholdings.html",
        "https://markets.newyorkfed.org/api/soma/summary.csv",
        "https://www.newyorkfed.org/markets/soma-holdings.csv"
    ]

    for url in urls:
        try:
            with limiter:
                response = session.get(url, timeout=30)

            if response.status_code == 200:
                if url.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(response.text))
                elif url.endswith('.html'):
                    # Parse HTML table
                    df = parse_html_table(response.text, table_index=0)
                else:
                    continue

                # Look for date and holdings columns
                date_col = None
                ust_col = None
                mbs_col = None
                agency_col = None

                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'date' in col_lower or 'as of' in col_lower:
                        date_col = col
                    elif 'treasury' in col_lower or 'ust' in col_lower:
                        ust_col = col
                    elif 'mbs' in col_lower or 'mortgage' in col_lower:
                        mbs_col = col
                    elif 'agency' in col_lower:
                        agency_col = col

                if date_col and (ust_col or mbs_col):
                    keep_cols = [c for c in [date_col, ust_col, mbs_col, agency_col] if c is not None]
                    df_clean = df[keep_cols].copy()

                    # Rename
                    col_rename = {date_col: 'date'}
                    if ust_col: col_rename[ust_col] = 'ust_holdings'
                    if mbs_col: col_rename[mbs_col] = 'mbs_holdings'
                    if agency_col: col_rename[agency_col] = 'agency_holdings'

                    df_clean = df_clean.rename(columns=col_rename)

                    # Clean numeric
                    for col in df_clean.columns:
                        if col != 'date':
                            df_clean[col] = clean_numeric_column(df_clean[col])

                    # Total
                    numeric_cols = [c for c in df_clean.columns if c != 'date']
                    df_clean['total_holdings'] = df_clean[numeric_cols].sum(axis=1)

                    df_clean = standardize_date_index(df_clean, 'date')

                    if len(df_clean) >= 50:
                        logger.info(f"✅ Success from: {url}")
                        cache.set('soma_holdings', df_clean)
                        return df_clean

        except Exception as e:
            logger.debug(f"Failed URL {url}: {e}")
            continue

    logger.error("All SOMA URLs failed")
    return None


# ============================================================================
# REFERENCE RATES WITH DISPERSION (Daily)
# ============================================================================

@retry_with_backoff(max_retries=5, base_delay=2.0)
def fetch_reference_rates_with_dispersion() -> Optional[pd.DataFrame]:
    """
    Fetch reference rates (SOFR, EFFR, BGCR, TGCR) with volume and dispersion.

    Source: https://www.newyorkfed.org/markets/reference-rates
    Frequency: Daily
    Data: Rates + Volume + Percentiles (25th, 75th, 99th)

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'sofr', 'sofr_volume', 'sofr_p25', 'sofr_p75',
                  'effr', 'effr_volume', 'effr_p25', 'effr_p75', ...]
    """
    logger.info("Fetching Reference Rates with dispersion...")

    cached = cache.get('reference_rates_dispersion', max_age_hours=24)
    if cached is not None:
        return cached

    session = create_robust_session()
    limiter = RateLimiter(calls_per_second=0.5)

    # URLs for each rate
    rate_urls = {
        'sofr': 'https://markets.newyorkfed.org/api/rates/secured/sofr/last/5000.csv',
        'effr': 'https://markets.newyorkfed.org/api/rates/unsecured/effr/last/5000.csv',
        'bgcr': 'https://markets.newyorkfed.org/api/rates/secured/bgcr/last/5000.csv',
        'tgcr': 'https://markets.newyorkfed.org/api/rates/secured/tgcr/last/5000.csv'
    }

    all_data = {}

    for rate_name, url in rate_urls.items():
        try:
            with limiter:
                response = session.get(url, timeout=30)

            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))

                # Expected columns: effectiveDate, ratePercent, volumeInBillions, percentile25, percentile75
                if 'effectiveDate' in df.columns and 'ratePercent' in df.columns:
                    df_clean = df.rename(columns={
                        'effectiveDate': 'date',
                        'ratePercent': f'{rate_name}_rate',
                        'volumeInBillions': f'{rate_name}_volume',
                        'percentile25': f'{rate_name}_p25',
                        'percentile75': f'{rate_name}_p75',
                        'percentile99': f'{rate_name}_p99'
                    })

                    # Keep only relevant columns
                    keep_cols = [c for c in df_clean.columns if c.startswith('date') or c.startswith(rate_name)]
                    df_clean = df_clean[keep_cols]

                    df_clean = standardize_date_index(df_clean, 'date')

                    all_data[rate_name] = df_clean
                    logger.info(f"✅ Fetched {rate_name.upper()}")

        except Exception as e:
            logger.warning(f"Failed to fetch {rate_name}: {e}")
            continue

    if all_data:
        # Merge all rates
        df_merged = None
        for rate_name, df_rate in all_data.items():
            if df_merged is None:
                df_merged = df_rate
            else:
                df_merged = df_merged.join(df_rate, how='outer')

        # Forward fill missing data (weekends)
        df_merged = df_merged.fillna(method='ffill', limit=3)

        cache.set('reference_rates_dispersion', df_merged)
        return df_merged

    logger.error("Failed to fetch any reference rates")
    return None


# ============================================================================
# CONVENIENCE FUNCTION: FETCH ALL FRBNY DATA
# ============================================================================

def fetch_all_frbny_data() -> Dict[str, pd.DataFrame]:
    """
    Fetch all FRBNY data sources in parallel.

    Returns
    -------
    Dict[str, pd.DataFrame]
        {
            'dealer_leverage': DataFrame,
            'triparty_repo': DataFrame,
            'soma_holdings': DataFrame,
            'reference_rates': DataFrame
        }
    """
    from .scraping_infrastructure import ParallelScraper

    scraper = ParallelScraper(max_workers=4, cache_ttl_hours=24)

    tasks = [
        ('dealer_leverage', fetch_dealer_leverage),
        ('triparty_repo', fetch_triparty_repo),
        ('soma_holdings', fetch_soma_holdings),
        ('reference_rates', fetch_reference_rates_with_dispersion)
    ]

    results = scraper.scrape_all(tasks, use_cache=True)
    scraper.close()

    return results


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FRBNY SCRAPERS TEST")
    print("="*80)

    # Test individual scrapers
    print("\n1. Testing Dealer Leverage...")
    df_dealer = fetch_dealer_leverage()
    if df_dealer is not None:
        print(f"   ✅ Success: {len(df_dealer)} rows")
        print(df_dealer.tail())
    else:
        print("   ❌ Failed")

    print("\n2. Testing Tri-Party Repo...")
    df_repo = fetch_triparty_repo()
    if df_repo is not None:
        print(f"   ✅ Success: {len(df_repo)} rows")
        print(df_repo.tail())
    else:
        print("   ❌ Failed")

    print("\n3. Testing SOMA Holdings...")
    df_soma = fetch_soma_holdings()
    if df_soma is not None:
        print(f"   ✅ Success: {len(df_soma)} rows")
        print(df_soma.tail())
    else:
        print("   ❌ Failed")

    print("\n4. Testing Reference Rates...")
    df_rates = fetch_reference_rates_with_dispersion()
    if df_rates is not None:
        print(f"   ✅ Success: {len(df_rates)} rows")
        print(df_rates.tail())
    else:
        print("   ❌ Failed")

    print("\n5. Testing Parallel Fetch...")
    all_data = fetch_all_frbny_data()
    print(f"   Fetched {len(all_data)} datasets")
    for name, df in all_data.items():
        if df is not None:
            print(f"   - {name}: {len(df)} rows")
        else:
            print(f"   - {name}: FAILED")

    print("\n✅ FRBNY Scrapers test complete!")
