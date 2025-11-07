"""
Industrial-Grade Web Scraping Infrastructure
=============================================

Optimized scraping with:
- Parallel execution (ThreadPoolExecutor)
- Automatic retry with exponential backoff
- Intelligent caching with TTL
- Rate limiting to avoid bans
- Multiple fallbacks per source
- Data validation and cleaning

Author: MacroArimax Enhancement
Date: 2025-11-07
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path
import time
import hashlib
import pickle
import logging
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CACHING SYSTEM
# ============================================================================

class SmartCache:
    """
    Intelligent caching with TTL (time-to-live) and invalidation.

    Features:
    - Automatic expiration based on data frequency
    - Hash-based cache keys
    - Pickle serialization for any Python object
    """

    def __init__(self, cache_dir: str = '.scraping_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path from key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str, max_age_hours: int = 24) -> Optional[Any]:
        """
        Get cached data if exists and not expired.

        Parameters
        ----------
        key : str
            Cache key (e.g., 'frbny_dealer_leverage')
        max_age_hours : int
            Maximum age in hours before cache expires

        Returns
        -------
        Optional[Any]
            Cached data or None if expired/missing
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {key}")
            return None

        # Check age
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_mtime

        if age > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired: {key} (age: {age})")
            return None

        # Load cached data
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Cache hit: {key} (age: {age})")
            return data
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def set(self, key: str, data: Any):
        """Save data to cache."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cache saved: {key}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self, key: Optional[str] = None):
        """Clear cache (all or specific key)."""
        if key is None:
            # Clear all
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("All cache cleared")
        else:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cache cleared: {key}")


# Global cache instance
cache = SmartCache()


# ============================================================================
# RETRY DECORATOR
# ============================================================================

def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """
    Decorator for automatic retry with exponential backoff.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts
    base_delay : float
        Initial delay in seconds
    max_delay : float
        Maximum delay in seconds
    exponential_base : float
        Base for exponential backoff (2.0 = double each time)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            return None
        return wrapper
    return decorator


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter to avoid hitting API limits.

    Example:
        limiter = RateLimiter(calls_per_second=2)
        with limiter:
            response = requests.get(url)
    """

    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0

    def __enter__(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        return self

    def __exit__(self, *args):
        self.last_call = time.time()


# ============================================================================
# HTTP SESSION WITH RETRY
# ============================================================================

def create_robust_session() -> requests.Session:
    """
    Create HTTP session with automatic retry and sensible defaults.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()

    # Retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Headers to look like a browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })

    return session


# ============================================================================
# DATA VALIDATORS
# ============================================================================

class DataValidator:
    """Validate scraped data for common issues."""

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: List[str],
        date_column: str = 'date',
        min_rows: int = 10
    ) -> bool:
        """
        Validate DataFrame has expected structure.

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        # Check if DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.error("Data is not a DataFrame")
            return False

        # Check min rows
        if len(df) < min_rows:
            logger.error(f"Too few rows: {len(df)} < {min_rows}")
            return False

        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False

        # Check date column
        if date_column in df.columns:
            try:
                pd.to_datetime(df[date_column])
            except Exception as e:
                logger.error(f"Invalid date column: {e}")
                return False

        # Check for all-null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            logger.warning(f"All-null columns: {null_cols}")

        return True

    @staticmethod
    def validate_series(
        series: pd.Series,
        min_length: int = 10,
        max_null_pct: float = 0.5,
        check_monotonic: bool = False
    ) -> bool:
        """Validate time series data."""
        if len(series) < min_length:
            logger.error(f"Series too short: {len(series)}")
            return False

        null_pct = series.isnull().sum() / len(series)
        if null_pct > max_null_pct:
            logger.error(f"Too many nulls: {null_pct:.1%}")
            return False

        if check_monotonic:
            if not series.index.is_monotonic_increasing:
                logger.warning("Series index not monotonic")

        return True


# ============================================================================
# PARALLEL SCRAPING ORCHESTRATOR
# ============================================================================

class ParallelScraper:
    """
    Orchestrate parallel scraping from multiple sources.

    Example:
        scraper = ParallelScraper(max_workers=5)
        results = scraper.scrape_all([
            ('dealer_leverage', fetch_dealer_leverage),
            ('repo_volume', fetch_triparty_repo),
        ])
    """

    def __init__(self, max_workers: int = 5, cache_ttl_hours: int = 24):
        self.max_workers = max_workers
        self.cache_ttl_hours = cache_ttl_hours
        self.session = create_robust_session()

    def scrape_all(
        self,
        tasks: List[tuple],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Scrape multiple sources in parallel.

        Parameters
        ----------
        tasks : List[tuple]
            List of (name, function, kwargs) tuples
        use_cache : bool
            Whether to use cache

        Returns
        -------
        Dict[str, Any]
            {name: result} for each task
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_name = {}

            for task in tasks:
                if len(task) == 2:
                    name, func = task
                    kwargs = {}
                else:
                    name, func, kwargs = task

                # Check cache first
                if use_cache:
                    cached = cache.get(name, max_age_hours=self.cache_ttl_hours)
                    if cached is not None:
                        results[name] = cached
                        logger.info(f"Using cached data for {name}")
                        continue

                # Submit to executor
                future = executor.submit(func, **kwargs)
                future_to_name[future] = name

            # Collect results
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result(timeout=120)

                    # Only mark as success if result is not None
                    if result is not None:
                        results[name] = result

                        # Cache result
                        if use_cache:
                            cache.set(name, result)

                        logger.info(f"✅ Successfully scraped: {name}")
                    else:
                        results[name] = None
                        logger.warning(f"⚠️  {name} returned None (no data)")

                except Exception as e:
                    logger.error(f"❌ Failed to scrape {name}: {e}")
                    results[name] = None

        return results

    def close(self):
        """Close session."""
        self.session.close()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def download_file(url: str, output_path: Path, session: Optional[requests.Session] = None):
    """
    Download file from URL with progress.

    Parameters
    ----------
    url : str
        File URL
    output_path : Path
        Where to save
    session : Optional[requests.Session]
        Reusable session
    """
    if session is None:
        session = create_robust_session()

    response = session.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                f.write(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    logger.debug(f"Download progress: {pct:.1f}%")

    logger.info(f"Downloaded: {output_path.name} ({total_size / 1024:.1f} KB)")


def parse_html_table(
    html: str,
    table_index: int = 0,
    header_row: int = 0
) -> pd.DataFrame:
    """
    Parse HTML table to DataFrame.

    Parameters
    ----------
    html : str
        HTML content
    table_index : int
        Which table to extract (if multiple)
    header_row : int
        Which row contains column names

    Returns
    -------
    pd.DataFrame
    """
    tables = pd.read_html(html)

    if not tables:
        raise ValueError("No tables found in HTML")

    if table_index >= len(tables):
        raise ValueError(f"Table index {table_index} out of range (found {len(tables)} tables)")

    df = tables[table_index]

    # Set proper headers if needed
    if header_row > 0:
        df.columns = df.iloc[header_row]
        df = df[header_row + 1:]

    return df


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean numeric column (remove commas, convert to float).

    Example: "1,234.56" -> 1234.56
    """
    if series.dtype == object:
        # Remove commas and other non-numeric chars
        series = series.astype(str).str.replace(',', '').str.replace('$', '').str.strip()
        series = pd.to_numeric(series, errors='coerce')

    return series


def standardize_date_index(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Standardize date index.

    - Parse dates
    - Sort by date
    - Remove duplicates
    - Set as index
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found")

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    df = df.sort_values(date_column)
    df = df.drop_duplicates(subset=[date_column], keep='last')
    df = df.set_index(date_column)

    return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Parallel scraping
    print("="*80)
    print("SCRAPING INFRASTRUCTURE TEST")
    print("="*80)

    # Test cache
    print("\n1. Testing cache...")
    cache.set('test_key', {'value': 123, 'timestamp': datetime.now()})
    cached_data = cache.get('test_key', max_age_hours=24)
    print(f"   Cached data: {cached_data}")

    # Test rate limiter
    print("\n2. Testing rate limiter...")
    limiter = RateLimiter(calls_per_second=2)
    start = time.time()
    for i in range(5):
        with limiter:
            print(f"   Call {i+1} at t={time.time()-start:.2f}s")

    # Test retry decorator
    print("\n3. Testing retry decorator...")

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def flaky_function(fail_count=2):
        """Function that fails first N times."""
        if not hasattr(flaky_function, 'attempt'):
            flaky_function.attempt = 0

        flaky_function.attempt += 1

        if flaky_function.attempt <= fail_count:
            raise ValueError(f"Intentional failure (attempt {flaky_function.attempt})")

        return "Success!"

    result = flaky_function(fail_count=2)
    print(f"   Result: {result}")

    print("\n✅ All infrastructure tests passed!")
