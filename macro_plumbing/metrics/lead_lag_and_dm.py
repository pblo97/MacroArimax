"""
Lead-Lag Analysis & Diebold-Mariano Tests
==========================================

Complete lead-lag correlation matrix (-10...+10 days) between signals and targets.
Rolling Diebold-Mariano tests to validate forecast superiority vs baseline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')


def compute_lead_lag_matrix(
    signals: pd.DataFrame,
    targets: pd.DataFrame,
    max_lag: int = 10,
    method: str = 'spearman'
) -> pd.DataFrame:
    """
    Compute lead-lag correlation matrix between signals and targets.

    Parameters
    ----------
    signals : pd.DataFrame
        Signal series (e.g., Stress Score, SFI, drivers)
    targets : pd.DataFrame
        Target series (e.g., ΔHY_OAS, SPX_ER, ΔVIX)
    max_lag : int
        Maximum lag in days (default 10)
    method : str
        'spearman' or 'pearson'

    Returns
    -------
    pd.DataFrame
        Matrix with signals as rows, targets as columns,
        values are (best_lag, best_corr, p_value)
    """
    # Align indices
    common_idx = signals.index.intersection(targets.index)
    signals = signals.loc[common_idx]
    targets = targets.loc[common_idx]

    results = []

    for signal_name in signals.columns:
        for target_name in targets.columns:
            signal = signals[signal_name].dropna()
            target = targets[target_name].dropna()

            # Compute correlations for all lags
            corrs = []
            pvals = []
            lags = range(-max_lag, max_lag + 1)

            for lag in lags:
                if lag == 0:
                    s = signal
                    t = target
                elif lag > 0:
                    # Positive lag: signal leads target by lag days
                    s = signal.shift(lag)
                    t = target
                else:
                    # Negative lag: target leads signal
                    s = signal
                    t = target.shift(-lag)

                # Align
                common = s.index.intersection(t.index)
                if len(common) < 30:
                    corrs.append(np.nan)
                    pvals.append(np.nan)
                    continue

                s_aligned = s.loc[common].values
                t_aligned = t.loc[common].values

                if method == 'spearman':
                    corr, pval = spearmanr(s_aligned, t_aligned, nan_policy='omit')
                else:
                    corr, pval = stats.pearsonr(s_aligned, t_aligned)

                corrs.append(corr)
                pvals.append(pval)

            # Find best lag
            corrs_arr = np.array(corrs)
            if not np.all(np.isnan(corrs_arr)):
                abs_corrs = np.abs(corrs_arr)
                best_idx = np.nanargmax(abs_corrs)
                best_lag = lags[best_idx]
                best_corr = corrs_arr[best_idx]
                best_pval = pvals[best_idx]
            else:
                best_lag = 0
                best_corr = np.nan
                best_pval = np.nan

            results.append({
                'Signal': signal_name,
                'Target': target_name,
                'Best_Lag': best_lag,
                'Best_Corr': best_corr,
                'P_Value': best_pval,
                'All_Lags': dict(zip(lags, corrs)),
                'All_PVals': dict(zip(lags, pvals))
            })

    return pd.DataFrame(results)


def compute_lead_lag_heatmap(
    signals: pd.DataFrame,
    targets: pd.DataFrame,
    max_lag: int = 10,
    method: str = 'spearman'
) -> Dict[str, pd.DataFrame]:
    """
    Compute full heatmap of correlations for all lags.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with target names as keys, each containing a DataFrame
        with signals as rows and lags as columns.
    """
    common_idx = signals.index.intersection(targets.index)
    signals = signals.loc[common_idx]
    targets = targets.loc[common_idx]

    lags = list(range(-max_lag, max_lag + 1))
    heatmaps = {}

    for target_name in targets.columns:
        target = targets[target_name].dropna()
        corr_matrix = []

        for signal_name in signals.columns:
            signal = signals[signal_name].dropna()
            row = []

            for lag in lags:
                if lag == 0:
                    s = signal
                    t = target
                elif lag > 0:
                    s = signal.shift(lag)
                    t = target
                else:
                    s = signal
                    t = target.shift(-lag)

                common = s.index.intersection(t.index)
                if len(common) < 30:
                    row.append(np.nan)
                    continue

                s_aligned = s.loc[common].values
                t_aligned = t.loc[common].values

                if method == 'spearman':
                    corr, _ = spearmanr(s_aligned, t_aligned, nan_policy='omit')
                else:
                    corr, _ = stats.pearsonr(s_aligned, t_aligned)

                row.append(corr)

            corr_matrix.append(row)

        heatmaps[target_name] = pd.DataFrame(
            corr_matrix,
            index=signals.columns,
            columns=[f"Lag_{lag:+d}" for lag in lags]
        )

    return heatmaps


def diebold_mariano_test(
    forecast_errors_1: np.ndarray,
    forecast_errors_2: np.ndarray,
    h: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for forecast comparison.

    H0: Both forecasts have equal accuracy
    H1: Forecast 1 is more accurate than Forecast 2

    Parameters
    ----------
    forecast_errors_1 : np.ndarray
        Forecast errors from model 1 (e.g., your stress score)
    forecast_errors_2 : np.ndarray
        Forecast errors from model 2 (e.g., baseline)
    h : int
        Forecast horizon (for HAC correction)

    Returns
    -------
    dm_stat : float
        DM test statistic
    p_value : float
        One-sided p-value (H1: model 1 better)
    """
    # Loss differential (MSE)
    d = forecast_errors_1**2 - forecast_errors_2**2

    # Mean loss differential
    d_bar = np.mean(d)

    # Variance with HAC correction (Newey-West)
    n = len(d)
    gamma_0 = np.var(d, ddof=1)

    if h > 1:
        # Add autocorrelation terms
        gamma_sum = 0
        for lag in range(1, h):
            gamma_lag = np.cov(d[:-lag], d[lag:])[0, 1]
            gamma_sum += 2 * gamma_lag

        var_d = (gamma_0 + gamma_sum) / n
    else:
        var_d = gamma_0 / n

    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d) if var_d > 0 else 0

    # P-value (one-sided: DM < 0 means model 1 is better)
    p_value = stats.norm.cdf(dm_stat)

    return dm_stat, p_value


def rolling_diebold_mariano(
    y_true: pd.Series,
    forecast_1: pd.Series,
    forecast_2: pd.Series,
    window: int = 60,
    h: int = 1
) -> pd.DataFrame:
    """
    Rolling Diebold-Mariano test.

    Parameters
    ----------
    y_true : pd.Series
        Actual values
    forecast_1 : pd.Series
        Forecasts from model 1 (your model)
    forecast_2 : pd.Series
        Forecasts from model 2 (baseline)
    window : int
        Rolling window size in days
    h : int
        Forecast horizon

    Returns
    -------
    pd.DataFrame
        Rolling DM statistics and p-values
    """
    # Align
    common_idx = y_true.index.intersection(forecast_1.index).intersection(forecast_2.index)
    y_true = y_true.loc[common_idx]
    forecast_1 = forecast_1.loc[common_idx]
    forecast_2 = forecast_2.loc[common_idx]

    # Compute errors
    errors_1 = (y_true - forecast_1).values
    errors_2 = (y_true - forecast_2).values

    dm_stats = []
    p_values = []
    dates = []

    for i in range(window, len(y_true)):
        e1_window = errors_1[i-window:i]
        e2_window = errors_2[i-window:i]

        dm_stat, p_val = diebold_mariano_test(e1_window, e2_window, h=h)

        dm_stats.append(dm_stat)
        p_values.append(p_val)
        dates.append(y_true.index[i])

    return pd.DataFrame({
        'DM_Stat': dm_stats,
        'P_Value': p_values,
        'Model1_Better': [p < 0.05 for p in p_values]
    }, index=dates)


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.05,
    method: str = 'TR'
) -> List[str]:
    """
    Model Confidence Set (Hansen, Lunde, Nason 2011).

    Simplified implementation using sequential testing.

    Parameters
    ----------
    losses : pd.DataFrame
        Loss matrix with models as columns, time as rows
    alpha : float
        Confidence level (default 0.05)
    method : str
        'TR' (t-range) or 'Tmax' (t-max)

    Returns
    -------
    List[str]
        Model names in the MCS
    """
    models = list(losses.columns)
    n_models = len(models)
    n_obs = len(losses)

    # Mean losses
    mean_losses = losses.mean()

    # Eliminate models sequentially
    surviving = models.copy()

    while len(surviving) > 1:
        # Pairwise differences
        diffs = {}
        for i, m1 in enumerate(surviving):
            for m2 in surviving[i+1:]:
                diff = losses[m1] - losses[m2]
                diffs[(m1, m2)] = diff.mean() / (diff.std() / np.sqrt(n_obs))

        # Find worst model (highest mean loss)
        worst_model = max(surviving, key=lambda m: mean_losses[m])

        # Test if worst model can be eliminated
        t_stats = []
        for m in surviving:
            if m != worst_model:
                diff = losses[worst_model] - losses[m]
                t_stat = diff.mean() / (diff.std() / np.sqrt(n_obs))
                t_stats.append(t_stat)

        # If all t_stats positive and significant, eliminate
        if t_stats and all(t > 1.96 for t in t_stats):  # Simplified critical value
            surviving.remove(worst_model)
        else:
            break

    return surviving


def compute_granger_causality(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 5
) -> pd.DataFrame:
    """
    Granger causality test: does x help predict y?

    Uses F-test comparing AR(p) model of y vs. ARX(p) model with x lags.

    Parameters
    ----------
    x : pd.Series
        Potential causal variable
    y : pd.Series
        Response variable
    max_lag : int
        Maximum lag to test

    Returns
    -------
    pd.DataFrame
        F-statistics and p-values for each lag
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests

        # Align
        common_idx = x.index.intersection(y.index)
        df = pd.DataFrame({'y': y.loc[common_idx], 'x': x.loc[common_idx]}).dropna()

        # Run test
        results = grangercausalitytests(df[['y', 'x']], maxlag=max_lag, verbose=False)

        # Extract results
        output = []
        for lag in range(1, max_lag + 1):
            f_stat = results[lag][0]['ssr_ftest'][0]
            p_value = results[lag][0]['ssr_ftest'][1]
            output.append({
                'Lag': lag,
                'F_Stat': f_stat,
                'P_Value': p_value,
                'Significant': p_value < 0.05
            })

        return pd.DataFrame(output)

    except ImportError:
        # Fallback: simple correlation-based approximation
        output = []
        for lag in range(1, max_lag + 1):
            x_lagged = x.shift(lag)
            common = x_lagged.index.intersection(y.index)
            if len(common) < 30:
                continue

            corr, p_val = spearmanr(
                x_lagged.loc[common].dropna(),
                y.loc[common].dropna(),
                nan_policy='omit'
            )
            output.append({
                'Lag': lag,
                'Correlation': corr,
                'P_Value': p_val,
                'Significant': p_val < 0.05
            })

        return pd.DataFrame(output)
