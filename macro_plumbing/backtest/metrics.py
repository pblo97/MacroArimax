"""
metrics.py
Performance metrics for liquidity stress models: IC, AUROC, Brier, Q4-Q1, etc.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss


def information_coefficient(predictions: pd.Series, actuals: pd.Series) -> float:
    """
    Spearman rank correlation (IC).

    Parameters
    ----------
    predictions : pd.Series
        Predicted values
    actuals : pd.Series
        Actual values

    Returns
    -------
    float
        IC (Spearman correlation)
    """
    df = pd.concat([predictions, actuals], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    ic, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return ic


def auroc(predictions: pd.Series, actuals: pd.Series) -> float:
    """Area Under ROC Curve."""
    df = pd.concat([predictions, actuals], axis=1).dropna()
    if len(df) < 10 or actuals.nunique() < 2:
        return np.nan
    return roc_auc_score(df.iloc[:, 1], df.iloc[:, 0])


def brier_score(predictions: pd.Series, actuals: pd.Series) -> float:
    """Brier score (lower is better)."""
    df = pd.concat([predictions, actuals], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    return brier_score_loss(df.iloc[:, 1], df.iloc[:, 0])


def quantile_spread(
    predictions: pd.Series,
    returns: pd.Series,
    n_quantiles: int = 4,
) -> Dict[str, float]:
    """
    Q4-Q1 spread: return difference between top and bottom quantiles.

    Parameters
    ----------
    predictions : pd.Series
        Signal/predictions
    returns : pd.Series
        Forward returns
    n_quantiles : int
        Number of quantiles

    Returns
    -------
    dict
        Quantile statistics
    """
    df = pd.concat([predictions, returns], axis=1).dropna()
    df.columns = ["signal", "returns"]

    # Assign quantiles
    df["quantile"] = pd.qcut(df["signal"], q=n_quantiles, labels=False, duplicates="drop")

    # Mean returns by quantile
    quantile_rets = df.groupby("quantile")["returns"].mean()

    if len(quantile_rets) < 2:
        return {"Q4_Q1_spread": np.nan, "monotonicity": np.nan}

    # Q4 - Q1 spread
    q_top = quantile_rets.iloc[-1]
    q_bot = quantile_rets.iloc[0]
    spread = q_top - q_bot

    # Monotonicity: correlation of quantile rank with mean return
    mono = quantile_rets.corr(pd.Series(range(len(quantile_rets))))

    return {
        "Q4_Q1_spread": spread,
        "Q4_return": q_top,
        "Q1_return": q_bot,
        "monotonicity": mono,
        "quantile_returns": quantile_rets.to_dict(),
    }


def sharpe_ratio(returns: pd.Series) -> float:
    """Sharpe ratio."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(252)  # Annualized


def sortino_ratio(returns: pd.Series) -> float:
    """Sortino ratio (downside deviation)."""
    r = returns.dropna()
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return np.nan
    return r.mean() / downside.std() * np.sqrt(252)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown."""
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def hit_rate(predictions: pd.Series, actuals: pd.Series, threshold: float = 0.5) -> float:
    """Hit rate (accuracy for binary predictions)."""
    df = pd.concat([predictions, actuals], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    binary_preds = (df.iloc[:, 0] > threshold).astype(int)
    return (binary_preds == df.iloc[:, 1]).mean()


def compute_all_metrics(
    predictions: pd.Series,
    actuals: pd.Series,
    returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Compute all standard metrics.

    Parameters
    ----------
    predictions : pd.Series
        Model predictions
    actuals : pd.Series
        Actual values (binary or continuous)
    returns : pd.Series, optional
        Forward returns for quantile analysis

    Returns
    -------
    dict
        All metrics
    """
    metrics = {
        "IC": information_coefficient(predictions, actuals),
        "AUROC": auroc(predictions, actuals),
        "Brier": brier_score(predictions, actuals),
        "HitRate": hit_rate(predictions, actuals),
    }

    if returns is not None:
        quant_metrics = quantile_spread(predictions, returns)
        metrics.update(quant_metrics)

    return metrics


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000

    # Simulate predictions and actuals
    true_signal = np.random.randn(n)
    predictions = true_signal + np.random.randn(n) * 0.5
    actuals = (true_signal > 0).astype(int)
    returns = true_signal * 0.01 + np.random.randn(n) * 0.02

    preds = pd.Series(predictions)
    acts = pd.Series(actuals)
    rets = pd.Series(returns)

    # Compute metrics
    metrics = compute_all_metrics(preds, acts, rets)

    print("Performance Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
