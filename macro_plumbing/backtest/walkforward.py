"""
walkforward.py
Walk-forward validation for liquidity stress models.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Optional
from .metrics import compute_all_metrics


class WalkForwardValidator:
    """Walk-forward cross-validation."""

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step: Optional[int] = None,
    ):
        """
        Initialize validator.

        Parameters
        ----------
        train_window : int
            Training window size (e.g., 252 days = 1 year)
        test_window : int
            Test window size (e.g., 63 days = 1 quarter)
        step : int, optional
            Step size (defaults to test_window)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step if step is not None else test_window

    def split(self, data_length: int) -> List[Tuple[slice, slice]]:
        """
        Generate train/test splits.

        Parameters
        ----------
        data_length : int
            Total data length

        Returns
        -------
        list
            List of (train_slice, test_slice) tuples
        """
        splits = []
        start = 0

        while start + self.train_window + self.test_window <= data_length:
            train_end = start + self.train_window
            test_end = train_end + self.test_window

            train_slice = slice(start, train_end)
            test_slice = slice(train_end, test_end)

            splits.append((train_slice, test_slice))

            start += self.step

        return splits

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_func: Callable,
        returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Perform walk-forward validation.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        model_func : callable
            Function that takes (X_train, y_train) and returns fitted model
            Model must have .predict(X) method
        returns : pd.Series, optional
            Forward returns for performance analysis

        Returns
        -------
        pd.DataFrame
            Results for each fold
        """
        data = pd.concat([y, X], axis=1).dropna()
        y_data = data[y.name]
        X_data = data.drop(columns=y.name)

        splits = self.split(len(data))
        results = []

        for i, (train_idx, test_idx) in enumerate(splits):
            # Split data
            X_train = X_data.iloc[train_idx]
            y_train = y_data.iloc[train_idx]
            X_test = X_data.iloc[test_idx]
            y_test = y_data.iloc[test_idx]

            # Train model
            try:
                model = model_func(X_train, y_train)
                predictions = model.predict(X_test)

                # Compute metrics
                if returns is not None:
                    ret_test = returns.loc[y_test.index]
                    metrics = compute_all_metrics(predictions, y_test, ret_test)
                else:
                    metrics = compute_all_metrics(predictions, y_test)

                results.append(
                    {
                        "fold": i,
                        "train_start": data.index[train_idx.start],
                        "train_end": data.index[train_idx.stop - 1],
                        "test_start": data.index[test_idx.start],
                        "test_end": data.index[test_idx.stop - 1],
                        **metrics,
                    }
                )
            except Exception as e:
                print(f"Fold {i} failed: {e}")
                continue

        return pd.DataFrame(results)


def expanding_window_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_func: Callable,
    min_train: int = 252,
    test_window: int = 63,
    step: int = 63,
) -> pd.DataFrame:
    """
    Expanding window validation (always train on all historical data).

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model_func : callable
        Model training function
    min_train : int
        Minimum training size
    test_window : int
        Test window size
    step : int
        Step size

    Returns
    -------
    pd.DataFrame
        Results for each fold
    """
    data = pd.concat([y, X], axis=1).dropna()
    y_data = data[y.name]
    X_data = data.drop(columns=y.name)

    results = []
    start_test = min_train

    while start_test + test_window <= len(data):
        # Expanding train window
        X_train = X_data.iloc[:start_test]
        y_train = y_data.iloc[:start_test]

        # Fixed test window
        X_test = X_data.iloc[start_test : start_test + test_window]
        y_test = y_data.iloc[start_test : start_test + test_window]

        # Train and predict
        try:
            model = model_func(X_train, y_train)
            predictions = model.predict(X_test)

            metrics = compute_all_metrics(predictions, y_test)

            results.append(
                {
                    "train_end": data.index[start_test - 1],
                    "test_start": data.index[start_test],
                    "test_end": data.index[start_test + test_window - 1],
                    "train_size": len(X_train),
                    **metrics,
                }
            )
        except Exception as e:
            print(f"Test window starting at {start_test} failed: {e}")

        start_test += step

    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2023-12-31", freq="D")

    # Synthetic data
    X = pd.DataFrame(
        {
            "feature1": np.cumsum(np.random.randn(len(dates))) * 0.1,
            "feature2": np.random.randn(len(dates)),
        },
        index=dates,
    )
    y = ((X["feature1"] + X["feature2"]) > 0).astype(int)
    y.name = "target"

    # Model function
    def train_model(X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    # Walk-forward validation
    validator = WalkForwardValidator(train_window=500, test_window=100, step=100)
    results = validator.validate(X, y, train_model)

    print("Walk-Forward Results:")
    print(results[["fold", "IC", "AUROC", "HitRate"]])
    print(f"\nAverage IC: {results['IC'].mean():.4f}")
    print(f"Average AUROC: {results['AUROC'].mean():.4f}")
