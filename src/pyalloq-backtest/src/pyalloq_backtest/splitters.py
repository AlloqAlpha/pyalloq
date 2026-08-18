import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Generator
from pyalloq_core.data import MarketData


class BaseWindowSplitter(ABC):
    @abstractmethod
    def split(
        self,
        data: MarketData,
        rebalance_dates: pd.DatetimeIndex,
    ) -> Generator[tuple[pd.Timestamp, MarketData], None, None]:
        """Yields (current date, sliced_market_data)"""
        pass


class RollingWindowSplitter(BaseWindowSplitter):
    def __init__(self, lookback_window: int = 252) -> None:
        self.lookback_window = lookback_window

    def split(
        self,
        data: MarketData,
        rebalance_dates: pd.DatetimeIndex,
    ) -> Generator[tuple[pd.Timestamp, MarketData], None, None]:
        for current_date in rebalance_dates:
            data_window = data.slice_time(
                end_date=current_date, lookback=self.lookback_window
            )

            if len(data_window.prices) >= self.lookback_window:
                yield current_date, data_window


class ExpandingWindowSplitter(BaseWindowSplitter):
    def __init__(self, min_periods: int = 252) -> None:
        self.min_periods = min_periods

    def split(
        self,
        data: MarketData,
        rebalance_dates: pd.DatetimeIndex,
    ) -> Generator[tuple[pd.Timestamp, MarketData], None, None]:
        for current_date in rebalance_dates:
            data_window = data.slice_time(end_date=current_date, lookback=None)

            if len(data_window.prices) >= self.min_periods:
                yield current_date, data_window


class PurgedKFoldSplitter:
    """
    Used strictly for Cross-Validation (Deep Learning / ML training).
    Splits data into K folds, applying a purge and embargo to prevent data leakage.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self, data: MarketData
    ) -> Generator[tuple[MarketData, MarketData], None, None]:
        """Yields (train_data, test_data) for neural network training."""

        total_length = len(data.prices)
        fold_size = total_length // self.n_splits
        embargo_size = int(total_length * self.embargo_pct)

        indices = np.arange(total_length)
        dates = data.prices.index

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else total_length

            test_indices = indices[test_start:test_end]
            train_indices_left = indices[: max(0, test_start - embargo_size)]
            train_indices_right = indices[min(total_length, test_end + embargo_size) :]

            train_indices = np.concatenate([train_indices_left, train_indices_right])

            if len(train_indices) == 0:
                continue

            train_dates = dates[train_indices]
            test_dates = dates[test_indices]

            train_data = MarketData(
                prices=data.prices.loc[train_dates],
                features={k: v.loc[train_dates] for k, v in data.features.items()}
                if data.features
                else {},
                cross_sectional=data.cross_sectional,
                risk_free_rate=data.risk_free_rate.loc[train_dates]
                if isinstance(data.risk_free_rate, pd.Series)
                else data.risk_free_rate,
            )

            test_data = MarketData(
                prices=data.prices.loc[test_dates],
                features={k: v.loc[test_dates] for k, v in data.features.items()}
                if data.features
                else {},
                cross_sectional=data.cross_sectional,
                risk_free_rate=data.risk_free_rate.loc[test_dates]
                if isinstance(data.risk_free_rate, pd.Series)
                else data.risk_free_rate,
            )

            yield train_data, test_data
