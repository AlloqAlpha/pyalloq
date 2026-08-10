from abc import ABC, abstractmethod
import pandas as pd
from typing import Any
from .enums import DataFrequency
from .data import MarketData
from .results import OptimizationResult


class BaseAllocator(ABC):
    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers

    def _resolve_annualization_factor(
        self,
        returns: pd.DataFrame,
        frequency: DataFrequency | str | int | None = None,
    ) -> float:
        """Determines annualization factor from Enum, int or inferred pandas index."""
        if isinstance(frequency, DataFrequency):
            return float(frequency.value)
        if isinstance(frequency, (int, float)):
            return float(frequency)
        if isinstance(frequency, str):
            try:
                float(DataFrequency[frequency.upper()].value)
            except KeyError:
                pass

        if isinstance(returns.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(returns.index)
            if inferred:
                if "W" in inferred:
                    return DataFrequency.WEEKLY.value
                if "M" in inferred or "MS" in inferred:
                    return DataFrequency.MONTHLY.value
                if "B" in inferred or "D" in inferred:
                    return DataFrequency.DAILY.value

        return DataFrequency.DAILY.value

    def _prepare_inputs(
        self,
        prices: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        window_len: int | None = None,
        frequency: DataFrequency | str | int = DataFrequency.DAILY,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Internal helper to resolve inputs. If raw prices are given,
        it computes basic historical matrices. Otherwise, it uses the provided ones.
        """
        final_returns = expected_returns
        final_cov = cov_matrix

        returns = prices.iloc[-window_len:] if window_len else prices
        scale_factor = self._resolve_annualization_factor(returns, frequency)
        df_returns = returns.pct_change().dropna()

        if final_returns is None:
            final_returns = df_returns.mean() * scale_factor

        if final_cov is None:
            final_cov = df_returns.cov() * scale_factor

        if final_returns is None or final_cov is None:
            raise ValueError(
                "You must provide either raw 'prices' or BOTH 'expected_returns' and 'cov_matrix'"
            )

        return (final_returns, final_cov)

    @abstractmethod
    def allocate(
        self,
        data: MarketData,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Base allocate function all subclasses must implement
        """
        ...


class BaseReturnEstimator(ABC):
    @abstractmethod
    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        """
        Takes raw prices (and optional macro/technical features) and
        returns an Nx1 pandas Series of expected returns for the assets.
        """
        pass


class BaseCovarianceEstimator(ABC):
    @abstractmethod
    def estimate(self, data: MarketData, **kwargs: Any) -> pd.DataFrame:
        """Takes a price DataFrame and returns an NxN covariance matrix."""
        pass
