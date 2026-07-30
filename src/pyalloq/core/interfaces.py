from abc import ABC, abstractmethod
import pandas as pd
from .enums import DataFrequency
from .results import OptimizationResult


class BaseAllocator(ABC):
    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers

    def _resolve_annualization_factor(
        self,
        prices: pd.DataFrame,
        frequency: DataFrequency | str | int = None,
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

        # Infer frequency directly from pandas DatetimeIndex
        if isinstance(prices.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(prices.index)
            if inferred:
                if "W" in inferred: return DataFrequency.WEEKLY.value
                if "M" in inferred or "MS" in inferred: return DataFrequency.MONTHLY.value
                if "B" in inferred or "D" in inferred: return DataFrequency.DAILY.value

        return DataFrequency.DAILY.value

    def _prepare_inputs(
        self,
        prices: pd.DataFrame | None = None,
        expected_returns: pd.DataFrame | None = None,
        cov_matrix: pd.DataFrame = None,
        window_len: int | None = None,
        frequency: DataFrequency | str | int = DataFrequency.DAILY,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Internal helper to resolve inputs. If raw prices are given, 
        it computes basic historical matrices. Otherwise, it uses the provided ones.
        """
        if prices is not None:
            df = prices.iloc[-window_len:] if window_len else prices
            scale_factor = self._resolve_annualization_factor(df, frequency)
            returns = df.pct_change().dropna()
            calculated_returns = returns.mean() * scale_factor
            calculated_cov = returns.cov() * scale_factor

            return (calculated_returns, calculated_cov)

        if expected_returns is None or cov_matrix is None:
            raise ValueError("You must provide either raw 'prices' or BOTH 'expected_returns' and 'cov_matrix'")

    @abstractmethod
    def allocate(self, *args, **kwargs) -> OptimizationResult:
        """
        All child classes must implement this method
        """
        ...

class BaseReturnEstimator(ABC):
    @abstractmethod
    def estimate(
        self, 
        prices: pd.DataFrame, 
        features: pd.DataFrame | None = None,
        **kwargs
    ) -> pd.Series:
        """
        Takes raw prices (and optional macro/technical features) and 
        returns an Nx1 pandas Series of expected returns for the assets.
        """
        pass