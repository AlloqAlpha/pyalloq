from typing import Any
import pandas as pd
import numpy as np
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from pyalloq_core.data import MarketData


class EqualWeightAllocator(BaseAllocator):
    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        N = len(data.prices.columns)
        weights = pd.Series(1.0 / N, index=data.prices.columns)

        return OptimizationResult(
            weights=weights,
            status="OPTIMAL_EQUAL_WEIGHT",
            metadata={"strategy": "1/N Equal Weight"},
        )


class RandomAllocator(BaseAllocator):
    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        tickers = data.prices.columns
        N = len(tickers)
        weights = pd.Series(np.random.dirichlet(np.ones(N)), index=tickers)

        return OptimizationResult(
            weights=weights,
            status="OPTIMAL_RANDOM",
            metadata={"strategy": "Random Weights"},
        )
