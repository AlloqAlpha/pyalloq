from typing import Any
import pandas as pd
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.results import OptimizationResult
from pyalloq.core.data import MarketData


class NaiveAllocator(BaseAllocator):
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
            status="OPTIMAL_NAIVE",
            metadata={"strategy": "1/N Equal Weight"},
        )
