import pandas as pd
from typing import Any
from pyalloq.core.interfaces import BaseAllocator, BaseReturnEstimator
from pyalloq.core.data import MarketData


class StrategyPipeline:
    def __init__(
        self,
        allocator: BaseAllocator,
        returns_estimator: BaseReturnEstimator | None = None,
        cov_estimator: Any | None = None,
        allocator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.allocator = allocator
        self.returns_estimator = returns_estimator
        self.cov_estimator = cov_estimator
        self.allocator_kwargs = allocator_kwargs or {}

    def generate_weights(
        self,
        data: MarketData,
    ) -> pd.Series:
        expected_returns: pd.Series | None = None
        if self.returns_estimator is not None:
            expected_returns = self.returns_estimator.estimate(data)

        cov_matrix: pd.DataFrame | None = None
        if self.cov_estimator is not None:
            cov_matrix = self.cov_estimator.estimate(data)

        result = self.allocator.allocate(data, expected_returns, cov_matrix)

        return result.weights
