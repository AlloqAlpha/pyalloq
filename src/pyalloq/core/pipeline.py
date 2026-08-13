import pandas as pd
from typing import Any
from pyalloq.core.interfaces import (
    BaseAllocator,
    BaseReturnEstimator,
    BaseCovarianceEstimator,
)
from pyalloq.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.core.data import MarketData


class StrategyPipeline:
    def __init__(
        self,
        allocator: BaseAllocator,
        returns_estimator: BaseReturnEstimator | None = None,
        cov_estimator: BaseCovarianceEstimator | None = None,
        allocator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.allocator = allocator
        self.returns_estimator = (
            returns_estimator
            if returns_estimator is not None
            else EWMAReturnEstimator()
        )
        self.cov_estimator = (
            cov_estimator if cov_estimator is not None else EmpiricalCovariance()
        )
        self.allocator_kwargs = allocator_kwargs or {}

    def generate_weights(
        self,
        data: MarketData,
    ) -> pd.Series:
        expected_returns = self.returns_estimator.estimate(data)
        cov_matrix = self.cov_estimator.estimate(data)

        result = self.allocator.allocate(
            data, cov_matrix=cov_matrix, expected_returns=expected_returns
        )

        return result.weights
