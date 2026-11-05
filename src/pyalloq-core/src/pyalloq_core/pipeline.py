from typing import Any

import pandas as pd

from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import (
    BaseAllocator,
    BaseCovarianceEstimator,
    BaseReturnEstimator,
)


class StrategyPipeline:
    def __init__(
        self,
        allocator: BaseAllocator,
        returns_estimator: BaseReturnEstimator | None = None,
        cov_estimator: BaseCovarianceEstimator | None = None,
        allocator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.allocator = allocator
        if returns_estimator is None:
            from pyalloq.classical.estimators.returns.classical.ewma import (
                EWMAReturnEstimator,
            )

            self.returns_estimator: BaseReturnEstimator = EWMAReturnEstimator()
        else:
            self.returns_estimator = returns_estimator

        if cov_estimator is None:
            from pyalloq.classical.estimators.covariance.empirical import (
                EmpiricalCovariance,
            )

            self.cov_estimator: BaseCovarianceEstimator = EmpiricalCovariance()
        else:
            self.cov_estimator = cov_estimator
        self.allocator_kwargs = allocator_kwargs or {}

    def generate_weights(
        self,
        data: MarketData,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        expected_returns = self.returns_estimator.estimate(data)
        cov_matrix = self.cov_estimator.estimate(data)

        call_kwargs = dict(self.allocator_kwargs)
        if current_weights is not None and "current_weights" not in call_kwargs:
            call_kwargs["current_weights"] = current_weights

        result = self.allocator.allocate(
            data,
            cov_matrix=cov_matrix,
            expected_returns=expected_returns,
            **call_kwargs,
        )

        return (
            result.clean_weights()
            if hasattr(result, "clean_weights")
            else result.weights
        )
