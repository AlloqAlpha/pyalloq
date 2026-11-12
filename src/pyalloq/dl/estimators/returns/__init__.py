"""Deep learning return estimators."""

from pyalloq.dl.estimators.returns.timesfm import (
    TimesFM3MultivariateReturnEstimator,
    TimesFM3UnivariateReturnEstimator,
)
from pyalloq.dl.estimators.returns.timestral import (
    TimestralMultivariateReturnEstimator,
    TimestralUnivariateReturnEstimator,
)

__all__ = [
    "TimesFM3MultivariateReturnEstimator",
    "TimesFM3UnivariateReturnEstimator",
    "TimestralMultivariateReturnEstimator",
    "TimestralUnivariateReturnEstimator",
]
