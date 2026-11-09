from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.estimators.returns.classical.factor import (
    CAPMReturnEstimator,
    MultiFactorReturnEstimator,
)
from pyalloq.classical.estimators.returns.classical.implied import (
    ImpliedReturnEstimator,
)
from pyalloq.classical.estimators.returns.classical.james_stein import (
    JamesSteinReturnEstimator,
)
from pyalloq.classical.estimators.returns.classical.mean import (
    MeanHistoricalReturnEstimator,
)
from pyalloq.classical.estimators.returns.classical.momentum import (
    ResidualMomentumEstimator,
    VolatilityScaledMultiHorizonEstimator,
)
from pyalloq.classical.estimators.returns.classical.vrp import (
    VarianceRiskPremiumEstimator,
)

__all__ = [
    "CAPMReturnEstimator",
    "EWMAReturnEstimator",
    "ImpliedReturnEstimator",
    "JamesSteinReturnEstimator",
    "MeanHistoricalReturnEstimator",
    "MultiFactorReturnEstimator",
    "ResidualMomentumEstimator",
    "VarianceRiskPremiumEstimator",
    "VolatilityScaledMultiHorizonEstimator",
]
