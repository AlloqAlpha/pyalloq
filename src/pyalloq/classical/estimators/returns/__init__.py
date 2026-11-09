from pyalloq.classical.estimators.returns.classical import (
    CAPMReturnEstimator,
    EWMAReturnEstimator,
    ImpliedReturnEstimator,
    JamesSteinReturnEstimator,
    MeanHistoricalReturnEstimator,
    MultiFactorReturnEstimator,
    ResidualMomentumEstimator,
    VarianceRiskPremiumEstimator,
    VolatilityScaledMultiHorizonEstimator,
)
from pyalloq.classical.estimators.returns.multi_horizon import (
    AlphaDecayReturnEstimator,
)

__all__ = [
    "AlphaDecayReturnEstimator",
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
