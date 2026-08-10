import pandas as pd
from pyalloq.core.interfaces import BaseReturnEstimator
from pyalloq.core.interfaces import BaseCovarianceEstimator
from pyalloq.core.data import MarketData


class ImpliedReturnEstimator(BaseReturnEstimator):
    def __init__(
        self, cov_estimator: BaseCovarianceEstimator, risk_aversion: float = 2.5
    ) -> None:
        self.cov_estimator = cov_estimator
        self.risk_aversion = risk_aversion

    def estimate(
        self,
        data: MarketData,
        **kwargs,
    ) -> pd.Series:
        cross_sectional = data.cross_sectional
        if cross_sectional is None or "market_cap" not in cross_sectional.columns:
            raise ValueError(
                "ImpliedReturnEstimator requires 'market_cap' in data.cross_sectional"
            )
        market_caps = cross_sectional["market_cap"]

        cov_matrix = self.cov_estimator.estimate(data)
        market_caps = market_caps.loc[cov_matrix.columns]
        market_weights = market_caps / market_caps.sum()

        implied_returns = self.risk_aversion * cov_matrix.dot(market_weights)

        return implied_returns
