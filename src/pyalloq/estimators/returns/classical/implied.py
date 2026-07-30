import pandas as pd
from pyalloq.core.interfaces import BaseReturnEstimator

class ImpliedReturnEstimator(BaseReturnEstimator):
    def __init__(self, risk_aversion: float = 2.5):
        self.risk_aversion = risk_aversion

    def estimate(
        self,
        cov_matrix: pd.DataFrame,
        market_caps: pd.Series,
        **kwargs,
    ) -> pd.Series:
        market_caps = market_caps.loc[cov_matrix.columns]
        market_weights = market_caps / market_caps.sum()

        implied_returns = self.risk_aversion * cov_matrix.dot(market_weights)

        return implied_returns