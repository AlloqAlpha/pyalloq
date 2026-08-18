import pandas as pd
from pyalloq_core.interfaces import BaseCovarianceEstimator
from pyalloq_core.data import MarketData
from typing import Any


class EmpiricalCovariance(BaseCovarianceEstimator):
    def __init__(self, annualization_factor: float = 252.0) -> None:
        self.annualization_factor = annualization_factor

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.DataFrame:
        returns = data.prices.pct_change().dropna()
        cov_matrix = returns.cov() * self.annualization_factor

        return cov_matrix
