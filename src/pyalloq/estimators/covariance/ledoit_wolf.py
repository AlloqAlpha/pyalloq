import pandas as pd
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]
from pyalloq.core.interfaces import BaseCovarianceEstimator
from pyalloq.core.data import MarketData
from typing import Any


class LedoitWolfShrinkage(BaseCovarianceEstimator):
    def __init__(self, annualization_factor: float = 252.0) -> None:
        self.annualization_factor = annualization_factor

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.DataFrame:
        returns = data.prices.pct_change().dropna()

        lw = LedoitWolf().fit(returns.values)
        shrunk_cov = lw.covariance_ * self.annualization_factor

        return pd.DataFrame(
            shrunk_cov, index=data.prices.columns, columns=data.prices.columns
        )
