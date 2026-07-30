import pandas as pd
from sklearn.covariance import LedoitWolf
from .base import BaseCovarianceEstimator

class LedoitWolfShrinkage(BaseCovarianceEstimator):
    def __init__(self, annualization_factor: float = 252.0) -> None:
        self.annualization_factor = annualization_factor

    def estimate(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change().dropna()

        lw = LedoitWolf().fit(returns.values)
        shrunk_cov = lw.covariance_ * self.annualization_factor

        return pd.DataFrame(
            shrunk_cov,
            index=prices.columns,
            columns=prices.columns
        )