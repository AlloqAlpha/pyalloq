import pandas as pd
import numpy as np
from .base import BaseCovarianceEstimator

class SemiCovariance(BaseCovarianceEstimator):
    def __init__(
        self, 
        benchmark_return: float = 0.0,
        annualization_factor: float = 252.0,
    ) -> None:
        self.benchmark_return = benchmark_return
        self.annualization_factor = annualization_factor

    def estimate(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change().dropna()

        daily_benchmark = self.benchmark_return / self.annualization_factor
        downside_returns = np.minimum(returns - daily_benchmark, 0.0)

        T = len(returns)
        semi_cov = (downside_returns.T @ downside_returns) / T
        semi_cov = semi_cov * self.annualization_factor

        return pd.DataFrame(
            semi_cov,
            index=prices.columns,
            columns=prices.columns,
        )