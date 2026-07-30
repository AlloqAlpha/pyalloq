import pandas as pd
import numpy as np
from .base import BaseCovarianceEstimator

class EmpiricalCovariance(BaseCovarianceEstimator):
    def __init__(self, annualization_factor: float = 252.0) -> None:
        self.annualization_factor = annualization_factor

    def estimate(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change().dropna()
        cov_matrix = returns.cov() * self.annualization_factor

        return cov_matrix