import pandas as pd
from .base import BaseCovarianceEstimator

class EWMACovariance(BaseCovarianceEstimator):
    def __init__(
        self, 
        span: int = 60,
        annualization_factor: float = 252.0,
    ) -> None:
        self.span = span
        self.annualization_factor = annualization_factor

    def estimate(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        returns = prices.pct_change().dropna()

        ewma_cov_series = returns.ewm(span=self.span).cov()
        latest_date = returns.index[-1]
        latest_cov = ewma_cov_series.xs(latest_date, level=0)

        return latest_cov * self.annualization_factor