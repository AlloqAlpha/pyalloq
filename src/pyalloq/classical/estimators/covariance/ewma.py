import pandas as pd
from pyalloq_core.interfaces import BaseCovarianceEstimator
from pyalloq_core.data import MarketData
from typing import Any, cast


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
        data: MarketData,
        **kwargs: Any,
    ) -> pd.DataFrame:
        returns = data.prices.pct_change().dropna()

        ewma_cov_series = returns.ewm(span=self.span).cov()
        latest_date = returns.index[-1]
        latest_cov = ewma_cov_series.xs(latest_date, level=0)

        final_cov = latest_cov * self.annualization_factor
        return cast(pd.DataFrame, final_cov)
