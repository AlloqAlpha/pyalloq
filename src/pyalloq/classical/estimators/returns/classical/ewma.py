import pandas as pd
from pyalloq_core.interfaces import BaseReturnEstimator
from pyalloq_core.data import MarketData


class EWMAReturnEstimator(BaseReturnEstimator):
    def __init__(self, span: int = 252) -> None:
        self.span = span

    def estimate(
        self,
        data: MarketData,
        **kwargs,
    ) -> pd.Series:
        daily_returns = data.prices.pct_change().dropna()
        ewma_daily = daily_returns.ewm(span=self.span, adjust=False).mean()
        expected_returns = ewma_daily.iloc[-1] * 252

        return expected_returns
