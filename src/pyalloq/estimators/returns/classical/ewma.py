import pandas as pd
from pyalloq.core.interfaces import BaseReturnEstimator

class EWMAReturnEstimator(BaseReturnEstimator):
    def __init__(self, span: int = 252) -> None:
        self.span = span

    def estimate(
        self,
        prices: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        daily_returns = prices.pct_change().dropna()
        ewma_daily = daily_returns.ewm(span=self.span, adjust=False).mean()
        expected_returns = ewma_daily.iloc[-1] * 252

        return expected_returns