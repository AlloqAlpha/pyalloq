from typing import Any

import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import DataFrequency
from pyalloq_core.interfaces import BaseReturnEstimator
from pyalloq_core.utils import resolve_annualization_factor


class EWMAReturnEstimator(BaseReturnEstimator):
    def __init__(
        self,
        span: int = 252,
        frequency: DataFrequency | str | int | float | None = DataFrequency.DAILY,
    ) -> None:
        self.span = span
        self.frequency = frequency

    def estimate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> pd.Series:
        daily_returns = data.prices.pct_change().dropna()
        scale_factor = resolve_annualization_factor(daily_returns, self.frequency)
        ewma_daily = daily_returns.ewm(span=self.span, adjust=False).mean()
        expected_returns = ewma_daily.iloc[-1] * scale_factor

        return expected_returns
