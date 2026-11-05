from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import DataFrequency
from pyalloq_core.interfaces import BaseReturnEstimator
from pyalloq_core.utils import resolve_annualization_factor


class MeanHistoricalReturnEstimator(BaseReturnEstimator):
    """
    Computes annualized expected returns from historical price series using either:
    - Arithmetic sample mean (`compounding=False`, default): `mean(r) * freq`
    - Geometric / CAGR mean (`compounding=True`): `(prod(1 + r)) ** (freq / T) - 1`
    """

    def __init__(
        self,
        compounding: bool = False,
        frequency: DataFrequency | str | int | float | None = DataFrequency.DAILY,
        window_len: int | None = None,
    ) -> None:
        self.compounding = compounding
        self.frequency = frequency
        self.window_len = window_len

    def estimate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> pd.Series:
        prices = (
            data.prices.iloc[-self.window_len :]
            if self.window_len is not None
            else data.prices
        )
        returns = prices.pct_change().dropna()
        scale_factor = resolve_annualization_factor(returns, self.frequency)

        if len(returns) == 0:
            return pd.Series(0.0, index=prices.columns, name="expected_returns")

        if self.compounding:
            growth = (1.0 + returns).prod(axis=0).clip(lower=1e-12)
            cagr_arr = (
                np.power(
                    growth.to_numpy(dtype=float), scale_factor / float(len(returns))
                )
                - 1.0
            )
            expected_returns = pd.Series(cagr_arr, index=prices.columns)
        else:
            expected_returns = returns.mean(axis=0) * scale_factor

        expected_returns.name = "expected_returns"
        return expected_returns
