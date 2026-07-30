import pandas as pd
from typing import List
from pyalloq.core.interfaces import BaseAllocator

class WalkForwardEngine:
    def __init__(
        self,
        allocator: BaseAllocator,
        lookback_window: int = 252,
        rebalance_freq: str = "6ME"
    ) -> None:
        self.allocator = allocator
        self.lookback_window = lookback_window
        self.rebalance_freq = rebalance_freq

    def run(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        rebalance_dates = prices.resample(self.rebalance_freq).last()

        weights_history = []
        for current_date in rebalance_dates.index:
            df_historical = prices.loc[:current_date]

            if len(df_historical) < self.lookback_window:
                continue

            df_window = df_historical.iloc[-self.lookback_window:]

            result = self.allocator.allocate(
                prices=df_window,
            )
            weights = result.clean_weights()
            weights.name = current_date
            weights_history.append(weights)

        return pd.DataFrame(weights_history)