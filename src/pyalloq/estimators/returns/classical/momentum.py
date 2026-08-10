import pandas as pd
from pyalloq.core.interfaces import BaseReturnEstimator
from pyalloq.core.data import MarketData


class CrossSectionalMomentumEstimator(BaseReturnEstimator):
    def estimate(
        self,
        data: MarketData,
        **kwargs,
    ) -> pd.Series:
        if len(data.prices) < 252:
            raise ValueError(
                "Momentum estimator requires at least 252 days (1 year) of price data."
            )

        p_12m = data.prices.iloc[-252]
        p_1m = data.prices.iloc[-21]

        momentum_scores = (p_1m / p_12m) - 1.0
        z_scores = (momentum_scores - momentum_scores.mean()) / momentum_scores.std()

        return z_scores
