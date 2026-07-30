import pandas as pd
from pyalloq.core.interfaces import BaseReturnEstimator

class CrossSectionalMomentumEstimator(BaseReturnEstimator):
    def estimate(
        self,
        prices: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        if len(prices) < 252:
            raise ValueError("Momentum estimator requires at least 252 days (1 year) of price data.")

        p_12m = prices.iloc[-252]
        p_1m = prices.iloc[-21]

        momentum_scores = (p_1m / p_12m) - 1.0
        z_scores = (momentum_scores - momentum_scores.mean()) / momentum_scores.std()

        return z_scores