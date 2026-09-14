import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseReturnEstimator


class VolatilityScaledMultiHorizonEstimator(BaseReturnEstimator):
    """
    Computes raw momentum across multiple lookbacks, scales each to its specific realized volatility, outputs a single composite Z-score vector.
    Acts as a stable relative-return forecast.
    """

    def __init__(
        self, horizons: tuple[int, ...] = (63, 126, 252), skip_period: int = 21
    ):
        self.horizons = horizons
        self.skip_period = skip_period

    def estimate(self, data: MarketData, **kwargs) -> pd.Series:
        max_lookback = max(self.horizons)
        if len(data.prices) < max_lookback:
            raise ValueError(f"Requires at least {max_lookback} price observations.")

        returns = data.prices.pct_change().dropna()
        p_skip = data.prices.iloc[-self.skip_period]

        horizon_z_scores = []
        for h in self.horizons:
            p_start = data.prices.iloc[-h]
            raw_ret = (p_skip / p_start) - 1.0

            # Annualized volatility over horizon h
            vol = returns.iloc[-h:].std() * np.sqrt(252.0)
            risk_adj_ret = raw_ret / (vol + 1e-6)

            # Cross-sectional standardization to convert to relative expected returns
            z = (risk_adj_ret - risk_adj_ret.mean()) / (risk_adj_ret.std() + 1e-6)
            horizon_z_scores.append(z)

        # Average z-scores across horizons
        composite_z = pd.concat(horizon_z_scores, axis=1).mean(axis=1)
        composite_z.name = "expected_returns"
        return composite_z


class ResidualMomentumEstimator(BaseReturnEstimator):
    """
    Isolates the idiosyncratic trend. By regressing out market beta, it ensures the outputted return estimate is purely driven by asset specific momentum, not just broader market movement.
    """

    def __init__(self, lookback: int = 252, skip_period: int = 21):
        self.lookback = lookback
        self.skip_period = skip_period

    def estimate(self, data: MarketData, **kwargs) -> pd.Series:
        if len(data.prices) < self.lookback:
            raise ValueError(f"Requires at least {self.lookback} price observations.")

        daily_rets = data.prices.pct_change().dropna()
        window_rets = daily_rets.iloc[-self.lookback : -self.skip_period]
        market_proxy = window_rets.mean(axis=1)

        mkt_var = market_proxy.var()
        mkt_mean = market_proxy.mean()

        scores = {}
        for asset in window_rets.columns:
            r = window_rets[asset]

            # Simple OLS regression math
            beta = r.cov(market_proxy) / (mkt_var + 1e-8)
            alpha = r.mean() - beta * mkt_mean
            residuals = r - (alpha + beta * market_proxy)

            # Standardized cumulative residual
            residual_score = residuals.sum() / (residuals.std() + 1e-6)
            scores[asset] = residual_score

        s = pd.Series(scores)
        z_scores = (s - s.mean()) / (s.std() + 1e-6)
        z_scores.name = "expected_returns"

        return z_scores
