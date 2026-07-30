import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pyalloq.core.interfaces import BaseReturnEstimator

class CAPMReturnEstimator(BaseReturnEstimator):
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        market_risk_premium: float = 0.05,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.market_risk_premium = market_risk_premium

    def estimate(
        self,
        prices: pd.DataFrame,
        market_prices: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        returns = prices.pct_change().dropna()
        market_prices = market_prices.pct_change().dropna()

        returns, market_returns = returns.align(market_returns, join="inner", axis=0)
        market_var = market_returns.var()
        expected_returns = {}

        for asset in returns.column:
            cov = returns[asset].cov(market_returns)
            beta = cov / market_var

            expected_returns[asset] = self.risk_free_rate + beta * self.market_risk_premium

        return pd.Series(expected_returns)

class MultiFactorReturnEstimator(BaseReturnEstimator):
    def __init__(
        self,
        factor_premium: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> None:
        self.factor_premium = factor_premium
        self.risk_free_rate = risk_free_rate
        self.model = LinearRegression()

    def estimate(
        self,
        prices: pd.DataFrame,
        factor_data: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        returns = prices.pct_change().dropna()
        returns, factor_data = returns.align(factor_data, join="inner", axis=0)

        expected_returns = {}
        for asset in returns.columns:
            y = returns[asset].values
            X = factor_data.values

            self.model.fit(X, y)
            betas = self.model.coef_

            asset_mu = self.risk_free_rate + np.dot(betas, self.factor_premium.values)
            expected_returns[asset] = asset_mu

        return pd.Series(expected_returns)