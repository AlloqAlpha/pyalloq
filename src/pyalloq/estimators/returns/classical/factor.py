import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseReturnEstimator
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]


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
        data: MarketData,
        **kwargs,
    ) -> pd.Series:
        returns = data.prices.pct_change().dropna()
        market_prices = data.features.get("market_prices")
        if market_prices is None:
            raise ValueError(
                "CAPMReturnEstimator requires market_prices in data.features['market_prices']"
            )

        market_series: pd.Series = (
            market_prices.iloc[:, 0]
            if isinstance(market_prices, pd.DataFrame)
            else market_prices
        )
        market_returns: pd.Series = market_series.pct_change().dropna()

        common_index = returns.index.intersection(market_returns.index)
        aligned_returns = returns.loc[common_index]
        aligned_market_returns = market_returns.loc[common_index]

        market_var = float(aligned_market_returns.var())
        expected_returns = {}

        for asset in aligned_returns.columns:
            cov = float(aligned_returns[asset].cov(aligned_market_returns))
            beta = cov / market_var if market_var > 0 else 1.0

            expected_returns[asset] = (
                self.risk_free_rate + beta * self.market_risk_premium
            )

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
        data: MarketData,
        **kwargs,
    ) -> pd.Series:
        factor_data = data.cross_sectional
        if factor_data is None:
            raise ValueError(
                "MultiFactorReturnEstimator requires 'factors' in data.cross_sectional"
            )

        returns = data.prices.pct_change().dropna()

        returns, factor_data = returns.align(factor_data, join="inner", axis=0)

        expected_returns = {}
        for asset in returns.columns:
            y = returns[asset].values
            X = factor_data.values

            self.model.fit(X, y)
            betas = self.model.coef_

            asset_mu = self.risk_free_rate + (betas @ self.factor_premium.values)
            expected_returns[asset] = asset_mu

        return pd.Series(expected_returns)
