from typing import Any
import pandas as pd
import numpy as np

try:
    from timesfm3 import TimesFM3Evaluator
except ImportError:
    TimesFM3Evaluator = Any  # type: ignore[misc,assignment]

from pyalloq_core.interfaces import BaseReturnEstimator
from pyalloq_core.data import MarketData


class TimesFM3UnivariateReturnEstimator(BaseReturnEstimator):
    """Zero-shot univariate return estimator using Google's TimesFM 3.0."""

    def __init__(self, model: TimesFM3Evaluator, horizon: int = 21):
        self.model = model
        self.horizon = horizon

    def estimate(self, data: MarketData, **kwargs) -> pd.Series:
        stock_prices = data.prices
        assets = stock_prices.columns

        # 1. TimesFM 3.0 expects a list of 1D numpy arrays for univariate series
        ts_list = [stock_prices[col].to_numpy(dtype=np.float32) for col in assets]

        # 2. Run zero-shot forecasting
        outputs = list(
            self.model.predict_batch(
                ts_list,
                horizon=self.horizon,
                return_quantiles=False,
                use_symmetric_averaging=False,
            )
        )

        # 3. Extract the point forecast at the exact target horizon
        # outputs[i].forecast shape is (horizon,)
        future_prices = np.array([out.forecast[self.horizon - 1] for out in outputs])
        current_prices = stock_prices.iloc[-1].to_numpy()

        # 4. Calculate and annualize expected returns
        predicted_returns = (future_prices / current_prices) - 1.0
        annualized_returns = predicted_returns * (252.0 / self.horizon)

        return pd.Series(annualized_returns, index=assets, name="expected_returns")


class TimesFM3MultivariateReturnEstimator(BaseReturnEstimator):
    """
    Zero-shot multivariate return estimator using Google's TimesFM 3.0.
    Captures cross-asset spillovers and incorporates MarketData features.
    """

    def __init__(
        self, model: TimesFM3Evaluator, horizon: int = 21, annualize: bool = True
    ):
        self.model = model
        self.horizon = horizon
        self.annualize = annualize

    def estimate(self, data: MarketData, **kwargs) -> pd.Series:
        stock_prices = data.prices
        assets = stock_prices.columns
        current_prices = stock_prices.iloc[-1].to_numpy()

        # 1. Target Variates: (N_assets, Context_Length)
        target = stock_prices.to_numpy(dtype=np.float32).T

        # 2. Past-Only Covariates: Process MarketData.features
        past_only_cov = None
        if data.features:
            past_covs = []
            for feat_name, df_feat in data.features.items():
                aligned_feat = df_feat.reindex(
                    index=stock_prices.index, columns=assets
                ).fillna(0.0)
                # Stack to create a flat list of feature channels: (Num_Features * N_assets, T)
                past_covs.append(aligned_feat.to_numpy(dtype=np.float32).T)
            past_only_cov = np.concatenate(past_covs, axis=0)

        past_future_cov = None
        if data.future_features:
            future_covs = [
                # data.future_features is already sliced to (Context + Horizon) by MarketData
                df.reindex(columns=assets).fillna(0.0).to_numpy(dtype=np.float32).T
                for df in data.future_features.values()
            ]
            past_future_cov = np.concatenate(future_covs, axis=0)

        predict_kwargs = {
            "contexts": [target],
            "horizon": self.horizon,
            "return_quantiles": False,  # We only need the point forecast for standard \mu
            "use_symmetric_averaging": False,
        }
        if past_only_cov is not None:
            predict_kwargs["past_only_covariates"] = [past_only_cov]
        if past_future_cov is not None:
            predict_kwargs["past_future_covariates"] = [past_future_cov]

        outputs = self.model.predict_batch(**predict_kwargs)
        outputs = list(outputs)

        # 4. Extract point forecast for the target horizon
        # outputs[0].forecast shape is (N_assets, Horizon)
        future_prices = outputs[0].forecast[:, self.horizon - 1]

        predicted_returns = (future_prices / current_prices) - 1.0

        if self.annualize:
            predicted_returns = predicted_returns * (252.0 / self.horizon)

        return pd.Series(predicted_returns, index=assets, name="expected_returns")
