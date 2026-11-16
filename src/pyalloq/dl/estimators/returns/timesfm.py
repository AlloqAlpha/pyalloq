from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import CovariateKey, CovariateMode
from pyalloq_core.interfaces import BaseReturnEstimator

try:
    from timesfm3 import TimesFM3Evaluator
except ImportError:
    TimesFM3Evaluator = Any  # type: ignore[misc,assignment]

P10_INDEX = 0
P50_INDEX = 4
P90_INDEX = 8


class TimesFM3UnivariateReturnEstimator(BaseReturnEstimator):
    """
    Zero-shot univariate return estimator using Google's TimesFM 3.0.
    """

    def __init__(
        self,
        model: Any = None,
        horizon: int = 21,
        annualize: bool = True,
        covariate_mode: CovariateMode = CovariateMode.NONE,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        device: str | None = None,
        tfm_model: Any = None,
    ) -> None:
        self.model = model if model is not None else tfm_model
        self.horizon = int(horizon)
        self.annualize = bool(annualize)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
        self.device = device
        self.last_posterior_std_: pd.Series | None = None
        self.last_quantiles_: dict[float, pd.Series] = {}

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        stock_prices = data.prices
        assets = list(stock_prices.columns)
        current_prices = stock_prices.iloc[-1].to_numpy(dtype=np.float64)

        ts_list = [stock_prices[col].to_numpy(dtype=np.float32) for col in assets]
        outputs = list(
            self.model.predict_batch(
                ts_list,
                horizon=self.horizon,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )

        future_prices = np.array(
            [out.forecast[self.horizon - 1] for out in outputs], dtype=np.float64
        )
        predicted_returns = (future_prices / current_prices) - 1.0
        scale = (252.0 / self.horizon) if self.annualize else 1.0
        annualized_returns = predicted_returns * scale

        # Extract quantile spread for uncertainty-aware ensembling when quantiles are returned
        if hasattr(outputs[0], "quantiles") and outputs[0].quantiles is not None:
            p10_p = np.array(
                [out.quantiles[self.horizon - 1, P10_INDEX] for out in outputs],
                dtype=np.float64,
            )
            p50_p = np.array(
                [out.quantiles[self.horizon - 1, P50_INDEX] for out in outputs],
                dtype=np.float64,
            )
            p90_p = np.array(
                [out.quantiles[self.horizon - 1, P90_INDEX] for out in outputs],
                dtype=np.float64,
            )
            r10 = ((p10_p / current_prices) - 1.0) * scale
            r50 = ((p50_p / current_prices) - 1.0) * scale
            r90 = ((p90_p / current_prices) - 1.0) * scale
            std_scale = float(np.sqrt(252.0 / self.horizon)) if self.annualize else 1.0
            ret_std = (
                np.maximum(p90_p - p10_p, 1e-8) / (2.56 * current_prices)
            ) * std_scale
            self.last_posterior_std_ = pd.Series(ret_std, index=assets)
            self.last_quantiles_ = {
                0.10: pd.Series(r10, index=assets),
                0.50: pd.Series(r50, index=assets),
                0.90: pd.Series(r90, index=assets),
            }

        return pd.Series(annualized_returns, index=assets, name="expected_returns")


class TimesFM3MultivariateReturnEstimator(BaseReturnEstimator):
    """
    Zero-shot multivariate return estimator using Google's TimesFM 3.0.
    Captures cross-asset spillovers and incorporates optional `MarketData` past and
    known-future covariates via `CovariateMode` and `CovariateKey`.
    """

    def __init__(
        self,
        model: Any = None,
        horizon: int = 21,
        annualize: bool = True,
        covariate_mode: CovariateMode = CovariateMode.AUTO,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        device: str | None = None,
        tfm_model: Any = None,
    ) -> None:
        self.model = model if model is not None else tfm_model
        self.horizon = int(horizon)
        self.annualize = bool(annualize)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
        self.device = device
        self.last_posterior_std_: pd.Series | None = None
        self.last_quantiles_: dict[float, pd.Series] = {}

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        stock_prices = data.prices
        assets = list(stock_prices.columns)
        current_prices = stock_prices.iloc[-1].to_numpy(dtype=np.float64)

        target = stock_prices.to_numpy(dtype=np.float32).T
        past_only_cov, past_future_cov = data.extract_covariates(
            mode=self.covariate_mode,
            covariate_keys=self.covariate_keys,
            horizon=self.horizon,
            dtype=np.float32,
        )

        predict_kwargs: dict[str, Any] = {
            "contexts": [target],
            "horizon": self.horizon,
            "return_quantiles": True,
            "use_symmetric_averaging": False,
        }
        if past_only_cov is not None:
            predict_kwargs["past_only_covariates"] = [past_only_cov]
        if past_future_cov is not None:
            predict_kwargs["past_future_covariates"] = [past_future_cov]

        outputs = list(self.model.predict_batch(**predict_kwargs))
        future_prices = np.asarray(
            outputs[0].forecast[:, self.horizon - 1], dtype=np.float64
        )
        predicted_returns = (future_prices / current_prices) - 1.0
        scale = (252.0 / self.horizon) if self.annualize else 1.0
        if self.annualize:
            predicted_returns = predicted_returns * scale

        if hasattr(outputs[0], "quantiles") and outputs[0].quantiles is not None:
            q_arr = np.asarray(outputs[0].quantiles, dtype=np.float64)
            p10_p = q_arr[:, self.horizon - 1, P10_INDEX]
            p50_p = q_arr[:, self.horizon - 1, P50_INDEX]
            p90_p = q_arr[:, self.horizon - 1, P90_INDEX]
            r10 = ((p10_p / current_prices) - 1.0) * scale
            r50 = ((p50_p / current_prices) - 1.0) * scale
            r90 = ((p90_p / current_prices) - 1.0) * scale
            std_scale = float(np.sqrt(252.0 / self.horizon)) if self.annualize else 1.0
            ret_std = (
                np.maximum(p90_p - p10_p, 1e-8) / (2.56 * current_prices)
            ) * std_scale
            self.last_posterior_std_ = pd.Series(ret_std, index=assets)
            self.last_quantiles_ = {
                0.10: pd.Series(r10, index=assets),
                0.50: pd.Series(r50, index=assets),
                0.90: pd.Series(r90, index=assets),
            }

        return pd.Series(predicted_returns, index=assets, name="expected_returns")
