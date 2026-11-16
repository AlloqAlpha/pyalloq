from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import CovariateKey, CovariateMode
from pyalloq_core.interfaces import BaseReturnEstimator

try:
    from timestral import TimestralForecaster
except ImportError:
    TimestralForecaster = None  # type: ignore[misc,assignment]

DEFAULT_TIMESTRAL_CHECKPOINT = "mistralai/Timestral-28M"


def _resolve_timestral_model(
    model: Any,
    device: str | None = None,
) -> Any:
    """Resolve a pre-loaded TimestralForecaster or lazy-load from a Hugging Face checkpoint."""
    if model is not None and not isinstance(model, str):
        return model

    checkpoint = model if isinstance(model, str) else DEFAULT_TIMESTRAL_CHECKPOINT
    if TimestralForecaster is None:
        raise ImportError(
            "The 'timestral' package is required to load Timestral models by checkpoint name. "
            "Install it via: pip install git+https://github.com/mistralai/timestral.git"
        )
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    return TimestralForecaster.from_pretrained(checkpoint, **kwargs)


def _prepare_multivariate_timestral_inputs(
    data: MarketData,
    horizon: int | None = None,
    covariate_mode: CovariateMode = CovariateMode.AUTO,
    covariate_keys: Sequence[CovariateKey | str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Convert MarketData prices, features, and future_features into 3D tensors
    expected by TimestralForecaster.forecast:
      - context: `(1, N_assets, T)`
      - past_covariates: `(1, C_past, T)` or `None`
      - known_covariates: `(1, C_future, T + H)` or `None`
    """
    stock_prices = data.prices
    target = stock_prices.to_numpy(dtype=np.float32).T[np.newaxis, :, :]

    past_cov, future_cov = data.extract_covariates(
        mode=covariate_mode,
        covariate_keys=covariate_keys,
        horizon=horizon,
        dtype=np.float32,
    )
    past_covariates = past_cov[np.newaxis, :, :] if past_cov is not None else None
    known_covariates = future_cov[np.newaxis, :, :] if future_cov is not None else None
    return target, past_covariates, known_covariates


class TimestralUnivariateReturnEstimator(BaseReturnEstimator):
    """
    Zero-shot univariate return estimator using Mistral AI's Timestral foundation model.
    Forecasts each asset series independently as a batch of 1-variate series.
    """

    def __init__(
        self,
        model: Any = DEFAULT_TIMESTRAL_CHECKPOINT,
        horizon: int = 21,
        annualize: bool = True,
        covariate_mode: CovariateMode = CovariateMode.NONE,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.horizon = int(horizon)
        self.annualize = bool(annualize)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
        self.device = device
        self._forecaster: Any | None = (
            model if (model is not None and not isinstance(model, str)) else None
        )
        self.last_posterior_std_: pd.Series | None = None
        self.last_posterior_cvar_: pd.Series | None = None
        self.last_posterior_downside_std_: pd.Series | None = None
        self.last_posterior_skewness_: pd.Series | None = None
        self.last_interdecile_range_: pd.Series | None = None
        self.last_quantiles_: dict[float, pd.Series] = {}

    def _get_forecaster(self) -> Any:
        if self._forecaster is None:
            self._forecaster = _resolve_timestral_model(self.model, device=self.device)
        return self._forecaster

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        stock_prices = data.prices
        assets = list(stock_prices.columns)
        current_prices = stock_prices.iloc[-1].to_numpy(dtype=np.float64)

        context = stock_prices.to_numpy(dtype=np.float32).T[:, np.newaxis, :]
        forecaster = self._get_forecaster()
        forecast = forecaster.forecast(context, prediction_length=self.horizon)

        median_arr = np.asarray(forecast.median, dtype=np.float64)
        future_prices = median_arr[:, 0, self.horizon - 1]

        scale = (252.0 / self.horizon) if self.annualize else 1.0
        predicted_returns = ((future_prices / current_prices) - 1.0) * scale

        if hasattr(forecast, "quantile"):
            p10_p = np.asarray(forecast.quantile(0.10), dtype=np.float64)[
                :, 0, self.horizon - 1
            ]
            p90_p = np.asarray(forecast.quantile(0.90), dtype=np.float64)[
                :, 0, self.horizon - 1
            ]
            std_scale = float(np.sqrt(252.0 / self.horizon)) if self.annualize else 1.0
            ret_std = (
                np.maximum(p90_p - p10_p, 1e-8) / (2.56 * current_prices)
            ) * std_scale
            r10 = ((p10_p / current_prices) - 1.0) * scale
            r90 = ((p90_p / current_prices) - 1.0) * scale
            self.last_posterior_std_ = pd.Series(ret_std, index=assets)
            self.last_quantiles_ = {
                0.10: pd.Series(r10, index=assets),
                0.50: pd.Series(predicted_returns, index=assets),
                0.90: pd.Series(r90, index=assets),
            }
            idr = np.maximum(r90 - r10, 1e-6)
            skew = ((r90 - predicted_returns) - (predicted_returns - r10)) / idr
            down_std = np.maximum(predicted_returns - r10, 1e-6) / 1.2816
            cvar_95 = np.maximum(-predicted_returns + 2.0627 * down_std, 0.5 * ret_std)
            self.last_interdecile_range_ = pd.Series(idr, index=assets)
            self.last_posterior_skewness_ = pd.Series(skew, index=assets)
            self.last_posterior_downside_std_ = pd.Series(down_std, index=assets)
            self.last_posterior_cvar_ = pd.Series(cvar_95, index=assets)

        return pd.Series(predicted_returns, index=assets, name="expected_returns")


class TimestralMultivariateReturnEstimator(BaseReturnEstimator):
    """
    Zero-shot multivariate return estimator using Mistral AI's Timestral foundation model.
    Captures cross-asset spillovers jointly and incorporates optional `MarketData` past and
    known-future covariates via `CovariateMode` and `CovariateKey`.
    """

    def __init__(
        self,
        model: Any = DEFAULT_TIMESTRAL_CHECKPOINT,
        horizon: int = 21,
        annualize: bool = True,
        covariate_mode: CovariateMode = CovariateMode.AUTO,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.horizon = int(horizon)
        self.annualize = bool(annualize)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
        self.device = device
        self._forecaster: Any | None = (
            model if (model is not None and not isinstance(model, str)) else None
        )
        self.last_posterior_std_: pd.Series | None = None
        self.last_posterior_cvar_: pd.Series | None = None
        self.last_posterior_downside_std_: pd.Series | None = None
        self.last_posterior_skewness_: pd.Series | None = None
        self.last_interdecile_range_: pd.Series | None = None
        self.last_quantiles_: dict[float, pd.Series] = {}

    def _get_forecaster(self) -> Any:
        if self._forecaster is None:
            self._forecaster = _resolve_timestral_model(self.model, device=self.device)
        return self._forecaster

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        stock_prices = data.prices
        assets = list(stock_prices.columns)
        current_prices = stock_prices.iloc[-1].to_numpy(dtype=np.float64)

        target, past_covariates, known_covariates = (
            _prepare_multivariate_timestral_inputs(
                data,
                horizon=self.horizon,
                covariate_mode=self.covariate_mode,
                covariate_keys=self.covariate_keys,
            )
        )

        forecast_kwargs: dict[str, Any] = {
            "prediction_length": self.horizon,
        }
        if past_covariates is not None:
            forecast_kwargs["past_covariates"] = past_covariates
        if known_covariates is not None:
            forecast_kwargs["known_covariates"] = known_covariates

        forecaster = self._get_forecaster()
        forecast = forecaster.forecast(target, **forecast_kwargs)

        median_arr = np.asarray(forecast.median, dtype=np.float64)
        future_prices = median_arr[0, :, self.horizon - 1]

        scale = (252.0 / self.horizon) if self.annualize else 1.0
        predicted_returns = ((future_prices / current_prices) - 1.0) * scale

        if hasattr(forecast, "quantile"):
            p10_p = np.asarray(forecast.quantile(0.10), dtype=np.float64)[
                0, :, self.horizon - 1
            ]
            p90_p = np.asarray(forecast.quantile(0.90), dtype=np.float64)[
                0, :, self.horizon - 1
            ]
            std_scale = float(np.sqrt(252.0 / self.horizon)) if self.annualize else 1.0
            ret_std = (
                np.maximum(p90_p - p10_p, 1e-8) / (2.56 * current_prices)
            ) * std_scale
            r10 = ((p10_p / current_prices) - 1.0) * scale
            r90 = ((p90_p / current_prices) - 1.0) * scale
            self.last_posterior_std_ = pd.Series(ret_std, index=assets)
            self.last_quantiles_ = {
                0.10: pd.Series(r10, index=assets),
                0.50: pd.Series(predicted_returns, index=assets),
                0.90: pd.Series(r90, index=assets),
            }
            idr = np.maximum(r90 - r10, 1e-6)
            skew = ((r90 - predicted_returns) - (predicted_returns - r10)) / idr
            down_std = np.maximum(predicted_returns - r10, 1e-6) / 1.2816
            cvar_95 = np.maximum(-predicted_returns + 2.0627 * down_std, 0.5 * ret_std)
            self.last_interdecile_range_ = pd.Series(idr, index=assets)
            self.last_posterior_skewness_ = pd.Series(skew, index=assets)
            self.last_posterior_downside_std_ = pd.Series(down_std, index=assets)
            self.last_posterior_cvar_ = pd.Series(cvar_95, index=assets)

        return pd.Series(predicted_returns, index=assets, name="expected_returns")
