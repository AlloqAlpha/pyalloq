from typing import Any

import numpy as np
import pandas as pd

try:
    from timestral import TimestralForecaster
except ImportError:
    TimestralForecaster = None  # type: ignore[misc,assignment]

from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseReturnEstimator

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
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Convert MarketData prices, features, and future_features into 3D tensors
    expected by TimestralForecaster.forecast:
      - context: (1, N_assets, T)
      - past_covariates: (1, Num_Features * N_assets, T) or None
      - known_covariates: (1, Num_Future_Features * N_assets, T + H) or None
    """
    stock_prices = data.prices
    assets = list(stock_prices.columns)
    t_steps = len(stock_prices)

    # 1. Target Variates: shape (1, N_assets, T)
    target = stock_prices.to_numpy(dtype=np.float32).T[np.newaxis, :, :]

    # 2. Past-Only Covariates: shape (1, Num_Features * N_assets, T)
    past_covariates: np.ndarray | None = None
    if data.features:
        past_covs = []
        for df_feat in data.features.values():
            aligned_feat = df_feat.reindex(
                index=stock_prices.index, columns=assets
            ).fillna(0.0)
            past_covs.append(aligned_feat.to_numpy(dtype=np.float32).T)
        if past_covs:
            past_covariates = np.concatenate(past_covs, axis=0)[np.newaxis, :, :]

    # 3. Known-Future Covariates: shape (1, Num_Future_Features * N_assets, T + H)
    known_covariates: np.ndarray | None = None
    if data.future_features:
        future_covs = []
        for df in data.future_features.values():
            arr = df.reindex(columns=assets).fillna(0.0).to_numpy(dtype=np.float32).T
            # If future_features has length T (due to MarketData.validate_alignment),
            # pad or slice to (T + horizon) when horizon is provided.
            if horizon is not None:
                expected_len = t_steps + horizon
                if arr.shape[1] < expected_len:
                    pad_width = expected_len - arr.shape[1]
                    arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="edge")
                elif arr.shape[1] > expected_len:
                    arr = arr[:, :expected_len]
            future_covs.append(arr)
        if future_covs:
            known_covariates = np.concatenate(future_covs, axis=0)[np.newaxis, :, :]

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
        device: str | None = None,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.annualize = annualize
        self.device = device
        self._forecaster: Any | None = (
            model if (model is not None and not isinstance(model, str)) else None
        )

    def _get_forecaster(self) -> Any:
        if self._forecaster is None:
            self._forecaster = _resolve_timestral_model(self.model, device=self.device)
        return self._forecaster

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        stock_prices = data.prices
        assets = stock_prices.columns
        current_prices = stock_prices.iloc[-1].to_numpy(dtype=np.float64)

        # 1. Timestral univariate batch shape: (N_assets, 1, Context_Length)
        context = stock_prices.to_numpy(dtype=np.float32).T[:, np.newaxis, :]

        # 2. Run zero-shot probabilistic forecasting
        forecaster = self._get_forecaster()
        forecast = forecaster.forecast(context, prediction_length=self.horizon)

        # 3. Extract the median point forecast at the exact target horizon
        # forecast.median has shape (N_assets, 1, horizon)
        median_arr = np.asarray(forecast.median, dtype=np.float64)
        future_prices = median_arr[:, 0, self.horizon - 1]

        # 4. Calculate and optionally annualize expected returns
        predicted_returns = (future_prices / current_prices) - 1.0
        if self.annualize:
            predicted_returns = predicted_returns * (252.0 / self.horizon)

        return pd.Series(predicted_returns, index=assets, name="expected_returns")


class TimestralMultivariateReturnEstimator(BaseReturnEstimator):
    """
    Zero-shot multivariate return estimator using Mistral AI's Timestral foundation model.
    Captures cross-asset spillovers jointly and incorporates MarketData past and known-future covariates.
    """

    def __init__(
        self,
        model: Any = DEFAULT_TIMESTRAL_CHECKPOINT,
        horizon: int = 21,
        annualize: bool = True,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.annualize = annualize
        self.device = device
        self._forecaster: Any | None = (
            model if (model is not None and not isinstance(model, str)) else None
        )

    def _get_forecaster(self) -> Any:
        if self._forecaster is None:
            self._forecaster = _resolve_timestral_model(self.model, device=self.device)
        return self._forecaster

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        stock_prices = data.prices
        assets = stock_prices.columns
        current_prices = stock_prices.iloc[-1].to_numpy(dtype=np.float64)

        # 1. Prepare 3D target and optional covariate tensors
        target, past_covariates, known_covariates = (
            _prepare_multivariate_timestral_inputs(data, horizon=self.horizon)
        )

        forecast_kwargs: dict[str, Any] = {
            "prediction_length": self.horizon,
        }
        if past_covariates is not None:
            forecast_kwargs["past_covariates"] = past_covariates
        if known_covariates is not None:
            forecast_kwargs["known_covariates"] = known_covariates

        # 2. Run joint multivariate forecast
        forecaster = self._get_forecaster()
        forecast = forecaster.forecast(target, **forecast_kwargs)

        # 3. Extract median forecast at target horizon
        # forecast.median shape is (1, N_assets, horizon)
        median_arr = np.asarray(forecast.median, dtype=np.float64)
        future_prices = median_arr[0, :, self.horizon - 1]

        predicted_returns = (future_prices / current_prices) - 1.0
        if self.annualize:
            predicted_returns = predicted_returns * (252.0 / self.horizon)

        return pd.Series(predicted_returns, index=assets, name="expected_returns")
