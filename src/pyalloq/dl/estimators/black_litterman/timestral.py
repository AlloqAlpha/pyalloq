from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseViewGenerator
from scipy.stats import norm

from pyalloq.dl.estimators.returns.timestral import (
    DEFAULT_TIMESTRAL_CHECKPOINT,
    _prepare_multivariate_timestral_inputs,
    _resolve_timestral_model,
)


def _validate_confidence_interval(
    confidence_interval: tuple[float, float],
) -> tuple[float, float, float]:
    """Validate quantile bounds (q_low, q_high) and return (q_low, q_high, z_spread)."""
    q_low, q_high = confidence_interval
    if not (0.0 < q_low < q_high < 1.0):
        raise ValueError(
            f"confidence_interval must satisfy 0 < q_low < q_high < 1, got {confidence_interval}."
        )
    # Use exact standard 2.56 divisor for the canonical (0.10, 0.90) interval for parity with TimesFM,
    # otherwise compute the Gaussian z-score spread dynamically via norm.ppf.
    if np.isclose(q_low, 0.10) and np.isclose(q_high, 0.90):
        z_spread = 2.56
    else:
        z_spread = float(norm.ppf(q_high) - norm.ppf(q_low))
    return float(q_low), float(q_high), z_spread


class TimestralUnivariateViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman views (Q) and uncertainty (Omega)
    using the 21 non-crossing probabilistic quantiles of Mistral AI's Timestral in univariate mode.
    """

    def __init__(
        self,
        model: Any = DEFAULT_TIMESTRAL_CHECKPOINT,
        horizon: int = 21,
        confidence_interval: tuple[float, float] = (0.10, 0.90),
        annualize: bool = True,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.confidence_interval = confidence_interval
        self.annualize = annualize
        self.device = device
        self._q_low, self._q_high, self._z_spread = _validate_confidence_interval(
            confidence_interval
        )
        self._forecaster: Any | None = (
            model if (model is not None and not isinstance(model, str)) else None
        )

    def _get_forecaster(self) -> Any:
        if self._forecaster is None:
            self._forecaster = _resolve_timestral_model(self.model, device=self.device)
        return self._forecaster

    def generate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        prices_df = data.prices
        assets = list(prices_df.columns)
        n_assets = len(assets)
        current_prices = prices_df.to_numpy(dtype=np.float64)[-1]

        # 1. Timestral univariate batch shape: (N_assets, 1, Context_Length)
        context = prices_df.to_numpy(dtype=np.float32).T[:, np.newaxis, :]

        # 2. Run zero-shot probabilistic forecasting
        forecaster = self._get_forecaster()
        forecast = forecaster.forecast(context, prediction_length=self.horizon)

        # 3. Extract median and lower/upper quantiles at target horizon
        # Shapes: (N_assets, 1, horizon)
        p50_prices = np.asarray(forecast.median, dtype=np.float64)[
            :, 0, self.horizon - 1
        ]
        q_low_prices = np.asarray(forecast.quantile(self._q_low), dtype=np.float64)[
            :, 0, self.horizon - 1
        ]
        q_high_prices = np.asarray(forecast.quantile(self._q_high), dtype=np.float64)[
            :, 0, self.horizon - 1
        ]

        # 4. View Expected Returns (Q) - based on the median forecast
        view_returns = (p50_prices / current_prices) - 1.0
        if self.annualize:
            view_returns = view_returns * (252.0 / self.horizon)
        Q = pd.Series(view_returns, index=assets, name="expected_returns")

        # 5. View Uncertainty Matrix (Omega)
        price_std = np.maximum(q_high_prices - q_low_prices, 1e-8) / self._z_spread
        return_std = price_std / current_prices
        if self.annualize:
            return_std = return_std * np.sqrt(252.0 / self.horizon)

        omega_variances = return_std**2
        Omega = pd.DataFrame(np.diag(omega_variances), index=assets, columns=assets)

        # 6. Picking Matrix (P) is the Identity Matrix
        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)

        return P, Q, Omega


class TimestralMultivariateViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman views (Q) and uncertainty (Omega)
    using the probabilistic outputs of Mistral AI's Timestral in joint multivariate mode.
    Incorporates both past-only covariates (MarketData.features) and known-future covariates
    (MarketData.future_features).
    """

    def __init__(
        self,
        model: Any = DEFAULT_TIMESTRAL_CHECKPOINT,
        horizon: int = 21,
        confidence_interval: tuple[float, float] = (0.10, 0.90),
        annualize: bool = True,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.confidence_interval = confidence_interval
        self.annualize = annualize
        self.device = device
        self._q_low, self._q_high, self._z_spread = _validate_confidence_interval(
            confidence_interval
        )
        self._forecaster: Any | None = (
            model if (model is not None and not isinstance(model, str)) else None
        )

    def _get_forecaster(self) -> Any:
        if self._forecaster is None:
            self._forecaster = _resolve_timestral_model(self.model, device=self.device)
        return self._forecaster

    def generate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        stock_prices = data.prices
        assets = list(data.assets)
        n_assets = len(assets)
        current_prices = stock_prices.to_numpy(dtype=np.float64)[-1]

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

        # 2. Generate the Joint Multivariate Forecast
        forecaster = self._get_forecaster()
        forecast = forecaster.forecast(target, **forecast_kwargs)

        # 3. Extract Quantiles for the specific horizon
        # Shape of forecast.median / forecast.quantile(q) is (1, N_assets, horizon)
        p50_prices = np.asarray(forecast.median, dtype=np.float64)[
            0, :, self.horizon - 1
        ]
        q_low_prices = np.asarray(forecast.quantile(self._q_low), dtype=np.float64)[
            0, :, self.horizon - 1
        ]
        q_high_prices = np.asarray(forecast.quantile(self._q_high), dtype=np.float64)[
            0, :, self.horizon - 1
        ]

        # 4. View Expected Returns (Q)
        view_returns = (p50_prices / current_prices) - 1.0
        if self.annualize:
            view_returns = view_returns * (252.0 / self.horizon)
        Q = pd.Series(view_returns, index=assets, name="expected_returns")

        # 5. View Uncertainty Matrix (Omega)
        price_std = np.maximum(q_high_prices - q_low_prices, 1e-8) / self._z_spread
        return_std = price_std / current_prices
        if self.annualize:
            return_std = return_std * np.sqrt(252.0 / self.horizon)

        Omega = pd.DataFrame(np.diag(return_std**2), index=assets, columns=assets)

        # 6. Picking Matrix (P)
        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)

        return P, Q, Omega
