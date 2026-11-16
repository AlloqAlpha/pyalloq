from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import CovariateKey, CovariateMode
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
    if np.isclose(q_low, 0.10) and np.isclose(q_high, 0.90):
        z_spread = 2.56
    else:
        z_spread = float(norm.ppf(q_high) - norm.ppf(q_low))
    return float(q_low), float(q_high), z_spread


class TimestralUnivariateViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman views (`Q`) and uncertainty (`Omega`)
    using the 21 non-crossing probabilistic quantiles of Mistral AI's Timestral in univariate mode.
    """

    def __init__(
        self,
        model: Any = DEFAULT_TIMESTRAL_CHECKPOINT,
        horizon: int = 21,
        confidence_interval: tuple[float, float] = (0.10, 0.90),
        annualize: bool = True,
        full_covariance_omega: bool = False,
        covariate_mode: CovariateMode = CovariateMode.NONE,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.horizon = int(horizon)
        self.confidence_interval = confidence_interval
        self.annualize = bool(annualize)
        self.full_covariance_omega = bool(full_covariance_omega)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
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

        context = prices_df.to_numpy(dtype=np.float32).T[:, np.newaxis, :]
        forecaster = self._get_forecaster()
        forecast = forecaster.forecast(context, prediction_length=self.horizon)

        p50_prices = np.asarray(forecast.median, dtype=np.float64)[
            :, 0, self.horizon - 1
        ]
        q_low_prices = np.asarray(forecast.quantile(self._q_low), dtype=np.float64)[
            :, 0, self.horizon - 1
        ]
        q_high_prices = np.asarray(forecast.quantile(self._q_high), dtype=np.float64)[
            :, 0, self.horizon - 1
        ]

        view_returns = (p50_prices / current_prices) - 1.0
        if self.annualize:
            view_returns = view_returns * (252.0 / self.horizon)
        Q = pd.Series(view_returns, index=assets, name="expected_returns")

        price_std = np.maximum(q_high_prices - q_low_prices, 1e-8) / self._z_spread
        return_std = price_std / current_prices
        if self.annualize:
            return_std = return_std * np.sqrt(252.0 / self.horizon)

        Omega = pd.DataFrame(np.diag(return_std**2), index=assets, columns=assets)
        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)
        return P, Q, Omega


class TimestralMultivariateViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman views (`Q`) and uncertainty (`Omega`)
    using the probabilistic outputs of Mistral AI's Timestral in joint multivariate mode.
    Incorporates both past-only covariates (`MarketData.features`) and known-future covariates
    (`MarketData.future_features`) via `CovariateMode` and `CovariateKey`, plus optional
    non-diagonal `Omega`.
    """

    def __init__(
        self,
        model: Any = DEFAULT_TIMESTRAL_CHECKPOINT,
        horizon: int = 21,
        confidence_interval: tuple[float, float] = (0.10, 0.90),
        annualize: bool = True,
        full_covariance_omega: bool = False,
        covariate_mode: CovariateMode = CovariateMode.AUTO,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.horizon = int(horizon)
        self.confidence_interval = confidence_interval
        self.annualize = bool(annualize)
        self.full_covariance_omega = bool(full_covariance_omega)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
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

        med_full = np.asarray(forecast.median, dtype=np.float64)[0]  # (N, H)
        p50_prices = med_full[:, self.horizon - 1]
        q_low_prices = np.asarray(forecast.quantile(self._q_low), dtype=np.float64)[
            0, :, self.horizon - 1
        ]
        q_high_prices = np.asarray(forecast.quantile(self._q_high), dtype=np.float64)[
            0, :, self.horizon - 1
        ]

        view_returns = (p50_prices / current_prices) - 1.0
        if self.annualize:
            view_returns = view_returns * (252.0 / self.horizon)
        Q = pd.Series(view_returns, index=assets, name="expected_returns")

        price_std = np.maximum(q_high_prices - q_low_prices, 1e-8) / self._z_spread
        return_std = price_std / current_prices
        if self.annualize:
            return_std = return_std * np.sqrt(252.0 / self.horizon)

        if self.full_covariance_omega and n_assets > 1 and med_full.shape[1] >= 2:
            med_traj = med_full / current_prices[:, None] - 1.0
            corr = np.corrcoef(med_traj)
            corr = np.nan_to_num(corr, nan=0.0)
            corr = 0.5 * (corr + corr.T)
            np.fill_diagonal(corr, 1.0)
            eigvals, eigvecs = np.linalg.eigh(corr)
            corr_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-4)) @ eigvecs.T
            d_inv = 1.0 / np.sqrt(np.diag(corr_psd))
            corr_psd = corr_psd * np.outer(d_inv, d_inv)
            omega_mat = np.outer(return_std, return_std) * (
                0.7 * np.eye(n_assets) + 0.3 * corr_psd
            )
        else:
            omega_mat = np.diag(return_std**2)

        Omega = pd.DataFrame(omega_mat, index=assets, columns=assets)
        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)
        return P, Q, Omega
