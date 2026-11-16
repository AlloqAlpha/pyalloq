from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import CovariateKey, CovariateMode
from pyalloq_core.interfaces import BaseViewGenerator
from scipy.stats import norm

try:
    from timesfm3 import TimesFM3Evaluator
except ImportError:
    TimesFM3Evaluator = Any  # type: ignore[misc,assignment]

P10_INDEX = 0
P50_INDEX = 4
P90_INDEX = 8
TIMESFM_QUANTILE_GRID = np.array(
    [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float64
)


def _validate_timesfm_confidence_interval(
    confidence_interval: tuple[float, float],
) -> tuple[float, float, float]:
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


def _interpolate_timesfm_quantile(q_slice: np.ndarray, target_q: float) -> np.ndarray:
    """
    Interpolates a `(N_assets, 9)` quantile slice along TimesFM-3's 9 decile levels
    `[0.10, ..., 0.90]` to evaluate arbitrary `target_q in (0, 1)`.
    """
    if np.isclose(target_q, 0.10):
        return q_slice[:, P10_INDEX]
    if np.isclose(target_q, 0.50):
        return q_slice[:, P50_INDEX]
    if np.isclose(target_q, 0.90):
        return q_slice[:, P90_INDEX]

    out = np.empty(len(q_slice), dtype=np.float64)
    for i in range(len(q_slice)):
        out[i] = float(np.interp(target_q, TIMESFM_QUANTILE_GRID, q_slice[i]))
    return out


class TimesFM3UnivariateViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman views (`Q`) and uncertainty (`Omega`)
    using the probabilistic quantile outputs of TimesFM-3.0 in univariate mode.
    """

    def __init__(
        self,
        model: Any = None,
        horizon: int = 21,
        confidence_interval: tuple[float, float] = (0.10, 0.90),
        annualize: bool = True,
        full_covariance_omega: bool = False,
        covariate_mode: CovariateMode = CovariateMode.NONE,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        uncertainty_mode: Literal[
            "posterior_variance", "downside_semivariance", "cvar_tail"
        ] = "posterior_variance",
        tail_skew_aversion: float = 0.15,
        device: str | None = None,
        tfm_model: Any = None,
    ) -> None:
        resolved = model if model is not None else tfm_model
        self.model = resolved
        self.tfm_model = resolved
        self.horizon = int(horizon)
        self.confidence_interval = confidence_interval
        self.annualize = bool(annualize)
        self.full_covariance_omega = bool(full_covariance_omega)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
        self.uncertainty_mode = uncertainty_mode
        self.tail_skew_aversion = float(max(0.0, tail_skew_aversion))
        self.device = device
        self._q_low, self._q_high, self._z_spread = (
            _validate_timesfm_confidence_interval(confidence_interval)
        )
        self.last_posterior_cvar_: pd.Series | None = None
        self.last_posterior_downside_std_: pd.Series | None = None
        self.last_posterior_skewness_: pd.Series | None = None
        self.last_interdecile_range_: pd.Series | None = None

    def generate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        prices_df = data.prices
        assets = list(cast(Any, prices_df.columns))
        n_assets = len(assets)
        current_prices = prices_df.to_numpy(dtype=np.float64)[-1]

        ts_list = [prices_df[col].to_numpy(dtype=np.float32) for col in assets]
        outputs = cast(
            list[Any],
            list(
                self.tfm_model.predict_batch(
                    ts_list,
                    horizon=self.horizon,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                )
            ),
        )

        q_horizon = np.stack(
            [
                np.asarray(out.quantiles[self.horizon - 1], dtype=np.float64)
                for out in outputs
            ],
            axis=0,
        )  # (N, 9)
        p50_prices = _interpolate_timesfm_quantile(q_horizon, 0.50)
        q_low_prices = _interpolate_timesfm_quantile(q_horizon, self._q_low)
        q_high_prices = _interpolate_timesfm_quantile(q_horizon, self._q_high)

        view_returns = (p50_prices / current_prices) - 1.0
        scale = (252.0 / self.horizon) if self.annualize else 1.0
        if self.annualize:
            view_returns = view_returns * scale

        price_std = np.maximum(q_high_prices - q_low_prices, 1e-8) / self._z_spread
        return_std = price_std / current_prices
        std_scale = float(np.sqrt(252.0 / self.horizon)) if self.annualize else 1.0
        if self.annualize:
            return_std = return_std * std_scale

        r_low = ((q_low_prices / current_prices) - 1.0) * scale
        r_high = ((q_high_prices / current_prices) - 1.0) * scale
        idr = np.maximum(r_high - r_low, 1e-6)
        skew = ((r_high - view_returns) - (view_returns - r_low)) / idr
        down_std = (
            np.maximum(p50_prices - q_low_prices, 1e-8)
            / (max(0.5 * self._z_spread, 1e-4) * current_prices)
        ) * std_scale
        cvar_95 = np.maximum(-view_returns + 2.0627 * down_std, 0.5 * return_std)

        self.last_interdecile_range_ = pd.Series(idr, index=assets)
        self.last_posterior_skewness_ = pd.Series(skew, index=assets)
        self.last_posterior_downside_std_ = pd.Series(down_std, index=assets)
        self.last_posterior_cvar_ = pd.Series(cvar_95, index=assets)

        if self.uncertainty_mode == "downside_semivariance":
            return_std = np.maximum(down_std, 1e-6)
        elif self.uncertainty_mode == "cvar_tail":
            return_std = np.maximum(cvar_95 / 2.0627, 0.5 * return_std)
            view_returns = (
                view_returns
                - self.tail_skew_aversion * np.maximum(0.0, -skew) * cvar_95
            )
        Q = pd.Series(view_returns, index=assets, name="expected_returns")

        Omega = pd.DataFrame(np.diag(return_std**2), index=assets, columns=assets)
        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)
        return P, Q, Omega


class TimesFM3MultivariateViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman views (`Q`) and uncertainty (`Omega`)
    using the probabilistic outputs of TimesFM 3.0's Multivariate setup,
    supporting both `MarketData.features` and `MarketData.future_features`
    via `CovariateMode` and `CovariateKey`, plus optional non-diagonal `Omega`.
    """

    def __init__(
        self,
        model: Any = None,
        horizon: int = 21,
        confidence_interval: tuple[float, float] = (0.10, 0.90),
        annualize: bool = True,
        full_covariance_omega: bool = False,
        covariate_mode: CovariateMode = CovariateMode.AUTO,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        uncertainty_mode: Literal[
            "posterior_variance", "downside_semivariance", "cvar_tail"
        ] = "posterior_variance",
        tail_skew_aversion: float = 0.15,
        device: str | None = None,
        tfm_model: Any = None,
    ) -> None:
        self.model = model if model is not None else tfm_model
        self.horizon = int(horizon)
        self.confidence_interval = confidence_interval
        self.annualize = bool(annualize)
        self.full_covariance_omega = bool(full_covariance_omega)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
        self.uncertainty_mode = uncertainty_mode
        self.tail_skew_aversion = float(max(0.0, tail_skew_aversion))
        self.device = device
        self._q_low, self._q_high, self._z_spread = (
            _validate_timesfm_confidence_interval(confidence_interval)
        )
        self.last_posterior_cvar_: pd.Series | None = None
        self.last_posterior_downside_std_: pd.Series | None = None
        self.last_posterior_skewness_: pd.Series | None = None
        self.last_interdecile_range_: pd.Series | None = None

    def generate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        stock_prices = data.prices
        assets = list(data.assets)
        n_assets = len(assets)
        current_prices = stock_prices.to_numpy(dtype=np.float64)[-1]

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
        quantiles = np.asarray(outputs[0].quantiles, dtype=np.float64)  # (N, H, 9)
        q_horizon = quantiles[:, self.horizon - 1, :]

        p50_prices = _interpolate_timesfm_quantile(q_horizon, 0.50)
        q_low_prices = _interpolate_timesfm_quantile(q_horizon, self._q_low)
        q_high_prices = _interpolate_timesfm_quantile(q_horizon, self._q_high)

        view_returns = (p50_prices / current_prices) - 1.0
        scale = (252.0 / self.horizon) if self.annualize else 1.0
        if self.annualize:
            view_returns = view_returns * scale

        price_std = np.maximum(q_high_prices - q_low_prices, 1e-8) / self._z_spread
        return_std = price_std / current_prices
        std_scale = float(np.sqrt(252.0 / self.horizon)) if self.annualize else 1.0
        if self.annualize:
            return_std = return_std * std_scale

        r_low = ((q_low_prices / current_prices) - 1.0) * scale
        r_high = ((q_high_prices / current_prices) - 1.0) * scale
        idr = np.maximum(r_high - r_low, 1e-6)
        skew = ((r_high - view_returns) - (view_returns - r_low)) / idr
        down_std = (
            np.maximum(p50_prices - q_low_prices, 1e-8)
            / (max(0.5 * self._z_spread, 1e-4) * current_prices)
        ) * std_scale
        cvar_95 = np.maximum(-view_returns + 2.0627 * down_std, 0.5 * return_std)

        self.last_interdecile_range_ = pd.Series(idr, index=assets)
        self.last_posterior_skewness_ = pd.Series(skew, index=assets)
        self.last_posterior_downside_std_ = pd.Series(down_std, index=assets)
        self.last_posterior_cvar_ = pd.Series(cvar_95, index=assets)

        if self.uncertainty_mode == "downside_semivariance":
            return_std = np.maximum(down_std, 1e-6)
        elif self.uncertainty_mode == "cvar_tail":
            return_std = np.maximum(cvar_95 / 2.0627, 0.5 * return_std)
            view_returns = (
                view_returns
                - self.tail_skew_aversion * np.maximum(0.0, -skew) * cvar_95
            )
        Q = pd.Series(view_returns, index=assets, name="expected_returns")

        if self.full_covariance_omega and n_assets > 1 and quantiles.shape[1] >= 2:
            # Compute cross-asset epistemic correlation from multi-horizon median trajectories
            med_traj = quantiles[:, :, P50_INDEX] / current_prices[:, None] - 1.0
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
