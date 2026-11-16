from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseReturnEstimator, BaseViewGenerator

WeightingMode = Literal["inverse_uncertainty", "scrps_adaptive", "equal", "custom"]


def compute_quantile_scrps(
    y_true: np.ndarray,
    quantiles_by_level: dict[float, np.ndarray],
) -> float:
    """
    Computes the scale-free Continuous Ranked Probability Score (`sCRPS`, Appendix B of
    Potapczynski et al., TMLR 2026) from a dictionary of quantile predictions:
        sCRPS(y, y_hat) = (2 / Q) sum_{j=1}^Q sum_i [ alpha_j (y_i - y_hat_i^{alpha_j})_+
                                                      + (1 - alpha_j)(y_hat_i^{alpha_j} - y_i)_+ ]
                          / sum_i |y_i|
    """
    if not quantiles_by_level:
        return 1.0
    y_arr = np.asarray(y_true, dtype=np.float64)
    pinball_total = 0.0
    for alpha_j, q_pred in quantiles_by_level.items():
        q_arr = np.asarray(q_pred, dtype=np.float64)
        diff = y_arr - q_arr
        loss_j = alpha_j * np.maximum(diff, 0.0) + (1.0 - alpha_j) * np.maximum(
            -diff, 0.0
        )
        pinball_total += float(np.sum(loss_j))
    numer = (2.0 / len(quantiles_by_level)) * pinball_total
    denom = max(float(np.sum(np.abs(y_arr))), 1e-6)
    return float(max(numer / denom, 1e-4))


class EnsembleReturnEstimator(BaseReturnEstimator):
    """
    Precision-Weighted & sCRPS-Adaptive Bayesian Model Averaging Return Estimator
    (`EnsembleReturnEstimator`).

    Combines multiple peer foundation models (`ApolloPFNReturnEstimator`,
    `TimesFM3MultivariateReturnEstimator`, `TimestralMultivariateReturnEstimator`,
    `NeuralSDEEstimator`) and/or classical return estimators & view generators:

        w_{m, i} = precision_{m, i} / sum_{k=1}^M precision_{k, i},
        mu_i^{ens} = sum_{m=1}^M w_{m, i} * mu_{m, i}

    Weighting Modes (`weighting_mode`):
    -----------------------------------
    1. `"inverse_uncertainty"` (default):
       Weights each model per asset by its inverse epistemic predictive variance
       `sigma_{m, i}^{-2}` (extracted from `last_posterior_std_`, view generator `diag(Omega)`,
       or `(p90 - p10) / 2.56` quantile spread).
    2. `"scrps_adaptive"`:
       Evaluates each model on a held-out trailing validation slice of `data` using the
       Appendix B `sCRPS` metric and weights each model by `(sigma_{m, i}^2 * sCRPS_m)^{-1}`.
    3. `"equal"`: Uniform `1 / M` weights across all models.
    4. `"custom"`: User-supplied `custom_weights` vector of length `M`.
    """

    def __init__(
        self,
        estimators: Sequence[BaseReturnEstimator | BaseViewGenerator],
        weighting_mode: WeightingMode = "inverse_uncertainty",
        custom_weights: Sequence[float] | None = None,
        validation_horizon: int = 10,
        model_names: Sequence[str] | None = None,
    ) -> None:
        if len(estimators) < 1:
            raise ValueError(
                "EnsembleReturnEstimator requires at least 1 estimator or view generator."
            )
        if weighting_mode not in (
            "inverse_uncertainty",
            "scrps_adaptive",
            "equal",
            "custom",
        ):
            raise ValueError(
                "`weighting_mode` must be 'inverse_uncertainty', 'scrps_adaptive', 'equal', or 'custom'."
            )
        self.estimators = list(estimators)
        self.weighting_mode: WeightingMode = weighting_mode
        self.custom_weights = (
            np.asarray(custom_weights, dtype=np.float64)
            if custom_weights is not None
            else None
        )
        self.validation_horizon = max(2, int(validation_horizon))
        self.model_names = (
            list(model_names)
            if model_names is not None
            else [
                f"{type(est).__name__}_{idx}" for idx, est in enumerate(self.estimators)
            ]
        )

        self.last_model_weights_: pd.DataFrame | None = None
        self.last_epistemic_std_: pd.Series | None = None
        self.last_posterior_std_: pd.Series | None = None
        self.last_scrps_scores_: pd.Series | None = None

    @staticmethod
    def _evaluate_single_model(
        est: BaseReturnEstimator | BaseViewGenerator,
        data: MarketData,
        assets: list[str],
        fallback_std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
        """
        Evaluates a single `BaseReturnEstimator` or `BaseViewGenerator` on `data`,
        returning `(mu_vec, std_vec, quantiles_dict)`.
        """
        if isinstance(est, BaseViewGenerator) and not isinstance(
            est, BaseReturnEstimator
        ):
            _, q_ser, omega_df = est.generate(data)
            mu_vec = q_ser.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            omega_np = (
                omega_df.reindex(index=assets, columns=assets)
                .fillna(0.01)
                .to_numpy(dtype=np.float64)
            )
            std_vec = np.sqrt(np.maximum(np.diag(omega_np), 1e-8))
            q_dict = {
                0.10: mu_vec - 1.28 * std_vec,
                0.50: mu_vec,
                0.90: mu_vec + 1.28 * std_vec,
            }
            return mu_vec, std_vec, q_dict

        assert isinstance(est, BaseReturnEstimator)
        mu_ser = est.estimate(data)
        mu_vec = mu_ser.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)

        post_std = getattr(est, "last_posterior_std_", None)
        if isinstance(post_std, pd.Series):
            std_vec = (
                post_std.reindex(assets)
                .fillna(pd.Series(fallback_std, index=assets))
                .to_numpy(dtype=np.float64)
            )
            std_vec = np.maximum(std_vec, 1e-6)
        else:
            std_vec = fallback_std.copy()

        raw_q = getattr(est, "last_quantiles_", None)
        q_dict = {}
        if isinstance(raw_q, dict) and len(raw_q) > 0:
            for q_lvl, s_q in raw_q.items():
                if isinstance(s_q, pd.Series):
                    q_dict[float(q_lvl)] = (
                        s_q.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
                    )
        if not q_dict:
            q_dict = {
                0.10: mu_vec - 1.28 * std_vec,
                0.50: mu_vec,
                0.90: mu_vec + 1.28 * std_vec,
            }
        return mu_vec, std_vec, q_dict

    def _compute_validation_scrps(
        self,
        data: MarketData,
        assets: list[str],
        fallback_std: np.ndarray,
    ) -> np.ndarray:
        """
        Computes trailing hold-out `sCRPS` for each model in `self.estimators`.
        """
        n_models = len(self.estimators)
        t_total = len(data.prices)
        h_val = min(self.validation_horizon, max(2, t_total // 5))
        if t_total < h_val + 8:
            return np.ones(n_models, dtype=np.float64)

        split_date = data.prices.index[-(h_val + 1)]
        train_md = data.slice_time(end_date=split_date, horizon_steps=h_val)
        p_start = data.prices.loc[split_date].reindex(assets).to_numpy(dtype=np.float64)
        p_end = data.prices.iloc[-1].reindex(assets).to_numpy(dtype=np.float64)
        realized_ann = ((p_end / np.maximum(p_start, 1e-8)) - 1.0) * (252.0 / h_val)

        scores = np.ones(n_models, dtype=np.float64)
        for m_idx, est in enumerate(self.estimators):
            try:
                _, _, q_dict = self._evaluate_single_model(
                    est, train_md, assets, fallback_std
                )
                scores[m_idx] = compute_quantile_scrps(realized_ann, q_dict)
            except (ValueError, KeyError, RuntimeError):
                scores[m_idx] = 1.0
        return np.clip(scores, 1e-4, 100.0)

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        assets = list(data.prices.columns)
        n_assets = len(assets)
        n_models = len(self.estimators)

        hist_rets = data.prices.pct_change().dropna()
        fallback_std = (
            hist_rets.std(ddof=1)
            .reindex(assets)
            .fillna(0.02)
            .to_numpy(dtype=np.float64)
            * np.sqrt(252.0)
            if len(hist_rets) >= 2
            else np.full(n_assets, 0.20, dtype=np.float64)
        )
        fallback_std = np.maximum(fallback_std, 1e-4)

        if self.weighting_mode == "scrps_adaptive":
            scrps_vec = self._compute_validation_scrps(data, assets, fallback_std)
        else:
            scrps_vec = np.ones(n_models, dtype=np.float64)

        mu_mat = np.empty((n_models, n_assets), dtype=np.float64)
        std_mat = np.empty((n_models, n_assets), dtype=np.float64)

        for m_idx, est in enumerate(self.estimators):
            mu_m, std_m, _ = self._evaluate_single_model(
                est, data, assets, fallback_std
            )
            mu_mat[m_idx] = mu_m
            std_mat[m_idx] = np.maximum(std_m, 1e-5)

        if self.weighting_mode == "equal":
            weights = np.full((n_models, n_assets), 1.0 / n_models, dtype=np.float64)
        elif self.weighting_mode == "custom":
            if self.custom_weights is None or len(self.custom_weights) != n_models:
                raise ValueError(
                    f"`custom_weights` must have length {n_models} matching `estimators`."
                )
            w_raw = np.maximum(self.custom_weights, 0.0)
            w_norm = w_raw / max(float(np.sum(w_raw)), 1e-12)
            weights = np.broadcast_to(w_norm[:, None], (n_models, n_assets)).copy()
        elif self.weighting_mode == "scrps_adaptive":
            prec = 1.0 / ((std_mat**2) * scrps_vec[:, None])
            weights = prec / np.sum(prec, axis=0, keepdims=True)
        else:
            # "inverse_uncertainty"
            prec = 1.0 / (std_mat**2)
            weights = prec / np.sum(prec, axis=0, keepdims=True)

        mu_ens = np.sum(weights * mu_mat, axis=0)
        # Law of total variance: within-model variance + between-model disagreement
        var_ens = np.sum(
            weights * ((std_mat**2) + (mu_mat - mu_ens[None, :]) ** 2), axis=0
        )
        std_ens = np.sqrt(np.maximum(var_ens, 1e-8))

        self.last_model_weights_ = pd.DataFrame(
            weights, index=self.model_names, columns=assets
        )
        self.last_epistemic_std_ = pd.Series(
            std_ens, index=assets, name="epistemic_std"
        )
        self.last_posterior_std_ = self.last_epistemic_std_.copy()
        self.last_scrps_scores_ = pd.Series(
            scrps_vec, index=self.model_names, name="sCRPS"
        )

        return pd.Series(mu_ens, index=assets, name="expected_returns")


class EnsembleViewGenerator(BaseViewGenerator):
    """
    Bayesian Precision-Pooling View Generator (`EnsembleViewGenerator`).

    Combines multiple `BaseViewGenerator` instances (such as `ApolloPFNViewGenerator`,
    `TimesFM3MultivariateViewGenerator`, and `TimestralMultivariateViewGenerator`) via
    exact Bayesian precision addition:

        Omega_ens = (sum_{m=1}^M Omega_m^{-1})^{-1}
        Q_ens     = Omega_ens @ (sum_{m=1}^M Omega_m^{-1} Q_m)
    """

    def __init__(
        self,
        view_generators: Sequence[BaseViewGenerator],
        model_names: Sequence[str] | None = None,
    ) -> None:
        if len(view_generators) < 1:
            raise ValueError(
                "EnsembleViewGenerator requires at least 1 view generator."
            )
        self.view_generators = list(view_generators)
        self.model_names = (
            list(model_names)
            if model_names is not None
            else [
                f"{type(vg).__name__}_{idx}"
                for idx, vg in enumerate(self.view_generators)
            ]
        )
        self.last_model_weights_: pd.DataFrame | None = None

    def generate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        assets = list(data.prices.columns)
        n_assets = len(assets)
        n_models = len(self.view_generators)

        prec_sum = np.zeros((n_assets, n_assets), dtype=np.float64)
        prec_q_sum = np.zeros(n_assets, dtype=np.float64)
        diag_precs = np.zeros((n_models, n_assets), dtype=np.float64)

        for m_idx, vg in enumerate(self.view_generators):
            _, q_ser, omega_df = vg.generate(data, **kwargs)
            q_vec = q_ser.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            omega_mat = (
                omega_df.reindex(index=assets, columns=assets)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            omega_mat = 0.5 * (omega_mat + omega_mat.T) + 1e-6 * np.eye(n_assets)
            omega_inv = np.linalg.inv(omega_mat)

            prec_sum += omega_inv
            prec_q_sum += omega_inv @ q_vec
            diag_precs[m_idx] = np.maximum(np.diag(omega_inv), 1e-8)

        omega_ens = np.linalg.inv(prec_sum + 1e-8 * np.eye(n_assets))
        omega_ens = 0.5 * (omega_ens + omega_ens.T)
        q_ens = omega_ens @ prec_q_sum

        weights = diag_precs / np.sum(diag_precs, axis=0, keepdims=True)
        self.last_model_weights_ = pd.DataFrame(
            weights, index=self.model_names, columns=assets
        )

        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)
        Q = pd.Series(q_ens, index=assets, name="expected_returns")
        Omega = pd.DataFrame(omega_ens, index=assets, columns=assets)
        return P, Q, Omega
