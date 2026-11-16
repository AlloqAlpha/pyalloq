from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import (
    BaseReturnEstimator,
    BaseScenarioEstimator,
    BaseViewGenerator,
)

WeightingMode = Literal[
    "inverse_uncertainty", "scrps_adaptive", "cvar_adaptive", "equal", "custom"
]


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


class EnsembleReturnEstimator(BaseScenarioEstimator, BaseReturnEstimator):
    """
    Precision-Weighted, sCRPS-Adaptive & Tail-CVaR-Adaptive Bayesian Model Averaging
    Return & Scenario Estimator (`EnsembleReturnEstimator`).

    Combines multiple peer foundation models (`ApolloPFNReturnEstimator`,
    `TimesFM3MultivariateReturnEstimator`, `TimestralMultivariateReturnEstimator`,
    `NeuralSDEEstimator`) and/or classical return estimators & view generators:

        w_{m, i} = precision_{m, i} / sum_{k=1}^M precision_{k, i},
        mu_i^{ens} = sum_{m=1}^M w_{m, i} * mu_{m, i}

    Also implements `BaseScenarioEstimator` (`.fit(data)` and `.generate(n_paths, horizon)`),
    synthesizing Bayesian model-averaged non-Gaussian scenario paths coupled via a
    Student-t or Gaussian copula (Demarta & McNeil, 2005; Meucci, 2006) so it can be
    passed directly as `scenario_generator=...` into `WassersteinDROAllocator`.

    Weighting Modes (`weighting_mode`):
    -----------------------------------
    1. `"inverse_uncertainty"` (default):
       Weights each model per asset by its inverse epistemic predictive variance
       `sigma_{m, i}^{-2}` (extracted from `last_posterior_std_`, view generator `diag(Omega)`,
       or `(p90 - p10) / 2.56` quantile spread).
    2. `"scrps_adaptive"`:
       Evaluates each model on a held-out trailing validation slice of `data` using the
       Appendix B `sCRPS` metric and weights each model by `(sigma_{m, i}^2 * sCRPS_m)^{-1}`.
    3. `"cvar_adaptive"`:
       Weights each model per asset by its inverse squared posterior Expected Shortfall
       `CVaR_{alpha, m, i}^{-2}` (Rockafellar & Uryasev, 2000; Müller et al., ICLR 2022).
    4. `"equal"`: Uniform `1 / M` weights across all models.
    5. `"custom"`: User-supplied `custom_weights` vector of length `M`.
    """

    def __init__(
        self,
        estimators: Sequence[BaseReturnEstimator | BaseViewGenerator],
        weighting_mode: WeightingMode = "inverse_uncertainty",
        custom_weights: Sequence[float] | None = None,
        validation_horizon: int = 10,
        model_names: Sequence[str] | None = None,
        copula_type: Literal["student_t", "gaussian"] = "student_t",
        copula_df: float = 4.0,
        random_state: int = 42,
    ) -> None:
        if len(estimators) < 1:
            raise ValueError(
                "EnsembleReturnEstimator requires at least 1 estimator or view generator."
            )
        if weighting_mode not in (
            "inverse_uncertainty",
            "scrps_adaptive",
            "cvar_adaptive",
            "equal",
            "custom",
        ):
            raise ValueError(
                "`weighting_mode` must be 'inverse_uncertainty', 'scrps_adaptive', 'cvar_adaptive', 'equal', or 'custom'."
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
        self.copula_type = copula_type
        self.copula_df = float(max(2.1, copula_df))
        self.random_state = int(random_state)

        self.is_fitted: bool = False
        self.assets: list[str] = []
        self._last_data: MarketData | None = None
        self._last_mu_ens: pd.Series | None = None
        self.last_model_weights_: pd.DataFrame | None = None
        self.last_epistemic_std_: pd.Series | None = None
        self.last_posterior_std_: pd.Series | None = None
        self.last_posterior_cvar_: pd.Series | None = None
        self.last_posterior_downside_std_: pd.Series | None = None
        self.last_posterior_skewness_: pd.Series | None = None
        self.last_interdecile_range_: pd.Series | None = None
        self.last_scenario_probabilities_: np.ndarray | None = None
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
        cvar_mat = np.empty((n_models, n_assets), dtype=np.float64)
        down_mat = np.empty((n_models, n_assets), dtype=np.float64)
        skew_mat = np.empty((n_models, n_assets), dtype=np.float64)
        idr_mat = np.empty((n_models, n_assets), dtype=np.float64)

        for m_idx, est in enumerate(self.estimators):
            mu_m, std_m, q_dict = self._evaluate_single_model(
                est, data, assets, fallback_std
            )
            mu_mat[m_idx] = mu_m
            std_mat[m_idx] = np.maximum(std_m, 1e-5)

            q10 = q_dict.get(0.10, mu_m - 1.2816 * std_mat[m_idx])
            q50 = q_dict.get(0.50, mu_m)
            q90 = q_dict.get(0.90, mu_m + 1.2816 * std_mat[m_idx])
            default_idr = np.maximum(q90 - q10, 1e-5)
            default_skew = ((q90 - q50) - (q50 - q10)) / default_idr
            default_down = np.maximum(q50 - q10, 1e-5) / 1.2816
            default_cvar = np.maximum(
                -q50 + 2.0627 * default_down, 0.5 * std_mat[m_idx]
            )

            cvar_ser = getattr(est, "last_posterior_cvar_", None)
            cvar_mat[m_idx] = (
                np.maximum(
                    cvar_ser.reindex(assets)
                    .fillna(pd.Series(default_cvar, index=assets))
                    .to_numpy(dtype=np.float64),
                    1e-5,
                )
                if isinstance(cvar_ser, pd.Series)
                else default_cvar
            )
            down_ser = getattr(est, "last_posterior_downside_std_", None)
            down_mat[m_idx] = (
                np.maximum(
                    down_ser.reindex(assets)
                    .fillna(pd.Series(default_down, index=assets))
                    .to_numpy(dtype=np.float64),
                    1e-5,
                )
                if isinstance(down_ser, pd.Series)
                else default_down
            )
            skew_ser = getattr(est, "last_posterior_skewness_", None)
            skew_mat[m_idx] = (
                skew_ser.reindex(assets)
                .fillna(pd.Series(default_skew, index=assets))
                .to_numpy(dtype=np.float64)
                if isinstance(skew_ser, pd.Series)
                else default_skew
            )
            idr_ser = getattr(est, "last_interdecile_range_", None)
            idr_mat[m_idx] = (
                np.maximum(
                    idr_ser.reindex(assets)
                    .fillna(pd.Series(default_idr, index=assets))
                    .to_numpy(dtype=np.float64),
                    1e-5,
                )
                if isinstance(idr_ser, pd.Series)
                else default_idr
            )

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
        elif self.weighting_mode == "cvar_adaptive":
            prec = 1.0 / (cvar_mat**2)
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
        cvar_ens = np.sum(weights * cvar_mat, axis=0)
        down_ens = np.sqrt(
            np.sum(
                weights
                * ((down_mat**2) + np.maximum(0.0, mu_ens[None, :] - mu_mat) ** 2),
                axis=0,
            )
        )
        skew_ens = np.sum(weights * skew_mat, axis=0)
        idr_ens = np.sum(weights * idr_mat, axis=0) + 1.28 * np.sqrt(
            np.sum(weights * ((mu_mat - mu_ens[None, :]) ** 2), axis=0)
        )

        self.assets = assets
        self._last_data = data
        self._record_index_metadata(data.prices.index)
        self.is_fitted = True
        self.last_model_weights_ = pd.DataFrame(
            weights, index=self.model_names, columns=assets
        )
        self.last_epistemic_std_ = pd.Series(
            std_ens, index=assets, name="epistemic_std"
        )
        self.last_posterior_std_ = self.last_epistemic_std_.copy()
        self.last_posterior_cvar_ = pd.Series(
            cvar_ens, index=assets, name="posterior_cvar"
        )
        self.last_posterior_downside_std_ = pd.Series(
            down_ens, index=assets, name="posterior_downside_std"
        )
        self.last_posterior_skewness_ = pd.Series(
            skew_ens, index=assets, name="posterior_skewness"
        )
        self.last_interdecile_range_ = pd.Series(
            idr_ens, index=assets, name="interdecile_range"
        )
        self.last_scrps_scores_ = pd.Series(
            scrps_vec, index=self.model_names, name="sCRPS"
        )
        self._last_mu_ens = pd.Series(mu_ens, index=assets, name="expected_returns")
        return self._last_mu_ens.copy()

    def fit(self, data: MarketData) -> None:
        self.estimate(data)

    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        """
        Synthesizes 3D `(n_paths, horizon, n_assets)` Bayesian Model-Averaged non-Gaussian
        scenario trajectories by combining constituent `BaseScenarioEstimator` Riemann Bar
        marginals and split-normal asymmetric quantile marginals via a Student-t or Gaussian
        copula (Demarta & McNeil, 2005; Meucci, 2006).
        """
        if not self.is_fitted or self._last_data is None:
            raise ValueError(
                "EnsembleReturnEstimator must be fitted via .fit(data) or .estimate_scenarios(data) first."
            )
        from scipy.stats import norm
        from scipy.stats import t as student_t

        data = self._last_data
        assets = self.assets
        n_assets = len(assets)
        rng = np.random.default_rng(self.random_state)

        if self._last_mu_ens is None:
            self.estimate(data)
        assert (
            self._last_mu_ens is not None
            and self.last_epistemic_std_ is not None
            and self.last_posterior_skewness_ is not None
        )

        # If any constituent is a BaseScenarioEstimator (e.g. ApolloPFNReturnEstimator),
        # blend its copula-coupled Riemann Bar scenarios with the ensemble's posterior moments & skewness
        scen_children = [
            est for est in self.estimators if isinstance(est, BaseScenarioEstimator)
        ]
        if len(scen_children) > 0:
            child = scen_children[0]
            if not getattr(child, "is_fitted", False):
                child.fit(data)
            base_scen = child.generate(n_paths=n_paths, horizon=horizon)
            R_base = np.asarray(base_scen.returns, dtype=np.float64)
        else:
            R_base = None

        returns_df = (
            data.prices.pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .dropna(how="all")
            .fillna(0.0)
        )
        ret_mat = np.nan_to_num(
            returns_df.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        if n_assets > 1 and len(ret_mat) >= 2:
            r_cent = ret_mat - np.mean(ret_mat, axis=0, keepdims=True)
            r_norm = np.linalg.norm(r_cent, axis=0, keepdims=True)
            r_unit = np.where(r_norm > 1e-8, r_cent / np.maximum(r_norm, 1e-8), 0.0)
            emp_corr = r_unit.T @ r_unit
            emp_corr = 0.5 * (emp_corr + emp_corr.T)
            np.fill_diagonal(emp_corr, 1.0)
            eigvals, eigvecs = np.linalg.eigh(emp_corr)
            corr_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-4)) @ eigvecs.T
        else:
            corr_psd = np.eye(n_assets, dtype=np.float64)
        chol = np.linalg.cholesky(corr_psd + 1e-6 * np.eye(n_assets))

        mu_ens_vec = self._last_mu_ens.reindex(assets).to_numpy(dtype=np.float64)
        mu_daily = (
            mu_ens_vec / 252.0
            if getattr(self.estimators[0], "annualize", True)
            else mu_ens_vec / max(1, horizon)
        )
        std_daily = self.last_epistemic_std_.reindex(assets).to_numpy(
            dtype=np.float64
        ) / np.sqrt(252.0)
        skew_vec = self.last_posterior_skewness_.reindex(assets).to_numpy(
            dtype=np.float64
        )

        nu = self.copula_df
        R_copula = np.empty((n_paths, horizon, n_assets), dtype=np.float64)
        for t_step in range(horizon):
            z_corr = rng.standard_normal(size=(n_paths, n_assets)) @ chol.T
            if self.copula_type == "student_t":
                chi2_s = rng.chisquare(df=nu, size=(n_paths, 1)) / nu
                t_var = z_corr / np.sqrt(np.maximum(chi2_s, 1e-6))
                u_unif = np.clip(student_t.cdf(t_var, df=nu), 1e-4, 1.0 - 1e-4)
                z_std = norm.ppf(u_unif)
            else:
                z_std = z_corr
            # Apply Cornish-Fisher / split-normal skewness shift from pooled Bowley skewness
            z_skewed = z_std + (skew_vec[None, :] / 6.0) * (z_std**2 - 1.0)
            R_copula[:, t_step, :] = mu_daily[None, :] + std_daily[None, :] * z_skewed

        if R_base is not None and R_base.shape == R_copula.shape:
            # Center R_base around the Bayesian ensemble daily mean while preserving Riemann Bar tails
            base_mean = np.mean(R_base, axis=(0, 1), keepdims=True)
            R_final = (
                0.65 * (R_base - base_mean + mu_daily[None, None, :]) + 0.35 * R_copula
            ).astype(np.float32)
        else:
            R_final = R_copula.astype(np.float32)

        self.last_scenario_probabilities_ = np.full(
            n_paths, 1.0 / max(1, n_paths), dtype=np.float64
        )
        init_prices = data.prices.iloc[-1].to_numpy(dtype=np.float32)
        prices_sim = init_prices * np.cumprod(1.0 + R_final, axis=1)
        return ScenarioMarketData(
            prices=prices_sim.astype(np.float32, copy=False),
            returns=R_final,
            asset_names=list(assets),
            initial_prices=init_prices.copy(),
            timestamps=self._build_future_timestamps(horizon),
        )


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
        self.last_posterior_cvar_: pd.Series | None = None
        self.last_posterior_downside_std_: pd.Series | None = None
        self.last_posterior_skewness_: pd.Series | None = None
        self.last_interdecile_range_: pd.Series | None = None

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
        cvar_mat = np.zeros((n_models, n_assets), dtype=np.float64)
        down_mat = np.zeros((n_models, n_assets), dtype=np.float64)
        skew_mat = np.zeros((n_models, n_assets), dtype=np.float64)
        idr_mat = np.zeros((n_models, n_assets), dtype=np.float64)

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
            std_vec = np.sqrt(np.maximum(np.diag(omega_mat), 1e-8))

            prec_sum += omega_inv
            prec_q_sum += omega_inv @ q_vec
            diag_precs[m_idx] = np.maximum(np.diag(omega_inv), 1e-8)

            cvar_s = getattr(vg, "last_posterior_cvar_", None)
            cvar_mat[m_idx] = (
                cvar_s.reindex(assets).to_numpy(dtype=np.float64)
                if isinstance(cvar_s, pd.Series)
                else np.maximum(-q_vec + 2.0627 * std_vec, 0.5 * std_vec)
            )
            down_s = getattr(vg, "last_posterior_downside_std_", None)
            down_mat[m_idx] = (
                down_s.reindex(assets).to_numpy(dtype=np.float64)
                if isinstance(down_s, pd.Series)
                else std_vec
            )
            skew_s = getattr(vg, "last_posterior_skewness_", None)
            skew_mat[m_idx] = (
                skew_s.reindex(assets).to_numpy(dtype=np.float64)
                if isinstance(skew_s, pd.Series)
                else np.zeros(n_assets, dtype=np.float64)
            )
            idr_s = getattr(vg, "last_interdecile_range_", None)
            idr_mat[m_idx] = (
                idr_s.reindex(assets).to_numpy(dtype=np.float64)
                if isinstance(idr_s, pd.Series)
                else 2.56 * std_vec
            )

        omega_ens = np.linalg.inv(prec_sum + 1e-8 * np.eye(n_assets))
        omega_ens = 0.5 * (omega_ens + omega_ens.T)
        q_ens = omega_ens @ prec_q_sum

        weights = diag_precs / np.sum(diag_precs, axis=0, keepdims=True)
        self.last_model_weights_ = pd.DataFrame(
            weights, index=self.model_names, columns=assets
        )
        self.last_posterior_cvar_ = pd.Series(
            np.sum(weights * cvar_mat, axis=0), index=assets, name="posterior_cvar"
        )
        self.last_posterior_downside_std_ = pd.Series(
            np.sum(weights * down_mat, axis=0),
            index=assets,
            name="posterior_downside_std",
        )
        self.last_posterior_skewness_ = pd.Series(
            np.sum(weights * skew_mat, axis=0), index=assets, name="posterior_skewness"
        )
        self.last_interdecile_range_ = pd.Series(
            np.sum(weights * idr_mat, axis=0), index=assets, name="interdecile_range"
        )

        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)
        Q = pd.Series(q_ens, index=assets, name="expected_returns")
        Omega = pd.DataFrame(omega_ens, index=assets, columns=assets)
        return P, Q, Omega
