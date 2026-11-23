from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseCovarianceEstimator, BaseReturnEstimator
from pyalloq_features.regimes.jump_models import extract_jump_regime_features
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]


def compute_mahalanobis_relevance_weights(
    state_history: np.ndarray,
    subsample_fraction: float = 0.35,
    ridge_eps: float = 1e-4,
) -> tuple[np.ndarray, float, float, float]:
    """
    Computes Mahalanobis Relevance weights `w_{it}` and analytical `fit` / `adjusted_fit`
    confidence scores following Czasonis, Kritzman & Turkington (JFDS 2023; JPM 2023, 2024, 2026)
    and Kritzman, Kulasekaran & Turkington (2023):

        sim(x_i, x_t)  = -0.5 * (x_i - x_t) Omega^{-1} (x_i - x_t)'
        info(x, x_bar) = (x - x_bar) Omega^{-1} (x - x_bar)'
        r_{it}         = sim(x_i, x_t) + 0.5 * (info(x_i, x_bar) + info(x_t, x_bar))

    Observations below relevance quantile threshold `r^*` ( retaining top `subsample_fraction`
    of historical periods) are censored: `delta(r_{it}) = 1{r_{it} >= r^*}`.

    Returns
    -------
    tuple[np.ndarray, float, float, float]
        `(normalized_weights, relevance_threshold, fit_score, adjusted_fit_confidence)`
    """
    t_len, d_dim = state_history.shape
    if t_len <= 2:
        return np.full(max(1, t_len), 1.0 / max(1, t_len)), 0.0, 1.0, 1.0

    x_bar = np.mean(state_history, axis=0)
    centered = state_history - x_bar[None, :]
    cov_x = (centered.T @ centered) / max(1, t_len - 1) + ridge_eps * np.eye(d_dim)
    omega_inv = np.linalg.pinv(cov_x)

    x_t = state_history[-1]
    diff_t = state_history - x_t[None, :]
    sim_vec = -0.5 * np.einsum("ti,ij,tj->t", diff_t, omega_inv, diff_t)
    info_vec = np.einsum("ti,ij,tj->t", centered, omega_inv, centered)
    info_t = float(info_vec[-1])

    relevance = sim_vec + 0.5 * (info_vec + info_t)

    phi = float(np.clip(subsample_fraction, max(3.0 / t_len, 0.05), 1.0))
    r_star = float(np.quantile(relevance, 1.0 - phi))
    active_mask = relevance >= r_star

    # Positive importance weights over censored relevant subset
    shifted = np.where(active_mask, relevance - r_star + 1e-3, 0.0)
    w_sum = float(np.sum(shifted))
    if w_sum <= 1e-12:
        weights = np.full(t_len, 1.0 / t_len, dtype=np.float64)
    else:
        weights = shifted / w_sum

    # Analytical fit & adjusted_fit confidence (proportional to informative concentration)
    rel_centered = relevance - float(np.mean(relevance))
    rel_var = float(np.mean(rel_centered**2))
    info_norm = float(np.clip(info_t / max(float(d_dim), 1.0), 0.0, 4.0))
    fit_raw = float(np.clip(rel_var / (rel_var + float(d_dim)), 0.05, 0.98))

    # Asymmetry adjustment rewards high-informativeness current states with strong historical analogs
    pos_analog_ratio = (
        float(np.mean(relevance[active_mask] > 0.0)) if np.any(active_mask) else 0.5
    )
    adjusted_fit = float(
        np.clip(
            0.5 * fit_raw + 0.25 * (info_norm / 2.0) + 0.25 * pos_analog_ratio,
            0.05,
            1.0,
        )
    )
    return weights, r_star, fit_raw, adjusted_fit


class RelevanceConditionedCovarianceEstimator(BaseCovarianceEstimator):
    """
    Mahalanobis Relevance-Conditioned Covariance Estimator
    (`RelevanceConditionedCovarianceEstimator`, Czasonis, Kritzman & Turkington, 2023–2026).

    Rather than slicing a rolling window into tiny discrete HMM regime buckets (which causes
    severe sample starvation `T_k < N`), weights every historical return observation by its
    Mahalanobis similarity and informativeness relative to current market conditions `x_t`,
    applies optional Ledoit-Wolf shrinkage to the relevance-weighted scatter matrix, and uses
    `adjusted_fit` confidence to shrink tactical covariance tilts automatically toward the
    full-window baseline when current conditions lack strong historical analogs.
    """

    def __init__(
        self,
        subsample_fraction: float = 0.35,
        use_ledoit_wolf: bool = True,
        shrink_by_adjusted_fit: bool = True,
        windows: Sequence[int] = (5, 20, 60),
        feature_keys: Sequence[str] | None = None,
        annualization_factor: float = 252.0,
    ) -> None:
        self.subsample_fraction = float(np.clip(subsample_fraction, 0.05, 1.0))
        self.use_ledoit_wolf = bool(use_ledoit_wolf)
        self.shrink_by_adjusted_fit = bool(shrink_by_adjusted_fit)
        self.windows = tuple(int(w) for w in windows)
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.annualization_factor = float(annualization_factor)

        self.last_relevance_weights_: pd.Series | None = None
        self.last_fit_: float = 1.0
        self.last_adjusted_fit_: float = 1.0
        self.last_threshold_: float = 0.0

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.DataFrame:
        returns = data.prices.pct_change().dropna()
        assets = list(data.prices.columns)
        n_assets = len(assets)
        if len(returns) < 3:
            cov_emp = returns.cov().fillna(0.0) * self.annualization_factor
            return cov_emp.reindex(index=assets, columns=assets, fill_value=0.0)

        state_df = extract_jump_regime_features(
            returns=returns,
            windows=self.windows,
            feature_data=data.features,
            feature_keys=self.feature_keys,
        )
        state_mat = (
            state_df.reindex(returns.index).fillna(0.0).to_numpy(dtype=np.float64)
        )
        weights, r_star, fit_val, adj_fit = compute_mahalanobis_relevance_weights(
            state_mat, subsample_fraction=self.subsample_fraction
        )
        self.last_relevance_weights_ = pd.Series(
            weights, index=returns.index, name="relevance_weights"
        )
        self.last_threshold_ = r_star
        self.last_fit_ = fit_val
        self.last_adjusted_fit_ = adj_fit

        ret_np = returns.to_numpy(dtype=np.float64)
        mu_rel = weights @ ret_np
        centered_rel = ret_np - mu_rel[None, :]
        # Relevance-weighted covariance matrix
        denom = max(1.0 - float(np.sum(weights**2)), 1e-6)
        cov_rel = ((centered_rel.T * weights) @ centered_rel) / denom

        if self.use_ledoit_wolf and len(ret_np) >= 4:
            # Apply Ledoit-Wolf shrinkage intensity to relevance-weighted scatter
            lw = LedoitWolf().fit(ret_np)
            alpha_lw = float(np.clip(getattr(lw, "shrinkage_", 0.15), 0.02, 0.85))
            mu_trace = float(np.trace(cov_rel)) / max(1, n_assets)
            cov_rel = (1.0 - alpha_lw) * cov_rel + alpha_lw * mu_trace * np.eye(
                n_assets
            )
            cov_prior = lw.covariance_
        else:
            cov_prior = np.cov(ret_np, rowvar=False)

        if self.shrink_by_adjusted_fit:
            cov_final = adj_fit * cov_rel + (1.0 - adj_fit) * cov_prior
        else:
            cov_final = cov_rel

        cov_final = 0.5 * (cov_final + cov_final.T) * self.annualization_factor
        eigvals, eigvecs = np.linalg.eigh(cov_final)
        cov_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T

        df_cov = pd.DataFrame(cov_psd, index=assets, columns=assets)
        df_cov.attrs["adjusted_fit"] = adj_fit
        df_cov.attrs["fit"] = fit_val
        return df_cov


class RelevanceConditionedReturnEstimator(BaseReturnEstimator):
    """
    Mahalanobis Relevance-Conditioned Return Estimator
    (`RelevanceConditionedReturnEstimator`, Czasonis, Kritzman & Turkington, 2023–2026).

    Computes relevance-weighted expected returns from historical analogs matching the current
    state `x_t` and automatically shrinks tactical tilts toward the full-sample / prior baseline
    proportional to `adjusted_fit` when current conditions lack strong historical analogs.
    """

    def __init__(
        self,
        subsample_fraction: float = 0.35,
        shrink_by_adjusted_fit: bool = True,
        windows: Sequence[int] = (5, 20, 60),
        feature_keys: Sequence[str] | None = None,
        annualization_factor: float = 252.0,
    ) -> None:
        self.subsample_fraction = float(np.clip(subsample_fraction, 0.05, 1.0))
        self.shrink_by_adjusted_fit = bool(shrink_by_adjusted_fit)
        self.windows = tuple(int(w) for w in windows)
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.annualization_factor = float(annualization_factor)

        self.last_relevance_weights_: pd.Series | None = None
        self.last_fit_: float = 1.0
        self.last_adjusted_fit_: float = 1.0

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        returns = data.prices.pct_change().dropna()
        assets = list(data.prices.columns)
        if len(returns) < 3:
            mu_emp = returns.mean().fillna(0.0) * self.annualization_factor
            return mu_emp.reindex(assets, fill_value=0.0).rename("expected_returns")

        state_df = extract_jump_regime_features(
            returns=returns,
            windows=self.windows,
            feature_data=data.features,
            feature_keys=self.feature_keys,
        )
        state_mat = (
            state_df.reindex(returns.index).fillna(0.0).to_numpy(dtype=np.float64)
        )
        weights, _, fit_val, adj_fit = compute_mahalanobis_relevance_weights(
            state_mat, subsample_fraction=self.subsample_fraction
        )
        self.last_relevance_weights_ = pd.Series(
            weights, index=returns.index, name="relevance_weights"
        )
        self.last_fit_ = fit_val
        self.last_adjusted_fit_ = adj_fit

        ret_np = returns.to_numpy(dtype=np.float64)
        mu_rel = (weights @ ret_np) * self.annualization_factor
        mu_prior = np.mean(ret_np, axis=0) * self.annualization_factor

        if self.shrink_by_adjusted_fit:
            mu_final = adj_fit * mu_rel + (1.0 - adj_fit) * mu_prior
        else:
            mu_final = mu_rel

        return pd.Series(mu_final, index=assets, name="expected_returns")
