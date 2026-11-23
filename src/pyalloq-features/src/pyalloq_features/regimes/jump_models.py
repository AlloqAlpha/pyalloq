from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseRegimeDetector
from pyalloq_core.results import RegimeState

from pyalloq_features.core.base import BaseFeatureTransformer

TuningCriterion = Literal["none", "ftic", "abic", "ts_cv"]


def extract_jump_regime_features(
    returns: pd.DataFrame,
    windows: Sequence[int] = (5, 20, 60),
    feature_data: dict[str, pd.DataFrame] | None = None,
    feature_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Constructs a standardized multivariate regime feature matrix from asset returns
    following Shu, Yu, and Mulvey (2024, 2025) and practitioner replications:
      - Equal-weight market return & EWMA momentum across `windows`,
      - Rolling realized volatility & downside deviation (strongest regime separator),
      - Rolling 60-day Sortino ratio & drawdown depth,
      - Cross-sectional return dispersion,
      - Optional exogenous `feature_keys` from `MarketData.features`.
    """
    clean_ret = returns.dropna(how="all").fillna(0.0)
    idx = clean_ret.index
    t_len = len(clean_ret)
    if t_len == 0:
        return pd.DataFrame(index=idx)

    mkt_ret = clean_ret.mean(axis=1)
    cs_disp = (
        clean_ret.std(axis=1).fillna(0.0)
        if clean_ret.shape[1] > 1
        else pd.Series(0.0, index=idx)
    )

    feats: dict[str, pd.Series] = {
        "mkt_return": mkt_ret,
        "cs_dispersion": cs_disp,
    }

    cum_eq = (1.0 + mkt_ret).cumprod()
    peak_eq = cum_eq.cummax().replace(0.0, np.nan)
    drawdown = ((cum_eq - peak_eq) / peak_eq).fillna(0.0)
    feats["drawdown"] = drawdown

    neg_ret = mkt_ret.clip(upper=0.0)
    for w in windows:
        w_eff = max(2, min(int(w), max(2, t_len // 2)))
        min_p = max(1, w_eff // 3)
        ewma_ret = mkt_ret.ewm(span=w_eff, min_periods=1).mean()
        roll_vol = mkt_ret.rolling(w_eff, min_periods=min_p).std().bfill().fillna(0.01)
        down_dev = (
            np.sqrt(
                (neg_ret**2)
                .rolling(w_eff, min_periods=min_p)
                .mean()
                .bfill()
                .fillna(1e-4)
            )
            + 1e-5
        )
        sortino = (ewma_ret / down_dev).clip(-5.0, 5.0)
        feats[f"ewma_ret_{w}"] = ewma_ret
        feats[f"realized_vol_{w}"] = roll_vol
        feats[f"downside_dev_{w}"] = down_dev
        feats[f"sortino_{w}"] = sortino

    if feature_data and feature_keys:
        for k in feature_keys:
            if k in feature_data:
                df_f = feature_data[k].reindex(idx).ffill().bfill().fillna(0.0)
                for col in df_f.columns:
                    feats[f"{k}_{col}"] = df_f[col]

    df_out = pd.DataFrame(feats, index=idx).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df_out


def _witten_tibshirani_l1_weights(
    bcss_vec: np.ndarray,
    kappa: float,
    n_bisect: int = 40,
) -> np.ndarray:
    """
    Solves the Witten & Tibshirani (2010) / Nystrup, Kolm & Lindström (2021) sparse feature
    weighting subproblem:
        max_w  w^T b   s.t.  ||w||_2 <= 1,  ||w||_1 <= kappa,  w >= 0
    via exact soft-thresholding bisection.
    """
    d = len(bcss_vec)
    b_pos = np.maximum(np.asarray(bcss_vec, dtype=np.float64), 0.0)
    norm_b = float(np.linalg.norm(b_pos, ord=2))
    if norm_b <= 1e-12:
        return np.full(d, 1.0 / np.sqrt(max(1, d)), dtype=np.float64)

    w_unconstrained = b_pos / norm_b
    kappa_eff = float(np.clip(kappa, 1.0, np.sqrt(float(d))))
    if float(np.sum(w_unconstrained)) <= kappa_eff + 1e-9:
        return w_unconstrained

    low_delta = 0.0
    high_delta = float(np.max(b_pos))
    for _ in range(n_bisect):
        mid = 0.5 * (low_delta + high_delta)
        soft = np.maximum(b_pos - mid, 0.0)
        s_norm = float(np.linalg.norm(soft, ord=2))
        if s_norm <= 1e-12:
            high_delta = mid
            continue
        w_cand = soft / s_norm
        if float(np.sum(w_cand)) > kappa_eff:
            low_delta = mid
        else:
            high_delta = mid

    soft = np.maximum(b_pos - 0.5 * (low_delta + high_delta), 0.0)
    s_norm = float(np.linalg.norm(soft, ord=2))
    if s_norm <= 1e-12:
        idx_max = int(np.argmax(b_pos))
        w_one = np.zeros(d, dtype=np.float64)
        w_one[idx_max] = 1.0
        return w_one
    return soft / s_norm


def _solve_discrete_jump_dp(
    loss_matrix: np.ndarray,
    jump_penalty: float,
) -> tuple[np.ndarray, float]:
    """
    Solves the exact O(T * K) dynamic programming Bellman recursion for discrete Jump Models:
        C_t(k) = l(y_t, theta_k) + min(C_{t-1}(k), min_{j != k} C_{t-1}(j) + lambda)
    Returns `(states, total_objective_cost)`.
    """
    t_len, k_states = loss_matrix.shape
    if t_len == 0:
        return np.zeros(0, dtype=np.int64), 0.0
    if k_states <= 1:
        return np.zeros(t_len, dtype=np.int64), float(np.sum(loss_matrix))

    lam = max(0.0, float(jump_penalty))
    cost_dp = np.zeros((t_len, k_states), dtype=np.float64)
    backptr = np.zeros((t_len, k_states), dtype=np.int64)
    cost_dp[0] = loss_matrix[0]

    for t in range(1, t_len):
        prev = cost_dp[t - 1]
        min_idx = int(np.argmin(prev))
        min_val = prev[min_idx]
        # Second minimum for j != min_idx
        temp = prev.copy()
        temp[min_idx] = np.inf
        sec_idx = int(np.argmin(temp))
        sec_val = temp[sec_idx]

        for k in range(k_states):
            stay_cost = prev[k]
            other_min = sec_val if k == min_idx else min_val
            other_idx = sec_idx if k == min_idx else min_idx
            switch_cost = other_min + lam
            if stay_cost <= switch_cost:
                cost_dp[t, k] = loss_matrix[t, k] + stay_cost
                backptr[t, k] = k
            else:
                cost_dp[t, k] = loss_matrix[t, k] + switch_cost
                backptr[t, k] = other_idx

    states = np.zeros(t_len, dtype=np.int64)
    states[-1] = int(np.argmin(cost_dp[-1]))
    total_cost = float(cost_dp[-1, states[-1]])
    for t in range(t_len - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]
    return states, total_cost


class SparseJumpModelDetector(BaseRegimeDetector, BaseFeatureTransformer):
    """
    Statistical & Sparse Jump Model Regime Detector (`SparseJumpModelDetector`,
    Bemporad et al., 2018; Nystrup, Kolm & Lindström, 2021; Shu, Yu & Mulvey, 2024–2025;
    Cortese et al., 2023, 2024).

    Solves the L0-penalized temporal clustering & Witten-Tibshirani sparse feature selection
    problem:
        min_{Theta, S, w}  sum_{t=1}^T sum_{d=1}^D w_d (y_{t, d} - theta_{s_t, d})^2
                           + lambda * sum_{t=2}^T 1{s_t != s_{t-1}}
        subject to:  ||w||_2 <= 1,  ||w||_1 <= kappa,  w >= 0.

    Supports:
    - Coordinate descent alternating between centroid updates, exact O(T K) dynamic programming,
      and Witten-Tibshirani BCSS soft-thresholded feature weights.
    - Automated hyperparameter selection over `(n_regimes_grid, jump_penalty_grid, sparsity_kappa_grid)`
      via Cortese et al. (2023, 2024) Generalized Information Criteria (`tuning_criterion="ftic"`
      or `"abic"`) or downstream risk-adjusted strategy performance (`tuning_criterion="ts_cv"`).
    - Dual inheritance from `BaseRegimeDetector` (`.detect(data) -> RegimeState`) and
      `BaseFeatureTransformer` (`.transform(data) -> MarketData` appending `regime_probs`
      and `regime_state` into `data.features`).
    """

    def __init__(
        self,
        n_regimes: int = 3,
        jump_penalty: float = 15.0,
        sparsity_kappa: float | None = 2.0,
        tuning_criterion: TuningCriterion = "none",
        n_regimes_grid: Sequence[int] | None = None,
        jump_penalty_grid: Sequence[float] | None = None,
        sparsity_kappa_grid: Sequence[float] | None = None,
        windows: Sequence[int] = (5, 20, 60),
        feature_keys: Sequence[str] | None = None,
        max_iter: int = 30,
        tol: float = 1e-5,
        prob_temperature: float = 1.0,
        output_probs_key: str = "regime_probs",
        output_state_key: str = "regime_state",
        annualization_factor: float = 252.0,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = max(2, int(n_regimes))
        self.jump_penalty = max(0.0, float(jump_penalty))
        self.sparsity_kappa = (
            float(sparsity_kappa) if sparsity_kappa is not None else None
        )
        self.tuning_criterion: TuningCriterion = tuning_criterion
        self.n_regimes_grid = (
            [max(2, int(k)) for k in n_regimes_grid]
            if n_regimes_grid is not None
            else [self.n_regimes]
        )
        self.jump_penalty_grid = (
            [max(0.0, float(lam)) for lam in jump_penalty_grid]
            if jump_penalty_grid is not None
            else [5.0, 15.0, 35.0]
        )
        self.sparsity_kappa_grid = (
            [float(kp) for kp in sparsity_kappa_grid]
            if sparsity_kappa_grid is not None
            else (
                [self.sparsity_kappa]
                if self.sparsity_kappa is not None
                else [1.5, 2.2, 3.0]
            )
        )
        self.windows = tuple(int(w) for w in windows)
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.max_iter = max(3, int(max_iter))
        self.tol = float(max(1e-8, tol))
        self.prob_temperature = float(max(1e-3, prob_temperature))
        self.output_probs_key = output_probs_key
        self.output_state_key = output_state_key
        self.annualization_factor = float(annualization_factor)
        self.random_state = int(random_state)

        self.is_fitted: bool = False
        self.feature_names_: list[str] = []
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.feature_weights_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        self.selected_k_: int = self.n_regimes
        self.selected_lambda_: float = self.jump_penalty
        self.selected_kappa_: float | None = self.sparsity_kappa
        self.last_state_: RegimeState | None = None

    def _prepare_standardized_features(
        self, data: MarketData, fit_scaler: bool = False
    ) -> tuple[np.ndarray, pd.Index, pd.Series]:
        returns = data.prices.pct_change().fillna(0.0)
        df_feat = extract_jump_regime_features(
            returns=returns,
            windows=self.windows,
            feature_data=data.features,
            feature_keys=self.feature_keys,
        )
        if fit_scaler or not self.feature_names_:
            self.feature_names_ = list(df_feat.columns)
        else:
            df_feat = df_feat.reindex(columns=self.feature_names_, fill_value=0.0)

        x_raw = df_feat.to_numpy(dtype=np.float64)
        if fit_scaler or self.feature_mean_ is None or self.feature_std_ is None:
            self.feature_mean_ = np.mean(x_raw, axis=0)
            std = np.std(x_raw, axis=0, ddof=0)
            self.feature_std_ = np.where(std > 1e-6, std, 1.0)

        x_std = np.clip((x_raw - self.feature_mean_) / self.feature_std_, -6.0, 6.0)
        mkt_ret = returns.mean(axis=1)
        return x_std, df_feat.index, mkt_ret

    def _fit_single_config(
        self,
        x_std: np.ndarray,
        mkt_ret: np.ndarray,
        k_states: int,
        lam: float,
        kappa: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Fits SJM for a single `(K, lambda, kappa)` configuration via coordinate descent.
        Returns `(states, centroids, feature_weights, raw_loss)`.
        """
        t_len, d_feat = x_std.shape
        k_eff = min(k_states, max(2, t_len // 3))

        w = np.full(d_feat, 1.0 / np.sqrt(max(1, d_feat)), dtype=np.float64)
        # Initialize centroids via quantiles along the first feature / volatility proxy
        vol_col = min(2, d_feat - 1)
        quantiles = np.linspace(0.15, 0.85, k_eff)
        q_idx = [
            int(
                np.argmin(np.abs(x_std[:, vol_col] - np.quantile(x_std[:, vol_col], q)))
            )
            for q in quantiles
        ]
        centroids = x_std[q_idx].copy()
        states = np.zeros(t_len, dtype=np.int64)

        total_ss = np.sum((x_std - np.mean(x_std, axis=0, keepdims=True)) ** 2, axis=0)
        prev_obj = np.inf

        for _it in range(self.max_iter):
            # 1. Compute feature-weighted squared distance loss matrix L_{t, k}
            diff = x_std[:, None, :] - centroids[None, :, :]  # (T, K, D)
            loss_mat = 0.5 * np.sum((diff**2) * w[None, None, :], axis=2)

            # 2. Exact dynamic programming state sequence decode
            states, obj_val = _solve_discrete_jump_dp(loss_mat, jump_penalty=lam)

            # 3. Centroid update per state
            for k in range(k_eff):
                mask_k = states == k
                if np.any(mask_k):
                    centroids[k] = np.mean(x_std[mask_k], axis=0)

            # 4. Sparse feature weight update via Witten-Tibshirani BCSS
            if kappa is not None and d_feat > 1:
                wcss = np.zeros(d_feat, dtype=np.float64)
                for k in range(k_eff):
                    mask_k = states == k
                    if np.any(mask_k):
                        wcss += np.sum((x_std[mask_k] - centroids[k]) ** 2, axis=0)
                bcss = np.maximum(total_ss - wcss, 0.0)
                w_new = _witten_tibshirani_l1_weights(bcss, kappa=kappa)
                if (
                    np.linalg.norm(w_new - w) < self.tol
                    and abs(prev_obj - obj_val) < self.tol
                ):
                    w = w_new
                    break
                w = w_new
            elif abs(prev_obj - obj_val) < self.tol:
                break
            prev_obj = obj_val

        # Canonically order regimes from lowest downside risk / highest Sortino to highest risk (crisis)
        regime_risk = np.zeros(k_eff, dtype=np.float64)
        for k in range(k_eff):
            mask_k = states == k
            if np.any(mask_k):
                r_k = mkt_ret[mask_k]
                vol_k = float(np.std(r_k)) if len(r_k) > 1 else 0.01
                mean_k = float(np.mean(r_k))
                regime_risk[k] = vol_k - 0.5 * mean_k
            else:
                regime_risk[k] = float(k)

        order = np.argsort(regime_risk)
        inv_order = np.zeros(k_eff, dtype=np.int64)
        for new_k, old_k in enumerate(order):
            inv_order[old_k] = new_k

        states_ordered = inv_order[states]
        centroids_ordered = centroids[order]
        return states_ordered, centroids_ordered, w, prev_obj

    def _score_configuration(
        self,
        x_std: np.ndarray,
        mkt_ret: np.ndarray,
        states: np.ndarray,
        centroids: np.ndarray,
        w: np.ndarray,
        k_states: int,
    ) -> float:
        """
        Evaluates configuration quality. Lower is better (minimization scale).
        Supports:
        - `"ftic"` / `"abic"` (Cortese et al. 2023, 2024 Generalized Information Criteria)
        - `"ts_cv"` (Negative downstream regime-conditioned Sharpe ratio net of 10 bps switch cost)
        """
        t_len, d_feat = x_std.shape
        n_jumps = int(np.sum(states[1:] != states[:-1]))
        active_feats = int(np.sum(w > 1e-3))

        if self.tuning_criterion == "ts_cv":
            # Causal walk-forward validation: bull regime (0) takes 1.0 exposure, neutral (1) 0.5, bear (K-1) 0.1
            exposures = np.linspace(1.0, 0.1, k_states)[states]
            # Lagged by 1 step for strict causality + 10 bps turnover penalty
            pos = np.roll(exposures, 1)
            pos[0] = 1.0
            strat_ret = pos * mkt_ret - 0.0010 * np.abs(np.diff(pos, prepend=1.0))
            vol = float(np.std(strat_ret, ddof=1))
            sharpe = (
                float(np.mean(strat_ret)) / vol * np.sqrt(self.annualization_factor)
                if vol > 1e-8
                else -10.0
            )
            return -sharpe

        diff = x_std - centroids[states]
        weighted_rss = float(np.sum((diff**2) * w[None, :]))
        sigma2 = max(weighted_rss / max(1, t_len), 1e-6)
        n_params = k_states * max(1, active_feats) + n_jumps
        if self.tuning_criterion == "abic":
            penalty = n_params * np.log(max(2, t_len))
        else:
            # FTIC: penalizes both state switches and feature dimension with log(log(T * D))
            penalty = (
                n_params
                * np.log(max(2, t_len))
                * np.log(max(2.0, np.log(max(3, t_len * d_feat))))
            )
        return float(t_len * np.log(sigma2) + penalty)

    def fit(self, data: MarketData) -> SparseJumpModelDetector:
        x_std, _, mkt_ret_ser = self._prepare_standardized_features(
            data, fit_scaler=True
        )
        mkt_ret = mkt_ret_ser.to_numpy(dtype=np.float64)

        if self.tuning_criterion == "none":
            states, centroids, w, _ = self._fit_single_config(
                x_std=x_std,
                mkt_ret=mkt_ret,
                k_states=self.n_regimes,
                lam=self.jump_penalty,
                kappa=self.sparsity_kappa,
            )
            self.selected_k_ = len(centroids)
            self.selected_lambda_ = self.jump_penalty
            self.selected_kappa_ = self.sparsity_kappa
            self.centroids_ = centroids
            self.feature_weights_ = w
        else:
            best_score = np.inf
            best_tuple: (
                tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float | None]
                | None
            ) = None
            for k_cand in self.n_regimes_grid:
                for lam_cand in self.jump_penalty_grid:
                    for kap_cand in self.sparsity_kappa_grid:
                        states, centroids, w, _ = self._fit_single_config(
                            x_std=x_std,
                            mkt_ret=mkt_ret,
                            k_states=k_cand,
                            lam=lam_cand,
                            kappa=kap_cand,
                        )
                        score = self._score_configuration(
                            x_std, mkt_ret, states, centroids, w, len(centroids)
                        )
                        if score < best_score or best_tuple is None:
                            best_score = score
                            best_tuple = (
                                states,
                                centroids,
                                w,
                                len(centroids),
                                lam_cand,
                                kap_cand,
                            )
            assert best_tuple is not None
            (
                _,
                self.centroids_,
                self.feature_weights_,
                self.selected_k_,
                self.selected_lambda_,
                self.selected_kappa_,
            ) = best_tuple

        self.is_fitted = True
        return self

    def detect(self, data: MarketData, refit: bool = True) -> RegimeState:
        if (
            refit
            or not self.is_fitted
            or self.centroids_ is None
            or self.feature_weights_ is None
        ):
            self.fit(data)
        assert self.centroids_ is not None and self.feature_weights_ is not None

        x_std, idx, _ = self._prepare_standardized_features(data, fit_scaler=False)
        k_eff = len(self.centroids_)
        diff = x_std[:, None, :] - self.centroids_[None, :, :]
        loss_mat = 0.5 * np.sum(
            (diff**2) * self.feature_weights_[None, None, :], axis=2
        )

        states, obj_val = _solve_discrete_jump_dp(
            loss_mat, jump_penalty=self.selected_lambda_
        )

        # Compute calibrated softmax posterior probabilities bolstered by persistent state path
        scaled_logits = -loss_mat / self.prob_temperature
        scaled_logits = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
        raw_probs = np.exp(np.clip(scaled_logits, -30.0, 0.0))
        raw_probs = raw_probs / np.sum(raw_probs, axis=1, keepdims=True)

        one_hot = np.eye(k_eff, dtype=np.float64)[states]
        persistence_blend = float(
            np.clip(self.selected_lambda_ / (self.selected_lambda_ + 10.0), 0.25, 0.85)
        )
        probs_np = persistence_blend * one_hot + (1.0 - persistence_blend) * raw_probs
        probs_np = probs_np / np.sum(probs_np, axis=1, keepdims=True)

        sorted_p = np.sort(probs_np, axis=1)
        conf_np = (
            sorted_p[:, -1] - sorted_p[:, -2]
            if k_eff >= 2
            else np.ones(len(probs_np), dtype=np.float64)
        )

        n_switches = int(np.sum(states[1:] != states[:-1])) if len(states) > 1 else 0
        years = max(len(states) / self.annualization_factor, 1e-4)
        shifts_per_year = float(n_switches / years)

        cols = [f"regime_{k}" for k in range(k_eff)]
        probs_df = pd.DataFrame(probs_np, index=idx, columns=cols)
        hard_ser = pd.Series(states, index=idx, name="regime_state")
        conf_ser = pd.Series(conf_np, index=idx, name="regime_confidence")
        fw_ser = pd.Series(
            self.feature_weights_, index=self.feature_names_, name="feature_weights"
        )
        cent_df = pd.DataFrame(self.centroids_, index=cols, columns=self.feature_names_)

        state_res = RegimeState(
            probs=probs_df,
            hard_state=hard_ser,
            confidence=conf_ser,
            feature_weights=fw_ser,
            regime_centroids=cent_df,
            metadata={
                "detector": "SparseJumpModelDetector",
                "n_regimes": k_eff,
                "jump_penalty": self.selected_lambda_,
                "sparsity_kappa": self.selected_kappa_,
                "tuning_criterion": self.tuning_criterion,
                "n_switches": n_switches,
                "shifts_per_year": shifts_per_year,
                "active_features": int(np.sum(self.feature_weights_ > 1e-3)),
                "objective_value": obj_val,
            },
        )
        self.last_state_ = state_res
        return state_res

    def transform(self, data: MarketData) -> MarketData:
        """
        `BaseFeatureTransformer` entry point: detects regimes and appends
        `output_probs_key` and `output_state_key` into `MarketData.features`.
        """
        state = self.detect(data, refit=not self.is_fitted)
        new_features = dict(data.features) if data.features else {}
        new_features[self.output_probs_key] = (
            state.probs.reindex(data.prices.index).ffill().bfill()
        )
        new_features[self.output_state_key] = (
            state.hard_state.to_frame(name="regime_state")
            .reindex(data.prices.index)
            .ffill()
            .bfill()
        )
        return replace(data, features=new_features)


class ContinuousJumpModelDetector(BaseRegimeDetector, BaseFeatureTransformer):
    """
    Continuous Jump Model Regime Detector (`ContinuousJumpModelDetector`,
    Aydınhan, Kolm, Mulvey & Shu, 2024, *Annals of Operations Research*).

    Generalizes discrete regime assignments into a continuous simplex probability trajectory
    `p_t in Delta^K` minimizing feature distance, temporal L1/L2 jump penalty, and a
    mode-sharpening loss that pushes probabilities toward simplex vertices while preserving
    smooth transitions across regime boundaries:

        min_{Theta, P in Delta^K}  sum_{t=1}^T <p_t, l_t>
                                   + lambda * sum_{t=2}^T ||p_t - p_{t-1}||_2^2
                                   - gamma_mode * sum_{t=1}^T ||p_t - 1/K||_2^2.

    Eliminates discontinuous Viterbi boundary jumps and minority-regime sample starvation.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        jump_penalty: float = 20.0,
        mode_loss_weight: float = 0.35,
        sparsity_kappa: float | None = 2.2,
        tuning_criterion: TuningCriterion = "none",
        jump_penalty_grid: Sequence[float] | None = None,
        windows: Sequence[int] = (5, 20, 60),
        feature_keys: Sequence[str] | None = None,
        max_iter: int = 25,
        smoothing_passes: int = 12,
        tol: float = 1e-5,
        output_probs_key: str = "regime_probs",
        output_state_key: str = "regime_state",
        annualization_factor: float = 252.0,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = max(2, int(n_regimes))
        self.jump_penalty = max(0.0, float(jump_penalty))
        self.mode_loss_weight = float(np.clip(mode_loss_weight, 0.0, 2.0))
        self.sparsity_kappa = sparsity_kappa
        self.tuning_criterion: TuningCriterion = tuning_criterion
        self.jump_penalty_grid = (
            [max(0.0, float(lam)) for lam in jump_penalty_grid]
            if jump_penalty_grid is not None
            else [8.0, 20.0, 45.0]
        )
        self.windows = tuple(int(w) for w in windows)
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.max_iter = max(3, int(max_iter))
        self.smoothing_passes = max(2, int(smoothing_passes))
        self.tol = float(max(1e-8, tol))
        self.output_probs_key = output_probs_key
        self.output_state_key = output_state_key
        self.annualization_factor = float(annualization_factor)
        self.random_state = int(random_state)

        self._sjm_initializer = SparseJumpModelDetector(
            n_regimes=self.n_regimes,
            jump_penalty=self.jump_penalty,
            sparsity_kappa=self.sparsity_kappa,
            tuning_criterion=self.tuning_criterion,
            jump_penalty_grid=self.jump_penalty_grid,
            windows=self.windows,
            feature_keys=self.feature_keys,
            max_iter=self.max_iter,
            annualization_factor=self.annualization_factor,
            random_state=self.random_state,
        )
        self.is_fitted: bool = False
        self.centroids_: np.ndarray | None = None
        self.feature_weights_: np.ndarray | None = None
        self.selected_lambda_: float = self.jump_penalty
        self.last_state_: RegimeState | None = None

    @staticmethod
    def _project_probability_simplex(v: np.ndarray) -> np.ndarray:
        """Exact Euclidean projection onto the K-dimensional probability simplex Delta^K."""
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, len(v) + 1)
        cond = u - cssv / ind > 0.0
        rho = int(ind[cond][-1]) if np.any(cond) else 1
        theta = float(cssv[rho - 1]) / float(rho)
        return np.maximum(v - theta, 0.0)

    def _solve_continuous_simplex_trajectory(
        self,
        loss_mat: np.ndarray,
        init_probs: np.ndarray,
        lam: float,
    ) -> np.ndarray:
        """
        Solves for the continuous simplex trajectory `P in (T, K)` with temporal jump
        regularization `lam * ||p_t - p_{t-1}||_2^2` and mode-sharpening penalty
        `mode_loss_weight` via block-coordinate proximal simplex updates.
        """
        t_len, k_eff = loss_mat.shape
        if t_len == 0:
            return np.zeros((0, k_eff), dtype=np.float64)

        p_traj = init_probs.copy()
        uniform = np.full(k_eff, 1.0 / k_eff, dtype=np.float64)
        lam_eff = max(0.0, float(lam))
        gamma_m = self.mode_loss_weight

        # Normalize loss rows to stable scale
        row_min = np.min(loss_mat, axis=1, keepdims=True)
        row_scale = np.maximum(np.std(loss_mat, axis=1, keepdims=True), 0.25)
        l_norm = (loss_mat - row_min) / row_scale

        for _ in range(self.smoothing_passes):
            # Forward-backward Gauss-Seidel simplex sweeps
            for t in range(t_len):
                nbr_sum = np.zeros(k_eff, dtype=np.float64)
                n_nbr = 0.0
                if t > 0:
                    nbr_sum += p_traj[t - 1]
                    n_nbr += 1.0
                if t < t_len - 1:
                    nbr_sum += p_traj[t + 1]
                    n_nbr += 1.0

                denom = max(2.0 * lam_eff * n_nbr + 1.0, 1e-6)
                # Mode loss sharpens p_t away from uniform 1/K toward vertices
                mode_boost = 2.0 * gamma_m * (p_traj[t] - uniform)
                target = (
                    2.0 * lam_eff * nbr_sum + p_traj[t] - l_norm[t] + mode_boost
                ) / denom
                p_traj[t] = self._project_probability_simplex(target)

        return p_traj

    def fit(self, data: MarketData) -> ContinuousJumpModelDetector:
        self._sjm_initializer.fit(data)
        self.centroids_ = (
            self._sjm_initializer.centroids_.copy()
            if self._sjm_initializer.centroids_ is not None
            else None
        )
        self.feature_weights_ = (
            self._sjm_initializer.feature_weights_.copy()
            if self._sjm_initializer.feature_weights_ is not None
            else None
        )
        self.selected_lambda_ = self._sjm_initializer.selected_lambda_

        if self.centroids_ is not None and self.feature_weights_ is not None:
            x_std, _, _ = self._sjm_initializer._prepare_standardized_features(
                data, fit_scaler=False
            )
            k_eff = len(self.centroids_)
            for _ in range(3):
                diff = x_std[:, None, :] - self.centroids_[None, :, :]
                loss_mat = 0.5 * np.sum(
                    (diff**2) * self.feature_weights_[None, None, :], axis=2
                )
                logits = -loss_mat - np.max(-loss_mat, axis=1, keepdims=True)
                init_p = np.exp(np.clip(logits, -25.0, 0.0))
                init_p /= np.sum(init_p, axis=1, keepdims=True)
                p_traj = self._solve_continuous_simplex_trajectory(
                    loss_mat, init_p, lam=self.selected_lambda_
                )
                for k in range(k_eff):
                    w_k = p_traj[:, k]
                    w_sum = float(np.sum(w_k))
                    if w_sum > 1e-6:
                        self.centroids_[k] = (w_k @ x_std) / w_sum

        self.is_fitted = True
        return self

    def detect(self, data: MarketData, refit: bool = True) -> RegimeState:
        if (
            refit
            or not self.is_fitted
            or self.centroids_ is None
            or self.feature_weights_ is None
        ):
            self.fit(data)
        assert self.centroids_ is not None and self.feature_weights_ is not None

        x_std, idx, _ = self._sjm_initializer._prepare_standardized_features(
            data, fit_scaler=False
        )
        k_eff = len(self.centroids_)
        diff = x_std[:, None, :] - self.centroids_[None, :, :]
        loss_mat = 0.5 * np.sum(
            (diff**2) * self.feature_weights_[None, None, :], axis=2
        )

        states_dp, _ = _solve_discrete_jump_dp(
            loss_mat, jump_penalty=self.selected_lambda_
        )
        logits = -loss_mat - np.max(-loss_mat, axis=1, keepdims=True)
        soft_p = np.exp(np.clip(logits, -25.0, 0.0))
        soft_p /= np.sum(soft_p, axis=1, keepdims=True)
        init_p = 0.5 * np.eye(k_eff, dtype=np.float64)[states_dp] + 0.5 * soft_p

        probs_np = self._solve_continuous_simplex_trajectory(
            loss_mat, init_p, lam=self.selected_lambda_
        )
        hard_states = np.argmax(probs_np, axis=1).astype(np.int64)

        sorted_p = np.sort(probs_np, axis=1)
        conf_np = (
            sorted_p[:, -1] - sorted_p[:, -2]
            if k_eff >= 2
            else np.ones(len(probs_np), dtype=np.float64)
        )

        n_switches = (
            int(np.sum(hard_states[1:] != hard_states[:-1]))
            if len(hard_states) > 1
            else 0
        )
        l1_prob_turnover = (
            float(np.mean(np.sum(np.abs(np.diff(probs_np, axis=0)), axis=1)))
            if len(probs_np) > 1
            else 0.0
        )
        years = max(len(hard_states) / self.annualization_factor, 1e-4)
        shifts_per_year = float(n_switches / years)

        cols = [f"regime_{k}" for k in range(k_eff)]
        probs_df = pd.DataFrame(probs_np, index=idx, columns=cols)
        hard_ser = pd.Series(hard_states, index=idx, name="regime_state")
        conf_ser = pd.Series(conf_np, index=idx, name="regime_confidence")
        fw_ser = pd.Series(
            self.feature_weights_,
            index=self._sjm_initializer.feature_names_,
            name="feature_weights",
        )
        cent_df = pd.DataFrame(
            self.centroids_,
            index=cols,
            columns=self._sjm_initializer.feature_names_,
        )

        state_res = RegimeState(
            probs=probs_df,
            hard_state=hard_ser,
            confidence=conf_ser,
            feature_weights=fw_ser,
            regime_centroids=cent_df,
            metadata={
                "detector": "ContinuousJumpModelDetector",
                "n_regimes": k_eff,
                "jump_penalty": self.selected_lambda_,
                "mode_loss_weight": self.mode_loss_weight,
                "n_switches": n_switches,
                "shifts_per_year": shifts_per_year,
                "mean_l1_prob_change": l1_prob_turnover,
            },
        )
        self.last_state_ = state_res
        return state_res

    def transform(self, data: MarketData) -> MarketData:
        state = self.detect(data, refit=not self.is_fitted)
        new_features = dict(data.features) if data.features else {}
        new_features[self.output_probs_key] = (
            state.probs.reindex(data.prices.index).ffill().bfill()
        )
        new_features[self.output_state_key] = (
            state.hard_state.to_frame(name="regime_state")
            .reindex(data.prices.index)
            .ffill()
            .bfill()
        )
        return replace(data, features=new_features)
