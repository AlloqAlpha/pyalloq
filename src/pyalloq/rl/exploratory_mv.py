from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.linalg as sla
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from sklearn.mixture import GaussianMixture


class WonhamRegimeFilter:
    """
    Continuous-Time Wonham Regime Belief Filter (`WonhamRegimeFilter`, arXiv:2501.16659).

    Estimates `K` latent market regimes `(mu_k, Sigma_k)` ordered from lowest to highest
    volatility alongside the continuous-time Markov generator matrix `Q in R^{K x K}`
    (`sum_j Q_{ij} = 0`, `Q_{ij} >= 0` for `i != j`), and filters the causal posterior
    regime probability trajectory `p_t = P(S_t = k | F_t) in Delta^K` via the Wonham SDE:

        dp_{t, k} = sum_{j=1}^K Q_{j, k} p_{t, j} dt
                    + p_{t, k} (mu_k - mu_bar_t)^T Sigma_bar_t^{-1} (dR_t - mu_bar_t dt)
    """

    def __init__(
        self,
        n_regimes: int = 2,
        annualization_factor: float = 252.0,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = max(2, int(n_regimes))
        self.annualization_factor = float(annualization_factor)
        self.random_state = int(random_state)

        self.is_fitted: bool = False
        self.regime_means_: np.ndarray | None = None  # (K, N) daily
        self.regime_covs_: np.ndarray | None = None  # (K, N, N) daily
        self.transition_matrix_: np.ndarray | None = None  # (K, K)
        self.generator_matrix_: np.ndarray | None = None  # (K, K)
        self.last_belief_: np.ndarray | None = None  # (K,)

    def fit(self, returns: pd.DataFrame) -> WonhamRegimeFilter:
        clean_ret = returns.dropna(how="all").fillna(0.0)
        ret_np = clean_ret.to_numpy(dtype=np.float64)
        t_obs, n_assets = ret_np.shape
        k_reg = min(self.n_regimes, max(2, t_obs // 4))

        if t_obs >= 2 * k_reg:
            gmm = GaussianMixture(
                n_components=k_reg,
                covariance_type="full",
                reg_covar=1e-5,
                random_state=self.random_state,
            )
            gmm.fit(ret_np)
            raw_means = np.asarray(gmm.means_, dtype=np.float64)
            raw_covs = np.asarray(gmm.covariances_, dtype=np.float64)
            states = gmm.predict(ret_np)
        else:
            raw_means = np.tile(np.mean(ret_np, axis=0), (k_reg, 1))
            base_cov = np.cov(ret_np, rowvar=False) + 1e-5 * np.eye(n_assets)
            raw_covs = np.stack(
                [base_cov * (0.7 + 0.6 * i) for i in range(k_reg)], axis=0
            )
            states = np.zeros(t_obs, dtype=int)

        # Order regimes canonically from lowest trace volatility (calm/bull) to highest (crisis/bear)
        vols = np.array([np.trace(raw_covs[k]) for k in range(k_reg)])
        order = np.argsort(vols)
        self.regime_means_ = raw_means[order]
        self.regime_covs_ = np.stack(
            [
                0.5 * (raw_covs[idx] + raw_covs[idx].T) + 1e-6 * np.eye(n_assets)
                for idx in order
            ],
            axis=0,
        )

        inv_order = np.zeros(k_reg, dtype=int)
        for new_idx, old_idx in enumerate(order):
            inv_order[old_idx] = new_idx
        mapped_states = inv_order[states]

        # Estimate empirical Markov transition matrix P and generator Q = (P - I) / dt
        trans_counts = np.full((k_reg, k_reg), 0.5, dtype=np.float64)
        for t in range(len(mapped_states) - 1):
            trans_counts[mapped_states[t], mapped_states[t + 1]] += 1.0
        p_mat = trans_counts / np.sum(trans_counts, axis=1, keepdims=True)
        self.transition_matrix_ = p_mat

        dt = 1.0 / self.annualization_factor
        q_mat = (p_mat - np.eye(k_reg, dtype=np.float64)) / dt
        for i in range(k_reg):
            q_mat[i, i] = -np.sum(np.delete(q_mat[i], i))
        self.generator_matrix_ = q_mat
        self.is_fitted = True
        return self

    def filter_series(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Runs causal continuous-time Wonham SDE filtering across `returns` and returns
        a `(T, K)` DataFrame of posterior regime probabilities `p_t in Delta^K`.
        """
        if not self.is_fitted:
            self.fit(returns)
        assert (
            self.regime_means_ is not None
            and self.regime_covs_ is not None
            and self.generator_matrix_ is not None
        )

        clean_ret = returns.dropna(how="all").fillna(0.0)
        ret_np = clean_ret.to_numpy(dtype=np.float64)
        t_obs, n_assets = ret_np.shape
        k_reg = len(self.regime_means_)
        dt = 1.0 / self.annualization_factor

        p_curr = np.full(k_reg, 1.0 / k_reg, dtype=np.float64)
        beliefs = np.empty((t_obs, k_reg), dtype=np.float64)

        prec_mats = [
            np.linalg.inv(self.regime_covs_[k] + 1e-5 * np.eye(n_assets))
            for k in range(k_reg)
        ]

        for t in range(t_obs):
            r_t = ret_np[t]
            mu_bar = p_curr @ self.regime_means_
            cov_bar = sum(
                p_curr[k] * self.regime_covs_[k] for k in range(k_reg)
            ) + 1e-5 * np.eye(n_assets)
            prec_bar = np.linalg.inv(cov_bar)
            innov = r_t - mu_bar

            # Wonham SDE drift + innovation martingale update
            prior_drift = (p_curr @ self.generator_matrix_) * dt
            innov_gain = np.array(
                [
                    p_curr[k]
                    * float((self.regime_means_[k] - mu_bar) @ prec_bar @ innov)
                    for k in range(k_reg)
                ],
                dtype=np.float64,
            )
            # Combine with quadratic log-likelihood innovation for second-order volatility discrimination
            quad_ll = np.array(
                [
                    -0.5
                    * float(
                        (r_t - self.regime_means_[k])
                        @ prec_mats[k]
                        @ (r_t - self.regime_means_[k])
                    )
                    - 0.5 * float(np.linalg.slogdet(self.regime_covs_[k])[1])
                    for k in range(k_reg)
                ],
                dtype=np.float64,
            )
            quad_ll = quad_ll - np.max(quad_ll)
            like_weights = np.exp(np.clip(quad_ll, -20.0, 0.0))

            p_euler = np.maximum(p_curr + prior_drift + innov_gain, 1e-6)
            p_euler = p_euler / np.sum(p_euler)
            p_post = 0.65 * p_euler + 0.35 * (p_euler * like_weights)
            p_curr = np.clip(p_post / np.sum(p_post), 1e-4, 1.0)
            p_curr = p_curr / np.sum(p_curr)
            beliefs[t] = p_curr

        self.last_belief_ = p_curr.copy()
        cols = [f"regime_{k}" for k in range(k_reg)]
        return pd.DataFrame(beliefs, index=clean_ret.index, columns=cols)


class ExploratoryRegimeMVAllocator(BaseAllocator):
    """
    Exploratory Mean-Variance Portfolio Optimizer with Regime-Switching
    (`ExploratoryRegimeMVAllocator`, arXiv:2501.16659).

    Implements the continuous-time entropy-regularized ("exploratory") Reinforcement Learning
    Mean-Variance control policy under latent Markov regime switching:

        max_{pi}  E[ int_0^H (w_t^T mu(S_t) - (gamma / 2) w_t^T Sigma(S_t) w_t
                              - (kappa_turn / 2) ||w_t - w_{t^-}||_2^2
                              + lambda_ent * H(pi_t)) dt ]

    Key Outputs:
    ------------
    1. **Wonham-Filtered Regime Beliefs `p_T in Delta^K`**:
       Tracks posterior probabilities across calm/bull and high-volatility/bear regimes.
    2. **Myopic + Intertemporal Regime-Hedging Mean Control `mu_pi^*`**:
       Combines current regime-blended risk-return trade-off with forward generator transition
       hedging `p_T^T exp(Q * H)` and quadratic turnover regularization toward `current_weights`.
    3. **Analytical Entropy-Regularized Exploration Covariance `Sigma_pi^*`**:
       `Sigma_pi^* = (lambda_entropy / gamma_risk) * (Sigma_bar(p_T) + kappa_turn I)^{-1}`,
       supporting both deterministic deployment (`sample_exploratory_action=False`) and
       stochastic RL policy exploration (`sample_exploratory_action=True` or `.sample_action()`).
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        n_regimes: int = 2,
        risk_aversion: float = 2.0,
        entropy_temperature: float = 0.05,
        turnover_penalty: float = 0.02,
        horizon_days: float = 10.0,
        hedging_weight: float = 0.5,
        sample_exploratory_action: bool = False,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
        random_state: int = 42,
    ) -> None:
        super().__init__(tickers or [])
        self.n_regimes = max(2, int(n_regimes))
        self.risk_aversion = float(max(1e-3, risk_aversion))
        self.entropy_temperature = float(max(1e-5, entropy_temperature))
        self.turnover_penalty = float(max(0.0, turnover_penalty))
        self.horizon_days = float(max(1.0, horizon_days))
        self.hedging_weight = float(np.clip(hedging_weight, 0.0, 2.0))
        self.sample_exploratory_action = bool(sample_exploratory_action)
        self.weight_bounds = weight_bounds
        self.random_state = int(random_state)

        self.regime_filter = WonhamRegimeFilter(
            n_regimes=self.n_regimes, random_state=self.random_state
        )
        self.last_policy_cov_: pd.DataFrame | None = None
        self.last_policy_entropy_: float = 0.0
        self.last_regime_beliefs_: pd.Series | None = None

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        assets = (
            list(cov_matrix.columns)
            if cov_matrix is not None
            else list(data.prices.columns)
        )
        n_assets = len(assets)
        returns_df = data.prices.reindex(columns=assets).pct_change().dropna()
        if len(returns_df) < 3:
            raise ValueError(
                "ExploratoryRegimeMVAllocator requires at least 3 return observations."
            )

        scale_factor = self._resolve_annualization_factor(returns_df)
        self.regime_filter.annualization_factor = scale_factor
        belief_df = self.regime_filter.filter_series(returns_df)
        p_t = belief_df.iloc[-1].to_numpy(dtype=np.float64)
        self.last_regime_beliefs_ = belief_df.iloc[-1].copy()

        assert (
            self.regime_filter.regime_means_ is not None
            and self.regime_filter.regime_covs_ is not None
            and self.regime_filter.generator_matrix_ is not None
        )
        reg_means_ann = self.regime_filter.regime_means_ * scale_factor
        reg_covs_ann = self.regime_filter.regime_covs_ * scale_factor
        k_reg = len(p_t)

        # Regime-blended expected return & total law-of-total-covariance matrix
        mu_myopic = p_t @ reg_means_ann
        if expected_returns is not None:
            mu_pass = (
                expected_returns.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
            mu_myopic = 0.5 * mu_myopic + 0.5 * mu_pass

        cov_blended = np.zeros((n_assets, n_assets), dtype=np.float64)
        for k in range(k_reg):
            diff_k = reg_means_ann[k] - mu_myopic
            cov_blended += p_t[k] * (reg_covs_ann[k] + np.outer(diff_k, diff_k))
        if cov_matrix is not None:
            cov_pass = (
                cov_matrix.reindex(index=assets, columns=assets)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            cov_blended = 0.5 * cov_blended + 0.5 * cov_pass

        cov_blended = 0.5 * (cov_blended + cov_blended.T)
        eigvals, eigvecs = np.linalg.eigh(cov_blended)
        eigvals_psd = np.maximum(eigvals, 1e-6)
        cov_psd = eigvecs @ np.diag(eigvals_psd) @ eigvecs.T

        # Intertemporal Markov regime-transition hedging demand via matrix exponential exp(Q * H)
        tau_years = self.horizon_days / scale_factor
        trans_h = sla.expm(self.regime_filter.generator_matrix_ * tau_years)
        p_forward = np.clip(p_t @ trans_h, 0.0, 1.0)
        p_forward = p_forward / max(float(np.sum(p_forward)), 1e-8)
        mu_forward = p_forward @ reg_means_ann
        hedging_drift = self.hedging_weight * (mu_forward - mu_myopic)
        mu_total = mu_myopic + hedging_drift

        # Analytical entropy-regularized exploratory Gaussian policy covariance Sigma_pi^*
        prec_reg = np.linalg.inv(
            cov_psd + (self.turnover_penalty + 1e-4) * np.eye(n_assets)
        )
        sigma_pi = (self.entropy_temperature / self.risk_aversion) * prec_reg
        sigma_pi = 0.5 * (sigma_pi + sigma_pi.T)
        self.last_policy_cov_ = pd.DataFrame(sigma_pi, index=assets, columns=assets)

        sign, logdet = np.linalg.slogdet(2.0 * np.pi * np.e * sigma_pi)
        self.last_policy_entropy_ = float(0.5 * logdet) if sign > 0 else 0.0

        # Solve for optimal projected mean control mu_pi^*
        w = cp.Variable(n_assets)
        low, high = float(self.weight_bounds[0]), float(self.weight_bounds[1])
        constraints: list[Any] = [
            cp.sum(w) == 1.0,
            w >= low,
            w <= high,
        ]
        quad_risk = 0.5 * self.risk_aversion * cp.quad_form(w, cov_psd)
        if current_weights is not None and self.turnover_penalty > 0.0:
            w_prev = (
                current_weights.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
            turn_pen: Any = 0.5 * self.turnover_penalty * cp.sum_squares(w - w_prev)
        else:
            turn_pen = 0.0

        prob = cp.Problem(cp.Maximize(mu_total @ w - quad_risk - turn_pen), constraints)
        try:
            prob.solve()
        except (cp.SolverError, ValueError, ArithmeticError):
            prob.solve(solver=cp.SCS)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            w_mean = np.asarray(w.value, dtype=np.float64)
            w_mean = np.clip(w_mean, low, high)
            w_mean = w_mean / max(float(np.sum(w_mean)), 1e-12)
            status_str = str(prob.status).upper()
        else:
            w_mean = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
            status_str = "FALLBACK"

        do_sample = kwargs.get(
            "sample_exploratory_action", self.sample_exploratory_action
        )
        if do_sample:
            rng = np.random.default_rng(self.random_state)
            w_raw = rng.multivariate_normal(w_mean, sigma_pi)
            w_final = np.clip(w_raw, max(0.0, low), high)
            w_sum = float(np.sum(w_final))
            w_final = (
                w_final / w_sum
                if w_sum > 1e-12
                else np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
            )
        else:
            w_final = w_mean

        ann_ret = float(mu_total @ w_final)
        ann_vol = float(np.sqrt(max(0.0, w_final @ cov_psd @ w_final)))
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

        return OptimizationResult(
            name="ExploratoryRegimeMV",
            weights=pd.Series(w_final, index=assets, name="weights"),
            status=status_str,
            expected_return=ann_ret,
            volatility=ann_vol,
            sharpe_ratio=sharpe,
            metadata={
                "regime_beliefs": p_t.tolist(),
                "forward_regime_beliefs": p_forward.tolist(),
                "policy_entropy": self.last_policy_entropy_,
                "entropy_temperature": self.entropy_temperature,
                "exploratory_sampled": bool(do_sample),
            },
        )

    def sample_action(
        self,
        data: MarketData,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Samples an entropy-regularized exploratory portfolio action `w ~ pi_T^*(. | x_T, p_T)`
        projected onto the simplex for continuous-time RL training (arXiv:2501.16659).
        """
        res = self.allocate(
            data,
            current_weights=current_weights,
            sample_exploratory_action=True,
        )
        return res.weights
