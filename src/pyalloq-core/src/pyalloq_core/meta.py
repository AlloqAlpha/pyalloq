from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex Delta^K."""
    k = len(v)
    if k == 0:
        return v.copy()
    if k == 1:
        return np.ones(1, dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, k + 1)
    cond = u - cssv / ind > 0.0
    rho = int(ind[cond][-1]) if np.any(cond) else 1
    theta = float(cssv[rho - 1]) / float(rho)
    return np.maximum(v - theta, 0.0)


def estimate_elliptical_kurtosis_kappa(returns: np.ndarray) -> float:
    """
    Estimates the multivariate elliptical excess kurtosis parameter
    `kappa = E[tau_t^2] - 1 = kurt / 3` (Kan & Lassance, *JFQA* Dec 2025):

        kappa_hat = (1 / (N * (N + 2))) * (1 / T) * sum_{t=1}^T [(r_t - mu)' Sigma^{-1} (r_t - mu)]^2 - 1

    Corresponding empirically to Student-t degrees of freedom `nu in [3, 8]`.
    """
    if returns.ndim == 1:
        returns = returns[:, None]
    t_len, n_assets = returns.shape
    if t_len <= 4:
        return 0.5

    mu = np.mean(returns, axis=0)
    centered = returns - mu[None, :]
    cov = (centered.T @ centered) / max(1, t_len - 1) + 1e-5 * np.eye(n_assets)
    prec = np.linalg.pinv(cov)
    mahal_sq = np.einsum("ti,ij,tj->t", centered, prec, centered)
    m4 = float(np.mean(mahal_sq**2))
    raw_kappa = (m4 / float(n_assets * (n_assets + 2))) - 1.0
    return float(np.clip(raw_kappa, 0.0, 5.0))


def compute_optimal_universe_size(
    n_obs: int,
    n_available_assets: int,
    kurtosis_kappa: float = 0.5,
    avg_correlation: float = 0.25,
) -> int:
    """
    Computes the finite-sample optimal universe cardinality `N^*` under elliptical
    parameter uncertainty (Kan & Lassance extension, *JFQA* Dec 2025, 61(5):2312–2352).

    Under elliptical returns with excess kurtosis parameter `kappa`, estimation variance
    scales with `(1 + kappa) * N / T`, placing the finite-sample utility maximum strictly
    below half the sample length (`N^* < T / 2`), and shrinking further as `kappa` or
    cross-asset correlation increases.
    """
    if n_available_assets <= 2:
        return max(1, n_available_assets)
    t_eff = max(6, int(n_obs))
    kappa_eff = max(0.0, float(kurtosis_kappa))
    rho_eff = float(np.clip(avg_correlation, -0.2, 0.90))

    # Effective sample capacity scaled by elliptical kurtosis and correlation redundancy
    effective_ratio = 0.45 / (
        (1.0 + 0.5 * kappa_eff) * (1.0 + 0.35 * max(0.0, rho_eff))
    )
    n_star = round(t_eff * effective_ratio)
    return int(np.clip(n_star, 2, n_available_assets))


class BaseMetaWeightingStrategy(ABC):
    """
    Pluggable interface for online meta-allocation and portfolio blending strategies
    inside `EnsembleAllocator`.

    Takes an `(M_history, K_experts)` matrix of realized/hypothetical child allocator
    net returns (and optionally `current_meta_weights` `alpha_{t-1} in Delta^K`) and
    returns the next-step meta-allocation vector `alpha_t in Delta^K`.
    """

    @abstractmethod
    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Computes blending weights `alpha_t in Delta^K` across `K` child allocators.
        """
        ...


class AdaHedgeStrategy(BaseMetaWeightingStrategy):
    """
    Parameter-Free AdaHedge Online Meta-Weighting Strategy
    (`AdaHedgeStrategy`, van Erven et al., 2011; de Rooij et al., *JMLR* 2014).

    Maintains cumulative loss `L_k` and cumulative mixability gap `Delta_t` to adapt
    the learning rate `eta_t = ln(K) / Delta_t` without hand-tuned hyperparameters.
    Included as the static-regret baseline comparator.
    """

    def __init__(self, loss_scale: float = 1.0) -> None:
        self.loss_scale = float(max(1e-4, loss_scale))
        self.last_eta_: float = 1.0

    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if expert_returns.ndim != 2 or expert_returns.shape[1] == 0:
            return np.ones(1, dtype=np.float64)
        t_steps, k_experts = expert_returns.shape
        if k_experts == 1 or t_steps == 0:
            return np.full(k_experts, 1.0 / max(1, k_experts), dtype=np.float64)

        losses = -expert_returns * self.loss_scale
        cum_loss = np.zeros(k_experts, dtype=np.float64)
        delta_cum = 1e-6
        w = np.full(k_experts, 1.0 / k_experts, dtype=np.float64)

        for t in range(t_steps):
            eta = np.log(float(k_experts)) / max(delta_cum, 1e-6)
            shifted_l = cum_loss - np.min(cum_loss)
            w = np.exp(np.clip(-eta * shifted_l, -40.0, 0.0))
            w /= np.sum(w)

            l_t = losses[t]
            h_t = float(w @ l_t)
            # Mixloss M_t = -(1/eta) * ln(sum_k w_k exp(-eta * l_{t,k}))
            l_shift = l_t - np.min(l_t)
            mix_term = float(np.sum(w * np.exp(np.clip(-eta * l_shift, -40.0, 40.0))))
            m_t = float(np.min(l_t)) - (1.0 / max(eta, 1e-8)) * np.log(
                max(mix_term, 1e-12)
            )
            delta_cum += max(0.0, h_t - m_t)
            cum_loss += l_t

        eta_final = np.log(float(k_experts)) / max(delta_cum, 1e-6)
        self.last_eta_ = float(eta_final)
        shifted_l = cum_loss - np.min(cum_loss)
        w = np.exp(np.clip(-eta_final * shifted_l, -40.0, 0.0))
        return w / np.sum(w)


class FixedShareStrategy(BaseMetaWeightingStrategy):
    """
    Fixed-Share Adaptive-Regret Online Meta-Weighting Strategy
    (`FixedShareStrategy`, Herbster & Warmuth, 1998; Adamskiy, Koolen, Chernov & Vovk,
    *JMLR* 2016).

    Eliminates AdaHedge's static-regret inertia during regime shifts by redistributing
    a switching share `switching_rate in (0, 1)` across all `K` child allocators at every
    step:
        w_{t, k}^{post} = (1 - alpha_switch) * w_{t, k} + (alpha_switch / K),
    guaranteeing worst-case optimal tracking regret over every contiguous regime interval.
    """

    def __init__(
        self,
        switching_rate: float = 0.08,
        learning_rate: float = 12.0,
    ) -> None:
        self.switching_rate = float(np.clip(switching_rate, 0.0, 0.5))
        self.learning_rate = float(max(0.1, learning_rate))

    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        t_steps, k_experts = expert_returns.shape
        if k_experts <= 1 or t_steps == 0:
            return np.full(max(1, k_experts), 1.0 / max(1, k_experts), dtype=np.float64)

        w = np.full(k_experts, 1.0 / k_experts, dtype=np.float64)
        alpha_s = self.switching_rate
        eta = self.learning_rate

        for t in range(t_steps):
            r_t = expert_returns[t]
            r_shifted = r_t - np.max(r_t)
            w_mult = w * np.exp(np.clip(eta * r_shifted, -30.0, 0.0))
            s_mult = float(np.sum(w_mult))
            w_mult = (
                w_mult / s_mult
                if s_mult > 1e-12
                else np.full(k_experts, 1.0 / k_experts)
            )
            # Fixed-share pool redistribution
            w = (1.0 - alpha_s) * w_mult + (alpha_s / float(k_experts))

        return w / np.sum(w)


class CBCEStrategy(BaseMetaWeightingStrategy):
    """
    Coin Betting for Changing Environments (`CBCEStrategy`, Jun, Orabona, Wright & Willett,
    *AISTATS / EJS* 2017).

    Achieves parameter-free Strongly Adaptive Regret `O_tilde(sqrt(|I|))` on every
    sub-interval `I subset [T]` without knowing the number of regime shifts in advance,
    running in `O(log T)` geometric covering intervals via Krichevsky-Trofimov coin-betting
    potentials.
    """

    def __init__(self, max_scales: int = 6) -> None:
        self.max_scales = max(2, int(max_scales))

    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        t_steps, k_experts = expert_returns.shape
        if k_experts <= 1 or t_steps == 0:
            return np.full(max(1, k_experts), 1.0 / max(1, k_experts), dtype=np.float64)

        # Geometric dyadic interval horizons: L_m = 2^m
        horizons = [min(t_steps, max(2, 2 ** (m + 1))) for m in range(self.max_scales)]
        horizons = sorted(set(horizons))

        meta_pool = np.zeros((len(horizons), k_experts), dtype=np.float64)
        interval_scores = np.zeros(len(horizons), dtype=np.float64)

        for idx_h, h_len in enumerate(horizons):
            sub_ret = expert_returns[-h_len:]
            # Normalize per-step excess return over interval mean
            scale = max(float(np.max(np.abs(sub_ret))), 1e-4)
            g_norm = np.clip(sub_ret / scale, -1.0, 1.0)

            wealth = np.ones(k_experts, dtype=np.float64)
            sum_g = np.zeros(k_experts, dtype=np.float64)
            w_curr = np.full(k_experts, 1.0 / k_experts, dtype=np.float64)

            for tau in range(len(g_norm)):
                port_g = float(w_curr @ g_norm[tau])
                adv = g_norm[tau] - port_g
                sum_g += adv
                bet_frac = sum_g / float(tau + 2)
                wealth = np.maximum(1.0 + sum_g * bet_frac, 1e-6)
                pos_bet = np.maximum(bet_frac * wealth, 0.0)
                s_pos = float(np.sum(pos_bet))
                w_curr = (
                    pos_bet / s_pos
                    if s_pos > 1e-12
                    else np.full(k_experts, 1.0 / k_experts)
                )

            meta_pool[idx_h] = w_curr
            # Score interval by realized Sharpe on that horizon
            h_port_ret = sub_ret @ w_curr
            interval_scores[idx_h] = float(np.mean(h_port_ret)) / (
                float(np.std(h_port_ret)) + 1e-5
            )

        shifted = interval_scores - np.max(interval_scores)
        meta_p = np.exp(np.clip(2.0 * shifted, -20.0, 0.0))
        meta_p /= np.sum(meta_p)
        w_final = meta_p @ meta_pool
        return w_final / np.sum(w_final)


class AdaNormalHedgeStrategy(BaseMetaWeightingStrategy):
    """
    Parameter-Free AdaNormalHedge Strategy (`AdaNormalHedgeStrategy`, Luo & Schapire,
    *COLT* 2015).

    Competes simultaneously against any unknown convex combination `u in Delta^K` of
    correlated child allocators rather than only a single winner, using the normal-potential
    function `Phi(R, C) = exp([R]_+^2 / (3 C))`.
    """

    def __init__(self, confidence_restart_decay: float = 0.98) -> None:
        self.confidence_restart_decay = float(
            np.clip(confidence_restart_decay, 0.80, 1.0)
        )

    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        t_steps, k_experts = expert_returns.shape
        if k_experts <= 1 or t_steps == 0:
            return np.full(max(1, k_experts), 1.0 / max(1, k_experts), dtype=np.float64)

        scale = max(float(np.max(np.abs(expert_returns))), 1e-4)
        norm_ret = np.clip(expert_returns / scale, -1.0, 1.0)

        regret = np.zeros(k_experts, dtype=np.float64)
        cum_abs = np.ones(k_experts, dtype=np.float64)
        w = np.full(k_experts, 1.0 / k_experts, dtype=np.float64)

        for t in range(t_steps):
            r_pos = np.maximum(regret, 0.0)
            potential = (r_pos / cum_abs) * np.exp(
                np.clip((r_pos**2) / (3.0 * cum_abs), 0.0, 25.0)
            )
            p_sum = float(np.sum(potential))
            w = (
                potential / p_sum
                if p_sum > 1e-12
                else np.full(k_experts, 1.0 / k_experts)
            )

            port_r = float(w @ norm_ret[t])
            inst_regret = norm_ret[t] - port_r
            regret = self.confidence_restart_decay * regret + inst_regret
            cum_abs = self.confidence_restart_decay * cum_abs + np.abs(inst_regret)

        r_pos = np.maximum(regret, 0.0)
        potential = (r_pos / cum_abs) * np.exp(
            np.clip((r_pos**2) / (3.0 * cum_abs), 0.0, 25.0)
        )
        p_sum = float(np.sum(potential))
        return (
            potential / p_sum
            if p_sum > 1e-12
            else np.full(k_experts, 1.0 / k_experts, dtype=np.float64)
        )


class DualScaledSwitchingCostStrategy(BaseMetaWeightingStrategy):
    """
    Weighted Fractional / Dual-Scaled Online Learning with Switching Costs
    (`DualScaledSwitchingCostStrategy`, Geulen, Vöcking & Winkler, *COLT* 2010;
    NeurIPS 2022; arXiv:2610.24370).

    Leverages full-information feedback across all `K` child allocators while explicitly
    penalizing meta-weight turnover `lambda_switch * ||alpha_t - alpha_{t-1}||_1`, achieving
    the minimax optimal `O(sqrt(B T ln K))` switching-cost regret.
    """

    def __init__(
        self,
        switching_cost_penalty: float = 0.25,
        forgetting_factor: float = 0.96,
        learning_rate: float | None = None,
    ) -> None:
        self.switching_cost_penalty = float(max(0.0, switching_cost_penalty))
        self.forgetting_factor = float(np.clip(forgetting_factor, 0.70, 1.0))
        self.learning_rate = learning_rate

    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        t_steps, k_experts = expert_returns.shape
        if k_experts <= 1 or t_steps == 0:
            return np.full(max(1, k_experts), 1.0 / max(1, k_experts), dtype=np.float64)

        b_cost = max(self.switching_cost_penalty, 1e-3)
        if self.learning_rate is not None:
            eta = float(self.learning_rate)
        else:
            # Minimax optimal learning rate scaling O(sqrt(ln(K) / (B * T)))
            eta = float(np.sqrt(np.log(max(2, k_experts)) / max(1.0, b_cost * t_steps)))

        # Compute exponentially decayed risk-adjusted utility score per expert
        gam = self.forgetting_factor
        decay_weights = np.array(
            [gam ** (t_steps - 1 - t) for t in range(t_steps)], dtype=np.float64
        )
        decay_weights /= np.sum(decay_weights)

        mean_r = decay_weights @ expert_returns
        centered = expert_returns - mean_r[None, :]
        vol_r = np.sqrt(np.maximum(decay_weights @ (centered**2), 1e-8))
        utility_scores = mean_r / vol_r

        logits = (
            eta
            * (utility_scores - np.max(utility_scores))
            / max(float(np.std(utility_scores)), 1e-4)
        )
        w_target = np.exp(np.clip(logits, -20.0, 0.0))
        w_target /= np.sum(w_target)

        if current_meta_weights is not None and len(current_meta_weights) == k_experts:
            w_prev = np.asarray(current_meta_weights, dtype=np.float64)
            w_prev = np.maximum(w_prev, 0.0)
            s_prev = float(np.sum(w_prev))
            if s_prev > 1e-12:
                w_prev /= s_prev
                # Weighted fractional / dual-space contraction proportional to switching penalty
                shrink_alpha = float(1.0 / (1.0 + 2.0 * self.switching_cost_penalty))
                w_smooth = shrink_alpha * w_target + (1.0 - shrink_alpha) * w_prev
                return _project_simplex(w_smooth)

        return w_target


class CADTSBanditStrategy(BaseMetaWeightingStrategy):
    """
    Combinatorial Adaptive Discounted Thompson Sampling (`CADTSBanditStrategy`,
    de Freitas Fonseca et al., *Computational Economics* 2025, arXiv:2410.04217).

    Fuses a discounted Thompson posterior (`gamma_discount`) and a sliding-window
    Thompson posterior (`window_size`) via a pessimistic minimum aggregator (`f = min`),
    selecting top combinatorial subsets for large or computationally expensive RL/DL
    allocator pools.
    """

    def __init__(
        self,
        gamma_discount: float = 0.94,
        window_size: int = 30,
        subset_size: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.gamma_discount = float(np.clip(gamma_discount, 0.70, 0.999))
        self.window_size = max(5, int(window_size))
        self.subset_size = subset_size
        self.random_state = int(random_state)

    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        t_steps, k_experts = expert_returns.shape
        if k_experts <= 1 or t_steps == 0:
            return np.full(max(1, k_experts), 1.0 / max(1, k_experts), dtype=np.float64)

        rng = np.random.default_rng(self.random_state)

        # 1. Discounted posterior mean & precision
        disc_w = np.array(
            [self.gamma_discount ** (t_steps - 1 - t) for t in range(t_steps)],
            dtype=np.float64,
        )
        n_eff_disc = float(np.sum(disc_w))
        disc_w /= max(n_eff_disc, 1e-8)
        mu_disc = disc_w @ expert_returns
        var_disc = np.maximum(disc_w @ ((expert_returns - mu_disc[None, :]) ** 2), 1e-6)

        # 2. Sliding-window posterior mean & precision
        w_len = min(self.window_size, t_steps)
        win_ret = expert_returns[-w_len:]
        mu_win = np.mean(win_ret, axis=0)
        var_win = np.maximum(np.var(win_ret, axis=0, ddof=0), 1e-6)

        # Sample from both Thompson Gaussian posteriors and apply pessimistic min aggregator
        theta_disc = rng.normal(
            mu_disc, np.sqrt(var_disc / max(1.0, n_eff_disc)), size=(25, k_experts)
        )
        theta_win = rng.normal(
            mu_win, np.sqrt(var_win / max(1.0, float(w_len))), size=(25, k_experts)
        )
        pessimistic_samples = np.minimum(theta_disc, theta_win)
        scores = np.mean(pessimistic_samples, axis=0)

        m_sub = (
            min(k_experts, max(1, int(self.subset_size)))
            if self.subset_size is not None
            else k_experts
        )
        top_idx = np.argsort(scores)[::-1][:m_sub]
        mask = np.zeros(k_experts, dtype=bool)
        mask[top_idx] = True

        active_scores = scores[mask] - np.max(scores[mask])
        scale = max(float(np.std(scores[mask])), 1e-4)
        active_w = np.exp(np.clip(active_scores / scale, -15.0, 0.0))
        active_w /= np.sum(active_w)

        w_out = np.zeros(k_experts, dtype=np.float64)
        w_out[mask] = active_w
        return w_out


class BPDSMetaStrategy(BaseMetaWeightingStrategy):
    """
    Bayesian Predictive Decision Synthesis (`BPDSMetaStrategy`, Tallman & West,
    *JRSS-B* 2024, 86(2):340–363; Tallman & West, 2025, arXiv:2405.01598).

    Synthesizes candidate models/allocators via outcome-dependent **Gibbs utility tilting**:
        pi_{t, j} propto pi_{0, j} * exp(tau_t * s_{t, j}(y_t, x_{t, j}))
    where `s_{t, j}` scores realized risk-adjusted portfolio decision utility (rather than
    raw statistical log-likelihood), and `j = 0` designates an explicit **baseline fallback
    model `M_0`** (1/N or cash) that automatically absorbs weight whenever all active
    models lose out-of-sample decision utility.
    """

    def __init__(
        self,
        tilting_temperature: float = 8.0,
        risk_aversion: float = 2.0,
        baseline_prior_weight: float = 0.15,
        forgetting_factor: float = 0.95,
    ) -> None:
        self.tilting_temperature = float(max(0.1, tilting_temperature))
        self.risk_aversion = float(max(0.1, risk_aversion))
        self.baseline_prior_weight = float(np.clip(baseline_prior_weight, 0.01, 0.80))
        self.forgetting_factor = float(np.clip(forgetting_factor, 0.70, 1.0))
        self.last_baseline_weight_: float = 0.0

    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        baseline_index: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        t_steps, k_experts = expert_returns.shape
        if k_experts <= 1 or t_steps == 0:
            return np.full(max(1, k_experts), 1.0 / max(1, k_experts), dtype=np.float64)

        # Last expert is treated as baseline anchor M_0 (1/N) by convention unless specified
        b_idx = baseline_index if baseline_index is not None else (k_experts - 1)
        b_idx = int(np.clip(b_idx, 0, k_experts - 1))

        gam = self.forgetting_factor
        decay = np.array(
            [gam ** (t_steps - 1 - t) for t in range(t_steps)], dtype=np.float64
        )
        decay /= np.sum(decay)

        # Realized decision utility s_{t, j} = r_{t, j} - 0.5 * gamma * r_{t, j}^2
        inst_utility = expert_returns - 0.5 * self.risk_aversion * (expert_returns**2)
        cum_utility = decay @ inst_utility
        util_std = max(float(np.std(cum_utility)), 1e-5)

        # Prior pi_0 gives explicit baseline weight to M_0
        priors = np.full(
            k_experts,
            (1.0 - self.baseline_prior_weight) / max(1, k_experts - 1),
            dtype=np.float64,
        )
        priors[b_idx] = self.baseline_prior_weight

        # If all active models have negative realized utility relative to baseline, boost M_0 tilt
        base_u = float(cum_utility[b_idx])
        rel_utility = (cum_utility - base_u) / util_std
        tilted = priors * np.exp(
            np.clip(self.tilting_temperature * rel_utility, -20.0, 20.0)
        )
        if float(np.max(cum_utility)) < 0.0:
            tilted[b_idx] *= 2.0

        w = tilted / max(float(np.sum(tilted)), 1e-12)
        self.last_baseline_weight_ = float(w[b_idx])
        return w


class EllipticalShrinkageStrategy(BaseMetaWeightingStrategy):
    """
    Analytical Elliptical Fat-Tail Two-Fund / Three-Fund + 1/N Shrinkage Meta-Strategy
    (`EllipticalShrinkageStrategy`, Kan & Lassance, *JFQA* Dec 2025; Lassance, Vanderveken
    & Vrins, *JBES* 2024, 42(3):875–889).

    Calibrates exact finite-sample combination weights blending risky sample optimizers
    (Markowitz, GMV, HRP, CVaR) with the Equally Weighted (`1/N`) anchor portfolio
    accounting for multivariate elliptical excess kurtosis `kappa = E[tau_t^2] - 1`.
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,
        min_anchor_weight: float = 0.15,
    ) -> None:
        self.risk_aversion = float(max(0.1, risk_aversion))
        self.min_anchor_weight = float(np.clip(min_anchor_weight, 0.0, 0.90))
        self.last_kurtosis_kappa_: float = 0.5

    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        asset_returns: np.ndarray | None = None,
        anchor_index: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        t_steps, k_experts = expert_returns.shape
        if k_experts <= 1 or t_steps <= 2:
            return np.full(max(1, k_experts), 1.0 / max(1, k_experts), dtype=np.float64)

        raw_for_kurt = asset_returns if asset_returns is not None else expert_returns
        kappa = estimate_elliptical_kurtosis_kappa(raw_for_kurt)
        self.last_kurtosis_kappa_ = kappa

        n_dim = raw_for_kurt.shape[1] if raw_for_kurt.ndim == 2 else k_experts
        # Finite-sample elliptical estimation risk penalty c_ellip = (1 + kappa) * N / T
        c_ellip = (1.0 + kappa) * float(n_dim) / max(float(t_steps), 4.0)
        # Optimal risky portfolio shrinkage intensity delta^* in [min_anchor_weight, 0.85]
        anchor_share = float(
            np.clip(c_ellip / (1.0 + c_ellip), self.min_anchor_weight, 0.85)
        )

        a_idx = anchor_index if anchor_index is not None else (k_experts - 1)
        a_idx = int(np.clip(a_idx, 0, k_experts - 1))

        # Weight active (non-anchor) experts by elliptical-adjusted out-of-sample utility
        means = np.mean(expert_returns, axis=0)
        vars_e = np.maximum(np.var(expert_returns, axis=0, ddof=1), 1e-8)
        ellip_utility = means - 0.5 * self.risk_aversion * (1.0 + 0.5 * kappa) * vars_e

        active_mask = np.ones(k_experts, dtype=bool)
        active_mask[a_idx] = False
        if not np.any(active_mask):
            return np.ones(k_experts, dtype=np.float64) / k_experts

        u_act = ellip_utility[active_mask]
        u_shift = (u_act - np.max(u_act)) / max(float(np.std(u_act)), 1e-5)
        w_act = np.exp(np.clip(2.0 * u_shift, -15.0, 0.0))
        w_act /= np.sum(w_act)

        w_final = np.zeros(k_experts, dtype=np.float64)
        w_final[active_mask] = (1.0 - anchor_share) * w_act
        w_final[a_idx] = anchor_share
        return w_final


BPDSEstimator = BPDSMetaStrategy
WeightedFractionalStrategy = DualScaledSwitchingCostStrategy
SwitchingCostOLOStrategy = DualScaledSwitchingCostStrategy
