from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import BaseScenarioEstimator

from pyalloq_simulation.utils.imputation import FactorBackcastImputer

FactorSelectionMode = Literal["fixed", "effective_rank", "sjm_screened"]


def compute_effective_spectral_rank(
    z_returns: np.ndarray,
    max_k: int = 18,
) -> int:
    """
    Computes the Roy & Vetterli (2007) spectral-entropy effective rank of standardized
    asset returns, capped at `max_k = 18` to prevent the diffusion conditioning
    bias-variance cliff documented in Gao et al. WRDS ablations (March 2026,
    arXiv:2603.10385).
    """
    _, n_assets = z_returns.shape
    if n_assets <= 2:
        return max(1, n_assets)
    _, s_vals, _ = np.linalg.svd(z_returns, full_matrices=False)
    eig_energy = np.maximum(s_vals**2, 1e-12)
    p = eig_energy / np.sum(eig_energy)
    entropy = -float(np.sum(p * np.log(p)))
    eff_rank = int(round(np.exp(entropy)))
    return int(np.clip(eff_rank, 2, min(n_assets, max(2, int(max_k)))))


class ConditionalDiffusionGenerator(BaseScenarioEstimator):
    """
    Factor-Based Conditional Denoising Diffusion Scenario Generator (`ConditionalDiffusionGenerator`).

    Inherits from `BaseScenarioEstimator` (and `BaseScenarioGenerator`) and synthesizes a 3D
    `ScenarioMarketData` tensor of shape `(M_paths, T_horizon, N_assets)`.

    Architecture:
    -------------
    1. **Factor-Bottleneck Decomposition**:
       Standardized cross-sectional returns `Z_t = (R_t - mu) / sigma` are decomposed via SVD into
       `K = min(n_factors, N)` latent factor coordinates `F_t = Z_t @ V_K` and idiosyncratic
       residuals `E_t = Z_t - F_t @ V_K^T`, avoiding the `N >> T` curse of dimensionality.
    2. **Multi-Modal Conditioning State `c_t`**:
       Each step conditions on:
       - Trailing factor return & realized volatility regime (`condition_window`),
       - Point-in-Time `implied_vol` level and cross-sectional dispersion (when available in `data.features`),
       - Depth-2 Rough-Path Log-Signature & Lévy area invariants over `signature_window`.
    3. **Cosine-Schedule Score Network & Fast DDIM Sampler**:
       Calibrates a conditional sinusoidal-time + random-Fourier residual denoising network
       `eps_theta(x_tau, tau, c_t)` across `diffusion_steps` noise scales and synthesizes forward
       trajectories in `ddim_steps` accelerated DDIM reverse steps (`eta in [0, 1]`).
    """

    def __init__(
        self,
        n_factors: int = 5,
        condition_window: int = 15,
        signature_window: int = 5,
        diffusion_steps: int = 100,
        ddim_steps: int = 20,
        ddim_eta: float = 0.25,
        guidance_scale: float = 1.0,
        condition_dropout: float = 0.15,
        hidden_dim: int = 64,
        ridge_penalty: float = 1e-2,
        implied_vol_feature: str = "implied_vol",
        factor_selection_mode: FactorSelectionMode = "fixed",
        max_conditioning_dim: int = 18,
        annualization_factor: float = 252.0,
        dtype: Any = np.float32,
        impute_missing: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_factors = max(1, int(n_factors))
        self.condition_window = max(3, int(condition_window))
        self.signature_window = max(2, int(signature_window))
        self.diffusion_steps = max(10, int(diffusion_steps))
        self.ddim_steps = max(2, min(int(ddim_steps), self.diffusion_steps))
        self.ddim_eta = float(np.clip(ddim_eta, 0.0, 1.0))
        self.guidance_scale = float(max(0.0, guidance_scale))
        self.condition_dropout = float(np.clip(condition_dropout, 0.0, 0.5))
        self.hidden_dim = max(16, int(hidden_dim))
        self.ridge_penalty = float(max(1e-6, ridge_penalty))
        self.implied_vol_feature = implied_vol_feature
        self.factor_selection_mode: FactorSelectionMode = factor_selection_mode
        self.max_conditioning_dim = max(4, int(max_conditioning_dim))
        self.annualization_factor = float(annualization_factor)
        self.dtype = np.dtype(dtype)
        self.impute_missing = bool(impute_missing)
        self.random_state = random_state

        self.is_fitted: bool = False
        self.log_returns: bool = False
        self.assets: list[str] = []
        self.N: int = 0
        self.K: int = 0

        self.mean_vector: np.ndarray | None = None
        self.std_vector: np.ndarray | None = None
        self.initial_prices: np.ndarray | None = None
        self.factor_loadings_: np.ndarray | None = None  # (N, K)
        self.residual_std_: np.ndarray | None = None  # (N,)
        self.residual_kurtosis_: np.ndarray | None = None  # (N,)

        # Cosine noise schedule parameters
        self.alphas_cumprod_: np.ndarray | None = None

        # Conditional denoising network parameters
        self._rff_weights: np.ndarray | None = None
        self._rff_bias: np.ndarray | None = None
        self._readout_weights: np.ndarray | None = None
        self._cond_mean: np.ndarray | None = None
        self._cond_std: np.ndarray | None = None
        self._terminal_factors_window: np.ndarray | None = None
        self._terminal_iv_ratio: float = 1.0
        self._terminal_iv_dispersion: float = 0.0
        self._cond_keep_indices: np.ndarray | None = None
        self.effective_rank_: int = 0
        self.signature_mmd_loss_: float = 0.0

    @staticmethod
    def _build_cosine_schedule(steps: int, s: float = 0.008) -> np.ndarray:
        """
        Cosine variance schedule (Nichol & Dhariwal, 2021) returning ` alpha_bar_t` in `(0, 1)`.
        """
        t = np.linspace(0, steps, steps + 1, dtype=np.float64)
        f_t = np.cos(((t / steps) + s) / (1.0 + s) * (np.pi * 0.5)) ** 2
        alphas_cumprod = f_t[1:] / f_t[0]
        return np.clip(alphas_cumprod, 1e-4, 0.9999)

    @staticmethod
    def _compute_factor_log_signature(factor_window: np.ndarray) -> np.ndarray:
        """
        Computes Depth-2 Rough Log-Signature summary (level-1 increments + antisymmetric
        Lévy area norms) over a `(W, K)` factor window.
        """
        w_len, k_dim = factor_window.shape
        if w_len < 2:
            return np.zeros(k_dim + 2, dtype=np.float64)

        dx = factor_window
        x_cum = np.cumsum(dx, axis=0)
        x_prev = np.vstack([np.zeros((1, k_dim), dtype=np.float64), x_cum[:-1]])
        s1 = x_cum[-1] / float(w_len)

        levy_mat = (
            0.5
            * (np.einsum("ti,tj->ij", x_prev, dx) - np.einsum("tj,ti->ij", x_prev, dx))
            / float(w_len)
        )
        levy_fro = float(np.linalg.norm(levy_mat))
        levy_asym = float(np.mean(np.abs(levy_mat)))
        return np.concatenate([s1, np.array([levy_fro, levy_asym], dtype=np.float64)])

    def _extract_conditioning_vector(
        self,
        factor_hist: np.ndarray,
        iv_ratio: float,
        iv_disp: float,
    ) -> np.ndarray:
        """
        Constructs the conditioning context vector `c_t` from a `(W_c, K)` slice of
        recent factor returns, PIT implied-vol ratio, and Depth-2 rough log-signatures.
        """
        f_mean = np.mean(factor_hist, axis=0)
        f_vol = np.std(factor_hist, axis=0, ddof=0)
        sig_win = factor_hist[-min(len(factor_hist), self.signature_window) :]
        log_sig = self._compute_factor_log_signature(sig_win)
        iv_stats = np.array([iv_ratio, iv_disp], dtype=np.float64)
        return np.concatenate([f_mean, f_vol, log_sig, iv_stats])

    def _encode_denoiser_features(
        self,
        x_noisy: np.ndarray,
        alpha_bar: np.ndarray,
        cond_norm: np.ndarray,
    ) -> np.ndarray:
        """
        Maps `(x_noisy, sqrt(alpha_bar), sqrt(1 - alpha_bar), sin/cos(tau), cond_norm)`
        through sinusoidal time embeddings and Random Fourier + skip features.
        """
        assert self._rff_weights is not None and self._rff_bias is not None
        sqrt_ab = np.sqrt(alpha_bar)
        sqrt_one_minus_ab = np.sqrt(1.0 - alpha_bar)
        time_emb = np.hstack(
            [
                sqrt_ab,
                sqrt_one_minus_ab,
                np.sin(np.pi * alpha_bar),
                np.cos(np.pi * alpha_bar),
            ]
        )
        raw_in = np.hstack([x_noisy, time_emb, cond_norm])
        proj = raw_in @ self._rff_weights + self._rff_bias
        rff = np.hstack([np.sin(proj), np.cos(proj), np.tanh(proj)]) / np.sqrt(
            float(self.hidden_dim)
        )
        return np.hstack([raw_in, rff, np.ones((len(x_noisy), 1), dtype=np.float64)])

    def fit(self, data: MarketData, log_returns: bool = False) -> None:
        clean_data = (
            FactorBackcastImputer().fit_transform(data) if self.impute_missing else data
        )
        self.assets = list(clean_data.assets)
        self.N = len(self.assets)
        self.K = min(self.n_factors, self.N)
        self.log_returns = log_returns
        self._record_index_metadata(clean_data.prices.index)
        self.initial_prices = clean_data.prices.iloc[-1].to_numpy(dtype=self.dtype)

        returns = clean_data.get_returns(log_returns)
        self._fit_exogenous_features(clean_data, returns)

        ret_np = returns.to_numpy(dtype=np.float64)
        t_obs = len(ret_np)
        if t_obs < 3:
            raise ValueError(
                "ConditionalDiffusionGenerator requires at least 3 return observations."
            )

        self.mean_vector = np.mean(ret_np, axis=0)
        std_vec = np.std(ret_np, axis=0, ddof=1)
        std_vec = np.where(std_vec > 1e-6, std_vec, 1e-3)
        self.std_vector = std_vec

        z_ret = (ret_np - self.mean_vector) / self.std_vector

        self.effective_rank_ = compute_effective_spectral_rank(
            z_ret, max_k=self.max_conditioning_dim
        )
        if self.factor_selection_mode in ("effective_rank", "sjm_screened"):
            self.K = min(self.effective_rank_, self.N, self.max_conditioning_dim)
        else:
            self.K = min(self.n_factors, self.N)

        # SVD factor decomposition: Z = U S V^T -> F = Z @ V_K
        _, _, vt = np.linalg.svd(z_ret, full_matrices=False)
        v_k = vt[: self.K].T  # (N, K)
        # Canonical sign orientation for reproducibility
        signs = np.where(v_k[0, :] >= 0.0, 1.0, -1.0)
        v_k = v_k * signs
        self.factor_loadings_ = v_k

        factors = z_ret @ v_k  # (T, K)
        recon = factors @ v_k.T
        residuals = z_ret - recon
        self.residual_std_ = np.maximum(np.std(residuals, axis=0, ddof=1), 1e-4)
        res_norm = residuals / self.residual_std_
        kurt = np.mean(res_norm**4, axis=0) - 3.0
        self.residual_kurtosis_ = np.clip(kurt, 0.0, 6.0)

        # Extract Point-in-Time implied volatility series if present
        iv_ratios = np.ones(t_obs, dtype=np.float64)
        iv_disps = np.zeros(t_obs, dtype=np.float64)
        hist_ann_vol = self.std_vector * np.sqrt(self.annualization_factor)
        if self.implied_vol_feature in clean_data.features:
            iv_df = (
                clean_data.features[self.implied_vol_feature]
                .reindex(index=returns.index, columns=self.assets)
                .ffill()
                .fillna(0.0)
            )
            iv_mat = iv_df.to_numpy(dtype=np.float64)
            valid_rows = np.all(np.isfinite(iv_mat) & (iv_mat > 0.0), axis=1)
            for idx in range(t_obs):
                if valid_rows[idx]:
                    ratio_vec = iv_mat[idx] / np.maximum(hist_ann_vol, 1e-4)
                    iv_ratios[idx] = float(np.clip(np.mean(ratio_vec), 0.25, 4.0))
                    iv_disps[idx] = float(np.clip(np.std(ratio_vec), 0.0, 2.0))

        self._terminal_iv_ratio = float(iv_ratios[-1])
        self._terminal_iv_dispersion = float(iv_disps[-1])

        w_c = min(self.condition_window, max(2, t_obs - 1))
        self._terminal_factors_window = factors[-w_c:].copy()

        # Build training pairs (c_{t-1}, f_t)
        cond_list: list[np.ndarray] = []
        target_list: list[np.ndarray] = []
        for t_idx in range(w_c, t_obs):
            f_win = factors[t_idx - w_c : t_idx]
            c_vec = self._extract_conditioning_vector(
                f_win, iv_ratios[t_idx - 1], iv_disps[t_idx - 1]
            )
            cond_list.append(c_vec)
            target_list.append(factors[t_idx])

        if not cond_list:
            c_vec = self._extract_conditioning_vector(
                factors, self._terminal_iv_ratio, self._terminal_iv_dispersion
            )
            cond_list = [c_vec for _ in range(t_obs)]
            target_list = [factors[i] for i in range(t_obs)]

        cond_mat = np.vstack(cond_list)
        target_mat = np.vstack(target_list)

        # Bias-variance cliff regularization on conditioning dimensionality k (arXiv:2603.10385):
        # Cap conditioning vector dimension at `max_conditioning_dim` via SJM/variance screening
        raw_c_dim = cond_mat.shape[1]
        if (
            self.factor_selection_mode == "sjm_screened"
            or raw_c_dim > self.max_conditioning_dim
        ):
            c_vars = np.var(cond_mat, axis=0, ddof=0)
            target_corr = np.mean(
                np.abs(
                    (cond_mat - np.mean(cond_mat, axis=0)).T
                    @ (target_mat - np.mean(target_mat, axis=0))
                ),
                axis=1,
            )
            relevance_score = c_vars * (target_corr + 1e-4)
            k_keep = min(raw_c_dim, self.max_conditioning_dim)
            keep_idx = np.sort(np.argsort(relevance_score)[::-1][:k_keep])
            self._cond_keep_indices = keep_idx
            cond_mat = cond_mat[:, keep_idx]
        else:
            self._cond_keep_indices = None

        self._cond_mean = np.mean(cond_mat, axis=0)
        cond_std = np.std(cond_mat, axis=0, ddof=0)
        self._cond_std = np.where(cond_std > 1e-6, cond_std, 1.0)
        cond_norm = (cond_mat - self._cond_mean) / self._cond_std

        self.alphas_cumprod_ = self._build_cosine_schedule(self.diffusion_steps)

        rng = np.random.default_rng(
            self.random_state if self.random_state is not None else 1729
        )
        n_samples = len(target_mat)
        # Replicate across multiple diffusion noise scales for score-matching regression
        n_aug = max(8, min(24, 600 // max(1, n_samples)))
        x0_rep = np.repeat(target_mat, n_aug, axis=0)
        cond_rep = np.repeat(cond_norm, n_aug, axis=0).copy()
        total_rows = len(x0_rep)

        # Classifier-Free Guidance conditioning dropout (arXiv:2509.22088):
        # zero out conditioning vector with probability `condition_dropout` to learn unconditional score
        if self.condition_dropout > 0.0:
            drop_mask = rng.random(size=total_rows) < self.condition_dropout
            cond_rep[drop_mask] = 0.0

        step_indices = rng.integers(0, self.diffusion_steps, size=total_rows)
        alpha_bar_sample = self.alphas_cumprod_[step_indices][:, None]
        eps_true = rng.standard_normal(size=(total_rows, self.K))
        x_noisy = (
            np.sqrt(alpha_bar_sample) * x0_rep
            + np.sqrt(1.0 - alpha_bar_sample) * eps_true
        )

        in_dim = self.K + 4 + cond_norm.shape[1]
        self._rff_weights = rng.normal(
            0.0, 1.0 / np.sqrt(float(in_dim)), size=(in_dim, self.hidden_dim)
        )
        self._rff_bias = rng.uniform(0.0, 2.0 * np.pi, size=(1, self.hidden_dim))

        phi = self._encode_denoiser_features(x_noisy, alpha_bar_sample, cond_rep)
        gram = phi.T @ phi + self.ridge_penalty * np.eye(phi.shape[1], dtype=np.float64)
        self._readout_weights = np.linalg.solve(gram, phi.T @ eps_true)
        self.is_fitted = True

        # Compute Signature-MMD validation discrepancy for Generative-WDRO (arXiv:2604.04342)
        emp_sig = self._compute_factor_log_signature(
            factors[-min(len(factors), max(3, self.signature_window)) :]
        )
        preview = self.generate(
            n_paths=min(24, max(6, t_obs)),
            horizon=min(max(3, self.signature_window), t_obs),
        )
        sim_z = (preview.returns - self.mean_vector) / self.std_vector
        sim_f = sim_z @ self.factor_loadings_
        sim_sigs = np.vstack(
            [self._compute_factor_log_signature(sim_f[m]) for m in range(len(sim_f))]
        )
        self.signature_mmd_loss_ = float(
            np.linalg.norm(emp_sig - np.mean(sim_sigs, axis=0))
            / np.sqrt(max(1, len(emp_sig)))
        )

    def _ddim_sample_step(
        self,
        cond_batch: np.ndarray,
        rng: np.random.Generator,
        guidance_scale: float | None = None,
    ) -> np.ndarray:
        """
        Runs accelerated DDIM reverse-time sampling with Classifier-Free Guidance
        (`guidance_scale`) for one cross-sectional step across `M` paths.
        """
        assert (
            self.alphas_cumprod_ is not None
            and self._cond_mean is not None
            and self._cond_std is not None
            and self._readout_weights is not None
        )
        m_paths = len(cond_batch)
        if self._cond_keep_indices is not None:
            cond_batch = cond_batch[:, self._cond_keep_indices]
        cond_norm = np.clip((cond_batch - self._cond_mean) / self._cond_std, -5.0, 5.0)
        uncond_norm = np.zeros_like(cond_norm)
        g_scale = (
            self.guidance_scale if guidance_scale is None else float(guidance_scale)
        )

        # Sub-sequence of diffusion steps for DDIM
        tau_seq = np.linspace(0, self.diffusion_steps - 1, self.ddim_steps, dtype=int)[
            ::-1
        ]
        x_tau = rng.standard_normal(size=(m_paths, self.K))

        for i, tau in enumerate(tau_seq):
            ab_t = float(self.alphas_cumprod_[tau])
            ab_prev = (
                float(self.alphas_cumprod_[tau_seq[i + 1]])
                if i + 1 < len(tau_seq)
                else 1.0
            )
            ab_vec = np.full((m_paths, 1), ab_t, dtype=np.float64)

            phi_cond = self._encode_denoiser_features(x_tau, ab_vec, cond_norm)
            eps_cond = phi_cond @ self._readout_weights
            if abs(g_scale - 1.0) > 1e-6:
                phi_uncond = self._encode_denoiser_features(x_tau, ab_vec, uncond_norm)
                eps_uncond = phi_uncond @ self._readout_weights
                eps_pred = g_scale * eps_cond + (1.0 - g_scale) * eps_uncond
            else:
                eps_pred = eps_cond

            # Predict x_0 and clip to prevent numerical blow-up
            x0_pred = (x_tau - np.sqrt(1.0 - ab_t) * eps_pred) / np.sqrt(ab_t)
            x0_pred = np.clip(x0_pred, -6.0, 6.0)

            if i + 1 == len(tau_seq):
                x_tau = x0_pred
            else:
                sigma_ddim = (
                    self.ddim_eta
                    * np.sqrt(max(0.0, (1.0 - ab_prev) / max(1.0 - ab_t, 1e-8)))
                    * np.sqrt(max(0.0, 1.0 - ab_t / max(ab_prev, 1e-8)))
                )
                dir_xt = np.sqrt(max(0.0, 1.0 - ab_prev - sigma_ddim**2)) * eps_pred
                z_noise = (
                    rng.standard_normal(size=(m_paths, self.K))
                    if sigma_ddim > 0.0
                    else 0.0
                )
                x_tau = np.sqrt(ab_prev) * x0_pred + dir_xt + sigma_ddim * z_noise

        return x_tau

    def generate(
        self,
        n_paths: int,
        horizon: int,
        guidance_scale: float | None = None,
        stress_iv_multiplier: float = 1.0,
    ) -> ScenarioMarketData:
        if (
            not self.is_fitted
            or self.mean_vector is None
            or self.std_vector is None
            or self.factor_loadings_ is None
            or self.residual_std_ is None
            or self.residual_kurtosis_ is None
            or self._terminal_factors_window is None
            or self.initial_prices is None
        ):
            raise ValueError(
                "ConditionalDiffusionGenerator needs to be fitted. Call .fit(data: MarketData) first."
            )

        rng = np.random.default_rng(self.random_state)
        w_c = len(self._terminal_factors_window)

        # Rolling factor buffer per path: (M, w_c, K)
        f_buffer = np.broadcast_to(
            self._terminal_factors_window[None, :, :], (n_paths, w_c, self.K)
        ).astype(np.float64, copy=True)

        R_sim = np.empty((n_paths, horizon, self.N), dtype=np.float64)
        IV_sim = np.empty((n_paths, horizon, self.N), dtype=np.float64)
        base_ann_vol = self.std_vector * np.sqrt(self.annualization_factor)
        iv_scale = self._terminal_iv_ratio * float(max(0.25, stress_iv_multiplier))

        for t in range(horizon):
            cond_batch = np.vstack(
                [
                    self._extract_conditioning_vector(
                        f_buffer[m], iv_scale, self._terminal_iv_dispersion
                    )
                    for m in range(n_paths)
                ]
            )
            f_next = self._ddim_sample_step(
                cond_batch, rng=rng, guidance_scale=guidance_scale
            )

            # Heavy-tailed idiosyncratic residuals calibrated to empirical excess kurtosis
            df_t = np.clip(
                6.0 / np.maximum(self.residual_kurtosis_, 0.5) + 4.0, 4.5, 30.0
            )
            g_noise = rng.standard_normal(size=(n_paths, self.N))
            chi_scale = np.sqrt(
                rng.chisquare(df=df_t, size=(n_paths, self.N)) / (df_t - 2.0)
            )
            eps_idio = (g_noise / chi_scale) * self.residual_std_

            # Modulate cross-sectional scale by evolving factor volatility & PIT IV ratio
            path_vol_mult = np.clip(
                0.7 * iv_scale
                + 0.3 * np.mean(np.std(f_buffer, axis=1), axis=1, keepdims=True),
                0.4,
                2.5,
            )
            z_step = (f_next @ self.factor_loadings_.T + eps_idio) * path_vol_mult
            R_sim[:, t, :] = self.mean_vector + self.std_vector * z_step
            IV_sim[:, t, :] = path_vol_mult * base_ann_vol[None, :]

            if w_c > 1:
                f_buffer = np.concatenate(
                    [f_buffer[:, 1:, :], f_next[:, None, :]], axis=1
                )
            else:
                f_buffer[:, 0, :] = f_next

        R_sim_out = R_sim.astype(self.dtype, copy=False)
        if self.log_returns:
            synthetic_prices = self.initial_prices * np.exp(
                np.cumsum(R_sim_out, axis=1)
            )
        else:
            synthetic_prices = self.initial_prices * np.cumprod(1.0 + R_sim_out, axis=1)

        synthetic_prices = synthetic_prices.astype(self.dtype, copy=False)
        sim_features, feat_cols = self._simulate_exogenous_features(R_sim_out)
        sim_features[self.implied_vol_feature] = IV_sim.astype(self.dtype, copy=False)
        feat_cols[self.implied_vol_feature] = list(self.assets)

        return ScenarioMarketData(
            prices=synthetic_prices,
            returns=R_sim_out,
            asset_names=self.assets,
            initial_prices=self.initial_prices.copy(),
            timestamps=self._build_future_timestamps(horizon),
            features=sim_features,
            feature_columns=feat_cols,
        )


FactorConditionalDiffusionEstimator = ConditionalDiffusionGenerator
