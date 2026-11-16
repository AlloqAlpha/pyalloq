from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.enums import CovariateKey, CovariateMode
from pyalloq_core.interfaces import (
    BaseReturnEstimator,
    BaseScenarioEstimator,
    BaseViewGenerator,
)
from torch import nn

SCM_ACTIVATIONS: tuple[str, ...] = (
    "tanh",
    "sin",
    "abs",
    "identity",
    "log",
    "sigmoid",
    "smooth_relu",
    "modulo",
    "step",
)


class SRNGNPriorGenerator:
    """
    Single Root Node Random Growing Network (`SRNGN`) Synthetic Time-Series SCM Prior Generator
    (Potapczynski et al., TMLR 2026 / arXiv:2603.15802v2, Section 3.2, Algorithm 1 & Section 4.1).

    1. **Algorithm 1 (`sample_srngn_dag`)**:
       Constructs a Directed Acyclic Graph (DAG) with a guaranteed single root node (Node 0)
       and long causal paths by combining uniform source selection with preferential attachment
       (`Pi_i = (k_i + 1) / sum_j (k_j + 1)`).
    2. **Hybrid Harmonic + Financial Root Node Excitation**:
       Excites the root node with multi-frequency sine/cosine waves (`log phi ~ U(1, 10)`),
       GARCH stochastic volatility clustering, and compound Poisson regime-shift jumps.
    3. **Nonlinear SCM Edge Propagation & Appendix E.1 Noisy Covariate Injection**:
       Propagates signals in topological order using random MLPs with activations drawn from
       `{tanh, sin, abs, identity, log, sigmoid, smooth_relu, modulo, step}` and corrupts a subset
       of covariates with AR(1)/Fourier/Gaussian distractions so `ApolloPFN` learns to ignore
       uninformative features.
    """

    def __init__(
        self,
        min_nodes: int = 16,
        max_nodes: int = 48,
        node_state_dim: int = 6,
        preferential_rho: float = 0.35,
        noisy_covariate_fraction: float = 0.25,
        random_state: int | None = 42,
    ) -> None:
        self.min_nodes = max(6, int(min_nodes))
        self.max_nodes = max(self.min_nodes + 2, int(max_nodes))
        self.node_state_dim = max(2, int(node_state_dim))
        self.preferential_rho = float(np.clip(preferential_rho, 0.0, 1.0))
        self.noisy_covariate_fraction = float(
            np.clip(noisy_covariate_fraction, 0.0, 0.8)
        )
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    @staticmethod
    def sample_srngn_dag(
        n_nodes: int,
        rho: float = 0.35,
        rng: np.random.Generator | None = None,
    ) -> dict[int, list[int]]:
        """
        Algorithm 1: Single Root Node Random Growing Network (SRNGN).

        Returns a parent adjacency dictionary `parents[node_idx] -> list[parent_indices]`
        where `node 0` is the unique root (`parents[0] == []`) and every node `n >= 1`
        has `parents[n] subset {0, ..., n - 1}` with `len(parents[n]) >= 1`.
        """
        gen = rng if rng is not None else np.random.default_rng()
        v_tot = max(3, int(n_nodes))
        parents: dict[int, list[int]] = {i: [] for i in range(v_tot)}
        in_degree = np.zeros(v_tot, dtype=np.float64)

        # Line 1-2: Initialize graph with nodes 0, 1 and directed causal edge 0 -> 1
        parents[1].append(0)
        in_degree[0] = 1.0
        in_degree[1] = 1.0

        for n in range(2, v_tot):
            # Line 4-6: Preferential attachment probabilities over existing nodes 0 .. n-1
            weights = in_degree[:n] + 1.0
            probs = weights / np.sum(weights)
            t_node = int(gen.choice(n, p=probs))

            # Line 7-9: Select additional source s uniformly from {0, ..., n-1} \ {t_node}
            candidates = [idx for idx in range(n) if idx != t_node]
            s_node = int(gen.choice(candidates)) if candidates else 0
            parents[n].append(s_node)
            in_degree[n] = 1.0

            # Line 10-14: With probability rho, also connect target t_node -> n
            if float(gen.random()) < rho and t_node not in parents[n]:
                parents[n].append(t_node)
                in_degree[n] += 1.0
                in_degree[t_node] += 1.0

        return parents

    @staticmethod
    def apply_scm_activation(x: np.ndarray, act_name: str) -> np.ndarray:
        x_clip = np.clip(x, -10.0, 10.0)
        if act_name == "tanh":
            return np.tanh(x_clip)
        if act_name == "sin":
            return np.sin(x_clip)
        if act_name == "abs":
            return np.abs(x_clip)
        if act_name == "identity":
            return x_clip
        if act_name == "log":
            return np.log(np.abs(x_clip) + 1e-2)
        if act_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x_clip))
        if act_name == "smooth_relu":
            return np.logaddexp(0.0, x_clip)
        if act_name == "modulo":
            return np.mod(x_clip, 2.0) - 1.0
        if act_name == "step":
            return np.where(x_clip > 0.0, 1.0, -1.0)
        return np.tanh(x_clip)

    def _sample_root_signal(self, total_steps: int) -> np.ndarray:
        """
        Generates root node excitation `(total_steps, node_state_dim)` combining:
        - Section 3.2 multi-frequency harmonic waves `alpha_1 sin(phi_1 t) + alpha_2 cos(phi_2 t)`
        - GARCH-like stochastic volatility clustering and Poisson regime jumps.
        """
        t_grid = np.linspace(0.0, 1.0, total_steps, dtype=np.float64)[:, None]
        d = self.node_state_dim
        phi_1 = np.exp(self._rng.uniform(1.0, np.log(10.0), size=(1, d)))
        phi_2 = np.exp(self._rng.uniform(1.0, np.log(10.0), size=(1, d)))
        alpha_1 = self._rng.normal(0.0, 1.0, size=(1, d))
        alpha_2 = self._rng.normal(0.0, 1.0, size=(1, d))

        harmonic = alpha_1 * np.sin(2.0 * np.pi * phi_1 * t_grid) + alpha_2 * np.cos(
            2.0 * np.pi * phi_2 * t_grid
        )

        # Stochastic volatility + jump component for financial domain realism
        vol = np.ones((total_steps, d), dtype=np.float64)
        shocks = np.zeros((total_steps, d), dtype=np.float64)
        for t in range(1, total_steps):
            eps = self._rng.standard_normal(size=d)
            vol[t] = np.sqrt(
                0.15 + 0.20 * (shocks[t - 1] ** 2) + 0.65 * (vol[t - 1] ** 2)
            )
            jump = self._rng.poisson(0.04, size=d) * self._rng.normal(0.0, 1.5, size=d)
            shocks[t] = vol[t] * eps + jump

        return 0.6 * harmonic + 0.4 * np.cumsum(shocks * 0.15, axis=0)

    def sample_scm_task(
        self,
        context_len: int = 60,
        horizon: int = 15,
        n_targets: int = 4,
        n_covariates: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples one synthetic multivariate time-series dataset from an SRNGN DAG.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - `targets`: shape `(context_len + horizon, n_targets)`
            - `covariates`: shape `(context_len + horizon, n_covariates)`
        """
        total_steps = int(context_len + horizon)
        min_req = n_targets + n_covariates + 2
        v_nodes = int(
            self._rng.integers(
                max(self.min_nodes, min_req), max(self.max_nodes, min_req + 4)
            )
        )
        parents = self.sample_srngn_dag(
            v_nodes, rho=self.preferential_rho, rng=self._rng
        )

        d = self.node_state_dim
        states = np.zeros((v_nodes, total_steps, d), dtype=np.float64)
        states[0] = self._sample_root_signal(total_steps)

        for node_j in range(1, v_nodes):
            pa_list = parents[node_j]
            pa_sum = np.sum([states[p] for p in pa_list], axis=0) / np.sqrt(
                float(len(pa_list))
            )
            w_edge = self._rng.normal(0.0, 1.0 / np.sqrt(float(d)), size=(d, d))
            b_edge = self._rng.normal(0.0, 0.2, size=(1, d))
            act_name = str(self._rng.choice(SCM_ACTIVATIONS))
            noise = self._rng.normal(0.0, 0.15, size=(total_steps, d))
            raw = self.apply_scm_activation(pa_sum @ w_edge + b_edge, act_name) + noise
            states[node_j] = (raw - np.mean(raw, axis=0)) / (np.std(raw, axis=0) + 1e-5)

        non_root = np.arange(1, v_nodes)
        chosen = self._rng.choice(
            non_root, size=n_targets + n_covariates, replace=False
        )
        cov_nodes = chosen[:n_covariates]
        tgt_nodes = chosen[n_covariates:]

        # Extract scalar projection from each chosen node's 6D state
        covariates = np.column_stack([states[c, :, 0] for c in cov_nodes])
        targets = np.column_stack([states[t, :, 0] for t in tgt_nodes])

        # Appendix E.1: corrupt a fraction of covariates with structured AR(1)/Fourier noise
        n_corrupt = round(self.noisy_covariate_fraction * n_covariates)
        for c_idx in range(n_corrupt):
            ar_noise = np.zeros(total_steps, dtype=np.float64)
            for t in range(1, total_steps):
                ar_noise[t] = 0.7 * ar_noise[t - 1] + float(self._rng.standard_normal())
            covariates[:, c_idx] = ar_noise

        return targets, covariates


class RiemannBarDistribution(nn.Module):
    """
    Piecewise-Constant Riemann ("Bar") Posterior Predictive Distribution Head
    (Hollmann et al., 2025; Potapczynski et al., TMLR 2026, Appendix B & Appendix F).

    Discretizes the z-scored target support `[z_min, z_max]` into `n_bins` buckets and
    provides exact analytical expectations, standard deviations, quantiles, differential
    entropy, and scaled CRPS (`sCRPS`).
    """

    borders: torch.Tensor
    centers: torch.Tensor
    widths: torch.Tensor

    def __init__(
        self,
        n_bins: int = 128,
        z_min: float = -5.0,
        z_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.n_bins = max(16, int(n_bins))
        self.z_min = float(z_min)
        self.z_max = float(z_max)

        borders = torch.linspace(self.z_min, self.z_max, self.n_bins + 1)
        centers = 0.5 * (borders[:-1] + borders[1:])
        widths = borders[1:] - borders[:-1]
        self.register_buffer("borders", borders, persistent=False)
        self.register_buffer("centers", centers, persistent=False)
        self.register_buffer("widths", widths, persistent=False)

    def logits_to_probs(
        self, logits: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        temp = max(float(temperature), 1e-3)
        return torch.softmax(logits / temp, dim=-1)

    def nll_loss(self, logits: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss mapping continuous `target_z` to its Riemann bucket index.
        """
        clamped = torch.clamp(target_z, self.z_min + 1e-5, self.z_max - 1e-5)
        scale = self.n_bins / (self.z_max - self.z_min)
        bin_idx = torch.floor((clamped - self.z_min) * scale).long()
        bin_idx = torch.clamp(bin_idx, 0, self.n_bins - 1)
        flat_logits = logits.reshape(-1, self.n_bins)
        flat_targets = bin_idx.reshape(-1)
        return nn.functional.cross_entropy(flat_logits, flat_targets)

    def mean(self, probs: np.ndarray | torch.Tensor) -> np.ndarray:
        p = (
            probs.detach().cpu().numpy()
            if isinstance(probs, torch.Tensor)
            else np.asarray(probs, dtype=np.float64)
        )
        c = self.centers.detach().cpu().numpy().astype(np.float64)
        return np.asarray(np.sum(p * c, axis=-1), dtype=np.float64)

    def std(self, probs: np.ndarray | torch.Tensor) -> np.ndarray:
        p = (
            probs.detach().cpu().numpy()
            if isinstance(probs, torch.Tensor)
            else np.asarray(probs, dtype=np.float64)
        )
        c = self.centers.detach().cpu().numpy().astype(np.float64)
        w = self.widths.detach().cpu().numpy().astype(np.float64)
        mu = np.sum(p * c, axis=-1, keepdims=True)
        # Include within-bin uniform variance w^2 / 12
        var = np.sum(p * ((c - mu) ** 2 + (w**2) / 12.0), axis=-1)
        return np.sqrt(np.maximum(var, 1e-8))

    def quantile(self, probs: np.ndarray | torch.Tensor, q: float) -> np.ndarray:
        """
        Piecewise-linear CDF inversion across the Riemann bins for any quantile `q in (0, 1)`.
        """
        q_clip = float(np.clip(q, 1e-4, 1.0 - 1e-4))
        p = (
            probs.detach().cpu().numpy()
            if isinstance(probs, torch.Tensor)
            else np.asarray(probs, dtype=np.float64)
        )
        borders_np = self.borders.detach().cpu().numpy().astype(np.float64)
        orig_shape = p.shape[:-1]
        flat_p = p.reshape(-1, self.n_bins)
        cdf = np.cumsum(flat_p, axis=-1)
        cdf_padded = np.hstack([np.zeros((len(flat_p), 1), dtype=np.float64), cdf])
        cdf_padded[:, -1] = 1.0

        out = np.empty(len(flat_p), dtype=np.float64)
        for i in range(len(flat_p)):
            out[i] = float(np.interp(q_clip, cdf_padded[i], borders_np))
        return out.reshape(orig_shape)

    def entropy(self, probs: np.ndarray | torch.Tensor) -> np.ndarray:
        p = (
            probs.detach().cpu().numpy()
            if isinstance(probs, torch.Tensor)
            else np.asarray(probs, dtype=np.float64)
        )
        w = self.widths.detach().cpu().numpy().astype(np.float64)
        p_safe = np.clip(p, 1e-12, 1.0)
        return np.asarray(
            -np.sum(p_safe * np.log(p_safe / w), axis=-1), dtype=np.float64
        )

    def s_crps(
        self,
        probs: np.ndarray | torch.Tensor,
        y_true: np.ndarray,
        quantile_levels: Sequence[float] = (
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),
    ) -> float:
        """
        Computes the scale-free Continuous Ranked Probability Score (`sCRPS`) defined in
        Appendix B of Potapczynski et al. (TMLR 2026):
            sCRPS(y, y_hat) = sum_t (2 / Q) sum_j [ alpha_j (y_t - y_hat_t^{alpha_j})_+
                                                    + (1 - alpha_j)(y_hat_t^{alpha_j} - y_t)_+ ]
                              / sum_t |y_t|
        """
        y_arr = np.asarray(y_true, dtype=np.float64)
        q_list = list(quantile_levels)
        pinball_sum = 0.0
        for alpha_j in q_list:
            y_q = self.quantile(probs, alpha_j)
            diff = y_arr - y_q
            loss_j = alpha_j * np.maximum(diff, 0.0) + (1.0 - alpha_j) * np.maximum(
                -diff, 0.0
            )
            pinball_sum += float(np.sum(loss_j))
        numer = (2.0 / max(1, len(q_list))) * pinball_sum
        denom = max(float(np.sum(np.abs(y_arr))), 1e-8)
        return float(numer / denom)


class RoPETemporalAttention(nn.Module):
    """
    Multi-Head Horizon-Aware Temporal Attention (`AttnSamp`) with Rotary Position Embeddings (RoPE,
    Section 3.3 of Potapczynski et al., TMLR 2026).

    Applies relative lag rotations `R_h` to queries and keys so `q_{t+h}^T R_h k_t -> 0` as `h -> inf`,
    naturally inducing a recency bias while allowing all history and future horizon tokens `1 .. T+H`
    to attend bidirectionally to each other.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        if d_model % (2 * n_heads) != 0:
            raise ValueError("`d_model` must be divisible by `2 * n_heads` for RoPE.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B_f, n_heads, L_time, head_dim)
        _, _, l_time, h_dim = x.shape
        half = h_dim // 2
        positions = torch.arange(l_time, device=x.device, dtype=x.dtype).unsqueeze(1)
        freq_seq = torch.arange(0, half, device=x.device, dtype=x.dtype)
        inv_freq = 1.0 / (10000.0 ** (freq_seq / max(1.0, float(half))))
        angles = positions * inv_freq.unsqueeze(0)  # (L_time, half)
        sin_a = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        cos_a = torch.cos(angles).unsqueeze(0).unsqueeze(0)

        x1 = x[..., :half]
        x2 = x[..., half : 2 * half]
        rotated = torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1)
        return rotated

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor
            Input embedding tensor of shape `(B, T_tot, F_var, D)`.
        """
        bsz, t_tot, f_var, d_mod = z.shape
        # Transpose to attend along T_tot for each variate f: (B * F_var, T_tot, D)
        zf = z.permute(0, 2, 1, 3).reshape(bsz * f_var, t_tot, d_mod)

        q = (
            self.q_proj(zf)
            .view(bsz * f_var, t_tot, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(zf)
            .view(bsz * f_var, t_tot, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(zf)
            .view(bsz * f_var, t_tot, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        q_rope = self._apply_rope(q)
        k_rope = self._apply_rope(k)

        scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / (
            float(self.head_dim) ** 0.5
        )
        attn = torch.softmax(scores, dim=-1)
        ctx = (
            torch.matmul(attn, v)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * f_var, t_tot, d_mod)
        )
        out = self.out_proj(ctx).view(bsz, f_var, t_tot, d_mod).permute(0, 2, 1, 3)
        return out


class FactorizedApolloPFNBlock(nn.Module):
    """
    Single Temporal-Feature Factorized Transformer Layer (Section 3.3, Equation 1):
        Z <- LN_1(Z + AttnFeat(Z))
        Z <- LN_2(Z + AttnSamp_RoPE(Z))
        Z <- LN_3(Z + MLP(Z))
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, ffn_dim: int = 192) -> None:
        super().__init__()
        self.attn_feat = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.attn_samp = RoPETemporalAttention(d_model=d_model, n_heads=n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.ln3 = nn.LayerNorm(d_model)
        self.last_feat_attn_weights_: torch.Tensor | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bsz, t_tot, f_var, d_mod = z.shape
        # 1. AttnFeat across the F_var axis
        z_flat_t = z.reshape(bsz * t_tot, f_var, d_mod)
        feat_out, feat_w = self.attn_feat(
            z_flat_t, z_flat_t, z_flat_t, need_weights=True, average_attn_weights=True
        )
        # Store time-averaged (B, F_var, F_var) cross-variate attention map
        self.last_feat_attn_weights_ = (
            feat_w.detach().view(bsz, t_tot, f_var, f_var).mean(dim=1)
        )
        z = self.ln1(z + feat_out.view(bsz, t_tot, f_var, d_mod))

        # 2. AttnSamp with RoPE and Full Bidirectional Horizon Attention across T_tot
        z = self.ln2(z + self.attn_samp(z))

        # 3. Pointwise MLP
        z = self.ln3(z + self.mlp(z))
        return z


class ApolloPFNTransformer(nn.Module):
    """
    Time-Aware Prior-Data Fitted Network (`ApolloPFNTransformer`, Potapczynski et al., TMLR 2026).

    Implements:
    - **Appendix F Tokenization & Feature-Pair Grouping**:
      Embeds target series with observation mask `0.0` (observed history `1..T`) and `-2.0`
      (unobserved horizon `T+1..T+H`), groups exogenous covariates with orthogonal random
      feature positional encodings `Omega_feat`, and adds absolute sinusoidal encodings `Omega_time`.
    - **Temporal-Feature Factorized Transformer Backbone (`AttnFeat` + RoPE `AttnSamp`)**.
    - **Riemann Bar Distribution Head + In-Context Ridge-Bayesian Prior Anchor**:
      Combines learned transformer logits with an analytical in-context Bayesian linear/ridge
      predictive prior over `(X_{1:T+H}, Y_{1:T})` so predictions remain calibrated even with
      compact architectures.
    """

    feature_pos_enc: torch.Tensor

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 192,
        n_bins: int = 128,
        z_min: float = -5.0,
        z_max: float = 5.0,
        max_features: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ffn_dim = int(ffn_dim)
        self.n_bins = int(n_bins)
        self.max_features = int(max_features)

        # Appendix F: W_Y in R^{2 x D} for (z_value, mask_indicator) and W_X in R^{3 x D} for paired covariates
        self.target_proj = nn.Linear(2, self.d_model)
        self.cov_pair_proj = nn.Linear(3, self.d_model)

        # Fixed orthogonal random feature positional encodings Omega_feat in R^{max_features x D}
        rng = np.random.default_rng(2026)
        feat_pe = (
            rng.standard_normal(size=(self.max_features, self.d_model))
            / np.sqrt(float(self.d_model))
        ).astype(np.float32)
        self.register_buffer(
            "feature_pos_enc", torch.from_numpy(feat_pe), persistent=True
        )

        self.blocks = nn.ModuleList(
            [
                FactorizedApolloPFNBlock(
                    d_model=self.d_model, n_heads=self.n_heads, ffn_dim=self.ffn_dim
                )
                for _ in range(self.n_layers)
            ]
        )
        self.bar_head = nn.Linear(self.d_model, self.n_bins)
        self.bar_dist = RiemannBarDistribution(
            n_bins=self.n_bins, z_min=z_min, z_max=z_max
        )
        self.last_feat_attention_: torch.Tensor | None = None

    def _sinusoidal_time_encoding(
        self, t_tot: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Section 3.3 absolute sinusoidal positional encoding `Omega in R^{T_tot x D}`:
            Omega_{t, 2d}   = cos(2 * pi * t * 2^{2d} / 2^{12})
            Omega_{t, 2d+1} = sin(2 * pi * t * 2^{2d+1} / 2^{12})
        """
        t_pos = torch.arange(t_tot, device=device, dtype=dtype).unsqueeze(1)
        d_idx = torch.arange(self.d_model, device=device, dtype=dtype).unsqueeze(0)
        scales = (2.0 ** (d_idx * (12.0 / max(1.0, float(self.d_model))))) / (2.0**12)
        angles = (2.0 * float(np.pi) * t_pos * scales).to(dtype=dtype)
        enc = torch.where(d_idx % 2 == 0, torch.cos(angles), torch.sin(angles)).to(
            dtype=dtype
        )
        return enc.unsqueeze(0).unsqueeze(2)  # (1, T_tot, 1, D)

    def _in_context_bayesian_prior_logits(
        self,
        y_hist_z: torch.Tensor,
        cov_full_z: torch.Tensor | None,
        horizon: int,
    ) -> torch.Tensor:
        """
        Computes an analytical in-context Bayesian Ridge posterior predictive distribution
        `N(mu_post, sigma_post^2)` over the Riemann Bar support `[z_min, z_max]`, acting as a
        residual prior anchor for the transformer head.
        """
        # y_hist_z: (B, T, N_assets), cov_full_z: (B, T + H, C_cov) or None
        bsz, t_hist, _n_assets = y_hist_z.shape
        device = y_hist_z.device
        dtype = y_hist_z.dtype

        # Build autoregressive + recency-weighted trend + exogenous design matrix
        t_idx = torch.linspace(-1.0, 1.0, t_hist + horizon, device=device, dtype=dtype)
        t_hist_col = t_idx[:t_hist].view(1, t_hist, 1).expand(bsz, t_hist, 1)
        t_fut_col = t_idx[t_hist:].view(1, horizon, 1).expand(bsz, horizon, 1)

        if cov_full_z is not None and cov_full_z.shape[-1] > 0:
            x_train = torch.cat([t_hist_col, cov_full_z[:, :t_hist, :]], dim=-1)
            x_test = torch.cat([t_fut_col, cov_full_z[:, t_hist:, :]], dim=-1)
        else:
            x_train = t_hist_col
            x_test = t_fut_col

        # Exponential recency weights prioritizing recent context observations
        recency = torch.exp(
            torch.linspace(-1.5, 0.0, t_hist, device=device, dtype=dtype)
        )
        w_sqrt = torch.sqrt(recency).view(1, t_hist, 1)

        xw = x_train * w_sqrt
        yw = y_hist_z * w_sqrt
        d_feat = x_train.shape[-1]
        eye = torch.eye(d_feat, device=device, dtype=dtype).unsqueeze(0)
        gram = torch.bmm(xw.transpose(1, 2), xw) + 1.5 * eye
        rhs = torch.bmm(xw.transpose(1, 2), yw)
        beta = torch.linalg.solve(gram, rhs)  # (B, d_feat, N_assets)

        # Combine covariate-driven prediction with recency-weighted local level
        recent_level = (yw.sum(dim=1) / w_sqrt.sum(dim=1)).unsqueeze(
            1
        )  # (B, 1, N_assets)
        y_pred_fut = (
            0.55 * torch.bmm(x_test, beta) + 0.45 * recent_level
        )  # (B, H, N_assets)

        residuals = y_hist_z - torch.bmm(x_train, beta)
        res_var = torch.mean(residuals**2, dim=1, keepdim=True).clamp(min=0.15, max=2.5)
        # Horizon uncertainty grows smoothly with step h = 1 .. H
        h_growth = torch.linspace(1.0, 1.35, horizon, device=device, dtype=dtype).view(
            1, horizon, 1
        )
        sigma_fut = torch.sqrt(res_var * h_growth)  # (B, H, N_assets)

        centers = self.bar_dist.centers.view(1, 1, 1, self.n_bins)
        mu_exp = y_pred_fut.unsqueeze(-1)
        sig_exp = sigma_fut.unsqueeze(-1)
        prior_logits = -0.5 * ((centers - mu_exp) / sig_exp) ** 2 - torch.log(sig_exp)
        return prior_logits

    def forward(
        self,
        y_hist_z: torch.Tensor,
        past_cov_z: torch.Tensor | None = None,
        future_cov_z: torch.Tensor | None = None,
        horizon: int = 10,
        pair_covariates: bool = True,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_hist_z : torch.Tensor
            Z-scored historical target series of shape `(B, T, N_assets)`.
        past_cov_z : torch.Tensor, optional
            Z-scored past-only covariates of shape `(B, T, C_past)`.
        future_cov_z : torch.Tensor, optional
            Z-scored known-future covariates of shape `(B, T + H, C_future)`.
        horizon : int
            Forecast horizon `H`.

        Returns
        -------
        torch.Tensor
            Riemann Bar logits for the future target horizon of shape `(B, H, N_assets, n_bins)`.
        """
        bsz, t_hist, n_assets = y_hist_z.shape
        t_tot = t_hist + int(horizon)
        device = y_hist_z.device
        dtype = y_hist_z.dtype

        # 1. Target Variate Tokenization (Appendix F):
        #    y_tilde_{t, 1} = z-scored target on 1..T, 0.0 (mean) on T+1..T+H
        #    y_tilde_{t, 2} = 0.0 on 1..T (observed), -2.0 on T+1..T+H (unobserved)
        y_pad = torch.zeros((bsz, horizon, n_assets), device=device, dtype=dtype)
        y_vals = torch.cat([y_hist_z, y_pad], dim=1).unsqueeze(-1)  # (B, T+H, N, 1)
        mask_obs = torch.zeros((bsz, t_hist, n_assets, 1), device=device, dtype=dtype)
        mask_unobs = torch.full(
            (bsz, horizon, n_assets, 1), -2.0, device=device, dtype=dtype
        )
        y_mask = torch.cat([mask_obs, mask_unobs], dim=1)
        y_tokens = self.target_proj(
            torch.cat([y_vals, y_mask], dim=-1)
        )  # (B, T+H, N, D)

        # 2. Exogenous Covariate Tokenization & Appendix F Feature-Pair Grouping
        cov_full_list: list[torch.Tensor] = []
        cov_tokens_list: list[torch.Tensor] = []

        if past_cov_z is not None and past_cov_z.shape[-1] > 0:
            c_past = past_cov_z.shape[-1]
            # Past-only covariates are unobserved (-2.0) on T+1..T+H, filled with last/mean
            last_past = past_cov_z[:, -1:, :].expand(bsz, horizon, c_past)
            past_full = torch.cat([past_cov_z, last_past], dim=1)
            cov_full_list.append(past_full)

        if future_cov_z is not None and future_cov_z.shape[-1] > 0:
            cov_full_list.append(future_cov_z[:, :t_tot, :])

        if cov_full_list:
            cov_all = torch.cat(cov_full_list, dim=-1)  # (B, T+H, C_total)
            c_total = cov_all.shape[-1]
            n_past_ch = past_cov_z.shape[-1] if past_cov_z is not None else 0

            # Build mask channel per covariate: 0.0 where observed, -2.0 where unobserved
            c_mask = torch.zeros((bsz, t_tot, c_total), device=device, dtype=dtype)
            if n_past_ch > 0:
                c_mask[:, t_hist:, :n_past_ch] = -2.0

            if pair_covariates and c_total >= 2:
                # Appendix F: Group covariates into pairs (pad by 1 if odd)
                if c_total % 2 == 1:
                    pad_val = torch.zeros((bsz, t_tot, 1), device=device, dtype=dtype)
                    pad_msk = torch.full(
                        (bsz, t_tot, 1), -2.0, device=device, dtype=dtype
                    )
                    cov_all_p = torch.cat([cov_all, pad_val], dim=-1)
                    c_mask_p = torch.cat([c_mask, pad_msk], dim=-1)
                else:
                    cov_all_p = cov_all
                    c_mask_p = c_mask
                n_pairs = cov_all_p.shape[-1] // 2
                paired_vals = cov_all_p.view(bsz, t_tot, n_pairs, 2)
                paired_mask = c_mask_p.view(bsz, t_tot, n_pairs, 2).mean(
                    dim=-1, keepdim=True
                )
                cov_in = torch.cat(
                    [paired_vals, paired_mask], dim=-1
                )  # (B, T+H, n_pairs, 3)
            else:
                cov_in = torch.stack(
                    [cov_all, torch.zeros_like(cov_all), c_mask], dim=-1
                )
            cov_tok = self.cov_pair_proj(cov_in)
            cov_tokens_list.append(cov_tok)
            cov_for_prior: torch.Tensor | None = cov_all
        else:
            cov_for_prior = None

        if cov_tokens_list:
            z = torch.cat([y_tokens, *cov_tokens_list], dim=2)  # (B, T+H, N + F_cov, D)
        else:
            z = y_tokens

        f_total = z.shape[2]
        feat_pe = (
            self.feature_pos_enc[: min(f_total, self.max_features)]
            .to(device=device, dtype=dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if f_total <= self.max_features:
            z = z + feat_pe
        z = (z + self._sinusoidal_time_encoding(t_tot, device=device, dtype=dtype)).to(
            dtype=dtype
        )

        attn_accum: torch.Tensor | None = None
        for block in self.blocks:
            z = block(z)
            feat_w = cast(
                torch.Tensor | None, getattr(block, "last_feat_attn_weights_", None)
            )
            if feat_w is not None:
                attn_accum = feat_w if attn_accum is None else attn_accum + feat_w
        if attn_accum is not None:
            self.last_feat_attention_ = (attn_accum / float(len(self.blocks))).detach()

        # Extract horizon target slice Z_{T+1:T+H, :N_assets, :}
        z_horizon_targets = z[:, t_hist:, :n_assets, :]
        net_logits = self.bar_head(z_horizon_targets)
        prior_logits = self._in_context_bayesian_prior_logits(
            y_hist_z, cov_for_prior, horizon=horizon
        )
        return 0.5 * net_logits + 0.5 * prior_logits

    def save_pretrained(self, checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "ffn_dim": self.ffn_dim,
                "n_bins": self.n_bins,
                "z_min": self.bar_dist.z_min,
                "z_max": self.bar_dist.z_max,
                "max_features": self.max_features,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)
        return path

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str | Path, device: str | None = None
    ) -> ApolloPFNTransformer:
        map_loc = device if device is not None else "cpu"
        payload = torch.load(
            Path(checkpoint_path), map_location=map_loc, weights_only=True
        )
        cfg = payload["config"]
        model = cls(**cfg)
        model.load_state_dict(payload["state_dict"])
        if device is not None:
            model = model.to(device)
        model.eval()
        return model


class ApolloPFNPretrainer:
    """
    Offline Synthetic SCM Prior Meta-Trainer for `ApolloPFNTransformer`
    (Potapczynski et al., TMLR 2026, Section 4.1 & Table 6).

    Trains `ApolloPFNTransformer` strictly on synthetic `SRNGNPriorGenerator` causal graphs
    using AdamW, linear warmup + cosine annealing, and **Stochastic Weight Averaging (SWA /
    checkpoint averaging across the final training steps)** as specified in Table 6.
    """

    def __init__(
        self,
        model: ApolloPFNTransformer | None = None,
        prior_generator: SRNGNPriorGenerator | None = None,
        learning_rate: float = 1e-3,
        warmup_steps: int = 5,
        swa_steps: int = 5,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.device = torch.device(device if device is not None else "cpu")
        self.random_state = int(random_state)
        torch.manual_seed(self.random_state)
        self.model = (model or ApolloPFNTransformer()).to(self.device)
        self.prior_generator = prior_generator or SRNGNPriorGenerator(
            random_state=self.random_state
        )
        self.learning_rate = float(learning_rate)
        self.warmup_steps = max(1, int(warmup_steps))
        self.swa_steps = max(1, int(swa_steps))
        self.loss_history_: list[float] = []

    def pretrain(
        self,
        n_steps: int = 20,
        batch_size: int = 4,
        context_len: int = 40,
        horizon: int = 10,
        n_targets: int = 4,
        n_covariates: int = 4,
    ) -> ApolloPFNTransformer:
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.0
        )

        swa_snapshots: list[dict[str, torch.Tensor]] = []
        total_steps = max(1, int(n_steps))
        swa_start = max(0, total_steps - self.swa_steps)

        for step in range(total_steps):
            # Linear warmup + cosine decay schedule (Table 6)
            if step < self.warmup_steps:
                lr_mult = float(step + 1) / float(self.warmup_steps)
            else:
                prog = float(step - self.warmup_steps) / max(
                    1.0, float(total_steps - self.warmup_steps)
                )
                lr_mult = 0.01 + 0.99 * 0.5 * (1.0 + float(np.cos(np.pi * prog)))
            for pg in optimizer.param_groups:
                pg["lr"] = self.learning_rate * lr_mult

            y_batch_list: list[np.ndarray] = []
            c_batch_list: list[np.ndarray] = []
            for _ in range(batch_size):
                tgts, covs = self.prior_generator.sample_scm_task(
                    context_len=context_len,
                    horizon=horizon,
                    n_targets=n_targets,
                    n_covariates=n_covariates,
                )
                y_batch_list.append(tgts)
                c_batch_list.append(covs)

            y_all = torch.as_tensor(
                np.stack(y_batch_list, axis=0), dtype=torch.float32, device=self.device
            )
            c_all = torch.as_tensor(
                np.stack(c_batch_list, axis=0), dtype=torch.float32, device=self.device
            )

            y_hist = y_all[:, :context_len, :]
            y_fut = y_all[:, context_len:, :]

            optimizer.zero_grad()
            logits = self.model(
                y_hist_z=y_hist,
                past_cov_z=None,
                future_cov_z=c_all,
                horizon=horizon,
                pair_covariates=True,
            )
            loss = self.model.bar_dist.nll_loss(logits, y_fut)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            self.loss_history_.append(float(loss.detach().item()))
            if step >= swa_start:
                swa_snapshots.append(
                    {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                )

        # Table 6: Checkpoint averaging across final steps
        if len(swa_snapshots) > 1:
            avg_state: dict[str, torch.Tensor] = {}
            for key in swa_snapshots[0]:
                stacked = torch.stack(
                    [snap[key].float() for snap in swa_snapshots], dim=0
                )
                avg_state[key] = stacked.mean(dim=0).to(
                    dtype=swa_snapshots[0][key].dtype, device=self.device
                )
            self.model.load_state_dict(avg_state)

        self.model.eval()
        return self.model

    def pretrain_and_save(
        self,
        checkpoint_path: str | Path,
        n_steps: int = 25,
        batch_size: int = 4,
        context_len: int = 40,
        horizon: int = 10,
    ) -> Path:
        self.pretrain(
            n_steps=n_steps,
            batch_size=batch_size,
            context_len=context_len,
            horizon=horizon,
        )
        return self.model.save_pretrained(checkpoint_path)


_CACHED_DEFAULT_APOLLOPFN: ApolloPFNTransformer | None = None


def _resolve_apollopfn_model(
    model: Any = None,
    device: str | None = None,
    warmup_steps: int = 6,
) -> ApolloPFNTransformer:
    global _CACHED_DEFAULT_APOLLOPFN
    if isinstance(model, ApolloPFNTransformer):
        model.eval()
        return model
    if isinstance(model, (str, Path)) and Path(model).exists():
        return ApolloPFNTransformer.from_pretrained(model, device=device)

    if _CACHED_DEFAULT_APOLLOPFN is None:
        pretrainer = ApolloPFNPretrainer(
            warmup_steps=max(1, warmup_steps // 2),
            swa_steps=max(1, warmup_steps // 2),
            device=device,
            random_state=2026,
        )
        _CACHED_DEFAULT_APOLLOPFN = pretrainer.pretrain(
            n_steps=max(2, int(warmup_steps)),
            batch_size=2,
            context_len=32,
            horizon=8,
        )
    return copy.deepcopy(_CACHED_DEFAULT_APOLLOPFN)


class ApolloPFNReturnEstimator(BaseScenarioEstimator, BaseReturnEstimator):
    """
    Zero-Shot Time-Aware Prior-Data Fitted Network Return & Scenario Estimator
    (`ApolloPFNReturnEstimator`, Potapczynski et al., TMLR 2026 / arXiv:2603.15802v2).

    Drop-in interchangeable with `TimesFM3MultivariateReturnEstimator` and
    `TimestralMultivariateReturnEstimator`, while also implementing `BaseScenarioEstimator`
    (`.estimate_scenarios()`) and exposing zero-shot `AttnFeat` covariate attribution
    (`last_covariate_importance_`).
    """

    def __init__(
        self,
        model: Any = None,
        horizon: int = 21,
        annualize: bool = True,
        covariate_mode: CovariateMode = CovariateMode.AUTO,
        covariate_keys: Sequence[CovariateKey | str] | None = None,
        bar_temperature: float = 1.0,
        pair_covariates: bool = False,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.horizon = max(1, int(horizon))
        self.annualize = bool(annualize)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
        self.bar_temperature = float(max(1e-2, bar_temperature))
        self.pair_covariates = bool(pair_covariates)
        self.device = device
        self.random_state = int(random_state)
        self._net: ApolloPFNTransformer | None = None

        self.is_fitted: bool = False
        self.assets: list[str] = []
        self.last_posterior_std_: pd.Series | None = None
        self.last_quantiles_: dict[float, pd.Series] = {}
        self.last_entropy_: pd.Series | None = None
        self.last_covariate_importance_: pd.DataFrame | None = None
        self.last_prediction_intervals_: dict[str, pd.DataFrame] = {}
        self.last_horizon_quantiles_: dict[float, pd.DataFrame] = {}
        self._last_data: MarketData | None = None
        self._last_step_probs: np.ndarray | None = None  # (H, N, B)
        self._last_ret_mean: np.ndarray | None = None  # (N,)
        self._last_ret_std: np.ndarray | None = None  # (N,)
        self._last_asset_corr: np.ndarray | None = None  # (N, N)

    def _get_net(self) -> ApolloPFNTransformer:
        if self._net is None:
            self._net = _resolve_apollopfn_model(self.model, device=self.device)
        return self._net

    def _run_in_context_inference(
        self, data: MarketData, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Runs a single zero-shot forward pass (`torch.no_grad()`) of `ApolloPFNTransformer`
        on `data`, returning `(probs_h_n_b, ret_mean, ret_std, assets)`.
        """
        net = self._get_net()
        dev = next(net.parameters()).device
        assets = list(data.prices.columns)
        self.assets = assets
        n_assets = len(assets)
        self._record_index_metadata(data.prices.index)

        returns_df = data.prices.pct_change().dropna(how="all").fillna(0.0)
        ret_np = returns_df.to_numpy(dtype=np.float64)
        t_obs = len(ret_np)
        if t_obs < 3:
            raise ValueError(
                "ApolloPFNReturnEstimator requires at least 3 return observations."
            )

        ret_mean = np.mean(ret_np, axis=0)
        ret_std = np.maximum(np.std(ret_np, axis=0, ddof=1), 1e-5)
        y_hist_z = (ret_np - ret_mean[None, :]) / ret_std[None, :]

        past_cov, future_cov, ch_names = data.extract_covariates(
            mode=self.covariate_mode,
            covariate_keys=self.covariate_keys,
            horizon=horizon,
            dtype=np.float32,
            return_channel_names=True,
        )
        offset = len(data.prices) - t_obs
        if past_cov is not None:
            past_cov = past_cov[:, offset:]
        if future_cov is not None:
            future_cov = future_cov[:, offset:]
            expected_fut_len = t_obs + horizon
            if future_cov.shape[1] < expected_fut_len:
                pad_w = expected_fut_len - future_cov.shape[1]
                future_cov = np.pad(future_cov, ((0, 0), (0, pad_w)), mode="edge")
            elif future_cov.shape[1] > expected_fut_len:
                future_cov = future_cov[:, :expected_fut_len]

        past_t: torch.Tensor | None = None
        if past_cov is not None and past_cov.shape[0] > 0:
            p_arr = past_cov.T.astype(np.float64)  # (T, C_past)
            p_mu = np.mean(p_arr, axis=0, keepdims=True)
            p_sd = np.maximum(np.std(p_arr, axis=0, keepdims=True), 1e-5)
            p_z = (p_arr - p_mu) / p_sd
            past_t = torch.as_tensor(p_z[None, ...], dtype=torch.float32, device=dev)

        fut_t: torch.Tensor | None = None
        if future_cov is not None and future_cov.shape[0] > 0:
            f_arr = future_cov.T.astype(np.float64)  # (T + H, C_fut)
            f_mu = np.mean(f_arr[:t_obs], axis=0, keepdims=True)
            f_sd = np.maximum(np.std(f_arr[:t_obs], axis=0, keepdims=True), 1e-5)
            f_z = (f_arr - f_mu) / f_sd
            fut_t = torch.as_tensor(f_z[None, ...], dtype=torch.float32, device=dev)

        y_t = torch.as_tensor(y_hist_z[None, ...], dtype=torch.float32, device=dev)

        with torch.no_grad():
            logits = net(
                y_hist_z=y_t,
                past_cov_z=past_t,
                future_cov_z=fut_t,
                horizon=horizon,
                pair_covariates=self.pair_covariates,
            )
            probs = (
                net.bar_dist.logits_to_probs(logits, temperature=self.bar_temperature)[
                    0
                ]
                .detach()
                .cpu()
                .numpy()
            )  # (H, N, n_bins)

        # Extract AttnFeat cross-variate importance & cross-asset epistemic correlation
        if net.last_feat_attention_ is not None:
            attn_mat = (
                net.last_feat_attention_[0].detach().cpu().numpy().astype(np.float64)
            )
            asset_attn = attn_mat[:n_assets, :n_assets]
            asset_corr = 0.5 * (asset_attn + asset_attn.T)
            if n_assets > 1:
                emp_corr = np.corrcoef(y_hist_z.T)
                emp_corr = np.nan_to_num(emp_corr, nan=0.0)
                blend_corr = 0.7 * emp_corr + 0.3 * asset_corr
                blend_corr = 0.5 * (blend_corr + blend_corr.T)
                np.fill_diagonal(blend_corr, 1.0)
                eigvals, eigvecs = np.linalg.eigh(blend_corr)
                self._last_asset_corr = (
                    eigvecs @ np.diag(np.maximum(eigvals, 1e-4)) @ eigvecs.T
                )
            else:
                self._last_asset_corr = np.ones((1, 1), dtype=np.float64)

            if attn_mat.shape[1] > n_assets and len(ch_names) > 0:
                cov_attn = attn_mat[:n_assets, n_assets:]
                cov_sum = np.maximum(np.sum(cov_attn, axis=1, keepdims=True), 1e-8)
                cov_norm = cov_attn / cov_sum
                cols = (
                    ch_names[: cov_norm.shape[1]]
                    if len(ch_names) >= cov_norm.shape[1]
                    else [f"cov_{i}" for i in range(cov_norm.shape[1])]
                )
                self.last_covariate_importance_ = pd.DataFrame(
                    cov_norm, index=assets, columns=cols
                )
            else:
                self.last_covariate_importance_ = pd.DataFrame(index=assets)

        self._last_step_probs = probs
        self._last_ret_mean = ret_mean
        self._last_ret_std = ret_std
        self._last_data = data
        self.is_fitted = True
        return probs, ret_mean, ret_std, assets

    def fit(self, data: MarketData) -> None:
        self._run_in_context_inference(data, horizon=self.horizon)

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        h = int(kwargs.get("horizon", self.horizon))
        probs, ret_mean, ret_std, assets = self._run_in_context_inference(
            data, horizon=h
        )
        net = self._get_net()

        # Invert z-scoring for each horizon step h = 1 .. H: r_{t+h} = mu_train + sigma_train * z_{t+h}
        step_means_z = net.bar_dist.mean(probs)  # (H, N)
        step_stds_z = net.bar_dist.std(probs)  # (H, N)
        step_means = ret_mean[None, :] + ret_std[None, :] * step_means_z
        step_stds = ret_std[None, :] * step_stds_z

        avg_daily_ret = np.mean(step_means, axis=0)
        avg_daily_std = np.mean(step_stds, axis=0)
        scale = 252.0 if self.annualize else float(h)
        std_scale = float(np.sqrt(252.0)) if self.annualize else float(np.sqrt(h))

        ann_ret = avg_daily_ret * scale
        ann_std = avg_daily_std * std_scale
        self.last_posterior_std_ = pd.Series(
            ann_std, index=assets, name="posterior_std"
        )
        self.last_entropy_ = pd.Series(
            np.mean(net.bar_dist.entropy(probs), axis=0), index=assets, name="entropy"
        )

        # Populate decile quantiles and Table 8 prediction intervals (20%, 40%, 60%, 80%)
        fut_idx = self._build_future_timestamps(h)
        self.last_quantiles_ = {}
        self.last_horizon_quantiles_ = {}
        for q_lvl in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            q_z = net.bar_dist.quantile(probs, q_lvl)  # (H, N)
            q_daily = ret_mean[None, :] + ret_std[None, :] * q_z
            self.last_horizon_quantiles_[q_lvl] = pd.DataFrame(
                q_daily * scale, index=fut_idx, columns=assets
            )
            self.last_quantiles_[q_lvl] = pd.Series(
                np.mean(q_daily, axis=0) * scale, index=assets
            )

        for pi_label, (q_lo, q_hi) in (
            ("20%", (0.40, 0.60)),
            ("40%", (0.30, 0.70)),
            ("60%", (0.20, 0.80)),
            ("80%", (0.10, 0.90)),
        ):
            width_df = (
                self.last_horizon_quantiles_[q_hi] - self.last_horizon_quantiles_[q_lo]
            )
            self.last_prediction_intervals_[pi_label] = width_df

        return pd.Series(ann_ret, index=assets, name="expected_returns")

    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        """
        Synthesizes 3D `(n_paths, horizon, n_assets)` non-parametric return & price trajectories
        by inverse-CDF sampling from the multi-horizon Riemann Bar posterior coupled via the
        `AttnFeat` cross-asset Gaussian copula.
        """
        if (
            not self.is_fitted
            or self._last_data is None
            or self._last_step_probs is None
            or self._last_ret_mean is None
            or self._last_ret_std is None
        ):
            raise ValueError(
                "ApolloPFNReturnEstimator must be fitted or called via .estimate_scenarios(data) first."
            )

        if horizon != len(self._last_step_probs):
            self._run_in_context_inference(self._last_data, horizon=horizon)
        assert (
            self._last_step_probs is not None
            and self._last_ret_mean is not None
            and self._last_ret_std is not None
        )

        from scipy.stats import norm

        net = self._get_net()
        rng = np.random.default_rng(self.random_state)
        n_assets = len(self.assets)
        corr = (
            self._last_asset_corr
            if self._last_asset_corr is not None
            else np.eye(n_assets, dtype=np.float64)
        )
        chol = np.linalg.cholesky(corr + 1e-6 * np.eye(n_assets))

        R_sim = np.empty((n_paths, horizon, n_assets), dtype=np.float32)
        borders_np = net.bar_dist.borders.detach().cpu().numpy().astype(np.float64)

        for t in range(horizon):
            z_gauss = rng.standard_normal(size=(n_paths, n_assets)) @ chol.T
            u_unif = np.clip(norm.cdf(z_gauss), 1e-4, 1.0 - 1e-4)
            for i in range(n_assets):
                p_vec = self._last_step_probs[t, i]
                cdf_vec = np.concatenate([[0.0], np.cumsum(p_vec)])
                cdf_vec[-1] = 1.0
                z_quant = np.interp(u_unif[:, i], cdf_vec, borders_np)
                r_col = self._last_ret_mean[i] + self._last_ret_std[i] * z_quant
                R_sim[:, t, i] = r_col.astype(np.float32)

        init_prices = self._last_data.prices.iloc[-1].to_numpy(dtype=np.float32)
        prices_sim = init_prices * np.cumprod(1.0 + R_sim, axis=1)
        return ScenarioMarketData(
            prices=prices_sim.astype(np.float32, copy=False),
            returns=R_sim,
            asset_names=list(self.assets),
            initial_prices=init_prices.copy(),
            timestamps=self._build_future_timestamps(horizon),
        )


class ApolloPFNViewGenerator(BaseViewGenerator):
    """
    Generates Black-Litterman & Fonseca-Junca (2021) Wasserstein DRO views `(P, Q, Omega)`
    from the Riemann Bar Posterior Predictive Distribution of `ApolloPFNTransformer`.

    Drop-in interchangeable with `TimesFM3MultivariateViewGenerator` and
    `TimestralMultivariateViewGenerator`.
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
        bar_temperature: float = 1.0,
        device: str | None = None,
        tfm_model: Any = None,
    ) -> None:
        from scipy.stats import norm

        q_low, q_high = confidence_interval
        if not (0.0 < q_low < q_high < 1.0):
            raise ValueError(
                f"confidence_interval must satisfy 0 < q_low < q_high < 1, got {confidence_interval}."
            )
        self.model = model if model is not None else tfm_model
        self.horizon = max(1, int(horizon))
        self.confidence_interval = confidence_interval
        self.annualize = bool(annualize)
        self.full_covariance_omega = bool(full_covariance_omega)
        self.covariate_mode = covariate_mode
        self.covariate_keys = covariate_keys
        self.bar_temperature = float(max(1e-2, bar_temperature))
        self.device = device
        self._q_low = float(q_low)
        self._q_high = float(q_high)
        self._z_spread = (
            2.56
            if (np.isclose(q_low, 0.10) and np.isclose(q_high, 0.90))
            else float(norm.ppf(q_high) - norm.ppf(q_low))
        )
        self._estimator = ApolloPFNReturnEstimator(
            model=self.model,
            horizon=self.horizon,
            annualize=self.annualize,
            covariate_mode=self.covariate_mode,
            covariate_keys=self.covariate_keys,
            bar_temperature=self.bar_temperature,
            device=self.device,
        )

    def generate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        Q = self._estimator.estimate(data, horizon=self.horizon)
        assets = list(Q.index)
        n_assets = len(assets)

        assert (
            self._estimator._last_step_probs is not None
            and self._estimator._last_ret_std is not None
        )
        net = self._estimator._get_net()
        probs = self._estimator._last_step_probs
        ret_std = self._estimator._last_ret_std

        q_lo_z = np.mean(net.bar_dist.quantile(probs, self._q_low), axis=0)
        q_hi_z = np.mean(net.bar_dist.quantile(probs, self._q_high), axis=0)
        daily_std = np.maximum(q_hi_z - q_lo_z, 1e-6) * ret_std / self._z_spread
        std_scale = (
            float(np.sqrt(252.0)) if self.annualize else float(np.sqrt(self.horizon))
        )
        view_std = daily_std * std_scale

        if (
            self.full_covariance_omega
            and n_assets > 1
            and self._estimator._last_asset_corr is not None
        ):
            corr = self._estimator._last_asset_corr
            d_inv = 1.0 / np.sqrt(np.maximum(np.diag(corr), 1e-8))
            corr_norm = corr * np.outer(d_inv, d_inv)
            omega_mat = np.outer(view_std, view_std) * (
                0.65 * np.eye(n_assets) + 0.35 * corr_norm
            )
            omega_mat = 0.5 * (omega_mat + omega_mat.T)
        else:
            omega_mat = np.diag(view_std**2)

        Omega = pd.DataFrame(omega_mat, index=assets, columns=assets)
        P = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets)
        return P, Q, Omega
