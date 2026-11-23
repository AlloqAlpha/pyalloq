from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from pyalloq.dl.layers.fragility_kkt import BasePolicyNetwork
from pyalloq.dl.layers.set_transformer import (
    NormStyle,
    SetBlockType,
    SetTransformerBackbone,
    TemporalEncoderType,
)

MandateHeadType = Literal[
    "capped_simplex",
    "sparsemax",
    "masked_dirichlet",
    "dollar_neutral",
    "vol_targeted",
]


class _CappedSimplexFunction(torch.autograd.Function):
    r"""
    Exact Euclidean Projection onto the Mask-Aware Capped Simplex:
        \min_{w} \frac{1}{2} \|w - y\|_2^2
        \text{s.t.} \quad \sum_{i: m_i = 1} w_i = 1, \quad 0 \le w_i \le w_{\max, i}, \quad w_j = 0 \;\forall m_j = 0

    Solved via exact vectorized bisection on the Lagrange multiplier ``\nu``:
        w_i(\nu) = m_i \cdot \operatorname{clip}(y_i - \nu, 0, w_{\max})
    with exact implicit Jacobian on the active free set ``S = \{i : m_i = 1 \text{ and } 0 < w_i < w_{\max}\}``:
        \frac{\partial \mathcal{L}}{\partial y_i} = \mathbb{I}_{i \in S} \left( g_i - \frac{1}{|S|} \sum_{k \in S} g_k \right)
    When ``max_weight >= 1.0``, reduces to exact Sparsemax (Martins & Astudillo, ICML 2016).
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        mask: torch.Tensor | None,
        max_weight: float,
        n_bisect_iter: int,
    ) -> torch.Tensor:
        B, N = logits.shape
        device = logits.device
        dtype = logits.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            m_f = m_bool.to(dtype=torch.float64)
            y = torch.where(
                m_bool,
                torch.nan_to_num(logits.to(torch.float64), nan=0.0),
                torch.zeros((B, N), device=device, dtype=torch.float64),
            )
        else:
            m_bool = torch.ones((B, N), device=device, dtype=torch.bool)
            m_f = torch.ones((B, N), device=device, dtype=torch.float64)
            y = torch.nan_to_num(logits.to(torch.float64), nan=0.0)

        n_active = m_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        # Ensure feasibility: if N_active * max_weight < 1.0, relax cap to 1 / N_active
        cap = torch.clamp(
            torch.full((B, 1), float(max_weight), device=device, dtype=torch.float64),
            min=1.0,
        )
        min_feasible_cap = (1.0 / n_active) + 1e-12
        cap = torch.maximum(
            torch.full((B, 1), float(max_weight), device=device, dtype=torch.float64),
            min_feasible_cap,
        ).clamp(max=1.0)

        y_masked_min = (
            torch.where(m_bool, y, torch.full_like(y, 1e12))
            .min(dim=-1, keepdim=True)
            .values
        )
        y_masked_max = (
            torch.where(m_bool, y, torch.full_like(y, -1e12))
            .max(dim=-1, keepdim=True)
            .values
        )

        nu_low = y_masked_min - cap - 1.0
        nu_high = y_masked_max + 1.0

        for _ in range(n_bisect_iter):
            nu_mid = 0.5 * (nu_low + nu_high)
            w_mid = torch.clamp(y - nu_mid, min=0.0)
            w_mid = torch.minimum(w_mid, cap) * m_f
            s_mid = w_mid.sum(dim=-1, keepdim=True)
            nu_low = torch.where(s_mid > 1.0, nu_mid, nu_low)
            nu_high = torch.where(s_mid <= 1.0, nu_mid, nu_high)

        nu_star = 0.5 * (nu_low + nu_high)
        w_star = torch.minimum(torch.clamp(y - nu_star, min=0.0), cap) * m_f
        # Normalize residual floating-point drift on active uncovered support
        w_sum = w_star.sum(dim=-1, keepdim=True)
        w_star = torch.where(w_sum > 1e-12, w_star, m_f / n_active)

        free_mask = m_bool & (w_star > 1e-11) & (w_star < (cap - 1e-11))
        ctx.save_for_backward(free_mask, m_bool)
        return w_star.to(dtype=dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        free_mask, m_bool = ctx.saved_tensors
        dtype = grad_output.dtype

        free_f = free_mask.to(dtype=dtype)
        count_free = free_f.sum(dim=-1, keepdim=True)

        # If all active assets are saturated at bounds, fall back to active mask for smooth gradient flow
        fallback_f = m_bool.to(dtype=dtype)
        use_free = count_free >= 1.0
        support_f = torch.where(use_free, free_f, fallback_f)
        support_count = support_f.sum(dim=-1, keepdim=True).clamp(min=1.0)

        g_masked = grad_output * support_f
        g_mean = g_masked.sum(dim=-1, keepdim=True) / support_count
        grad_logits = support_f * (grad_output - g_mean)
        # Explicitly zero gradients on inactive masked slots
        grad_logits = grad_logits * m_bool.to(dtype=dtype)
        return grad_logits, None, None, None


class CappedSimplexProjection(nn.Module):
    r"""
    Strictly Equivariant Capped Simplex / Sparsemax Portfolio Head:
        w_i(\nu) = m_{i,t} \cdot \operatorname{clip}(y_{i,t} - \nu, 0, w_{\max})
    Setting ``max_weight=1.0`` yields Sparsemax (Martins & Astudillo, ICML 2016).
    """

    def __init__(
        self,
        max_weight: float = 0.20,
        n_bisect_iter: int = 55,
    ) -> None:
        super().__init__()
        self.max_weight = float(max(1e-3, min(1.0, max_weight)))
        self.n_bisect_iter = int(max(25, n_bisect_iter))

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _CappedSimplexFunction.apply(
            logits, mask, self.max_weight, self.n_bisect_iter
        )


class MaskedDirichletCashHead(nn.Module):
    r"""
    Masked Dirichlet + Cash Allocation Head (Xue & Ye, Oct 2025 / arXiv:2510.06466):
        l_t^{(\text{cash})} = a_{\text{cash}}^T z_t^{[\text{REGIME}]}
        \alpha_t = \left(\operatorname{softplus}([y_t, l_t^{(\text{cash})}]) + 1\right) \odot \tilde{m}_t
        w_t = \frac{\alpha_t}{\mathbf{1}^T \alpha_t}  \quad (\text{deterministic})
              \quad \text{or} \quad w_t \sim \operatorname{Dirichlet}(\alpha_t)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        include_cash: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.include_cash = bool(include_cash)
        self.cash_proj = nn.Linear(self.hidden_dim, 1)

    def get_concentrations(
        self,
        asset_logits: torch.Tensor,
        z_regime: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = asset_logits.shape
        device = asset_logits.device
        dtype = asset_logits.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            m_f = m_bool.to(dtype=dtype)
            y_clean = torch.where(
                m_bool,
                torch.nan_to_num(asset_logits, nan=0.0),
                torch.zeros_like(asset_logits),
            )
        else:
            m_f = torch.ones((B, N), device=device, dtype=dtype)
            y_clean = torch.nan_to_num(asset_logits, nan=0.0)

        alpha_assets = (
            torch.nn.functional.softplus(y_clean.clamp(min=-20.0, max=20.0)) + 1.0
        ) * m_f

        if self.include_cash:
            cash_logit = self.cash_proj(z_regime)  # (B, 1)
            alpha_cash = (
                torch.nn.functional.softplus(cash_logit.clamp(min=-20.0, max=20.0))
                + 1.0
            )
            alpha_full = torch.cat([alpha_assets, alpha_cash], dim=-1)
            cash_ones = torch.ones((B, 1), device=device, dtype=dtype)
            mask_full = torch.cat([m_f, cash_ones], dim=-1)
        else:
            alpha_full = alpha_assets
            mask_full = m_f

        return alpha_full, mask_full

    def forward(
        self,
        asset_logits: torch.Tensor,
        z_regime: torch.Tensor,
        mask: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        alpha_full, mask_full = self.get_concentrations(
            asset_logits=asset_logits, z_regime=z_regime, mask=mask
        )
        if deterministic:
            denom = alpha_full.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            return alpha_full / denom

        # Reparameterized Gamma sampling on active slots so padded slots stay strictly 0
        alpha_safe = torch.where(
            mask_full > 0.0, alpha_full.clamp(min=1e-3), torch.ones_like(alpha_full)
        )
        gamma_samples = torch._standard_gamma(alpha_safe) * mask_full
        denom = gamma_samples.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return gamma_samples / denom


class DollarNeutralStatArbHead(nn.Module):
    r"""
    Strictly Equivariant Dollar-Neutral Statistical Arbitrage Head:
        a_t = \tanh(y_t) \odot m_t
        \tilde{a}_{i,t} = \left(a_{i,t} - \frac{1}{N_{\text{active}}} \sum_{j=1}^N a_{j,t}\right) m_{i,t}
        w_t = G_{\text{gross}} \frac{\tilde{a}_t}{\|\tilde{a}_t\|_1 + \epsilon}
    Enforces exact zero net exposure ``\sum_i w_{i,t} = 0`` and target gross leverage
    ``\sum_i |w_{i,t}| = G_{\text{gross}}`` across any active universe size ``N_{\text{active}} >= 2``.
    """

    def __init__(
        self,
        gross_leverage: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.gross_leverage = float(max(1e-3, gross_leverage))
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N = logits.shape
        device = logits.device
        dtype = logits.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            m_f = m_bool.to(dtype=dtype)
            y_clean = torch.where(
                m_bool,
                torch.nan_to_num(logits, nan=0.0),
                torch.zeros_like(logits),
            )
        else:
            m_f = torch.ones((B, N), device=device, dtype=dtype)
            y_clean = torch.nan_to_num(logits, nan=0.0)

        n_active = m_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        a_raw = torch.tanh(y_clean) * m_f
        mean_a = a_raw.sum(dim=-1, keepdim=True) / n_active
        a_demeaned = (a_raw - mean_a) * m_f

        l1_norm = torch.abs(a_demeaned).sum(dim=-1, keepdim=True)
        w = self.gross_leverage * a_demeaned / (l1_norm + self.eps)
        return w * m_f


class VolTargetedContinuousRiskHead(nn.Module):
    r"""
    DeePM Volatility-Targeted Continuous Risk Head (Wood, Roberts & Zohren, Jan 2026):
        p_{i,t} = \tanh(y_{i,t}) \cdot m_{i,t}, \qquad
        \tilde{w}_{i,t} = \frac{p_{i,t}}{\hat{\sigma}_{i,t}}
    Decouples per-asset directional conviction from universe cardinality. When
    ``long_only=True`` (default for simplex allocators), projects ``\operatorname{softplus}(y_{i,t}) / \hat{\sigma}_{i,t}``
    onto the active probability simplex; when ``long_only=False``, normalizes to target
    gross risk budget ``target_leverage``.
    """

    def __init__(
        self,
        long_only: bool = True,
        target_leverage: float = 1.0,
        eps_vol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.long_only = bool(long_only)
        self.target_leverage = float(max(1e-3, target_leverage))
        self.eps_vol = float(eps_vol)

    def forward(
        self,
        logits: torch.Tensor,
        x_returns: torch.Tensor,
        mask: torch.Tensor | None = None,
        implied_vol: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N = logits.shape
        device = logits.device
        dtype = logits.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            m_f = m_bool.to(dtype=dtype)
            y_clean = torch.where(
                m_bool,
                torch.nan_to_num(logits, nan=0.0),
                torch.zeros_like(logits),
            )
            r_clean = torch.where(
                m_bool.unsqueeze(1),
                torch.nan_to_num(x_returns, nan=0.0),
                torch.zeros_like(x_returns),
            )
        else:
            m_f = torch.ones((B, N), device=device, dtype=dtype)
            y_clean = torch.nan_to_num(logits, nan=0.0)
            r_clean = torch.nan_to_num(x_returns, nan=0.0)

        if implied_vol is not None:
            sigma_hat = torch.nan_to_num(
                implied_vol.to(device=device, dtype=dtype), nan=0.01
            ).clamp(min=self.eps_vol)
            if sigma_hat.dim() == 1:
                sigma_hat = sigma_hat.unsqueeze(0).expand(B, N)
        else:
            sigma_hat = r_clean.std(dim=1, unbiased=False).clamp(min=self.eps_vol)

        if self.long_only:
            conviction = torch.nn.functional.softplus(y_clean) * m_f
            raw_w = (conviction / sigma_hat) * m_f
            denom = raw_w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            return raw_w / denom

        conviction = torch.tanh(y_clean) * m_f
        raw_w = (conviction / sigma_hat) * m_f
        denom = torch.abs(raw_w).sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return self.target_leverage * (raw_w / denom) * m_f


class SetTransformerPolicyNet(BasePolicyNetwork):
    r"""
    Unified Pillar 4 Size- and Permutation-Equivariant Portfolio Policy Network.

    Honors PyAlloq's ``BasePolicyNetwork.forward(x, w_current=None)`` contract while
    supporting optional ``mask``, ``attn_bias``, and ``static_metadata`` arguments:
    - Extracts equivariant asset states ``Z_t in R^{B x N x d}`` and the strictly
      permutation-invariant ``PMA_1`` global ``[REGIME]`` token ``z_regime in R^{B x d}``.
    - Routes equivariant asset logits into one of five mandate-specific heads:
      ``"capped_simplex"``, ``"sparsemax"``, ``"masked_dirichlet"``, ``"dollar_neutral"``,
      or ``"vol_targeted"``.
    - Exposes an invariant Actor-Critic value head ``evaluate_value(x, ...) -> V_psi(z_regime)``
      and Dirichlet log-probability evaluation for ``POMDPPortfolioEnv`` and ``SB3PortfolioAgent``.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_inds: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        mandate: MandateHeadType = "capped_simplex",
        max_weight: float = 0.25,
        include_cash: bool = False,
        gross_leverage: float = 1.0,
        vol_targeted_long_only: bool = True,
        use_ssmax: bool = True,
        block_type: SetBlockType = "isab",
        temporal_type: TemporalEncoderType = "lstm",
        temporal_encoder: nn.Module | None = None,
        norm_style: NormStyle = "pre",
        directed_delay: bool = False,
        static_metadata_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_features = int(max(1, n_features))
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_inds = int(num_inds)
        self.mandate: MandateHeadType = mandate
        self.max_weight = float(max_weight)
        self.include_cash = bool(include_cash)

        self.backbone = SetTransformerBackbone(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_inds=self.num_inds,
            num_layers=num_layers,
            dropout=dropout,
            use_ssmax=use_ssmax,
            block_type=block_type,
            temporal_type=temporal_type,
            temporal_encoder=temporal_encoder,
            norm_style=norm_style,
            directed_delay=directed_delay,
            static_metadata_dim=static_metadata_dim,
        )

        # Equivariant asset scoring head combining z_{i,t} with global z_regime context
        self.scoring_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

        # Permutation-invariant RL critic V_psi(z_regime) and auxiliary regime logits
        self.value_critic_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Mandate output heads
        self.capped_simplex_head = CappedSimplexProjection(max_weight=self.max_weight)
        self.sparsemax_head = CappedSimplexProjection(max_weight=1.0)
        self.dirichlet_head = MaskedDirichletCashHead(
            hidden_dim=self.hidden_dim, include_cash=self.include_cash
        )
        self.dollar_neutral_head = DollarNeutralStatArbHead(
            gross_leverage=gross_leverage
        )
        self.vol_targeted_head = VolTargetedContinuousRiskHead(
            long_only=vol_targeted_long_only,
            target_leverage=gross_leverage,
        )

        self.last_regime_token_: torch.Tensor | None = None
        self.last_value_estimate_: torch.Tensor | None = None
        self.last_raw_logits_: torch.Tensor | None = None

    def encode_representations(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        static_metadata: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns ``(raw_logits, z_assets, h_inducing, z_regime)``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        B, _, N, _ = x.shape
        device = x.device
        dtype = x.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            if m_bool.dim() == 1:
                m_bool = m_bool.unsqueeze(0).expand(B, N)
        else:
            m_bool = None

        z_assets, h_inducing, z_regime = self.backbone(
            x=x,
            w_current=w_current,
            mask=m_bool,
            attn_bias=attn_bias,
            static_metadata=static_metadata,
        )
        z_reg_exp = z_regime.unsqueeze(1).expand(B, N, self.hidden_dim)
        joint = torch.cat([z_assets, z_reg_exp], dim=-1)
        raw_logits = self.scoring_head(joint).squeeze(-1)
        if m_bool is not None:
            raw_logits = raw_logits * m_bool.to(dtype=dtype)

        self.last_regime_token_ = z_regime
        self.last_value_estimate_ = self.value_critic_head(z_regime).squeeze(-1)
        self.last_raw_logits_ = raw_logits
        return raw_logits, z_assets, h_inducing, z_regime

    def evaluate_value(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        static_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the strictly permutation-invariant state value critic ``V_psi(z_regime)`` ``(B,)``.
        """
        _, _, _, z_regime = self.encode_representations(
            x=x,
            w_current=w_current,
            mask=mask,
            attn_bias=attn_bias,
            static_metadata=static_metadata,
        )
        return self.value_critic_head(z_regime).squeeze(-1)

    def get_dirichlet_concentrations(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        static_metadata: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns ``(alpha_full, mask_full, state_value)`` for Masked Dirichlet Actor-Critic RL.
        """
        raw_logits, _, _, z_regime = self.encode_representations(
            x=x,
            w_current=w_current,
            mask=mask,
            attn_bias=attn_bias,
            static_metadata=static_metadata,
        )
        alpha_full, mask_full = self.dirichlet_head.get_concentrations(
            asset_logits=raw_logits, z_regime=z_regime, mask=mask
        )
        val = self.value_critic_head(z_regime).squeeze(-1)
        return alpha_full, mask_full, val

    def forward(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        static_metadata: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        raw_logits, _, _, z_regime = self.encode_representations(
            x=x,
            w_current=w_current,
            mask=mask,
            attn_bias=attn_bias,
            static_metadata=static_metadata,
        )

        if self.mandate == "capped_simplex":
            return self.capped_simplex_head(raw_logits, mask=mask)
        if self.mandate == "sparsemax":
            return self.sparsemax_head(raw_logits, mask=mask)
        if self.mandate == "masked_dirichlet":
            return self.dirichlet_head(
                asset_logits=raw_logits,
                z_regime=z_regime,
                mask=mask,
                deterministic=deterministic,
            )
        if self.mandate == "dollar_neutral":
            return self.dollar_neutral_head(raw_logits, mask=mask)
        if self.mandate == "vol_targeted":
            return self.vol_targeted_head(
                logits=raw_logits,
                x_returns=x[:, :, :, 0],
                mask=mask,
            )

        raise ValueError(f"Unsupported mandate: {self.mandate}")
