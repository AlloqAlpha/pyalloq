from __future__ import annotations

import copy

import torch
from torch import nn

from pyalloq.dl.layers.set_transformer import (
    ISAB,
    NormStyle,
    SetBlockType,
    SetTransformerBackbone,
    TemporalEncoderType,
)


class VICRegLoss(nn.Module):
    r"""
    Mask-Aware Variance-Invariance-Covariance Regularization (`VICReg`, Bardes et al., ICLR 2022)
    for Joint-Embedding Predictive Architectures (`MarketJEPA`):
        \mathcal{L}_{\text{VICReg}}(Z_{\text{pred}}, Z_{\text{tgt}}; M_t)
        = \lambda_{\text{inv}} \mathcal{L}_{\text{inv}}
          + \mu_{\text{var}} \mathcal{L}_{\text{var}}
          + \nu_{\text{cov}} \mathcal{L}_{\text{cov}}

    Crucially, variance and covariance penalties are computed strictly over active tokens
    where ``mask == True`` (plus the global ``[REGIME]`` token at index 0) so padded
    pre-IPO or delisted asset slots never bias latent statistics.
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma_target_std: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.sim_coeff = float(sim_coeff)
        self.std_coeff = float(std_coeff)
        self.cov_coeff = float(cov_coeff)
        self.gamma_target_std = float(gamma_target_std)
        self.eps = float(eps)

    def _variance_and_covariance_loss(
        self,
        z_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes variance hinge loss and off-diagonal covariance penalty over ``M_act``
        active token embeddings of shape ``(M_act, d)``.
        """
        m_count, d = z_active.shape
        if m_count <= 1:
            zero = torch.tensor(0.0, device=z_active.device, dtype=z_active.dtype)
            return zero, zero

        z_centered = z_active - z_active.mean(dim=0, keepdim=True)
        std = torch.sqrt(z_centered.var(dim=0, unbiased=False) + self.eps)
        var_loss = torch.mean(torch.relu(self.gamma_target_std - std))

        cov = (z_centered.T @ z_centered) / float(max(1, m_count - 1))
        eye = torch.eye(d, device=z_active.device, dtype=z_active.dtype)
        off_diag = cov * (1.0 - eye)
        cov_loss = off_diag.pow(2).sum() / float(d)
        return var_loss, cov_loss

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_pred : torch.Tensor
            Predicted latent representation of shape ``(B, N + 1, d)`` with ``[REGIME]`` at index 0.
        z_target : torch.Tensor
            Target EMA latent representation of shape ``(B, N + 1, d)`` with ``[REGIME]`` at index 0.
        mask : torch.Tensor, optional
            Asset tradability mask of shape ``(B, N)`` or token mask ``(B, N + 1)``.
        """
        B, n_tokens, d = z_pred.shape
        device = z_pred.device
        dtype = z_pred.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            if m_bool.dim() == 1:
                m_bool = m_bool.unsqueeze(0).expand(B, -1)
            if m_bool.shape[1] == n_tokens - 1:
                # Prepend True for the global [REGIME] token at index 0
                reg_true = torch.ones((B, 1), device=device, dtype=torch.bool)
                token_mask = torch.cat([reg_true, m_bool], dim=1)
            else:
                token_mask = m_bool
        else:
            token_mask = torch.ones((B, n_tokens), device=device, dtype=torch.bool)

        m_f = token_mask.unsqueeze(-1).to(dtype=dtype)
        active_count = token_mask.sum().clamp(min=1).to(dtype=dtype)

        # 1. Invariance (MSE) loss strictly over active tokens
        diff_sq = ((z_pred - z_target) * m_f).pow(2).sum(dim=-1)
        sim_loss = diff_sq.sum() / active_count

        # 2. Variance & Covariance regularization strictly over active unmasked tokens
        flat_mask = token_mask.reshape(-1)
        pred_active = z_pred.reshape(-1, d)[flat_mask]
        tgt_active = z_target.reshape(-1, d)[flat_mask]

        var_pred, cov_pred = self._variance_and_covariance_loss(pred_active)
        var_tgt, cov_tgt = self._variance_and_covariance_loss(tgt_active)

        var_loss = 0.5 * (var_pred + var_tgt)
        cov_loss = 0.5 * (cov_pred + cov_tgt)

        return (
            self.sim_coeff * sim_loss
            + self.std_coeff * var_loss
            + self.cov_coeff * cov_loss
        )


class LatentPredictor(nn.Module):
    r"""
    Permutation-Equivariant Set Transformer Latent Predictor ``g_\phi`` for ``MarketJEPA``.

    Maps the context representation ``(B, N + 1, d)`` (where index 0 is ``[REGIME]`` and
    indices ``1..N`` are asset tokens) to predicted target horizon representations
    ``(B, N + 1, d)`` in ``O(N m)`` time using an ``ISAB`` block.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_inds: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.isab = ISAB(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            num_inds=num_inds,
            dropout=dropout,
            use_ssmax=True,
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        z_tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, n_tokens, _ = z_tokens.shape
        device = z_tokens.device
        dtype = z_tokens.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            if m_bool.dim() == 1:
                m_bool = m_bool.unsqueeze(0).expand(B, -1)
            if m_bool.shape[1] == n_tokens - 1:
                reg_true = torch.ones((B, 1), device=device, dtype=torch.bool)
                token_mask = torch.cat([reg_true, m_bool], dim=1)
            else:
                token_mask = m_bool
        else:
            token_mask = None

        h = self.in_proj(z_tokens)
        z_att = self.isab(h, mask=token_mask, return_inducing=False)
        assert isinstance(z_att, torch.Tensor)
        out = z_tokens + self.out_proj(z_att)
        if token_mask is not None:
            out = out * token_mask.unsqueeze(-1).to(dtype=dtype)
        return out


class MarketJEPA(nn.Module):
    r"""
    Pillar 3 Joint-Embedding Predictive Architecture (`MarketJEPA`) powered by
    ``SetTransformerBackbone`` + ``PMA_1``.

    - **Context Encoder ``f_\theta``**: Stacked ``ISAB_m`` layers followed by ``PMA_1(Z)``
      prepended at token index 0, outputting ``(B, N + 1, d)`` where token 0 is the strictly
      permutation-invariant ``[REGIME]`` token and tokens ``1..N`` are 1D permutation-equivariant
      asset representations in ``O(N m)`` time.
    - **EMA Target Encoder ``f_\xi``**: Momentum-updated copy of ``f_\theta`` (no gradient flow).
    - **Latent Predictor ``g_\phi`` & Mask-Aware ``VICRegLoss``**: Predicts future target latent
      representations and evaluates variance/covariance penalties strictly over ``mask == True``
      tokens.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_inds: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        ema_decay: float = 0.99,
        use_ssmax: bool = True,
        block_type: SetBlockType = "isab",
        temporal_type: TemporalEncoderType = "lstm",
        norm_style: NormStyle = "pre",
        directed_delay: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = int(max(1, n_features))
        self.hidden_dim = int(hidden_dim)
        self.ema_decay = float(ema_decay)

        self.context_encoder = SetTransformerBackbone(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            num_inds=num_inds,
            num_layers=num_layers,
            dropout=dropout,
            use_ssmax=use_ssmax,
            block_type=block_type,
            temporal_type=temporal_type,
            norm_style=norm_style,
            directed_delay=directed_delay,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        self.predictor = LatentPredictor(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            num_inds=num_inds,
            dropout=dropout,
        )
        self.vicreg_loss = VICRegLoss()

    @torch.no_grad()
    def update_target_ema(self, decay: float | None = None) -> None:
        """
        Updates the EMA target encoder weights ``xi <- tau * xi + (1 - tau) * theta``.
        """
        tau = float(decay if decay is not None else self.ema_decay)
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(tau).add_(p_ctx.data, alpha=1.0 - tau)

    def encode_context(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes market window ``x`` ``(B, L, N, F)`` into ``(B, N + 1, d)`` with the
        strictly permutation-invariant ``[REGIME]`` token at index 0 and equivariant
        asset tokens at indices ``1..N``.
        """
        z_assets, _, z_regime = self.context_encoder(
            x=x, mask=mask, attn_bias=attn_bias
        )
        return torch.cat([z_regime.unsqueeze(1), z_assets], dim=1)

    @torch.no_grad()
    def encode_target(
        self,
        x_target: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes future target window ``x_target`` ``(B, L, N, F)`` via the EMA target encoder
        into ``(B, N + 1, d)`` with ``[REGIME]`` at index 0.
        """
        z_assets, _, z_regime = self.target_encoder(
            x=x_target, mask=mask, attn_bias=attn_bias
        )
        return torch.cat([z_regime.unsqueeze(1), z_assets], dim=1)

    def forward(
        self,
        x_context: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass returning the context token tensor ``(B, N + 1, d)`` with ``[REGIME]``
        at index 0.
        """
        return self.encode_context(x=x_context, mask=mask, attn_bias=attn_bias)

    def compute_jepa_loss(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the self-supervised mask-aware VICReg predictive loss between predicted
        latent tokens ``g_phi(f_theta(x_context))`` and EMA target tokens ``f_xi(x_target)``.
        """
        z_ctx = self.encode_context(x=x_context, mask=mask, attn_bias=attn_bias)
        z_pred = self.predictor(z_ctx, mask=mask)
        with torch.no_grad():
            z_tgt = self.encode_target(
                x_target=x_target, mask=mask, attn_bias=attn_bias
            )
        return self.vicreg_loss(z_pred=z_pred, z_target=z_tgt, mask=mask)
