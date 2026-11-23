from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from pyalloq.dl.layers.neural_jump_rde import (
    NeuralJumpRDE,
    TorchLogSignatureExtractor,
)

NormStyle = Literal["pre", "post"]
SetBlockType = Literal["sab", "isab", "auto"]
TemporalEncoderType = Literal["lstm", "vsn_conv", "logsig_ssm", "neural_jump_rde"]


class ScalableMultiheadAttention(nn.Module):
    r"""
    Multi-Head Scaled Dot-Product Attention with Scalable-Softmax (SSMax, Nakanishi 2025),
    strict boolean key/query masking for ragged asset universes, and Graphormer-style
    optional additive attention biases (Ying et al., NeurIPS 2021).

    When ``use_ssmax=True``, scales query vectors per attention head by:
        (s_head * log(max(2.0, N_active))) / sqrt(d_k)
    where ``N_active = sum_{j=1}^{N_k} key_mask_j`` is the active asset cardinality
    for each sample in the batch, eliminating attention fading across 50-to-500 stock transfers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_ssmax: bool = True,
        ssmax_init: float = 0.5,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.use_ssmax = bool(use_ssmax)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable per-head SSMax temperature scale s_head (initialized near 0.5)
        self.ssmax_scale = nn.Parameter(
            torch.full((1, self.num_heads, 1, 1), float(ssmax_init))
        )
        # Learnable per-head Graphormer prior gate gamma_head (initialized near 0.0 for smooth warmup)
        self.attn_bias_gate = nn.Parameter(torch.zeros((1, self.num_heads, 1, 1)))
        self.attn_bias_proj = nn.Sequential(
            nn.Linear(1, self.num_heads),
            nn.Tanh(),
        )

        self.last_attention_weights_: torch.Tensor | None = None
        self.last_attention_entropy_: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape ``(B, N_q, d)``.
        key : torch.Tensor
            Key tensor of shape ``(B, N_k, d)``.
        value : torch.Tensor
            Value tensor of shape ``(B, N_k, d)``.
        key_mask : torch.Tensor, optional
            Boolean or binary tradability mask of shape ``(B, N_k)`` where ``True`` / ``1``
            indicates an active tradable asset and ``False`` / ``0`` indicates a padded slot.
        query_mask : torch.Tensor, optional
            Boolean or binary mask of shape ``(B, N_q)`` zeroing out padded query rows.
        attn_bias : torch.Tensor, optional
            Optional Graphormer relational prior of shape ``(B, N_q, N_k)`` or ``(N_q, N_k)``
            or ``(B, H, N_q, N_k)``.

        Returns
        -------
        torch.Tensor
            Attended output tensor of shape ``(B, N_q, d)``.
        """
        B, N_q, _ = query.shape
        N_k = key.shape[1]
        device = query.device
        dtype = query.dtype

        # Sanitize masked key/value and query rows BEFORE linear projections so extreme
        # sentinel or NaN values in padded slots never contaminate active representations
        if key_mask is not None:
            k_bool = key_mask.to(device=device, dtype=torch.bool)
            k_mask_3d = k_bool.unsqueeze(-1)
            key = torch.where(
                k_mask_3d, torch.nan_to_num(key, nan=0.0), torch.zeros_like(key)
            )
            value = torch.where(
                k_mask_3d, torch.nan_to_num(value, nan=0.0), torch.zeros_like(value)
            )
        else:
            k_bool = None

        if query_mask is not None:
            q_bool = query_mask.to(device=device, dtype=torch.bool)
            q_mask_3d = q_bool.unsqueeze(-1)
            query = torch.where(
                q_mask_3d, torch.nan_to_num(query, nan=0.0), torch.zeros_like(query)
            )
        else:
            q_bool = None

        q = (
            self.q_proj(query)
            .view(B, N_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = (
            self.v_proj(value)
            .view(B, N_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        inv_sqrt_d = float(self.head_dim) ** -0.5
        if self.use_ssmax:
            if k_bool is not None:
                n_active = (
                    k_bool.sum(dim=-1, keepdim=True).to(dtype=dtype).clamp(min=2.0)
                )
            else:
                n_active = torch.full(
                    (B, 1), max(2.0, float(N_k)), device=device, dtype=dtype
                )
            log_n = torch.log(n_active).view(B, 1, 1, 1)
            scale = self.ssmax_scale.clamp(min=0.05, max=4.0) * log_n * inv_sqrt_d
            q = q * scale
        else:
            q = q * inv_sqrt_d

        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, N_q, N_k)

        if attn_bias is not None:
            bias_t = torch.nan_to_num(attn_bias.to(device=device, dtype=dtype), nan=0.0)
            if bias_t.dim() == 2:
                bias_t = bias_t.unsqueeze(0).expand(B, -1, -1)
            if bias_t.dim() == 3:
                # Project scalar edge weights to H heads: (B, N_q, N_k, 1) -> (B, H, N_q, N_k)
                mapped_bias = self.attn_bias_proj(bias_t.unsqueeze(-1)).permute(
                    0, 3, 1, 2
                )
            elif bias_t.dim() == 4:
                mapped_bias = bias_t
            else:
                raise ValueError(f"Unsupported attn_bias shape: {attn_bias.shape}")
            scores = scores + self.attn_bias_gate * mapped_bias

        if k_bool is not None:
            # Mask inactive keys with -inf pre-softmax
            key_keep = k_bool.view(B, 1, 1, N_k)
            all_masked = ~k_bool.any(dim=-1, keepdim=True).view(B, 1, 1, 1)
            scores = torch.where(
                key_keep | all_masked,
                scores,
                torch.full_like(scores, float("-inf")),
            )

        attn_weights = torch.softmax(scores, dim=-1)
        if k_bool is not None:
            key_keep_f = k_bool.view(B, 1, 1, N_k).to(dtype=dtype)
            attn_weights = attn_weights * key_keep_f
            denom = attn_weights.sum(dim=-1, keepdim=True)
            attn_weights = torch.where(
                denom > 0.0,
                attn_weights / denom.clamp(min=1e-12),
                torch.zeros_like(attn_weights),
            )

        if q_bool is not None:
            q_keep_f = q_bool.view(B, 1, N_q, 1).to(dtype=dtype)
            attn_weights = attn_weights * q_keep_f

        with torch.no_grad():
            self.last_attention_weights_ = attn_weights.mean(dim=1).detach()
            p_safe = attn_weights.clamp(min=1e-12)
            ent = -(attn_weights * torch.log(p_safe)).sum(dim=-1)
            self.last_attention_entropy_ = ent.mean(dim=(1, 2)).detach()

        attn_dropped = self.dropout(attn_weights)
        context = torch.matmul(attn_dropped, v)  # (B, H, N_q, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)
        out = self.out_proj(context)

        if q_bool is not None:
            out = out * q_bool.unsqueeze(-1).to(dtype=dtype)

        return out


class MAB(nn.Module):
    r"""
    Multihead Attention Block (Lee et al., ICML 2019, Section 3.1):
        H = LayerNorm(X + Multihead(X, Y, Y))
        MAB(X, Y) = LayerNorm(H + rFF(H))
    Supports both Pre-LayerNorm (``norm_style="pre"``, Graphormer) and Post-LayerNorm
    (``norm_style="post"``, Lee et al. 2019), operating strictly row-wise per element.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        use_ssmax: bool = True,
        norm_style: NormStyle = "pre",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim if ffn_dim is not None else 2 * embed_dim)
        self.norm_style: NormStyle = norm_style

        self.attn = ScalableMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            use_ssmax=use_ssmax,
        )
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.ln_y = nn.LayerNorm(self.embed_dim)
        self.rff = nn.Sequential(
            nn.Linear(self.embed_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, self.embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        if query_mask is not None:
            q_bool = query_mask.to(device=device, dtype=torch.bool)
            x = torch.where(
                q_bool.unsqueeze(-1),
                torch.nan_to_num(x, nan=0.0),
                torch.zeros_like(x),
            )
        else:
            q_bool = None

        if key_mask is not None:
            k_bool = key_mask.to(device=device, dtype=torch.bool)
            y = torch.where(
                k_bool.unsqueeze(-1),
                torch.nan_to_num(y, nan=0.0),
                torch.zeros_like(y),
            )
        else:
            k_bool = None

        if self.norm_style == "pre":
            x_norm = self.ln1(x)
            y_norm = self.ln_y(y)
            if q_bool is not None:
                x_norm = x_norm * q_bool.unsqueeze(-1).to(dtype=dtype)
            if k_bool is not None:
                y_norm = y_norm * k_bool.unsqueeze(-1).to(dtype=dtype)
            attn_out = self.attn(
                query=x_norm,
                key=y_norm,
                value=y_norm,
                key_mask=k_bool,
                query_mask=q_bool,
                attn_bias=attn_bias,
            )
            h = x + attn_out
            if q_bool is not None:
                h = h * q_bool.unsqueeze(-1).to(dtype=dtype)
            h_norm = self.ln2(h)
            if q_bool is not None:
                h_norm = h_norm * q_bool.unsqueeze(-1).to(dtype=dtype)
            out = h + self.rff(h_norm)
        else:
            attn_out = self.attn(
                query=x,
                key=y,
                value=y,
                key_mask=k_bool,
                query_mask=q_bool,
                attn_bias=attn_bias,
            )
            h = self.ln1(x + attn_out)
            if q_bool is not None:
                h = h * q_bool.unsqueeze(-1).to(dtype=dtype)
            out = self.ln2(h + self.rff(h))

        if q_bool is not None:
            out = out * q_bool.unsqueeze(-1).to(dtype=dtype)
        return out


class SAB(nn.Module):
    r"""
    Set Attention Block (Lee et al., ICML 2019):
        SAB(X) := MAB(X, X)
    Strictly permutation-equivariant pairwise self-attention across all N assets,
    supporting optional Directed Delay (``x_kv`` lagged by t-1) and Graphormer ``attn_bias``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        use_ssmax: bool = True,
        norm_style: NormStyle = "pre",
    ) -> None:
        super().__init__()
        self.mab = MAB(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_ssmax=use_ssmax,
            norm_style=norm_style,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        x_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = x if x_kv is None else x_kv
        return self.mab(
            x=x,
            y=kv,
            key_mask=mask,
            query_mask=mask,
            attn_bias=attn_bias,
        )


class ISAB(nn.Module):
    r"""
    Induced Set Attention Block (Lee et al., ICML 2019):
        H = MAB_1(I, X) in R^{B x m x d}   (Stage 1: O(Nm) permutation-invariant inducing factors)
        Z = MAB_2(X, H) in R^{B x N x d}   (Stage 2: O(Nm) permutation-equivariant asset states)

    - Stage 1 attends over N_active variable-cardinality asset keys and uses Scalable-Softmax (SSMax).
    - Stage 2 attends over the fixed m inducing points and uses standard Softmax.
    - When an optional N x N graph adjacency prior ``attn_bias`` is provided to ISAB, computes
      O(Nm) Graphormer Node Degree / Centrality structural encodings (normalized weighted degree,
      max neighbor coupling, and active degree ratio) and fuses them into X without materializing
      quadratic O(N^2) attention layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_inds: int = 16,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        use_ssmax: bool = True,
        norm_style: NormStyle = "pre",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_inds = int(num_inds)

        self.inducing_points = nn.Parameter(
            torch.randn(1, self.num_inds, self.embed_dim)
            / (float(self.embed_dim) ** 0.5)
        )
        # Stage 1: Inducing points query N_active assets (uses SSMax)
        self.mab1 = MAB(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_ssmax=use_ssmax,
            norm_style=norm_style,
        )
        # Stage 2: Assets query fixed m inducing points (standard Softmax)
        self.mab2 = MAB(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_ssmax=False,
            norm_style=norm_style,
        )

        # Graphormer Node Degree / Centrality structural encoding projection for O(Nm) prior fusion
        self.centrality_gate = nn.Parameter(torch.tensor(0.0))
        self.centrality_proj = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Tanh(),
        )
        self.last_inducing_attention_: torch.Tensor | None = None

    def _inject_centrality_encoding(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        device = x.device
        dtype = x.dtype

        adj = torch.nan_to_num(attn_bias.to(device=device, dtype=dtype), nan=0.0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        elif adj.dim() == 4:
            adj = adj.mean(dim=1)

        if mask is not None:
            m_f = mask.to(device=device, dtype=dtype)
            pair_mask = m_f.unsqueeze(-1) * m_f.unsqueeze(-2)
            adj = adj * pair_mask
            n_act = m_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            m_f = torch.ones((B, N), device=device, dtype=dtype)
            n_act = torch.full((B, 1), max(1.0, float(N)), device=device, dtype=dtype)

        abs_adj = torch.abs(adj)
        mean_deg = abs_adj.sum(dim=-1) / n_act
        max_deg = abs_adj.max(dim=-1).values
        pos_ratio = (adj > 0.0).to(dtype=dtype).sum(dim=-1) / n_act

        struct_feats = torch.stack([mean_deg, max_deg, pos_ratio], dim=-1)  # (B, N, 3)
        struct_emb = self.centrality_proj(struct_feats) * m_f.unsqueeze(-1)
        return x + torch.tanh(self.centrality_gate) * struct_emb

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        x_kv: torch.Tensor | None = None,
        return_inducing: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        device = x.device

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            x = torch.where(
                m_bool.unsqueeze(-1),
                torch.nan_to_num(x, nan=0.0),
                torch.zeros_like(x),
            )
            if x_kv is not None:
                x_kv = torch.where(
                    m_bool.unsqueeze(-1),
                    torch.nan_to_num(x_kv, nan=0.0),
                    torch.zeros_like(x_kv),
                )
        else:
            m_bool = None

        if attn_bias is not None:
            x = self._inject_centrality_encoding(x, attn_bias, m_bool)
            if x_kv is not None:
                x_kv = self._inject_centrality_encoding(x_kv, attn_bias, m_bool)

        kv = x if x_kv is None else x_kv
        ind = self.inducing_points.expand(B, -1, -1)

        # Stage 1: H = MAB_1(I, X_kv) -> strictly permutation-invariant in (B, m, d)
        h_inducing = self.mab1(
            x=ind,
            y=kv,
            key_mask=m_bool,
            query_mask=None,
            attn_bias=None,
        )
        self.last_inducing_attention_ = self.mab1.attn.last_attention_weights_

        # Stage 2: Z = MAB_2(X_q, H) -> strictly permutation-equivariant in (B, N, d)
        z_assets = self.mab2(
            x=x,
            y=h_inducing,
            key_mask=None,
            query_mask=m_bool,
            attn_bias=None,
        )

        if return_inducing:
            return z_assets, h_inducing
        return z_assets


class PMA(nn.Module):
    r"""
    Pooling by Multihead Attention (Lee et al., ICML 2019, Section 3.2):
        PMA_k(Z) = MAB(S, rFF(Z)) in R^{B x k x d}
    Aggregates an arbitrary-size encoded asset set Z in R^{B x N x d} into k
    strictly permutation-invariant global summary tokens (e.g., k=1 for [REGIME]).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_seeds: int = 1,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        use_ssmax: bool = True,
        norm_style: NormStyle = "pre",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_seeds = int(num_seeds)

        self.seed_vectors = nn.Parameter(
            torch.randn(1, self.num_seeds, self.embed_dim)
            / (float(self.embed_dim) ** 0.5)
        )
        self.rff_in = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.mab = MAB(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_ssmax=use_ssmax,
            norm_style=norm_style,
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z.shape[0]
        device = z.device
        dtype = z.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            z_clean = torch.where(
                m_bool.unsqueeze(-1),
                torch.nan_to_num(z, nan=0.0),
                torch.zeros_like(z),
            )
            z_ff = self.rff_in(z_clean) * m_bool.unsqueeze(-1).to(dtype=dtype)
        else:
            m_bool = None
            z_ff = self.rff_in(z)

        seeds = self.seed_vectors.expand(B, -1, -1)
        return self.mab(
            x=seeds,
            y=z_ff,
            key_mask=m_bool,
            query_mask=None,
            attn_bias=None,
        )


class VectorizedVariableSelectionNetwork(nn.Module):
    r"""
    Vectorized Variable Selection Network & Temporal Conv1D Encoder (`V-VSN`, Wood et al. 2026).

    Processes each asset's multivariate lookback sequence ``(B * N, L, F)`` identically:
    1. Projects each feature channel ``f = 1..F`` into a shared latent space and computes
       Gated Residual Network (GRN) softmax variable selection weights across channels.
    2. Applies causal temporal depthwise-separable Conv1D + gated residual summarization
       to produce a per-asset temporal embedding of shape ``(B * N, hidden_dim)``.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(max(1, n_features))
        self.hidden_dim = int(hidden_dim)

        self.var_proj = nn.Linear(1, self.hidden_dim)
        self.var_gate = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.n_features),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
        )
        self.grn_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GLU(dim=-1),
        )
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_flat : torch.Tensor
            Flattened asset sequence tensor of shape ``(M, L, F)`` where ``M = B * N``.
        """
        M, L, F = x_flat.shape
        if F < self.n_features:
            pad = torch.zeros(
                (M, L, self.n_features - F), device=x_flat.device, dtype=x_flat.dtype
            )
            x_in = torch.cat([x_flat, pad], dim=-1)
        elif F > self.n_features:
            x_in = x_flat[..., : self.n_features]
        else:
            x_in = x_flat

        # Channel selection weights: (M, L, F)
        v_weights = torch.softmax(self.var_gate(x_in), dim=-1).unsqueeze(-1)
        # Per-feature embedding: (M, L, F, H)
        v_emb = self.var_proj(x_in.unsqueeze(-1))
        selected = (v_weights * v_emb).sum(dim=2)  # (M, L, H)

        conv_in = selected.transpose(1, 2)  # (M, H, L)
        conv_out = self.temporal_conv(conv_in).transpose(1, 2)  # (M, L, H)
        last_step = selected[:, -1, :] + conv_out[:, -1, :]
        return self.layer_norm(last_step + self.grn_out(last_step))


class _SelectiveLogSigTemporalAdapter(nn.Module):
    """
    Lightweight selective log-signature temporal encoder wrapping ``TorchLogSignatureExtractor``
    so ``SetTransformerBackbone`` can use ``temporal_type="logsig_ssm"`` without circular imports.
    """

    def __init__(self, n_features: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_features = int(max(1, n_features))
        self.hidden_dim = int(hidden_dim)
        self.logsig = TorchLogSignatureExtractor(
            n_channels=self.n_features, window_size=4
        )
        self.proj = nn.Sequential(
            nn.Linear(self.logsig.logsig_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.logsig(x_flat)
        h_seq = self.proj(tokens)
        out, _ = self.gru(h_seq)
        return self.norm(out[:, -1, :])


class SetTransformerBackbone(nn.Module):
    r"""
    Unified Permutation-Equivariant Set Transformer Backbone for PyAlloq.

    Architecture:
    1. **Per-Asset Temporal Encoder**: Processes each asset's time series ``(L, F)`` identically
       with shared weights (``"lstm"``, ``"vsn_conv"``, ``"logsig_ssm"``, ``"neural_jump_rde"``,
       or a custom ``nn.Module``), guaranteeing 1D permutation equivariance across assets.
    2. **Ticker-Agnostic Metadata & Cardinality-Normalized Drifted Weight Fusion**:
       Optionally fuses ``static_metadata`` ``(B, N, F_static)`` and pre-trade holdings
       ``[w_{i,t^-}, N_active * w_{i,t^-}]`` into each asset's embedding without binding to
       fixed ticker IDs or universe sizes.
    3. **Directed Delay (t - 1) Causal Sieve** (``directed_delay=True``):
       Feeds cross-sectional attention keys and values from lagged lookback ``1 .. L-1``
       while querying with each asset's local time-t state ``1 .. L``, preventing same-day
       asynchronous close leakage across global markets.
    4. **Hybrid SAB / ISAB Cross-Sectional Stack** (``block_type: Literal["sab", "isab", "auto"]``):
       - ``"isab"``: Linear-time ``O(Nm)`` Induced Set Attention Blocks producing both
         equivariant asset embeddings ``Z_t in R^{B x N x d}`` and invariant latent factor
         states ``H_{I,t} in R^{B x m x d}``.
       - ``"sab"``: Full pairwise ``O(N^2)`` Set Attention Blocks (with Graphormer ``attn_bias``),
         followed by a ``PMA_m`` factor extraction head so ``H_{I,t}`` is always available.
       - ``"auto"``: Routes compact panels (``N <= auto_sab_threshold`` when ``attn_bias`` is
         provided) through ``SAB`` and large universes through ``ISAB``.
    5. **Pooling by Multihead Attention (PMA_1)**:
       Extracts the strictly permutation-invariant global ``[REGIME]`` token
       ``z_regime in R^{B x d}``.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_inds: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_ssmax: bool = True,
        block_type: SetBlockType = "isab",
        temporal_type: TemporalEncoderType = "lstm",
        temporal_encoder: nn.Module | None = None,
        norm_style: NormStyle = "pre",
        directed_delay: bool = False,
        static_metadata_dim: int = 0,
        auto_sab_threshold: int = 64,
    ) -> None:
        super().__init__()
        self.n_features = int(max(1, n_features))
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_inds = int(num_inds)
        self.num_layers = int(max(1, num_layers))
        self.use_ssmax = bool(use_ssmax)
        self.block_type: SetBlockType = block_type
        self.temporal_type: TemporalEncoderType = temporal_type
        self.norm_style: NormStyle = norm_style
        self.directed_delay = bool(directed_delay)
        self.static_metadata_dim = int(max(0, static_metadata_dim))
        self.auto_sab_threshold = int(auto_sab_threshold)

        # 1. Per-Asset Temporal Encoder
        self.custom_temporal_encoder = temporal_encoder
        if temporal_encoder is None:
            if temporal_type == "lstm":
                self.lstm = nn.LSTM(
                    input_size=self.n_features,
                    hidden_size=self.hidden_dim,
                    batch_first=True,
                )
                self.temporal_norm = nn.LayerNorm(self.hidden_dim)
                self.temporal_drop = nn.Dropout(dropout)
            elif temporal_type == "vsn_conv":
                self.vsn = VectorizedVariableSelectionNetwork(
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    dropout=dropout,
                )
            elif temporal_type == "logsig_ssm":
                self.logsig_ssm = _SelectiveLogSigTemporalAdapter(
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                )
            elif temporal_type == "neural_jump_rde":
                self.jump_rde = NeuralJumpRDE(
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unsupported temporal_type: {temporal_type}")

        # 2. Equivariant Drifted Weight & Static Metadata Fusion Projections
        self.weight_proj = nn.Linear(2, self.hidden_dim)
        self.metadata_proj: nn.Linear | None
        if self.static_metadata_dim > 0:
            self.metadata_proj = nn.Linear(self.static_metadata_dim, self.hidden_dim)
        else:
            self.metadata_proj = None

        # 3. Cross-Sectional Set Attention Blocks (ISAB and SAB layers)
        self.isab_layers = nn.ModuleList(
            [
                ISAB(
                    embed_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    num_inds=self.num_inds,
                    dropout=dropout,
                    use_ssmax=use_ssmax,
                    norm_style=norm_style,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.sab_layers = nn.ModuleList(
            [
                SAB(
                    embed_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=dropout,
                    use_ssmax=use_ssmax,
                    norm_style=norm_style,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.sab_factor_pma = PMA(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_seeds=self.num_inds,
            dropout=dropout,
            use_ssmax=use_ssmax,
            norm_style=norm_style,
        )

        # 4. Global [REGIME] Token Pooling via PMA_1
        self.regime_pma = PMA(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_seeds=1,
            dropout=dropout,
            use_ssmax=use_ssmax,
            norm_style=norm_style,
        )

    def _encode_temporal_slice(
        self,
        x_slice: torch.Tensor,
        mask_bool: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Encodes a 4D market slice ``(B, L_sub, N, F)`` into per-asset embeddings ``(B, N, d)``.
        """
        B, L_sub, N, F = x_slice.shape
        device = x_slice.device
        dtype = x_slice.dtype

        if mask_bool is not None:
            x_clean = torch.where(
                mask_bool.view(B, 1, N, 1),
                torch.nan_to_num(x_slice, nan=0.0),
                torch.zeros_like(x_slice),
            )
        else:
            x_clean = torch.nan_to_num(x_slice, nan=0.0)

        if F < self.n_features:
            pad = torch.zeros(
                (B, L_sub, N, self.n_features - F), device=device, dtype=dtype
            )
            x_clean = torch.cat([x_clean, pad], dim=-1)
        elif F > self.n_features:
            x_clean = x_clean[..., : self.n_features]

        x_flat = (
            x_clean.permute(0, 2, 1, 3).contiguous().view(B * N, L_sub, self.n_features)
        )

        if self.custom_temporal_encoder is not None:
            raw_emb = self.custom_temporal_encoder(x_flat)
            if isinstance(raw_emb, tuple):
                raw_emb = raw_emb[0]
            if raw_emb.dim() == 3:
                raw_emb = raw_emb[:, -1, :]
            h_temp = raw_emb.view(B, N, self.hidden_dim)
        elif self.temporal_type == "lstm":
            lstm_out, _ = self.lstm(x_flat)
            last_step = self.temporal_drop(lstm_out[:, -1, :])
            h_temp = self.temporal_norm(last_step).view(B, N, self.hidden_dim)
        elif self.temporal_type == "vsn_conv":
            h_temp = self.vsn(x_flat).view(B, N, self.hidden_dim)
        elif self.temporal_type == "logsig_ssm":
            h_temp = self.logsig_ssm(x_flat).view(B, N, self.hidden_dim)
        else:
            h_rde = self.jump_rde(x_flat)
            if isinstance(h_rde, tuple):
                h_rde = h_rde[0]
            h_temp = h_rde.view(B, N, self.hidden_dim)

        if mask_bool is not None:
            h_temp = h_temp * mask_bool.unsqueeze(-1).to(dtype=dtype)
        return h_temp

    def _augment_embeddings(
        self,
        h_temp: torch.Tensor,
        w_current: torch.Tensor | None,
        static_metadata: torch.Tensor | None,
        mask_bool: torch.Tensor | None,
    ) -> torch.Tensor:
        B, N, _ = h_temp.shape
        device = h_temp.device
        dtype = h_temp.dtype

        if mask_bool is not None:
            m_f = mask_bool.to(dtype=dtype)
            n_active = m_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            m_f = torch.ones((B, N), device=device, dtype=dtype)
            n_active = torch.full(
                (B, 1), max(1.0, float(N)), device=device, dtype=dtype
            )

        if w_current is not None:
            w_t = torch.nan_to_num(w_current.to(device=device, dtype=dtype), nan=0.0)
            if w_t.dim() == 1:
                w_t = w_t.unsqueeze(0).expand(B, N)
            w_t = w_t * m_f
            w_scaled = w_t * n_active
            w_feat = torch.stack([w_t, w_scaled], dim=-1)  # (B, N, 2)
            h_temp = h_temp + self.weight_proj(w_feat) * m_f.unsqueeze(-1)

        if static_metadata is not None and self.metadata_proj is not None:
            meta_t = torch.nan_to_num(
                static_metadata.to(device=device, dtype=dtype), nan=0.0
            )
            if meta_t.dim() == 2:
                meta_t = meta_t.unsqueeze(0).expand(B, -1, -1)
            h_temp = h_temp + self.metadata_proj(meta_t) * m_f.unsqueeze(-1)

        return h_temp * m_f.unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        static_metadata: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Market state tensor of shape ``(B, L, N, F)`` or ``(B, L, N)``.
        w_current : torch.Tensor, optional
            Pre-trade drifted weights ``(B, N)`` or ``(N,)``.
        mask : torch.Tensor, optional
            Boolean tradability mask ``(B, N)`` where ``True`` marks active tradable assets.
        attn_bias : torch.Tensor, optional
            Optional Graphormer relational prior ``(B, N, N)`` or ``(N, N)``.
        static_metadata : torch.Tensor, optional
            Optional ticker-agnostic structural metadata ``(B, N, F_static)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - ``z_assets``: Permutation-equivariant asset embeddings ``(B, N, hidden_dim)``
            - ``h_inducing``: Permutation-invariant latent factor embeddings ``(B, m, hidden_dim)``
            - ``z_regime``: Permutation-invariant global ``[REGIME]`` token ``(B, hidden_dim)``
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        B, L, N, _ = x.shape
        device = x.device
        dtype = x.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            if m_bool.dim() == 1:
                m_bool = m_bool.unsqueeze(0).expand(B, N)
        else:
            m_bool = None

        h_q = self._encode_temporal_slice(x, m_bool)
        h_q = self._augment_embeddings(h_q, w_current, static_metadata, m_bool)

        if self.directed_delay and L > 1:
            # Directed Delay (t - 1) Causal Sieve: cross-sectional keys/values come strictly
            # from lagged observations 1..L-1 so time-t cross-asset closes never leak
            h_kv = self._encode_temporal_slice(x[:, :-1, :, :], m_bool)
            h_kv = self._augment_embeddings(h_kv, w_current, static_metadata, m_bool)
        else:
            h_kv = None

        use_sab = self.block_type == "sab" or (
            self.block_type == "auto"
            and attn_bias is not None
            and N <= self.auto_sab_threshold
        )

        z = h_q
        h_ind: torch.Tensor | None = None

        if use_sab:
            for idx, sab in enumerate(self.sab_layers):
                kv_in = h_kv if (idx == 0 and h_kv is not None) else None
                z = sab(z, mask=m_bool, attn_bias=attn_bias, x_kv=kv_in)
            pma_source = h_kv if (self.directed_delay and h_kv is not None) else z
            h_ind = self.sab_factor_pma(pma_source, mask=m_bool)
        else:
            for idx, isab in enumerate(self.isab_layers):
                kv_in = h_kv if (idx == 0 and h_kv is not None) else None
                z_out, h_ind = isab(
                    z,
                    mask=m_bool,
                    attn_bias=attn_bias,
                    x_kv=kv_in,
                    return_inducing=True,
                )
                z = z_out

        assert h_ind is not None
        regime_source = h_kv if (self.directed_delay and h_kv is not None) else z
        z_regime = self.regime_pma(regime_source, mask=m_bool).squeeze(1)

        if m_bool is not None:
            z = z * m_bool.unsqueeze(-1).to(dtype=dtype)

        return z, h_ind, z_regime
