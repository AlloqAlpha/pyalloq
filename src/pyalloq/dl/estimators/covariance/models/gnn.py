from __future__ import annotations

import torch
from torch import nn

from pyalloq.dl.layers.fragility_kkt import FragilityShrinkageCholeskyLayer
from pyalloq.dl.layers.neural_jump_rde import NeuralJumpRDE


class CRISPSparseSTGAT(nn.Module):
    """
    CRISP Sparse Spatio-Temporal Graph Attention Network (`CRISPSparseSTGAT`).

    Combines:
    1. `NeuralJumpRDE` temporal rough-path encoder per asset: maps `(B, L, N, F)` into
       continuous-jump node embeddings `H_temp` of shape `(B, N, hidden_dim)`.
    2. Composite Prior + Learnable Graph Attention with CRISP Top-k / Thresholded Edge Sparsity:
       - Prior adjacency blends positive return correlations and regularized partial correlations
         from the input lookback window.
       - Multi-head graph attention scores are computed across asset nodes and pruned via
         `keep_ratio` (preserving self-loops and the top `ceil(keep_ratio * N)` neighbors per node)
         to eliminate over-smoothing in dense financial correlation networks.
    3. `FragilityShrinkageCholeskyLayer` PSD covariance head and expected return head:
       - `forward(x)` returns `(B, N, N)` strictly PSD covariance tensors (compatible with
         `DeepCovarianceEstimator`).
       - `predict_returns_and_cov(x)` returns `(mu, cov, sparse_adj)` for end-to-end training
         with `DifferentiableRiskParityLayer`, `DifferentiableCVaRLayer`, `MDFPOptimizationLayer`,
         or downstream `SpectralLaplacianHRPAllocator`.
    """

    def __init__(
        self,
        num_assets: int,
        num_features: int,
        hidden_dim: int = 32,
        num_heads: int = 4,
        keep_ratio: float = 0.35,
        signature_window: int = 5,
        ridge_lambda: float = 1e-4,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_assets = int(num_assets)
        self.num_features = int(num_features)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.keep_ratio = float(max(0.05, min(1.0, keep_ratio)))

        self.temporal_encoder = NeuralJumpRDE(
            n_features=num_features,
            hidden_dim=hidden_dim,
            window_size=signature_window,
        )

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prior_bias_scale = nn.Parameter(torch.tensor(0.5))

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, 1)
        self.cov_head = FragilityShrinkageCholeskyLayer(
            hidden_dim=hidden_dim,
            eps=ridge_lambda,
        )

    def _compute_composite_prior_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts a composite prior adjacency `(B, N, N)` from the primary return feature
        `x[:, :, :, 0]` combining sample correlation and precision partial correlation.
        """
        # r: (B, L, N)
        r = x[:, :, :, 0]
        bsz, seq_len, n_assets = r.shape
        if seq_len < 2:
            return (
                torch.eye(n_assets, device=x.device, dtype=x.dtype)
                .unsqueeze(0)
                .expand(bsz, n_assets, n_assets)
            )

        r_centered = r - r.mean(dim=1, keepdim=True)
        cov = torch.bmm(r_centered.transpose(1, 2), r_centered) / max(1, seq_len - 1)
        eye = torch.eye(n_assets, device=x.device, dtype=x.dtype).unsqueeze(0)
        std = torch.sqrt(torch.clamp(torch.diagonal(cov, dim1=-2, dim2=-1), min=1e-8))
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-8)

        # Regularized precision partial correlation
        trace_mean = (
            torch.diagonal(cov, dim1=-2, dim2=-1)
            .mean(dim=-1, keepdim=True)
            .unsqueeze(-1)
        )
        reg_cov = (
            0.85 * cov + 0.15 * torch.clamp(trace_mean, min=1e-6) * eye + 1e-5 * eye
        )
        prec = torch.linalg.inv(reg_cov)
        prec_diag = torch.sqrt(
            torch.clamp(torch.diagonal(prec, dim1=-2, dim2=-1), min=1e-8)
        )
        partial_corr = -prec / (
            prec_diag.unsqueeze(-1) * prec_diag.unsqueeze(-2) + 1e-8
        )

        composite = 0.5 * torch.relu(corr) + 0.5 * torch.relu(partial_corr)
        composite = composite * (1.0 - eye) + eye
        return composite

    def _sparse_graph_attention(
        self, h: torch.Tensor, prior_adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes CRISP sparse multi-head attention over asset nodes.
        """
        bsz, n_assets, _ = h.shape
        q = (
            self.q_proj(h)
            .view(bsz, n_assets, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(h)
            .view(bsz, n_assets, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(h)
            .view(bsz, n_assets, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores + self.prior_bias_scale * prior_adj.unsqueeze(1)

        # CRISP Top-k edge pruning (always keeping self-loops)
        k_keep = max(1, min(n_assets, int(round(self.keep_ratio * n_assets))))
        eye_mask = torch.eye(n_assets, device=h.device, dtype=torch.bool).view(
            1, 1, n_assets, n_assets
        )
        scores_boosted = torch.where(eye_mask, scores + 1e4, scores)
        _, topk_idx = torch.topk(scores_boosted, k=k_keep, dim=-1)
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        keep_mask.scatter_(-1, topk_idx, True)
        keep_mask = keep_mask | eye_mask

        masked_scores = scores.masked_fill(~keep_mask, -1e9)
        attn_weights = torch.softmax(masked_scores, dim=-1)
        attn_weights = attn_weights * keep_mask.to(attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        )

        context = torch.matmul(attn_weights, v)
        context = (
            context.transpose(1, 2).contiguous().view(bsz, n_assets, self.hidden_dim)
        )
        h_out = h + self.out_proj(context)

        # Symmetrized mean sparse adjacency across heads: (B, N, N)
        mean_adj = attn_weights.mean(dim=1)
        sparse_adj = 0.5 * (mean_adj + mean_adj.transpose(-2, -1))
        return h_out, sparse_adj

    def encode_nodes(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes input `(B, L, N, F)` into spatiotemporal node embeddings `(B, N, hidden_dim)`,
        sparse learned adjacency `(B, N, N)`, and node Hawkes jump intensities `(B, N)`.
        """
        h_temp, jump_intensity = self.temporal_encoder(x, return_jump_intensity=True)
        prior_adj = self._compute_composite_prior_adjacency(x)
        h_st, sparse_adj = self._sparse_graph_attention(h_temp, prior_adj)
        return h_st, sparse_adj, jump_intensity

    @staticmethod
    def _compute_batch_fiedler_value(sparse_adj: torch.Tensor) -> torch.Tensor:
        """
        Computes the Fiedler algebraic connectivity `lambda_2(L_sparse)` per batch element
        from the learned symmetric sparse adjacency `(B, N, N)`.
        """
        bsz, n_assets, _ = sparse_adj.shape
        if n_assets <= 1:
            return torch.zeros(bsz, device=sparse_adj.device, dtype=sparse_adj.dtype)
        with torch.no_grad():
            eye = torch.eye(
                n_assets, device=sparse_adj.device, dtype=sparse_adj.dtype
            ).unsqueeze(0)
            w_off = sparse_adj * (1.0 - eye)
            deg = w_off.sum(dim=-1).clamp(min=1e-8)
            inv_sqrt_deg = deg.rsqrt()
            norm_w = w_off * inv_sqrt_deg.unsqueeze(-1) * inv_sqrt_deg.unsqueeze(-2)
            lap = eye - 0.5 * (norm_w + norm_w.transpose(-2, -1))
            eigvals = torch.linalg.eigvalsh(lap)
            return eigvals[:, 1].clamp(min=0.0, max=2.0)

    def predict_returns_and_cov(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts expected return vector `mu` `(B, N)`, strictly PSD covariance matrix `cov`
        `(B, N, N)`, and learned sparse adjacency `sparse_adj` `(B, N, N)`.
        """
        h_st, sparse_adj, jump_intensity = self.encode_nodes(x)
        mu = self.mu_head(h_st).squeeze(-1)
        fiedler_val = self._compute_batch_fiedler_value(sparse_adj)
        hawkes_proxy = torch.tanh(jump_intensity.mean(dim=-1)).clamp(0.0, 0.95)
        cov = self.cov_head(
            h_st,
            fiedler_value=fiedler_val,
            hawkes_radius=hawkes_proxy,
        )
        return mu, cov, sparse_adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard covariance prediction pass returning `(B, N, N)` PSD covariance matrix
        for seamless compatibility with `DeepCovarianceEstimator`.
        """
        _, cov, _ = self.predict_returns_and_cov(x)
        return cov
