from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pyalloq_core.data import MarketData
from torch import nn


class RelationalGraphPriorModule(nn.Module):
    """
    Relational Information Network & GNN Edge-Refinement Prior Module
    (`RelationalGraphPriorModule`, Chen, Fan & Chin, Sept 12, 2026,
    *Journal of Financial Data Science*, "Portfolio Optimization with Covariance from
    News-Derived Information Networks").

    Instead of averaging document-level text embeddings (which dilutes firm-specific
    relational signals), takes a structured entity-relation / co-mention / supply-chain
    or sector-linkage adjacency matrix `A_rel in R^{N x N}` (or extracts it from
    `MarketData.features["relational_graph"]` / `MarketData.cross_sectional`), refines
    edge weights against rolling market return data via a GNN edge-gating layer combined
    with the regularized partial-correlation Laplacian, and emits a sparse symmetric
    `attn_bias` tensor `(B, N, N)` for `SetTransformerPolicyNet` and
    `SetTransformerCovarianceNet`.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        keep_ratio: float = 0.40,
        relational_blend: float = 0.60,
        ridge_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = max(4, int(hidden_dim))
        self.keep_ratio = float(np.clip(keep_ratio, 0.05, 1.0))
        self.relational_blend = float(np.clip(relational_blend, 0.0, 1.0))
        self.ridge_eps = float(max(1e-6, ridge_eps))

        # GNN node feature projection from rolling return statistics (mean, vol, downside vol, autocorr)
        self.node_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        # Edge refinement bilinear / MLP gate combining [h_i, h_j, a_rel_{i,j}, partial_corr_{i,j}]
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def extract_relational_prior_from_data(
        data: MarketData,
        prior_adjacency: pd.DataFrame | np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        Resolves an `(N, N)` relational adjacency matrix from:
          1. Explicit `prior_adjacency` argument,
          2. `data.features["relational_graph"]` or `data.features["news_graph"]`,
          3. `data.cross_sectional["sector"]` co-sector linkage matrix.
        Returns `None` if no relational prior is present (preserving zero-graph fallback).
        """
        assets = list(data.prices.columns)
        n_assets = len(assets)
        if prior_adjacency is not None:
            if isinstance(prior_adjacency, pd.DataFrame):
                return prior_adjacency.reindex(
                    index=assets, columns=assets, fill_value=0.0
                ).to_numpy(dtype=np.float32)
            arr = np.asarray(prior_adjacency, dtype=np.float32)
            if arr.shape == (n_assets, n_assets):
                return arr

        if data.features:
            for key in ("relational_graph", "news_graph", "entity_graph"):
                if key in data.features:
                    df_g = data.features[key]
                    if df_g.shape == (n_assets, n_assets):
                        return df_g.to_numpy(dtype=np.float32)
                    if (
                        set(assets).issubset(set(df_g.columns))
                        and len(df_g) >= n_assets
                    ):
                        sub = df_g.iloc[-n_assets:].reindex(
                            columns=assets, fill_value=0.0
                        )
                        return sub.to_numpy(dtype=np.float32)

        if (
            data.cross_sectional is not None
            and "sector" in data.cross_sectional.columns
        ):
            sec_ser = data.cross_sectional["sector"].reindex(assets)
            if sec_ser.notna().any():
                sec_vals = sec_ser.astype(str).to_numpy()
                same_sec = (sec_vals[:, None] == sec_vals[None, :]).astype(np.float32)
                np.fill_diagonal(same_sec, 1.0)
                return same_sec

        return None

    def forward(
        self,
        x_returns: torch.Tensor,
        relational_adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Refines relational adjacency `relational_adj` `(B, N, N)` or `(N, N)` against
        rolling asset returns `x_returns` `(B, L, N)` or `(B, L, N, F)` and outputs a
        sparse symmetric `attn_bias` tensor `(B, N, N)`.
        """
        if x_returns.dim() == 4:
            r = x_returns[:, :, :, 0]
        else:
            r = x_returns

        bsz, seq_len, n_assets = r.shape
        device = r.device
        dtype = r.dtype
        eye = torch.eye(n_assets, device=device, dtype=dtype).unsqueeze(0)

        r_clean = torch.nan_to_num(r, nan=0.0)
        r_mean = r_clean.mean(dim=1)
        r_std = r_clean.std(dim=1, unbiased=False).clamp(min=1e-6)
        r_down = torch.clamp(-r_clean, min=0.0).mean(dim=1)
        if seq_len >= 3:
            r_prev = r_clean[:, :-1, :] - r_mean.unsqueeze(1)
            r_next = r_clean[:, 1:, :] - r_mean.unsqueeze(1)
            ac1 = (r_prev * r_next).mean(dim=1) / (r_std**2 + 1e-6)
        else:
            ac1 = torch.zeros_like(r_mean)

        node_feats = torch.stack([r_mean, r_std, r_down, ac1], dim=-1)  # (B, N, 4)
        h_nodes = self.node_encoder(node_feats)  # (B, N, H)

        # Compute regularized partial-correlation Laplacian from return window
        r_cent = r_clean - r_mean.unsqueeze(1)
        cov = torch.bmm(r_cent.transpose(1, 2), r_cent) / float(max(1, seq_len - 1))
        trace_scale = (
            torch.diagonal(cov, dim1=-2, dim2=-1)
            .mean(dim=-1, keepdim=True)
            .unsqueeze(-1)
            .clamp(min=1e-6)
        )
        reg_cov = (1.0 - 0.15) * cov + 0.15 * trace_scale * eye + self.ridge_eps * eye
        prec = torch.linalg.inv(reg_cov)
        prec_diag = torch.sqrt(torch.diagonal(prec, dim1=-2, dim2=-1).clamp(min=1e-8))
        partial_corr = -prec / (
            prec_diag.unsqueeze(-1) * prec_diag.unsqueeze(-2) + 1e-8
        )
        partial_corr = torch.relu(partial_corr) * (1.0 - eye) + eye

        if relational_adj is not None:
            rel_t = torch.nan_to_num(
                relational_adj.to(device=device, dtype=dtype), nan=0.0
            )
            if rel_t.dim() == 2:
                rel_t = rel_t.unsqueeze(0).expand(bsz, -1, -1)
        else:
            rel_t = partial_corr

        # Pairwise GNN edge gating
        h_i = h_nodes.unsqueeze(2).expand(bsz, n_assets, n_assets, self.hidden_dim)
        h_j = h_nodes.unsqueeze(1).expand(bsz, n_assets, n_assets, self.hidden_dim)
        edge_in = torch.cat(
            [h_i, h_j, rel_t.unsqueeze(-1), partial_corr.unsqueeze(-1)], dim=-1
        )
        gate_w = self.edge_gate(edge_in).squeeze(-1)
        gate_sym = 0.5 * (gate_w + gate_w.transpose(1, 2))

        blended_adj = gate_sym * (
            self.relational_blend * rel_t + (1.0 - self.relational_blend) * partial_corr
        )
        blended_adj = 0.5 * (blended_adj + blended_adj.transpose(1, 2))

        # Sparse top-k pruning per node (preserving self-loops)
        k_keep = max(1, min(n_assets, round(self.keep_ratio * n_assets)))
        eye_bool = eye.to(torch.bool)
        boosted = torch.where(eye_bool, blended_adj + 1e4, blended_adj)
        _, topk_idx = torch.topk(boosted, k=k_keep, dim=-1)
        keep_mask = torch.zeros_like(blended_adj, dtype=torch.bool)
        keep_mask.scatter_(-1, topk_idx, True)
        keep_mask = keep_mask | keep_mask.transpose(1, 2) | eye_bool

        sparse_bias = torch.where(keep_mask, blended_adj, torch.zeros_like(blended_adj))
        return sparse_bias

    def compute_attn_bias(
        self,
        data: MarketData,
        x_batch: torch.Tensor,
        prior_adjacency: pd.DataFrame | np.ndarray | None = None,
    ) -> torch.Tensor | None:
        """
        Resolves the relational prior from `data` or `prior_adjacency` and runs GNN edge
        refinement against `x_batch` to produce `attn_bias` `(B, N, N)`.
        """
        raw_prior = self.extract_relational_prior_from_data(data, prior_adjacency)
        rel_tensor = (
            torch.tensor(raw_prior, dtype=x_batch.dtype, device=x_batch.device)
            if raw_prior is not None
            else None
        )
        with torch.no_grad():
            return self.forward(x_batch, relational_adj=rel_tensor)
