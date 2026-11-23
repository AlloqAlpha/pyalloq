import torch
from torch import nn

from pyalloq.dl.estimators.e2e.models.set_policy import CappedSimplexProjection
from pyalloq.dl.layers.fragility_kkt import BasePolicyNetwork
from pyalloq.dl.layers.set_transformer import ISAB, PMA


class BoundedSoftmax(nn.Module):
    """
    Differentiable Bounded Softmax
    Ensures weight sum to 1.0, but strictly caps max exposure per asset.
    Uses an iterative 'water-filling' algorithmm to maintain the computational graph.
    """

    def __init__(
        self, max_weight: float = 0.25, iterations: int = 3, eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.max_weight = max_weight
        self.iterations = iterations
        self.eps = eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: logits shape: (Batch Size, N Assets)
        """
        w = torch.softmax(logits, dim=-1)

        for _ in range(self.iterations):
            overshoot = torch.relu(w - self.max_weight)
            w = w - overshoot
            excess_mass = overshoot.sum(dim=-1, keepdim=True)
            under_cap_mask = (w < self.max_weight).float()
            eligable_weights = w * under_cap_mask
            sum_eligable = eligable_weights.sum(dim=-1, keepdim=True) + self.eps

            w = w + (excess_mass * (eligable_weights / sum_eligable))

        return w


class AssetTemporalEncoder(nn.Module):
    """
    Compresses the historical time-series of each asset into a single
    'momentum and volatility' state vector.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape: (Batch * N Assets, Lookback, N_features)
        """
        lstm_out, _ = self.lstm(x)

        last_step = lstm_out[:, -1, :]
        last_step = self.dropout(last_step)

        return self.layer_norm(last_step)


class PortfolioSpatialAttention(nn.Module):
    """
    Allows every asset to 'look' at the state vectors of every other asset
    to implicitly to learn cross-asset correlations, replacing the Covariance Matrix.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, asset_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            asset_embeddings: shape: (Batch Size, N Assets, Hidden Dim)
        """
        attention_out, _ = self.attention(
            query=asset_embeddings,
            key=asset_embeddings,
            value=asset_embeddings,
        )

        return self.layer_norm(asset_embeddings + attention_out)


class AssetScoringLayer(nn.Module):
    """
    Projects the high-dimensional, context-aware portfolio embeddings down
    into a single unbounded raw allocation score per asset.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, context_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_embeddings: shape: (Batch Size, N Assets, hidden dim)
        Output:
            Score: shape: (Batch Size, N Assets)
        """

        scores = self.mlp(context_embeddings)

        return scores.squeeze(-1)


class ConstrainedSpatioTemporalAttentionNet(nn.Module):
    """
    (CSTAN) Portfolio Allocator.
    Directly maps raw market tensors to mathematically bounded optimal weights,
    bypassing classical return and covariance estimators entirely.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_asset_weight: float = 0.25,
        waterfill_iterations: int = 3,
    ) -> None:
        super().__init__()

        self.temporal_encoder = AssetTemporalEncoder(n_features, hidden_dim, dropout)
        self.spatial_attention = PortfolioSpatialAttention(
            hidden_dim, num_heads, dropout
        )
        self.scoring_layer = AssetScoringLayer(hidden_dim, dropout)
        self.bounded_softmax = BoundedSoftmax(max_asset_weight, waterfill_iterations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch Size, Lookback, N Assets, N Features)
        Returns:
            weights: Tensor of shape (Batch, N Assets)
        """
        batch_size, lookback, n_assets, n_features = x.size()

        x_permuted = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_permuted.view(batch_size * n_assets, lookback, n_features)

        temporal_state = self.temporal_encoder(x_reshaped)

        cross_sectional_state = temporal_state.view(batch_size, n_assets, -1)

        attended_state = self.spatial_attention(cross_sectional_state)

        raw_logits = self.scoring_layer(attended_state)

        optimal_weights = self.bounded_softmax(raw_logits)

        return optimal_weights


class ConvTransformerAllocator(BasePolicyNetwork):
    r"""
    Size- and Permutation-Equivariant Conv1D + Set Transformer Portfolio Policy
    (`ConvTransformerAllocator`).

    Replaces the legacy ``encoded.view(B, -1) -> nn.Linear(d_model * n_assets, 32)``
    flattening regime head with ``PMA_k(Z)`` (Pooling by Multihead Attention with
    Scalable-Softmax), making both ``regime_head`` (permutation-invariant) and
    ``allocation_head`` (1D permutation-equivariant) completely independent of asset
    count ``N`` and ticker column ordering.
    """

    def __init__(
        self,
        n_features: int = 1,
        d_model: int = 64,
        num_heads: int = 4,
        num_inds: int = 16,
        num_pma_seeds: int = 1,
        num_regimes: int = 3,
        dropout: float = 0.1,
        max_asset_weight: float = 0.25,
        n_assets: int | None = None,
    ) -> None:
        super().__init__()
        # `n_assets` is accepted only for backward signature compatibility and never hardcoded
        self.n_features = int(max(1, n_features))
        self.d_model = int(d_model)
        self.num_pma_seeds = int(max(1, num_pma_seeds))
        self.num_regimes = int(max(2, num_regimes))

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.n_features, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal_norm = nn.LayerNorm(self.d_model)

        self.cross_asset_isab = ISAB(
            embed_dim=self.d_model,
            num_heads=num_heads,
            num_inds=num_inds,
            dropout=dropout,
            use_ssmax=True,
        )
        self.pma_pool = PMA(
            embed_dim=self.d_model,
            num_heads=num_heads,
            num_seeds=self.num_pma_seeds,
            dropout=dropout,
            use_ssmax=True,
        )
        # Size-invariant regime head operating on PMA_k(Z) instead of d_model * n_assets
        self.regime_head = nn.Sequential(
            nn.Linear(self.d_model * self.num_pma_seeds, 32),
            nn.GELU(),
            nn.Linear(32, self.num_regimes),
        )
        self.weight_proj = nn.Linear(2, self.d_model)
        self.allocation_head = nn.Sequential(
            nn.Linear(self.d_model + self.num_regimes, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 1),
        )
        self.capped_simplex = CappedSimplexProjection(max_weight=max_asset_weight)
        self.last_regime_probs_: torch.Tensor | None = None

    def predict_regime_probs(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes strictly permutation-invariant and size-invariant regime probabilities
        ``(B, num_regimes)`` via ``PMA_k(Z)``.
        """
        _, regime_probs = self._encode_assets_and_regime(x=x, w_current=None, mask=mask)
        return regime_probs

    def _encode_assets_and_regime(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        B, L, N, F = x.shape
        device = x.device
        dtype = x.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            if m_bool.dim() == 1:
                m_bool = m_bool.unsqueeze(0).expand(B, N)
            m_f = m_bool.to(dtype=dtype)
            x_clean = torch.where(
                m_bool.view(B, 1, N, 1),
                torch.nan_to_num(x, nan=0.0),
                torch.zeros_like(x),
            )
            n_active = m_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            m_bool = None
            m_f = torch.ones((B, N), device=device, dtype=dtype)
            x_clean = torch.nan_to_num(x, nan=0.0)
            n_active = torch.full(
                (B, 1), max(1.0, float(N)), device=device, dtype=dtype
            )

        if F < self.n_features:
            pad = torch.zeros(
                (B, L, N, self.n_features - F), device=device, dtype=dtype
            )
            x_clean = torch.cat([x_clean, pad], dim=-1)
        elif F > self.n_features:
            x_clean = x_clean[..., : self.n_features]

        # Per-asset Conv1D over (B * N, F, L)
        x_flat = (
            x_clean.permute(0, 2, 3, 1).contiguous().view(B * N, self.n_features, L)
        )
        conv_out = self.temporal_conv(x_flat)[:, :, -1]
        h_temp = self.temporal_norm(conv_out).view(B, N, self.d_model)

        if w_current is not None:
            w_t = torch.nan_to_num(w_current.to(device=device, dtype=dtype), nan=0.0)
            if w_t.dim() == 1:
                w_t = w_t.unsqueeze(0).expand(B, N)
            w_t = w_t * m_f
            w_feat = torch.stack([w_t, w_t * n_active], dim=-1)
            h_temp = h_temp + self.weight_proj(w_feat)

        h_temp = h_temp * m_f.unsqueeze(-1)
        z_assets = self.cross_asset_isab(h_temp, mask=m_bool, return_inducing=False)
        assert isinstance(z_assets, torch.Tensor)

        # PMA_k pooled global summary -> regime probabilities (strictly permutation & size invariant)
        pma_summary = self.pma_pool(z_assets, mask=m_bool).reshape(
            B, self.d_model * self.num_pma_seeds
        )
        regime_probs = torch.softmax(self.regime_head(pma_summary), dim=-1)
        self.last_regime_probs_ = regime_probs
        return z_assets, regime_probs

    def forward(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_assets, regime_probs = self._encode_assets_and_regime(
            x=x, w_current=w_current, mask=mask
        )
        B, N, _ = z_assets.shape
        reg_exp = regime_probs.unsqueeze(1).expand(B, N, self.num_regimes)
        logits = self.allocation_head(torch.cat([z_assets, reg_exp], dim=-1)).squeeze(
            -1
        )
        return self.capped_simplex(logits, mask=mask)
