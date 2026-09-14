import torch
import torch.nn as nn


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
