import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    Compresses the historical time-series of each asset into a single
    'regime state' vector.
    """

    def __init__(self, n_features: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch * N_assets, Lookback, N_features)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]

        return self.layer_norm(last_step)


class CrossAssetAttention(nn.Module):
    """
    Allows every asset to 'look' at the regime states of every other asset
    to determine non-linear correlation and tail-risk clustering.
    """

    def __init__(
        self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, asset_embeddings: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(
            query=asset_embeddings, key=asset_embeddings, value=asset_embeddings
        )
        # Residual connection
        return self.layer_norm(asset_embeddings + attn_out)


class CholeskyCovarianceLayer(nn.Module):
    """
    Generates a dynamically sized, mathematically guaranteed Positive Definite
    covariance matrix using the Cholesky factorization: \Sigma = L L^T + \epsilon I.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, cross_asset_features: torch.Tensor) -> torch.Tensor:
        batch_size, n_assets, _ = cross_asset_features.size()

        q = self.proj_q(cross_asset_features)
        k = self.proj_k(cross_asset_features)

        raw_matrix = torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)

        L_strict = torch.tril(raw_matrix, diagonal=-1)
        D_raw = torch.diagonal(raw_matrix, dim1=-2, dim2=-1)

        D_positive = torch.nn.functional.softplus(D_raw) + self.eps

        D_matrix = torch.diag_embed(D_positive)
        L = L_strict + D_matrix

        covariance_matrix = torch.bmm(L, L.transpose(1, 2))

        return covariance_matrix


class CrossAttentionCholeskyNet(nn.Module):
    """
    End-to-End Deep Learning Covariance Estimator.
    Guarantees PSD outputs suitable for cvxpy Markowitz optimization.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.temporal_encoder = TemporalEncoder(n_features, hidden_dim, dropout)
        self.cross_attention = CrossAssetAttention(hidden_dim, num_heads, dropout)
        self.cholesky_output = CholeskyCovarianceLayer(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Lookback, N_Assets, N_Features)
        Returns:
            covariance_matrix: Tensor of shape (Batch, N_Assets, N_Assets)
        """

        batch_size, lookback, n_assets, n_features = x.size()
        x_reshaped = x.view(batch_size * n_assets, lookback, n_features)
        temporal_state = self.temporal_encoder(x_reshaped)
        cross_sectional_state = temporal_state.view(batch_size, n_assets, -1)
        attended_state = self.cross_attention(cross_sectional_state)
        cov_matrix = self.cholesky_output(attended_state)

        return cov_matrix
