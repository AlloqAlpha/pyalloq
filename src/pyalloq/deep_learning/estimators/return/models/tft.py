import torch
import torch.nn as nn
import math


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU).
    Splits input in half:
        1. Passed one half through a Sigmoid activation (the Gate)
        2. Multiples the result with the other half
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        out, gate = x.chunk(2, dim=-1)

        return out * torch.sigmoid(gate)


class GRN(nn.Module):
    """
    Gated Residual Network (GRN).
    The core processing engine of TFT
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context_fc = (
            nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        )

        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.glu = GLU(hidden_size, hidden_size)

        self.skip_layer = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = self.skip_layer(x)

        x = self.fc1(x)
        if self.context_fc is not None and context is not None:
            x = x + self.context_fc(context)

        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)

        return self.layer_norm(residual + x)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN).
    Dynamically weights the importance of each financial features at every single time step.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        self.flattened_grn = GRN(
            input_size=num_features * hidden_size,
            hidden_size=num_features,
            dropout=dropout,
        )

        self.single_feature_grns = nn.ModuleList(
            [
                GRN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
                for _ in range(num_features)
            ]
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor shape (Batch Size, Seq Len, No. Features, Hidden Size)
        Returns:
            processed_features: The noise-filtered, weighted combination of features.
            selection_weights: The attention weights showing which features mattered.
        """
        batch_size, seq_len, num_features, hidden_size = x.size()
        flattened = x.view(batch_size, seq_len, -1)
        selection_weights = self.softmax(self.flattened_grn(flattened))

        processed_list = []
        for i in range(self.num_features):
            feat_i = x[:, :, i, :]
            processed_feat = self.single_feature_grns[i](feat_i)
            processed_list.append(processed_feat)

        processed_stack = torch.stack(processed_list, dim=1)
        weighted_features = processed_stack * selection_weights.unsqueeze(-1)

        output = torch.sum(weighted_features, dim=2)

        return (output, selection_weights)


class ContinuousFeatureEmbedder(nn.Module):
    """
    Expandas 1D Scalar features (raw returns, RSI, volumes) into high dimensional
    vectors (hidden_size) so VSN can process them
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
    ) -> None:
        self.num_features = num_features
        self.embeddings = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(num_features)]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (Batch, Seq Len, No. Features)
        Returns:
            Tensor of shape (Batch, Seq Len, No. Features, Hidden)
        """
        batch_size, seq_len, num_features = x.size()
        embedded_features = []

        for i in range(self.num_features):
            feat_i = x[:, :, i].unsqueeze(-1)
            embedded_i = self.embeddings[i](feat_i)
            embedded_features.append(embedded_i)

        return torch.stack(embedded_features, dim=2)


class LocalityEncoder(nn.Module):
    """ """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.post_lstm_gate = GLU(input_size=hidden_size, hidden_size=hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape: (Batch Size, Seq Len, Hidden Size). VSN Output
        Returns:
            encoded_sequence: Tensor of shape (Batch Size, Seq Len, Hidden Size)
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        gated_out = self.post_lstm_gate(lstm_out)

        return self.layer_norm(x + gated_out)


class MultiHeadAttention(nn.Module):
    """
    TFT's custom attention mechanism.
    Averages the attention weights across all heads to the network's
    historical focus is more transparent and interpretable.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)

        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.gate = GLU(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: The attention-processed tensor
            attention_weights: The transparency matrix showing which days mattered.
        """
        batch_size, seq_len, hidden_size = x.size()
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)
        out = self.out_proj(context)

        gated_out = self.gate(out)
        final_out = self.layer_norm(x + gated_out)

        interpretable_weights = attention_weights.mean(dim=1)

        return (final_out, interpretable_weights)


class TFTDeepReturnsNet(nn.Module):
    """
    Temporal Fusion Transformer for Expected Return Estimator.
    Ingests noisy market data, filters it, captures locality, and attends to history.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedder = ContinuousFeatureEmbedder(
            num_features=n_features, hidden_size=hidden_size
        )
        self.vsn = VariableSelectionNetwork(
            num_features=n_features, hidden_size=hidden_size, dropout=dropout
        )
        self.locality_encoder = LocalityEncoder(
            hidden_size=hidden_size, dropout=dropout
        )
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, lookback, n_assets, n_features = x.size()
        x_reshaped = x.view(batch_size * n_assets, lookback, n_features)

        embedded = self.embedder(x_reshaped)
        vsn_out, _ = self.vsn(embedded)
        encoded = self.locality_encoder(vsn_out)
        attention_out, _ = self.attention(encoded)

        final_step = attention_out[:, -1, :]
        mu_pred = self.output_head(final_step)

        expected_returns = mu_pred.view(batch_size, n_assets)

        return expected_returns
