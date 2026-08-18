import math
import torch
import torch.nn as nn
from typing import Optional


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU).
    Splits the input in half; passes one half through a Sigmoid activation (the Gate),
    and multiplies it by the other half.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # We double the hidden size so we can split it in half later
        self.fc = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        # Split the tensor along the last dimension
        out, gate = x.chunk(2, dim=-1)
        # Apply the gate
        return out * torch.sigmoid(gate)


class GRN(nn.Module):
    """
    Gated Residual Network (GRN).
    The core processing engine of the TFT. It dynamically suppresses noise.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Optional context vector (used later for static features like Asset Sector)
        self.context_fc = (
            nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        )

        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.glu = GLU(hidden_size, hidden_size)

        # If input size doesn't match hidden size, we need a projection layer for the residual connection
        self.skip_layer = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = self.skip_layer(x)

        x = self.fc1(x)
        if self.context_fc is not None and context is not None:
            x = x + self.context_fc(context)

        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)

        # The Residual Connection: Add the original input back to the processed input
        return self.layer_norm(residual + x)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN).
    Dynamically weighs the importance of each financial feature at every single time step.
    """

    def __init__(
        self, num_features: int, hidden_size: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        # 1. A single GRN that looks at ALL features to generate the "Selection Weights"
        self.flattened_grn = GRN(
            input_size=num_features * hidden_size,
            hidden_size=num_features,
            dropout=dropout,
        )

        # 2. Independent GRNs for each individual feature to process them
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
            x: Tensor of shape (Batch, Seq_Len, Num_Features, Hidden_Size)
               Note: Raw features must be embedded to Hidden_Size before passing here!
        Returns:
            processed_features: The noise-filtered, weighted combination of features.
            selection_weights: The attention weights showing WHICH features mattered.
        """
        batch_size, seq_len, num_features, hidden_size = x.size()

        # Flatten the features to calculate global importance weights
        # Shape: (Batch, Seq_Len, Num_Features * Hidden_Size)
        flattened = x.reshape(batch_size, seq_len, -1)

        # Calculate the weights and apply Softmax so they sum to 1.0
        # Shape: (Batch, Seq_Len, Num_Features)
        selection_weights = self.softmax(self.flattened_grn(flattened))

        # Process each feature individually through its own GRN
        processed_list = []
        for i in range(self.num_features):
            # Extract feature 'i' across all batches and timesteps -> Shape: (Batch, Seq_Len, Hidden_Size)
            feat_i = x[:, :, i, :]
            processed_feat = self.single_feature_grns[i](feat_i)
            processed_list.append(processed_feat)

        # Stack them back together -> Shape: (Batch, Seq_Len, Num_Features, Hidden_Size)
        processed_stack = torch.stack(processed_list, dim=2)

        # Multiply each feature by its calculated importance weight
        # We add unsqueeze(-1) to broadcast the weight across the Hidden_Size dimension
        weighted_features = processed_stack * selection_weights.unsqueeze(-1)

        # Sum across the feature dimension to create one unified, noise-filtered representation
        # Shape: (Batch, Seq_Len, Hidden_Size)
        output = torch.sum(weighted_features, dim=2)

        return output, selection_weights


class ContinuousFeatureEmbedder(nn.Module):
    """
    Expands 1D scalar features (like raw returns, RSI, or Volume) into
    high-dimensional vectors (Hidden_Size) so the VSN can process them.
    """

    def __init__(self, num_features: int, hidden_size: int) -> None:
        super().__init__()
        self.num_features = num_features

        # We create a separate Linear projection layer for EACH feature.
        # RSI needs to be embedded differently than Volume!
        self.embeddings = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(num_features)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Num_Features)
        Returns:
            Tensor of shape (Batch, Seq_Len, Num_Features, Hidden_Size)
        """
        batch_size, seq_len, num_features = x.size()
        embedded_features = []

        for i in range(self.num_features):
            # Extract single feature: (Batch, Seq_Len, 1)
            feat_i = x[:, :, i].unsqueeze(-1)
            # Pass through its dedicated embedding layer
            embedded_i = self.embeddings[i](feat_i)
            embedded_features.append(embedded_i)

        # Stack them back together along the feature dimension
        # Shape: (Batch, Seq_Len, Num_Features, Hidden_Size)
        return torch.stack(embedded_features, dim=2)


class LocalityEncoder(nn.Module):
    """
    Standard Transformers ignore the order of time (Permutation Invariant).
    This LSTM layer forces the network to respect strict chronology (e.g.,
    recognizing that a price drop yesterday matters more than one 30 days ago).
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()

        # We use a standard LSTM to encode the sequence
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # In TFT, the output of the LSTM is passed through a Gated Residual Network
        # combined with a LayerNorm to ensure gradients flow cleanly.
        self.post_lstm_gate = GLU(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Hidden_Size) - The output from VSN
        Returns:
            encoded_sequence: Tensor of shape (Batch, Seq_Len, Hidden_Size)
        """
        # Pass through the LSTM to capture temporal momentum
        lstm_out, _ = self.lstm(x)

        lstm_out = self.dropout(lstm_out)

        # Apply the Gated Linear Unit to suppress irrelevant temporal noise
        gated_out = self.post_lstm_gate(lstm_out)

        # Residual connection: Add the original VSN output (x) to the LSTM output
        # This solves the vanishing gradient problem in deep time-series networks
        return self.layer_norm(x + gated_out)


class InterpretableMultiHeadAttention(nn.Module):
    """
    TFT's custom attention mechanism.
    Averages the attention weights across all heads so the network's
    historical focus is 100% transparent and interpretable.
    """

    def __init__(
        self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q (Queries) and K (Keys) are split across heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)

        # V (Values) is SHARED across all heads (The TFT Interpretability trick)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # TFT Post-Attention Gating
        self.gate = GLU(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: The attention-processed tensor.
            attention_weights: The transparency matrix showing which days mattered.
        """
        batch_size, seq_len, hidden_size = x.size()

        # 1. Project Queries, Keys, and Values
        q = (
            self.q_proj(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Value is not split by head in the same way; it's shared, but reshaped for batched matmul
        v = (
            self.v_proj(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 2. Calculate Attention Scores (Q * K^T / sqrt(d))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3. Apply weights to Values
        context = torch.matmul(attn_weights, v)

        # 4. Concatenate heads back together
        context = (
            context.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, hidden_size)
        )
        out = self.out_proj(context)

        # 5. Apply the TFT Gate and Residual Connection
        gated_out = self.gate(out)
        final_out = self.layer_norm(x + gated_out)

        # 6. Average the attention weights across all heads for interpretability
        # Shape: (Batch, Seq_Len, Seq_Len)
        interpretable_weights = attn_weights.mean(dim=1)

        return final_out, interpretable_weights


class TFTDeepReturnsNet(nn.Module):
    """
    The Ultimate Temporal Fusion Transformer for Expected Return Estimation (\mu).
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

        # 1. Expand features
        self.embedder = ContinuousFeatureEmbedder(n_features, hidden_size)

        # 2. Filter the noise
        self.vsn = VariableSelectionNetwork(n_features, hidden_size, dropout)

        # 3. Capture short-term chronological momentum
        self.locality_encoder = LocalityEncoder(hidden_size, dropout)

        # 4. Capture long-term structural regimes
        self.attention = InterpretableMultiHeadAttention(
            hidden_size, num_heads, dropout
        )

        # 5. Output Layer (Linear mapping to predict expected return)
        # We use a 2-layer MLP to distill the final hidden state into a single return prediction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Lookback, N_Assets, N_Features)
        Returns:
            expected_returns: Tensor of shape (Batch, N_Assets)
        """
        batch_size, lookback, n_assets, n_features = x.size()

        # 1. Reshape to treat each asset as an independent time-series sample
        # Shape: (Batch * N_Assets, Lookback, N_Features)
        x_permuted = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_permuted.reshape(batch_size * n_assets, lookback, n_features)

        # 2. Pass through TFT Pipeline
        embedded = self.embedder(x_reshaped)
        vsn_out, _ = self.vsn(embedded)  # (We can log the VSN weights later!)
        encoded = self.locality_encoder(vsn_out)
        attn_out, _ = self.attention(
            encoded
        )  # (We can log the Attention weights later!)

        # 3. Extract the final time step's representation
        # Shape: (Batch * N_Assets, Hidden_Size)
        final_step = attn_out[:, -1, :]

        # 4. Predict Expected Return (\mu)
        # Shape: (Batch * N_Assets, 1)
        mu_pred = self.output_head(final_step)

        # 5. Reshape back to cross-sectional layout
        # Shape: (Batch, N_Assets)
        expected_returns = mu_pred.reshape(batch_size, n_assets)

        return expected_returns
