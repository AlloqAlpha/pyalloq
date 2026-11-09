from __future__ import annotations

import torch
from torch import nn


class TorchLogSignatureExtractor(nn.Module):
    """
    Pure-PyTorch Windowed Depth-2 Log-Signature Extractor in the Lyndon Basis.

    Partitions an input sequence X of shape (B_flat, L, C) into S local intervals of
    length W (with time augmentation channel t in [0, 1]) and computes for each interval:
    - Level-1 path increments Delta X^a (dimension C + 1)
    - Level-2 Lévy areas / Lie brackets [e_a, e_b] = 1/2 int (X^a dX^b - X^b dX^a)
      for 0 <= a < b <= C (dimension (C + 1) * C // 2).
    """

    pair_i: torch.Tensor
    pair_j: torch.Tensor

    def __init__(self, n_channels: int, window_size: int = 5) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.window_size = int(max(2, window_size))
        aug_c = self.n_channels + 1  # +1 for normalized time channel
        self.aug_channels = aug_c
        self.logsig_dim = aug_c + (aug_c * (aug_c - 1)) // 2

        # Precompute upper-triangular pair indices (a < b) for vectorized Lévy area
        triu_i, triu_j = torch.triu_indices(aug_c, aug_c, offset=1)
        self.register_buffer("pair_i", triu_i, persistent=False)
        self.register_buffer("pair_j", triu_j, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (B_flat, L, C).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - `logsig_tokens`: shape (B_flat, S, logsig_dim)
            - `interval_shocks`: terminal step increments of shape (B_flat, S, C)
        """
        B_flat, L, C = x.shape
        device = x.device
        dtype = x.dtype

        W = min(self.window_size, L)
        stride = max(1, W - 1)
        # Ensure at least one window
        starts = list(range(0, max(1, L - W + 1), stride))
        if starts[-1] + W < L:
            starts.append(L - W)

        t_grid = (
            torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
            .view(1, W, 1)
            .expand(B_flat, W, 1)
        )

        tokens_list: list[torch.Tensor] = []
        shocks_list: list[torch.Tensor] = []
        pair_i = self.pair_i
        pair_j = self.pair_j

        for s_idx in starts:
            seg = x[:, s_idx : s_idx + W, :]  # (B_flat, W, C)
            aug = torch.cat([t_grid, seg], dim=-1)  # (B_flat, W, C + 1)
            rel = aug - aug[:, 0:1, :]
            dX = rel[:, 1:, :] - rel[:, :-1, :]  # (B_flat, W - 1, C + 1)
            X_mid = 0.5 * (rel[:, 1:, :] + rel[:, :-1, :])

            level_1 = rel[:, -1, :]  # (B_flat, C + 1)
            # Vectorized Lévy area for all pairs (a < b): shape (B_flat, n_pairs)
            area_ab = 0.5 * torch.sum(
                X_mid[:, :, pair_i] * dX[:, :, pair_j]
                - X_mid[:, :, pair_j] * dX[:, :, pair_i],
                dim=1,
            )
            tokens_list.append(torch.cat([level_1, area_ab], dim=-1))
            shocks_list.append(seg[:, -1, :] - seg[:, 0, :])

        logsig_tokens = torch.stack(tokens_list, dim=1)  # (B_flat, S, logsig_dim)
        interval_shocks = torch.stack(shocks_list, dim=1)  # (B_flat, S, C)
        return logsig_tokens, interval_shocks


class NeuralJumpRDE(nn.Module):
    """
    Continuous-Time Neural Rough Differential Equation with Hawkes Jump Discontinuities
    (`NeuralJumpRDE`, Node-Level Point-in-Time Volatility Backbone).

    Synthesizes:
    1. **Log-ODE Continuous Flow**: Advances the hidden state h_{i, k} in R^d over local
       intervals via a neural vector field f_theta contracting with the Lyndon-basis
       log-signature token LogSig(X_{i, [t_k, t_{k+1}]}).
    2. **Discontinuous Hawkes Jump Encoder**: Tracks a self-exciting jump intensity lambda_{i, k}
       driven by standardized path shocks; when jump probability p_{jump} = 1 - exp(-lambda_{i, k})
       spikes, applies an instantaneous state transformation J_psi(h, Delta X) to recalibrate
       volatility expectations without smoothing lag.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        window_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.logsig_extractor = TorchLogSignatureExtractor(
            n_channels=n_features,
            window_size=window_size,
        )
        logsig_dim = self.logsig_extractor.logsig_dim

        # Initial hidden state projection from first observation
        self.init_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Log-ODE continuous vector field f_theta: maps (h, logsig) -> dh
        self.log_ode_field = nn.Sequential(
            nn.Linear(hidden_dim + logsig_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Self-exciting Hawkes jump intensity gate & discontinuous jump encoder J_psi
        self.hawkes_decay = nn.Parameter(torch.tensor(0.65))
        self.hawkes_excitation = nn.Linear(n_features + hidden_dim, 1)
        self.jump_encoder = nn.Sequential(
            nn.Linear(hidden_dim + n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_jump_intensity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(B, L, N, F)` or `(B_flat, L, F)`.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Node embeddings of shape `(B, N, hidden_dim)` (or `(B_flat, hidden_dim)`),
            plus optional terminal Hawkes jump intensity of shape `(B, N)`.
        """
        is_4d = x.dim() == 4
        if is_4d:
            B, L, N, F = x.shape
            x_flat = x.permute(0, 2, 1, 3).reshape(B * N, L, F)
        else:
            B_flat, L, F = x.shape
            x_flat = x

        logsig_tokens, shocks = self.logsig_extractor(x_flat)
        _, S, _ = logsig_tokens.shape

        h = self.init_proj(x_flat[:, 0, :])
        lam = torch.full((x_flat.shape[0], 1), 0.05, device=x.device, dtype=x.dtype)
        decay = torch.sigmoid(self.hawkes_decay)

        for k in range(S):
            lsig_k = logsig_tokens[:, k, :]
            shock_k = shocks[:, k, :]

            # 1. Continuous Log-ODE step
            dh = self.log_ode_field(torch.cat([h, lsig_k], dim=-1))
            h_cont = h + dh

            # 2. Self-exciting Hawkes jump intensity update & discontinuous Jump Encoder
            exc = torch.nn.functional.softplus(
                self.hawkes_excitation(torch.cat([h_cont, torch.abs(shock_k)], dim=-1))
            )
            lam = decay * lam + (1.0 - decay) * exc
            p_jump = 1.0 - torch.exp(-lam.clamp(min=0.0, max=10.0))

            jump_delta = self.jump_encoder(torch.cat([h_cont, shock_k], dim=-1))
            h = self.out_norm(h_cont + p_jump * jump_delta)

        if is_4d:
            h_out = h.view(B, N, self.hidden_dim)
            if return_jump_intensity:
                return h_out, lam.view(B, N)
            return h_out

        if return_jump_intensity:
            return h, lam.squeeze(-1)
        return h
