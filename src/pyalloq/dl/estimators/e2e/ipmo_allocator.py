from typing import Any

import numpy as np
import pandas as pd
import torch
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import MultiPeriodOptimizationResult
from torch import nn

from pyalloq.dl.layers.mdfp import MDFPOptimizationLayer


class MultiHorizonPredictorNet(nn.Module):
    """
    Spatio-Temporal Neural Network predicting H-step expected returns mu_{t+1:t+H}
    from a lookback window of asset features X in R^{B x L x N x F}.
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int = 1,
        horizon: int = 5,
        d_model: int = 32,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.n_features = n_features
        self.horizon = horizon
        self.input_proj = nn.Linear(n_features, d_model)
        self.temporal_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.cross_asset_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.horizon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, horizon),
            nn.Tanh(),
        )
        # Scale predicted daily returns into realistic range [-5%, +5%]
        self.output_scale = nn.Parameter(torch.tensor(0.02))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (B, L, N, F).

        Returns
        -------
        torch.Tensor
            Predicted multi-horizon returns mu_seq of shape (B, H, N).
        """
        B, L, N, F = x.shape
        # Fold assets into batch for temporal encoding: (B * N, L, F)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, L, F)
        h_in = self.input_proj(x_flat)
        _, h_last = self.temporal_gru(h_in)  # (1, B * N, d_model)
        asset_emb = h_last.squeeze(0).reshape(B, N, -1)  # (B, N, d_model)

        # Cross-asset self-attention
        attn_out, _ = self.cross_asset_attn(asset_emb, asset_emb, asset_emb)
        emb = asset_emb + attn_out  # (B, N, d_model)

        # Predict H horizons per asset -> (B, N, H) -> permute to (B, H, N)
        mu_pred = self.horizon_head(emb).permute(0, 2, 1) * self.output_scale
        return mu_pred


class IPMOModule(nn.Module):
    """
    Integrated Prediction and Multi-Period Optimization (IPMO) Module combining a
    multi-horizon return predictor with a differentiable MDFPOptimizationLayer.
    When called with `return_trajectory=False` (default for standard trainers),
    returns the immediate trade execution weights z*_{t,0} of shape (B, N).
    """

    def __init__(
        self,
        predictor: nn.Module,
        mdfp_layer: MDFPOptimizationLayer,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.mdfp_layer = mdfp_layer

    def forward(
        self,
        x: torch.Tensor,
        cov_seq: torch.Tensor | None = None,
        w_current: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mu_seq = self.predictor(x)
        z_traj = self.mdfp_layer(mu_seq, cov_seq=cov_seq, w_current=w_current)
        if return_trajectory:
            return z_traj, mu_seq
        return z_traj[:, 0, :]


class IPMOAllocator(BaseAllocator):
    """
    Integrated Prediction and Multi-Period Optimization (IPMO) Allocator (Pillar 2).

    Maps raw feature tensors X_t -> Multi-Horizon Predictions mu_{t+1:t+H} ->
    Differentiable MDFPOptimizationLayer -> Optimal H-step Weight Trajectory Z*_t,
    and executes the first-stage portfolio weights w*_t = Z*_{t,0}.
    """

    def __init__(
        self,
        predictor: nn.Module,
        mdfp_layer: MDFPOptimizationLayer | None = None,
        lookback_window: int = 60,
        device: str = "cpu",
        tickers: list[str] | None = None,
    ) -> None:
        super().__init__(tickers or [])
        self.device = device
        self.predictor = predictor.to(self.device)
        self.mdfp_layer = (mdfp_layer or MDFPOptimizationLayer()).to(self.device)
        self.lookback_window = lookback_window

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        **kwargs: Any,
    ) -> MultiPeriodOptimizationResult:
        if not self.tickers and data.assets:
            self.tickers = list(data.assets)
        tickers = list(data.prices.columns)
        n_assets = len(tickers)
        H = self.mdfp_layer.horizon

        returns_df = data.prices.pct_change().fillna(0.0).iloc[-self.lookback_window :]
        base_tensor = torch.tensor(returns_df.values, dtype=torch.float32).unsqueeze(-1)
        tensor_list = [base_tensor]

        if data.features is not None:
            for _, df_feat in data.features.items():
                feat_window = df_feat.iloc[-self.lookback_window :]
                feat_tensor = torch.tensor(
                    feat_window.values, dtype=torch.float32
                ).unsqueeze(-1)
                tensor_list.append(feat_tensor)

        x_tensor = torch.cat(tensor_list, dim=-1).unsqueeze(0).to(self.device)

        # Build covariance tensor (1, H, N, N) from cov_matrix or sample covariance
        if cov_matrix is not None:
            cov_np = (
                cov_matrix.reindex(index=tickers, columns=tickers).fillna(0.0).values
            )
        else:
            cov_np = returns_df.cov().fillna(0.0).values + np.eye(n_assets) * 1e-4

        cov_tensor = (
            torch.tensor(cov_np, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, H, n_assets, n_assets)
        )

        if current_weights is not None:
            w0_np = (
                current_weights.reindex(tickers)
                .fillna(1.0 / n_assets)
                .to_numpy(dtype=np.float32)
            )
            w0_sum = float(w0_np.sum())
            if w0_sum > 0:
                w0_np = w0_np / w0_sum
        else:
            w0_np = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)

        w_curr_tensor = torch.tensor(
            w0_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.predictor.eval()
        self.mdfp_layer.eval()
        with torch.no_grad():
            mu_seq = self.predictor(x_tensor)
            z_traj = self.mdfp_layer(
                mu_seq=mu_seq,
                cov_seq=cov_tensor,
                w_current=w_curr_tensor,
            )

        z_np = z_traj.squeeze(0).cpu().numpy()  # (H, N)
        stage_index = pd.Index([f"t+{k}" for k in range(H)], name="stage")
        weights_traj_df = pd.DataFrame(z_np, index=stage_index, columns=tickers)

        trades_np = np.zeros_like(z_np)
        trades_np[0] = z_np[0] - w0_np
        for k in range(1, H):
            trades_np[k] = z_np[k] - z_np[k - 1]
        trades_traj_df = pd.DataFrame(trades_np, index=stage_index, columns=tickers)

        w_first = pd.Series(z_np[0], index=tickers, name="weights")
        return MultiPeriodOptimizationResult(
            weights=w_first,
            status="OPTIMAL_IPMO_MDFP",
            weights_trajectory=weights_traj_df,
            trades_trajectory=trades_traj_df,
            metadata={
                "horizon": H,
                "predicted_returns_mean": float(mu_seq.mean().item()),
                "planned_total_turnover": float(np.abs(trades_np).sum()),
            },
        )
