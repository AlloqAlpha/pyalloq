from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from pyalloq.dl.estimators.covariance.models.gnn import CRISPSparseSTGAT
from pyalloq.dl.layers.fragility_kkt import (
    BasePolicyNetwork,
    DifferentiableCVaRLayer,
    DifferentiableRiskParityLayer,
)
from pyalloq.dl.layers.mdfp import MDFPOptimizationLayer
from pyalloq.dl.layers.neural_jump_rde import TorchLogSignatureExtractor

PolicyHeadMode = Literal["mdfp", "risk_parity_kkt", "cvar_kkt", "residual_actor"]


def compute_von_neumann_entropy(
    cov: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Computes the Von Neumann Entropy (CVNE) of a batch of SPD covariance matrices
    (PDF p. 8, Kronecker coVariance Neural Network regularity metric):
        S_vN(Sigma) = - sum_{i=1}^N p_i log(p_i),   where p_i = lambda_i / Tr(Sigma).

    When `normalize=True`, divides by `log(N)` so `S_vN in [0, 1]`:
    - `S_vN -> 1`: isotropic / well-diversified eigenvalue spectrum.
    - `S_vN -> 0`: rank-1 contagion collapse (single dominant market mode).
    """
    squeezed = cov.dim() == 2
    if squeezed:
        cov = cov.unsqueeze(0)

    B, N, _ = cov.shape
    if N <= 1:
        out = torch.ones(B, device=cov.device, dtype=cov.dtype)
        return out.squeeze(0) if squeezed else out

    sym_cov = 0.5 * (cov + cov.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(sym_cov).clamp(min=1e-8)
    probs = eigvals / eigvals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
    if normalize:
        entropy = entropy / float(np.log(max(2, N)))
    return entropy.squeeze(0) if squeezed else entropy


class LogSigSSMBlock(nn.Module):
    """
    Log-Signature Selective State-Space Model (LogSig-SSM) Encoder (PDF p. 5 & p. 12).

    Couples multi-window Lyndon Log-Signature tokenization (`TorchLogSignatureExtractor`)
    with an S6/Mamba-inspired input-dependent selective state-space recurrence:
        h_k = exp(-Delta_k * A) * h_{k-1} + (Delta_k * B_k) * u_k
        y_k = C_k * h_k + D * u_k
    where u_k is the local Lyndon log-signature token over sub-interval k, and
    (Delta_k, B_k, C_k) are selectively projected from u_k.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 32,
        state_dim: int = 16,
        num_windows: int = 4,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.num_windows = max(1, int(num_windows))

        self.logsig = TorchLogSignatureExtractor(
            n_channels=self.in_channels, window_size=max(2, int(window_size))
        )
        sig_dim = self.logsig.logsig_dim

        self.in_proj = nn.Sequential(
            nn.Linear(sig_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        # Selective SSM projections: Delta (step size), B (input matrix), C (readout matrix)
        self.delta_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.b_proj = nn.Linear(self.hidden_dim, self.state_dim)
        self.c_proj = nn.Linear(self.hidden_dim, self.state_dim)
        self.log_a = nn.Parameter(
            torch.linspace(0.0, 2.0, self.state_dim)
            .unsqueeze(0)
            .repeat(self.hidden_dim, 1)
        )
        self.skip_d = nn.Parameter(torch.ones(self.hidden_dim))
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x_asset: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_asset : torch.Tensor
            Asset path tensor of shape `(M, L, C)`.

        Returns
        -------
        torch.Tensor
            Terminal selective state-space representation of shape `(M, hidden_dim)`.
        """
        m_batch = x_asset.shape[0]
        logsig_tokens, _ = self.logsig(x_asset)  # (M, S, sig_dim)
        n_steps = logsig_tokens.shape[1]

        a_neg = -torch.exp(self.log_a.clamp(min=-4.0, max=3.0))  # (H, S)
        h_state = torch.zeros(
            (m_batch, self.hidden_dim, self.state_dim),
            device=x_asset.device,
            dtype=x_asset.dtype,
        )
        y_last = torch.zeros(
            (m_batch, self.hidden_dim),
            device=x_asset.device,
            dtype=x_asset.dtype,
        )

        for step_idx in range(n_steps):
            sig_tok = logsig_tokens[:, step_idx, :]  # (M, sig_dim)
            u_k = self.in_proj(sig_tok)  # (M, H)
            delta_k = torch.nn.functional.softplus(self.delta_proj(u_k)).clamp(
                min=1e-3, max=2.0
            )  # (M, H)
            b_k = self.b_proj(u_k)  # (M, S)
            c_k = self.c_proj(u_k)  # (M, S)

            decay = torch.exp(delta_k.unsqueeze(-1) * a_neg.unsqueeze(0))  # (M, H, S)
            drive = (delta_k * u_k).unsqueeze(-1) * b_k.unsqueeze(1)  # (M, H, S)
            h_state = decay * h_state + drive
            y_last = (h_state * c_k.unsqueeze(1)).sum(dim=-1) + self.skip_d * u_k

        return self.out_norm(y_last)


class UnifiedE2EPolicyNetwork(BasePolicyNetwork):
    """
    Unified 3-Stage End-to-End Portfolio Policy Network (Pillar 4 Controller):
    1. **Stage 1 (Topological & Rough-Signature Perception):**
       Combines `CRISPSparseSTGAT` (NeuralJumpRDE + CRISP sparse graph attention)
       and `LogSigSSMBlock` selective state-space encoding to extract regime-aware
       expected returns `mu_t`, fragility-shrunk SPD covariance `Sigma_t`, and
       Von Neumann Entropy `S_vN(Sigma_t)`.
    2. **Stage 2 (Differentiable Optimization / Actor Head):**
       Dispatches `(mu_t, Sigma_t, w_current)` to one of four interchangeable layers:
       - `"mdfp"`: Multi-period Entropic Mirror-Descent Fixed-Point (`MDFPOptimizationLayer`)
       - `"risk_parity_kkt"`: Implicit KKT Equal/Budgeted Risk Contribution (`DifferentiableRiskParityLayer`)
       - `"cvar_kkt"`: Implicit KKT CVaR-Tilted Risk Budgeting (`DifferentiableCVaRLayer`)
       - `"residual_actor"`: Direct Neural Actor head in log-weight space around `w_current`
    3. **Stage 3 (Turnover-Aware No-Trade-Zone Controller):**
       Applies an endogenous Von-Neumann-Entropy and turnover-modulated retention gate
       between `w_current` and the Stage 2 target weights.
    """

    def __init__(
        self,
        num_assets: int,
        num_features: int = 1,
        hidden_dim: int = 32,
        num_heads: int = 2,
        head_mode: PolicyHeadMode = "mdfp",
        backbone: nn.Module | None = None,
        horizon: int = 5,
        risk_aversion: float = 2.0,
        cost_penalty: float = 0.002,
        no_trade_buffer: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_assets = int(num_assets)
        self.num_features = int(num_features)
        self.hidden_dim = int(hidden_dim)
        self.head_mode: PolicyHeadMode = head_mode
        self.no_trade_buffer = float(np.clip(no_trade_buffer, 0.0, 0.95))

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = CRISPSparseSTGAT(
                num_assets=self.num_assets,
                num_features=self.num_features,
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
            )

        self.logsig_ssm = LogSigSSMBlock(
            in_channels=self.num_features,
            hidden_dim=self.hidden_dim,
        )
        self.ssm_mu_proj = nn.Linear(self.hidden_dim, 1)

        self.mdfp_layer = MDFPOptimizationLayer(
            horizon=horizon,
            risk_aversion=risk_aversion,
            cost_penalty=cost_penalty,
        )
        self.rp_layer = DifferentiableRiskParityLayer()
        self.cvar_layer = DifferentiableCVaRLayer(risk_aversion=risk_aversion)
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.last_von_neumann_entropy_: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() == 3:
            # (B, L, N) -> (B, L, N, 1)
            x = x.unsqueeze(-1)

        if x.shape[-1] > self.num_features:
            x = x[..., : self.num_features]
        elif x.shape[-1] < self.num_features:
            pad_c = self.num_features - x.shape[-1]
            x = torch.cat(
                [
                    x,
                    torch.zeros((*x.shape[:-1], pad_c), device=x.device, dtype=x.dtype),
                ],
                dim=-1,
            )

        B, L, N, F = x.shape
        device = x.device
        dtype = x.dtype

        if w_current is None:
            w_curr = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
        elif w_current.dim() == 1:
            w_curr = w_current.unsqueeze(0).expand(B, N).to(device=device, dtype=dtype)
        else:
            w_curr = w_current.to(device=device, dtype=dtype)
        w_curr = self._normalize_simplex(w_curr)

        # Stage 1: Spatio-Temporal Graph + LogSig-SSM Perception
        pred_fn = getattr(self.backbone, "predict_returns_and_cov", None)
        if callable(pred_fn):
            mu_stgat, cov_stgat, _ = pred_fn(x)
        else:
            cov_stgat = self.backbone(x)
            mu_stgat = x[:, -1, :, 0]

        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, L, F)
        if F == self.num_features:
            ssm_emb = self.logsig_ssm(x_flat).reshape(B, N, self.hidden_dim)
            mu_ssm = 0.05 * torch.tanh(self.ssm_mu_proj(ssm_emb).squeeze(-1))
            mu = 0.5 * (mu_stgat + mu_ssm)
        else:
            ssm_emb = torch.zeros((B, N, self.hidden_dim), device=device, dtype=dtype)
            mu = mu_stgat

        vn_entropy = compute_von_neumann_entropy(cov_stgat, normalize=True)
        self.last_von_neumann_entropy_ = vn_entropy.detach()

        # Stage 2: Differentiable Decision / Optimization Layer
        if self.head_mode == "mdfp":
            z_traj = self.mdfp_layer(mu_seq=mu, cov_seq=cov_stgat, w_current=w_curr)
            w_opt = z_traj[:, 0, :]
        elif self.head_mode == "risk_parity_kkt":
            w_opt = self.rp_layer(cov_stgat)
        elif self.head_mode == "cvar_kkt":
            ret_scenarios = x[:, :, :, 0]
            w_opt = self.cvar_layer(
                expected_returns=mu,
                cov_matrix=cov_stgat,
                scenarios=ret_scenarios,
            )
        else:
            # "residual_actor": Condition directly on [ssm_emb, mu, w_curr]
            actor_in = torch.cat(
                [ssm_emb, mu.unsqueeze(-1), w_curr.unsqueeze(-1)], dim=-1
            )
            logits = self.actor_head(actor_in).squeeze(-1)
            log_prior = torch.log(w_curr.clamp(min=1e-6))
            w_opt = torch.softmax(log_prior + logits, dim=-1)

        # Stage 3: Turnover-Aware No-Trade-Zone Controller
        if self.no_trade_buffer > 0.0 and self.head_mode in (
            "risk_parity_kkt",
            "cvar_kkt",
        ):
            trade_rate = 1.0 - self.no_trade_buffer
            w_target = trade_rate * w_opt + self.no_trade_buffer * w_curr
            return self._normalize_simplex(w_target)

        return self._normalize_simplex(w_opt)
