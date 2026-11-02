import torch
from torch import nn

from pyalloq.dl.layers.mdfp import MDFPOptimizationLayer


class SPOPlusLoss(nn.Module):
    """
    Smart "Predict, then Optimize" Plus (SPO+) Convex Surrogate Loss (Elmachtoub & Grigas 2022)
    extended to Multi-Period Portfolio Trajectories.

    Given predicted return trajectory mu_hat (B, H, N) and realized return trajectory y_true (B, H, N),
    evaluates the perturbed objective vector:
        mu_spo = 2 * mu_hat - y_true
    and computes the convex upper bound on decision regret via the MDFP oracle:
        L_SPO+(mu_hat, y_true) = J(z*(2*mu_hat - y_true); 2*mu_hat - y_true)
                                 + 2 * <mu_hat, w_true - z*(2*mu_hat - y_true)>
    """

    def __init__(
        self,
        mdfp_layer: MDFPOptimizationLayer | None = None,
        horizon: int = 5,
        risk_aversion: float = 2.0,
        cost_penalty: float = 0.001,
    ) -> None:
        super().__init__()
        self.mdfp_layer = mdfp_layer or MDFPOptimizationLayer(
            horizon=horizon,
            risk_aversion=risk_aversion,
            cost_penalty=cost_penalty,
        )
        self.cost_penalty = cost_penalty

    def forward(
        self,
        mu_hat: torch.Tensor,
        y_true: torch.Tensor,
        cov_seq: torch.Tensor | None = None,
        w_current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mu_hat.dim() == 2:
            mu_hat = mu_hat.unsqueeze(1)
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(1)

        B, H, N = mu_hat.shape
        if w_current is None:
            w_current = torch.full(
                (B, N), 1.0 / N, device=mu_hat.device, dtype=mu_hat.dtype
            )

        # Perturbed return vector 2 * mu_hat - y_true
        mu_spo = 2.0 * mu_hat - y_true

        # Solve oracle trajectories for perturbed and ground-truth returns
        z_spo = self.mdfp_layer(mu_spo, cov_seq=cov_seq, w_current=w_current)
        with torch.no_grad():
            z_true = self.mdfp_layer(y_true, cov_seq=cov_seq, w_current=w_current)

        # Compute turnover penalties along z_spo and z_true
        z_spo_prev = torch.cat([w_current.unsqueeze(1), z_spo[:, :-1, :]], dim=1)
        z_true_prev = torch.cat([w_current.unsqueeze(1), z_true[:, :-1, :]], dim=1)
        tc_spo = self.cost_penalty * torch.abs(z_spo - z_spo_prev).sum(dim=-1)
        tc_true = self.cost_penalty * torch.abs(z_true - z_true_prev).sum(dim=-1)

        # SPO+ convex regret surrogate over H stages
        term_spo = (mu_spo * z_spo).sum(dim=-1) - tc_spo
        term_true = (y_true * z_true).sum(dim=-1) - tc_true
        correction = 2.0 * (mu_hat * (z_true - z_spo)).sum(dim=-1)

        loss_per_stage = term_spo + correction - term_true
        return torch.relu(loss_per_stage).mean()


class HybridDFLLoss(nn.Module):
    """
    Hybrid Decision-Focused Learning (DFL) Loss (Research Section 6.3):
        L_Hybrid = alpha * L_MPO(z*(mu_hat), y_true) + (1 - alpha) * L_MSE(mu_hat, y_true)

    Blends multi-period net-of-cost decision utility (negative Sharpe or negative net utility
    across the trajectory z*) with regularizing MSE return prediction loss to prevent overfitting
    in low signal-to-noise financial regimes.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        cost_penalty: float = 0.001,
        risk_aversion: float = 2.0,
        discount: float = 0.98,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0.0, 1.0].")
        self.alpha = alpha
        self.cost_penalty = cost_penalty
        self.risk_aversion = risk_aversion
        self.discount = discount
        self.mse = nn.MSELoss()

    def forward(
        self,
        z_trajectory: torch.Tensor,
        mu_hat: torch.Tensor,
        y_true: torch.Tensor,
        w_current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_trajectory : torch.Tensor
            Differentiable optimal weight trajectory z*(mu_hat) of shape (B, H, N) or (B, N).
        mu_hat : torch.Tensor
            Predicted return trajectory of shape (B, H, N) or (B, N).
        y_true : torch.Tensor
            Realized forward return trajectory of shape (B, H, N) or (B, N).
        w_current : torch.Tensor, optional
            Starting pre-trade weights of shape (B, N).
        """
        if z_trajectory.dim() == 2:
            z_trajectory = z_trajectory.unsqueeze(1)
        if mu_hat.dim() == 2:
            mu_hat = mu_hat.unsqueeze(1)
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(1)

        B, H, N = z_trajectory.shape
        device = z_trajectory.device
        dtype = z_trajectory.dtype

        if w_current is None:
            w_current = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)

        stage_idx = torch.arange(H, device=device, dtype=dtype).view(1, H)
        discounts = self.discount**stage_idx

        z_prev = torch.cat([w_current.unsqueeze(1), z_trajectory[:, :-1, :]], dim=1)
        turnover = torch.abs(z_trajectory - z_prev).sum(dim=-1)  # (B, H)

        # Realized net portfolio return at each stage s: r_{p,s} = y_{true,s}^T z_s - kappa * turnover_s
        gross_port_ret = (z_trajectory * y_true).sum(dim=-1)  # (B, H)
        net_port_ret = gross_port_ret - self.cost_penalty * turnover  # (B, H)

        # Multi-period mean-variance / Sharpe decision objective
        stage_utility = net_port_ret - 0.5 * self.risk_aversion * (net_port_ret.pow(2))
        l_mpo = -(discounts * stage_utility).mean()

        l_mse = self.mse(mu_hat, y_true)
        return self.alpha * l_mpo + (1.0 - self.alpha) * l_mse
