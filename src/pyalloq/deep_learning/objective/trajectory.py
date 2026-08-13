import torch
import torch.nn as nn
from pyalloq.deep_learning.costs import TorchCostAdapter
from pyalloq.backtest.costs import BaseCostModel


class TrajectoryLoss(nn.Module):
    """
    For Multi-Horizon Models.
    Calculates the Sharpe ratio across K future days deducing
    transaction costs based on portfolio turnover required to navigate the path.
    """

    def __init__(self, cost_model: BaseCostModel, eps: float = 1e-6) -> None:
        super().__init__()
        self.cost_model = TorchCostAdapter(cost_model=cost_model)
        self.eps = eps

    def forward(
        self,
        weights_matrix: torch.Tensor,
        true_returns_matrix: torch.Tensor,
        volume_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            weights_matrix: Tensor of shape (Batch Size, Horizon, N assets)
            true_returns_matrix: Tensor of shape (Batch Size, Horizon, N assets)
            volume_matrix: Tensor of shape (Batch Size, Horizon, N assets) - Required fir Almgren-Chriss
        """
        gross_returns = torch.sum(weights_matrix * true_returns_matrix, dim=-1)

        zeros = torch.zeros(
            (weights_matrix.size(0), 1, weights_matrix.size(2)),
            device=weights_matrix.device,
        )
        shifted_weights = torch.cat([zeros, weights_matrix[:, :-1, :]], dim=1)

        weights_delta = weights_matrix - shifted_weights

        cost_pct = self.cost_model(weights_delta, volume=volume_matrix)
        total_costs = torch.sum(cost_pct, dim=-1)

        net_returns = gross_returns - total_costs
        expected_net_returns = torch.mean(net_returns)
        volatility = torch.std(net_returns)

        sharpe_ratio = expected_net_returns / (volatility + self.eps)
        return -sharpe_ratio
