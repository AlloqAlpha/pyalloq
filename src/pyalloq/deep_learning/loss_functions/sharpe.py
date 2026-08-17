import torch
import torch.nn as nn
from pyalloq.backtest.costs import BaseCostModel, FlatBpsCostModel
from pyalloq.deep_learning.costs import TorchCostAdapter


class SharpeLoss(nn.Module):
    """ """

    def __init__(
        self,
        cost_model: BaseCostModel = FlatBpsCostModel(),
        risk_free_rate: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.eps = eps
        self.cost_model = TorchCostAdapter(cost_model=cost_model)

    def forward(
        self,
        weights_matrix: torch.Tensor,
        true_returns_matrix: torch.Tensor,
        volume_matrix: torch.Tensor | None = None,
        initial_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        weights: Tensor of shape (Batch Size, N assets). Must sum to 1
        true_returns: Tensor of shape(Batch Size, N assets). T+1 return
        """

        if weights_matrix.dim() == 2:
            weights_matrix = weights_matrix.unsqueeze(1)
            true_returns_matrix = true_returns_matrix.squeeze(1)
            if volume_matrix is not None:
                volume_matrix = volume_matrix.unsqueeze(1)

        gross_returns = torch.sum(weights_matrix * true_returns_matrix, dim=-1)

        if initial_weights is None:
            initial_weights = torch.zeros_like(weights_matrix[:, :1, :])
        else:
            initial_weights = initial_weights.unsqueeze(1)

        shifted_weights = torch.cat([initial_weights, weights_matrix[:, :-1, :]], dim=1)
        weights_delta = weights_matrix - shifted_weights

        cost_pct = self.cost_model(weights_delta, volume=volume_matrix)
        total_costs = torch.sum(cost_pct, dim=-1)

        net_returns = gross_returns - total_costs
        trajectory_returns = torch.sum(net_returns, dim=-1)

        expected_net_return = torch.mean(trajectory_returns) - self.risk_free_rate
        volatility = torch.std(trajectory_returns)

        sharpe_ratio = expected_net_return / (volatility + self.eps)

        return -sharpe_ratio
