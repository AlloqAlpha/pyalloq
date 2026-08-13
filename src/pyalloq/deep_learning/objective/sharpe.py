import torch
import torch.nn as nn


class SharpeLoss(nn.Module):
    """
    Calculates the negative Sharpe Ratio on a batch of predicted portfolio weights.
    By minimizing this loss, the network learns risk-adjusted allocation.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.eps = eps

    def forward(
        self, weights: torch.Tensor, future_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        weights: Tensor of shape (Batch Size, N assets). Must sum to 1
        true_returns: Tensor of shape(Batch Size, N assets). T+1 return
        """

        portfolio_returns = torch.sum(weights * future_returns, dim=1)
        expected_return = torch.mean(portfolio_returns)
        volatility = torch.std(portfolio_returns)

        sharpe_ratio = (expected_return - self.risk_free_rate) / (volatility + self.eps)

        return -sharpe_ratio
