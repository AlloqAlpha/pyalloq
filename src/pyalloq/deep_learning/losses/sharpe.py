import torch
import torch.nn as nn

class SharpeLoss(nn.Module):
    def __init__(self, risk_free_rate: float = 0.0) -> None:
        super().__init__()
        self.rf_rate = risk_free_rate

    def forward(
        self,
        weights: torch.Tensor,
        future_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        weights shape: (Batch, N_assets)
        future_returns shape: (Batch, N_assets)
        """
        portfolio_returns = torch.sum(weights * future_returns, dim=-1)
        expected_returns = torch.mean(portfolio_returns) - self.rf_rate
        portfolio_volatility = torch.std(portfolio_returns)

        sharpe_ratio = expected_returns / (portfolio_volatility + 1e-6)
        return -sharpe_ratio