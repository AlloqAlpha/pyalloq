import torch
import torch.nn as nn
from typing import cast
from pyalloq_backtest.costs import BaseCostModel, CostData


class TorchCostAdapter(nn.Module):
    def __init__(self, cost_model: BaseCostModel) -> None:
        super().__init__()
        self.cost_model = cost_model

    def forward(
        self,
        weights_delta: torch.Tensor,
        volume: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features_slice: dict[str, CostData] | None = None
        if volume is not None:
            features_slice = {"volume": volume}
        costs_tensor = self.cost_model.calculate_costs(weights_delta, features_slice)

        return cast(torch.Tensor, costs_tensor)

    def decompose(
        self,
        weights_delta: torch.Tensor,
        volume: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features_slice: dict[str, CostData] | None = None
        if volume is not None:
            features_slice = {"volume": volume}
        spread_t, impact_t = self.cost_model.decompose_costs(
            weights_delta, features_slice
        )
        return cast(torch.Tensor, spread_t), cast(torch.Tensor, impact_t)
