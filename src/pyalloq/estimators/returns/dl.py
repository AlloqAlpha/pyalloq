import torch
import pandas as pd
from typing import Any
from pyalloq.core.interfaces import BaseReturnEstimator
from pyalloq.core.data import MarketData


class DLReturnEstimator(BaseReturnEstimator):
    def __init__(self, model: torch.nn.Module, lookback_window: int):
        self.model = model
        self.model.eval()
        self.lookback_window = lookback_window

    def estimate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> pd.Series:
        sequence_data = data.prices.iloc[-self.lookback_window :].values
        tensor_data = torch.tensor(sequence_data, dtype=torch.float32).T.unsqueeze(-1)
        with torch.no_grad():
            predicted_returns = self.model(tensor_data)
            predicted_returns = predicted_returns.squeeze().numpy()

        return pd.Series(predicted_returns, index=data.prices.columns)
