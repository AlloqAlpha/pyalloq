import torch
import numpy as np
from torch.utils.data import Dataset
from pyalloq_core.data import MarketData


class MarketDataset(Dataset):
    def __init__(
        self,
        data: MarketData,
        lookback_window: int = 60,
        horizon: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.lookback_window = lookback_window
        self.horizon = horizon
        self._dtype = dtype

        df_returns = data.prices.pct_change().dropna()
        self.assets = df_returns.columns.tolist()
        self.dates = df_returns.index.tolist()

        features_list = [df_returns.to_numpy()]

        if data.features is not None:
            for feat_name, df_feature in data.features.items():
                aligned_feat = df_feature.reindex(
                    index=df_returns.index, columns=self.assets
                ).fillna(0.0)
                features_list.append(aligned_feat.to_numpy())

        stacked_features = np.stack(features_list, axis=-1)
        self.X_data = torch.tensor(stacked_features, dtype=self._dtype)
        self.y_data = torch.tensor(df_returns.to_numpy(), dtype=self._dtype)

        self.valid_length = len(df_returns) - self.lookback_window - self.horizon + 1

        if self.valid_length <= 0:
            raise ValueError(
                f"Data length ({len(df_returns)}) is too short for lookback"
                f"({self.lookback_window}) and horizon ({self.horizon})"
            )

    def __len__(self) -> int:
        return self.valid_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: Tensor of shape (Lookback, No. Assets, No. No. Features)
            y: Tensor of shape (Horizon. No. Assets)
        """
        x_start = idx
        x_end = idx + self.lookback_window
        X_tensor = self.X_data[x_start:x_end]

        y_start = x_end
        y_end = y_start + self.horizon
        y_tensor = self.y_data[y_start:y_end]

        if self.horizon == 1:
            y_tensor = y_tensor.unsqueeze(0)

        return (X_tensor, y_tensor)
