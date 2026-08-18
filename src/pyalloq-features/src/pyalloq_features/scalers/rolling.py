from pyalloq_features.core.base import BaseFeatureTransformer
from pyalloq_core.data import MarketData


class RollingZScoreScaler(BaseFeatureTransformer):
    """Point-in-Time Z-Score normalization utilizing backward-looking statistics."""

    def __init__(self, target_feature: str, window: int = 252) -> None:
        self.target_feature = target_feature
        self.window = window
        self.out_col = f"{target_feature}_zscore_{window}"

    def transform(self, data: MarketData) -> MarketData:
        df_feature = data.features[self.target_feature]

        rolling_view = df_feature.rolling(
            window=self.window, min_periods=self.window // 2
        )
        rolling_mean = rolling_view.mean()
        rolling_std = rolling_view.std()

        epsilon = 1e-8
        data.features[self.out_col] = (df_feature - rolling_mean) / (
            rolling_std + epsilon
        )

        return data
