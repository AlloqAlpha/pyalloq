import pandas as pd
from .base import BaseReturnEstimator

class MLReturnEstimator(BaseReturnEstimator):

    def __init__(
        self, 
        trained_model,
        feature_columns: list[str],
    ) -> None:
        self.model = trained_model
        self.features = feature_columns

    def estimate(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.Series:
        latest_features = features.iloc[-1][self.features]
        predictions = self.model.predict(latest_features)

        return pd.Series(
            predictions,
            index=prices.columns,
        )