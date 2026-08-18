import talib
from pyalloq_features.core.base import BaseFeatureTransformer
from pyalloq_core.data import MarketData
import pandas as pd
import numpy as np


class TechnicalIndicator(BaseFeatureTransformer):
    """
    Applies TA-Lib functions across the asset columns of a 2D DataFrame.
    """

    def __init__(
        self,
        indicator: str,
        source: str = "prices",
        **kwargs,
    ) -> None:
        self.indicator_name = indicator.upper()
        self.source = source  # prices or a key in data.features
        self.kwargs = kwargs
        self._func = getattr(talib, self.indicator_name)

        params_str = "_".join([str(v) for v in self.kwargs.values()])
        self.out_col = (
            f"{self.indicator_name}_{params_str}" if params_str else self.indicator_name
        )

    def transform(self, data: MarketData) -> MarketData:
        if self.source == "prices":
            df_source = data.prices
        else:
            df_source = data.features[self.source]

        df_out = pd.DataFrame(
            index=df_source.index, columns=df_source.columns, dtype=np.float64
        )

        for ticker in df_source.columns:
            asset_series = df_source[ticker].to_numpy()
            try:
                df_out[ticker] = self._func(asset_series, **self.kwargs)
            except Exception:
                pass

        data.features[self.out_col] = df_out

        return data
