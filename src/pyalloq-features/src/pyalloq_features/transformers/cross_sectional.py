from pyalloq_features.core.base import BaseFeatureTransformer
from pyalloq_core.data import MarketData


class CrossSectionalRank(BaseFeatureTransformer):
    """
    Ranks a feature cross-sectionality across all available assets at timestamp T.
    """

    def __init__(self, target_feature: str) -> None:
        self.target_feature = target_feature
        self.out_col = f"{target_feature}_cs_rank"

    def transform(self, data: MarketData) -> MarketData:
        df_features = data.features[self.target_feature]

        cs_rank = df_features.rank(axis=1, pct=True)
        data.features[self.out_col] = cs_rank

        return data
