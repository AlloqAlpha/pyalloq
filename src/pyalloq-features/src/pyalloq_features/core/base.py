from abc import ABC, abstractmethod
from pyalloq_core.data import MarketData


class BaseFeatureTransformer(ABC):
    """
    Abstract interface for all feature engineering modules.
    Operates directly on the MarketData dataclass.
    """

    def fit(self, data: MarketData) -> "BaseFeatureTransformer":
        return self

    @abstractmethod
    def transform(self, data: MarketData) -> MarketData:
        """Applies transformation and strictly appends new columns."""
        ...
