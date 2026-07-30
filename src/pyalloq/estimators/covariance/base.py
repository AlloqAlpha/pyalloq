from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseCovarianceEstimator(ABC):

    @abstractmethod
    def estimate(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Takes a price DataFrame and returns an NxN covariance matrix."""
        pass
