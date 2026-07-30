import pandas as pd
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.results import OptimizationResult

class NaiveAllocator(BaseAllocator):
    def allocate(
        self,
        prices: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        n_assets = len(prices.columns)
        weights = pd.Series(1.0, n_assets, index=prices.columns)

        return OptimizationResult(
            weights=weights,
            status="OPTIMAL_NAIVE",
            metadata={"strategy": "1/N Equal Weight"}
        )

        