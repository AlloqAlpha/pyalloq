from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult


class EqualWeightAllocator(BaseAllocator):
    def __init__(self, tickers: list[str] | None = None) -> None:
        super().__init__(tickers or [])

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        N = len(data.prices.columns)
        weights = pd.Series(1.0 / N, index=data.prices.columns)

        return OptimizationResult(
            weights=weights,
            status="OPTIMAL_EQUAL_WEIGHT",
            metadata={"strategy": "1/N Equal Weight"},
        )


class InverseVolatilityAllocator(BaseAllocator):
    """
    Allocates weights inversely proportional to each asset's standalone volatility:
        w_i = (1 / sigma_i) / sum_j (1 / sigma_j).
    Ignores cross-asset correlations, serving as a fast analytical baseline for Risk Parity.
    """

    def __init__(self, tickers: list[str] | None = None) -> None:
        super().__init__(tickers or [])

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if cov_matrix is not None:
            assets = list(cov_matrix.columns)
            vols = np.sqrt(np.maximum(np.diag(cov_matrix.values), 1e-12))
        else:
            assets = list(data.prices.columns)
            rets = data.prices.pct_change().dropna()
            vols = np.maximum(rets.std().to_numpy(dtype=float), 1e-12)

        inv_vols = 1.0 / vols
        weights_arr = inv_vols / float(np.sum(inv_vols))
        weights = pd.Series(weights_arr, index=assets)

        return OptimizationResult(
            name="InverseVolatility",
            weights=weights,
            status="OPTIMAL_INVERSE_VOLATILITY",
            metadata={"strategy": "Inverse Volatility (1/sigma)"},
        )


class MarketCapWeightAllocator(BaseAllocator):
    """
    Allocates weights proportional to asset market capitalizations (`market_cap` in
    `data.cross_sectional` or `market_caps` passed via kwargs). Falls back to 1/N if unavailable.
    """

    def __init__(self, tickers: list[str] | None = None) -> None:
        super().__init__(tickers or [])

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        assets = list(data.prices.columns)
        mcaps: pd.Series | None = kwargs.get("market_caps")
        if (
            mcaps is None
            and data.cross_sectional is not None
            and "market_cap" in data.cross_sectional.columns
        ):
            mcaps = data.cross_sectional["market_cap"]

        if mcaps is None:
            weights = pd.Series(1.0 / len(assets), index=assets)
            return OptimizationResult(
                name="MarketCapWeight",
                weights=weights,
                status="FALLBACK_EQUAL_WEIGHT",
                metadata={"strategy": "Market Cap Weight (fallback 1/N)"},
            )

        aligned = mcaps.reindex(assets).fillna(0.0).clip(lower=0.0).astype(float)
        total_mcap = float(aligned.sum())
        if total_mcap <= 1e-12:
            weights = pd.Series(1.0 / len(assets), index=assets)
        else:
            weights = aligned / total_mcap

        return OptimizationResult(
            name="MarketCapWeight",
            weights=weights,
            status="OPTIMAL_MARKET_CAP",
            metadata={"strategy": "Market Cap Weighted"},
        )


class RandomAllocator(BaseAllocator):
    """
    Generate random weights that sum to 1 using Dirichlet distribution.
    """

    def __init__(
        self, tickers: list[str] | None = None, seed: int | None = None
    ) -> None:
        super().__init__(tickers or [])
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        tickers = data.prices.columns
        weights = pd.Series(
            self._rng.dirichlet(np.ones(len(tickers))),
            index=tickers,
        )

        return OptimizationResult(
            weights=weights,
            status="OPTIMAL_RANDOM",
            metadata={"strategy": "Random (Dirichlet)", "seed": self.seed},
        )
