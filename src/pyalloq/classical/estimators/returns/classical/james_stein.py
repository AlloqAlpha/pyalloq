from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.enums import DataFrequency
from pyalloq_core.interfaces import BaseReturnEstimator
from pyalloq_core.utils import resolve_annualization_factor


class JamesSteinReturnEstimator(BaseReturnEstimator):
    def __init__(
        self,
        frequency: DataFrequency | str | int | float | None = DataFrequency.DAILY,
    ) -> None:
        self.frequency = frequency

    def estimate(
        self,
        data: MarketData,
        **kwargs: Any,
    ) -> pd.Series:
        returns = data.prices.pct_change().dropna()
        scale_factor = resolve_annualization_factor(returns, self.frequency)
        hist_mu = returns.mean() * scale_factor

        N = len(hist_mu)
        if N < 3:
            return hist_mu

        grand_mean = hist_mu.mean()
        target = np.full(N, grand_mean)

        cov_matrix = returns.cov() * scale_factor
        inv_cov = np.linalg.pinv(cov_matrix.values)

        denom = (hist_mu.values - target).T @ inv_cov @ (hist_mu.values - target)

        w = (N - 2) / denom if denom > 0 else 1.0
        w = min(max(w, 0.0), 1.0)

        hist_mu_array = hist_mu.to_numpy()
        shrunk_mu = (1.0 - w) * hist_mu_array + w * target

        return pd.Series(shrunk_mu, index=data.prices.columns)
