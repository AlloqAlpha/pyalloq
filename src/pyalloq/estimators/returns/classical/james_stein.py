import numpy as np
import pandas as pd
from pyalloq_core.interfaces import BaseReturnEstimator
from pyalloq_core.data import MarketData


class JamesSteinReturnEstimator(BaseReturnEstimator):
    def estimate(
        self,
        data: MarketData,
        **kwargs,
    ) -> pd.Series:
        returns = data.prices.pct_change().dropna()
        hist_mu = returns.mean() * 252

        N = len(hist_mu)
        if N < 3:
            return hist_mu

        grand_mean = hist_mu.mean()
        target = np.full(N, grand_mean)

        cov_matrix = returns.cov() * 252
        inv_cov = np.linalg.pinv(cov_matrix.values)

        denom = (hist_mu.values - target).T @ inv_cov @ (hist_mu.values - target)

        w = (N - 2) / denom if denom > 0 else 1.0
        w = min(max(w, 0.0), 1.0)

        hist_mu_array = hist_mu.to_numpy()
        shrunk_mu = (1.0 - w) * hist_mu_array + w * target

        return pd.Series(shrunk_mu, index=data.prices.columns)
