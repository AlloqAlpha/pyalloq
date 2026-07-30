import numpy as np
import pandas as pd
from pyalloq.core.interfaces import BaseReturnEstimator

class JamesSteinReturnEstimator(BaseReturnEstimator):
    def estimate(
        self,
        prices: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        returns = prices.pct_change().dropna()
        hist_mu = returns.mean() * 252

        N = len(hist_mu)
        if N < 3:
            return hist_mu

        grand_mean = hist_mu.mean()
        target = np.full(N, grand_mean)

        cov_matrix = returns.cov() * 252
        inv_cov = np.linalg.pinv(cov_matrix.values)

        ones = np.ones(N)

        denom = (hist_mu.values - target).T @ inv_cov @ (hist_mu.values - target)

        w = (N - 2) / denom if denom > 0 else 1.0
        w = min(max(w, 0.0, 1.0))

        shrunk_mu = (1 - w) * hist_mu.values + w * target

        return pd.Series(shrunk_mu, index=prices.columns)