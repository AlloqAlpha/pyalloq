import numpy as np
import pandas as pd
from pyalloq.optimizers.machine_learning.hrp import HRPAllocator
from pyalloq.optimizers.classical.risk_parity import RiskParityAllocator

class HERCAllocator(HRPAllocator):
    def __init__(self, tickers: list[str]) -> None:
        super().__init__(tickers)
        self.erc_solver = RiskParityAllocator(tickers)

    def _get_cluster_risk(self, cov: np.ndarray, cluster: list) -> float:

        if len(cluster) == 1:
            return cov[cluster[0], cluster[0]]

        df_sub_cov = pd.DataFrame(cov[np.ix_(cluster, cluster)])
        erc_result = self.erc_solver.allocate(df_sub_cov)
        w_erc = erc_result.weights.values
        sub_cov_val = df_sub_cov.values
        cluster_variance = w_erc.T @ sub_cov_val @ w_erc

        return cluster_variance

    def _recursive_bisection(
        self, 
        cov: np.ndarray,
        sort_ix: np.ndarray,
        weights: pd.Series
    ) -> pd.Series:

        clusters = [sort_ix.tolist()]
        while len(clusters) > 0:
            clusters = [c[i:j] for c in clusters for i,j in ((0, int(len(c)/2)), (int(len(c)/2), len(c))) if len(c) > 1]

            for i in range(0, len(clusters, 2)):
                cluster_left = clusters[i]
                cluster_right = clusters[i+1]

                risk_left = self._get_cluster_risk(cov, cluster_left)
                risk_right = self._get_cluster_risk(cov, cluster_right)

                alpha = 1.0 - (risk_left / (risk_left + risk_right))

                weights.loc[cluster_left] *= alpha
                weights.loc[cluster_right] *= (1 - alpha)

        return weights
        