import numpy as np
import pandas as pd
from typing import Any
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from pyalloq_core.utils import cov_to_corr
from pyalloq_core.data import MarketData


class HRPAllocator(BaseAllocator):
    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if cov_matrix is None:
            raise ValueError("HRPAllocator requires cov_matrix")

        corr = cov_to_corr(cov_matrix.to_numpy())
        dist = np.sqrt(0.5 * (1 - corr))

        condensed_dist = ssd.squareform(dist, checks=False)
        link = sch.linkage(condensed_dist, method="single")
        sort_ix = self._get_quasi_diag(link)

        weights = pd.Series(1.0, index=cov_matrix.columns)
        weights = self._recursive_bisection(cov_matrix.to_numpy(), sort_ix, weights)

        final_weights = pd.Series(0.0, index=cov_matrix.columns)
        for idx in sort_ix:
            final_weights.iloc[idx] = weights.iloc[idx]

        return OptimizationResult(
            weights=final_weights,
            status="OPTIMAL_HRP",
        )

    def _get_quasi_diag(self, link: np.ndarray) -> np.ndarray:
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = pd.Index(range(0, sort_ix.shape[0] * 2, 2))

            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]).sort_index()

        return np.array(sort_ix.tolist())

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sort_ix: np.ndarray,
        weights: pd.Series,
    ) -> pd.Series:
        clusters = [sort_ix.tolist()]
        while len(clusters) > 0:
            clusters = [
                c[i:j]
                for c in clusters
                for i, j in ((0, int(len(c) / 2)), (int(len(c) / 2), len(c)))
                if len(c) > 1
            ]
            for i in range(0, len(clusters), 2):
                cluster_left = clusters[i]
                cluster_right = clusters[i + 1]

                var_left = self._get_cluster_var(cov, cluster_left)
                var_right = self._get_cluster_var(cov, cluster_right)

                alpha = 1 - var_left / (var_left + var_right)
                weights.iloc[cluster_left] *= alpha
                weights.iloc[cluster_right] *= 1 - alpha

        return weights

    def _get_cluster_var(
        self,
        cov: np.ndarray,
        cluster: list,
    ) -> float:
        sub_cov = cov[np.ix_(cluster, cluster)]
        ivp = 1.0 / np.diag(sub_cov)
        ivp /= ivp.sum()

        return np.dot(ivp.T, np.dot(sub_cov, ivp))
