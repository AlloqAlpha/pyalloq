import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pyalloq.optimizers.classical.markowitz import MarkowitzAllocator
from pyalloq.core.interfaces import BaseAllocator
from pyalloq.core.results import OptimizationResult
from pyalloq.core.utils import cov_to_corr
from pyalloq.core.enums import ObjectiveFunction

class NCOAllocator(BaseAllocator):
    def __init__(
        self,
        tickers: list[str],
        inner_optimizer: BaseAllocator | None = None,
        outer_optimizer: BaseAllocator | None = None,
        n_clusters: int = 4
    ) -> None:
        super().__init__(
            tickers=tickers
        )
        self.inner_optimizer = inner_optimizer or MarkowitzAllocator(tickers)
        self.outer_optimizer = outer_optimizer or MarkowitzAllocator(tickers)
        self.n_clusters = n_clusters

    def allocate(
        self, 
        cov_matrix: pd.DataFrame,
        objective: ObjectiveFunction = ObjectiveFunction.MAX_SHARPE,
        **kwargs,
    ) -> OptimizationResult:
        corr = cov_to_corr(cov_matrix.values)
        distance_matrix = np.sqrt(0.5 * (1 - corr))

        kmeans = KMeans(n_clusters=self.n_clusters)
        clusters = kmeans.fit_predict(distance_matrix)

        final_weights = pd.Series(0.0, index=cov_matrix.columns)
        cluster_covariance = np.zeros((self.n_clusters, self.n_clusters))

        inner_weights_dict = {}
        for i in range(self.n_clusters):
            cluster_assets = cov_matrix.columns[clusters == i]
            sub_cov = cov_matrix.loc[cluster_assets, cluster_assets]

            inner_result = self.inner_optimizer.allocate(cov_matrix=sub_cov, objective=objective)
            inner_weights_dict[i] = inner_result.weights

            cluster_covariance[i, i] = inner_result.weights.T @ sub_cov @ inner_result.weights

        outer_cov_df = pd.DataFrame(cluster_covariance)
        outer_result = self.outer_optimizer.allocate(cov_matrix=outer_cov_df, objective=objective)

        for i in range(self.n_clusters):
            cluster_assets = cov_matrix.columns[clusters == i]
            final_weights[cluster_assets] = inner_weights_dict[i] * outer_result.weights.iloc[i]

        return OptimizationResult(
            weights=final_weights,
            status="OPTIMAL_NCO"
        )


        

