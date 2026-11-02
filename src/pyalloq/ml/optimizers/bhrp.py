from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from pyalloq_core.utils import cov_to_corr


class BayesianHRPAllocator(BaseAllocator):
    """
    Two-Level Bayesian Hierarchical Risk Parity (RL-BHRP) Allocator (Research Section 7.2).

    Combines:
    1. Hierarchical Correlation Clustering into K macro-clusters (or user-supplied sectors).
    2. Intra-Cluster HRP Recursive Bisection to form risk-balanced cluster sub-portfolios w^{(k)}.
    3. Inter-Cluster Empirical Bayes / Black-Litterman Posterior Return Estimation:
       Shrinks sample cluster return views toward the equilibrium risk-parity implied return prior
       pi_k = delta * (Sigma_cluster w_hrp)_k with confidence weight tau.
    4. Multi-Period Turnover-Regularized Inter-Cluster Tilt:
       Tilts cluster budgets away from baseline HRP cluster weights toward high-posterior-Sharpe
       clusters while penalizing cluster-level turnover against `current_weights`.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        n_clusters: int | None = None,
        tau: float = 0.25,
        return_tilt_strength: float = 2.0,
        turnover_penalty: float = 0.5,
        linkage_method: Literal[
            "single", "complete", "average", "weighted", "centroid", "median", "ward"
        ] = "ward",
    ) -> None:
        super().__init__(tickers or [])
        self.n_clusters = n_clusters
        self.tau = tau
        self.return_tilt_strength = return_tilt_strength
        self.turnover_penalty = turnover_penalty
        self.linkage_method = linkage_method

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if not self.tickers and data.assets:
            self.tickers = list(data.assets)
        tickers = list(data.prices.columns)
        n_assets = len(tickers)

        returns_df = data.prices.pct_change().dropna(how="all").fillna(0.0)
        if cov_matrix is None:
            cov_df = returns_df.cov()
        else:
            cov_df = cov_matrix.reindex(index=tickers, columns=tickers).fillna(0.0)

        cov_np = cov_df.to_numpy(dtype=float) + np.eye(n_assets) * 1e-8
        corr = cov_to_corr(cov_np)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        condensed_dist = ssd.squareform(dist, checks=False)

        link = sch.linkage(condensed_dist, method=self.linkage_method)
        sort_ix = self._get_quasi_diag(link)

        # Base HRP weights via recursive bisection
        hrp_weights = pd.Series(1.0, index=tickers, dtype=float)
        hrp_weights = self._recursive_bisection(cov_np, sort_ix, hrp_weights)

        # Determine flat cluster assignments
        k_clusters = self.n_clusters or min(max(2, int(np.sqrt(n_assets))), n_assets)
        cluster_labels = sch.fcluster(link, t=k_clusters, criterion="maxclust")
        unique_clusters = sorted(int(c) for c in np.unique(cluster_labels))

        # Expected returns vector (sample mean or provided estimator)
        if expected_returns is not None:
            mu_vec = expected_returns.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        else:
            mu_vec = (
                returns_df.mean().reindex(tickers).fillna(0.0).to_numpy(dtype=float)
            )

        # Build intra-cluster normalized weight matrix W_intra (N x K)
        K = len(unique_clusters)
        W_intra = np.zeros((n_assets, K), dtype=float)
        base_cluster_budgets = np.zeros(K, dtype=float)

        for k_idx, cid in enumerate(unique_clusters):
            mask = cluster_labels == cid
            c_sum = float(hrp_weights.iloc[mask].sum())
            base_cluster_budgets[k_idx] = c_sum
            if c_sum > 0:
                W_intra[mask, k_idx] = hrp_weights.iloc[mask].to_numpy() / c_sum
            else:
                W_intra[mask, k_idx] = 1.0 / max(1, int(mask.sum()))

        # Cluster-level covariance Sigma_C (K x K) and sample cluster views mu_C (K,)
        Sigma_C = W_intra.T @ cov_np @ W_intra + np.eye(K) * 1e-8
        mu_C_view = W_intra.T @ mu_vec

        # Implied equilibrium prior pi_C = Sigma_C @ b_hrp
        pi_C = Sigma_C @ base_cluster_budgets

        # Bayesian Posterior Cluster Return: shrinkage between equilibrium prior pi_C and view mu_C_view
        post_mu_C = (1.0 - self.tau) * pi_C + self.tau * mu_C_view
        cluster_vols = np.sqrt(np.clip(np.diag(Sigma_C), 1e-8, None))
        posterior_ir = post_mu_C / cluster_vols

        # Current cluster budgets from pre-trade weights (if provided) for turnover damping
        if current_weights is not None:
            w_curr = current_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
            curr_cluster_budgets = np.array(
                [float(w_curr[cluster_labels == cid].sum()) for cid in unique_clusters],
                dtype=float,
            )
            c_tot = float(curr_cluster_budgets.sum())
            if c_tot > 0:
                curr_cluster_budgets /= c_tot
            else:
                curr_cluster_budgets = base_cluster_budgets.copy()
        else:
            curr_cluster_budgets = base_cluster_budgets.copy()

        # Multiplicative Bayesian tilt on cluster budgets with softmax normalization
        centered_ir = posterior_ir - float(np.mean(posterior_ir))
        scale_ir = float(np.std(posterior_ir)) + 1e-8
        norm_ir = np.clip(centered_ir / scale_ir, -3.0, 3.0)

        tilted_budgets = base_cluster_budgets * np.exp(
            self.return_tilt_strength * 0.25 * norm_ir
        )
        tilted_budgets /= float(np.sum(tilted_budgets))

        # Apply multi-period turnover damping toward current cluster budgets
        lam = float(
            np.clip(self.turnover_penalty / (1.0 + self.turnover_penalty), 0.0, 0.9)
        )
        final_cluster_budgets = (
            1.0 - lam
        ) * tilted_budgets + lam * curr_cluster_budgets
        final_cluster_budgets = np.clip(final_cluster_budgets, 0.0, None)
        final_cluster_budgets /= float(np.sum(final_cluster_budgets))

        # Reconstruct asset-level weights w = W_intra @ final_cluster_budgets
        final_w_np = W_intra @ final_cluster_budgets
        final_w_np = np.clip(final_w_np, 0.0, None)
        final_w_np /= float(np.sum(final_w_np))

        final_weights = pd.Series(final_w_np, index=tickers, name="weights")
        cluster_map = {
            str(asset): f"Cluster {int(c)}" for asset, c in zip(tickers, cluster_labels)
        }

        return OptimizationResult(
            weights=final_weights,
            status="OPTIMAL_BAYESIAN_HRP",
            metadata={
                "linkage_matrix": link,
                "sort_ix": sort_ix.tolist(),
                "ordered_assets": [str(tickers[i]) for i in sort_ix],
                "clusters": cluster_map,
                "base_cluster_budgets": base_cluster_budgets.tolist(),
                "posterior_cluster_budgets": final_cluster_budgets.tolist(),
                "posterior_cluster_returns": post_mu_C.tolist(),
            },
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

                alpha = 1.0 - var_left / (var_left + var_right)
                weights.iloc[cluster_left] *= alpha
                weights.iloc[cluster_right] *= 1.0 - alpha

        return weights

    def _get_cluster_var(self, cov: np.ndarray, cluster: list[int]) -> float:
        sub_cov = cov[np.ix_(cluster, cluster)]
        ivp = 1.0 / np.diag(sub_cov)
        ivp /= ivp.sum()
        return float(np.dot(ivp.T, np.dot(sub_cov, ivp)))
