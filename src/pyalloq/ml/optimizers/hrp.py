from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from pyalloq_core.utils import cov_to_corr

ClusteringMode = Literal["single_linkage", "spectral_laplacian"]


class HRPAllocator(BaseAllocator):
    """
    Hierarchical Risk Parity (HRP) Portfolio Allocator (López de Prado 2016),
    extended with Spectral Graph Laplacian Fiedler Cuts (`clustering_mode="spectral_laplacian"`).

    Modes:
    - `"single_linkage"` (default): Classical agglomerative clustering on correlation distance
      `d_{ij} = sqrt(0.5 * (1 - rho_{ij}))` followed by quasi-diagonalization and recursive bisection.
    - `"spectral_laplacian"`: Graph-theoretic recursive spectral partitioning using the second
      smallest eigenvector (Fiedler vector `v_2`) of the normalized Graph Laplacian `L = I - D^{-1/2} W D^{-1/2}`,
      with optional fragility-aware bisection penalizing sub-graphs with high internal algebraic
      connectivity `lambda_2(L_S)`.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        clustering_mode: ClusteringMode = "single_linkage",
        fragility_penalty: float = 0.0,
        fragility_tilt: float | None = None,
        use_precision_graph: bool = True,
    ) -> None:
        super().__init__(tickers or [])
        self.clustering_mode: ClusteringMode = clustering_mode
        self.fragility_penalty = float(
            fragility_tilt if fragility_tilt is not None else fragility_penalty
        )
        self.use_precision_graph = bool(use_precision_graph)

    @staticmethod
    def _build_adjacency_from_cov(
        cov: np.ndarray,
        use_precision: bool = True,
    ) -> np.ndarray:
        n = cov.shape[0]
        if n <= 1:
            return np.zeros((n, n), dtype=np.float64)

        max_diag = float(np.max(np.abs(np.diag(cov))))
        cov_unit = cov / max_diag if max_diag > 1e-12 else cov
        corr = cov_to_corr(cov_unit)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        if use_precision:
            theta = np.linalg.inv(corr + 1e-3 * np.eye(n, dtype=np.float64))
            d_th = np.sqrt(np.clip(np.diag(theta), 1e-10, None))
            pcorr = -theta / np.outer(d_th, d_th)
            adj = np.maximum(0.0, pcorr)
        else:
            adj = np.maximum(0.0, corr)

        # Fallback to positive correlation if partial correlation graph is completely disconnected
        if float(np.sum(adj)) < 1e-8:
            adj = np.maximum(0.0, corr)
        np.fill_diagonal(adj, 0.0)
        return adj

    @staticmethod
    def _subgraph_laplacian_and_fiedler(
        adjacency: np.ndarray,
        nodes: list[int],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Computes the normalized Graph Laplacian L_S, Fiedler value lambda_2(L_S),
        and Fiedler vector v_2(L_S) for the induced subgraph on `nodes`.
        """
        k = len(nodes)
        if k <= 1:
            return (
                np.zeros((k, k), dtype=np.float64),
                0.0,
                np.ones(k, dtype=np.float64),
            )

        sub_w = adjacency[np.ix_(nodes, nodes)].astype(np.float64, copy=True)
        np.fill_diagonal(sub_w, 0.0)
        deg = np.sum(sub_w, axis=1)
        deg_inv_sqrt = np.where(deg > 1e-10, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
        norm_w = sub_w * np.outer(deg_inv_sqrt, deg_inv_sqrt)
        lap = np.eye(k, dtype=np.float64) - norm_w
        lap = 0.5 * (lap + lap.T)

        eigvals, _ = np.linalg.eigh(lap)
        order = np.argsort(eigvals)
        fiedler_val = float(max(0.0, eigvals[order[1]]))

        # Compute Fiedler vector on a mildly regularized connected graph to prevent
        # zero-eigenvalue multiplicity rotations when subgraphs are disconnected
        idx_grid = np.arange(1, k + 1, dtype=np.float64)
        tie_break = 1e-4 * (1.0 + 0.1 * np.add.outer(idx_grid, idx_grid) / float(k))
        np.fill_diagonal(tie_break, 0.0)
        reg_w = sub_w + tie_break
        reg_deg = np.sum(reg_w, axis=1)
        reg_inv = 1.0 / np.sqrt(np.maximum(reg_deg, 1e-10))
        reg_lap = np.eye(k, dtype=np.float64) - reg_w * np.outer(reg_inv, reg_inv)
        reg_lap = 0.5 * (reg_lap + reg_lap.T)
        r_vals, r_vecs = np.linalg.eigh(reg_lap)
        r_order = np.argsort(r_vals)
        fiedler_vec = r_vecs[:, r_order[1]].astype(np.float64)

        if float(np.dot(fiedler_vec, idx_grid)) < 0.0:
            fiedler_vec = -fiedler_vec
        return lap, fiedler_val, fiedler_vec

    @classmethod
    def _spectral_sweep_partition(
        cls,
        adjacency: np.ndarray,
        nodes: list[int],
    ) -> tuple[list[int], list[int], float]:
        """
        Partitions `nodes` into `(left_nodes, right_nodes)` along the Fiedler vector ordering
        by minimizing the Cheeger normalized cut conductance, with balanced-cut safeguard.
        """
        k = len(nodes)
        if k == 2:
            return [nodes[0]], [nodes[1]], 0.0

        _, fiedler_val, fiedler_vec = cls._subgraph_laplacian_and_fiedler(
            adjacency, nodes
        )
        local_order = np.argsort(fiedler_vec)
        ordered_nodes = [nodes[int(i)] for i in local_order]

        sub_w = adjacency[np.ix_(ordered_nodes, ordered_nodes)]
        total_deg = np.sum(sub_w, axis=1)
        total_vol = float(np.sum(total_deg))

        if total_vol <= 1e-10:
            mid = k // 2
            return ordered_nodes[:mid], ordered_nodes[mid:], fiedler_val

        # Search interior split points [1 .. k-1] minimizing conductance + mild balance regularizer
        best_split = k // 2
        best_score = float("inf")
        for s in range(1, k):
            left_slice = slice(0, s)
            right_slice = slice(s, k)
            cut_weight = float(np.sum(sub_w[left_slice, right_slice]))
            vol_left = float(np.sum(total_deg[left_slice]))
            vol_right = float(np.sum(total_deg[right_slice]))
            denom = min(vol_left, vol_right)
            conductance = cut_weight / denom if denom > 1e-10 else cut_weight
            balance_pen = 0.15 * abs(s - 0.5 * k) / float(k)
            score = conductance + balance_pen
            if score < best_score:
                best_score = score
                best_split = s

        return (
            ordered_nodes[:best_split],
            ordered_nodes[best_split:],
            fiedler_val,
        )

    def _spectral_recursive_bisection(
        self,
        cov: np.ndarray,
        adjacency: np.ndarray,
        columns: pd.Index,
    ) -> tuple[pd.Series, list[int], dict[str, str], float]:
        n_assets = len(columns)
        weights_arr = np.ones(n_assets, dtype=np.float64)
        root_nodes = list(range(n_assets))
        _, root_fiedler, _ = self._subgraph_laplacian_and_fiedler(adjacency, root_nodes)

        # Build recursive Fiedler tree to simultaneously obtain topological ordering & weights
        def _recurse(
            nodes: list[int], depth: int, prefix: str
        ) -> tuple[list[int], dict[str, str]]:
            if len(nodes) <= 1:
                cmap = {str(columns[nodes[0]]): prefix or "Cluster 1"} if nodes else {}
                return list(nodes), cmap

            left_nodes, right_nodes, _ = self._spectral_sweep_partition(
                adjacency, nodes
            )
            var_left = self._get_cluster_var(cov, left_nodes)
            var_right = self._get_cluster_var(cov, right_nodes)

            if self.fragility_penalty > 0.0:
                _, f_left, _ = self._subgraph_laplacian_and_fiedler(
                    adjacency, left_nodes
                )
                _, f_right, _ = self._subgraph_laplacian_and_fiedler(
                    adjacency, right_nodes
                )
                var_left = var_left * (1.0 + self.fragility_penalty * f_left)
                var_right = var_right * (1.0 + self.fragility_penalty * f_right)

            alpha = 1.0 - var_left / max(var_left + var_right, 1e-12)
            weights_arr[left_nodes] *= alpha
            weights_arr[right_nodes] *= 1.0 - alpha

            c_left = f"{prefix}L" if depth < 2 else prefix
            c_right = f"{prefix}R" if depth < 2 else prefix
            ord_left, map_left = _recurse(left_nodes, depth + 1, c_left)
            ord_right, map_right = _recurse(right_nodes, depth + 1, c_right)
            merged_map = {**map_left, **map_right}
            return ord_left + ord_right, merged_map

        sort_ix, raw_clusters = _recurse(root_nodes, depth=0, prefix="")
        weights = pd.Series(weights_arr, index=columns, dtype=np.float64)
        # Normalize cluster names to "Cluster 1", "Cluster 2", ...
        unique_tags = list(dict.fromkeys(raw_clusters.values()))
        tag_to_label = {tag: f"Cluster {i + 1}" for i, tag in enumerate(unique_tags)}
        cluster_map = {asset: tag_to_label[tag] for asset, tag in raw_clusters.items()}
        return weights, sort_ix, cluster_map, root_fiedler

    def allocate(
        self,
        data: MarketData | None = None,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if cov_matrix is None and isinstance(data, pd.DataFrame):
            cov_matrix = data
            data = None
        if cov_matrix is None:
            raise ValueError("HRPAllocator requires cov_matrix")

        mode: ClusteringMode = kwargs.get("clustering_mode", self.clustering_mode)
        cov_np = cov_matrix.to_numpy(dtype=np.float64)

        if mode == "spectral_laplacian":
            adj_override = kwargs.get("adjacency_matrix")
            if adj_override is not None:
                adjacency = np.asarray(adj_override, dtype=np.float64)
            else:
                adjacency = self._build_adjacency_from_cov(
                    cov_np, use_precision=self.use_precision_graph
                )

            weights, sort_ix_list, cluster_map, fiedler_val = (
                self._spectral_recursive_bisection(
                    cov=cov_np,
                    adjacency=adjacency,
                    columns=cov_matrix.columns,
                )
            )
            return OptimizationResult(
                weights=weights,
                status="OPTIMAL_SPECTRAL_HRP",
                metadata={
                    "clustering_mode": "spectral_laplacian",
                    "fiedler_value": fiedler_val,
                    "sort_ix": sort_ix_list,
                    "ordered_assets": [
                        str(cov_matrix.columns[i]) for i in sort_ix_list
                    ],
                    "clusters": cluster_map,
                },
            )

        corr = cov_to_corr(cov_np)
        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0.0, 1.0))

        condensed_dist = ssd.squareform(dist, checks=False)
        link = sch.linkage(condensed_dist, method="single")
        sort_ix = self._get_quasi_diag(link)

        weights = pd.Series(1.0, index=cov_matrix.columns)
        weights = self._recursive_bisection(cov_np, sort_ix, weights)

        final_weights = pd.Series(0.0, index=cov_matrix.columns)
        for idx in sort_ix:
            final_weights.iloc[idx] = weights.iloc[idx]

        # Extract flat cluster assignment (default 2-4 clusters based on tree depth) for visualization
        n_assets = len(cov_matrix.columns)
        n_clust = min(max(2, int(np.sqrt(n_assets))), n_assets)
        cluster_labels = sch.fcluster(link, t=n_clust, criterion="maxclust")
        cluster_map = {
            str(asset): f"Cluster {int(c)}"
            for asset, c in zip(cov_matrix.columns, cluster_labels)
        }

        return OptimizationResult(
            weights=final_weights,
            status="OPTIMAL_HRP",
            metadata={
                "clustering_mode": "single_linkage",
                "linkage_matrix": link,
                "sort_ix": sort_ix.tolist(),
                "ordered_assets": [str(cov_matrix.columns[i]) for i in sort_ix],
                "clusters": cluster_map,
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

        return float(np.dot(ivp.T, np.dot(sub_cov, ivp)))


class SpectralLaplacianHRPAllocator(HRPAllocator):
    """
    Graph-Theoretic Hierarchical Risk Parity Allocator using Recursive Fiedler Vector
    Spectral Cuts on the Financial Graph Laplacian L = I - D^{-1/2} W D^{-1/2}.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        fragility_penalty: float = 0.25,
        fragility_tilt: float | None = None,
        use_precision_graph: bool = True,
    ) -> None:
        super().__init__(
            tickers=tickers,
            clustering_mode="spectral_laplacian",
            fragility_penalty=(
                fragility_tilt if fragility_tilt is not None else fragility_penalty
            ),
            use_precision_graph=use_precision_graph,
        )
