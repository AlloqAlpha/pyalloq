# Graph-Based Allocator Mathematics

This page covers the mathematical foundations of the hierarchical clustering-based allocators: HRP, HERC, and NCO.

These methods avoid inverting the covariance matrix, making them more numerically stable for large universes and ill-conditioned matrices.

---

## Hierarchical Risk Parity (HRP)

López de Prado (2016). HRP uses hierarchical clustering on the correlation matrix to build a tree of related assets, then allocates weights top-down through the tree using inverse-variance weighting.

### Step 1: Distance Matrix

Convert the correlation matrix \(\rho\) into a metric distance matrix:

\[
d_{ij} = \sqrt{\frac{1 - \rho_{ij}}{2}}
\]

Properties: \(d_{ii} = 0\), \(d_{ij} = d_{ji}\), \(0 \leq d_{ij} \leq 1\).

### Step 2: Hierarchical Clustering (Dendrogram)

Apply single-linkage agglomerative clustering to the distance matrix to build a binary tree (dendrogram). At each merge step, the distance between two clusters is:

\[
d(C_1, C_2) = \min_{i \in C_1, j \in C_2} d_{ij}
\]

### Step 3: Quasi-Diagonalization

Re-order the assets (leaves of the dendrogram) such that similar assets are adjacent. The resulting reordering makes the covariance matrix approximately block-diagonal.

### Step 4: Recursive Bisection

Traverse the dendrogram top-down. At each level, split the current cluster into two sub-clusters and allocate capital inversely proportional to their sub-portfolio variances:

\[
w_{left} \leftarrow w_{left} \times \alpha, \quad w_{right} \leftarrow w_{right} \times (1 - \alpha)
\]

where the allocation split \(\alpha\) is:

\[
\alpha = 1 - \frac{\tilde{\sigma}^2_{left}}{\tilde{\sigma}^2_{left} + \tilde{\sigma}^2_{right}}
\]

and \(\tilde{\sigma}^2_C\) is the variance of the **inverse-variance portfolio** of cluster \(C\):

\[
\tilde{w}_i = \frac{1/\sigma^2_i}{\sum_{j \in C} 1/\sigma^2_j}, \quad \tilde{\sigma}^2_C = \tilde{w}^\top_C \Sigma_C \tilde{w}_C
\]

### Spectral Graph Laplacian HRP (`SpectralLaplacianHRPAllocator`)

Instead of single-linkage agglomerative clustering on pairwise correlation distances (which is vulnerable to "chaining" noise), **Spectral Graph Laplacian HRP** constructs a non-negative precision/partial-correlation adjacency matrix \(W_S\) on any induced subgraph \(S \subseteq V\) and computes the normalized Graph Laplacian \(\mathcal{L}_S = I_{|S|} - D_S^{-1/2} W_S D_S^{-1/2}\). Using the **Fiedler vector** \(v_2(\mathcal{L}_S)\) (the eigenvector of the second smallest eigenvalue \(\lambda_2(\mathcal{L}_S)\)):

1. **Cheeger Conductance Sweep Cut:** Sort nodes \(i \in S\) by their Fiedler coordinate \(v_{2,i}(\mathcal{L}_S)\) and select the split index \(s^*\) minimizing the regularized Cheeger normalized cut conductance \(\phi(S_L, S_R)\).
2. **Fragility-Tilted Recursive Bisection:** Modify each sub-cluster's effective IVP variance by its internal algebraic connectivity \(\lambda_2(\mathcal{L}_{S_c})\):

\[
\tilde{V}_{S_c} = \tilde{\sigma}^2_{S_c} \left(1 + \eta_{\text{frag}} \, \lambda_2(\mathcal{L}_{S_c})\right), \qquad \alpha_L = 1 - \frac{\tilde{V}_{S_L}}{\tilde{V}_{S_L} + \tilde{V}_{S_R}}
\]

---

## Differentiable Risk Parity KKT Layer (`DifferentiableRiskParityLayer`)

By Spinu's (2013) convex log-barrier formulation of Risk Budgeting / Risk Parity:

\[
y^*(\hat{\Sigma}, b) = \arg\min_{y > 0} \frac{1}{2} y^\top \hat{\Sigma} y - \sum_{i=1}^N b_i \ln y_i, \qquad w^* = \frac{y^*}{\mathbf{1}^\top y^*}
\]

the KKT stationarity condition \(F(y^*, \hat{\Sigma}, b) = \hat{\Sigma} y^* - b / y^* = \mathbf{0}\) has an **exact closed-form positive-definite Jacobian** \(J_y = \hat{\Sigma} + \operatorname{diag}(b / (y^*)^2)\), enabling exact implicit differentiation without unrolling Newton iterations:

\[
\frac{\partial \mathcal{L}_{\text{task}}}{\partial \hat{\Sigma}} = -\frac{1}{2}\left(u (y^*)^\top + y^* u^\top\right), \qquad \text{where } \left(\hat{\Sigma} + \operatorname{diag}\!\left(\frac{b}{(y^*)^2}\right)\right) u = \frac{\partial \mathcal{L}_{\text{task}}}{\partial y^*}
\]

---

## Hierarchical Equal Risk Contribution (HERC)

HERC extends HRP by replacing the inverse-variance cluster weights at each bisection step with **equal risk contribution weights** within each cluster. This enforces risk parity not just at the asset level but at every level of the hierarchy.

At each cluster node, the intra-cluster weights are the ERC solution:

\[
\min_{y_C > 0} \; \frac{1}{2} y_C^\top \Sigma_C y_C - \sum_{i \in C} \ln y_{C,i}
\]

The inter-cluster allocation still follows the same recursive bisection as HRP, but using the ERC cluster variance instead of the IVP cluster variance.

---

## Nested Cluster Optimization (NCO)

López de Prado (2019). NCO applies two-level Markowitz optimization:

### Step 1: Cluster Detection

Apply k-means clustering directly to the correlation distance matrix \(d_{ij} = \sqrt{\frac{1}{2}(1 - \rho_{ij})}\) to partition assets into \(K\) clusters.

### Step 2: Intra-Cluster Optimization

Run a **within-cluster** minimum-variance optimization for each cluster \(C_k\) independently:

\[
w^{\text{intra}}_{C_k} = \arg\min_{w \ge 0,\; \mathbf{1}^\top w = 1} w^\top \Sigma_{C_k} w
\]

This produces one synthetic "cluster portfolio" per cluster with variance \(\sigma_{C_k}^2 = (w^{\text{intra}}_{C_k})^\top \Sigma_{C_k} w^{\text{intra}}_{C_k}\).

### Step 3: Inter-Cluster Optimization

Construct the reduced \(K \times K\) cluster covariance matrix (`NCOAllocator` uses the diagonalized cluster variance matrix \(\Sigma^{\text{cluster}} = \operatorname{diag}(\sigma_{C_1}^2, \dots, \sigma_{C_K}^2)\)) and run a second minimum-variance optimization across cluster representatives:

\[
w^{\text{inter}} = \arg\min_{u \ge 0,\; \mathbf{1}^\top u = 1} u^\top \Sigma^{\text{cluster}} u
\]

### Step 4: Combine

The final weight for asset \(i\) in cluster \(C_k\) is:

\[
w_i = w^{\text{intra}}_{i \in C_k} \times w^{\text{inter}}_{C_k}
\]

---

## Comparison

| Property | HRP | HERC | NCO |
|----------|-----|------|-----|
| Matrix inversion | ❌ | ❌ | ✅ (within clusters only) |
| Risk budget control | Implicit (IVP) | Explicit (ERC) | Implicit (MinVol) |
| Cluster method | Hierarchical linkage | Hierarchical linkage | K-Means |
| Complexity | Low | Medium | Higher |
| Large universe (100+) | ✅ | ✅ | ✅ |

---

## References

- [López de Prado (2016) — Building Diversified Portfolios that Outperform Out-of-Sample](../references/papers.md)
- [Raffinot (2018) — Hierarchical Clustering-Based Asset Allocation](../references/papers.md)
