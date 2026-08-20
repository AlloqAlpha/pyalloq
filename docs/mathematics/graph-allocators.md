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

---

## Hierarchical Equal Risk Contribution (HERC)

HERC extends HRP by replacing the inverse-variance cluster weights at each bisection step with **equal risk contribution weights** within each cluster. This enforces risk parity not just at the asset level but at every level of the hierarchy.

At each cluster node, the intra-cluster weights are the ERC solution:

\[
\min_{y_C \geq 0} \; \frac{1}{2} y_C^\top \Sigma_C y_C - \sum_{i \in C} \ln y_{C,i}
\]

The inter-cluster allocation still follows the same recursive bisection as HRP, but using the ERC cluster variance instead of the IVP cluster variance.

---

## Nested Cluster Optimization (NCO)

López de Prado (2019). NCO applies two-level Markowitz optimization:

### Step 1: Cluster Detection

Apply k-means clustering to the correlation matrix (using the first \(k\) principal components as cluster features) to partition assets into \(K\) clusters.

### Step 2: Intra-Cluster Optimization

Run a **within-cluster** Markowitz optimization for each cluster \(C_k\) independently:

\[
w^{intra}_{C_k} = \arg\min_{w \geq 0} w^\top \Sigma_{C_k} w
\]

This produces one synthetic "cluster portfolio" per cluster.

### Step 3: Inter-Cluster Optimization

Build a reduced \(K \times K\) covariance matrix from the cluster portfolio returns, and run a second Markowitz optimization across cluster representatives:

\[
w^{inter} = \arg\min_{u \geq 0} u^\top \Sigma^{cluster} u
\]

### Step 4: Combine

The final weight for asset \(i\) in cluster \(C_k\) is:

\[
w_i = w^{intra}_{i \in C_k} \times w^{inter}_{C_k}
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
