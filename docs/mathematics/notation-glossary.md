# Notation & Symbol Glossary

This glossary standardizes mathematical notation across all PyAlloq documentation pages so that every symbol has a consistent meaning and dimension.

---

## Dimensions & Indices

| Symbol | Meaning |
|--------|---------|
| \(N\) | Number of assets in the investment universe (or \(N_{\text{active}}\) for active tradable assets in ragged universes) |
| \(T\) | Number of historical time periods (or simulation horizon steps) |
| \(M\) | Number of Monte Carlo scenario paths in `ScenarioMarketData` |
| \(H\) | Multi-period planning or forecast horizon (steps) |
| \(L\) | Lookback window length for rolling estimators, datasets, and neural sequence encoders |
| \(F\) | Number of input feature channels per asset in `MarketDataset` (`X` has shape \((L, N, F)\)) |
| \(K\) | Number of clusters, regimes, views, factors, or child strategies in an ensemble |
| \(m\) | Number of inducing points in `ISAB` (`SetTransformerBackbone`) or scenario index \(m \in \{1, \dots, M\}\) |
| \(A\) | Annualization factor (`DataFrequency`: \(252\) daily, \(52\) weekly, \(12\) monthly, \(365\) crypto) |

---

## Returns, Prices & Risk Parameters

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| \(P_t\) | \(\mathbb{R}_{++}^N\) | Asset price vector at time \(t\) |
| \(r_t\) | \(\mathbb{R}^N\) | Simple asset return vector over \((t-1, t]\) |
| \(\mu, \, \hat{\mu}\) | \(\mathbb{R}^N\) | True / estimated annualized expected return vector |
| \(\Sigma, \, \hat{\Sigma}\) | \(\mathbb{S}_{++}^{N \times N}\) | True / estimated annualized positive-definite covariance matrix |
| \(C, \, \Gamma\) | \([-1, 1]^{N \times N}\) | Correlation matrix (\(C_{ij} = \Sigma_{ij} / (\sigma_i \sigma_j)\)) |
| \(\sigma_i\) | \(\mathbb{R}_+\) | Individual asset volatility (\(\sqrt{\Sigma_{ii}}\)) |
| \(\sigma_p\) | \(\mathbb{R}_+\) | Portfolio volatility (\(\sqrt{w^\top \Sigma w}\)) |
| \(r_f\) | \(\mathbb{R}\) | Risk-free rate |
| \(\gamma, \, \lambda\) | \(\mathbb{R}_+\) | Risk-aversion parameter (context-specific; \(\lambda\) also denotes eigenvalues, jump penalties, or Lagrange multipliers) |

---

## Portfolio Weights & Execution

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| \(\Delta^N\) | \(\subset \mathbb{R}^N\) | Standard probability simplex \(\{w \in \mathbb{R}^N : w \ge 0, \, \mathbf{1}^\top w = 1\}\) |
| \(w_t\) | \(\mathbb{R}^N\) | Post-trade target portfolio weight vector at time \(t\) |
| \(w_{t^-}\) | \(\mathbb{R}^N\) | Pre-trade drifted portfolio weight vector entering rebalance date \(t\) |
| \(z_t = \Delta w_t\) | \(\mathbb{R}^N\) | Trade vector \(w_t - w_{t^-}\) |
| \(\phi_{\text{trade}}(z_t)\) | \(\mathbb{R}_+\) | Realized transaction cost / market impact drag |
| \(r_{p,t}^{\text{net}}\) | \(\mathbb{R}\) | Net portfolio return after transaction costs: \((1 - \phi_{\text{trade}}(z_t))(1 + w_t^\top r_t) - 1\) |
| \(\alpha_t\) | \(\Delta^K\) | Meta-portfolio weights over \(K\) child allocators in `EnsembleAllocator` / `MetaPortfolioEnv` |

---

## Graph, Spectral & Regime Symbols

| Symbol | Meaning |
|--------|---------|
| \(W\) | Non-negative symmetric graph adjacency matrix |
| \(D\) | Diagonal degree matrix \(D_{ii} = \sum_j W_{ij}\) (or diagonal volatility matrix \(D_\sigma = \operatorname{diag}(\sigma)\)) |
| \(\mathcal{L}\) | Normalized Graph Laplacian \(\mathcal{L} = I - D^{-1/2} W D^{-1/2}\) |
| \(\lambda_2, \, v_2\) | Fiedler algebraic connectivity (second smallest eigenvalue of \(\mathcal{L}\)) and Fiedler eigenvector |
| \(G, \, \rho(G)\) | Hawkes excitation branching matrix and its spectral radius \(\max_i |\lambda_i(G)|\) |
| \(p_t\) | Regime belief / probability vector \(p_t \in \Delta^K\) |
| \(W_2\) | Bures-Wasserstein 2-distance between fast and slow Gaussian market baths |
| \(\mathcal{S}_{\text{vN}}(\Sigma)\) | Normalized Von Neumann covariance entropy \(-\frac{1}{\ln N} \sum_i p_i \ln p_i\) |

---

## Operators & Conventions

- \(\mathbf{1}\): all-ones column vector of conformal dimension.
- \(\circ\) or \(\odot\): elementwise (Hadamard) product; \(\oslash\): elementwise division.
- \([x]_+ = \max(x, 0)\): positive part.
- \(\|x\|_1, \|x\|_2, \|x\|_\infty, \|A\|_F\): \(\ell_1\), Euclidean \(\ell_2\), infinity, and Frobenius norms.
- \(\operatorname{Tr}(\cdot)\), \(\operatorname{diag}(\cdot)\): matrix trace and diagonal operator.
- \(\Pi_{\mathcal{C}}(\cdot)\): Euclidean projection onto convex set \(\mathcal{C}\).
