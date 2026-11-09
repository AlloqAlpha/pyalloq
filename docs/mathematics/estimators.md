# Estimator Mathematics

This page covers the mathematical foundations of the return and covariance estimators implemented in `pyalloq`.

---

## Return Estimators

### Exponentially Weighted Mean (EWMA)

Weights observations exponentially so that recent returns matter more:

\[
\hat{\mu}^{EWMA}_i = \frac{\sum_{t=1}^{T} \lambda^{T-t} r_{i,t}}{\sum_{t=1}^{T} \lambda^{T-t}}
\]

where the decay factor \(\lambda = 1 - \frac{2}{\text{span}+1}\).

---

### James-Stein Shrinkage

Shrinks each asset's historical mean towards the grand mean of all assets. For \(N \geq 3\) assets, the James-Stein estimator strictly dominates the sample mean in terms of MSE:

\[
\hat{\mu}^{JS} = (1 - w^*) \hat{\mu} + w^* \bar{\mu} \mathbf{1}
\]

The optimal shrinkage intensity \(w^*\) is:

\[
w^* = \min\left(1,\; \frac{N-2}{(\hat{\mu} - \bar{\mu}\mathbf{1})^\top \hat{\Sigma}^{-1} (\hat{\mu} - \bar{\mu}\mathbf{1})}\right)
\]

where \(\bar{\mu} = \frac{1}{N}\sum_i \hat{\mu}_i\) is the grand mean.

---

### Implied Returns (Reverse Optimization)

Given observed market-capitalization weights \(w^{mkt}\) and the market risk aversion \(\lambda\), the implied equilibrium excess returns are:

\[
\Pi = \lambda \Sigma w^{mkt}
\]

This is the equilibrium prior used in Black-Litterman.

---

### Black-Litterman

Combines the market equilibrium prior \(\Pi\) with investor views \((P, Q)\) through Bayesian updating.

**Views:** \(P \in \mathbb{R}^{K \times N}\) is a view-picking matrix, \(Q \in \mathbb{R}^K\) are the expected view returns, and \(\Omega\) is the uncertainty matrix for the views.

**Posterior expected returns:**

\[
\mu^* = \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
       \left[(\tau\Sigma)^{-1} \Pi + P^\top \Omega^{-1} Q\right]
\]

**Posterior covariance:**

\[
\Sigma^* = \Sigma + \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
\]

where \(\tau\) is a scalar controlling the uncertainty in the prior (typically 0.01–0.05).

---

## Covariance Estimators

### Ledoit-Wolf Shrinkage

Shrinks the sample covariance \(\hat{\Sigma}\) towards a structured target \(T\) (e.g., diagonal matrix):

\[
\hat{\Sigma}^{LW} = (1 - \alpha^*) \hat{\Sigma} + \alpha^* \hat{\mu}_{cov} I
\]

The optimal shrinkage intensity \(\alpha^*\) is derived analytically by minimizing the Frobenius norm of the estimation error:

\[
\alpha^* = \frac{\delta^2}{\gamma^2}
\]

where \(\delta^2\) and \(\gamma^2\) are data-dependent scaling constants derived from the Oracle Approximating Shrinkage (OAS) method.

See: [Ledoit & Wolf (2004)](../references/papers.md#ledoit-wolf-2004)

---

### Random Matrix Theory (Marchenko-Pastur)

For a random matrix \(X \in \mathbb{R}^{T \times N}\) with i.i.d. entries, the eigenvalue distribution of \(\hat{\Sigma} = \frac{1}{T} X^\top X\) converges to the Marchenko-Pastur distribution with upper bound:

\[
\lambda_+ = \sigma^2 \left(1 + \sqrt{\frac{N}{T}}\right)^2
\]

Any empirical eigenvalue \(\lambda_i < \lambda_+\) is attributable to noise, not signal. The denoising step replaces these with the average noise eigenvalue:

\[
\bar{\lambda}_{noise} = \frac{\sum_{i: \lambda_i < \lambda_+} \lambda_i}{|\{i: \lambda_i < \lambda_+\}|}
\]

Then the clean covariance is reconstructed: \(\hat{\Sigma}^{RMT} = V \Lambda_{clean} V^\top\)

The optional **detoning** step removes the largest eigenvalue (corresponding to the market-wide factor) from the correlation matrix before reconstruction, isolating sector-level correlations.

---

### Semi-Covariance

Uses only negative return periods to estimate covariance, focusing on downside co-movement:

\[
\hat{\Sigma}^{semi}_{ij} = \frac{1}{T^-} \sum_{t: r_{i,t} < 0} r_{i,t} \cdot r_{j,t}
\]

where \(T^-\) is the count of periods where asset \(i\) had a negative return.

---

### EWMA Covariance

Uses exponential weighting to capture time-varying covariance:

\[
\hat{\Sigma}_t = (1 - \lambda) r_t r_t^\top + \lambda \hat{\Sigma}_{t-1}
\]

where \(\lambda = 1 - \frac{2}{\text{span}+1}\) is the decay factor. This models volatility clustering observed in financial time series (similar to the RiskMetrics model).

---

### Point-in-Time Model-Free Implied Volatility & Buss-Vilkov Implied Covariance

#### 1. Model-Free Implied Volatility (`PITOptionsMFIVTransformer`)
Following Britten-Jones & Neuberger (2000) and the CBOE VIX methodology, the risk-neutral integrated variance over horizon \(\tau\) is extracted without assuming Black-Scholes dynamics by integrating out-of-the-money (OTM) option mid-prices \(Q_t(K_k, \tau)\) across strikes \(K_k\) strictly stamped prior to the decision cut-off (\(\text{quote\_timestamp} \le t\)):

\[
\sigma_{\text{MFIV}, i}^2(t, \tau) = \frac{2 e^{r \tau}}{\tau} \sum_{k} \frac{\Delta K_k}{K_k^2} Q_t(K_k, \tau) - \frac{1}{\tau} \left( \frac{F_{t,\tau}}{K_0} - 1 \right)^2
\]

#### 2. Buss & Vilkov (2012) Option-Implied Correlation (`ImpliedCovarianceEstimator`)
Given the diagonal matrix of forward-looking implied volatilities \(D_{\text{IV}, t} = \operatorname{diag}(\sigma_{1,t}^{\text{IV}}, \dots, \sigma_{N,t}^{\text{IV}})\) and the Ledoit-Wolf shrunk physical correlation matrix \(\Gamma_t^{\text{hist}}\), the implied correlation matrix is parameterized as a convex shift toward the equicorrelation target \(\mathbf{1}\mathbf{1}^\top\):

\[
\Gamma_t^{\text{imp}} = (1 - \gamma_t) \Gamma_t^{\text{hist}} + \gamma_t \mathbf{1}\mathbf{1}^\top, \qquad \hat{\Sigma}_t^{\text{imp}} = D_{\text{IV}, t} \, \Gamma_t^{\text{imp}} \, D_{\text{IV}, t}
\]

When the index implied volatility \(\sigma_{M,t}^{\text{IV}}\) for market weights \(w_M \in \Delta^N\) is observed, \(\gamma_t\) is solved in closed form:

\[
\gamma_t = \frac{\left(\sigma_{M,t}^{\text{IV}}\right)^2 - w_M^\top D_{\text{IV}, t} \Gamma_t^{\text{hist}} D_{\text{IV}, t} w_M}{\left(w_M^\top \sigma_t^{\text{IV}}\right)^2 - w_M^\top D_{\text{IV}, t} \Gamma_t^{\text{hist}} D_{\text{IV}, t} w_M}
\]

---

### Graph Reaction-Diffusion Covariance (`ReactionDiffusionCovarianceEstimator`)

Let \(\Theta_t = \left(\Gamma_t^{\text{LW}} + \varepsilon I_N\right)^{-1}\) denote the regularized precision matrix. The non-negative partial correlation adjacency matrix \(W_t\) and symmetric normalized Graph Laplacian \(\mathcal{L}_t\) are defined by:

\[
W_{ij, t} = \max\!\left(0, -\frac{\Theta_{ij, t}}{\sqrt{\Theta_{ii, t} \Theta_{jj, t}}}\right) \quad (i \neq j), \qquad \mathcal{L}_t = I_N - D_t^{-1/2} W_t D_t^{-1/2}
\]

The normalized log-volatility excitation field \(v_t \in \mathbb{R}^N\) and slow liquidity inhibitor \(u_t \in \mathbb{R}^N\) evolve over horizon \(H\) via the coupled FitzHugh-Nagumo Graph Reaction-Diffusion system:

\[
\frac{dv}{dt} = -D_v \mathcal{L}_t v - \kappa_v v + \gamma_{\text{react}} \, v (v - \theta_{\text{crit}})(1 - v) - \beta_u u
\]

\[
\frac{du}{dt} = -D_u \mathcal{L}_t u + \epsilon_u (v - 0.8 u)
\]

Integrating \((v_t, u_t)\) to \(t = H\) via 4th-order Runge-Kutta (RK4) yields the forward volatility vector \(\hat{\sigma}_i(H) = \bar{\sigma}_i \exp(v_i(H))\) and contagion-tightened correlation matrix \(\hat{\Gamma}_H\), forming \(\hat{\Sigma}_{t+H} = \operatorname{diag}(\hat{\sigma}(H)) \, \hat{\Gamma}_H \, \operatorname{diag}(\hat{\sigma}(H))\).

---

### Spectral Graph Fragility & Wiener-Hopf Multivariate Hawkes Process (`SpectralGraphFragility`, `WienerHopfHawkesSolver`)

1. **Spectral Algebraic Connectivity (Fiedler Value \(\lambda_2\)):**
   Let \(0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N \le 2\) denote the eigenvalues of the symmetric normalized Graph Laplacian \(\mathcal{L}_t = I_N - D_t^{-1/2} W_t D_t^{-1/2}\). The **Fiedler value** \(\lambda_2(\mathcal{L}_t)\) measures algebraic connectivity and systemic cascade propensity, while \(\tau_{\text{mix}} = \lambda_2(\mathcal{L}_t)^{-1}\) defines the network mixing time and \(v_2 \in \mathbb{R}^N\) provides the optimal spectral bisection coordinate.

2. **Non-Parametric Wiener-Hopf Multivariate Hawkes Excitation (`WienerHopfHawkesSolver`):**
   For the \(N\)-dimensional self-exciting jump point process \(\boldsymbol{\lambda}_t = \boldsymbol{\mu} + \int_0^t \boldsymbol{\Phi}(t - s) \, d\mathbf{N}_s\), the causal excitation kernels \(\Phi_{ij}(\tau)\) satisfy the coupled Wiener-Hopf integral system:

\[
C_{ij}(\tau) = \Lambda_j \Phi_{ij}(\tau) + \sum_{k=1}^N \int_0^\infty \Phi_{ik}(u) C_{kj}(\tau - u) \, du, \qquad \forall \tau > 0
\]

   Discretizing over \(Q\) lag quadrature nodes yields a block-Toeplitz linear system whose solution determines the **branching ratio matrix** \(G = \|\boldsymbol{\Phi}\|_1 \in \mathbb{R}_+^{N \times N}\) and **criticality spectral radius** \(\rho(G) = \max_k |\lambda_k(G)| \in [0, 1)\).

---

### Markovian Lifted Heston Rough Volatility & Log-Signatures (`LiftedHestonRoughVolTransformer`, `LogSignatureTransformer`)

1. **Lyndon-Basis Truncated Log-Signature:**
   Over a local window \([s, t]\) of the time-augmented price-volatility path \(X_u = (u, \log P_u, \sigma_u)\), the depth-2 log-signature combines Level-1 increments \(\Delta X_{s,t}\) with the signed **Lévy areas** (Lie brackets):

\[
\mathcal{A}^{a,b}_{s,t} = \frac{1}{2} \int_s^t \left( (X_u^a - X_s^a)\,dX_u^b - (X_u^b - X_s^b)\,dX_u^a \right)
\]

2. **Abi Jaber (2019) \(K\)-Factor Markovian Lifted Heston Kernel:**
   Approximates the singular fractional power-law kernel \(K(t) = t^{H - 1/2} / \Gamma(H + 1/2)\) via a finite superposition of \(K\) Markovian Ornstein-Uhlenbeck volatility factors \(U_{t,k}\) with geometric mean-reversion speeds \(x_k = r^{k - K/2}\) and analytic weights \(c_k(H_{\text{eff}})\), where \(H_{\text{eff}} \in (0.05, 0.50]\) blends the realized variogram Hurst \(H_{\text{RV}}\) and implied surface Hurst \(H_{\text{IV}}\).
