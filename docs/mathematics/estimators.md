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

3. **Bakshi-Kapadia-Madan (2003) Risk-Neutral Moments & Cornish-Fisher Tail Expansion:**
   From out-of-the-money (OTM) option prices \(Q(K_i)\), the quadratic (\(V\)), cubic (\(W\)), and quartic (\(X\)) model-free contracts determine risk-neutral skewness (\(\text{MFIS}_t\)) and excess kurtosis (\(\text{MFIK}_t\)). When `use_cornish_fisher_tail=True`, `ImpliedCovarianceEstimator` scales each asset's implied volatility by the Cornish-Fisher quantile ratio at confidence \(z_\alpha\):

\[
m_{\text{CF}}(S, K_{\text{ex}}) = \frac{z_{\text{CF}}}{z_\alpha}, \qquad z_{\text{CF}} = z_\alpha + \frac{z_\alpha^2 - 1}{6} S + \frac{z_\alpha^3 - 3 z_\alpha}{24} K_{\text{ex}} - \frac{2 z_\alpha^3 - 5 z_\alpha}{36} S^2
\]

4. **Von Neumann Covariance Entropy (`compute_von_neumann_entropy`):**
   Given normalized eigenvalues \(p_i = \lambda_i(\Sigma) / \operatorname{Tr}(\Sigma)\), the normalized Von Neumann Entropy measures spectral diversification vs. rank-1 contagion collapse:

\[
\mathcal{S}_{\text{vN}}(\Sigma) = -\frac{1}{\ln N} \sum_{i=1}^N p_i \ln p_i \in [0, 1]
\]

---

### Time-Aware Prior-Data Fitted Networks (`ApolloPFN`, Potapczynski et al., TMLR 2026)

1. **Amortized In-Context Posterior Predictive Distribution (PPD):**
   Trained offline on synthetic Structural Causal Model (SCM) time-series tasks \(\mathcal{D} \sim p(\mathcal{D})\) generated via **Algorithm 1 (`SRNGNPriorGenerator`, Single Root Node Random Growing Network)**, `ApolloPFNTransformer` directly approximates the Bayesian posterior predictive distribution in a single forward pass:

\[
q_\theta\!\left(Y_{T+1:T+H} \;\middle|\; X_{1:T+H}, \, Y_{1:T}\right) \;\approx\; \int p\!\left(Y_{T+1:T+H} \mid X_{T+1:T+H}, \xi\right) p\!\left(\xi \mid X_{1:T}, Y_{1:T}\right) d\xi
\]

2. **Temporal-Feature Factorized Transformer (`AttnFeat` + RoPE `AttnSamp`):**
   Given embedding tensor \(Z \in \mathbb{R}^{(T+H) \times F \times D}\) with observation mask tokens \(0\) (observed) and \(-2\) (unobserved horizon targets) plus absolute sinusoidal encodings \(\Omega_{t, 2d}\), each layer \(\ell = 1, \dots, L\) alternates cross-variate attention and Rotary Position Embedded (RoPE) full-horizon temporal attention:

\[
Z \leftarrow \operatorname{LN}_1\!\left(Z + \operatorname{AttnFeat}(Z)\right), \quad
Z \leftarrow \operatorname{LN}_2\!\left(Z + \operatorname{AttnSamp}_{\text{RoPE}}(Z)\right), \quad
Z \leftarrow \operatorname{LN}_3\!\left(Z + \operatorname{MLP}(Z)\right)
\]

3. **Riemann Bar Distribution & `sCRPS`-Adaptive Bayesian Model Averaging (`EnsembleReturnEstimator`):**
   The output head parameterizes a piecewise-constant density over \(B\) Riemann bins spanning \([z_{\min}, z_{\max}]\). Evaluated across quantile levels \(\alpha_1 < \dots < \alpha_Q\), the scale-free Continuous Ranked Probability Score (`sCRPS`, Appendix B) is:

\[
\operatorname{sCRPS}(y, \hat{y}) = \frac{\sum_{t=T+1}^{T+H} \frac{2}{Q} \sum_{j=1}^Q \left[ \alpha_j \left(y_t - \hat{y}_t^{\alpha_j}\right)_+ + (1 - \alpha_j)\left(\hat{y}_t^{\alpha_j} - y_t\right)_+ \right]}{\sum_{t=T+1}^{T+H} |y_t|}
\]

   In `EnsembleReturnEstimator`, model \(m\)'s weight on asset \(i\) under `weighting_mode="scrps_adaptive"` is proportional to its calibrated precision \(w_{m,i} \propto \left(\sigma_{m,i}^2 \cdot \operatorname{sCRPS}_m\right)^{-1}\), or under `weighting_mode="cvar_adaptive"` to its inverse squared tail risk \(w_{m,i} \propto \operatorname{CVaR}_{\alpha, m, i}^{-2}\).

4. **Closed-Form Riemann Bar Tail Functionals & Copula-Coupled Scenario Generation (Müller et al., ICLR 2022; Demarta & McNeil, 2005; Patton, 2006):**
   Given Riemann bin edges \(b_0 < b_1 < \dots < b_B\), widths \(w_k = b_{k+1} - b_k\), probabilities \(p_k\), and cumulative probabilities \(C_k = \sum_{j < k} p_j\), the truncated probability mass in bin \(k\) below tail threshold \(1 - \alpha\) is \(\Delta p_k = \max(0, \min(p_k, 1 - \alpha - C_k))\). With the Müller et al. (2022) half-normal left-tail shift \(\delta_0 = \sqrt{2/\pi} w_0\) on the outermost bucket \(k = 0\), `RiemannBarDistribution` evaluates exact Expected Shortfall (\(\operatorname{CVaR}_\alpha\)), Sortino downside semi-deviation (\(\sigma_{-}\)), and Bowley skewness (\(\mathcal{S}_{\text{Bowley}}\)):

\[
\operatorname{CVaR}_\alpha(Z) = -\frac{1}{1 - \alpha} \sum_{k=0}^{B-1} \Delta p_k \left(b_k + \frac{\Delta p_k}{2 p_k} w_k - \delta_0 \mathbb{I}_{\{k=0\}}\right), \qquad
\mathcal{S}_{\text{Bowley}} = \frac{(q_{0.90} - q_{0.50}) - (q_{0.50} - q_{0.10})}{q_{0.90} - q_{0.10}}
\]

   To synthesize joint \(N\)-asset scenarios (`BaseScenarioEstimator`), correlated uniform variates \(u^{(m, h)} \sim C_{\nu, R_{\text{imp}}}^{\text{t}}\) are drawn from a Student-\(t\) (or stratified LHS-\(t\)) copula parameterized by the Buss & Vilkov (2012) option-implied correlation matrix blended with `AttnFeat` epistemic correlation, and inverted through each asset's exact Riemann Bar marginal CDF: \(\tilde{r}_{i, h}^{(m)} = \mu_{i,\text{ctx}} + \sigma_{i,\text{ctx}} F_{\text{bar}, i, h}^{-1}(u_i^{(m, h)})\).

---

### Permutation-Invariant Set Transformers & Equivariant Deep Covariance (`SetTransformerCovarianceNet`, `SetTransformerPolicyNet`)

1. **Three Portfolio Symmetry Laws:**
   Let \(X_t \in \mathbb{R}^{N \times d}\) denote the cross-sectional matrix of per-asset temporal embeddings at decision time \(t\), and let \(\Pi \in \{0, 1\}^{N \times N}\) be any permutation matrix reordering tickers:
   - **Permutation Invariance** (Global `[REGIME]` token \(z_{\text{regime}}\), cash logit \(l_t^{(\text{cash})}\), RL critic \(V_\psi(s_t)\), and latent factor covariance \(\Sigma_{F,t} \in \mathbb{S}_{++}^m\)):
     \[
     g(\Pi X_t) = g(X_t) \qquad \forall \Pi \in S_N
     \]
   - **1D Permutation Equivariance** (Expected returns \(\hat{\mu}_t\), factor loadings \(B_t \in \mathbb{R}^{N \times m}\), idiosyncratic variances \(d_t \in \mathbb{R}_{++}^N\), and portfolio weights \(w_t \in \mathbb{R}^N\)):
     \[
     f(\Pi X_t) = \Pi f(X_t) \qquad \forall \Pi \in S_N
     \]
   - **2D Congruence Equivariance** (Cross-asset covariance matrix \(\hat{\Sigma}_t \in \mathbb{S}_{++}^N\)):
     \[
     \hat{\Sigma}(\Pi X_t) = \Pi \hat{\Sigma}(X_t) \Pi^\top \qquad \forall \Pi \in S_N
     \]

2. **Induced Set Attention Blocks (`ISAB`), Pooling by Multihead Attention (`PMA`), and Scalable-Softmax (`SSMax`):**
   Following Lee et al. (ICML 2019), `ISAB` replaces quadratic \(\mathcal{O}(N^2)\) pairwise self-attention (`SAB`) with \(m \ll N\) learnable inducing points \(I \in \mathbb{R}^{m \times d}\) functioning as dynamic latent risk factors:
   \[
   H_{I,t} = \operatorname{MAB}_1(I, X_t) \in \mathbb{R}^{m \times d}, \qquad
   Z_t = \operatorname{MAB}_2(X_t, H_{I,t}) \in \mathbb{R}^{N \times d}, \qquad
   z_{\text{regime}, t} = \operatorname{PMA}_1(Z_t) = \operatorname{MAB}(S, \operatorname{rFF}(Z_t))
   \]
   To eliminate **Softmax attention fading** when transferring from a 500-stock training universe to a 50-stock inference sleeve, every variable-cardinality attention key axis replaces standard Softmax with **Scalable-Softmax** (Nakanishi, 2025) over active tradable assets \(N_{\text{active}} = \sum_{j=1}^N m_{j,t}\):
   \[
   \operatorname{SSMax}(z_i; N_{\text{active}}, s) = \frac{m_{i,t} \exp\!\left((s \ln N_{\text{active}})\, z_i\right)}{\sum_{j=1}^N m_{j,t} \exp\!\left((s \ln N_{\text{active}})\, z_j\right)}
   \]

3. **Three Strictly Congruence-Equivariant PSD Covariance Heads:**
   - **ISAB Low-Rank Factor-Plus-Diagonal Head (`ISABFactorDiagonalHead`, `head_type="factor_diag"`):**
     Reuses both the equivariant asset embeddings \(Z_t\) and the invariant inducing-point factor state \(H_{I,t}\):
     \[
     \hat{\Sigma}_t = B_t \Sigma_{F,t} B_t^\top + D_t, \qquad
     B_t = \phi_B(Z_t) \in \mathbb{R}^{N \times m}, \quad
     D_t = \operatorname{diag}\!\left(\operatorname{softplus}(\phi_D(Z_t)) + \epsilon_d\right), \quad
     \Sigma_{F,t} = \frac{1}{d} H_{I,t} H_{I,t}^\top + \epsilon_F I_m
     \]
   - **Volatility-Scaled Equivariant Gram Correlation Head (`EquivariantGramCorrHead`, `head_type="gram_corr"`):**
     Maps each asset to \(\ell_2\)-normalized row vectors \(\tilde{u}_{i,t} = u_{i,t} / (\|u_{i,t}\|_2 + \epsilon)\) blended with learnable `[REGIME]` shrinkage \(\lambda_t \in (0, 1)\):
     \[
     \hat{\Gamma}_t = (1 - \lambda_t) \tilde{U}_t \tilde{U}_t^\top + \lambda_t I_N, \qquad
     \hat{\Sigma}_t = D_{\sigma, t} \hat{\Gamma}_t D_{\sigma, t}
     \]
   - **Rotation- and Permutation-Invariant Spectral Cleaning (`RIESpectralHead`, `head_type="rie_spectral"`, Bongiorno et al., 2026):**
     Decomposes the sample correlation matrix \(C_t = V_t \Lambda_t V_t^\top\), passes the unordered eigenvalue set \(\{(\lambda_{k,t}, q_{\text{in}})\}\) with aspect ratio \(q_{\text{in}} = N_{\text{active}} / L\) through a 1D equivariant Set Attention block to predict cleaned eigenvalues \(\lambda_t^*\), and recombines \(\hat{\Sigma}_t = D_{\sigma, t} (V_t \operatorname{diag}(\lambda_t^*) V_t^\top) D_{\sigma, t}\).

4. **Exact \(\mathcal{O}(N m^2 + m^3)\) Woodbury Inversion & Cardinality-Normalized Losses (`WoodburyGaussianNLLLoss`, `WoodburySteinLoss`):**
   Via the **Woodbury Matrix Identity** and **Matrix Determinant Lemma** with capacitance matrix \(M_t = \Sigma_{F,t}^{-1} + B_t^\top D_t^{-1} B_t \in \mathbb{S}_{++}^m\):
   \[
   \hat{\Sigma}_t^{-1} v = D_t^{-1} v - D_t^{-1} B_t M_t^{-1} B_t^\top D_t^{-1} v, \qquad
   \ln |\hat{\Sigma}_t| = \sum_{i \in \text{active}} \ln d_{i,t} + \ln |\Sigma_{F,t}| + \ln |M_t|
   \]
   Normalizing Gaussian NLL and James-Stein loss by \(N_{\text{active}}\) holds loss magnitudes and gradient norms invariant across mixed-cardinality batches.
