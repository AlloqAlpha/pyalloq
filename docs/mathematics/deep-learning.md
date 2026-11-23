# Deep Learning & Differentiable Optimization Mathematics

This page documents the exact mathematical formulations behind the neural architectures, differentiable optimization layers, covariance heads, loss functions, and self-supervised world models in `pyalloq.dl`.

---

## 1. Size- & Permutation-Equivariant Set Transformer (`SetTransformerBackbone`)

Financial universes change size over time due to IPOs, delistings, and Point-in-Time tradability masks \(M_t \in \{0, 1\}^N\). `SetTransformerBackbone` enforces strict permutation equivariance \(f(\Pi X) = \Pi f(X)\) and cardinality stability across variable \(N_{\text{active}} = \sum_i M_{t,i}\).

### Scalable-Softmax Attention (`ScalableMultiheadAttention`)

Standard dot-product attention suffers from entropy dilution as \(N_{\text{active}}\) grows. `ScalableMultiheadAttention` scales queries by \(\log N_{\text{active}}\) (Scalable-Softmax, SSMax) and supports additive Graphormer relational bias \(B_{\text{attn}} \in \mathbb{R}^{N \times N}\) from `RelationalGraphPriorModule`:

\[
\operatorname{SSMaxAttn}(Q, K, V) = \operatorname{softmax}\!\left(\frac{s_{\text{head}} \ln(\max(2, N_{\text{active}}))}{\sqrt{d_k}} Q K^\top + B_{\text{attn}} + M_{\text{pad}}\right) V
\]

where \(M_{\text{pad}}\) sets masked keys to \(-\infty\) prior to softmax.

### Induced Set Attention Blocks (`ISAB`) & Global Regime Pooling (`PMA`)

To avoid \(O(N^2)\) cross-asset attention, `ISAB` routes information through \(m\) learnable inducing points \(I \in \mathbb{R}^{m \times d}\) in \(O(N m)\) time:

\[
H_I = \operatorname{MAB}(I, Z_{\text{in}}), \qquad Z_{\text{out}} = \operatorname{MAB}(Z_{\text{in}}, H_I)
\]

A single-seed Pooling by Multihead Attention block (`PMA`) aggregates active assets into a permutation-invariant global market token \(z_{\text{regime}} = \operatorname{MAB}(S_{\text{seed}}, Z_{\text{out}}) \in \mathbb{R}^d\).

---

## 2. Equivariant Neural Covariance Heads & Woodbury Identities

`SetTransformerCovarianceNet` guarantees \(\hat{\Sigma}(\Pi X) = \Pi \hat{\Sigma}(X) \Pi^\top\) via three selectable heads:

### A. Factor + Diagonal Head (`ISABFactorDiagonalHead`)

Projects the \(m\) inducing tokens \(H_I \in \mathbb{R}^{m \times d}\) into a strictly SPD latent factor covariance \(\Sigma_{F,t} = \frac{1}{d} H_I H_I^\top + \epsilon_F I_m\), asset loadings \(B_t \in \mathbb{R}^{N \times m}\), and strictly positive idiosyncratic variances \(D_t = \operatorname{diag}(d_t)\) with \(d_{t,i} = \operatorname{softplus}(\phi_D(z_{t,i})) + \epsilon_d\):

\[
\hat{\Sigma}_t = B_t \Sigma_{F,t} B_t^\top + D_t
\]

Defining the \(m \times m\) **Woodbury capacitance matrix** \(M_t = \Sigma_{F,t}^{-1} + B_t^\top D_t^{-1} B_t\), the matrix determinant lemma and Sherman-Morrison-Woodbury identity evaluate exact log-determinants and linear solves in \(O(N m^2 + m^3)\) time instead of \(O(N^3)\):

\[
\ln |\hat{\Sigma}_t| = \sum_{i=1}^N \ln d_{t,i} + \ln |\Sigma_{F,t}| + \ln |M_t|
\]

\[
\hat{\Sigma}_t^{-1} v = D_t^{-1} v - D_t^{-1} B_t M_t^{-1} B_t^\top D_t^{-1} v
\]

These identities directly power `WoodburyGaussianNLLLoss` and `WoodburySteinLoss`:

\[
\mathcal{L}_{\text{WNLL}} = \frac{1}{N_{\text{active}}}\left(\ln |\hat{\Sigma}_t| + r_{t+1}^\top \hat{\Sigma}_t^{-1} r_{t+1}\right), \qquad
\mathcal{L}_{\text{WStein}} = \frac{1}{N_{\text{active}}}\left(\operatorname{Tr}(\hat{\Sigma}_t^{-1} \Sigma_{\text{true}}) + \ln |\hat{\Sigma}_t| - \ln |\Sigma_{\text{true}}| - N_{\text{active}}\right)
\]

### B. Equivariant Gram Correlation Head (`EquivariantGramCorrHead`)

Computes \(\ell_2\)-normalized asset embeddings \(\tilde{U}_t \in \mathbb{R}^{N \times d}\), shrinks their Gram matrix toward identity using a regime-gated shrinkage weight \(\lambda_t = \sigma(\psi(z_{\text{regime}})) \in [0.05, 0.95]\), and scales by predicted volatilities \(D_{\sigma,t}\):

\[
\hat{\Gamma}_t = (1 - \lambda_t) \tilde{U}_t \tilde{U}_t^\top + \lambda_t I_N, \qquad \hat{\Sigma}_t = D_{\sigma,t} \hat{\Gamma}_t D_{\sigma,t}
\]

### C. Rotational Invariant Estimator Spectral Head (`RIESpectralHead`)

Eigendecomposes the sample correlation matrix \(C_t = V_t \Lambda_t V_t^\top\), passes the spectrum \(\Lambda_t\) conditioned on aspect ratio \(q_{\text{in}} = N_{\text{active}} / L\) through a `SAB` block to obtain cleaned eigenvalues \(\lambda_k^*\) normalized so \(\sum_k \lambda_k^* = N_{\text{active}}\), reconstructs \(\hat{C}_t = V_t \operatorname{diag}(\lambda^*) V_t^\top\) rescaled to unit diagonal, and scales by \(D_{\sigma,t}\).

---

## 3. Differentiable Portfolio Mandate Projections (`SetTransformerPolicyNet`)

### Capped Simplex Projection (`CappedSimplexProjection`)

Maps unconstrained asset logits \(y \in \mathbb{R}^N\) with tradability mask \(m \in \{0, 1\}^N\) to the exact Euclidean projection onto the capped simplex:

\[
w^*(y) = \arg\min_{w \in \mathbb{R}^N} \frac{1}{2}\|w - y\|_2^2 \quad \text{s.t.} \quad \sum_{i : m_i = 1} w_i = 1, \quad 0 \le w_i \le w_{\max} m_i
\]

By KKT optimality, \(w_i^* = \operatorname{clip}(y_i - \nu^*, \, 0, \, w_{\max}) \cdot m_i\), where the unique scalar Lagrange multiplier \(\nu^*\) is found via 55-step bisection. Letting \(S = \{i : m_i = 1 \text{ and } 0 < w_i^* < w_{\max}\}\) be the active interior support, the exact implicit backward Jacobian applied to upstream gradient \(g = \frac{\partial \mathcal{L}}{\partial w^*}\) is:

\[
\frac{\partial \mathcal{L}}{\partial y_i} = \mathbf{1}_{i \in S} \left(g_i - \frac{1}{|S|} \sum_{k \in S} g_k\right)
\]

Setting \(w_{\max} = 1.0\) yields exact **Sparsemax**.

### Masked Dirichlet Cash Head (`MaskedDirichletCashHead`) & Dollar-Neutral Head

- **Masked Dirichlet**: Forms concentration parameters \(\alpha = (\operatorname{softplus}([y, \, a_{\text{cash}}^\top z_{\text{regime}}]) + 1) \odot \tilde{m}\), outputting deterministic mean \(\alpha / (\mathbf{1}^\top \alpha)\) or reparameterized Gamma samples during exploration.
- **Dollar-Neutral Stat-Arb (`DollarNeutralStatArbHead`)**: Sets \(a = \tanh(y) \odot m\), demeans across active assets \(\tilde{a} = (a - \bar{a}) \odot m\), and normalizes \(w = G_{\text{gross}} \tilde{a} / (\|\tilde{a}\|_1 + \epsilon)\) so \(\sum_i w_i = 0\) and \(\sum_i |w_i| = G_{\text{gross}}\).

---

## 4. Fragility-Gated Cholesky & Rough Path Encoders

### `FragilityShrinkageCholeskyLayer` (`CRISPSparseSTGAT`)

Blends a raw Cholesky covariance \(\Sigma_{\text{raw}} = L L^\top + \epsilon I\) toward a Buss-Vilkov equicorrelation target \(\Sigma_{\text{target}}\) using a fragility gate driven by the Graph Laplacian Fiedler value \(\lambda_2\) and Hawkes branching spectral radius \(\rho_G\):

\[
\alpha_{\text{frag}} = \sigma\!\left(w_0 + w_\lambda \lambda_2 + w_\rho \frac{\rho_G}{1 - \rho_G + 0.02}\right) \in [0.02, 0.95], \qquad
\hat{\Sigma} = (1 - \alpha_{\text{frag}}) \Sigma_{\text{raw}} + \alpha_{\text{frag}} \Sigma_{\text{target}}
\]

### `NeuralJumpRDE` & `LogSigSSMBlock`

Given windowed depth-2 Lyndon log-signature tokens \(u_k = \operatorname{LogSig}_2(X_k)\) (`TorchLogSignatureExtractor`):

- **`NeuralJumpRDE`** alternates continuous log-ODE drift \(h_{\text{cont}} = h + f_\theta(h, u_k)\) with a self-exciting Hawkes jump intensity \(\lambda_k = \delta \lambda_{k-1} + (1 - \delta)\operatorname{softplus}(g(h_{\text{cont}}, |\Delta X_k|))\), jump probability \(p_{\text{jump}} = 1 - e^{-\lambda_k}\), and discontinuous state update \(h_k = \operatorname{LN}(h_{\text{cont}} + p_{\text{jump}} J_\psi(h_{\text{cont}}, \Delta X_k))\).
- **`LogSigSSMBlock`** evolves a selective state-space recurrence with input-dependent step \(\Delta_k = \operatorname{softplus}(W_\Delta u_k)\), \(A = -\exp(\log A)\), \(B_k = W_B u_k\), \(C_k = W_C u_k\):
  \[
  h_k = \exp(\Delta_k A) \odot h_{k-1} + (\Delta_k u_k) \otimes B_k, \qquad y_k = \sum_{s} h_{k,\cdot,s} C_{k,s} + D \odot u_k
  \]

---

## 5. Entropic Value-at-Risk & Self-Supervised World Models

### `SoftMinEVaRLoss` (DeePM)

Combines the smooth Entropic Value-at-Risk of net portfolio returns with a log-sum-exp soft-maximum drawdown penalty over batch paths \(b = 1, \dots, B\):

\[
\mathcal{L}_{\text{EVaR}} = \frac{1}{\beta} \ln\!\left(\frac{1}{B} \sum_{b=1}^B \exp\!\left(-\beta \, r_{p,b}^{\text{net}}\right)\right) + \lambda_{\text{dd}} \cdot \frac{1}{\beta} \ln\!\left(\sum_{b=1}^B \exp\!\left(\beta \cdot \text{DD}_b\right)\right)
\]

### `MarketJEPA` & Mask-Aware `VICRegLoss`

`MarketJEPA` encodes context \(Z_{\text{ctx}} = f_\theta(X_{\text{ctx}})\) and stop-gradient EMA target \(Z_{\text{tgt}} = f_\xi(X_{\text{tgt}})\) (\(\xi \leftarrow \tau \xi + (1-\tau)\theta\)), predicts \(\hat{Z}_{\text{tgt}} = g_\phi(Z_{\text{ctx}})\), and minimizes mask-aware VICReg strictly over active tokens:

\[
\mathcal{L}_{\text{VICReg}} = \lambda_{\text{inv}} \operatorname{MSE}_{\text{act}}(\hat{Z}_{\text{tgt}}, Z_{\text{tgt}}) + \mu_{\text{var}} \operatorname{mean}\!\left(\operatorname{ReLU}(\gamma - \sigma_{\text{act}})\right) + \nu_{\text{cov}} \frac{1}{d} \sum_{i \neq j} C_{ij}^2
\]

---

## 6. Foundation Model Ensembles (`EnsembleReturnEstimator` & `EnsembleViewGenerator`)

### Scale-Free Quantile sCRPS & Law of Total Variance

For \(M\) base estimators producing predictive means \(\mu_{m,i}\) and standard deviations \(\sigma_{m,i}\), `EnsembleReturnEstimator` computes per-asset model weights \(w_{m,i}\) (e.g., `"scrps_adaptive"`: \(w_{m,i} \propto (\sigma_{m,i}^2 \cdot \text{sCRPS}_m)^{-1}\)) and combines uncertainty via the exact **Law of Total Variance**:

\[
\mu_{\text{ens}, i} = \sum_{m=1}^M w_{m,i} \mu_{m,i}, \qquad
\sigma_{\text{ens}, i}^2 = \sum_{m=1}^M w_{m,i} \left(\sigma_{m,i}^2 + (\mu_{m,i} - \mu_{\text{ens}, i})^2\right)
\]

### Exact Bayesian Precision Pooling (`EnsembleViewGenerator`)

Combines \(M\) Black-Litterman view generators \((P=I_N, Q_m, \Omega_m)\) into a single posterior view pair:

\[
\Omega_{\text{ens}} = \left(\sum_{m=1}^M \Omega_m^{-1}\right)^{-1}, \qquad
Q_{\text{ens}} = \Omega_{\text{ens}} \left(\sum_{m=1}^M \Omega_m^{-1} Q_m\right)
\]
