# Conformal Risk, Online Meta-Weighting, Jump Regimes & Spectral Fragility

This page documents the exact mathematical formulations behind PyAlloq's distribution-free conformal risk controllers, online meta-weighting strategies, statistical jump models, Mahalanobis relevance estimators, and Wiener-Hopf Hawkes spectral fragility diagnostics.

---

## 1. Conformal Prediction & Finite-Sample Risk Control (`pyalloq_core.conformal`)

### Online Conformal Calibration (`ConformalPIDCalibrator`)

Given nonconformity residual scores \(s_t = |y_t - \hat{y}_t|\) and target miscoverage \(\alpha \in [0.01, 0.95]\), let \(\text{err}_t = \mathbf{1}\{s_t > q_t\}\). `ConformalPIDCalibrator` supports three non-exchangeable time-series methods:

1. **Conformal PID Control (`method="pid"`, Angelopoulos, Candès & Tibshirani, NeurIPS 2023)**:
   \[
   q_{p,t+1} = \max\!\left(q_{p,t} + \eta_p \, s_{\text{scale}} (\text{err}_t - \alpha), \, 10^{-6}\right)
   \]
   \[
   I_t = K_I \, s_{\text{scale}} \tan\!\left(\operatorname{clip}\!\left(\frac{\sum_{\tau \le t}(\text{err}_\tau - \alpha)}{c_{\text{sat}}\sqrt{t \ln t}}, -1.35, 1.35\right)\right), \qquad
   D_t = K_D \left(\text{ewma}_t - q_{p,t}\right)
   \]
   \[
   q_t = \max(q_{p,t} + I_t + D_t, \, 10^{-6})
   \]
2. **Dynamically Tuned ACI (`method="dtaci"`, Gibbs & Candès, JMLR 2024)**: Runs parallel adaptive experts \(\alpha_{t+1}^{(m)} = \alpha_t^{(m)} + \gamma_m(\alpha - \text{err}_t^{(m)})\) weighted by exponential pinball loss with 5% fixed-share mixing.
3. **Non-Exchangeable Weighted Conformal (`method="nexcp"`, Barber et al., AoS 2023)**: Computes the weighted \((1 - \alpha)\)-quantile using exponentially decayed weights \(w_i = \rho^{t - i}\).

### Conformal Risk Control (`ConformalRiskController`, Angelopoulos et al., ICLR 2024)

For any monotone bounded loss \(L_i(\lambda) \in [0, B]\) with empirical risk \(\hat{R}_n(\lambda) = \frac{1}{n}\sum_{i=1}^n L_i(\lambda)\), selects:

\[
\hat{\lambda} = \inf\!\left\{\lambda \in \Lambda \;:\; \frac{n}{n+1} \hat{R}_n(\lambda) + \frac{B}{n+1} \le \alpha\right\}
\]

guaranteeing finite-sample expected risk control \(\mathbb{E}[L_{n+1}(\hat{\lambda})] \le \alpha\).

### Conformal Portfolio Optimizers (`pyalloq.classical.optimizers.conformal`)

- **`CPPSAllocator` (Kato, 2025)**: Forms candidate portfolios on the simplex, calibrates portfolio-level conformal intervals \([L_T^{w,\alpha}, U_T^{w,\alpha}]\), retains the top `top_m_defense` portfolios by lower bound \(L_T^{w,\alpha}\), and selects \(\arg\max U_T^{w,\alpha}\) among survivors.
- **`ConformalKellyOverlay`**: Replaces Gaussian variance in fractional Kelly sizing with squared conformal interval width \(W_{t,i} = U_{t,i} - L_{t,i}\):
  \[
  f_{t,i}^{\text{kelly}} = \eta_{\text{kelly}} \frac{\max(0, \mu_{t,i} - r_f)}{W_{t,i}^2}
  \]
- **`ConformalSafetyGateAllocator` (Gated-CSRC, Zarei et al., 2026)**: Given base policy \(\hat{w}_t\) and calibrated per-asset tail loss \(l_i = \max(0.002, -L_{i,\text{daily}} \cdot \max(0.75, \hat{\lambda}_{\text{crc}}))\), solves the convex Euclidean projection onto the `RiskCertificate` feasible set:
  \[
  w_t^* = \arg\min_{w} \frac{1}{2}\|w - \hat{w}_t\|_2^2 \quad \text{s.t.} \quad
  l^\top |w| \le L_{\text{tail}}^{\max}, \;
  \|w\|_1 \le \min(1 - c_{\min}, L_{\max}), \;
  \|\Sigma^{1/2} w\|_2 \le \sigma_{\max}, \;
  \|w - w_{t^-}\|_1 \le \tau_{\max}
  \]
  along with box, sector, and restricted-asset caps.

---

## 2. Online Meta-Weighting & Ensemble Allocation (`pyalloq_core.meta`)

Given trailing expert return matrix \(R^{\text{exp}} \in \mathbb{R}^{T_{\text{eval}} \times K}\) over \(K\) child allocators, `EnsembleAllocator` computes meta-weights \(\alpha_t \in \Delta^K\) and executes pre-TCA internal trade crossing \(w_{\text{net}} = \Pi_{\mathcal{C}}(W_t \alpha_t)\):

| Strategy | Core Update / Weighting Formula |
|----------|--------------------------------|
| **`AdaHedgeStrategy`** | Adapts learning rate \(\eta_t = \ln(K) / \Delta_t\) from cumulative mixability gap \(\Delta_t = \sum_\tau (h_\tau - m_\tau)\), where \(m_\tau = -\frac{1}{\eta}\ln(\sum_k w_k e^{-\eta l_{\tau,k}})\). |
| **`FixedShareStrategy`** | Multiplicative weights followed by fixed-share regime pool mixing: \(w_t^{\text{post}} = (1 - \alpha_s) w_t^{\text{mult}} + \frac{\alpha_s}{K}\). |
| **`CBCEStrategy`** | Krichevsky-Trofimov coin betting over dyadic horizons \(L_m = 2^{m+1}\) (`bet_frac = sum_g / (tau + 2)`), softmax-aggregated by interval Sharpe. |
| **`AdaNormalHedgeStrategy`** | Potential weights \(w_k \propto \frac{[R_k]_+}{C_k} \exp\!\left(\frac{[R_k]_+^2}{3 C_k}\right)\) from discounted cumulative regret \(R_k\) and absolute regret \(C_k\). |
| **`DualScaledSwitchingCostStrategy`** | Minimax rate \(\eta = \sqrt{\ln K / (B T)}\), exponentially decayed Sharpe softmax target contracted toward \(\alpha_{t-1}\) by \(\frac{1}{1 + 2B}\) and projected onto \(\Delta^K\). |
| **`CADTSBanditStrategy`** | Pessimistic \(\min(\theta_{\text{disc}}, \theta_{\text{win}})\) across 25 Thompson samples from discounted and sliding-window Gaussian posteriors. |
| **`BPDSMetaStrategy`** | Gibbs utility tilting \(\pi_{t,j} \propto \pi_{0,j} \exp(\tau_t \tilde{u}_{t,j})\) with quadratic decision utility \(s_{t,j} = r_{t,j} - \frac{\gamma}{2} r_{t,j}^2\) and baseline fallback \(M_0\). |
| **`EllipticalShrinkageStrategy`** | Estimates elliptical kurtosis \(\hat{\kappa} = \frac{1}{N(N+2)T}\sum_t [(r_t - \mu)^\top \Sigma^{-1}(r_t - \mu)]^2 - 1\), sets \(1/N\) anchor weight \(\delta^* = \operatorname{clip}\!\left(\frac{c_{\text{ellip}}}{1 + c_{\text{ellip}}}, \delta_{\min}, 0.85\right)\) where \(c_{\text{ellip}} = (1 + \hat{\kappa}) N / T\), and weights active children by \(\mu - \frac{\gamma}{2}(1 + 0.5\hat{\kappa})\sigma^2\). |

---

## 3. Statistical Jump Models (`pyalloq_features.regimes.jump_models`)

### Sparse Jump Model (`SparseJumpModelDetector`, Nystrup et al., 2021; Shu et al., 2024)

Alternates exact \(O(T K)\) Bellman dynamic programming over state sequence \(s_{1:T} \in \{0, \dots, K-1\}^T\) with Witten-Tibshirani \(\ell_1/\ell_2\) feature-weight optimization:

\[
C_t(k) = l(y_t, \theta_k; w) + \min\!\left(C_{t-1}(k), \; \min_{j \neq k} C_{t-1}(j) + \lambda\right)
\]

\[
\max_{w \ge 0} \; w^\top b_{\text{BCSS}} \quad \text{s.t.} \quad \|w\|_2 \le 1, \quad \|w\|_1 \le \kappa
\]

### Continuous Jump Model (`ContinuousJumpModelDetector`, Aydınhan et al., AoOR 2024)

Relaxes hard state indicators to a smooth probability trajectory \(P = (p_1, \dots, p_T) \in (\Delta^K)^T\) solved by forward-backward Gauss-Seidel proximal Euclidean simplex projections:

\[
\min_{\Theta, \, p_t \in \Delta^K} \; \sum_{t=1}^T \langle p_t, l_t \rangle + \lambda \sum_{t=2}^T \|p_t - p_{t-1}\|_2^2 - \gamma_{\text{mode}} \sum_{t=1}^T \|p_t - \mathbf{1}/K\|_2^2
\]

---

## 4. Mahalanobis Relevance (`pyalloq.classical.estimators.relevance`)

Following Czasonis, Kritzman & Turkington (2023–2026), given historical market state vectors \(x_i\) with mean \(\bar{x}\) and covariance \(\Omega\), the relevance of historical observation \(i\) to current state \(x_t\) combines pairwise Mahalanobis similarity and informativeness:

\[
r_{it} = -\frac{1}{2}(x_i - x_t)^\top \Omega^{-1} (x_i - x_t) + \frac{1}{2}\left[(x_i - \bar{x})^\top \Omega^{-1}(x_i - \bar{x}) + (x_t - \bar{x})^\top \Omega^{-1}(x_t - \bar{x})\right]
\]

Observations below the \((1 - \phi)\) quantile \(r^*\) are censored, and `RelevanceConditionedCovarianceEstimator` / `RelevanceConditionedReturnEstimator` blend the relevance-weighted estimate with the unconditional prior weighted by the analytical `adjusted_fit` score.

---

## 5. Spectral Graph Fragility & Wiener-Hopf Hawkes (`hawkes_spectral.py`)

- **`SpectralGraphFragility`**: Computes the Fiedler value \(\lambda_2\) (second smallest eigenvalue of normalized Graph Laplacian \(\mathcal{L} = I - D^{-1/2} W D^{-1/2}\)), structural mixing time \(\tau_{\text{mix}} = 1 / \max(\lambda_2, 10^{-6})\), and canonically oriented Fiedler vector \(v_2\).
- **`WienerHopfHawkesSolver`**: Extracts discrete jump indicators \(dN_{t,i} = \mathbf{1}\{|z_{t,i}| \ge z_{\text{thresh}}\}\), constructs the \(Q N \times Q N\) Nyström block-Toeplitz Wiener-Hopf operator \(K_{\text{WH}}\) from empirical lag covariances \(C(h)\), solves \(K_{\text{WH}} \Phi = \text{rhs}\) with ridge regularization, projects \(\Phi \ge 0\), integrates the branching matrix \(G = \frac{1}{Q}\sum_{q=1}^Q \Phi(q)\), and caps the spectral radius \(\rho(G) \le \rho_{\max} < 1\).
