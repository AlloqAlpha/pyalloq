# Monte Carlo & Entropy Pooling Mathematics

This page covers the mathematical foundations of the scenario generators, missing-history imputation, Meucci's Entropy Pooling (first- and second-moment views), and path-dependent risk metrics implemented in `pyalloq-simulation`.

---

## 1. PCA Factor Back-Casting Imputation

When asset \(j\) has missing historical observations (e.g., a recent IPO missing \(t \in \mathcal{M}_j\) and observed on \(t \in \mathcal{O}_j\)), `FactorBackcastImputer` avoids truncating the \(N\)-asset matrix:

1. Identify anchor assets \(\mathcal{A}\) with coverage \(\ge c_{\min}\) and extract \(K\) orthogonal latent factor series \(F \in \mathbb{R}^{T \times K}\) via Principal Component Analysis on anchor returns.
2. Estimate ridge-stabilized factor loadings for asset \(j\) on its observed window \(\mathcal{O}_j\):

\[
\hat{\beta}_j = \left(X_{\mathcal{O}_j}^\top X_{\mathcal{O}_j} + \lambda I\right)^{-1} X_{\mathcal{O}_j}^\top r_{j, \mathcal{O}_j}, \quad X_t = [1, F_t]
\]

3. Impute missing returns for \(t \in \mathcal{M}_j\) with matched idiosyncratic residual variance \(\hat{\sigma}_{\epsilon, j}^2\):

\[
\hat{r}_{j, t} = X_t \hat{\beta}_j + \epsilon_{j, t}, \quad \epsilon_{j, t} \sim \mathcal{N}(0, \hat{\sigma}_{\epsilon, j}^2)
\]

4. Back-cast pre-IPO prices recursively from the first observed price \(P_{j, t_0}\):

\[
P_{j, t} = \frac{P_{j, t+1}}{1 + \hat{r}_{j, t+1}}, \quad t = t_0 - 1, \dots, 0
\]

---

## 2. Scenario Generation Models

### Filtered Historical Simulation (`FHSGenerator`)

Filtered Historical Simulation (Barone-Adesi et al., 1999) strips volatility clustering using univariate GARCH(1,1) models while preserving empirical cross-asset dependence and fat tails via non-parametric residual bootstrapping:

\[
r_{i, t} = \mu_i + \epsilon_{i, t}, \quad \epsilon_{i, t} = \sigma_{i, t} z_{i, t}
\]

\[
\sigma_{i, t}^2 = \omega_i + \alpha_i \epsilon_{i, t-1}^2 + \beta_i \sigma_{i, t-1}^2
\]

At each simulated step \(t\), a random historical date index \(\tau \sim \text{Uniform}\{1, \dots, T_{\text{hist}}\}\) is drawn synchronously across all \(N\) assets, preserving cross-sectional co-movement \(Z_{\tau} = (\hat{z}_{1, \tau}, \dots, \hat{z}_{N, \tau})\).

---

### GARCH-EVT-Student-\(t\) Copula (`GARCHEVTCopulaGenerator`)

To model extreme tail risk and positive lower-tail dependence (\(\lambda_L > 0\)), `GARCHEVTCopulaGenerator` combines AR(1)-GARCH(1,1) filtration with Extreme Value Theory (McNeil & Frey, 2000) and a Student-\(t\) Copula:

1. **AR(1)-GARCH(1,1) Filtration:** Extracts standardized residuals \(z_{i, t} = \epsilon_{i, t} / \sigma_{i, t}\).
2. **Semi-Parametric Generalized Pareto (GPD) Marginals:** Given lower and upper quantile thresholds \(u_L, u_U\) at probabilities \(q_L, q_U\), exceedances in the tails follow the Generalized Pareto Distribution:

\[
F_i(z) =
\begin{cases}
q_L \left(1 - G_{\xi_{L,i}, \beta_{L,i}}(u_L - z)\right) & z < u_L \\
\hat{F}_{\text{emp}}(z) & u_L \le z \le u_U \\
q_U + (1 - q_U) G_{\xi_{U,i}, \beta_{U,i}}(z - u_U) & z > u_U
\end{cases}
\]

where the GPD cumulative distribution function is:

\[
G_{\xi, \beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}
\]

3. **Student-\(t\) Copula Binding:** Uniform marginals \(u_{i, t} = F_i(z_{i, t})\) are mapped via the univariate Student-\(t\) quantile \(t_\nu^{-1}(u_{i, t})\) to estimate the copula correlation matrix \(R\) with degrees of freedom \(\nu\). Lower-tail dependence between assets \(i\) and \(j\) is strictly positive:

\[
\lambda_L = 2 \, t_{\nu + 1}\!\left(-\sqrt{\frac{(\nu + 1)(1 - \rho_{ij})}{1 + \rho_{ij}}}\right) > 0
\]

---

### Gaussian Mixture Markov Regimes (`GMMRegimeGenerator`)

Models discrete latent market states \(s_t \in \{1, \dots, K\}\) via a \(K\)-component Gaussian Mixture Model:

\[
p(r_t) = \sum_{k=1}^K \pi_k \, \mathcal{N}(r_t \mid \mu_k, \Sigma_k)
\]

Historical states are decoded via maximum a posteriori assignment \(\hat{s}_t = \arg\max_k \gamma_{t, k}\), yielding the empirical Markov transition matrix \(P \in \mathbb{R}^{K \times K}\) with entries \(P_{ab} = \mathbb{P}(s_{t} = b \mid s_{t-1} = a)\). Forward paths evolve \(s_{m, t} \sim P_{s_{m, t-1}, \cdot}\) and draw \(r_{m, t} \sim \mathcal{N}(\mu_{s_{m, t}}, \Sigma_{s_{m, t}})\).

---

## 3. Meucci's Entropy Pooling (`EntropyPooler` & `ViewBuilder`)

Given \(M\) simulated scenarios with prior probability vector \(p \in \Delta^M\) (typically uniform \(p_m = 1/M\)), **Entropy Pooling** (Meucci, 2008) finds the posterior probability distribution \(x \in \Delta^M\) that satisfies linear equality and inequality views while minimizing Relative Entropy (Kullback-Leibler divergence):

\[
\min_{x \in \mathbb{R}^M} \mathcal{E}(x \,\|\, p) = \sum_{m=1}^{M} x_m \ln\!\left(\frac{x_m}{p_m}\right)
\]

subject to:

\[
A x \le b, \quad C x = d, \quad \mathbf{1}^\top x = 1, \quad x \ge 0
\]

### Unconstrained Dual Formulation

Rather than solving an \(M\)-dimensional primal problem (\(M \sim 10,000\)), `EntropyPooler` solves the low-dimensional Lagrange dual over the \(V\) view multipliers \(\gamma = (\lambda, \nu)\) where \(\nu \ge 0\):

\[
\min_{\lambda, \nu \ge 0} \; \ln\!\left(\sum_{m=1}^M p_m \exp\!\left(-[G^\top \gamma]_m\right)\right) + \gamma^\top \text{targets}, \quad G = \begin{bmatrix} C \\ A \end{bmatrix}
\]

The posterior scenario weights are recovered analytically via the log-sum-exp stabilized Boltzmann-Gibbs distribution:

\[
x^*_m = \frac{p_m \exp\!\left(-[G^\top \gamma^*]_m\right)}{\sum_{k=1}^M p_k \exp\!\left(-[G^\top \gamma^*]_k\right)}
\]

### First- and Second-Moment View Linearization

Because expectation under the scenario measure is a linear inner product \(\mathbb{E}_x[g(R)] = \sum_{m=1}^M x_m g(R_m)\), both first- and second-moment views are **linear** in the probability vector \(x\):

**Absolute Return View (`AbsoluteView`):**

\[
\sum_{m=1}^M x_m R_{m, i} \;\in\; \{\le, =, \ge\}\; \mu_i^{\text{target}}
\]

**Relative Return View (`RelativeView`):**

\[
\sum_{m=1}^M x_m (R_{m, i} - R_{m, j}) \;\in\; \{\le, =, \ge\}\; \delta_{ij}^{\text{target}}
\]

**Volatility View (`VolatilityView`)** — centered around the reference mean \(\bar{\mu}_i\):

\[
\sum_{m=1}^M x_m (R_{m, i} - \bar{\mu}_i)^2 \;\in\; \{\le, =, \ge\}\; \left(\sigma_i^{\text{target}}\right)^2
\]

**Covariance / Correlation View (`CovarianceView`):**

\[
\sum_{m=1}^M x_m (R_{m, i} - \bar{\mu}_i)(R_{m, j} - \bar{\mu}_j) \;\in\; \{\le, =, \ge\}\; \rho_{ij}^{\text{target}} \bar{\sigma}_i \bar{\sigma}_j
\]

---

## 4. Path-Dependent Risk Metrics (`ScenarioEvaluator`)

Let \(W_{m, t}\) denote the simulated portfolio wealth on path \(m \in \{1, \dots, M\}\) at time step \(t \in \{0, \dots, T\}\) with \(W_{m, 0} = 1\):

**Path Maximum Drawdown (\(\text{MDD}_m\)):**

\[
\text{MDD}_m = \min_{0 \le t \le T} \frac{W_{m, t} - \max_{0 \le \tau \le t} W_{m, \tau}}{\max_{0 \le \tau \le t} W_{m, \tau}} \le 0
\]

**View-Adjusted Maximum Drawdown:**

\[
\text{MDD}_{\text{view}} = \sum_{m=1}^M x^*_m \, \text{MDD}_m
\]

**Probability of Ruin (\(\mathbb{P}_{\text{ruin}}\))** — for critical drawdown threshold \(\theta_{\text{ruin}} < 0\) (e.g., \(-0.30\)):

\[
\mathbb{P}_{\text{ruin}}(\theta_{\text{ruin}}) = \sum_{m=1}^M x^*_m \, \mathbb{I}\!\left(\text{MDD}_m \le \theta_{\text{ruin}}\right)
\]

---

## 5. Probability of Backtest Overfitting (PBO) & CPCV

### Combinatorial Symmetric Cross-Validation (`PBOEvaluator`)

Given a return matrix \(M_{\text{trials}} \in \mathbb{R}^{T \times M}\) of \(M\) candidate strategy configurations evaluated over \(T\) observations, **CSCV** (Bailey et al., 2017) partitions time into \(S\) even submatrices and forms all \(C = \binom{S}{S/2}\) training (In-Sample, \(J\)) and testing (Out-of-Sample, \(\bar{J}\)) combinations:

1. For each split \(c \in \{1, \dots, C\}\), compute the vector of In-Sample Sharpe ratios \(R_c \in \mathbb{R}^M\) on \(J_c\) and Out-of-Sample Sharpe ratios \(\bar{R}_c \in \mathbb{R}^M\) on \(\bar{J}_c\).
2. Select the In-Sample optimal strategy \(m^* = \arg\max_m R_{c, m}\) and find its relative rank \(\omega_c \in (0, 1)\) within the Out-of-Sample distribution \(\bar{R}_c\).
3. Define the **Rank Logit** \(\lambda_c\):

\[
\lambda_c = \ln\!\left(\frac{\omega_c}{1 - \omega_c}\right)
\]

When \(\lambda_c \le 0\), the strategy selected as best In-Sample performs worse than the median strategy Out-of-Sample. The **Probability of Backtest Overfitting (PBO)** is the empirical probability mass in the left tail:

\[
\text{PBO} = \frac{1}{C} \sum_{c=1}^C \mathbb{I}\!\left(\lambda_c \le 0\right)
\]

### Combinatorial Purged Cross-Validation (`CPCVSplitter`)

Given \(N\) chronological groups and \(k\) test groups per split, there are \(\binom{N}{k}\) train/test splits with purge window \(p\) and embargo window \(e\) applied around every test block boundary. Because each group appears in exactly \(\varphi(N, k) = \binom{N - 1}{k - 1}\) test sets across all splits, the out-of-sample predictions can be reassembled into \(\varphi(N, k)\) full-horizon backtest paths rather than a single point estimate (López de Prado, 2018).
