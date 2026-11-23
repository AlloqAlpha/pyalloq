# Portfolio Theory

This page documents the mathematical formulations behind each classical allocator implemented in `pyalloq`.

---

## Markowitz Mean-Variance Optimization

Markowitz (1952) introduced the canonical portfolio optimization framework: find the weights \(w\) that minimize portfolio variance for a given level of expected return.

### Minimum Variance

\[
\min_{w} \; w^\top \Sigma w
\quad \text{subject to} \quad
\mathbf{1}^\top w = 1, \quad w \geq 0
\]

where \(\Sigma \in \mathbb{R}^{N \times N}\) is the covariance matrix of asset returns.

---

### Maximum Sharpe Ratio

Directly maximizing the Sharpe ratio is non-convex. Tobin's mutual fund separation theorem allows the problem to be recast as a convex quadratic program via the **substitution trick**:

Let \(y = \kappa w\) where \(\kappa = \frac{1}{(\mu - r_f \mathbf{1})^\top w}\). Then:

\[
\max_{w} \; \frac{(\mu - r_f \mathbf{1})^\top w}{\sqrt{w^\top \Sigma w}}
\;\equiv\;
\min_{y} \; y^\top \Sigma y
\quad \text{s.t.} \quad
(\mu - r_f \mathbf{1})^\top y = 1, \quad y \geq 0
\]

Recover weights: \(w^* = y^* / \mathbf{1}^\top y^*\)

---

### Maximum Utility (Mean-Variance)

Maximize the mean-variance utility function with a risk aversion parameter \(\lambda\):

\[
\max_{w} \; \mu^\top w - \lambda \cdot w^\top \Sigma w
\quad \text{s.t.} \quad
\mathbf{1}^\top w = 1, \quad w \geq 0
\]

The `risk_aversion` field in `MarketData` controls \(\lambda\).

---

## Equal Risk Contribution (Risk Parity)

Maillard, Roncalli & Teïletche (2010) propose portfolios where every asset contributes the same amount of marginal risk to the portfolio.

The **marginal risk contribution** of asset \(i\) is:

\[
\text{MRC}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}
\]

The **total risk contribution** of asset \(i\) is:

\[
\text{TRC}_i = w_i \cdot \text{MRC}_i = \frac{w_i (\Sigma w)_i}{\sigma_p}
\]

ERC requires \(\text{TRC}_i = \text{TRC}_j\) for all \(i, j\). This is achieved by solving the equivalent convex problem:

\[
\min_{y > 0} \; \frac{1}{2} y^\top \Sigma y - \sum_{i=1}^{N} \ln y_i
\]

Then normalize: \(w_i = y_i / \sum_j y_j\)

---

## Risk Budgeting

Generalizes ERC by assigning target risk budgets \(b_i \geq 0\) (with \(\sum b_i = 1\)) to each asset:

\[
\min_{y > 0} \; \frac{1}{2} y^\top \Sigma y - \sum_{i=1}^{N} b_i \ln y_i
\]

When \(b_i = 1/N\) for all \(i\), this reduces to Risk Parity. Pass budgets via `MarketData.features["risk_budgets"]`.

---

## Maximum Diversification

Choueifaty & Coignard (2008) define the **Diversification Ratio**:

\[
D(w) = \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}} = \frac{\sum_i w_i \sigma_i}{\sigma_p}
\]

where \(\sigma_i = \sqrt{\Sigma_{ii}}\) is the individual asset volatility and \(\sigma_p = \sqrt{w^\top \Sigma w}\) is the portfolio volatility.

The maximum diversification portfolio maximizes \(D(w)\), which is equivalent (via the same substitution trick as Max Sharpe) to:

\[
\min_{y \geq 0} \; y^\top \Sigma y
\quad \text{s.t.} \quad
\sigma^\top y = 1
\]

Recover weights: \(w^* = y^* / \mathbf{1}^\top y^*\)

---

## Conditional Value-at-Risk (Rockafellar & Uryasev, 2000)

For a confidence level \(\beta \in (0, 1)\) (e.g., \(\beta = 0.95\)) and historical return matrix \(R \in \mathbb{R}^{T \times N}\), `MeanCVaRAllocator` minimizes the portfolio Expected Shortfall using the convex Rockafellar-Uryasev linear programming auxiliary formulation:

\[
\min_{w, \alpha, u \ge 0} \; \alpha + \frac{1}{(1 - \beta) T} \sum_{t=1}^T u_t
\quad \text{subject to} \quad
u_t \ge -r_t^\top w - \alpha, \quad \mathbf{1}^\top w = 1, \quad l \le w \le b
\]

At optimality, \(\alpha^*\) equals the daily Value-at-Risk (\(\text{VaR}_\beta\)) and the objective equals \(\text{CVaR}_\beta(w^*)\).

---

## Conditional Drawdown-at-Risk (Chekhlov, Uryasev & Zabarankin, 2005)

Let \(C_t(w) = \sum_{\tau=1}^t r_\tau^\top w\) denote the cumulative uncompounded portfolio return at time \(t\), and let \(M_t = \max_{1 \le \tau \le t} C_\tau(w)\) track the running high-water mark. `CDaRAllocator` minimizes the tail average of drawdowns \(D_t = M_t - C_t(w)\) at level \(\beta\):

\[
\min_{w, M, d \ge 0, \alpha, u \ge 0} \; \alpha + \frac{1}{(1 - \beta) T} \sum_{t=1}^T u_t
\]

subject to linear path-monotonicity and budget constraints \(M_1 \ge 0\), \(M_t \ge C_t(w)\), \(M_t \ge M_{t-1}\), \(d_t \ge M_t - C_t(w)\), \(u_t \ge d_t - \alpha\), \(\mathbf{1}^\top w = 1\), and \(l \le w \le b\).

---

## Wasserstein Distributionally Robust Optimization (`WassersteinDROAllocator`)

Let \(\hat{\mathbb{P}} = \sum_{i=1}^M p_i \delta_{r_i}\) denote the empirical (or synthetic scenario) distribution over \(M\) return atoms \(r_i \in \mathbb{R}^N\). To hedge against distributional misspecification and structural regime shifts, `WassersteinDROAllocator` optimizes against the worst-case measure \(\mathbb{Q}\) inside the **Expected-Value-Constrained Wasserstein Ambiguity Set** (Esfahani & Kuhn, 2018; Fonseca & Junca, 2021; Li, 2023):

\[
\mathcal{P}_{\varepsilon_t}(G, q) = \left\{ \mathbb{Q} \in \mathcal{M}(\mathbb{R}^N) \;:\; W_p(\mathbb{Q}, \hat{\mathbb{P}}) \le \varepsilon_t, \quad \mathbb{E}_{\mathbb{Q}}[G r] = q \right\}
\]

### 1. Ground Metrics & Dual Norm Regularization
For transport cost \(d(r, \hat{r}) = \|r - \hat{r}\|_{\mathcal{G}}\) and effective exposure vector \(u = w - G^\top \lambda\), Kantorovich duality yields the exact dual norm penalty \(\varepsilon_t \|u\|_{\mathcal{G}^*}\):

- **Mahalanobis (`"mahalanobis"`)**: \(d(r, \hat{r}) = \|\Sigma^{-1/2}(r - \hat{r})\|_2 \implies \|u\|_{\mathcal{G}^*} = \|\Sigma^{1/2} u\|_2\).
- **Anisotropic Quantile-Mahalanobis (`"quantile_mahalanobis"`, Blanchet, Kang & Murthy, 2019)**: Calibrates an anisotropic daily transport matrix \(\Sigma_{\text{tail}} = D_{\text{tail}} C_{\text{corr}} D_{\text{tail}}\) from daily volatility \(\sigma_i\), the inter-decile spread \((q_{0.90, i} - q_{0.10, i})\) (divided by \(\sqrt{A}\) when annualized with \(A = 252\), or without \(\sqrt{A}\) for daily empirical atoms), and negative Bowley skewness \(\mathcal{S}_{\text{Bowley}, i}\):
  \[
  \left(D_{\text{tail}}\right)_{i,i} = \frac{1}{2}\left(\sigma_i + \frac{q_{0.90, i} - q_{0.10, i}}{2.56 \sqrt{A}}\right)\sqrt{1 + \kappa_{\text{tail}} \max(0, -\mathcal{S}_{\text{Bowley}, i})}, \qquad \|u\|_{\mathcal{G}^*} = \left\|\Sigma_{\text{tail}}^{1/2} u\right\|_2
  \]
- **Euclidean (`"l2"`)**: \(d(r, \hat{r}) = \|r - \hat{r}\|_2 \implies \|u\|_{\mathcal{G}^*} = \|u\|_2\).
- **Manhattan (`"l1"`)**: \(d(r, \hat{r}) = \|r - \hat{r}\|_1 \implies \|u\|_{\mathcal{G}^*} = \|u\|_\infty\).

### 2. Dynamic Bures-Wasserstein Radius Calibration
When `radius_mode="regime_adaptive"`, the ambiguity radius \(\varepsilon_t\) scales dynamically with the Bures-Wasserstein \(W_2\) distance between fast and slow market covariance baths (via `WassersteinRegimeGate`):

\[
\varepsilon_t = \varepsilon_{\text{base}} + \kappa_{\text{regime}} \cdot W_2\!\left(\mathcal{N}(\mu_{\text{fast}}, \Sigma_{\text{fast}}), \, \mathcal{N}(\mu_{\text{slow}}, \Sigma_{\text{slow}})\right)
\]

### 3. Wasserstein-Kelly Log-Growth Formulation (Li, 2023) & Fonseca-Junca Views (2021)
Given view-projected atoms \(\tilde{r}_i = r_i + \Sigma G^\top (G \Sigma G^\top + \rho_{\text{view}} \Omega + 10^{-8} I)^{-1}(q - G \hat{\mu})\) and fractional Kelly parameter \(\eta \in [0.05, 2.0]\) (standard fractional Kelly \(\eta \in (0, 1]\)), the worst-case expected geometric log-growth program is solved as a single convex exponential-cone program over \((w, \lambda)\):

\[
\max_{w \in \mathcal{W}, \, \lambda \in \mathbb{R}^K} \; \sum_{i=1}^M p_i \ln\!\left(1 + \eta \, \tilde{r}_i^\top w\right) - \varepsilon_t \, \eta \, \|w - G^\top \lambda\|_{\mathcal{G}^*} - \rho_{\text{view}} \|\lambda\|_2^2
\]

For **`WASSERSTEIN_CVAR`**, the worst-case Expected Shortfall at confidence \(\beta\) reduces to:

\[
\min_{w \in \mathcal{W}, \, \alpha, \, s \ge 0, \, \lambda} \; \alpha + \frac{1}{1 - \beta}\left(\sum_{i=1}^M p_i s_i + \varepsilon_t \|w - G^\top \lambda\|_{\mathcal{G}^*}\right) + \rho_{\text{view}} \|\lambda\|_2^2
\quad \text{s.t.} \quad s_i \ge -\tilde{r}_i^\top w - \alpha
\]

---

## Summary Table

| Allocator | Optimization Problem | Requires Returns? |
|-----------|----------------------|-------------------|
| Equal Weight | \(w_i = 1/N\) | No |
| Inverse Volatility | \(w_i = (1/\sigma_i) / \sum_j (1/\sigma_j)\) | No |
| Market Cap Weight | \(w_i = \text{MCap}_i / \sum_j \text{MCap}_j\) | No |
| Min Volatility | \(\min w^\top \Sigma w\) | No |
| Max Sharpe | \(\min y^\top \Sigma y \text{ s.t. } (\mu-r_f\mathbf{1})^\top y=1\) | Yes |
| Max Utility | \(\max \mu^\top w - \lambda w^\top \Sigma w\) | Yes |
| Efficient Return | \(\min w^\top \Sigma w \text{ s.t. } \mu^\top w \ge r_{\text{target}}\) | Yes |
| Efficient Risk | \(\max \mu^\top w \text{ s.t. } w^\top \Sigma w \le \sigma_{\text{target}}^2\) | Yes |
| Mean-CVaR | \(\min \alpha + \frac{1}{(1-\beta)T}\sum u_t \text{ s.t. } u_t \ge -r_t^\top w - \alpha\) | Optional |
| CDaR | \(\min \text{CDaR}_\beta(w)\) over sample path drawdowns | Optional |
| Wasserstein DRO | \(\max_w \inf_{\mathbb{Q} \in \mathcal{P}_{\varepsilon_t}} \mathbb{E}_{\mathbb{Q}}[U(w^\top r)]\) (Kelly / CVaR / Mean-Risk) | Optional |
| Risk Parity | \(\min \frac{1}{2}y^\top\Sigma y - \sum\ln y_i\) | No |
| Risk Budgeting | \(\min \frac{1}{2}y^\top\Sigma y - \sum b_i\ln y_i\) | No |
| Max Diversification | \(\min y^\top\Sigma y \text{ s.t. } \sigma^\top y=1\) | No |
