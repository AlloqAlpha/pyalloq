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
\min_{y \geq 0} \; \frac{1}{2} y^\top \Sigma y - \sum_{i=1}^{N} \ln y_i
\]

Then normalize: \(w_i = y_i / \sum_j y_j\)

---

## Risk Budgeting

Generalizes ERC by assigning target risk budgets \(b_i \geq 0\) (with \(\sum b_i = 1\)) to each asset:

\[
\min_{y \geq 0} \; \frac{1}{2} y^\top \Sigma y - \sum_{i=1}^{N} b_i \ln y_i
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

## Summary Table

| Allocator | Optimization Problem | Requires Returns? |
|-----------|----------------------|-------------------|
| Equal Weight | \(w_i = 1/N\) | No |
| Min Volatility | \(\min w^\top \Sigma w\) | No |
| Max Sharpe | \(\min y^\top \Sigma y \text{ s.t. } (\mu-r_f)^\top y=1\) | Yes |
| Max Utility | \(\max \mu^\top w - \lambda w^\top \Sigma w\) | Yes |
| Risk Parity | \(\min \frac{1}{2}y^\top\Sigma y - \sum\ln y_i\) | No |
| Risk Budgeting | \(\min \frac{1}{2}y^\top\Sigma y - \sum b_i\ln y_i\) | No |
| Max Diversification | \(\min y^\top\Sigma y \text{ s.t. } \sigma^\top y=1\) | No |
