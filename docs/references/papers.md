# References & Papers

PyAlloq's algorithms are grounded in seminal academic research. This page provides full citations, summaries, and links to the papers that inspired each component.

---

## Portfolio Optimization

### Markowitz (1952) — Portfolio Selection {#markowitz-1952}

> Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.
> [https://doi.org/10.1111/j.1540-6261.1952.tb01525.x](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x)

**Contribution:** Foundational mean-variance framework. Introduced the concept of the efficient frontier — the set of portfolios offering maximum expected return for a given level of risk.

**Key formula:**
\[
\min_w w^\top \Sigma w \quad \text{s.t.} \quad \mu^\top w = \mu^*, \; \mathbf{1}^\top w = 1
\]

**Implemented in:** `MarkowitzAllocator` (MIN_VOLATILITY, MAX_SHARPE, MAX_RETURN)

---

### Maillard, Roncalli & Teïletche (2010) — Equal Risk Contributions {#erc-2010}

> Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of Equally Weighted Risk Contributions Portfolios. *The Journal of Portfolio Management*, 36(4), 60–70.
> [https://doi.org/10.3905/jpm.2010.36.4.060](https://doi.org/10.3905/jpm.2010.36.4.060)

**Contribution:** Defined the Equal Risk Contribution (ERC) portfolio — a practical middle ground between Equal Weight and Minimum Variance. Showed that ERC portfolios can be computed via a convex optimization problem.

**Key formula:**
\[
w_i (\Sigma w)_i = w_j (\Sigma w)_j \quad \forall i, j
\]

**Implemented in:** `RiskParityAllocator`, `RiskBudgetingAllocator`

---

### Choueifaty & Coignard (2008) — Maximum Diversification {#max-div-2008}

> Choueifaty, Y., & Coignard, Y. (2008). Toward Maximum Diversification. *The Journal of Portfolio Management*, 35(1), 40–51.
> [https://doi.org/10.3905/JPM.2008.35.1.40](https://doi.org/10.3905/JPM.2008.35.1.40)

**Contribution:** Introduced the Diversification Ratio as a measure of portfolio diversification. The Maximum Diversification portfolio maximizes this ratio.

**Key formula:**
\[
D(w) = \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}}
\]

**Implemented in:** `MaxDiversificationAllocator`

---

## Hierarchical Methods

### López de Prado (2016) — Hierarchical Risk Parity {#hrp-2016}

> López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample. *The Journal of Portfolio Management*, 42(4), 59–69.
> [https://doi.org/10.3905/jpm.2016.42.4.059](https://doi.org/10.3905/jpm.2016.42.4.059)
> [arXiv:1603.05831](https://arxiv.org/abs/1603.05831)

**Contribution:** Proposed HRP — a machine learning approach to portfolio construction that avoids covariance matrix inversion. Uses hierarchical clustering and recursive bisection to allocate weights in a tree-like structure.

**Steps:** Distance matrix → Hierarchical clustering → Quasi-diagonalization → Recursive bisection

**Implemented in:** `HRPAllocator`

---

## Return Estimation

### Black & Litterman (1992) — Global Portfolio Optimization {#bl-1992}

> Black, F., & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*, 48(5), 28–43.
> [https://doi.org/10.2469/faj.v48.n5.28](https://doi.org/10.2469/faj.v48.n5.28)

**Contribution:** Combined CAPM equilibrium returns with subjective investor views through Bayesian updating. Produces more stable and intuitive return estimates than pure historical means.

**Key formula:**
\[
\mu^* = \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
        \left[(\tau\Sigma)^{-1} \Pi + P^\top \Omega^{-1} Q\right]
\]

**Implemented in:** `BlackLittermanEstimator`

---

## Covariance Estimation

### Ledoit & Wolf (2004) — Shrinkage Estimation {#ledoit-wolf-2004}

> Ledoit, O., & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix. *The Journal of Portfolio Management*, 30(4), 110–119.
> [https://doi.org/10.3905/jpm.2004.110](https://doi.org/10.3905/jpm.2004.110)

**Contribution:** Derived an analytically optimal shrinkage intensity for the sample covariance matrix, providing a closed-form, well-conditioned estimator that improves out-of-sample performance for Markowitz optimization.

**Key formula:**
\[
\hat{\Sigma}^{LW} = (1 - \alpha^*) \hat{\Sigma} + \alpha^* \hat{\mu}_{cov} I
\]

**Implemented in:** `LedoitWolfEstimator`

---

## Transaction Cost Models

### Almgren & Chriss (2001) — Optimal Execution {#almgren-chriss-2001}

> Almgren, R., & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. *Journal of Risk*, 3, 5–39.
> [https://doi.org/10.21314/JOR.2001.041](https://doi.org/10.21314/JOR.2001.041)

**Contribution:** Derived an optimal trade execution strategy that minimizes total transaction costs (market impact + risk) for a given liquidation horizon. The market impact cost model is widely used in institutional portfolio management.

**Market impact formula:**
\[
\text{impact}_i = \gamma \sqrt{\frac{\text{trade\_size}_i}{\text{daily\_volume}_i}}
\]

**Implemented in:** `AlmgrenChrissCostModel`
