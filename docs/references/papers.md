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

---

## Monte Carlo Simulation & Entropy Pooling

### Meucci (2008) — Fully Flexible Views: Theory and Practice {#meucci-2008}

> Meucci, A. (2008). Fully Flexible Views: Theory and Practice. *Risk*, 21(10), 97–102.
> [https://ssrn.com/abstract=1213325](https://ssrn.com/abstract=1213325)

**Contribution:** Introduced Entropy Pooling to blend arbitrary non-Gaussian Monte Carlo distributions with linear equality and inequality views on returns, volatilities, and correlations via Relative Entropy (Kullback-Leibler) minimization solved in the low-dimensional Lagrange dual space.

**Key formula:**
\[
x^* = \arg\min_{x \in \Delta^M} \sum_{m=1}^M x_m \ln\!\left(\frac{x_m}{p_m}\right) \quad \text{s.t.} \quad A x \le b, \; C x = d
\]

**Implemented in:** `EntropyPooler`, `ViewBuilder` (`AbsoluteView`, `RelativeView`, `VolatilityView`, `CovarianceView`)

---

### Barone-Adesi, Giannopoulos & Vosper (1999) — Filtered Historical Simulation {#fhs-1999}

> Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999). VaR Without Correlations for Portfolios of Derivative Securities. *Journal of Futures Markets*, 19(5), 583–602.

**Contribution:** Combined parametric GARCH volatility filtration with non-parametric historical residual bootstrapping, preserving empirical fat tails and cross-asset co-movement while adapting dynamically to current volatility regimes.

**Implemented in:** `FHSGenerator`

---

### McNeil & Frey (2000) & Demarta & McNeil (2005) — GARCH-EVT & Student-\(t\) Copula {#garch-evt-2000}

> McNeil, A. J., & Frey, R. (2000). Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series: An Extreme Value Approach. *Journal of Empirical Finance*, 7(3–4), 271–300.
>
> Demarta, S., & McNeil, A. J. (2005). The \(t\) Copula and Related Copulas. *International Statistical Review*, 73(1), 111–129.

**Contribution:** Modeled conditional heteroscedasticity via AR-GARCH filtration, fitted Generalized Pareto Distributions (GPD) to extreme tail exceedances, and coupled marginals with a Student-\(t\) Copula to capture asymmetric tail risk and positive lower-tail crash dependence (\(\lambda_L > 0\)).

**Implemented in:** `GARCHEVTCopulaGenerator`

---

## Backtest Overfitting & Combinatorial Validation

### Bailey, Borwein, López de Prado & Zhu (2017) — Probability of Backtest Overfitting {#pbo-2017}

> Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4), 39–69.
> [https://ssrn.com/abstract=2326253](https://ssrn.com/abstract=2326253)

**Contribution:** Introduced the non-parametric Combinatorial Symmetric Cross-Validation (CSCV) framework to estimate the probability that a strategy configuration selected for optimal In-Sample performance underperforms the median Out-of-Sample (\(\text{PBO} = \mathbb{P}[\lambda_c \le 0]\)).

**Implemented in:** `PBOEvaluator`, `plot_pbo_distribution`, `RobustnessTearSheet`

---

### López de Prado (2018) — Combinatorial Purged Cross-Validation {#cpcv-2018}

> López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Chapter 12: Backtesting through Cross-Validation).

**Contribution:** Formulated Combinatorial Purged Cross-Validation (CPCV) to eliminate look-ahead leakage via purging and embargoing while generating \(\varphi(N, k) = \binom{N-1}{k-1}\) complete out-of-sample backtest paths across recombined market regimes.

**Implemented in:** `CPCVSplitter`, `plot_cpcv_paths`, `RobustnessTearSheet`
