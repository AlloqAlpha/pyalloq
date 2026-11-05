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

---

## Multi-Period Portfolio Optimization, Decision-Focused Learning & FinRL

### Boyd, Busseti, Diamond, Kahn, Koh, Nystrup & Speth (2017) — Multi-Period Trading via Convex Optimization {#boyd-2017}

> Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P., & Speth, J. (2017). Multi-Period Trading via Convex Optimization. *Foundations and Trends in Optimization*, 3(1), 1–76.
> [https://doi.org/10.1561/2400000023](https://doi.org/10.1561/2400000023)

**Contribution:** Unified multi-period portfolio optimization under a receding-horizon Model Predictive Control (MPC) convex program incorporating proportional bid-ask spread costs (\(\ell_1\)), square-root market impact (\(|z|^{3/2}\) power cones), holding costs, and terminal risk penalties.

**Implemented in:** `MultiPeriodMPCAllocator`, `StochasticScenarioMPCAllocator`

---

### Gârleanu & Pedersen (2013) — Dynamic Trading with Predictable Returns and Transaction Costs {#garleanu-pedersen-2013}

> Gârleanu, N., & Pedersen, L. H. (2013). Dynamic Trading with Predictable Returns and Transaction Costs. *The Journal of Finance*, 68(6), 2309–2340.
> [https://doi.org/10.1111/jofi.12080](https://doi.org/10.1111/jofi.12080)

**Contribution:** Proved that under multi-horizon mean-reverting alpha signals and quadratic/permanent price impact, the optimal portfolio trades partially at a constant rate toward an intertemporal **Aim Portfolio** — a discounted weighted average of future Markowitz target portfolios.

**Implemented in:** `AlphaDecayReturnEstimator`, `MultiPeriodMPCAllocator`

---

### Elmachtoub & Grigas (2022) & Agrawal et al. (2019) — Decision-Focused Learning & Differentiable Convex Layers {#dfl-2022}

> Elmachtoub, A. N., & Grigas, P. (2022). Smart "Predict, then Optimize". *Management Science*, 68(1), 9–26.
>
> Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S., & Kolter, J. Z. (2019). Differentiable Convex Optimization Layers. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

**Contribution:** Formulated the convex SPO+ regret surrogate loss and implicit differentiation through fixed-point / KKT optimality conditions so neural predictors are trained directly on downstream multi-period decision utility.

**Implemented in:** `MDFPOptimizationLayer`, `SPOPlusLoss`, `HybridDFLLoss`, `IPMOAllocator`

---

### Buehler, Gonon, Teichmann & Wood (2019) & Liu et al. (2022) — Deep Hedging & FinRL {#deep-hedging-2019}

> Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep Hedging. *Quantitative Finance*, 19(8), 1271–1291.
>
> Liu, X.-Y., Xia, Z., Rui, J., Gao, J., Yang, H., Zhu, M., Wang, C., Wang, Z., & Guo, J. (2022). FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 35.

**Contribution:** Formulated recurrent pathwise trajectory optimization under proportional/convex market frictions and standardized Gymnasium environments with Mahalanobis turbulence circuit-breakers for continuous portfolio control.

**Implemented in:** `PortfolioGymEnv`, `ScenarioPortfolioEnv`, `TrajectoryPolicyNet`, `DeepHedgingTrajectoryTrainer`, `BayesianHRPAllocator`, `RLAllocator`

---

### Moroke et al. (2026) — Thermodynamic & Bures-Wasserstein Regime Gating {#moroke-2026}

> Moroke, N., et al. (2026). Thermodynamic Efficiency Bounds and Wasserstein Optimal Transport in Non-Equilibrium Portfolio Rebalancing.

**Contribution:** Modeled portfolio rebalancing between fast and slow market covariance baths as a stochastic heat engine bounded by Carnot efficiency \(\eta_{\text{Carnot}} = 1 - T_{\min}/T_{\max}\), using the Bures-Wasserstein \(W_2\) distance to gate frictional turnover during structural regime ruptures.

**Implemented in:** `WassersteinRegimeGate`, `ThermodynamicDiagnostics`

---

## Multi-Strategy Evaluation, Deflated Sharpe Ratio & Performance Attribution

### Bailey & López de Prado (2014) — The Deflated Sharpe Ratio {#dsr-2014}

> Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *The Journal of Portfolio Management*, 40(5), 94–107.
> [https://doi.org/10.3905/jpm.2014.40.5.094](https://doi.org/10.3905/jpm.2014.40.5.094)

**Contribution:** Formulated the Deflated Sharpe Ratio (DSR) to correct for selection bias under multiple testing (\(N\) trials) and non-Normal return distributions (skewness \(\hat{\gamma}_3\) and kurtosis \(\hat{\gamma}_4\)), deriving the analytical expected maximum Sharpe hurdle \(\widehat{\text{SR}}_0\) via extreme value theory.

**Key formula:**
\[
\widehat{\text{SR}}_0 = \sqrt{\text{Var}[\{\widehat{\text{SR}}_n\}]} \left( (1 - \gamma)\,\Phi^{-1}\!\left(1 - \frac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \frac{1}{N e}\right) \right)
\]

**Implemented in:** `BacktestTracker`, `MultiStrategyComparator` (`BenchmarkSuite`), `MetricsTearSheet`, `compute_deflated_sharpe_ratio`, `ComparisonTearSheet`

---

### Brinson & Fachler (1985) & Carino (1999) — Active Performance Attribution {#brinson-fachler-1985}

> Brinson, G. P., & Fachler, N. (1985). Measuring Non-U.S. Equity Portfolio Performance. *The Journal of Portfolio Management*, 11(3), 73–76.
> [https://doi.org/10.3905/jpm.1985.409005](https://doi.org/10.3905/jpm.1985.409005)
>
> Carino, D. R. (1999). Combining Attribution Effects Over Time. *The Journal of Performance Measurement*, 3(4), 5–14.

**Contribution:** Decomposed active return \(R_p - R_b\) into Allocation, Selection, and Interaction effects across sectors or asset clusters, with logarithmic multi-period geometric linking so multi-period attribution sums exactly to total geometric active return.

**Implemented in:** `PerformanceAttributor`, `BrinsonAttributionResult`, `plot_attribution_waterfall`, `ComparisonTearSheet`

---

### Ma & Smith (2025) — Alpha Decay and Transaction Costs in Multi-Period Trading {#ma-smith-2025}

> Ma, C., & Smith, P. (2025). On the Effect of Alpha Decay and Transaction Costs on the Multi-period Optimal Trading Strategy. *arXiv*.
> [https://doi.org/10.48550/arxiv.2502.04284](https://doi.org/10.48550/arxiv.2502.04284)

**Contribution:** Analyzed the compounding interaction between multi-horizon alpha decay, intra-period weight drift, and linear/non-linear transaction costs when transitioning from theoretical paper signals to executed multi-period trajectories.

**Implemented in:** `PerformanceAttributor.compute_frictional_drag`, `FrictionalDragResult`, `WalkForwardEngine`, `plot_attribution_waterfall`
