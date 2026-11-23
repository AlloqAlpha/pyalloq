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

### Das et al. (2024), Mistral AI (2026) & Potapczynski et al. (2026) — Time-Series Foundation Models & Prior-Data Fitted Networks (TimesFM-3, Timestral & ApolloPFN) {#tsfm-2026}

>Potapczynski, A., Selvam, R. K., Konstantinova, T., Wolff, M., Olivares, K. G., Ma, R., Mahoney, M. W., Wilson, A. G., Oreshkin, B. N., & Efimov, D. (2026). Time-Aware Prior Fitted Networks for Zero-Shot Forecasting with Exogenous Variables (`ApolloPFN`). *Transactions on Machine Learning Research (TMLR, 06/2026)*.
> [https://doi.org/10.48550/arxiv.2603.15802](https://doi.org/10.48550/arxiv.2603.15802)
>
> Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model (`TabPFN-v2`). *Nature*, 637, 319–326.
>
> Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). A Decoder-Only Foundation Model for Time-Series Forecasting (`TimesFM`). *International Conference on Machine Learning (ICML)*.
>
> Mistral AI (2026). Timestral: Multivariate Probabilistic Time-Series Foundation Models with Decoupled Horizon Forecasting.
> [https://github.com/mistralai/timestral](https://github.com/mistralai/timestral)

**Contribution:** Pretrained patch-based causal transformers (`TimesFM-3`, `Timestral`) and Time-Aware Prior-Data Fitted Networks (`ApolloPFN`, trained on `SRNGN` Algorithm 1 causal graphs with `AttnFeat` + RoPE `AttnSamp` and Riemann Bar distributions) that execute zero-shot Bayesian inference on multivariate time series with past-only and known-future exogenous covariates (`CovariateKey`, `CovariateMode`). Combined via inverse-variance and `sCRPS`-adaptive Bayesian Model Averaging in `EnsembleReturnEstimator` and `EnsembleViewGenerator`.

**Implemented in:** `ApolloPFNReturnEstimator`, `ApolloPFNViewGenerator`, `ApolloPFNTransformer`, `ApolloPFNPretrainer`, `SRNGNPriorGenerator`, `RiemannBarDistribution`, `EnsembleReturnEstimator`, `EnsembleViewGenerator`, `TimesFM3UnivariateReturnEstimator`, `TimesFM3MultivariateReturnEstimator`, `TimesFM3UnivariateViewGenerator`, `TimesFM3MultivariateViewGenerator`, `TimestralUnivariateReturnEstimator`, `TimestralMultivariateReturnEstimator`, `TimestralUnivariateViewGenerator`, `TimestralMultivariateViewGenerator`

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

## Generative Diffusion, Neural SDEs & Wasserstein Distributionally Robust Optimization

### Li (2023) — Wasserstein-Kelly Portfolios {#li-2023}

> Li, J. Y.-M. (2023). Wasserstein-Kelly Portfolios: A Robust Data-Driven Solution to Optimize Portfolio Growth. *arXiv*.
> [https://doi.org/10.48550/arxiv.2302.13979](https://doi.org/10.48550/arxiv.2302.13979)

**Contribution:** Formulated Distributionally Robust Kelly (log-optimal growth) portfolio optimization over a Wasserstein ball around the empirical distribution, proving tractable convex dual reformulations that prevent the extreme leverage and parameter sensitivity of classical Kelly portfolios.

**Implemented in:** `WassersteinDROAllocator` (`ObjectiveFunction.WASSERSTEIN_KELLY`)

---

### Fonseca & Junca (2021) & Esfahani & Kuhn (2018) — Wasserstein DRO with Expected Value Constraints {#fonseca-junca-2021}

> Fonseca, D., & Junca, M. (2021). Wasserstein Distributionally Robust Optimization with Expected Value Constraints. *arXiv*.
> [https://doi.org/10.48550/arxiv.2111.04663](https://doi.org/10.48550/arxiv.2111.04663)
>
> Mohajerin Esfahani, P., & Kuhn, D. (2018). Data-driven distributionally robust optimization using the Wasserstein metric: Performance guarantees and tractable reformulations. *Mathematical Programming*, 171(1–2), 115–166.

**Contribution:** Derived strong duality and finite convex program reformulations for worst-case expectation and CVaR minimization over Wasserstein balls intersected with linear expected-value constraints \(\mathbb{E}_{\mathbb{Q}}[G r] = q\), preserving all empirical data atoms while anchoring first-moment views.

**Implemented in:** `WassersteinDROAllocator` (`ObjectiveFunction.WASSERSTEIN_CVAR`, `ObjectiveFunction.WASSERSTEIN_MEAN_RISK`, `view_matrix`, `view_targets`)

---

### Cohen, Reisinger & Wang (2021) & Kidger et al. (2021) — Arbitrage-Free Neural SDE Market Models {#cohen-2021}

> Cohen, S. N., Reisinger, C., & Wang, S. (2021). Arbitrage-free neural-SDE market models. *arXiv*.
> [https://doi.org/10.48550/arxiv.2105.11053](https://doi.org/10.48550/arxiv.2105.11053)
>
> Kidger, P., Foster, J., Li, X., & Lyons, T. J. (2021). Neural SDEs as Infinite-Dimensional GANs. *International Conference on Machine Learning (ICML)*.

**Contribution:** Parameterized continuous-time asset price and implied-volatility factor SDEs with neural drift and diffusion coefficients subject to exact martingale / market-price-of-risk and calendar-spread no-arbitrage constraints, calibrated via pathwise rough-path signature discriminators.

**Implemented in:** `NeuralSDEGenerator`, `NeuralSDEEstimator`, `ArbitrageFreeNeuralSDENet`

---

### Factor-Based Conditional Diffusion & Generative WDRO (2025–2026) {#diffusion-wdro-2026}

> Factor-Based Conditional Diffusion Model for Portfolio Optimization (2025). *arXiv*.
> [https://doi.org/10.48550/arxiv.2509.22088](https://doi.org/10.48550/arxiv.2509.22088)
>
> Generative Models for Decision-Making under Distributional Shift (2026). *arXiv*.
> [https://doi.org/10.48550/arxiv.2604.04342](https://doi.org/10.48550/arxiv.2604.04342)

**Contribution:** Combined factor-bottleneck conditional denoising diffusion models (with classifier-free guidance and DDIM sampling) with Wasserstein Distributionally Robust Optimization (Generative-WDRO), centering the Wasserstein ball on conditionally synthesized forward scenarios while scaling the ambiguity radius by regime shift and generator signature-MMD discrepancy.

**Implemented in:** `ConditionalDiffusionGenerator`, `WassersteinDROAllocator(scenario_generator=...)`

---

### Exploratory Mean-Variance Portfolio Optimization with Regime-Switching (2026) {#exploratory-mv-2026}

> Exploratory Mean-Variance Portfolio Optimization with Regime-Switching (2026). *arXiv*.
> [https://doi.org/10.48550/arxiv.2501.16659](https://doi.org/10.48550/arxiv.2501.16659)

**Contribution:** Formulated continuous-time entropy-regularized ("exploratory") reinforcement learning for mean-variance portfolio control under latent Markov regime switching, combining continuous-time Wonham regime belief filtering with analytical Gaussian exploration covariance \(\Sigma_{\pi, t}^* = (\lambda_{\text{ent}}/\gamma)(\bar{\Sigma}(p_t) + \kappa_{\text{turn}} I)^{-1}\).

**Implemented in:** `ExploratoryRegimeMVAllocator`, `WonhamRegimeFilter`

---

### Potapczynski et al. (2026) & Müller et al. (2022) — Time-Aware Prior-Data Fitted Networks (`ApolloPFN`) & Riemann Bar Posteriors {#apollopfn-2026}

> Potapczynski, A., Finzi, M., Goldblum, M., & Wilson, A. G. (2026). Time-Aware Prior-Data Fitted Networks for Time Series Forecasting (`ApolloPFN`). *Transactions on Machine Learning Research (TMLR)*.
> [https://doi.org/10.48550/arxiv.2603.15802](https://doi.org/10.48550/arxiv.2603.15802)
>
> Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., & Hutter, F. (2022). Transformers Can Do Bayesian Inference. *International Conference on Learning Representations (ICLR)*.
> [https://doi.org/10.48550/arxiv.2112.10510](https://doi.org/10.48550/arxiv.2112.10510)

**Contribution:** Introduced synthetic Structural Causal Model (SCM) prior meta-training via Single Root Node Random Growing Networks (`SRNGNPriorGenerator`, Algorithm 1), factorized cross-variate (`AttnFeat`) and Rotary Position Embedded temporal (`AttnSamp`) attention, and the piecewise-constant **Riemann Bar Posterior Predictive Distribution** with half-normal outer tails—enabling exact closed-form expectations, quantiles, Expected Shortfall (\(\text{CVaR}_\alpha\)), Sortino downside semi-deviation, and Bowley skewness in a single zero-shot forward pass.

**Implemented in:** `SRNGNPriorGenerator`, `ApolloPFNTransformer`, `RiemannBarDistribution`, `ApolloPFNPretrainer`, `ApolloPFNReturnEstimator`, `ApolloPFNViewGenerator`, `EnsembleReturnEstimator`, `EnsembleViewGenerator`

---

### Blanchet, Kang & Murthy (2019) & Patton (2006) — Anisotropic Wasserstein Geometry & Conditional Copula Tail Integration {#blanchet-2019}

> Blanchet, J., Kang, Y., & Murthy, K. (2019). Robust Wasserstein Profile Inference and Applications to Machine Learning. *Journal of Applied Probability*, 56(3), 830–857.
> [https://doi.org/10.48550/arxiv.1610.05627](https://doi.org/10.48550/arxiv.1610.05627)
>
> Patton, A. J. (2006). Modelling Asymmetric Exchange Rate Dependence. *International Economic Review*, 47(2), 527–556.

**Contribution:** Established anisotropic optimal-transport Mahalanobis cost geometries \(c(\xi, \hat{\xi}) = \|\Sigma_{\text{tail}}^{-1/2}(\xi - \hat{\xi})\|_2\) that weight adversarial perturbations according to directional predictive uncertainty, and conditional Student-\(t\) copula factorizations coupling non-Gaussian marginal inverse CDFs \(F_{\text{bar}, i}^{-1}(u_i)\) with forward-looking option-implied correlations.

**Implemented in:** `WassersteinDROAllocator(ground_metric="quantile_mahalanobis")`, `ApolloPFNReturnEstimator(copula_type="student_t" | "stratified_t")`, `ApolloPFNViewGenerator(uncertainty_mode="cvar_tail" | "downside_semivariance")`

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

---

## Permutation-Invariant Set Transformers, Scalable-Softmax & Equivariant Portfolio Policies

### Lee, Lee, Kim, Kosiorek, Choi & Teh (2019) — Set Transformer {#lee-2019}

> Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97, 3744–3753.
> [https://proceedings.mlr.press/v97/lee19d/lee19d.pdf](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf)

**Contribution:** Formulated Multihead Attention Blocks (`MAB`), Set Attention Blocks (`SAB`), linear-time \(\mathcal{O}(Nm)\) Induced Set Attention Blocks (`ISAB`) via \(m\) trainable inducing points, and Pooling by Multihead Attention (`PMA`) as universal approximators of continuous permutation-equivariant and permutation-invariant functions on variable-cardinality sets.

**Implemented in:** `MAB`, `SAB`, `ISAB`, `PMA`, `SetTransformerBackbone`, `SetTransformerCovarianceNet`, `SetTransformerPolicyNet`

---

### Nakanishi (2025) & Martins & Astudillo (2016) — Scalable-Softmax (`SSMax`) & Sparsemax {#ssmax-2025}

> Nakanishi, K. M. (2025). Scalable-Softmax Is Superior for Attention. *arXiv*.
> [https://doi.org/10.48550/arxiv.2501.19399](https://doi.org/10.48550/arxiv.2501.19399)
>
> Martins, A. F. T., & Astudillo, R. F. (2016). From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification. *International Conference on Machine Learning (ICML)*.

**Contribution:** Proved that standard Softmax suffers from attention fading as set cardinality \(N\) scales, and introduced logarithmic logit scaling \(\operatorname{SSMax}(z_i; N, s) \propto \exp((s \ln N) z_i)\) alongside Euclidean simplex projections (`Sparsemax` / `CappedSimplexProjection`) to preserve concentration across 50-to-500 asset cardinality shifts.

**Implemented in:** `ScalableMultiheadAttention`, `CappedSimplexProjection`, `UniverseSubsampler`, `plot_set_transformer_diagnostics`

---

### Lin et al. (2021) & Bongiorno, Manolakis & Mantegna (2026) — Deep Risk Factor & RIE Spectral Covariance {#deep-cov-2026}

> Lin, H., Zhou, D., Liu, W., & Bian, J. (2021). Deep Risk Model: A Deep Learning Solution for Mining Latent Risk Factors to Improve Covariance Matrix Estimation. *ACM International Conference on AI in Finance (ICAIF)*.
> [https://doi.org/10.48550/arxiv.2107.05201](https://doi.org/10.48550/arxiv.2107.05201)
>
> Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2026). End-to-End Large Portfolio Optimization for Variance Minimization. *arXiv*.
> [https://doi.org/10.48550/arxiv.2507.01918](https://doi.org/10.48550/arxiv.2507.01918)

**Contribution:** Formulated strictly congruence-equivariant low-rank factor-plus-diagonal PSD covariance heads (\(\hat{\Sigma}_t = B_t \Sigma_{F,t} B_t^\top + D_t\) with exact \(\mathcal{O}(Nm^2 + m^3)\) Woodbury inversion) and rotation-invariant eigenvalue spectral cleaning (`RIESpectralHead`) that generalize zero-shot across universe sizes.

**Implemented in:** `ISABFactorDiagonalHead`, `EquivariantGramCorrHead`, `RIESpectralHead`, `SetTransformerCovarianceNet`, `WoodburyGaussianNLLLoss`, `WoodburySteinLoss`

---

### Wood, Roberts & Zohren (2026), Xue & Ye (2025) & Ying et al. (2021) — DeePM, Masked Dirichlet RL & Graphormer Priors {#deepm-2026}

> Wood, K., Roberts, S., & Zohren, S. (2026). DeePM: Regime-Robust Deep Learning for Systematic Macro Portfolio Management. *arXiv*.
> [https://doi.org/10.48550/arxiv.2601.05975](https://doi.org/10.48550/arxiv.2601.05975)
>
> Xue, J., & Ye, Y. (2025). Cross-Sectional Transformer Reinforcement Learning for Dynamic Constituent Portfolios. *arXiv*.
> [https://doi.org/10.48550/arxiv.2510.06466](https://doi.org/10.48550/arxiv.2510.06466)
>
> Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., & Liu, T.-Y. (2021). Do Transformers Really Perform Bad for Graph Representation? (`Graphormer`). *Advances in Neural Information Processing Systems (NeurIPS)*, 34.

**Contribution:** Introduced Vectorized Variable Selection Networks (`V-VSN`), the Directed Delay (\(t-1\)) causal sieve for asynchronous global market closes, differentiable `SoftMinEVaRLoss`, ragged binary tradability masking without price imputation, Masked Dirichlet + Cash PPO Actor-Critic policies, and Graphormer structural attention biases.

**Implemented in:** `VectorizedVariableSelectionNetwork`, `SetTransformerBackbone(directed_delay=True)`, `SoftMinEVaRLoss`, `MaskedDirichletCashHead`, `DollarNeutralStatArbHead`, `VolTargetedContinuousRiskHead`, `SetTransformerPolicyNet`, `MarketDataset(return_mask=True)`, `SB3PortfolioAgent(policy_backbone="set_transformer")`

---

## Conformal Risk, Online Meta-Weighting, Jump Regimes & Relevance

### Angelopoulos et al. (2023–2024), Kato (2025) & Zarei et al. (2026) — Conformal PID Control & Safety-Gated Portfolios {#conformal-risk-2026}

> Angelopoulos, A. N., Candès, E. J., & Tibshirani, R. J. (2023). Conformal PID Control for Time Series Prediction. *Advances in Neural Information Processing Systems (NeurIPS)*.
>
> Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal Risk Control. *International Conference on Learning Representations (ICLR)*.
>
> Kato, M. (2025). Conformal Predictive Portfolio Selection (`CPPS`). *arXiv:2410.16333*.
>
> Zarei, M., et al. (2026). Gated Conformal Safety Risk Certificates for Portfolio Control (`Gated-CSRC`). *arXiv:2611.14699*.

**Contribution:** Distribution-free finite-sample coverage and expected risk bounds for non-exchangeable financial time series, enabling conformal Kelly sizing, two-stage HR-LR portfolio selection, and convex Euclidean projection onto certified tail-loss sets.

**Implemented in:** `ConformalPIDCalibrator`, `ConformalRiskController`, `CPPSAllocator`, `ConformalKellyOverlay`, `ConformalSafetyGateAllocator`, `RiskCertificate`

---

### Online Meta-Weighting, BPDS & Elliptical Shrinkage {#meta-weighting-2025}

> de Rooij, S., van Erven, T., Grünwald, P. D., & Koolen, W. M. (2014). Follow the Leader if You Can, Hedge if You Must (`AdaHedge`). *JMLR*, 15(37), 1281–1316.
>
> Luo, H., & Schapire, R. E. (2015). Achieving All with No Parameters: `AdaNormalHedge`. *COLT*.
>
> Jun, K.-S., Orabona, F., Wright, S., & Willett, R. (2017). Improved Strongly Adaptive Online Learning using Coin Betting (`CBCE`). *AISTATS*.
>
> Tallman, E., & West, M. (2024). Bayesian Predictive Decision Synthesis (`BPDS`). *Journal of the Royal Statistical Society Series B*.
>
> Kan, R., & Lassance, N. (2025). Optimal Portfolio Shrinkage under Elliptical Distributions. *Journal of Financial and Quantitative Analysis (JFQA)*.

**Contribution:** Adaptive no-regret and Bayesian utility-tilted meta-combination rules for blending multiple portfolio allocators with switching-cost control and elliptical kurtosis \(\kappa\) shrinkage toward \(1/N\).

**Implemented in:** `AdaHedgeStrategy`, `FixedShareStrategy`, `CBCEStrategy`, `AdaNormalHedgeStrategy`, `DualScaledSwitchingCostStrategy`, `CADTSBanditStrategy`, `BPDSMetaStrategy`, `EllipticalShrinkageStrategy`, `EnsembleAllocator`

---

### Nystrup et al. (2021), Aydınhan et al. (2024) & Shu, Yu & Mulvey (2025) — Statistical Jump Models {#jump-models-2025}

> Nystrup, P., Lindström, E., Pinson, P., & Madsen, H. (2021). Temporal Regularization in Regime-Switching Models (`Sparse Jump Model`). *Quantitative Finance*.
>
> Aydınhan, A. O., Kolm, P. N., Mulvey, J. M., & Shu, Y. (2024). Identifying Patterns in Financial Markets: Extending the Statistical Jump Model for Regime Identification (`Continuous Jump Model`). *Annals of Operations Research*.
>
> Shu, Y., Yu, C., & Mulvey, J. M. (2025). Dynamic Asset Allocation with Asset-Specific Regime Forecasts. *Journal of Portfolio Management*.

**Contribution:** Exact Bellman dynamic programming and continuous simplex-trajectory jump models with \(\ell_1/\ell_2\) feature sparsity that eliminate HMM switch churn and sample starvation.

**Implemented in:** `SparseJumpModelDetector`, `ContinuousJumpModelDetector`, `AssetSpecificJumpViewGenerator`, `RegimeSwitchingAllocator`

---

### Czasonis, Kritzman & Turkington (2023–2026) — Relevance-Based Prediction {#relevance-2026}

> Czasonis, M., Kritzman, M., & Turkington, D. (2023). *Prediction Revisited: The Importance of Observation*. Wiley.

**Contribution:** Replaces arbitrary rolling windows with Mahalanobis similarity and informativeness weights (`relevance`) plus an analytical `adjusted_fit` shrinkage factor.

**Implemented in:** `compute_mahalanobis_relevance_weights`, `RelevanceConditionedReturnEstimator`, `RelevanceConditionedCovarianceEstimator`
