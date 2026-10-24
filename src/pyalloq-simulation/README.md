# `pyalloq-simulation`

Institutional Monte Carlo scenario generation, missing-history imputation, and Meucci Entropy Pooling for the PyAlloq SDK.

## Capabilities

- **Missing Data Imputation (`FactorBackcastImputer`)**: Back-casts missing history for recent IPOs and fills interior gaps in return space via PCA factor regression and idiosyncratic volatility matching.
- **Scenario Generators (`BaseScenarioGenerator`)**:
  - `GaussianGenerator`: Multivariate Normal baseline with Ledoit-Wolf shrinkage and Cholesky factorization.
  - `PCAFactorGenerator`: Latent PCA factor simulation for high-dimensional universes.
  - `FHSGenerator`: Filtered Historical Simulation (GARCH(1,1) de-volatization + synchronous residual bootstrapping).
  - `GARCHEVTCopulaGenerator`: AR(1)-GARCH(1,1) filtration + Extreme Value Theory (Generalized Pareto Distribution tails) + Student-$t$ Copula.
  - `GMMRegimeGenerator`: $K$-regime Gaussian Mixture Model with empirical Markov transition matrix.
- **Memory & Batching**: Default `float32` tensor precision and `generate_batches(n_paths, horizon, batch_size)` streaming.
- **Meucci Entropy Pooling (`ViewBuilder`, `EntropyPooler`)**: First-moment (`AbsoluteView`, `RelativeView`) and second-moment (`VolatilityView`, `CovarianceView`) market views solved via Lagrange dual optimization.
- **Path-Dependent Risk Evaluation (`ScenarioEvaluator`)**: Posterior probability-weighted Expected Return, Volatility, Sharpe Ratio, 95% CVaR, View-Adjusted Maximum Drawdown (`view_adjusted_mdd`), and Probability of Ruin (`probability_of_ruin`).
