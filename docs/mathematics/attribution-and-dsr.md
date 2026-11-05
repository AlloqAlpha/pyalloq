# Deflated Sharpe Ratio & Performance Attribution

This page documents the mathematical foundations behind `pyalloq-backtest`'s multiple-testing correction (`BacktestTracker`, `MultiStrategyComparator`, and the Deflated Sharpe Ratio) and 3-pillar active return decomposition (`PerformanceAttributor`).

---

## 1. The Multiple Testing Problem & Deflated Sharpe Ratio (DSR)

When evaluating \(N\) candidate strategies or hyperparameter configurations on historical data, selecting the strategy with the highest sample Sharpe ratio \(\widehat{\text{SR}}\) introduces severe selection bias (the "Single-Path Illusion").

### Expected Maximum Sharpe Ratio (False Strategy Theorem)

Bailey & López de Prado (2014) prove via extreme value theory (Gumbel asymptotic distribution of Gaussian maxima) that across \(N \ge 2\) independent trials with cross-sectional Sharpe variance \(\text{Var}[\{\widehat{\text{SR}}_n\}]\) and null mean \(\mathbb{E}[\widehat{\text{SR}}_n] = \text{SR}_{\text{target}}\), the expected maximum Sharpe ratio under the null hypothesis of zero true skill is:

\[
\widehat{\text{SR}}_0 = \text{SR}_{\text{target}} + \sqrt{\text{Var}[\{\widehat{\text{SR}}_n\}]} \left( (1 - \gamma)\,\Phi^{-1}\!\left(1 - \frac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \frac{1}{N e}\right) \right)
\]

where \(\gamma \approx 0.5772156649\) is the Euler-Mascheroni constant, \(e \approx 2.71828\), and \(\Phi^{-1}(\cdot)\) is the standard normal quantile function.

---

### Non-Normal Sharpe Estimator Variance & DSR

For a return series of length \(T\) with non-annualized sample Sharpe ratio \(\widehat{\text{SR}}\), sample skewness \(\hat{\gamma}_3\), and Pearson kurtosis \(\hat{\gamma}_4 = \hat{\kappa}_{\text{excess}} + 3\), Mertens (2002) and Bailey & López de Prado (2012, 2014) show that the asymptotic standard error of \(\widehat{\text{SR}}\) is:

\[
\hat{\sigma}(\widehat{\text{SR}}) = \sqrt{\frac{1 - \hat{\gamma}_3\,\widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\,\widehat{\text{SR}}^2}{T - 1}}
\]

The **Deflated Sharpe Ratio (DSR)** is the probability that the true Sharpe ratio exceeds the multiple-testing hurdle \(\widehat{\text{SR}}_0\):

\[
\text{DSR} = \Phi\!\left( \frac{(\widehat{\text{SR}} - \widehat{\text{SR}}_0)\sqrt{T - 1}}{\sqrt{1 - \hat{\gamma}_3\,\widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\,\widehat{\text{SR}}^2}} \right) \in [0, 1]
\]

When \(N = 1\), \(\widehat{\text{SR}}_0 = \text{SR}_{\text{target}}\) and DSR reduces to the **Probabilistic Sharpe Ratio (PSR)**.

---

## 2. Classical Brinson-Fachler Decomposition

Given a universe of \(M\) assets partitioned into \(J\) disjoint sectors or hierarchical clusters \(\{S_1, \dots, S_J\}\) at period \(t\):

- **Portfolio cluster weight:** \(w_{p,j,t} = \sum_{i \in S_j} w_{p,i,t}\)
- **Benchmark cluster weight:** \(w_{b,j,t} = \sum_{i \in S_j} w_{b,i,t}\)
- **Portfolio within-cluster return:** \(R_{p,j,t} = \sum_{i \in S_j} \frac{w_{p,i,t}}{w_{p,j,t}} r_{i,t}\)
- **Benchmark within-cluster return:** \(R_{b,j,t} = \sum_{i \in S_j} \frac{w_{b,i,t}}{w_{b,j,t}} r_{i,t}\)
- **Total portfolio & benchmark return:** \(R_{p,t} = \sum_{j=1}^J w_{p,j,t} R_{p,j,t}\), \(\; R_{b,t} = \sum_{j=1}^J w_{b,j,t} R_{b,j,t}\)

Brinson & Fachler (1985) decompose the single-period active return \(R_{p,t} - R_{b,t}\) into three orthogonal economic effects for each cluster \(j\):

\[
\begin{aligned}
\text{Allocation Effect:} \quad A_{j,t} &= (w_{p,j,t} - w_{b,j,t})(R_{b,j,t} - R_{b,t}) \\
\text{Selection Effect:} \quad S_{j,t} &= w_{b,j,t}(R_{p,j,t} - R_{b,j,t}) \\
\text{Interaction Effect:} \quad I_{j,t} &= (w_{p,j,t} - w_{b,j,t})(R_{p,j,t} - R_{b,j,t})
\end{aligned}
\]

Summing across all clusters \(j = 1, \dots, J\) satisfies the exact single-period identity:

\[
\sum_{j=1}^J \left( A_{j,t} + S_{j,t} + I_{j,t} \right) = R_{p,t} - R_{b,t}
\]

---

### Carino (1999) Geometric Multi-Period Linking

Because geometric multi-period returns compound multiplicatively, \(R_p = \prod_{t=1}^T (1 + R_{p,t}) - 1\) and \(R_b = \prod_{t=1}^T (1 + R_{b,t}) - 1\), simple arithmetic sums of single-period attribution effects leave a compounding residual. Carino (1999) eliminates this residual via logarithmic linking coefficients:

\[
k_t = \frac{\ln(1 + R_{p,t}) - \ln(1 + R_{b,t})}{R_{p,t} - R_{b,t}}, \qquad K = \frac{\ln(1 + R_p) - \ln(1 + R_b)}{R_p - R_b}
\]

with continuous limit \(k_t = \frac{1}{1 + R_{p,t}}\) when \(R_{p,t} = R_{b,t}\). Scaling each period's effect by \(k_t / K\) guarantees exact multi-period additivity:

\[
\sum_{t=1}^T \sum_{j=1}^J \frac{k_t}{K}\left( A_{j,t} + S_{j,t} + I_{j,t} \right) = R_p - R_b
\]

---

## 3. Factor-Implied Active Return Attribution

To test whether an allocator's active return \(y_t = R_{p,t} - R_{b,t}\) arises from genuine idiosyncratic skill (\(\alpha\)) or passive exposure to systematic risk premia \(F_t \in \mathbb{R}^K\) (e.g., Market, Size, Value, Momentum, Low-Volatility), `PerformanceAttributor` estimates the linear factor model:

\[
y_t = \alpha + \sum_{k=1}^K \beta_k F_{k,t} + \varepsilon_t
\]

To account for serial correlation and conditional heteroscedasticity in daily portfolio returns, standard errors and \(t\)-statistics are computed using the **Newey-West (1987) HAC** covariance estimator with Bartlett kernel bandwidth \(L\):

\[
\widehat{\text{Cov}}_{\text{HAC}}(\hat{\theta}) = \frac{T}{T - K - 1}(X^\top X)^{-1} \widehat{S}_{\text{NW}} (X^\top X)^{-1}
\]

where:

\[
\widehat{S}_{\text{NW}} = \sum_{t=1}^T \hat{\varepsilon}_t^2 x_t x_t^\top + \sum_{\ell=1}^L \left(1 - \frac{\ell}{L + 1}\right) \sum_{t=\ell+1}^T \hat{\varepsilon}_t \hat{\varepsilon}_{t-\ell}\left(x_t x_{t-\ell}^\top + x_{t-\ell} x_t^\top\right)
\]

---

## 4. Frictional Drag Accounting (Ma & Smith, 2025)

In multi-period trading under alpha decay and convex market impact (Ma & Smith, 2025; Almgren & Chriss, 2001), the gap between a theoretical signal's paper return and net realized return decomposes into an exact additive bridge at each step \(t\):

1. **Theoretical Paper Return (\(R_{t}^{\text{paper}}\)):** Return of un-drifted target weights \(w_t^*\):
   \[
   R_t^{\text{paper}} = (w_t^*)^\top r_t + \max(0, 1 - \mathbf{1}^\top w_t^*) r_{f,t}
   \]
2. **Alpha Decay & Intra-Period Drift Drag (\(\Delta_t^{\text{drift}}\)):** Performance wedge between actual held weights \(w_t^{\text{held}}\) (subject to intra-period price drift and rebalance threshold gating) and theoretical target weights:
   \[
   \Delta_t^{\text{drift}} = R_t^{\text{gross}} - R_t^{\text{paper}}
   \]
3. **Proportional Spread & Commission Drag (\(C_t^{\text{spread}}\)):** Linear \(\ell_1\) execution cost:
   \[
   C_t^{\text{spread}} = \sum_{i=1}^M c_{\text{spread}, i} |\Delta w_{i,t}|
   \]
4. **Square-Root Market Impact Drag (\(C_t^{\text{impact}}\)):** Non-linear Almgren-Chriss price impact:
   \[
   C_t^{\text{impact}} = \sum_{i=1}^M \gamma_i \sqrt{\frac{\text{AUM} \cdot |\Delta w_{i,t}|}{V_{i,t}}} |\Delta w_{i,t}|
   \]
5. **Short Borrow Drag (\(C_t^{\text{borrow}}\)):** Financing cost on short exposure \(\sum_i \max(0, -w_{i,t}^{\text{held}})\).

Combining all terms yields the exact Gross-to-Net Waterfall identity:

\[
R_t^{\text{net}} = R_t^{\text{paper}} + \Delta_t^{\text{drift}} - C_t^{\text{spread}} - C_t^{\text{impact}} - C_t^{\text{borrow}}
\]
