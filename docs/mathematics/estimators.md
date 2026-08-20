# Estimator Mathematics

This page covers the mathematical foundations of the return and covariance estimators implemented in `pyalloq`.

---

## Return Estimators

### Exponentially Weighted Mean (EWMA)

Weights observations exponentially so that recent returns matter more:

\[
\hat{\mu}^{EWMA}_i = \frac{\sum_{t=1}^{T} \lambda^{T-t} r_{i,t}}{\sum_{t=1}^{T} \lambda^{T-t}}
\]

where the decay factor \(\lambda = 1 - \frac{2}{\text{span}+1}\).

---

### James-Stein Shrinkage

Shrinks each asset's historical mean towards the grand mean of all assets. For \(N \geq 3\) assets, the James-Stein estimator strictly dominates the sample mean in terms of MSE:

\[
\hat{\mu}^{JS} = (1 - w^*) \hat{\mu} + w^* \bar{\mu} \mathbf{1}
\]

The optimal shrinkage intensity \(w^*\) is:

\[
w^* = \min\left(1,\; \frac{N-2}{(\hat{\mu} - \bar{\mu}\mathbf{1})^\top \hat{\Sigma}^{-1} (\hat{\mu} - \bar{\mu}\mathbf{1})}\right)
\]

where \(\bar{\mu} = \frac{1}{N}\sum_i \hat{\mu}_i\) is the grand mean.

---

### Implied Returns (Reverse Optimization)

Given observed market-capitalization weights \(w^{mkt}\) and the market risk aversion \(\lambda\), the implied equilibrium excess returns are:

\[
\Pi = \lambda \Sigma w^{mkt}
\]

This is the equilibrium prior used in Black-Litterman.

---

### Black-Litterman

Combines the market equilibrium prior \(\Pi\) with investor views \((P, Q)\) through Bayesian updating.

**Views:** \(P \in \mathbb{R}^{K \times N}\) is a view-picking matrix, \(Q \in \mathbb{R}^K\) are the expected view returns, and \(\Omega\) is the uncertainty matrix for the views.

**Posterior expected returns:**

\[
\mu^* = \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
       \left[(\tau\Sigma)^{-1} \Pi + P^\top \Omega^{-1} Q\right]
\]

**Posterior covariance:**

\[
\Sigma^* = \Sigma + \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
\]

where \(\tau\) is a scalar controlling the uncertainty in the prior (typically 0.01–0.05).

---

## Covariance Estimators

### Ledoit-Wolf Shrinkage

Shrinks the sample covariance \(\hat{\Sigma}\) towards a structured target \(T\) (e.g., diagonal matrix):

\[
\hat{\Sigma}^{LW} = (1 - \alpha^*) \hat{\Sigma} + \alpha^* \hat{\mu}_{cov} I
\]

The optimal shrinkage intensity \(\alpha^*\) is derived analytically by minimizing the Frobenius norm of the estimation error:

\[
\alpha^* = \frac{\delta^2}{\gamma^2}
\]

where \(\delta^2\) and \(\gamma^2\) are data-dependent scaling constants derived from the Oracle Approximating Shrinkage (OAS) method.

See: [Ledoit & Wolf (2004)](../references/papers.md#ledoit-wolf-2004)

---

### Random Matrix Theory (Marchenko-Pastur)

For a random matrix \(X \in \mathbb{R}^{T \times N}\) with i.i.d. entries, the eigenvalue distribution of \(\hat{\Sigma} = \frac{1}{T} X^\top X\) converges to the Marchenko-Pastur distribution with upper bound:

\[
\lambda_+ = \sigma^2 \left(1 + \sqrt{\frac{N}{T}}\right)^2
\]

Any empirical eigenvalue \(\lambda_i < \lambda_+\) is attributable to noise, not signal. The denoising step replaces these with the average noise eigenvalue:

\[
\bar{\lambda}_{noise} = \frac{\sum_{i: \lambda_i < \lambda_+} \lambda_i}{|\{i: \lambda_i < \lambda_+\}|}
\]

Then the clean covariance is reconstructed: \(\hat{\Sigma}^{RMT} = V \Lambda_{clean} V^\top\)

The optional **detoning** step removes the largest eigenvalue (corresponding to the market-wide factor) from the correlation matrix before reconstruction, isolating sector-level correlations.

---

### Semi-Covariance

Uses only negative return periods to estimate covariance, focusing on downside co-movement:

\[
\hat{\Sigma}^{semi}_{ij} = \frac{1}{T^-} \sum_{t: r_{i,t} < 0} r_{i,t} \cdot r_{j,t}
\]

where \(T^-\) is the count of periods where asset \(i\) had a negative return.

---

### EWMA Covariance

Uses exponential weighting to capture time-varying covariance:

\[
\hat{\Sigma}_t = (1 - \lambda) r_t r_t^\top + \lambda \hat{\Sigma}_{t-1}
\]

where \(\lambda = 1 - \frac{2}{\text{span}+1}\) is the decay factor. This models volatility clustering observed in financial time series (similar to the RiskMetrics model).
