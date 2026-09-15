# Return Estimators

Return estimators produce a `pd.Series` of **annualized expected returns** — one per asset — given a `MarketData` object. The result is consumed by `StrategyPipeline` and passed to the allocator.

All estimators implement `BaseReturnEstimator`:

```python
class BaseReturnEstimator(ABC):
    @abstractmethod
    def estimate(self, data: MarketData, **kwargs) -> pd.Series:
        ...
```

---

## EWMA (`EWMAReturnEstimator`)

Exponentially Weighted Moving Average returns. Recent observations receive higher weight. The **default** estimator in `StrategyPipeline`.

```python
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=EWMAReturnEstimator(span=60),  # 60-day span
)
```

**When to use:** When you believe recent performance is more predictive than long-run historical averages.

---

## James-Stein (`JamesSteinReturnEstimator`)

Shrinks historical mean returns towards the grand cross-sectional mean. Reduces estimation error in high-dimensional settings. Optimal for universes with \(N \geq 3\) assets.

\[
\hat{\mu}^{JS} = (1 - w) \hat{\mu} + w \bar{\mu} \mathbf{1}
\]

where the shrinkage intensity \(w\) is:

\[
w = \min\left(1, \frac{N-2}{(\hat{\mu} - \bar{\mu}\mathbf{1})^\top \Sigma^{-1} (\hat{\mu} - \bar{\mu}\mathbf{1})}\right)
\]

```python
from pyalloq.classical.estimators.returns.classical.james_stein import JamesSteinReturnEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=JamesSteinReturnEstimator(),
)
```

**When to use:** Large universes (10+ assets) where sample mean is noisy and you want a theoretically grounded shrinkage.

---

## Volatility-Scaled Multi-Horizon Momentum (`VolatilityScaledMultiHorizonEstimator`)

Computes raw momentum over multiple lookback horizons (by default 63, 126, and 252 trading days, skipping the most recent 21 days to avoid 1-month short-term reversal). Each horizon return is normalized by realized volatility, and the resulting scores are standardized cross-sectionally to produce stable relative return forecasts.

```python
from pyalloq.classical.estimators.returns.classical.momentum import VolatilityScaledMultiHorizonEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=VolatilityScaledMultiHorizonEstimator(
        horizons=(63, 126, 252),
        skip_period=21,
    ),
)
```

**When to use:** Equity or multi-asset universes where momentum signals across varying horizons need to be aggregated and risk-adjusted without letting high-volatility assets dominate.

---

## Residual Momentum (`ResidualMomentumEstimator`)

Isolates idiosyncratic price momentum by regressing each asset's historical returns on a cross-sectional market proxy to remove systemic market beta:

\[
R_{i, t} = \alpha_i + \beta_i R_{m, t} + \epsilon_{i, t}
\]

Standardized cumulative residuals $\sum \epsilon_{i, t} / \sigma_{\epsilon, i}$ are then converted to cross-sectional Z-scores.

```python
from pyalloq.classical.estimators.returns.classical.momentum import ResidualMomentumEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=ResidualMomentumEstimator(
        lookback=252,
        skip_period=21,
    ),
)
```

**When to use:** You want trend-following exposure driven purely by asset-specific performance rather than co-movement with broader equity indices.

---

## Factor Returns (`MultiFactorReturnEstimator`)

Estimates expected returns using a factor model. Asset returns are attributed to systematic risk factors (e.g., Fama-French). The factor loadings are estimated from `data.features`.

```python
from pyalloq.classical.estimators.returns.classical.factor import MultiFactorReturnEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=MultiFactorReturnEstimator(),
)
```

**When to use:** You have factor data attached to `MarketData.features` and want a structured return model.

---

## Implied Returns (`ImpliedReturnEstimator`)

Reverse-engineers expected returns from observed market-capitalization weights using the Black-Litterman equilibrium formula:

\[
\Pi = \lambda \Sigma w^{mkt}
\]

where \(\lambda\) is the implied risk aversion from the market, \(\Sigma\) is the covariance matrix, and \(w^{mkt}\) are cap-weighted market weights.

```python
from pyalloq.classical.estimators.returns.classical.implied import ImpliedReturnEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=ImpliedReturnEstimator(),
)
```

**When to use:** You want returns that are consistent with market equilibrium, as a neutral starting point before overlaying views.

---

## Black-Litterman (`BlackLittermanEstimator`)

Blends the market equilibrium returns \(\Pi\) with investor views \(Q\) expressed through a picking matrix \(P\). Returns both a posterior expected return vector and a posterior covariance matrix.

\[
\mu^* = \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1} \left[(\tau\Sigma)^{-1} \Pi + P^\top \Omega^{-1} Q\right]
\]

```python
from pyalloq.classical.estimators.black_litterman.bayesian import BlackLittermanEstimator
import pandas as pd

# Define views: "AAPL will outperform MSFT by 5%"
P = pd.DataFrame([[1, -1, 0]], columns=["AAPL", "MSFT", "GOOGL"])
Q = pd.DataFrame([[0.05]])

bl = BlackLittermanEstimator(tau=0.05, risk_aversion=2.5)
mu_bl, cov_bl = bl.estimate(
    market_caps=market_caps,
    cov_matrix=cov_matrix,
    P=P,
    Q=Q,
)
```

!!! note
    `BlackLittermanEstimator` is not a `BaseReturnEstimator` — it returns both `(mu, cov)` and is typically called directly or wrapped in a custom pipeline before the allocator step.

---

## Deep Learning / Foundation Models: TimesFM 3.0

Google's [TimesFM 3.0](https://github.com/google-research/timesfm) is a pretrained foundation model for zero-shot time-series forecasting. PyAlloq integrates TimesFM 3.0 for both **point return estimation** and **probabilistic view generation** for Black-Litterman.

!!! note "Installation requirement"
    TimesFM integration requires the deep learning optional dependency:
    ```bash
    uv add pyalloq[dl]
    # or pip install "pyalloq[dl]"
    ```

### Zero-Shot Return Estimation

PyAlloq provides both univariate and multivariate return estimators based on TimesFM 3.0:

- **`TimesFM3UnivariateReturnEstimator`**: Forecasts each asset series independently.
- **`TimesFM3MultivariateReturnEstimator`**: Forecasts cross-asset relationships jointly and optionally ingests past and future covariates from `MarketData.features` and `MarketData.future_features`.

For a forecast horizon $H$ (in trading days) and forecasted price $\hat{P}_{T+H, i}$, expected returns are annualized:

\[
\hat{\mu}_i = \left(\frac{\hat{P}_{T+H, i}}{P_{T, i}} - 1\right) \times \frac{252}{H}
\]

```python
from pyalloq.dl.estimators.returns.timesfm import (
    TimesFM3MultivariateReturnEstimator,
    TimesFM3UnivariateReturnEstimator,
)
from timesfm3 import TimesFM3Evaluator

# Initialize the TimesFM 3.0 evaluator
tfm = TimesFM3Evaluator.load_pretrained(...)

# 1. Univariate return estimator (forecasts assets independently)
univariate_est = TimesFM3UnivariateReturnEstimator(model=tfm, horizon=21)

# 2. Multivariate return estimator (leverages cross-asset dynamics & features)
multivariate_est = TimesFM3MultivariateReturnEstimator(
    model=tfm, horizon=21, annualize=True
)

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=multivariate_est,
)
```

**When to use:** Zero-shot price forecasting across regimes without needing to train or fine-tune neural networks locally. Multivariate mode is ideal when macro/factor features or cross-asset dependencies exist.

---

### TimesFM 3.0 Black-Litterman View Generation

Black-Litterman typically requires subjective investor views ($Q$) and a diagonal confidence/uncertainty matrix ($\Omega$). TimesFM 3.0 natively outputs probabilistic forecast quantiles ($p_{10}, p_{50}, p_{90}$), which PyAlloq turns into objective, probabilistic views:

1. **View Vector ($Q$)**: The median forecast $p_{50}$ defines the expected return view for each asset:
   \[
   Q_i = \left(\frac{p_{50, i}}{P_{T, i}} - 1\right) \times \frac{252}{H}
   \]
2. **Uncertainty Matrix ($\Omega$)**: Under a Gaussian assumption, the 80% credible interval ($p_{10}$ to $p_{90}$) corresponds to $2 \times 1.28 \approx 2.56$ standard deviations. The annualized forecast standard deviation is:
   \[
   \sigma_{i} = \frac{p_{90, i} - p_{10, i}}{2.56 \cdot P_{T, i}} \times \sqrt{\frac{252}{H}}
   \]
   The uncertainty matrix $\Omega$ is the diagonal matrix of these view variances:
   \[
   \Omega = \operatorname{diag}\left(\sigma_1^2, \sigma_2^2, \ldots, \sigma_N^2\right)
   \]
3. **Picking Matrix ($P$)**: Identity matrix $I_N$, since views are generated for all assets.

```python
from pyalloq.dl.estimators.black_litterman.timesfm import (
    TimesFM3MultivariateViewGenerator,
    TimesFM3UnivariateViewGenerator,
)
from pyalloq.classical.estimators.black_litterman.bayesian import BlackLittermanEstimator
from timesfm3 import TimesFM3Evaluator

tfm = TimesFM3Evaluator.load_pretrained(...)

# Generate objective views and uncertainty directly from TimesFM quantiles
view_generator = TimesFM3MultivariateViewGenerator(model=tfm, horizon=21)
P, Q, Omega = view_generator.generate(market_data)

# Blend with market equilibrium in Black-Litterman
bl = BlackLittermanEstimator(tau=0.05, risk_aversion=2.5)
mu_bl, cov_bl = bl.estimate(
    market_caps=market_data.cross_sectional,
    cov_matrix=cov_matrix,
    P=P,
    Q=Q,
)
```

**When to use:** You want Black-Litterman portfolio optimization but want systematic, probabilistic foundation model forecasts to drive the views and quantify view uncertainty.

---

## Implementing a Custom Return Estimator

```python
from pyalloq_core.interfaces import BaseReturnEstimator
from pyalloq_core.data import MarketData
import pandas as pd

class MyReturnEstimator(BaseReturnEstimator):
    def estimate(self, data: MarketData, **kwargs) -> pd.Series:
        # Annualized mean of log returns
        log_returns = data.prices.apply(lambda col: col.pct_change().add(1).apply(lambda x: x if x > 0 else 1e-6))
        return log_returns.mean() * 252

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=MyReturnEstimator(),
)
```
