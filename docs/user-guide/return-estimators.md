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
from pyalloq.estimators.returns.classical.ewma import EWMAReturnEstimator

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
from pyalloq.estimators.returns.classical.james_stein import JamesSteinReturnEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=JamesSteinReturnEstimator(),
)
```

**When to use:** Large universes (10+ assets) where sample mean is noisy and you want a theoretically grounded shrinkage.

---

## Momentum (`CrossSectionalMomentumEstimator`)

Uses cross-sectional price momentum as a return forecast. Assets with stronger recent performance receive higher expected return estimates.

```python
from pyalloq.estimators.returns.classical.momentum import CrossSectionalMomentumEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    returns_estimator=CrossSectionalMomentumEstimator(),
)
```

**When to use:** Equity universes where the momentum factor is significant, or when you want a trend-following return signal.

---

## Factor Returns (`MultiFactorReturnEstimator`)

Estimates expected returns using a factor model. Asset returns are attributed to systematic risk factors (e.g., Fama-French). The factor loadings are estimated from `data.features`.

```python
from pyalloq.estimators.returns.classical.factor import MultiFactorReturnEstimator

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
from pyalloq.estimators.returns.classical.implied import ImpliedReturnEstimator

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
from pyalloq.estimators.black_litterman.bayesian import BlackLittermanEstimator
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
