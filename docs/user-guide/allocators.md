# Allocators

Allocators solve for optimal portfolio weights given market data, a covariance matrix, and optionally expected returns. All allocators implement `BaseAllocator` and can be dropped into any `StrategyPipeline`.

---

## Classical Allocators

### Equal Weight (`EqualWeightAllocator`)

Assigns identical weight \(1/N\) to every asset. The simplest possible portfolio. Surprisingly hard to beat out-of-sample.

\[
w_i = \frac{1}{N}, \quad \forall i
\]

```python
from pyalloq.optimizers.classical.naive import EqualWeightAllocator

pipeline = StrategyPipeline(
    allocator=EqualWeightAllocator(tickers=data.assets)
)
```

!!! tip "Use as a baseline"
    Always include `EqualWeightAllocator` as a benchmark in your backtest comparisons.

---

### Markowitz (`MarkowitzAllocator`)

Classical mean-variance optimizer. Supports three objective functions via the `ObjectiveFunction` enum.

```python
from pyalloq.optimizers.classical.markowitz import MarkowitzAllocator
from pyalloq_core.enums import ObjectiveFunction
```

=== "Min Volatility"
    Minimizes total portfolio variance with no expected return input needed.

    \[
    \min_w \; w^\top \Sigma w \quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; w \geq 0
    \]

    ```python
    allocator = MarkowitzAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.MIN_VOLATILITY
    )
    ```

=== "Max Sharpe"
    Maximizes the Sharpe ratio using the Tobin substitution trick. Requires expected returns.

    \[
    \max_w \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}} \;\equiv\; \min_y \; y^\top \Sigma y \quad \text{s.t.} \quad (\mu - r_f)^\top y = 1,\; y \geq 0
    \]

    ```python
    allocator = MarkowitzAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.MAX_SHARPE
    )
    # MAX_SHARPE requires a returns estimator in the pipeline
    pipeline = StrategyPipeline(
        allocator=allocator,
        returns_estimator=JamesSteinReturnEstimator(),
    )
    ```

=== "Max Return (Utility)"
    Maximizes mean-variance utility, controlled by `risk_aversion` in `MarketData`.

    \[
    \max_w \; \mu^\top w - \lambda \cdot w^\top \Sigma w \quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; w \geq 0
    \]

    ```python
    allocator = MarkowitzAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.MAX_RETURN
    )
    ```

---

### Risk Parity (`RiskParityAllocator`)

Equal Risk Contribution (ERC). Every asset contributes the same amount of marginal risk to the portfolio. No expected return input needed — purely covariance-driven.

\[
\min_y \; \frac{1}{2} y^\top \Sigma y - \sum_{i=1}^{N} \ln y_i \quad \text{s.t.} \quad y \geq 0
\]

Then normalize: \(w = y / \mathbf{1}^\top y\)

```python
from pyalloq.optimizers.classical.risk_parity import RiskParityAllocator

pipeline = StrategyPipeline(
    allocator=RiskParityAllocator(tickers=data.assets)
)
```

---

### Risk Budgeting (`RiskBudgetingAllocator`)

Generalization of Risk Parity where each asset is assigned a **target risk budget** \(b_i\). If no budgets are provided, defaults to equal budgeting (identical to Risk Parity).

\[
\min_y \; \frac{1}{2} y^\top \Sigma y - \sum_{i=1}^{N} b_i \ln y_i \quad \text{s.t.} \quad y \geq 0
\]

```python
from pyalloq.optimizers.classical.risk_budgeting import RiskBudgetingAllocator
import pandas as pd

# Provide risk budgets through MarketData.features
risk_budgets = pd.DataFrame(
    {"AAPL": [0.4], "MSFT": [0.3], "GOOGL": [0.3]},
    index=data.prices.index[:1]
).reindex(data.prices.index).ffill()

data_with_budgets = MarketData(
    prices=data.prices,
    features={"risk_budgets": risk_budgets},
)

pipeline = StrategyPipeline(
    allocator=RiskBudgetingAllocator(tickers=data.assets)
)
```

---

### Max Diversification (`MaxDiversificationAllocator`)

Maximizes the **Diversification Ratio** — the weighted sum of individual asset volatilities divided by portfolio volatility.

\[
\max_w \; D(w) = \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}}
\]

Implemented via the equivalent dual: minimize portfolio variance subject to weighted volatilities summing to 1.

```python
from pyalloq.optimizers.classical.max_diversification import MaxDiversificationAllocator

pipeline = StrategyPipeline(
    allocator=MaxDiversificationAllocator(tickers=data.assets)
)
```

---

## Graph-Based ML Allocators

These allocators use hierarchical clustering on the correlation matrix to build a dendrogram-based weight allocation. They **do not invert the covariance matrix**, making them more numerically stable for large universes.

### HRP (`HRPAllocator`)

Hierarchical Risk Parity (López de Prado, 2016). Uses single-linkage clustering on correlation distance, quasi-diagonalization of the covariance matrix, and recursive bisection to assign inverse-variance weights.

```python
from pyalloq.optimizers.machine_learning.hrp import HRPAllocator

pipeline = StrategyPipeline(
    allocator=HRPAllocator(tickers=data.assets)
)
```

Steps:
1. Compute correlation distance matrix: \(d_{ij} = \sqrt{\frac{1 - \rho_{ij}}{2}}\)
2. Build hierarchical cluster tree (dendrogram)
3. Sort assets by quasi-diagonalization
4. Assign weights via recursive bisection using cluster inverse-variance portfolios

---

### HERC (`HERCAllocator`)

Hierarchical Equal Risk Contribution. Similar to HRP but allocates risk budget equally across the hierarchical clusters at each level, then inverse-variance within each cluster.

```python
from pyalloq.optimizers.machine_learning.herc import HERCAllocator

pipeline = StrategyPipeline(
    allocator=HERCAllocator(tickers=data.assets)
)
```

---

### NCO (`NCOAllocator`)

Nested Cluster Optimization. Divides the universe into clusters via k-means on the correlation matrix, runs a Markowitz optimization within each cluster (intra-cluster), and then a second Markowitz optimization across cluster representatives (inter-cluster).

```python
from pyalloq.optimizers.machine_learning.nco import NCOAllocator

pipeline = StrategyPipeline(
    allocator=NCOAllocator(tickers=data.assets)
)
```

---

## Choosing an Allocator

| Scenario | Recommended Allocator |
|----------|----------------------|
| Quick baseline | `EqualWeightAllocator` |
| Minimize portfolio risk, no return forecast needed | `RiskParityAllocator` |
| Minimize portfolio risk, trust your return forecast | `MarkowitzAllocator(MIN_VOLATILITY)` |
| Maximize Sharpe, have a good return model | `MarkowitzAllocator(MAX_SHARPE)` |
| Large universe (50+ assets), covariance unstable | `HRPAllocator` or `NCOAllocator` |
| Maximize portfolio diversification | `MaxDiversificationAllocator` |
| Assign specific risk budgets per asset | `RiskBudgetingAllocator` |

---

## Implementing a Custom Allocator

```python
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult
from pyalloq_core.data import MarketData
import pandas as pd

class MyCustomAllocator(BaseAllocator):
    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs,
    ) -> OptimizationResult:
        # Your optimization logic here
        weights = pd.Series(1.0 / len(data.assets), index=data.assets)
        return OptimizationResult(weights=weights, status="OPTIMAL")

# Drop it into any pipeline — no other changes needed
pipeline = StrategyPipeline(allocator=MyCustomAllocator(tickers=data.assets))
```
