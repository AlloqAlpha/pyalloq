# Allocators

Allocators solve for optimal portfolio weights given market data, a covariance matrix, and optionally expected returns. All allocators implement `BaseAllocator` and can be dropped into any `StrategyPipeline`.

---

## Classical Allocators

### Equal Weight, Inverse Volatility & Market Cap Baselines

PyAlloq provides four fast analytical baselines in `pyalloq.classical.optimizers.naive`:

- **`EqualWeightAllocator`**: Assigns identical weight \(1/N\) to every asset.
- **`InverseVolatilityAllocator`**: Assigns weights inversely proportional to standalone asset volatility \(w_i \propto 1/\sigma_i\) (ignoring cross-correlations).
- **`MarketCapWeightAllocator`**: Assigns weights proportional to `data.cross_sectional["market_cap"]` (or falls back to \(1/N\)).
- **`RandomAllocator`**: Samples simplex weights from a symmetric Dirichlet distribution.

\[
w_i^{\text{EW}} = \frac{1}{N}, \qquad w_i^{\text{IV}} = \frac{1/\sigma_i}{\sum_{j=1}^N 1/\sigma_j}, \qquad w_i^{\text{MCap}} = \frac{\text{MCap}_i}{\sum_{j=1}^N \text{MCap}_j}
\]

```python
from pyalloq import (
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    MarketCapWeightAllocator,
    StrategyPipeline,
)

pipeline_ew = StrategyPipeline(allocator=EqualWeightAllocator(tickers=data.assets))
pipeline_iv = StrategyPipeline(allocator=InverseVolatilityAllocator(tickers=data.assets))
pipeline_mc = StrategyPipeline(allocator=MarketCapWeightAllocator(tickers=data.assets))
```

!!! tip "Use as baselines"
    Always include `EqualWeightAllocator` and `InverseVolatilityAllocator` as benchmarks in your backtest comparisons.

---

### Markowitz (`MarkowitzAllocator`)

Classical mean-variance optimizer supporting five objectives (`MIN_VOLATILITY`, `MAX_SHARPE`, `MAX_RETURN`, `EFFICIENT_RETURN`, `EFFICIENT_RISK`), institutional constraints, and efficient frontier tracing.

```python
from pyalloq import MarkowitzAllocator, ObjectiveFunction
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
    Maximizes the Sharpe ratio using the homogenized Cornuéjols-Tütüncü / Tobin transformation. Requires expected returns.

    \[
    \max_w \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}} \;\equiv\; \min_{y, \kappa \ge 0} \; y^\top \Sigma y \quad \text{s.t.} \quad (\mu - r_f)^\top y = 1,\; \mathbf{1}^\top y = \kappa
    \]

    ```python
    allocator = MarkowitzAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.MAX_SHARPE
    )
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

=== "Efficient Return & Efficient Risk"
    Target a specific expected return (`EFFICIENT_RETURN`) or volatility ceiling (`EFFICIENT_RISK`), or trace the entire **Efficient Frontier**:

    ```python
    # Minimize volatility subject to portfolio return >= 12%
    alloc_ret = MarkowitzAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.EFFICIENT_RETURN,
        target_return=0.12,
    )

    # Maximize expected return subject to portfolio volatility <= 15%
    alloc_risk = MarkowitzAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.EFFICIENT_RISK,
        target_volatility=0.15,
    )

    # Trace 20 portfolios along the Efficient Frontier
    frontier_results = alloc_ret.efficient_frontier(
        data, cov_matrix=cov, expected_returns=mu, n_points=20
    )
    ```

=== "Constraints (Bounds, Market-Neutral, Sectors, Turnover)"
    `MarkowitzAllocator` natively supports box bounds, dollar-neutrality, gross leverage limits, single-period turnover limits, L2 diversification regularization, and sector caps/floors:

    ```python
    constrained_allocator = MarkowitzAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.MAX_SHARPE,
        weight_bounds=(0.02, 0.35),                     # Per-asset min/max bounds
        market_neutral=False,                           # Set True for sum(w) == 0
        max_leverage=1.0,                               # L1 gross leverage ||w||_1 <= L
        max_turnover=0.40,                              # ||w - w_prev||_1 <= 0.40
        l2_gamma=0.01,                                  # L2 weight regularization
        sector_mapper={"AAPL": "Tech", "MSFT": "Tech", "JPM": "Finance"},
        sector_upper={"Tech": 0.50},
        sector_lower={"Finance": 0.15},
    )
    ```

---

### Conditional Value-at-Risk & Drawdown (`MeanCVaRAllocator` & `CDaRAllocator`)

When asset returns exhibit fat tails or negative skewness, variance-based optimization underestimates tail risk. PyAlloq provides two single-period linear/conic downside risk allocators in `pyalloq.classical.optimizers.cvar`:

- **`MeanCVaRAllocator`**: Minimizes portfolio Expected Shortfall (\(\text{CVaR}_\beta\)) via the Rockafellar & Uryasev (2000) linear program (supports `MIN_CVAR`, `EFFICIENT_RETURN`, and `MAX_RETURN` objectives plus box bounds, market neutrality, turnover, and L2 regularization).
- **`CDaRAllocator`**: Minimizes Conditional Drawdown-at-Risk (\(\text{CDaR}_\beta\)) over the historical cumulative return path (Chekhlov, Uryasev & Zabarankin, 2005).

```python
from pyalloq import CDaRAllocator, MeanCVaRAllocator, ObjectiveFunction

# 95% CVaR minimization with 40% per-asset cap
cvar_allocator = MeanCVaRAllocator(
    tickers=data.assets,
    beta=0.95,
    objective=ObjectiveFunction.MIN_CVAR,
    weight_bounds=(0.0, 0.40),
)

# 90% Conditional Drawdown-at-Risk minimization
cdar_allocator = CDaRAllocator(
    tickers=data.assets,
    beta=0.90,
    weight_bounds=(0.0, 0.50),
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
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator

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
from pyalloq.classical.optimizers.risk_budgeting import RiskBudgetingAllocator
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
from pyalloq.classical.optimizers.max_diversification import MaxDiversificationAllocator

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
from pyalloq.ml.optimizers.hrp import HRPAllocator

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
from pyalloq.ml.optimizers.herc import HERCAllocator

pipeline = StrategyPipeline(
    allocator=HERCAllocator(tickers=data.assets)
)
```

---

### NCO (`NCOAllocator`)

Nested Cluster Optimization. Divides the universe into clusters via k-means on the correlation matrix, runs a Markowitz optimization within each cluster (intra-cluster), and then a second Markowitz optimization across cluster representatives (inter-cluster).

```python
from pyalloq.ml.optimizers.nco import NCOAllocator

pipeline = StrategyPipeline(
    allocator=NCOAllocator(tickers=data.assets)
)
```

---

## Choosing an Allocator

| Scenario | Recommended Allocator |
|----------|----------------------|
| Quick analytical baselines | `EqualWeightAllocator`, `InverseVolatilityAllocator`, `MarketCapWeightAllocator` |
| Minimize portfolio risk, no return forecast needed | `RiskParityAllocator` |
| Minimize portfolio risk, trust your return forecast | `MarkowitzAllocator(MIN_VOLATILITY)` |
| Maximize Sharpe with box/sector/turnover constraints | `MarkowitzAllocator(MAX_SHARPE, weight_bounds=..., sector_upper=...)` |
| Target return or target volatility / Efficient Frontier | `MarkowitzAllocator(EFFICIENT_RETURN)` / `MarkowitzAllocator(EFFICIENT_RISK)` |
| Heavy-tailed downside or drawdown control | `MeanCVaRAllocator` or `CDaRAllocator` |
| Large universe (50+ assets), covariance unstable | `HRPAllocator`, `NCOAllocator`, or `BayesianHRPAllocator` |
| Maximize portfolio diversification | `MaxDiversificationAllocator` |
| Assign specific risk budgets per asset | `RiskBudgetingAllocator` |

---

## Converting Continuous Weights to Integer Shares (`DiscreteShareAllocator`)

In live trading, continuous portfolio weights \(w \in \mathbb{R}^N\) must be converted into whole share or lot orders given current asset prices and total account cash. `DiscreteShareAllocator` supports both `method="greedy"` and Mixed-Integer Linear Programming (`method="lp"`), handling both long-only and long-short portfolios:

```python
from pyalloq import DiscreteShareAllocator

weights = pipeline.generate_weights(data)

disc = DiscreteShareAllocator(total_portfolio_value=100_000.0, lot_size=1)
order_plan = disc.allocate(weights, latest_prices=data, method="greedy")

print(order_plan.shares)          # {'AAPL': 142, 'MSFT': 88, 'GOOGL': 115, ...}
print(order_plan.leftover_cash)   # 43.18
print(order_plan.rmse_error)      # 0.0004
```

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
