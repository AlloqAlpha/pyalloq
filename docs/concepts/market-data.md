# MarketData

`MarketData` is the **single source of truth** for all financial data in PyAlloq. Every building block — estimators, allocators, the backtest engine — consumes and produces `MarketData`.

Understanding how it works is essential to avoiding look-ahead bias in backtesting.

---

## Data Fields

```python
from pyalloq_core.data import MarketData
import pandas as pd

data = MarketData(
    prices=prices_df,                    # required: (T × N) closing prices
    features={"volume": volume_df},      # optional: time-series features
    cross_sectional=market_caps_df,      # optional: static cross-sectional data
    risk_free_rate=0.04,                 # constant or pd.Series
    risk_aversion=1.0,                   # 0 = aggressive, 1 = conservative
)
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prices` | `pd.DataFrame` | ✅ | Closing prices. Rows = dates, Columns = ticker symbols |
| `assets` | `list[str]` | Auto | Auto-inferred from `prices.columns` if not provided |
| `features` | `dict[str, pd.DataFrame]` | ❌ | Named time-series feature matrices. Must share the same DatetimeIndex as `prices` |
| `cross_sectional` | `pd.DataFrame \| None` | ❌ | Cross-sectional data (e.g., market caps, sector labels). Not time-indexed |
| `risk_free_rate` | `pd.Series \| float` | ❌ | Annual risk-free rate. Defaults to `0.0` |
| `risk_aversion` | `pd.Series \| float` | ❌ | Used by `MarkowitzAllocator` for MAX_RETURN utility. Defaults to `1.0` |

---

## The Features Dictionary

`features` is a flexible dictionary of named `pd.DataFrame` matrices. Use it to attach any time-series signal to your market data:

```python
data = MarketData(
    prices=prices_df,
    features={
        "volume":        volume_df,           # daily traded volume
        "momentum_12m":  momentum_df,         # 12-month momentum signals
        "risk_budgets":  risk_budget_df,      # per-asset risk budgets (for RiskBudgetingAllocator)
        "vix":           vix_df,              # macro features
    }
)
```

!!! tip "Risk Budgets"
    `RiskBudgetingAllocator` reads its target budgets from `data.features["risk_budgets"]`. This is the standard way to pass allocator-specific inputs through the pipeline.

!!! tip "Almgren-Chriss Cost Model"
    `AlmgrenChrissCostModel` requires `data.features["volume"]` to be set for calculating market impact costs.

---

## Zero-Lookahead Guarantee

### `validate_alignment()`

Called automatically at construction time. Raises a `ValueError` if any feature matrix's index does not exactly match `prices.index`. This prevents silent data misalignment bugs.

```python
# This raises ValueError immediately — catching the bug at construction, not later
data = MarketData(
    prices=daily_prices,
    features={"macro": weekly_macro_df},  # ❌ weekly index ≠ daily index
)
# ValueError: Data misalignment: Feature: macro index does not perfectly match 'prices' index.
```

### `slice_time(end_date, lookback)`

Creates a **new** `MarketData` instance safely sliced up to (and including) `end_date`. The backtest engine calls this for every rebalance window.

```python
# Inside WalkForwardEngine — what happens at each rebalance date:
window_data = data.slice_time(
    end_date=pd.Timestamp("2023-06-30"),
    lookback=252,  # use 1 year of history
)

# window_data.prices ends at 2023-06-30 — no future data leaks through
weights = pipeline.generate_weights(window_data)
```

`slice_time` slices **all** fields consistently: `prices`, all `features`, and `risk_free_rate` (if a Series). The `cross_sectional` field is passed through unchanged since it is not time-indexed.

---

## Working with `MarketData` Directly

You can generate returns from prices:

```python
returns = data.prices.pct_change().dropna()
```

Or select a sub-universe:

```python
us_data = MarketData(
    prices=data.prices[["AAPL", "MSFT", "GOOGL"]],
    risk_free_rate=data.risk_free_rate,
)
```

---

## Design Philosophy

`MarketData` is intentionally a **dumb container**, not a class with business logic. This design choice means:

- All components receive the same type of input
- There is one canonical data format — no per-component data munging
- Time alignment is validated once at construction, not repeatedly inside each component
- The backtest engine can slice the data safely without knowing what's inside it
