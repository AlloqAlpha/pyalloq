# Feature Engineering

`pyalloq-features` provides tools to compute and attach time-series features — technical indicators, cross-sectional signals, and scaled inputs — to your `MarketData`. Features flow through the entire pipeline, enabling advanced estimators and ML models.

---

## Technical Transformers

Compute price-based technical indicators from `MarketData.prices`.

```python
from pyalloq_features.transformers.technical import TechnicalTransformer

transformer = TechnicalTransformer()
features = transformer.transform(data)

# features is a dict[str, pd.DataFrame] ready to attach to MarketData
```

Available indicators include momentum, rolling volatility, RSI, and moving average crossovers. Each indicator is computed per-asset and aligned to the price index to prevent look-ahead bias.

```python
from pyalloq_core.data import MarketData

enriched = MarketData(
    prices=data.prices,
    features=features,
)
```

---

## Cross-Sectional Transformers

Compute cross-sectional signals — ranking assets relative to each other at each point in time.

```python
from pyalloq_features.transformers.cross_sectional import CrossSectionalTransformer

cs_transformer = CrossSectionalTransformer()
cs_features = cs_transformer.transform(data)
```

Cross-sectional features are particularly useful with `FactorReturnEstimator` and custom ML-based allocators.

---

## Scalers

Normalize features before feeding them to ML estimators or deep learning models.

```python
from pyalloq_features.scalers import StandardFeatureScaler

scaler = StandardFeatureScaler()
scaled_features = scaler.fit_transform(features)
```

---

## Attaching Features to `MarketData`

Features must share the exact same `DatetimeIndex` as `data.prices` — `MarketData.validate_alignment()` enforces this at construction.

```python
from pyalloq_core.data import MarketData

data = MarketData(
    prices=prices_df,
    features={
        "momentum":   momentum_df,    # time-series momentum signals
        "volume":     volume_df,      # daily volume (required by AlmgrenChriss)
        "risk_budgets": budgets_df,   # target risk allocations (for RiskBudgetingAllocator)
    },
)
```

---

## Using Features in Estimators

Several components read from `MarketData.features`:

| Feature key | Used by |
|-------------|---------|
| `"risk_budgets"` | `RiskBudgetingAllocator` |
| `"volume"` | `AlmgrenChrissCostModel` |
| Any factor data | `FactorReturnEstimator` |
| Any signal | Custom ML estimators / DL models |
