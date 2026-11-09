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
from pyalloq_features.transformers.cross_sectional import CrossSectionalRank

cs_transformer = CrossSectionalRank(target_feature="volume")
data = cs_transformer.transform(data)
```

Cross-sectional features are particularly useful with `FactorReturnEstimator` and custom ML-based allocators.

---

## Point-in-Time Option-Implied Volatility (`PITOptionsMFIVTransformer`)

Compute 30-day **Model-Free Implied Volatility (MFIV)** (Britten-Jones & Neuberger 2000 / CBOE VIX methodology) from raw historical option chain snapshots while enforcing strict **Point-in-Time (PIT)** timestamp guards (`quote_timestamp <= decision_timestamp`) to prevent look-ahead bias.

When option chains are absent for a given asset or date, `PITOptionsMFIVTransformer` automatically falls back to Parkinson High-Low range volatility (if `"high"` and `"low"` exist in `data.features`) or EWMA realized volatility so `data.features["implied_vol"]` is always complete and aligned with `data.prices.index`.

```python
from pyalloq_features.transformers import PITOptionsMFIVTransformer

pit_iv = PITOptionsMFIVTransformer(
    option_chain_df=raw_option_chains,  # optional; uses causal fallback if None
    target_horizon_days=30.0,
    market_close_offset_hours=16.0,
)
data = pit_iv.transform(data)
# Populates data.features["implied_vol"]
```

### Rough Path Log-Signatures & Lifted Heston Rough Volatility

To capture local non-Markovian path geometry, price-volatility lead-lag (Lévy areas), and multi-scale rough volatility without microstructure noise contamination (Cont & Das 2023), use `LogSignatureTransformer` and `LiftedHestonRoughVolTransformer`:

```python
from pyalloq_features.transformers import (
    LiftedHestonRoughVolTransformer,
    LogSignatureTransformer,
)

# 1. Lyndon-basis truncated Log-Signatures (Level 1 increments + Level 2 Lévy areas)
data = LogSignatureTransformer(window=10, depth=2).transform(data)
# Populates: "logsig_levy_price_vol", "logsig_time_price_area", "logsig_norm"

# 2. Abi Jaber (2019) K-factor Markovian Lifted Heston Rough Volatility & Adaptive Hurst H_eff
data = LiftedHestonRoughVolTransformer(n_factors=5, estimation_window=30).transform(data)
# Populates: "lifted_heston_vol", "hurst_exponent"
```

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
| `"implied_vol"` | `ImpliedCovarianceEstimator`, `ReactionDiffusionCovarianceEstimator`, `ReactionDiffusionGenerator` |
| `"risk_budgets"` | `RiskBudgetingAllocator` |
| `"volume"` | `AlmgrenChrissCostModel` |
| Any factor data | `FactorReturnEstimator` |
| Any signal | Custom ML estimators / DL models |
