# pyalloq-features

`pyalloq-features` provides feature engineering transformers, TA-Lib indicator integration, scaling utilities, and feature pipeline orchestration for **PyAlloq**.

## Key Modules

- **`BaseFeatureTransformer`**: Abstract base class for all feature transformation modules operating on `MarketData`.
- **`TechnicalIndicator`**: Applies TA-Lib technical indicators (e.g., `SMA`, `RSI`, `MACD`, `ATR`, `BBANDS`) across asset columns of a 2D DataFrame and appends resulting features to `MarketData.features`.
- **`FeaturePipeline`**: Sequential feature pipeline builder for chaining feature transformers.
- **`scalers`**: Cross-sectional and time-series feature normalization and standardization utilities.

## Quick Example

```python
from pyalloq_features.transformers.technical import TechnicalIndicator
from pyalloq_features.core.pipeline import FeaturePipeline

# Define feature engineering pipeline
pipeline = FeaturePipeline()
pipeline.add(TechnicalIndicator(indicator="RSI", timeperiod=14))
pipeline.add(TechnicalIndicator(indicator="SMA", timeperiod=50))

# Apply transformations directly to MarketData
# updated_data = pipeline.run(market_data)
```
