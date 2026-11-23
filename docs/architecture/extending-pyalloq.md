# Extending PyAlloq

Every major layer in PyAlloq exposes a small abstract base class so you can write custom proprietary models and immediately inherit `WalkForwardEngine`, `ParallelMonteCarloEngine`, `CPCVSplitter`, `EnsembleAllocator`, `ConformalSafetyGateAllocator`, and all `pyalloq-viz` Tear Sheets.

---

## 1. Custom Data Client (`BaseDataClient`)

Subclass `pyalloq_data_connector.base.BaseDataClient` and implement `fetch_raw_data()`. The base class automatically aligns closing prices and optional OHLCV features into a validated `MarketData` instance.

```python
import datetime as dt
import pandas as pd
from pyalloq_data_connector.base import BaseDataClient

class ParquetDataClient(BaseDataClient):
    def __init__(self, root_dir: str):
        super().__init__(api_key=None)
        self.root_dir = root_dir

    def fetch_raw_data(
        self,
        tickers: list[str],
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = pd.read_parquet(f"{self.root_dir}/{ticker}.parquet")
            out[ticker] = df.loc[start_date:end_date]  # expects 'close' (and optional 'volume')
        return out
```

---

## 2. Custom Feature Transformer (`BaseFeatureTransformer`)

Subclass `pyalloq_features.core.base.BaseFeatureTransformer` to plug into `FeaturePipeline`. Always return a new or shallow-copied `MarketData` with aligned feature indices (`data.validate_alignment()`).

```python
from dataclasses import replace
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_features.core.base import BaseFeatureTransformer

class OvernightGapTransformer(BaseFeatureTransformer):
    def __init__(self, out_key: str = "overnight_gap"):
        self.out_key = out_key

    def transform(self, data: MarketData) -> MarketData:
        if "open" not in data.features:
            raise KeyError("OvernightGapTransformer requires 'open' in data.features")
        gap = (data.features["open"] / data.prices.shift(1) - 1.0).fillna(0.0)
        new_features = dict(data.features)
        new_features[self.out_key] = gap.reindex(data.prices.index)
        return replace(data, features=new_features)
```

---

## 3. Custom Return & Covariance Estimators

### `BaseReturnEstimator`

```python
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseReturnEstimator

class WinsorizedMeanEstimator(BaseReturnEstimator):
    def __init__(self, clip_z: float = 3.0, annualization_factor: float = 252.0):
        self.clip_z = clip_z
        self.annualization_factor = annualization_factor

    def estimate(self, data: MarketData, **kwargs) -> pd.Series:
        rets = data.get_returns().dropna()
        mu = rets.mean()
        sigma = rets.std().replace(0.0, 1e-8)
        clipped = rets.clip(lower=mu - self.clip_z * sigma, upper=mu + self.clip_z * sigma, axis=1)
        return clipped.mean() * self.annualization_factor
```

### `BaseCovarianceEstimator`

```python
import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseCovarianceEstimator

class DiagonalShrinkageCovariance(BaseCovarianceEstimator):
    def __init__(self, shrink: float = 0.25, annualization_factor: float = 252.0):
        self.shrink = shrink
        self.annualization_factor = annualization_factor

    def estimate(self, data: MarketData, **kwargs) -> pd.DataFrame:
        cov = data.get_returns().dropna().cov() * self.annualization_factor
        diag = pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
        return (1.0 - self.shrink) * cov + self.shrink * diag
```

---

## 4. Custom Black-Litterman View Generator (`BaseViewGenerator`)

Return `(P, Q, Omega)` where `P` is \(K \times N\), `Q` is length-\(K\) `pd.Series`, and `Omega` is \(K \times K\) PSD view-uncertainty `pd.DataFrame`. It plugs directly into `BlackLittermanReturnEstimator(view_generator=...)` or `EnsembleViewGenerator`.

```python
import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseViewGenerator

class MomentumSpreadViewGenerator(BaseViewGenerator):
    def __init__(self, lookback: int = 63, view_vol: float = 0.15):
        self.lookback = lookback
        self.view_vol = view_vol

    def generate(self, data: MarketData, **kwargs):
        mom = data.prices.pct_change(self.lookback).iloc[-1]
        winner, loser = mom.idxmax(), mom.idxmin()
        P = pd.DataFrame(0.0, index=["mom_spread"], columns=data.assets)
        P.loc["mom_spread", winner] = 1.0
        P.loc["mom_spread", loser] = -1.0
        Q = pd.Series({"mom_spread": float(np.clip(mom[winner] - mom[loser], -0.25, 0.25))})
        Omega = pd.DataFrame([[self.view_vol ** 2]], index=["mom_spread"], columns=["mom_spread"])
        return P, Q, Omega
```

---

## 5. Custom Allocator (`BaseAllocator`)

Implement `allocate(data, cov_matrix, expected_returns=None, **kwargs) -> OptimizationResult`. Accept `current_weights` via `kwargs` if your allocator penalizes turnover.

```python
import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult

class InverseVarianceAllocator(BaseAllocator):
    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        **kwargs,
    ) -> OptimizationResult:
        inv_var = 1.0 / np.maximum(np.diag(cov_matrix.values), 1e-10)
        weights = pd.Series(inv_var / inv_var.sum(), index=cov_matrix.index)
        vol = float(np.sqrt(weights.values @ cov_matrix.values @ weights.values))
        exp_ret = float(weights @ expected_returns) if expected_returns is not None else None
        return OptimizationResult(
            weights=weights,
            status="OPTIMAL",
            name="InverseVariance",
            expected_return=exp_ret,
            volatility=vol,
        )
```

---

## 6. Custom Meta-Weighting Strategy (`BaseMetaWeightingStrategy`)

Plug custom online learning or bandit rules into `EnsembleAllocator(meta_strategy=...)`:

```python
import numpy as np
from pyalloq_core.meta import BaseMetaWeightingStrategy

class TrailingSharpeSoftmaxStrategy(BaseMetaWeightingStrategy):
    def __init__(self, temperature: float = 5.0):
        self.temperature = temperature

    def compute_meta_weights(
        self,
        expert_returns: np.ndarray,
        current_meta_weights: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        # expert_returns has shape (T_eval, K)
        mu = expert_returns.mean(axis=0)
        sigma = np.maximum(expert_returns.std(axis=0), 1e-6)
        logits = self.temperature * (mu / sigma)
        logits -= logits.max()
        w = np.exp(logits)
        return w / w.sum()
```

---

## 7. Custom Transaction Cost Model (`BaseCostModel`)

Subclass `pyalloq_backtest.costs.BaseCostModel` to support both NumPy/Pandas backtesting (`WalkForwardEngine`) and PyTorch differentiable training (`TorchCostAdapter`).

```python
from pyalloq_backtest.costs import BaseCostModel

class TieredSlippageCostModel(BaseCostModel):
    def __init__(self, base_bps: float = 5.0, large_trade_bps: float = 15.0, threshold: float = 0.10):
        self.base_bps = base_bps
        self.large_trade_bps = large_trade_bps
        self.threshold = threshold

    def calculate_costs(self, weights_delta, features_slice=None):
        abs_delta = abs(weights_delta)
        turnover = abs_delta.sum()
        rate = self.large_trade_bps if float(turnover) > self.threshold else self.base_bps
        return turnover * (rate / 10_000.0)
```

---

## 8. Custom Scenario Generator (`BaseScenarioGenerator`)

Subclass `pyalloq_core.interfaces.BaseScenarioGenerator` to plug into `EntropyPooler`, `ParallelMonteCarloEngine`, `StochasticScenarioMPCAllocator`, and `POMDPPortfolioEnv`:

```python
import numpy as np
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import BaseScenarioGenerator

class BlockBootstrapGenerator(BaseScenarioGenerator):
    def __init__(self, block_size: int = 10, random_state: int = 42):
        self.block_size = block_size
        self.rng = np.random.default_rng(random_state)

    def fit(self, data: MarketData) -> None:
        self._returns = data.get_returns().dropna().values
        self._last_prices = data.prices.iloc[-1].values
        self._assets = list(data.assets)
        self._record_index_metadata(data.prices.index)

    def generate(self, n_paths: int, horizon: int) -> ScenarioMarketData:
        T_hist, N = self._returns.shape
        n_blocks = int(np.ceil(horizon / self.block_size))
        sim_rets = np.empty((n_paths, horizon, N), dtype=float)
        for m in range(n_paths):
            starts = self.rng.integers(0, T_hist - self.block_size, size=n_blocks)
            blocks = np.concatenate([self._returns[s : s + self.block_size] for s in starts], axis=0)
            sim_rets[m] = blocks[:horizon]
        sim_prices = self._last_prices[None, None, :] * np.cumprod(1.0 + sim_rets, axis=1)
        return ScenarioMarketData(
            prices=sim_prices,
            returns=sim_rets,
            asset_names=self._assets,
            initial_prices=self._last_prices,
            timestamps=self._build_future_timestamps(horizon),
        )
```

---

## 9. Custom Visualization Tear Sheet (`BaseTearSheet`)

Subclass `pyalloq_viz.tearsheets.base.BaseTearSheet` to inherit `show()`, `save_image()`, and themed responsive HTML export (`to_html()`):

```python
import plotly.graph_objects as go
from pyalloq_viz.tearsheets.base import BaseTearSheet
from pyalloq_viz.themes import apply_plotly_theme

class CustomRiskTearSheet(BaseTearSheet):
    title = "Custom Risk Monitor"
    subtitle = "Tail-risk and exposure overview"

    def __init__(self, returns_series):
        self.returns = returns_series

    def get_kpis(self) -> dict[str, str]:
        return {
            "Cumulative Return": f"{(1 + self.returns).prod() - 1:.2%}",
            "Daily Volatility": f"{self.returns.std():.2%}",
        }

    def get_component_figures(self, theme: str = "alloq_dark") -> list[go.Figure]:
        fig = go.Figure(go.Scatter(x=self.returns.index, y=(1 + self.returns).cumprod(), name="Equity"))
        return [apply_plotly_theme(fig, theme=theme)]

    def to_figure(self, theme: str = "alloq_dark") -> go.Figure:
        return self.get_component_figures(theme=theme)[0]
```
