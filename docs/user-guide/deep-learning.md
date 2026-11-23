# Deep Learning & Foundation Models

`pyalloq.dl` provides a complete PyTorch deep learning and zero-shot foundation model stack that integrates seamlessly with `MarketData`, `StrategyPipeline`, `WalkForwardEngine`, and `BaseCostModel`.

Install the deep learning extra to enable PyTorch and TimesFM support:

```bash
uv pip install "pyalloq[dl]"
```

---

## 1. Point-in-Time Datasets & Causal Walk-Forward Training

### `MarketDataset` & `UniverseSubsampler`

`MarketDataset` converts a `MarketData` object into sliding windows `(X, y)` (and optional Point-in-Time tradability `mask`) with shape `X: (L, N, F)` (channel 0 is always simple returns; remaining channels come from `MarketData.features`) and `y: (H, N)`.

- **`allow_ragged=True`**: Prevents backward-filling pre-IPO prices and computes a strictly causal boolean mask \(M_t \in \{0, 1\}^N\) inside `[idx, idx + lookback_window)` with zero look-ahead into `y`.
- **`UniverseSubsampler`**: Randomly subsamples \(N_{\text{sub}} \sim \text{LogUniform}(N_{\min}, N_{\text{active}})\) active assets per minibatch so size-equivariant networks (`SetTransformerCovarianceNet`, `SetTransformerPolicyNet`) generalize across arbitrary universe sizes.

### `WalkForwardTrainer` & `TorchCostAdapter`

`WalkForwardTrainer` trains any PyTorch model causally across `BaseWindowSplitter` splits, clipping gradients (`max_norm=1.0`) and optionally invoking `model.forward_factors(...)` for \(O(N m^2 + m^3)\) Woodbury covariance losses.

```python
import torch
from pyalloq.dl.dataset import UniverseSubsampler
from pyalloq.dl.trainer import WalkForwardTrainer
from pyalloq.dl.loss_functions.woodbury import WoodburyGaussianNLLLoss
from pyalloq.dl.estimators.covariance.models.set_covariance import SetTransformerCovarianceNet
from pyalloq_backtest.splitters import RollingWindowSplitter

model = SetTransformerCovarianceNet(
    n_features=1,
    hidden_dim=64,
    num_heads=4,
    num_inds=16,
    head_type="factor_diag",
    use_ssmax=True,
)

trainer = WalkForwardTrainer(
    model=model,
    loss_function=WoodburyGaussianNLLLoss(),
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    splitter=RollingWindowSplitter(lookback_window=252),
    lookback_window=60,
    horizon=1,
    batch_size=32,
    epochs_per_window=10,
    return_mask=True,
    allow_ragged=True,
    universe_subsampler=UniverseSubsampler(min_assets=4),
    use_forward_factors=True,
)

rebal_dates = data.prices.resample("QE").last().index[4:]
history = trainer.train(data, rebalance_dates=rebal_dates)
```

For end-to-end portfolio losses, `TorchCostAdapter` wraps any `pyalloq_backtest.costs.BaseCostModel` inside `SharpeLoss` so neural policies optimize net of the exact same spread and market-impact costs used during backtesting.

---

## 2. Zero-Shot Foundation Models & Bayesian Ensembles

PyAlloq ships interchangeable zero-shot time-series foundation models that require **no task-specific training** and expose rich distributional diagnostics (`last_posterior_std_`, `last_posterior_cvar_`, `last_posterior_downside_std_`, `last_posterior_skewness_`, `last_quantiles_`):

| Component | Role(s) | Key Features |
|-----------|---------|--------------|
| `ApolloPFNReturnEstimator` | `BaseReturnEstimator` & `BaseScenarioEstimator` | Time-aware Prior-Data Fitted Network (Potapczynski et al. 2026) with `RiemannBarDistribution` (128 bins), in-context Bayesian Ridge anchor, and Student-\(t\) / Gaussian / stratified-\(t\) copula scenario generation. |
| `ApolloPFNViewGenerator` | `BaseViewGenerator` | Emits Black-Litterman \((P=I, Q, \Omega)\) with `uncertainty_mode` in `{"posterior_variance", "downside_semivariance", "cvar_tail"}` and optional `full_covariance_omega`. |
| `TimesFM3UnivariateReturnEstimator` / `TimesFM3MultivariateReturnEstimator` | `BaseReturnEstimator` | Google TimesFM 3.0 zero-shot forecaster with `CovariateMode` / `CovariateKey` support. |
| `TimestralUnivariateReturnEstimator` / `TimestralMultivariateReturnEstimator` | `BaseReturnEstimator` | Mistral `Timestral-28M` zero-shot forecaster with covariate extraction. |
| `TimesFM3*ViewGenerator` / `Timestral*ViewGenerator` | `BaseViewGenerator` | Converts quantile forecasts \((p_{10}, p_{50}, p_{90})\) into CVaR- or variance-calibrated Black-Litterman views. |
| `EnsembleReturnEstimator` | `BaseReturnEstimator` & `BaseScenarioEstimator` | Combines multiple estimators via `"inverse_uncertainty"`, `"scrps_adaptive"`, `"cvar_adaptive"`, `"bpds"` (Bayesian Predictive Decision Synthesis), `"equal"`, or `"custom"` weighting with Law of Total Variance uncertainty. |
| `EnsembleViewGenerator` | `BaseViewGenerator` | Exact Bayesian precision pooling: \(\Omega_{\text{ens}} = (\sum_m \Omega_m^{-1})^{-1}\), \(Q_{\text{ens}} = \Omega_{\text{ens}}(\sum_m \Omega_m^{-1} Q_m)\). |

```python
from pyalloq.dl.estimators.apollopfn import ApolloPFNReturnEstimator
from pyalloq.dl.estimators.returns.timesfm import TimesFM3MultivariateReturnEstimator
from pyalloq.dl.estimators.ensemble import EnsembleReturnEstimator
from pyalloq_core.enums import CovariateMode
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq.classical.optimizers.markowitz import MarkowitzAllocator

ens_est = EnsembleReturnEstimator(
    estimators=[
        ApolloPFNReturnEstimator(horizon=21, covariate_mode=CovariateMode.AUTO),
        TimesFM3MultivariateReturnEstimator(horizon=21, covariate_mode=CovariateMode.AUTO),
    ],
    weighting_mode="scrps_adaptive",
    validation_horizon=10,
)

pipeline = StrategyPipeline(
    returns_estimator=ens_est,
    allocator=MarkowitzAllocator(tickers=data.assets),
)
```

---

## 3. Neural Return & Continuous-Time SDE Estimators

- **`DeepReturnEstimator(model, lookback_window=60)`**: Wraps any PyTorch return network (such as `TFTDeepReturnsNet` — Temporal Fusion Transformer with Variable Selection Network, LSTM locality encoder, and Interpretable Multi-Head Attention) as a drop-in `BaseReturnEstimator`.
- **`NeuralSDEEstimator`**: Combines `ArbitrageFreeNeuralSDENet` (enforcing a Hansen-Jagannathan market-price-of-risk cap \(\|\theta\|_2 \le \theta_{\max}\)) with depth-2 Lyndon log-signature MMD matching. Implements three paradigms: `estimate(data)` (annualized returns), `estimate_covariance(data)` (50/50 instantaneous + horizon-integrated PSD covariance), and `estimate_scenarios(data)` / `generate(n_paths, horizon)` (`ScenarioMarketData`).

---

## 4. Deep Covariance Estimators

Wrap any neural covariance architecture in `DeepCovarianceEstimator(model, lookback_window=60, allow_ragged=False, graph_prior=None)` to plug it directly into `StrategyPipeline(cov_estimator=...)`.

| Architecture | Complexity | Key Mechanism |
|--------------|------------|---------------|
| `SetTransformerCovarianceNet` | \(O(N m)\) (`ISAB`) | Permutation- and size-equivariant backbone with 3 selectable heads: `"factor_diag"` (\(\hat{\Sigma}_t = B_t \Sigma_{F,t} B_t^\top + D_t\) with \(O(Nm^2+m^3)\) Woodbury ops), `"gram_corr"` (shrinkage equicorrelation), and `"rie_spectral"` (Rotational Invariant Estimator eigenvalue cleaning). |
| `CRISPSparseSTGAT` | \(O(N^2)\) pruned | `NeuralJumpRDE` temporal encoder + CRISP top-\(k\) sparse spatio-temporal graph attention + Fiedler \(\lambda_2\) / Hawkes \(\rho_G\) fragility-gated `FragilityShrinkageCholeskyLayer`. |
| `CrossAttentionCholeskyNet` | \(O(N^2)\) | LSTM temporal encoder + cross-asset multi-head attention + `CholeskyCovarianceLayer` (\(\Sigma = L L^\top\)). |

When `RelationalGraphPriorModule` (or `data.features["relational_graph"]` / `data.cross_sectional["sector"]`) is provided, `DeepCovarianceEstimator` automatically injects an additive Graphormer `attn_bias` into cross-asset attention.

---

## 5. End-to-End Neural & Differentiable Optimization Allocators

### `DeepAllocator` & Policy Backbones

`DeepAllocator(model, lookback_window=60)` wraps any `BasePolicyNetwork` as a standard `BaseAllocator`:

- **`SetTransformerPolicyNet`**: Size- and permutation-equivariant policy network supporting 5 institutional mandates:
  - `"capped_simplex"` (`CappedSimplexProjection`: exact Euclidean projection onto \(\sum w_i = 1, 0 \le w_i \le w_{\max}\) with exact implicit Jacobian)
  - `"sparsemax"` (exact Euclidean simplex projection, \(w_{\max} = 1\))
  - `"masked_dirichlet"` (`MaskedDirichletCashHead`: supports optional cash sleeve and reparameterized Dirichlet sampling)
  - `"dollar_neutral"` (`DollarNeutralStatArbHead`: \(\sum w_i = 0, \sum |w_i| = G_{\text{gross}}\))
  - `"vol_targeted"` (`VolTargetedContinuousRiskHead`: inverse-volatility risk-scaled weights)
- **`ConvTransformerAllocator`** & **`ConstrainedSpatioTemporalAttentionNet`**: Spatio-temporal attention allocators with `CappedSimplexProjection` / `BoundedSoftmax` water-filling caps.
- **`UnifiedE2EPolicyNetwork`**: 3-stage perception (`CRISPSparseSTGAT` + `LogSigSSMBlock` selective state-space model) \(\to\) differentiable optimization head (`"mdfp"`, `"risk_parity_kkt"`, `"cvar_kkt"`, or `"residual_actor"`) \(\to\) no-trade-zone buffer.
- **`DifferentiableKKTAllocatorModule`**: Chains any covariance/return backbone directly into `DifferentiableRiskParityLayer` (Spinu Newton-KKT with exact implicit Jacobian) or `DifferentiableCVaRLayer` (entropic mirror descent with Neumann-series implicit backward pass).

### `IPMOAllocator` (Implicit Multi-Period Decision-Focused Learning)

`IPMOAllocator` pairs `MultiHorizonPredictorNet` with `MDFPOptimizationLayer` (coupled \(H\)-step entropic mirror descent with \(O(1)\)-memory Neumann implicit differentiation) and returns a `MultiPeriodOptimizationResult`. Train it end-to-end with `SPOPlusLoss` or `HybridDFLLoss`.

```python
from pyalloq.dl.estimators.e2e.ipmo_allocator import MultiHorizonPredictorNet, IPMOModule, IPMOAllocator
from pyalloq.dl.layers.mdfp import MDFPOptimizationLayer
from pyalloq.dl.loss_functions.dfl import HybridDFLLoss

predictor = MultiHorizonPredictorNet(n_assets=len(data.assets), n_features=1, horizon=5)
mdfp_layer = MDFPOptimizationLayer(horizon=5, risk_aversion=2.0, cost_penalty=0.002)
ipmo_module = IPMOModule(predictor=predictor, mdfp_layer=mdfp_layer)

# Train ipmo_module with HybridDFLLoss or SPOPlusLoss, then deploy:
allocator = IPMOAllocator(predictor=predictor, mdfp_layer=mdfp_layer, lookback_window=60)
```

---

## 6. Self-Supervised Market World Model (`MarketJEPA`)

`MarketJEPA` implements a Joint-Embedding Predictive Architecture over `SetTransformerBackbone` with an EMA target encoder (\(\xi \leftarrow \tau \xi + (1-\tau)\theta\)), `LatentPredictor`, and mask-aware `VICRegLoss` (invariance + variance hinge + covariance decorrelation). Pretrain `MarketJEPA` on unlabeled multi-asset panels, then transfer its backbone weights into `SetTransformerCovarianceNet` or `SetTransformerPolicyNet`.

---

## 7. Differentiable Loss Functions Summary

| Loss Function | Module | Target Use Case |
|---------------|--------|-----------------|
| `WoodburyGaussianNLLLoss` | `pyalloq.dl.loss_functions.woodbury` | \(O(N m^2 + m^3)\) factor+diagonal Gaussian NLL for `SetTransformerCovarianceNet`. |
| `WoodburySteinLoss` | `pyalloq.dl.loss_functions.woodbury` | Cardinality-normalized James-Stein / KL covariance loss. |
| `CholeskyNLLLoss` | `pyalloq.dl.loss_functions.cholesky_nll` | Triangular Cholesky Gaussian NLL for `CrossAttentionCholeskyNet`. |
| `SharpeLoss` | `pyalloq.dl.loss_functions.sharpe` | Differentiable net-of-cost Sharpe ratio using `TorchCostAdapter`. |
| `SoftMinEVaRLoss` | `pyalloq.dl.loss_functions.woodbury` | Entropic Value-at-Risk + soft-max drawdown penalty (DeePM). |
| `SPOPlusLoss` / `HybridDFLLoss` | `pyalloq.dl.loss_functions.dfl` | Decision-Focused Learning (Smart Predict-then-Optimize) through `MDFPOptimizationLayer`. |
| `VICRegLoss` | `pyalloq.dl.world_models.jepa` | Mask-aware self-supervised representation pretraining in `MarketJEPA`. |

---

## Further Reading

- **[Deep Learning Mathematics](../mathematics/deep-learning.md)** — Exact equations for Woodbury identities, Capped Simplex & Risk-Parity KKT Jacobians, MDFP Neumann IFT, ApolloPFN Riemann Bar Distribution, and VICReg.
- **[API Reference: Deep Learning](../api/dl.md)** — Full class and method signatures.
