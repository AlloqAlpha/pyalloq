# Covariance Estimators

Covariance estimators produce an **NxN annualized covariance matrix** from a `MarketData` object. The choice of estimator has a major impact on optimizer stability, especially for Markowitz-family allocators.

All estimators implement `BaseCovarianceEstimator`:

```python
class BaseCovarianceEstimator(ABC):
    @abstractmethod
    def estimate(self, data: MarketData, **kwargs) -> pd.DataFrame:
        ...
```

---

## Empirical (`EmpiricalCovariance`)

The classical sample covariance matrix. Simple and unbiased, but sensitive to outliers and unreliable when the number of observations \(T\) is close to the number of assets \(N\).

\[
\hat{\Sigma} = \frac{1}{T-1} \sum_{t=1}^{T} (r_t - \bar{r})(r_t - \bar{r})^\top
\]

```python
from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance

pipeline = StrategyPipeline(
    allocator=allocator,
    cov_estimator=EmpiricalCovariance(),
)
```

**When to use:** Large \(T/N\) ratio (e.g., 10+ years of daily data for 20 stocks). Stable, liquid markets.

---

## EWMA (`EWMACovariance`)

Exponentially Weighted Moving Average covariance. More recent observations receive exponentially higher weight, capturing volatility clustering (GARCH-like behaviour).

\[
\hat{\Sigma}_t = (1-\lambda) r_{t} r_{t}^\top + \lambda \hat{\Sigma}_{t-1}
\]

```python
from pyalloq.classical.estimators.covariance.ewma import EWMACovariance

pipeline = StrategyPipeline(
    allocator=allocator,
    cov_estimator=EWMACovariance(span=120),  # 120-day span
)
```

**When to use:** Volatile or crisis periods where recent correlations matter more. Time-varying risk models.

---

## Ledoit-Wolf (`LedoitWolfShrinkage`)

Analytically optimal shrinkage of the sample covariance towards a structured target (e.g., identity or scaled identity). Minimizes mean squared error of the estimator.

\[
\hat{\Sigma}^{LW} = (1 - \alpha^*) \hat{\Sigma} + \alpha^* \mu \mathbf{I}
\]

where \(\alpha^*\) is the analytically derived optimal shrinkage intensity.

```python
from pyalloq.classical.estimators.covariance.ledoit_wolf import LedoitWolfShrinkage

pipeline = StrategyPipeline(
    allocator=allocator,
    cov_estimator=LedoitWolfShrinkage(),
)
```

**When to use:** High-dimensional portfolios (\(N > 50\)), short histories. The go-to robust estimator for Markowitz optimization.

See: [Ledoit & Wolf (2004)](../references/papers.md#ledoit-wolf-2004)

---

## Random Matrix Theory (`RandomMatrixTheoryEstimator`)

Filters the empirical covariance matrix by removing **noise eigenvalues** that fall below the Marchenko-Pastur upper bound. These eigenvalues are statistical artifacts of finite sample size, not true economic factors.

**Denoising step:** Any eigenvalue \(\lambda_i\) below the Marchenko-Pastur bound \(\lambda_+\) is replaced by the average noise eigenvalue:

\[
\lambda_+ = \sigma^2 \left(1 + \sqrt{\frac{N}{T}}\right)^2
\]

**Detoning step (optional):** Removes the largest eigenvalue (the "market mode") to isolate sector-level correlations.

```python
from pyalloq.classical.estimators.covariance.random_matrix_theory import RandomMatrixTheoryEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    cov_estimator=RandomMatrixTheoryEstimator(
        denoise=True,   # remove noise eigenvalues
        detone=False,   # keep the market factor
    ),
)
```

**When to use:** Large universes where the empirical correlation matrix is dominated by noise. Pairs well with HRP and NCO allocators.

See: [López de Prado (2020) — Advances in Financial Machine Learning](../references/papers.md)

---

## Semi-Covariance (`SemiCovariance`)

Computes covariance using only **downside returns** (negative returns). This targets downside risk specifically, which is more relevant for loss-averse investors than full variance.

\[
\hat{\Sigma}^{semi}_{ij} = \frac{1}{T^-} \sum_{t: r_{i,t} < 0} r_{i,t} \cdot r_{j,t}
\]

```python
from pyalloq.classical.estimators.covariance.semi_covariance import SemiCovariance

pipeline = StrategyPipeline(
    allocator=allocator,
    cov_estimator=SemiCovariance(),
)
```

**When to use:** When minimizing downside risk is the primary objective. Pairs naturally with `MarkowitzAllocator(MIN_VOLATILITY)` for drawdown-conscious portfolios.

---

## Option-Implied Covariance (`ImpliedCovarianceEstimator`)

Combines forward-looking **Point-in-Time (PIT) Option-Implied Volatilities** (from `data.features["implied_vol"]`, such as 30-day Model-Free Implied Volatility computed by `PITOptionsMFIVTransformer`) with a Ledoit-Wolf shrunk correlation matrix and the **Buss & Vilkov (2012)** implied correlation adjustment:

\[
\Gamma_t^{\text{imp}} = (1 - \gamma_t) \Gamma_t^{\text{hist}} + \gamma_t \mathbf{1}\mathbf{1}^\top, \qquad \hat{\Sigma}_t^{\text{imp}} = D_{\text{IV}, t} \, \Gamma_t^{\text{imp}} \, D_{\text{IV}, t}
\]

When an index implied volatility `index_implied_vol` is supplied, \(\gamma_t\) is solved in closed form so the implied market portfolio variance matches \(\sigma_{M,\text{IV}}^2\). Enabling `use_cornish_fisher_tail=True` applies a **Cornish-Fisher four-moment tail expansion** using Bakshi-Kapadia-Madan option-implied skewness (`"implied_skew"`) and excess kurtosis (`"implied_kurtosis"`) to inflate standalone implied volatilities for negatively skewed or fat-tailed assets. You can also directly extract the \(N \times N\) implied correlation matrix \(\Gamma_t^{\text{imp}}\) via `.estimate_correlation(data)` or `estimator.last_implied_correlation_`.

```python
from pyalloq_features.transformers import PITOptionsMFIVTransformer
from pyalloq.classical.estimators.covariance import ImpliedCovarianceEstimator

data = PITOptionsMFIVTransformer(option_chain_df=chains_df).transform(data)

imp_cov = ImpliedCovarianceEstimator(
    index_implied_vol=0.22,
    correlation_window=60,
    use_cornish_fisher_tail=True,
    tail_confidence=0.95,
)
corr_imp_df = imp_cov.estimate_correlation(data)  # Direct N x N implied correlation matrix

pipeline = StrategyPipeline(
    allocator=allocator,
    cov_estimator=imp_cov,
)
```

**When to use:** Earnings seasons, macro event windows, or regime transitions where forward-looking option surfaces price in volatility and correlation risk before historical returns react.

---

## Graph Reaction-Diffusion (`ReactionDiffusionCovarianceEstimator`)

Models cross-asset volatility spillover and non-linear self-excitation on a dynamic **Financial Precision Graph** \(\mathcal{L}_t = I - D_t^{-1/2} W_t D_t^{-1/2}\) via a coupled FitzHugh-Nagumo Reaction-Diffusion ODE system integrated via 4th-order Runge-Kutta (RK4):

\[
\frac{dv}{dt} = -D_v \mathcal{L}_t v - \kappa_v v + \gamma_{\text{react}} v (v - \theta_{\text{crit}})(1 - v) - \beta_u u
\]

```python
from pyalloq.classical.estimators.covariance import ReactionDiffusionCovarianceEstimator

pipeline = StrategyPipeline(
    allocator=allocator,
    cov_estimator=ReactionDiffusionCovarianceEstimator(
        diffusion_coeff=0.35,
        horizon_steps=5,
    ),
)
```

**When to use:** Sector contagion and supply-chain spillover regimes where a localized volatility shock on one asset propagates to connected neighbors ahead of trailing sample windows. Also computes `last_diagnostics_` containing the Graph Laplacian's **Fiedler value** \(\lambda_2(\mathcal{L}_t)\), **network mixing time** \(\tau_{\text{mix}}\), **Fiedler vector** \(v_2\), and **Wiener-Hopf Multivariate Hawkes** branching matrix \(G = \|\Phi\|_1\) and spectral radius \(\rho(G)\).

---

## Deep Spatio-Temporal Graph Covariance (`CRISPSparseSTGAT` + `DeepCovarianceEstimator`)

Combines a continuous-time **Neural Rough Differential Equation with Hawkes Jump Discontinuities** (`NeuralJumpRDE`) at the node level with a **CRISP Sparse Multi-Head Graph Attention** layer and a **Network-Fragility Shrinkage Cholesky Layer** (`FragilityShrinkageCholeskyLayer`) to forecast strictly Positive Semi-Definite (PSD) covariance matrices:

```python
from pyalloq.dl import CRISPSparseSTGAT, DeepCovarianceEstimator

stgat = CRISPSparseSTGAT(
    num_assets=len(data.assets),
    num_features=1,
    hidden_dim=32,
    num_heads=4,
    keep_ratio=0.35,
)

cov_estimator = DeepCovarianceEstimator(model=stgat, lookback_window=30)
cov_df = cov_estimator.estimate(data)
```

---

## Size- and Permutation-Equivariant Set Transformer Covariance (`SetTransformerCovarianceNet`)

Unlike lower-triangular Cholesky heads (`torch.tril`) whose output depends on arbitrary DataFrame column ordering, `SetTransformerCovarianceNet` enforces exact **2D Congruence Equivariance** (\(\hat{\Sigma}(\Pi X) = \Pi \hat{\Sigma}(X) \Pi^\top\)) and zero-shot universe-size transfer via **Induced Set Attention Blocks (`ISAB`)** and **Scalable-Softmax (`SSMax`)**.

It supports three pluggable PSD covariance heads:

- **`head_type="factor_diag"` (`ISABFactorDiagonalHead`, Default)**: Reuses the equivariant asset embeddings \(Z_t\) and invariant inducing-point factor states \(H_{I,t}\) to form \(\hat{\Sigma}_t = B_t \Sigma_{F,t} B_t^\top + D_t\). Supports exact \(\mathcal{O}(N m^2 + m^3)\) training and GMV solves via `forward_factors(x, mask)` and `WoodburyGaussianNLLLoss` / `WoodburySteinLoss`.
- **`head_type="gram_corr"` (`EquivariantGramCorrHead`)**: Volatility-scaled \(\ell_2\)-normalized Gram correlation matrix blended with learnable `[REGIME]` shrinkage \(\lambda_t \in (0, 1)\) and optional Point-in-Time implied-volatility anchoring.
- **`head_type="rie_spectral"` (`RIESpectralHead`)**: Rotation- and permutation-invariant eigenvalue cleaning inside the empirical correlation eigenbasis conditioned on the aspect ratio \(q_{\text{in}} = N_{\text{active}} / L\).

```python
import torch
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq.dl import (
    DeepCovarianceEstimator,
    SetTransformerCovarianceNet,
    UniverseSubsampler,
    WoodburyGaussianNLLLoss,
)
from pyalloq.dl.trainer import WalkForwardTrainer

set_cov_net = SetTransformerCovarianceNet(
    n_features=1,
    hidden_dim=64,
    num_heads=4,
    num_inds=16,
    head_type="factor_diag",      # or "gram_corr" | "rie_spectral"
    temporal_type="vsn_conv",     # "lstm" | "vsn_conv" | "logsig_ssm" | "neural_jump_rde"
    directed_delay=True,          # Optional (t-1) causal sieve for multi-timezone closes
)

trainer = WalkForwardTrainer(
    model=set_cov_net,
    loss_function=WoodburyGaussianNLLLoss(normalize_by_cardinality=True),
    optimizer=torch.optim.AdamW(set_cov_net.parameters(), lr=2e-3),
    splitter=RollingWindowSplitter(lookback_window=120),
    lookback_window=30,
    return_mask=True,
    universe_subsampler=UniverseSubsampler(min_assets=10),
    use_forward_factors=True,     # Invokes O(Nm^2 + m^3) Woodbury pathway
)

cov_est = DeepCovarianceEstimator(model=set_cov_net, lookback_window=30, allow_ragged=True)
cov_df = cov_est.estimate(data)
factor_loadings_df, sigma_f_np, idio_var_ser = cov_est.estimate_factors(data)
```

---

## Choosing a Covariance Estimator

| Scenario | Recommended Estimator |
|----------|-----------------------|
| Many observations, stable market | `EmpiricalCovariance` |
| Volatile regime, recent correlations important | `EWMACovariance` |
| High-dimensional, short history | `LedoitWolfShrinkage` |
| Large universe, correlation noise removal | `RandomMatrixTheoryEstimator` |
| Downside risk focus | `SemiCovariance` |
| Forward-looking option-implied risk & correlation | `ImpliedCovarianceEstimator` |
| Cross-asset network volatility spillover & contagion | `ReactionDiffusionCovarianceEstimator` |
| State-conditioned historical similarity (Mahalanobis relevance) | `RelevanceConditionedCovarianceEstimator` |
| Spectral graph fragility & Wiener-Hopf Hawkes jump contagion | `SpectralGraphFragility`, `WienerHopfHawkesSolver` |

---

## Relevance-Conditioned & Spectral Hawkes Diagnostics

- **`RelevanceConditionedCovarianceEstimator`** (`pyalloq.classical.estimators.relevance`): Computes Czasonis-Kritzman-Turkington Mahalanobis relevance weights over historical market state vectors, censors low-relevance observations below the \((1-\phi)\) quantile, and blends relevance-weighted scatter with a Ledoit-Wolf prior weighted by `adjusted_fit`.
- **`SpectralGraphFragility` & `WienerHopfHawkesSolver`** (`pyalloq.classical.estimators.covariance.hawkes_spectral`): Extracts the Graph Laplacian Fiedler value \(\lambda_2\), mixing time \(\tau_{\text{mix}} = 1/\lambda_2\), and non-parametric Wiener-Hopf Hawkes jump branching matrix \(G\) (with spectral radius \(\rho(G)\)) via `compute_spectral_hawkes_diagnostics(returns, laplacian)`.

---

## Implementing a Custom Covariance Estimator

```python
from pyalloq_core.interfaces import BaseCovarianceEstimator
from pyalloq_core.data import MarketData
import pandas as pd

class MyCovariance(BaseCovarianceEstimator):
    def estimate(self, data: MarketData, **kwargs) -> pd.DataFrame:
        returns = data.prices.pct_change().dropna()
        return returns.cov() * 252

pipeline = StrategyPipeline(
    allocator=allocator,
    cov_estimator=MyCovariance(),
)
```
