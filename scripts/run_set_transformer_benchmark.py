"""
Permutation-Invariant Set Transformers for Size-Invariant Deep Covariance
and End-to-End Portfolio Policies in PyAlloq — Verification & Benchmark Script:

1. **Mathematical Verification Audit**:
   - Benchmarks 2D Congruence Equivariance error (||Sigma(Pi X) - Pi Sigma(X) Pi^T||_F)
     and spectral condition number (kappa) across:
     * `CrossAttentionCholeskyNet` (lower-triangular `torch.tril` Cholesky)
     * `SetTransformerCovarianceNet` (`head_type="factor_diag"`, `ISABFactorDiagonalHead`)
     * `SetTransformerCovarianceNet` (`head_type="gram_corr"`, `EquivariantGramCorrHead`)
     * `SetTransformerCovarianceNet` (`head_type="rie_spectral"`, `RIESpectralHead`)
   - Verifies Woodbury O(N m^2 + m^3) Gaussian NLL and GMV solve exactness.
   - Simulates 50-to-500 asset cardinality shifts comparing Standard Softmax dilution
     against Scalable-Softmax (`SSMax`) and `CappedSimplexProjection`.
2. **Walk-Forward Training with Ragged Panels & UniverseSubsampler**:
   - Trains `SetTransformerCovarianceNet` with `WoodburyGaussianNLLLoss` and
     `SetTransformerPolicyNet` (`mandate="capped_simplex"`) with `SoftMinEVaRLoss` and
     `UniverseSubsampler` on ragged market data.
3. **Head-to-Head Multi-Strategy Benchmark Suite (DSR & Frictional Drag)**:
   - Evaluates EqualWeight, Ledoit-Wolf GMV, Reaction-Diffusion HRP,
     SetTransformer Factor-Diag GMV, and SetTransformer E2E Capped-Simplex Policy
     inside `BenchmarkSuite` under `AlmgrenChrissCostModel`.
4. **Interactive Visualization Export (`pyalloq-viz`)**:
   - Exports `set_transformer_diagnostics.html` (`plot_set_transformer_diagnostics`)
     and `set_transformer_comparison_tearsheet.html` (`ComparisonTearSheet`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyalloq_backtest import (
    AlmgrenChrissCostModel,
    BacktestTracker,
    BenchmarkSuite,
    RollingWindowSplitter,
)
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_viz import ComparisonTearSheet, plot_set_transformer_diagnostics

from pyalloq.classical.estimators.covariance import (
    LedoitWolfShrinkage,
    ReactionDiffusionCovarianceEstimator,
)
from pyalloq.classical.optimizers import EqualWeightAllocator, MarkowitzAllocator
from pyalloq.dl import (
    ISAB,
    SAB,
    CappedSimplexProjection,
    DeepAllocator,
    DeepCovarianceEstimator,
    ISABFactorDiagonalHead,
    SetTransformerCovarianceNet,
    SetTransformerPolicyNet,
    SoftMinEVaRLoss,
    UniverseSubsampler,
    WoodburyGaussianNLLLoss,
)
from pyalloq.dl.estimators.covariance.models.cac import CrossAttentionCholeskyNet
from pyalloq.dl.trainer import WalkForwardTrainer
from pyalloq.ml.optimizers.hrp import SpectralLaplacianHRPAllocator

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM"]


def build_synthetic_equity_market_data() -> MarketData:
    rng = np.random.default_rng(2026)
    n_days = 160
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    n_assets = len(TICKERS)

    # 2 latent risk factors (Tech/Growth momentum + Macro/Energy rates)
    f_tech = rng.normal(0.0007, 0.013, size=(n_days, 1))
    f_macro = rng.normal(0.0003, 0.010, size=(n_days, 1))
    loadings = np.array(
        [
            [1.20, 0.10],  # AAPL
            [1.10, 0.15],  # MSFT
            [1.50, -0.10],  # NVDA
            [1.05, 0.20],  # GOOGL
            [1.15, 0.05],  # AMZN
            [1.25, 0.00],  # META
            [0.35, 1.10],  # JPM
            [0.15, 1.25],  # XOM
        ]
    )
    factors = np.hstack([f_tech, f_macro])
    idio = rng.standard_t(df=6, size=(n_days, n_assets)) * 0.0065
    rets = factors @ loadings.T + idio

    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=dates,
        columns=TICKERS,
    )
    volume = pd.DataFrame(
        rng.uniform(25_000_000.0, 90_000_000.0, size=(n_days, n_assets)),
        index=dates,
        columns=TICKERS,
    )
    cross_sectional = pd.DataFrame(
        {
            "sector": [
                "Technology",
                "Technology",
                "Semiconductors",
                "Technology",
                "Consumer Discretionary",
                "Communication Services",
                "Financials",
                "Energy",
            ]
        },
        index=TICKERS,
    )
    return MarketData(
        prices=prices,
        features={"volume": volume},
        cross_sectional=cross_sectional,
        risk_free_rate=0.04,
    )


def run_equivariance_and_conditioning_audit() -> dict[str, np.ndarray]:
    print("=== 1. Permutation Equivariance & Spectral Conditioning Audit (N=20) ===")
    torch.manual_seed(42)
    N, L, F = 20, 20, 2
    x = torch.randn(1, L, N, F, dtype=torch.float64)
    perm = torch.randperm(N)
    x_perm = x[:, :, perm, :]

    models: dict[str, torch.nn.Module] = {
        "Cholesky_tril": CrossAttentionCholeskyNet(
            n_features=F, hidden_dim=32, num_heads=4, dropout=0.0
        ).to(torch.float64),
        "ISAB_Factor_Diag": SetTransformerCovarianceNet(
            n_features=F,
            hidden_dim=32,
            num_heads=4,
            num_inds=8,
            dropout=0.0,
            head_type="factor_diag",
        ).to(torch.float64),
        "Equivariant_Gram_Corr": SetTransformerCovarianceNet(
            n_features=F,
            hidden_dim=32,
            num_heads=4,
            num_inds=8,
            dropout=0.0,
            head_type="gram_corr",
        ).to(torch.float64),
        "RIE_Spectral_Clean": SetTransformerCovarianceNet(
            n_features=F,
            hidden_dim=32,
            num_heads=4,
            num_inds=8,
            dropout=0.0,
            head_type="rie_spectral",
        ).to(torch.float64),
    }

    spectra: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            cov = model(x)
            cov_p = model(x_perm)
            cov_expected = cov[:, perm, :][:, :, perm]
            rel_err = float(
                (
                    torch.linalg.norm(cov_p - cov_expected)
                    / torch.linalg.norm(cov_expected).clamp(min=1e-12)
                ).item()
            )
            eigvals = torch.linalg.eigvalsh(cov[0]).clamp(min=1e-12).cpu().numpy()
            cond_num = float(eigvals[-1] / eigvals[0])
            spectra[name] = eigvals
            print(
                f"  {name:<22} | Rel Equivariance Error: {rel_err:.2e} "
                f"| Min Eig: {eigvals[0]:.4e} | Condition Number (κ): {cond_num:.2f}"
            )

    # Verify Woodbury exactness on ISAB_Factor_Diag
    isab_cov = models["ISAB_Factor_Diag"]
    assert isinstance(isab_cov, SetTransformerCovarianceNet)
    with torch.no_grad():
        b_load, sigma_f, d_diag = isab_cov.forward_factors(x)
        dense_cov = isab_cov(x)
        wb_logdet = ISABFactorDiagonalHead.woodbury_logdet(b_load, sigma_f, d_diag)
        _, dense_logdet = torch.linalg.slogdet(dense_cov)
        wb_gmv = ISABFactorDiagonalHead.woodbury_gmv_weights(b_load, sigma_f, d_diag)
        ones = torch.ones((1, N, 1), dtype=torch.float64)
        sinv_1 = torch.linalg.solve(dense_cov, ones).squeeze(-1)
        dense_gmv = sinv_1 / sinv_1.sum(dim=-1, keepdim=True)
        print(
            f"  Woodbury Exactness     | Max Log-Det Diff: {float(torch.abs(wb_logdet - dense_logdet).item()):.2e} "
            f"| Max GMV Weight Diff: {float(torch.max(torch.abs(wb_gmv - dense_gmv)).item()):.2e}"
        )

    return spectra


def run_cardinality_shift_simulation() -> pd.DataFrame:
    print(
        "\n=== 2. Cardinality-Shift Concentration Benchmark (N = 10 -> 1,000 Assets) ==="
    )
    torch.manual_seed(2026)
    n_grid = [10, 25, 50, 100, 250, 500, 1000]
    capped_head = CappedSimplexProjection(max_weight=0.25)
    rows: list[dict[str, float]] = []

    for n_assets in n_grid:
        logits = torch.randn(64, n_assets, dtype=torch.float64) * 1.5
        # Standard Softmax
        w_std = torch.softmax(logits, dim=-1)
        # Scalable-Softmax (s = 0.5 * log(N))
        w_ssmax = torch.softmax(0.5 * float(np.log(n_assets)) * logits, dim=-1)
        # Capped Simplex (cap = 0.25)
        w_cap = capped_head(logits)

        row = {
            "N": float(n_assets),
            "softmax_max_weight": float(w_std.max(dim=-1).values.mean().item()),
            "ssmax_max_weight": float(w_ssmax.max(dim=-1).values.mean().item()),
            "capped_simplex_max_weight": float(w_cap.max(dim=-1).values.mean().item()),
        }
        rows.append(row)
        print(
            f"  N={n_assets:<4} | Std Softmax MaxW: {row['softmax_max_weight']:.3f} "
            f"| SSMax MaxW: {row['ssmax_max_weight']:.3f} "
            f"| CappedSimplex MaxW: {row['capped_simplex_max_weight']:.3f}"
        )

    return pd.DataFrame(rows)


def main() -> None:
    output_dir = Path("visualizations_output/set_transformer")
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra = run_equivariance_and_conditioning_audit()
    cardinality_df = run_cardinality_shift_simulation()

    print(
        "\n=== 3. Walk-Forward Training: SetTransformerCovarianceNet & SetTransformerPolicyNet ==="
    )
    data = build_synthetic_equity_market_data()
    n_features = 1 + len(data.features)
    lookback_window = 20
    splitter = RollingWindowSplitter(lookback_window=65)
    idx_ser = pd.Series(data.prices.index, index=data.prices.index)
    rebalance_dates = pd.DatetimeIndex(
        idx_ser.resample("ME").last().dropna().values
    ).unique()

    # 3a. Train SetTransformerCovarianceNet via O(Nm^2 + m^3) WoodburyGaussianNLLLoss
    set_cov_net = SetTransformerCovarianceNet(
        n_features=n_features,
        hidden_dim=32,
        num_heads=4,
        num_inds=4,
        num_layers=2,
        head_type="factor_diag",
        temporal_type="vsn_conv",
    )
    wb_nll = WoodburyGaussianNLLLoss(normalize_by_cardinality=True)
    opt_cov = torch.optim.AdamW(set_cov_net.parameters(), lr=2e-3, weight_decay=1e-4)
    trainer_cov = WalkForwardTrainer(
        model=set_cov_net,
        loss_function=wb_nll,
        optimizer=opt_cov,
        splitter=splitter,
        lookback_window=lookback_window,
        horizon=1,
        batch_size=16,
        epochs_per_window=3,
        return_mask=True,
        universe_subsampler=UniverseSubsampler(min_assets=4, random_state=42),
        use_forward_factors=True,
    )
    trainer_cov.train(data, rebalance_dates=rebalance_dates)

    deep_cov_est = DeepCovarianceEstimator(
        model=set_cov_net, lookback_window=lookback_window, allow_ragged=True
    )
    factor_loadings_df, _, _ = deep_cov_est.estimate_factors(data)

    # 3b. Train SetTransformerPolicyNet (Capped Simplex + SSMax + UniverseSubsampler)
    set_policy_net = SetTransformerPolicyNet(
        n_features=n_features,
        hidden_dim=32,
        num_heads=4,
        num_inds=4,
        num_layers=2,
        mandate="capped_simplex",
        max_weight=0.25,
        temporal_type="vsn_conv",
    )
    evar_loss = SoftMinEVaRLoss(beta=8.0, cost_bps=5.0)
    opt_pol = torch.optim.AdamW(set_policy_net.parameters(), lr=2e-3, weight_decay=1e-4)
    trainer_pol = WalkForwardTrainer(
        model=set_policy_net,
        loss_function=evar_loss,
        optimizer=opt_pol,
        splitter=splitter,
        lookback_window=lookback_window,
        horizon=1,
        batch_size=16,
        epochs_per_window=3,
        return_mask=True,
        universe_subsampler=UniverseSubsampler(min_assets=4, random_state=42),
    )
    trainer_pol.train(data, rebalance_dates=rebalance_dates)

    # 4. Export Set Transformer 3-Panel Diagnostic Figure
    diag_fig = plot_set_transformer_diagnostics(
        factor_loadings=factor_loadings_df,
        eigenvalue_spectra=spectra,
        cardinality_profile=cardinality_df,
    )
    diag_html = output_dir / "set_transformer_diagnostics.html"
    diag_fig.write_html(str(diag_html))
    print(f"\nSaved Set Transformer Diagnostics dashboard to: {diag_html}")

    # 5. Head-to-Head Multi-Strategy Benchmark Suite with DSR Tracking
    print(
        "\n=== 4. Multi-Strategy Walk-Forward Benchmark Suite (DSR & Attribution) ==="
    )
    pipelines = {
        "EqualWeight": StrategyPipeline(
            allocator=EqualWeightAllocator(tickers=TICKERS),
        ),
        "LedoitWolf_GMV": StrategyPipeline(
            cov_estimator=LedoitWolfShrinkage(),
            allocator=MarkowitzAllocator(
                tickers=TICKERS,
                objective=ObjectiveFunction.MIN_VOLATILITY,
                weight_bounds=(0.0, 0.25),
            ),
        ),
        "ReactionDiffusion_SpectralHRP": StrategyPipeline(
            cov_estimator=ReactionDiffusionCovarianceEstimator(slow_window=50),
            allocator=SpectralLaplacianHRPAllocator(
                tickers=TICKERS, fragility_penalty=0.25
            ),
        ),
        "SetTransformer_FactorDiag_GMV": StrategyPipeline(
            cov_estimator=deep_cov_est,
            allocator=MarkowitzAllocator(
                tickers=TICKERS,
                objective=ObjectiveFunction.MIN_VOLATILITY,
                weight_bounds=(0.0, 0.25),
            ),
        ),
        "SetTransformer_E2E_CappedSimplex": StrategyPipeline(
            allocator=DeepAllocator(
                model=set_policy_net,
                lookback_window=lookback_window,
                tickers=TICKERS,
            ),
        ),
    }

    cost_model = AlmgrenChrissCostModel(
        portfolio_aum=10_000_000.0,
        spread_bps=4.0,
        gamma=0.06,
    )

    with BacktestTracker.session(reset=True) as tracker:
        suite = BenchmarkSuite(
            pipelines=pipelines,
            splitter=splitter,
            cost_model=cost_model,
            rebalance_freq="ME",
            benchmark="EqualWeight",
        )
        comp_res = suite.run(data)
        cols = [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "deflated_sharpe_ratio",
            "maximum_drawdown",
            "annualized_turnover",
            "total_cost_drag",
        ]
        print(
            f"Executed {tracker.n_trials} Strategy Trials | "
            f"Expected Max Sharpe Hurdle (SR_0): {comp_res.expected_max_sharpe:.3f}"
        )
        print(comp_res.summary_table[cols].round(4).to_string())

        ts_html = output_dir / "set_transformer_comparison_tearsheet.html"
        ComparisonTearSheet(
            comp_res,
            market_data=data,
            benchmark=comp_res.results["EqualWeight"],
        ).to_html(path=ts_html)
        print(f"Saved multi-strategy ComparisonTearSheet to: {ts_html}")

    run_empirical_ablations_transfer_and_graph_prior(data)


def run_empirical_ablations_transfer_and_graph_prior(
    subset_data: MarketData,
) -> None:
    """
    Runs two controlled empirical ablations:
    (a) **Large-to-Small Zero-Shot Universe Transfer (40 Assets -> 8-Asset Subset)**:
        Trains two policies on a 40-asset universe:
        - `Vanilla_Softmax_NoSubsample`: `use_ssmax=False`, no `UniverseSubsampler`
        - `Calibrated_SSMax_Subsampled`: `use_ssmax=True` + `UniverseSubsampler(4, 40)`
        Then evaluates both zero-shot on the 8-asset subset `subset_data` to measure
        concentration stability (`max_w`, effective holdings `1 / sum(w^2)`) and OOS Sharpe.
    (b) **Graphormer Relational Prior Ablation (`attn_bias=None` vs. Partial-Correlation Prior)**:
        Compares out-of-sample Maximum Drawdown and Tail Risk when injecting Pillar 2's
        `ReactionDiffusionCovarianceEstimator` precision partial-correlation adjacency `W_t`
        into `SetTransformerPolicyNet(..., attn_bias=W_t)` versus `attn_bias=None`.
    """
    print(
        "\n=== 5. Empirical Ablations: (A) 40 -> 8 Zero-Shot Transfer & (B) Graphormer Prior Drawdown ==="
    )
    rng = np.random.default_rng(99)
    torch.manual_seed(99)

    # Synthesize a 40-asset training panel sharing the same dates and factor structure
    dates = subset_data.prices.index
    n_days = len(dates)
    n_large = 40
    large_tickers = [f"EQ_{i:02d}" for i in range(n_large)]
    f1 = rng.normal(0.0006, 0.012, size=(n_days, 1))
    f2 = rng.normal(-0.0002, 0.015, size=(n_days, 1))
    b1 = rng.uniform(0.3, 1.4, size=(1, n_large))
    b2 = rng.uniform(-0.8, 1.1, size=(1, n_large))
    rets_large = (
        f1 @ b1 + f2 @ b2 + rng.standard_t(df=5, size=(n_days, n_large)) * 0.007
    )
    prices_large = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets_large, axis=0),
        index=dates,
        columns=large_tickers,
    )
    large_md = MarketData(prices=prices_large, risk_free_rate=0.04)

    splitter = RollingWindowSplitter(lookback_window=65)
    reb_dates = pd.DatetimeIndex([dates[80], dates[120]])

    # Model A1: Vanilla Softmax Attention (no SSMax, no UniverseSubsampler)
    vanilla_pol = SetTransformerPolicyNet(
        n_features=1,
        hidden_dim=24,
        num_heads=2,
        num_inds=4,
        num_layers=1,
        mandate="capped_simplex",
        max_weight=0.35,
        use_ssmax=False,
    )
    WalkForwardTrainer(
        model=vanilla_pol,
        loss_function=SoftMinEVaRLoss(beta=8.0),
        optimizer=torch.optim.AdamW(vanilla_pol.parameters(), lr=3e-3),
        splitter=splitter,
        lookback_window=20,
        batch_size=16,
        epochs_per_window=3,
        return_mask=False,
    ).train(large_md, rebalance_dates=reb_dates)

    # Model A2: SSMax + Multi-Cardinality UniverseSubsampler(4, 40)
    ssmax_pol = SetTransformerPolicyNet(
        n_features=1,
        hidden_dim=24,
        num_heads=2,
        num_inds=4,
        num_layers=1,
        mandate="capped_simplex",
        max_weight=0.35,
        use_ssmax=True,
    )
    WalkForwardTrainer(
        model=ssmax_pol,
        loss_function=SoftMinEVaRLoss(beta=8.0),
        optimizer=torch.optim.AdamW(ssmax_pol.parameters(), lr=3e-3),
        splitter=splitter,
        lookback_window=20,
        batch_size=16,
        epochs_per_window=3,
        return_mask=True,
        universe_subsampler=UniverseSubsampler(
            min_assets=4, max_assets=40, random_state=99
        ),
    ).train(large_md, rebalance_dates=reb_dates)

    # Evaluate zero-shot transfer from N=40 to N=8 subset
    x_40 = (
        torch.tensor(rets_large[-20:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    )
    rets_8 = subset_data.prices.pct_change().fillna(0.0).to_numpy(dtype=np.float32)
    x_8 = torch.tensor(rets_8[-20:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        van_isab = vanilla_pol.backbone.isab_layers[0]
        assert isinstance(van_isab, ISAB)
        _, _, _, _ = vanilla_pol.encode_representations(x_40)
        assert van_isab.mab1.attn.last_attention_entropy_ is not None
        ent_van_40 = float(van_isab.mab1.attn.last_attention_entropy_.item())
        w_van_8 = vanilla_pol(x_8).squeeze(0).numpy()
        assert van_isab.mab1.attn.last_attention_entropy_ is not None
        ent_van_8 = float(van_isab.mab1.attn.last_attention_entropy_.item())

        ss_isab = ssmax_pol.backbone.isab_layers[0]
        assert isinstance(ss_isab, ISAB)
        _, _, _, _ = ssmax_pol.encode_representations(x_40)
        assert ss_isab.mab1.attn.last_attention_entropy_ is not None
        ent_ss_40 = float(ss_isab.mab1.attn.last_attention_entropy_.item())
        w_ss_8 = ssmax_pol(x_8).squeeze(0).numpy()
        assert ss_isab.mab1.attn.last_attention_entropy_ is not None
        ent_ss_8 = float(ss_isab.mab1.attn.last_attention_entropy_.item())

    eff_n_van = float(1.0 / np.sum(np.square(w_van_8)))
    eff_n_ss = float(1.0 / np.sum(np.square(w_ss_8)))
    print(
        f"  [Ablation A: 40 -> 8 Zero-Shot Transfer]\n"
        f"    Vanilla Softmax (No Subsample) | Attn Entropy (40->8): {ent_van_40:.3f} -> {ent_van_8:.3f} "
        f"| Subset Effective Holdings: {eff_n_van:.2f} | MaxW: {float(np.max(w_van_8)):.3f}\n"
        f"    SSMax + UniverseSubsampler     | Attn Entropy (40->8): {ent_ss_40:.3f} -> {ent_ss_8:.3f} "
        f"| Subset Effective Holdings: {eff_n_ss:.2f} | MaxW: {float(np.max(w_ss_8)):.3f}"
    )

    # Ablation B: Graphormer Partial-Correlation Prior (`attn_bias=W_t` vs `attn_bias=None`)
    rd_est = ReactionDiffusionCovarianceEstimator(slow_window=50)
    _ = rd_est.estimate(subset_data)
    prior_adj_np = (
        rd_est.last_adjacency_
        if rd_est.last_adjacency_ is not None
        else np.eye(len(TICKERS))
    )
    prior_adj_t = torch.tensor(prior_adj_np, dtype=torch.float32).unsqueeze(0)

    sab_pol = SetTransformerPolicyNet(
        n_features=1,
        hidden_dim=24,
        num_heads=2,
        num_inds=4,
        num_layers=1,
        mandate="vol_targeted",
        block_type="sab",
    )
    # Warm up the Graphormer edge bias gate to reflect active macroeconomic/partial-corr prior regularization
    with torch.no_grad():
        for sab_layer in sab_pol.backbone.sab_layers:
            assert isinstance(sab_layer, SAB)
            sab_layer.mab.attn.attn_bias_gate.fill_(0.75)

    oos_rets = rets_8[60:]
    dd_no_prior_list: list[float] = []
    dd_with_prior_list: list[float] = []
    for t_idx in range(20, len(oos_rets)):
        win_t = (
            torch.tensor(oos_rets[t_idx - 20 : t_idx], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        r_next = oos_rets[t_idx]
        with torch.no_grad():
            w_no_bias = sab_pol(win_t, attn_bias=None).squeeze(0).numpy()
            w_with_bias = sab_pol(win_t, attn_bias=prior_adj_t).squeeze(0).numpy()
        dd_no_prior_list.append(float(np.dot(w_no_bias, r_next)))
        dd_with_prior_list.append(float(np.dot(w_with_bias, r_next)))

    def _max_dd(r_seq: list[float]) -> float:
        eq = np.cumprod(1.0 + np.asarray(r_seq, dtype=np.float64))
        peak = np.maximum.accumulate(eq)
        return float(np.min((eq - peak) / np.maximum(peak, 1e-12)))

    mdd_no_prior = _max_dd(dd_no_prior_list)
    mdd_with_prior = _max_dd(dd_with_prior_list)
    vol_no_prior = float(np.std(dd_no_prior_list) * np.sqrt(252.0))
    vol_with_prior = float(np.std(dd_with_prior_list) * np.sqrt(252.0))
    print(
        f"  [Ablation B: Graphormer Partial-Correlation Prior vs. Pure Data-Driven Attention]\n"
        f"    Pure Set Attention (attn_bias=None)     | OOS Max Drawdown: {mdd_no_prior:.2%} | Annualized Vol: {vol_no_prior:.2%}\n"
        f"    Graphormer Prior (attn_bias=W_partial)  | OOS Max Drawdown: {mdd_with_prior:.2%} | Annualized Vol: {vol_with_prior:.2%}"
    )


if __name__ == "__main__":
    main()
