from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from pyalloq_backtest.engine import WalkForwardEngine
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.enums import CovariateKey, CovariateMode, ObjectiveFunction
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_simulation import ConditionalDiffusionGenerator, NeuralSDEGenerator
from pyalloq_viz import plot_foundation_posterior_and_attribution

from pyalloq.classical.estimators.covariance import ImpliedCovarianceEstimator
from pyalloq.classical.optimizers import WassersteinDROAllocator
from pyalloq.dl import (
    ApolloPFNPretrainer,
    ApolloPFNReturnEstimator,
    ApolloPFNTransformer,
    ApolloPFNViewGenerator,
    EnsembleReturnEstimator,
    EnsembleViewGenerator,
    SRNGNPriorGenerator,
    TimesFM3MultivariateReturnEstimator,
    TimesFM3MultivariateViewGenerator,
    TimestralMultivariateReturnEstimator,
    TimestralMultivariateViewGenerator,
)
from pyalloq.rl import ExploratoryRegimeMVAllocator


class _OfflineTimesFM3Adapter:
    """Offline fallback evaluator used when the external `timesfm3` package is not installed."""

    def __init__(self, horizon: int = 15) -> None:
        self.horizon = horizon

    def predict_batch(
        self, contexts: list[np.ndarray], **kwargs: object
    ) -> list[SimpleNamespace]:
        ctx = contexts[0]
        n_assets = ctx.shape[0]
        last_vals = ctx[:, -1]
        mom = (ctx[:, -1] / np.maximum(ctx[:, -min(20, ctx.shape[1])], 1e-6)) - 1.0
        forecast = np.zeros((n_assets, self.horizon), dtype=np.float32)
        for i in range(n_assets):
            drift = np.clip(0.35 * mom[i], -0.04, 0.06)
            forecast[i] = np.linspace(
                last_vals[i] * (1.0 + drift / self.horizon),
                last_vals[i] * (1.0 + drift),
                self.horizon,
            )
        quantiles = np.zeros((n_assets, self.horizon, 9), dtype=np.float32)
        for q_idx, q_lvl in enumerate(np.linspace(0.1, 0.9, 9)):
            quantiles[:, :, q_idx] = forecast * (1.0 + (q_lvl - 0.5) * 0.12)
        return [SimpleNamespace(forecast=forecast, quantiles=quantiles)]


class _OfflineTimestralAdapter:
    """Offline fallback forecaster used when the external `timestral` package is not installed."""

    def forecast(
        self, context: np.ndarray, prediction_length: int = 15, **kwargs: object
    ) -> SimpleNamespace:
        bsz, n_var, t_len = context.shape
        last_vals = context[:, :, -1]
        mom = (last_vals / np.maximum(context[:, :, -min(15, t_len)], 1e-6)) - 1.0
        med = np.zeros((bsz, n_var, prediction_length), dtype=np.float32)
        for h in range(prediction_length):
            frac = float(h + 1) / float(prediction_length)
            med[:, :, h] = last_vals * (1.0 + np.clip(0.3 * mom, -0.04, 0.06) * frac)

        def _quantile(q: float) -> np.ndarray:
            return med * (1.0 + (float(q) - 0.5) * 0.14)

        return SimpleNamespace(median=med, quantile=_quantile)


def build_synthetic_macro_market_data() -> MarketData:
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2024-01-02", periods=140)
    future_dates = pd.bdate_range("2024-01-02", periods=160)
    assets = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]

    factor = rng.normal(0.0006, 0.011, size=(140, 1))
    betas = np.array([[1.15, 1.0, 1.45, 0.85, 0.70]])
    idio = rng.standard_t(df=5, size=(140, 5)) * 0.008
    rets = factor @ betas + idio
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=assets
    )

    iv_df = pd.DataFrame(
        0.20 + 0.06 * np.abs(rng.standard_normal(size=(140, 5))),
        index=dates,
        columns=assets,
    )
    cot_df = pd.DataFrame(
        {
            "COT_Net_Speculative": np.sin(np.linspace(0.0, 4.0 * np.pi, 140))
            + rng.normal(0, 0.2, 140),
            "COT_Dealer_Hedging": np.cos(np.linspace(0.0, 4.0 * np.pi, 140))
            + rng.normal(0, 0.2, 140),
        },
        index=dates,
    )
    rates_df = pd.DataFrame(
        {
            "US10Y_Yield": 4.10 + np.cumsum(rng.normal(0.0, 0.02, 140)),
            "Term_Spread_10Y2Y": -0.25 + np.linspace(0.0, 0.45, 140),
        },
        index=dates,
    )
    calendar_df = pd.DataFrame(
        {
            "FOMC_Decision": (np.arange(160) % 21 == 0).astype(float),
            "CPI_Print": (np.arange(160) % 15 == 0).astype(float),
        },
        index=future_dates,
    )

    return MarketData(
        prices=prices,
        features={
            CovariateKey.IMPLIED_VOL: iv_df,
            CovariateKey.COT_CROWDING: cot_df,
            CovariateKey.INTEREST_RATES: rates_df,
        },
        future_features={
            CovariateKey.CALENDAR_EVENTS: calendar_df,
        },
        risk_free_rate=0.04,
    )


def main() -> None:
    output_dir = Path("visualizations_output/apollopfn_wdro")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Building MarketData with Strongly-Typed CovariateKey Enums ===")
    data = build_synthetic_macro_market_data()
    past_cov, fut_cov, _ch_names = data.extract_covariates(
        mode=CovariateMode.AUTO, horizon=15, return_channel_names=True
    )
    assert past_cov is not None and fut_cov is not None
    print(
        f"Assets: {data.assets} | Past Covariate Channels: {past_cov.shape[0]} "
        f"| Future Covariate Channels: {fut_cov.shape[0]}"
    )

    print(
        "\n=== 2. Offline SRNGN Prior Meta-Training & Checkpoint Save (Table 6 SWA) ==="
    )
    ckpt_path = output_dir / "apollopfn_financial_prior.pt"
    prior_gen = SRNGNPriorGenerator(min_nodes=14, max_nodes=28, random_state=2026)
    net = ApolloPFNTransformer(
        d_model=64, n_heads=4, n_layers=3, ffn_dim=128, n_bins=96
    )
    pretrainer = ApolloPFNPretrainer(
        model=net,
        prior_generator=prior_gen,
        warmup_steps=4,
        swa_steps=4,
        random_state=2026,
    )
    pretrainer.pretrain_and_save(
        ckpt_path, n_steps=12, batch_size=4, context_len=40, horizon=15
    )
    print(f"Saved pretrained ApolloPFN checkpoint to: {ckpt_path}")

    print(
        "\n=== 3. Interchangeable ApolloPFN, TimesFM-3 & Timestral + Bayesian Ensemble ==="
    )
    horizon = 15
    shared_kwargs: dict[str, Any] = {
        "horizon": horizon,
        "annualize": True,
        "covariate_mode": CovariateMode.AUTO,
        "covariate_keys": [
            CovariateKey.COT_CROWDING,
            CovariateKey.INTEREST_RATES,
            CovariateKey.CALENDAR_EVENTS,
        ],
    }

    apfn_ret = ApolloPFNReturnEstimator(model=ckpt_path, **shared_kwargs)
    tfm_ret = TimesFM3MultivariateReturnEstimator(
        model=_OfflineTimesFM3Adapter(horizon=horizon), **shared_kwargs
    )
    tms_ret = TimestralMultivariateReturnEstimator(
        model=_OfflineTimestralAdapter(), **shared_kwargs
    )

    ensemble_ret = EnsembleReturnEstimator(
        estimators=[apfn_ret, tfm_ret, tms_ret],
        weighting_mode="scrps_adaptive",
        validation_horizon=10,
        model_names=["ApolloPFN", "TimesFM-3", "Timestral"],
    )
    mu_ens = ensemble_ret.estimate(data)
    print("Ensemble Expected Annualized Returns:\n", mu_ens.round(4))
    print("\nModel sCRPS Scores:\n", ensemble_ret.last_scrps_scores_.round(4))  # type: ignore[union-attr]

    print(
        "\n=== 4. Generative-WDRO + Foundation Ensemble View Pooling & Walk-Forward ==="
    )
    ensemble_vg = EnsembleViewGenerator(
        view_generators=[
            ApolloPFNViewGenerator(
                model=ckpt_path, horizon=horizon, full_covariance_omega=True
            ),
            TimesFM3MultivariateViewGenerator(
                model=_OfflineTimesFM3Adapter(horizon=horizon),
                horizon=horizon,
                full_covariance_omega=True,
            ),
            TimestralMultivariateViewGenerator(
                model=_OfflineTimestralAdapter(),
                horizon=horizon,
                full_covariance_omega=True,
            ),
        ],
        model_names=["ApolloPFN", "TimesFM-3", "Timestral"],
    )

    diff_gen = ConditionalDiffusionGenerator(
        n_factors=3, ddim_steps=12, guidance_scale=1.25, random_state=42
    )
    sde_gen = NeuralSDEGenerator(max_market_price_of_risk=1.5, random_state=42)
    sde_scen = sde_gen.estimate_scenarios(data, n_paths=60, horizon=10)
    print(
        f"NeuralSDE Scenarios: {sde_scen.returns.shape} | "
        f"Calendar Arbitrage Violation: {sde_gen.calendar_arbitrage_violation_:.6f}"
    )

    wdro_alloc = WassersteinDROAllocator(
        tickers=data.assets,
        objective=ObjectiveFunction.WASSERSTEIN_KELLY,
        epsilon=0.004,
        ground_metric="mahalanobis",
        radius_mode="regime_adaptive",
        scenario_generator=diff_gen,
        generator_n_paths=60,
        generator_horizon=10,
        view_generator=ensemble_vg,
    )
    imp_cov = ImpliedCovarianceEstimator(index_implied_vol=0.22)
    corr_imp = imp_cov.estimate_correlation(data)
    print(
        "Buss-Vilkov Implied Correlation Matrix (mean off-diag):",
        round(float(corr_imp.to_numpy().mean()), 4),
    )

    pipeline_wdro = StrategyPipeline(
        allocator=wdro_alloc,
        returns_estimator=ensemble_ret,
        cov_estimator=imp_cov,
    )
    wf = WalkForwardEngine(
        pipeline=pipeline_wdro,
        splitter=RollingWindowSplitter(lookback_window=60),
        rebalance_freq="ME",
    )
    bt_res = wf.run(data)
    final_eq = float(np.prod(1.0 + bt_res.returns.to_numpy(dtype=np.float64)))
    print(f"Generative-WDRO Walk-Forward Final Equity: {final_eq:.4f}")

    emv_alloc = ExploratoryRegimeMVAllocator(
        tickers=data.assets, n_regimes=2, entropy_temperature=0.04
    )
    emv_res = emv_alloc.allocate(data)
    print(
        f"Exploratory Regime-MV Weights: {emv_res.weights.round(3).to_dict()} "
        f"| Policy Entropy: {emv_alloc.last_policy_entropy_:.4f}"
    )

    fig = plot_foundation_posterior_and_attribution(
        horizon_quantiles=apfn_ret.last_horizon_quantiles_,
        covariate_importance=apfn_ret.last_covariate_importance_,
        ensemble_model_weights=ensemble_ret.last_model_weights_,
        asset="NVDA",
    )
    html_out = output_dir / "foundation_posterior_attribution.html"
    fig.write_html(str(html_out))
    print(
        f"Saved interactive Foundation Posterior & Attribution dashboard to: {html_out}"
    )


if __name__ == "__main__":
    main()
