from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.enums import CovariateKey, CovariateMode, ObjectiveFunction
from pyalloq_viz import plot_foundation_posterior_and_attribution

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


class _MockTimesFM3Multivariate:
    def __init__(self, horizon: int = 10, spread_mult: float = 0.05) -> None:
        self.horizon = horizon
        self.spread_mult = spread_mult
        self.last_past_cov_shape: tuple[int, ...] | None = None
        self.last_fut_cov_shape: tuple[int, ...] | None = None

    def predict_batch(
        self, contexts: list[np.ndarray], **kwargs: object
    ) -> list[SimpleNamespace]:
        ctx = contexts[0]
        n_assets = ctx.shape[0]
        last_vals = ctx[:, -1]
        if kwargs.get("past_only_covariates"):
            self.last_past_cov_shape = kwargs["past_only_covariates"][0].shape  # type: ignore[index]
        if kwargs.get("past_future_covariates"):
            self.last_fut_cov_shape = kwargs["past_future_covariates"][0].shape  # type: ignore[index]

        forecast = np.zeros((n_assets, self.horizon), dtype=np.float32)
        for i in range(n_assets):
            forecast[i] = np.linspace(
                last_vals[i] * 1.005, last_vals[i] * (1.02 + 0.01 * i), self.horizon
            )
        quantiles = np.zeros((n_assets, self.horizon, 9), dtype=np.float32)
        for q_idx, q_lvl in enumerate(np.linspace(0.1, 0.9, 9)):
            shift = (q_lvl - 0.5) * 2.5 * self.spread_mult
            quantiles[:, :, q_idx] = forecast * (1.0 + shift)
        return [SimpleNamespace(forecast=forecast, quantiles=quantiles)]


class _MockTimestralForecast:
    def __init__(self, med: np.ndarray, spread_mult: float = 0.08) -> None:
        self.median = med
        self._spread = spread_mult

    def quantile(self, q: float) -> np.ndarray:
        shift = (float(q) - 0.5) * 2.5 * self._spread
        return self.median * (1.0 + shift)


class _MockTimestralForecaster:
    def __init__(self, spread_mult: float = 0.08) -> None:
        self.spread_mult = spread_mult
        self.last_past_cov_shape: tuple[int, ...] | None = None
        self.last_known_cov_shape: tuple[int, ...] | None = None

    def forecast(
        self, context: np.ndarray, prediction_length: int = 10, **kwargs: object
    ) -> _MockTimestralForecast:
        bsz, n_var, _ = context.shape
        last_vals = context[:, :, -1]
        if "past_covariates" in kwargs and kwargs["past_covariates"] is not None:
            self.last_past_cov_shape = kwargs["past_covariates"].shape  # type: ignore[attr-defined]
        if "known_covariates" in kwargs and kwargs["known_covariates"] is not None:
            self.last_known_cov_shape = kwargs["known_covariates"].shape  # type: ignore[attr-defined]

        med = np.zeros((bsz, n_var, prediction_length), dtype=np.float32)
        for h in range(prediction_length):
            med[:, :, h] = last_vals * (1.0 + 0.002 * (h + 1))
        return _MockTimestralForecast(med, spread_mult=self.spread_mult)


@pytest.fixture
def macro_market_data() -> MarketData:
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2024-01-02", periods=60)
    future_dates = pd.bdate_range("2024-01-02", periods=70)
    assets = ["AAPL", "MSFT", "NVDA", "JPM"]

    rets = rng.normal(0.0008, 0.014, size=(60, 4))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=assets
    )

    # Asset-level feature (volume) + non-ticker Macro/COT features using CovariateKey
    vol_df = pd.DataFrame(
        rng.uniform(1e6, 5e6, size=(60, 4)), index=dates, columns=assets
    )
    cot_df = pd.DataFrame(
        {
            "COT_Net_Spec": rng.normal(0.0, 1.0, size=60),
            "COT_Commercial": rng.normal(0.0, 1.0, size=60),
        },
        index=dates,
    )
    rates_df = pd.DataFrame(
        {"US10Y": np.linspace(4.0, 4.35, 60)},
        index=dates,
    )
    cal_df = pd.DataFrame(
        {
            "FOMC_Day": (np.arange(70) % 15 == 0).astype(float),
            "CPI_Release": (np.arange(70) % 20 == 0).astype(float),
        },
        index=future_dates,
    )
    return MarketData(
        prices=prices,
        features={
            CovariateKey.VOLUME: vol_df,
            CovariateKey.COT_CROWDING: cot_df,
            CovariateKey.INTEREST_RATES: rates_df,
        },
        future_features={
            CovariateKey.CALENDAR_EVENTS: cal_df,
        },
    )


def test_covariate_key_and_mode_extraction(macro_market_data: MarketData) -> None:
    # 1. AUTO extracts both asset-aligned (4 channels) and macro/COT (3 channels) = 7 past channels
    past_auto, fut_auto, names = macro_market_data.extract_covariates(
        mode=CovariateMode.AUTO, horizon=10, return_channel_names=True
    )
    assert past_auto is not None and past_auto.shape == (7, 60)
    assert fut_auto is not None and fut_auto.shape == (2, 70)
    assert len(names) == 9

    # 2. NONE returns (None, None)
    past_none, fut_none = macro_market_data.extract_covariates(
        mode=CovariateMode.NONE, horizon=10
    )
    assert past_none is None and fut_none is None

    # 3. MACRO_ONLY extracts only non-ticker macro/COT features (3 past channels, 2 future channels)
    past_macro, fut_macro = macro_market_data.extract_covariates(
        mode=CovariateMode.MACRO_ONLY, horizon=10
    )
    assert past_macro is not None and past_macro.shape == (3, 60)
    assert fut_macro is not None and fut_macro.shape == (2, 70)

    # 4. ASSET_ONLY extracts only asset-aligned volume (4 past channels, None future)
    past_asset, fut_asset = macro_market_data.extract_covariates(
        mode=CovariateMode.ASSET_ONLY, horizon=10
    )
    assert past_asset is not None and past_asset.shape == (4, 60)
    assert fut_asset is None

    # 5. SELECTED filters by specific CovariateKey
    past_sel, fut_sel = macro_market_data.extract_covariates(
        mode=CovariateMode.SELECTED,
        covariate_keys=[CovariateKey.COT_CROWDING],
        horizon=10,
    )
    assert past_sel is not None and past_sel.shape == (2, 60)
    assert fut_sel is None


def test_srngn_prior_and_apollopfn_pretrainer_checkpoint(tmp_path: Path) -> None:
    prior = SRNGNPriorGenerator(min_nodes=10, max_nodes=18, random_state=7)
    dag = prior.sample_srngn_dag(12, rho=0.4)
    # Node 0 is the unique root; every node >= 1 has at least 1 parent < n
    assert dag[0] == []
    for n in range(1, 12):
        assert len(dag[n]) >= 1
        assert all(0 <= p < n for p in dag[n])

    targets, covariates = prior.sample_scm_task(
        context_len=30, horizon=8, n_targets=3, n_covariates=4
    )
    assert targets.shape == (38, 3)
    assert covariates.shape == (38, 4)

    # Test offline pretraining with SWA checkpoint averaging & save/load
    ckpt_file = tmp_path / "apollopfn_test.pt"
    net = ApolloPFNTransformer(d_model=32, n_heads=2, n_layers=2, ffn_dim=64, n_bins=64)
    pretrainer = ApolloPFNPretrainer(
        model=net,
        prior_generator=prior,
        warmup_steps=2,
        swa_steps=2,
        random_state=11,
    )
    saved_path = pretrainer.pretrain_and_save(
        ckpt_file, n_steps=4, batch_size=2, context_len=24, horizon=6
    )
    assert saved_path.exists()
    assert len(pretrainer.loss_history_) == 4

    loaded_net = ApolloPFNTransformer.from_pretrained(saved_path)
    assert loaded_net.d_model == 32
    assert loaded_net.n_bins == 64

    # Verify RiemannBarDistribution metrics including Appendix B sCRPS
    bar = loaded_net.bar_dist
    dummy_logits = torch.zeros((1, 6, 3, 64))
    probs = bar.logits_to_probs(dummy_logits, temperature=0.8)
    assert np.allclose(bar.mean(probs), 0.0, atol=1e-4)
    assert np.all(bar.std(probs) > 0.0)
    q10 = bar.quantile(probs, 0.10)
    q90 = bar.quantile(probs, 0.90)
    assert np.all(q90 > q10)
    scrps_val = bar.s_crps(probs, np.ones((1, 6, 3)) * 0.25)
    assert scrps_val > 0.0


def test_interchangeable_apollopfn_timesfm_and_timestral(
    macro_market_data: MarketData,
) -> None:
    horizon = 10
    mock_tfm = _MockTimesFM3Multivariate(horizon=horizon, spread_mult=0.04)
    mock_tms = _MockTimestralForecaster(spread_mult=0.09)

    # 1. Identical kwargs across all 3 Return Estimators
    common_kwargs = {
        "horizon": horizon,
        "annualize": True,
        "covariate_mode": CovariateMode.AUTO,
        "covariate_keys": [
            CovariateKey.COT_CROWDING,
            CovariateKey.INTEREST_RATES,
            CovariateKey.CALENDAR_EVENTS,
        ],
    }
    apfn_ret = ApolloPFNReturnEstimator(model=None, **common_kwargs)
    tfm_ret = TimesFM3MultivariateReturnEstimator(model=mock_tfm, **common_kwargs)
    tms_ret = TimestralMultivariateReturnEstimator(model=mock_tms, **common_kwargs)

    for est in (apfn_ret, tfm_ret, tms_ret):
        mu = est.estimate(macro_market_data)
        assert isinstance(mu, pd.Series)
        assert list(mu.index) == macro_market_data.assets
        assert np.all(np.isfinite(mu.to_numpy(dtype=np.float64)))
        assert est.last_posterior_std_ is not None

    # Verify macro covariates were preserved (2 COT + 1 US10Y = 3 past channels; 2 future calendar channels)
    assert mock_tfm.last_past_cov_shape == (3, 60)
    assert mock_tfm.last_fut_cov_shape == (2, 70)
    assert mock_tms.last_past_cov_shape == (1, 3, 60)
    assert mock_tms.last_known_cov_shape == (1, 2, 70)

    # Verify ApolloPFN covariate importance & 3D scenario generation
    assert apfn_ret.last_covariate_importance_ is not None
    assert apfn_ret.last_covariate_importance_.shape == (4, 5)
    scen_3d = apfn_ret.estimate_scenarios(
        macro_market_data, n_paths=30, horizon=horizon, refit=False
    )
    assert isinstance(scen_3d, ScenarioMarketData)
    assert scen_3d.returns.shape == (30, horizon, 4)

    # 2. Identical kwargs across all 3 View Generators (with full_covariance_omega=True)
    vg_kwargs = {
        "horizon": horizon,
        "confidence_interval": (0.10, 0.90),
        "annualize": True,
        "full_covariance_omega": True,
        "covariate_mode": CovariateMode.AUTO,
    }
    apfn_vg = ApolloPFNViewGenerator(model=None, **vg_kwargs)
    tfm_vg = TimesFM3MultivariateViewGenerator(model=mock_tfm, **vg_kwargs)
    tms_vg = TimestralMultivariateViewGenerator(model=mock_tms, **vg_kwargs)

    for vg in (apfn_vg, tfm_vg, tms_vg):
        P, Q, Omega = vg.generate(macro_market_data)
        assert P.shape == (4, 4)
        assert len(Q) == 4
        assert Omega.shape == (4, 4)
        assert np.all(np.linalg.eigvalsh(Omega.to_numpy(dtype=np.float64)) > 0.0)


def test_ensemble_estimators_wdro_views_and_viz(
    macro_market_data: MarketData,
) -> None:
    horizon = 10
    # Give TimesFM tighter uncertainty (spread=0.03) than Timestral (spread=0.12)
    mock_tfm = _MockTimesFM3Multivariate(horizon=horizon, spread_mult=0.03)
    mock_tms = _MockTimestralForecaster(spread_mult=0.12)

    apfn_est = ApolloPFNReturnEstimator(horizon=horizon)
    tfm_est = TimesFM3MultivariateReturnEstimator(model=mock_tfm, horizon=horizon)
    tms_est = TimestralMultivariateReturnEstimator(model=mock_tms, horizon=horizon)

    ens_inv = EnsembleReturnEstimator(
        estimators=[apfn_est, tfm_est, tms_est],
        weighting_mode="inverse_uncertainty",
        model_names=["ApolloPFN", "TimesFM-3", "Timestral"],
    )
    mu_ens = ens_inv.estimate(macro_market_data)
    assert len(mu_ens) == 4
    assert ens_inv.last_model_weights_ is not None
    # TimesFM-3 (tighter quantile spread) must receive strictly higher weight than Timestral
    w_tfm = float(ens_inv.last_model_weights_.loc["TimesFM-3"].mean())
    w_tms = float(ens_inv.last_model_weights_.loc["Timestral"].mean())
    assert w_tfm > w_tms

    # Test sCRPS-adaptive ensemble weighting
    ens_scrps = EnsembleReturnEstimator(
        estimators=[apfn_est, tfm_est, tms_est],
        weighting_mode="scrps_adaptive",
        validation_horizon=6,
        model_names=["ApolloPFN", "TimesFM-3", "Timestral"],
    )
    mu_scrps = ens_scrps.estimate(macro_market_data)
    assert len(mu_scrps) == 4
    assert ens_scrps.last_scrps_scores_ is not None
    assert np.all(ens_scrps.last_scrps_scores_.to_numpy(dtype=np.float64) > 0.0)

    # Test EnsembleViewGenerator plugged directly into WassersteinDROAllocator
    ens_vg = EnsembleViewGenerator(
        view_generators=[
            ApolloPFNViewGenerator(horizon=horizon, full_covariance_omega=True),
            TimesFM3MultivariateViewGenerator(
                model=mock_tfm, horizon=horizon, full_covariance_omega=True
            ),
            TimestralMultivariateViewGenerator(
                model=mock_tms, horizon=horizon, full_covariance_omega=True
            ),
        ],
        model_names=["ApolloPFN", "TimesFM-3", "Timestral"],
    )
    wdro = WassersteinDROAllocator(
        objective=ObjectiveFunction.WASSERSTEIN_MEAN_RISK,
        epsilon=0.005,
        view_generator=ens_vg,
    )
    res_wdro = wdro.allocate(macro_market_data)
    assert np.isclose(float(res_wdro.weights.sum()), 1.0, atol=1e-5)
    assert res_wdro.metadata["active_view_multipliers"] is not None

    # Test Foundation Posterior & Attribution visualization
    fig = plot_foundation_posterior_and_attribution(
        horizon_quantiles=apfn_est.last_horizon_quantiles_,
        covariate_importance=apfn_est.last_covariate_importance_,
        ensemble_model_weights=ens_inv.last_model_weights_,
        asset="AAPL",
    )
    assert len(fig.data) >= 5
