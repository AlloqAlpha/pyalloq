from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pyalloq_backtest import (
    MetricsTearSheet,
    PerformanceAttributor,
    compute_bayesian_fdr,
    compute_location_scale_dsr,
    compute_minimum_track_record_length,
)
from pyalloq_core import (
    AdaHedgeStrategy,
    AdaNormalHedgeStrategy,
    BPDSMetaStrategy,
    CADTSBanditStrategy,
    CBCEStrategy,
    ConformalPIDCalibrator,
    ConformalRiskController,
    DualScaledSwitchingCostStrategy,
    EllipticalShrinkageStrategy,
    FixedShareStrategy,
    MarketData,
    RegimeState,
    compute_optimal_universe_size,
    estimate_elliptical_kurtosis_kappa,
)
from pyalloq_features import (
    ContinuousJumpModelDetector,
    FeaturePipeline,
    SparseJumpModelDetector,
)
from pyalloq_simulation import (
    FactorConditionalDiffusionEstimator,
)

from pyalloq.classical.estimators import (
    AssetSpecificJumpViewGenerator,
    BlackLittermanEstimator,
    LedoitWolfShrinkage,
    MeanHistoricalReturnEstimator,
    RelevanceConditionedCovarianceEstimator,
    RelevanceConditionedReturnEstimator,
)
from pyalloq.classical.optimizers import (
    ConformalKellyOverlay,
    ConformalSafetyGateAllocator,
    CPPSAllocator,
    EnsembleAllocator,
    EqualWeightAllocator,
    MarkowitzAllocator,
    MeanCVaRAllocator,
    MultiPeriodMPCAllocator,
    RegimeSwitchingAllocator,
    RiskCertificate,
    RiskParityAllocator,
    RollingGaussianHMMDetector,
    WassersteinDROAllocator,
)
from pyalloq.dl import (
    DeepAllocator,
    DeepCovarianceEstimator,
    EnsembleReturnEstimator,
    RelationalGraphPriorModule,
    SetTransformerCovarianceNet,
    SetTransformerPolicyNet,
)
from pyalloq.ml.optimizers.hrp import HRPAllocator
from pyalloq.rl import ExploratoryRegimeRLAllocator


def _make_sample_market_data(n_days: int = 120, n_assets: int = 5) -> MarketData:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    # Two distinct regimes: low-vol positive drift then high-vol negative drift
    half = n_days // 2
    r1 = rng.normal(0.0008, 0.008, size=(half, n_assets))
    r2 = rng.standard_t(df=4, size=(n_days - half, n_assets)) * 0.018 - 0.0010
    rets = np.vstack([r1, r2])
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=tickers
    )
    volume = pd.DataFrame(
        rng.uniform(1e7, 5e7, size=(n_days, n_assets)), index=dates, columns=tickers
    )
    cs = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Bonds", "Gold", "Energy"][:n_assets],
            "market_cap": np.linspace(1e11, 5e11, n_assets),
        },
        index=tickers,
    )
    return MarketData(
        prices=prices,
        features={"volume": volume},
        cross_sectional=cs,
        risk_free_rate=0.03,
    )


def test_phase1_jump_models_and_feature_pipeline() -> None:
    data = _make_sample_market_data(n_days=110, n_assets=4)

    sjm = SparseJumpModelDetector(
        n_regimes=3,
        jump_penalty=12.0,
        sparsity_kappa=2.0,
        tuning_criterion="ftic",
        n_regimes_grid=(2, 3),
        jump_penalty_grid=(5.0, 15.0),
    )
    state_sjm = sjm.detect(data)
    assert isinstance(state_sjm, RegimeState)
    assert state_sjm.probs.shape[0] == len(data.prices)
    assert np.allclose(state_sjm.probs.sum(axis=1).to_numpy(), 1.0, atol=1e-5)
    assert state_sjm.feature_weights is not None
    assert (
        float(np.linalg.norm(state_sjm.feature_weights.to_numpy(), ord=2)) <= 1.0 + 1e-5
    )

    cjm = ContinuousJumpModelDetector(
        n_regimes=3,
        jump_penalty=18.0,
        mode_loss_weight=0.35,
    )
    state_cjm = cjm.detect(data)
    assert np.allclose(state_cjm.probs.sum(axis=1).to_numpy(), 1.0, atol=1e-5)
    assert 0.0 <= state_cjm.current_confidence <= 1.0

    # Verify FeaturePipeline integration
    fp = FeaturePipeline().add(cjm)
    transformed = fp.run(data)
    assert "regime_probs" in transformed.features
    assert "regime_state" in transformed.features

    # AssetSpecificJumpViewGenerator -> BlackLittermanEstimator
    vg = AssetSpecificJumpViewGenerator(n_regimes=2, jump_penalty=10.0)
    P, Q, Omega = vg.generate(data)
    assert P.shape == (4, 4)
    assert len(Q) == 4
    assert Omega.shape == (4, 4)
    bl = BlackLittermanEstimator(view_generator=vg)
    mu_bl, cov_bl = bl.estimate_from_data(data)
    assert len(mu_bl) == 4
    assert cov_bl.shape == (4, 4)

    # RegimeSwitchingAllocator (CJM vs RollingGaussianHMMDetector)
    rsa_cjm = RegimeSwitchingAllocator(
        regime_allocators=[
            HRPAllocator(),
            RiskParityAllocator(),
            EqualWeightAllocator(),
        ],
        regime_detector=cjm,
        mode="probability_blend",
        bear_defensive_cash_weight=0.20,
    )
    res_rsa = rsa_cjm.allocate(data)
    assert len(res_rsa.weights) == 4
    assert float(res_rsa.weights.sum()) <= 1.0 + 1e-6

    rsa_hmm = RegimeSwitchingAllocator(
        regime_allocators=[EqualWeightAllocator(), RiskParityAllocator()],
        regime_detector=RollingGaussianHMMDetector(n_regimes=2),
        mode="partitioned_hmm",
    )
    res_hmm = rsa_hmm.allocate(data)
    assert len(res_hmm.weights) == 4


def test_phase1_relevance_conditioning_and_sharpe_inference2() -> None:
    data = _make_sample_market_data(n_days=100, n_assets=4)

    rel_cov_est = RelevanceConditionedCovarianceEstimator(
        subsample_fraction=0.30,
        use_ledoit_wolf=True,
        shrink_by_adjusted_fit=True,
    )
    cov_df = rel_cov_est.estimate(data)
    assert cov_df.shape == (4, 4)
    eigvals = np.linalg.eigvalsh(cov_df.to_numpy())
    assert np.all(eigvals > 0.0)
    assert 0.0 < rel_cov_est.last_adjusted_fit_ <= 1.0

    rel_ret_est = RelevanceConditionedReturnEstimator(subsample_fraction=0.30)
    mu_ser = rel_ret_est.estimate(data)
    assert len(mu_ser) == 4
    assert np.all(np.isfinite(mu_ser.to_numpy()))

    # Sharpe Inference 2.0: Location-Scale DSR, MinTRL, Bayesian FDR, Top-2 Gap, Factor Mirage
    rets = data.prices.pct_change().dropna().mean(axis=1)
    dsr_ls, dsr_lo, sr0 = compute_location_scale_dsr(
        rets, n_trials=10, risk_free_rate=0.03
    )
    assert 0.0 <= dsr_ls <= 1.0
    assert 0.0 <= dsr_lo <= 1.0
    assert sr0 >= 0.0

    min_trl = compute_minimum_track_record_length(
        rets + 0.002, risk_free_rate=0.03, target_sharpe=0.0
    )
    assert min_trl >= 2.0

    bfdr = compute_bayesian_fdr(dsr_ls, prior_true_prob=0.15)
    assert 0.0 <= bfdr <= 1.0

    ts = MetricsTearSheet.generate(rets, risk_free_rate=0.03, n_trials=5)
    for key in (
        "location_scale_dsr",
        "location_only_dsr",
        "min_track_record_length",
        "bayesian_fdr",
    ):
        assert key in ts.index

    attributor = PerformanceAttributor()
    mirage_audit = attributor.audit_factor_mirage(rets, data)
    assert "factor_mirage_detected" in mirage_audit
    assert "sign_flip_rates" in mirage_audit
    assert "collider_sign_reversals" in mirage_audit


def test_phase2_meta_strategies_bpds_and_ensemble_allocator() -> None:
    data = _make_sample_market_data(n_days=90, n_assets=5)
    ret_np = data.prices.pct_change().dropna().to_numpy(dtype=np.float64)

    kappa = estimate_elliptical_kurtosis_kappa(ret_np)
    assert kappa >= 0.0
    n_star = compute_optimal_universe_size(
        n_obs=len(ret_np), n_available_assets=10, kurtosis_kappa=kappa
    )
    assert 2 <= n_star <= 10

    rng = np.random.default_rng(7)
    expert_rets = rng.normal(0.0005, 0.01, size=(60, 4))
    prev_w = np.array([0.25, 0.25, 0.25, 0.25])

    for strat in (
        AdaHedgeStrategy(),
        FixedShareStrategy(switching_rate=0.10),
        CBCEStrategy(),
        AdaNormalHedgeStrategy(),
        DualScaledSwitchingCostStrategy(switching_cost_penalty=0.30),
        CADTSBanditStrategy(subset_size=3),
        BPDSMetaStrategy(),
        EllipticalShrinkageStrategy(),
    ):
        w_meta = strat.compute_meta_weights(
            expert_rets, current_meta_weights=prev_w, asset_returns=ret_np
        )
        assert len(w_meta) == 4
        assert np.all(w_meta >= -1e-9)
        assert np.isclose(np.sum(w_meta), 1.0, atol=1e-5)

    # EnsembleReturnEstimator with weighting_mode="bpds"
    ens_ret = EnsembleReturnEstimator(
        estimators=[
            MeanHistoricalReturnEstimator(),
            RelevanceConditionedReturnEstimator(),
        ],
        weighting_mode="bpds",
    )
    mu_bpds = ens_ret.estimate(data)
    assert len(mu_bpds) == 5
    assert ens_ret.last_model_weights_ is not None

    # EnsembleAllocator with pre-TCA internal crossing & cardinality cap
    ens_alloc = EnsembleAllocator(
        child_allocators=[
            HRPAllocator(),
            MeanCVaRAllocator(),
            MarkowitzAllocator(),
        ],
        meta_strategy="dual_scaled_olo",
        include_1n_anchor=True,
        optimal_cardinality_cap=True,
        max_gross_leverage=1.0,
    )
    curr_w = pd.Series(0.20, index=data.prices.columns)
    res_ens = ens_alloc.allocate(data, current_weights=curr_w)
    assert len(res_ens.weights) == 5
    assert np.isclose(float(res_ens.weights.sum()), 1.0, atol=1e-4)
    assert res_ens.metadata["internal_crossing_turnover_saved"] >= 0.0


def test_phase3_conformal_calibrator_and_safety_gates() -> None:
    data = _make_sample_market_data(n_days=90, n_assets=4)

    for method in ("pid", "dtaci", "nexcp"):
        cal = ConformalPIDCalibrator(
            base_estimator=MeanHistoricalReturnEstimator(),
            alpha=0.25,
            method=method,  # type: ignore[arg-type]
        )
        res_c = cal.calibrate(data)
        assert len(res_c.point_forecast) == 4
        assert np.all(res_c.upper_bound.to_numpy() > res_c.lower_bound.to_numpy())
        assert 0.0 <= res_c.empirical_coverage <= 1.0
        _P, _Q, Omega = cal.generate(data)
        assert Omega.shape == (4, 4)

    crc = ConformalRiskController(alpha=0.15, bound_b=1.0)
    scores = np.abs(np.random.default_rng(11).standard_t(df=4, size=80)) * 0.02
    lam_hat = crc.calibrate(scores)
    assert 0.0 <= lam_hat <= 1.0

    # CPPSAllocator (HR-LR two-stage screen)
    cpps = CPPSAllocator(alpha=0.20, n_candidates=60, top_m_defense=10)
    res_cpps = cpps.allocate(data)
    assert len(res_cpps.weights) == 4
    assert np.isclose(float(res_cpps.weights.sum()), 1.0, atol=1e-5)

    # ConformalKellyOverlay
    kelly = ConformalKellyOverlay(
        base_allocator=HRPAllocator(),
        alpha=0.25,
        kelly_fraction=0.60,
        allow_cash_shrinkage=True,
    )
    res_kelly = kelly.allocate(data)
    assert len(res_kelly.weights) == 4
    assert float(res_kelly.weights.sum()) <= 1.0 + 1e-5

    # ConformalSafetyGateAllocator (Gated-CSRC convex projection)
    cert = RiskCertificate(
        max_tail_loss=0.025,
        max_gross_leverage=0.85,
        max_turnover=0.25,
        max_annual_volatility=0.18,
        min_cash_buffer=0.15,
        max_asset_weight=0.35,
        sector_caps={"Tech": 0.40},
        restricted_assets=("ASSET_3",),
    )
    gate = ConformalSafetyGateAllocator(
        base_allocator=MarkowitzAllocator(),
        certificate=cert,
    )
    curr_w = pd.Series([0.25, 0.25, 0.25, 0.25], index=data.prices.columns)
    res_gate = gate.allocate(data, current_weights=curr_w)
    w_g = res_gate.weights
    assert float(w_g.abs().sum()) <= cert.max_gross_leverage + 1e-4
    assert float(w_g["ASSET_3"]) == 0.0
    assert float(w_g["ASSET_0"] + w_g["ASSET_1"]) <= 0.40 + 1e-3


def test_phase4_mpc_diffusion_dro_emvj_and_relational_graph() -> None:
    data = _make_sample_market_data(n_days=80, n_assets=4)
    cov_df = LedoitWolfShrinkage().estimate(data)
    mu_ser = MeanHistoricalReturnEstimator().estimate(data)

    # 4A. Reference-Regulated High-Dimensional Multi-Period MPC (CVXPY & ADMM)
    for backend in ("cvxpy", "admm"):
        mpc = MultiPeriodMPCAllocator(
            horizon=3,
            reference_allocator=HRPAllocator(),
            reference_penalty=2.5,
            backend=backend,
        )
        res_mpc = mpc.allocate(data, cov_matrix=cov_df, expected_returns=mu_ser)
        assert res_mpc.metadata["reference_regulated"] is True
        assert np.isclose(float(res_mpc.weights.sum()), 1.0, atol=1e-4)

    # 4B. Factor-Based Conditional Diffusion + Wasserstein DRO Quadratic Shift Floor
    diff_gen = FactorConditionalDiffusionEstimator(
        n_factors=4,
        factor_selection_mode="effective_rank",
        max_conditioning_dim=10,
        diffusion_steps=20,
        ddim_steps=5,
        random_state=42,
    )
    diff_gen.fit(data)
    assert 2 <= diff_gen.K <= 4
    scen = diff_gen.generate(n_paths=20, horizon=5)
    assert scen.returns.shape == (20, 5, 4)

    wdro = WassersteinDROAllocator(
        scenario_generator=diff_gen,
        generator_n_paths=20,
        generator_horizon=5,
        quadratic_shift_floor_coeff=0.60,
    )
    res_wdro = wdro.allocate(data, cov_matrix=cov_df)
    assert res_wdro.metadata["generative_wdro"] is True
    assert res_wdro.metadata["quadratic_shift_floor"] >= 0.0

    # 4C. Exploratory Mean-Variance with Jumps (EMVJ) & CJM Beliefs + Tsallis Entropy
    cjm = ContinuousJumpModelDetector(n_regimes=2, jump_penalty=15.0)
    emvj = ExploratoryRegimeRLAllocator(
        n_regimes=2,
        regime_detector=cjm,
        include_jump_risk=True,
        entropy_type="tsallis",
        tsallis_q=1.5,
    )
    res_emvj = emvj.allocate(data, cov_matrix=cov_df, expected_returns=mu_ser)
    assert res_emvj.metadata["entropy_type"] == "tsallis"
    assert res_emvj.metadata["include_jump_risk"] is True
    assert np.isclose(float(res_emvj.weights.sum()), 1.0, atol=1e-4)

    # 5B. RelationalGraphPriorModule -> SetTransformerCovarianceNet & SetTransformerPolicyNet
    torch.manual_seed(42)
    graph_mod = RelationalGraphPriorModule(hidden_dim=8, keep_ratio=0.50)
    st_cov = SetTransformerCovarianceNet(
        n_features=2, hidden_dim=16, num_heads=2, num_inds=4, num_layers=1
    )
    deep_cov = DeepCovarianceEstimator(
        model=st_cov, lookback_window=20, graph_prior=graph_mod
    )
    est_cov = deep_cov.estimate(data)
    assert est_cov.shape == (4, 4)
    assert deep_cov.last_attn_bias_ is not None
    assert deep_cov.last_attn_bias_.shape == (4, 4)

    st_pol = SetTransformerPolicyNet(
        n_features=2, hidden_dim=16, num_heads=2, num_inds=4, num_layers=1
    )
    deep_alloc = DeepAllocator(model=st_pol, lookback_window=20, graph_prior=graph_mod)
    res_dl = deep_alloc.allocate(data)
    assert len(res_dl.weights) == 4
    assert deep_alloc.last_attn_bias_ is not None
    assert deep_alloc.last_attn_bias_.shape == (4, 4)
