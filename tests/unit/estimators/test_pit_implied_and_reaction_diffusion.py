from __future__ import annotations

import numpy as np
import pandas as pd
from pyalloq.classical.estimators.covariance import (
    ImpliedCovarianceEstimator,
    ReactionDiffusionCovarianceEstimator,
)
from pyalloq.classical.optimizers import (
    MarkowitzAllocator,
    StochasticScenarioMPCAllocator,
)
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import BaseScenarioGenerator
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_features.transformers.alternative import PITOptionsMFIVTransformer
from pyalloq_simulation import (
    EntropyPooler,
    ReactionDiffusionGenerator,
    ScenarioEvaluator,
)
from scipy.stats import norm


def _bs_otm_option_price(
    spot: float,
    strike: float,
    tau: float,
    vol: float,
    r: float = 0.0,
) -> tuple[float, str]:
    """Black-Scholes OTM option price helper for testing MFIV recovery."""
    d1 = (np.log(spot / strike) + (r + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    if strike <= spot * np.exp(r * tau):
        price = strike * np.exp(-r * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        return float(max(price, 1e-6)), "put"
    price = spot * norm.cdf(d1) - strike * np.exp(-r * tau) * norm.cdf(d2)
    return float(max(price, 1e-6)), "call"


class TestPITOptionsAndReactionDiffusion:
    def test_pit_mfiv_recovers_true_vol_and_blocks_future_quotes(
        self, synthetic_market_data: MarketData
    ) -> None:
        data = MarketData(
            prices=synthetic_market_data.prices.copy(),
            features={k: v.copy() for k, v in synthetic_market_data.features.items()},
        )
        target_date = data.prices.index[-1]
        ticker = data.prices.columns[0]
        spot = float(data.prices.loc[target_date, ticker])
        true_vol = 0.42
        tau = 30.0 / 365.25
        exp_date = target_date + pd.Timedelta(days=30)

        strikes = np.linspace(0.6 * spot, 1.4 * spot, 41)
        rows = []
        for k in strikes:
            mid, opt_type = _bs_otm_option_price(spot, float(k), tau, true_vol)
            # Valid PIT quote at 15:30 (before 16:00 market close)
            rows.append(
                {
                    "date": target_date,
                    "ticker": ticker,
                    "expiration": exp_date,
                    "strike": float(k),
                    "option_type": opt_type,
                    "bid": mid * 0.99,
                    "ask": mid * 1.01,
                    "quote_timestamp": target_date + pd.Timedelta(hours=15, minutes=30),
                }
            )
            # Contaminated future quote at 18:30 (after 16:00 close) with bogus 250% vol
            bad_mid, _ = _bs_otm_option_price(spot, float(k), tau, 2.50)
            rows.append(
                {
                    "date": target_date,
                    "ticker": ticker,
                    "expiration": exp_date,
                    "strike": float(k),
                    "option_type": opt_type,
                    "bid": bad_mid,
                    "ask": bad_mid,
                    "quote_timestamp": target_date + pd.Timedelta(hours=18, minutes=30),
                }
            )

        chain_df = pd.DataFrame(rows)
        tf = PITOptionsMFIVTransformer(option_chain_df=chain_df)
        out = tf.transform(data)

        assert "implied_vol" in out.features
        iv_df = out.features["implied_vol"]
        assert iv_df.shape == data.prices.shape
        assert not iv_df.isna().any().any()

        recovered_vol = float(iv_df.loc[target_date, ticker])
        assert abs(recovered_vol - true_vol) < 0.04

    def test_buss_vilkov_implied_covariance_matches_index_variance(
        self, synthetic_market_data: MarketData
    ) -> None:
        data = PITOptionsMFIVTransformer().transform(
            MarketData(
                prices=synthetic_market_data.prices.copy(),
                features={
                    k: v.copy() for k, v in synthetic_market_data.features.items()
                },
            )
        )
        tickers = list(data.prices.columns)
        n = len(tickers)
        asset_ivs = np.linspace(0.20, 0.40, n)
        data.features["implied_vol"].iloc[-1] = asset_ivs

        w_m = pd.Series(np.full(n, 1.0 / n), index=tickers)
        target_index_iv = 0.27

        estimator = ImpliedCovarianceEstimator(
            index_implied_vol=target_index_iv,
            market_weights=w_m,
        )
        cov = estimator.estimate(data)

        assert cov.shape == (n, n)
        eigvals = np.linalg.eigvalsh(cov.to_numpy())
        assert (eigvals > 0).all()

        np.testing.assert_allclose(
            np.sqrt(np.diag(cov.to_numpy())), asset_ivs, rtol=1e-6
        )

        port_var = float(w_m.to_numpy() @ cov.to_numpy() @ w_m.to_numpy())
        np.testing.assert_allclose(np.sqrt(port_var), target_index_iv, rtol=1e-4)

    def test_reaction_diffusion_covariance_propagates_network_contagion(self) -> None:
        rng = np.random.default_rng(123)
        dates = pd.date_range("2025-01-01", periods=120, freq="B")
        common_factor = rng.normal(0.0, 0.01, size=120)
        ret_a = common_factor + rng.normal(0.0, 0.003, size=120)
        ret_b = 0.9 * common_factor + rng.normal(0.0, 0.003, size=120)
        ret_c = rng.normal(0.0, 0.01, size=120)

        prices_df = pd.DataFrame(
            100.0 * np.cumprod(1.0 + np.column_stack([ret_a, ret_b, ret_c]), axis=0),
            index=dates,
            columns=["A", "B", "C"],
        )
        base_iv = pd.DataFrame(0.18, index=dates, columns=["A", "B", "C"])
        calm_data = MarketData(
            prices=prices_df.copy(), features={"implied_vol": base_iv.copy()}
        )

        shock_iv = base_iv.copy()
        shock_iv.iloc[-1, 0] = 0.85
        shock_data = MarketData(
            prices=prices_df.copy(), features={"implied_vol": shock_iv}
        )

        rd_est = ReactionDiffusionCovarianceEstimator(
            diffusion_coeff=0.45,
            horizon_steps=5,
        )
        cov_calm = rd_est.estimate(calm_data)
        cov_shock = rd_est.estimate(shock_data)

        eigvals = np.linalg.eigvalsh(cov_shock.to_numpy())
        assert (eigvals > 0).all()

        b_ratio = float(cov_shock.loc["B", "B"] / cov_calm.loc["B", "B"])
        c_ratio = float(cov_shock.loc["C", "C"] / cov_calm.loc["C", "C"])
        assert b_ratio > 1.10
        assert b_ratio > c_ratio * 1.08

    def test_reaction_diffusion_generator_3d_scenario_tensor_and_ecosystem(
        self, synthetic_market_data: MarketData
    ) -> None:
        data = PITOptionsMFIVTransformer().transform(
            MarketData(
                prices=synthetic_market_data.prices.copy(),
                features={
                    k: v.copy() for k, v in synthetic_market_data.features.items()
                },
            )
        )
        first_ticker = data.assets[0]
        gen = ReactionDiffusionGenerator(diffusion_coeff=0.35, leverage_rho=-0.5)
        assert isinstance(gen, BaseScenarioGenerator)
        gen.fit(data)

        scenarios = gen.generate(
            n_paths=16,
            horizon=15,
            initial_shock={first_ticker: 0.8},
        )
        assert isinstance(scenarios, ScenarioMarketData)
        assert scenarios.prices.shape == (16, 15, len(data.assets))
        assert scenarios.returns.shape == (16, 15, len(data.assets))
        assert scenarios.initial_prices is not None
        assert scenarios.initial_prices.shape == (len(data.assets),)
        assert "implied_vol" in scenarios.features
        assert scenarios.features["implied_vol"].shape == (16, 15, len(data.assets))

        # 1. Verify compatibility with EntropyPooler + ScenarioEvaluator
        pooler = EntropyPooler(n_scenarios=scenarios.n_paths)
        entropy_res = pooler.fit()
        evaluator = ScenarioEvaluator(ruin_threshold=-0.30)
        metrics = evaluator.evaluate(
            scenarios,
            asset_weights=np.full(len(data.assets), 1.0 / len(data.assets)),
            entropy_result=entropy_res,
        )
        assert np.isfinite(metrics.expected_return)
        assert np.isfinite(metrics.cvar_95)

        # 2. Verify compatibility with StochasticScenarioMPCAllocator
        mpc = StochasticScenarioMPCAllocator(tickers=data.assets, horizon=5)
        mpc_res = mpc.allocate(scenarios)
        assert np.isclose(mpc_res.weights.sum(), 1.0, atol=1e-4)

        # 3. Verify compatibility with StrategyPipeline via get_path(0)
        path_md = scenarios.get_path(0)
        pipeline = StrategyPipeline(
            cov_estimator=ReactionDiffusionCovarianceEstimator(),
            allocator=MarkowitzAllocator(tickers=data.assets),
        )
        weights = pipeline.generate_weights(path_md)
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)


class TestRoughSignaturesAndLiftedHeston:
    def test_log_signature_and_lifted_heston_rough_vol(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq_features.transformers import (
            LiftedHestonRoughVolTransformer,
            LogSignatureTransformer,
        )

        data = PITOptionsMFIVTransformer().transform(
            MarketData(
                prices=synthetic_market_data.prices.copy(),
                features={
                    k: v.copy() for k, v in synthetic_market_data.features.items()
                },
            )
        )
        sig_tf = LogSignatureTransformer(window=8, depth=2)
        data = sig_tf.transform(data)
        assert "logsig_levy_price_vol" in data.features
        assert "logsig_time_price_area" in data.features
        assert "logsig_norm" in data.features
        assert not data.features["logsig_levy_price_vol"].isna().any().any()

        rough_tf = LiftedHestonRoughVolTransformer(n_factors=4, estimation_window=20)
        data = rough_tf.transform(data)
        assert "hurst_exponent" in data.features
        assert "lifted_heston_vol" in data.features
        hurst_vals = data.features["hurst_exponent"].to_numpy()
        assert np.all((hurst_vals >= 0.05) & (hurst_vals <= 0.50))
        assert np.all(data.features["lifted_heston_vol"].to_numpy() > 0.0)


class TestSpectralHawkesAndLaplacianHRP:
    def test_spectral_fragility_and_wiener_hopf_hawkes(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq.classical.estimators.covariance import (
            SpectralGraphFragility,
            WienerHopfHawkesSolver,
        )

        rd_est = ReactionDiffusionCovarianceEstimator()
        cov_df = rd_est.estimate(synthetic_market_data)
        assert rd_est.last_diagnostics_ is not None
        assert rd_est.last_diagnostics_.fiedler_value >= 0.0
        assert 0.0 <= rd_est.last_diagnostics_.hawkes_spectral_radius < 1.0
        assert rd_est.last_diagnostics_.fiedler_vector.shape == (
            len(synthetic_market_data.assets),
        )
        assert cov_df.shape == (
            len(synthetic_market_data.assets),
            len(synthetic_market_data.assets),
        )

        # Direct solver verification
        returns_np = (
            synthetic_market_data.prices.pct_change()
            .dropna()
            .to_numpy(dtype=np.float64)
        )
        solver = WienerHopfHawkesSolver(max_lag=4, max_spectral_radius=0.85)
        branching, rho, baseline_mu = solver.fit_from_returns(returns_np)
        assert branching.shape == (returns_np.shape[1], returns_np.shape[1])
        assert baseline_mu.shape == (returns_np.shape[1],)
        assert np.all(branching >= 0.0)
        assert 0.0 <= rho <= 0.85 + 1e-6

        lap, _, _ = (
            ReactionDiffusionCovarianceEstimator.build_precision_graph_laplacian(
                returns_np
            )
        )
        lam2, tau_mix, v2 = SpectralGraphFragility.analyze_laplacian(lap)
        assert lam2 >= 0.0
        assert tau_mix > 0.0
        assert v2.shape == (returns_np.shape[1],)

    def test_spectral_laplacian_hrp_allocator(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq.ml.optimizers import HRPAllocator, SpectralLaplacianHRPAllocator

        rd_est = ReactionDiffusionCovarianceEstimator()
        cov_df = rd_est.estimate(synthetic_market_data)

        hrp_spec = SpectralLaplacianHRPAllocator(
            tickers=synthetic_market_data.assets,
            fragility_penalty=0.25,
        )
        res = hrp_spec.allocate(synthetic_market_data, cov_matrix=cov_df)
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
        assert (res.weights >= 0.0).all()
        assert res.metadata["clustering_mode"] == "spectral_laplacian"
        assert "fiedler_value" in res.metadata

        # Also verify via HRPAllocator(clustering_mode="spectral_laplacian", fragility_penalty=0.25)
        hrp_mode = HRPAllocator(
            tickers=synthetic_market_data.assets,
            clustering_mode="spectral_laplacian",
            fragility_penalty=0.25,
        )
        res2 = hrp_mode.allocate(synthetic_market_data, cov_matrix=cov_df)
        assert np.allclose(res.weights.to_numpy(), res2.weights.to_numpy(), atol=1e-6)


class TestDeepGraphAndKKTLayers:
    def test_crisp_sparse_stgat_and_deep_covariance_estimator(
        self, synthetic_market_data: MarketData
    ) -> None:
        import torch
        from pyalloq.dl import CRISPSparseSTGAT, DeepCovarianceEstimator

        n_assets = len(synthetic_market_data.assets)
        model = CRISPSparseSTGAT(
            num_assets=n_assets,
            num_features=1,
            hidden_dim=16,
            num_heads=4,
            keep_ratio=0.5,
        )
        x_dummy = torch.randn(2, 12, n_assets, 1, requires_grad=True)
        mu, cov, sparse_adj = model.predict_returns_and_cov(x_dummy)
        assert mu.shape == (2, n_assets)
        assert cov.shape == (2, n_assets, n_assets)
        assert sparse_adj.shape == (2, n_assets, n_assets)

        eigvals = torch.linalg.eigvalsh(cov)
        assert torch.all(eigvals > 0.0)

        # Verify DeepCovarianceEstimator integration
        OnlyPricesData = MarketData(prices=synthetic_market_data.prices.copy())
        deep_est = DeepCovarianceEstimator(model=model, lookback_window=15)
        est_df = deep_est.estimate(OnlyPricesData)
        assert est_df.shape == (n_assets, n_assets)
        assert np.all(np.linalg.eigvalsh(est_df.to_numpy()) > 0.0)

    def test_differentiable_risk_parity_and_cvar_kkt_layers(self) -> None:
        import torch
        from pyalloq.dl.layers import (
            DifferentiableCVaRLayer,
            DifferentiableRiskParityLayer,
        )

        bsz, n_assets = 2, 4
        raw = torch.randn(bsz, n_assets, n_assets, dtype=torch.float64)
        cov = (
            torch.bmm(raw, raw.transpose(1, 2))
            + 0.2 * torch.eye(n_assets, dtype=torch.float64).unsqueeze(0)
        ).requires_grad_(True)

        rp_layer = DifferentiableRiskParityLayer(max_iter=25)
        weights_rp = rp_layer(cov)
        assert weights_rp.shape == (bsz, n_assets)
        assert torch.allclose(
            weights_rp.sum(dim=-1),
            torch.ones(bsz, dtype=torch.float64),
            atol=1e-6,
        )
        assert torch.all(weights_rp > 0.0)

        # Verify Equal Risk Contribution condition: w_i * (Sigma w)_i / (w^T Sigma w) == 1/N
        sigma_w = torch.bmm(cov, weights_rp.unsqueeze(-1)).squeeze(-1)
        rc = weights_rp * sigma_w
        rc_share = rc / rc.sum(dim=-1, keepdim=True)
        target_share = torch.full_like(rc_share, 1.0 / n_assets)
        assert torch.allclose(rc_share, target_share, atol=1e-4)

        # Verify exact implicit KKT gradient flow
        loss_rp = (weights_rp[:, 0] ** 2).sum()
        loss_rp.backward()
        assert cov.grad is not None
        assert torch.all(torch.isfinite(cov.grad))
        assert torch.norm(cov.grad) > 0.0

        # Differentiable CVaR Layer
        scenarios = torch.randn(
            bsz, 30, n_assets, dtype=torch.float64, requires_grad=True
        )
        mu = torch.randn(bsz, n_assets, dtype=torch.float64, requires_grad=True)
        cvar_layer = DifferentiableCVaRLayer(beta=0.90, max_iter=20, step_size=0.1)
        weights_cvar = cvar_layer(mu, cov, scenarios=scenarios)
        assert weights_cvar.shape == (bsz, n_assets)
        assert torch.allclose(
            weights_cvar.sum(dim=-1),
            torch.ones(bsz, dtype=torch.float64),
            atol=1e-5,
        )
        loss_cvar = (weights_cvar[:, 0] - weights_cvar[:, 1]).sum()
        loss_cvar.backward()
        assert scenarios.grad is not None
        assert mu.grad is not None
        assert torch.all(torch.isfinite(scenarios.grad))


class TestEndogenousLiquidityImpactWrapper:
    def test_endogenous_impact_wrapper_on_portfolio_and_scenario_env(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq.rl import (
            EndogenousLiquidityImpactWrapper,
            PortfolioGymEnv,
            ScenarioPortfolioEnv,
        )

        base_env = PortfolioGymEnv(
            MarketData(prices=synthetic_market_data.prices.copy()),
            lookback_window=10,
        )
        wrapped_env = EndogenousLiquidityImpactWrapper(
            base_env,
            impact_lambda=0.01,
            resilience_rho=0.7,
            diffusion_kappa=0.005,
        )
        obs, info = wrapped_env.reset(seed=42)
        assert obs.shape == base_env.observation_space.shape
        assert "transient_impact_state" in info

        action = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = wrapped_env.step(action)
        assert next_obs.shape == base_env.observation_space.shape
        assert np.isfinite(reward)
        assert step_info["endogenous_impact_cost"] > 0.0
        assert step_info["graph_contagion_spillover"] >= 0.0
        assert np.any(step_info["transient_impact_state"] > 0.0)

        # Also test with ReactionDiffusionGenerator -> ScenarioPortfolioEnv
        gen = ReactionDiffusionGenerator(diffusion_coeff=0.25)
        gen.fit(MarketData(prices=synthetic_market_data.prices.copy()))
        scen = gen.generate(n_paths=4, horizon=12)
        scen_env = EndogenousLiquidityImpactWrapper(
            ScenarioPortfolioEnv(scen, lookback_window=4)
        )
        s_obs, _ = scen_env.reset(seed=7)
        s_next, s_rew, _, _, s_info = scen_env.step(action)
        assert s_obs.shape == s_next.shape
        assert np.isfinite(s_rew)
        assert "endogenous_impact_cost" in s_info


class TestAuditEdgeCasesAndGradcheck:
    def test_log_signature_reparameterization_invariance_and_depth3(self) -> None:
        from pyalloq_features.transformers import LogSignatureTransformer

        rng = np.random.default_rng(123)
        path = np.cumsum(rng.normal(0.0, 0.02, size=(6, 3)), axis=0)

        # Subdivide each linear segment into 2 half-steps (piecewise-linear refinement)
        refined_pts: list[np.ndarray] = []
        for k in range(len(path) - 1):
            refined_pts.append(path[k])
            refined_pts.append(0.5 * (path[k] + path[k + 1]))
        refined_pts.append(path[-1])
        path_refined = np.vstack(refined_pts)

        lsig_d2 = LogSignatureTransformer.compute_window_log_signature(path, depth=2)
        lsig_d2_ref = LogSignatureTransformer.compute_window_log_signature(
            path_refined, depth=2
        )
        # Depth-2 log-signature (increments + Lévy area) is strictly invariant to linear subdivision
        assert np.allclose(lsig_d2, lsig_d2_ref, atol=1e-12)

        lsig_d3 = LogSignatureTransformer.compute_window_log_signature(path, depth=3)
        assert lsig_d3.shape == (9,)
        assert np.all(np.isfinite(lsig_d3))

    def test_risk_parity_kkt_numerical_gradcheck(self) -> None:
        import torch
        from pyalloq.dl.layers import DifferentiableRiskParityLayer

        torch.manual_seed(99)
        n = 3
        base = torch.randn(1, n, n, dtype=torch.float64)
        cov = (
            torch.bmm(base, base.transpose(1, 2))
            + 0.35 * torch.eye(n, dtype=torch.float64).unsqueeze(0)
        ).requires_grad_(True)
        budgets = torch.tensor(
            [[0.5, 0.3, 0.2]], dtype=torch.float64, requires_grad=True
        )

        layer = DifferentiableRiskParityLayer(max_iter=30, tol=1e-12)

        def _func(c: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            c_sym = 0.5 * (c + c.transpose(-1, -2))
            return layer(c_sym, risk_budgets=b)

        assert torch.autograd.gradcheck(_func, (cov, budgets), eps=1e-6, atol=1e-4)

    def test_e2e_kkt_allocator_module_with_macro_feature_and_walkforward(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq.dl import CRISPSparseSTGAT, DifferentiableKKTAllocatorModule
        from pyalloq.dl.dataset import MarketDataset
        from pyalloq.dl.estimators.e2e.deep_allocator import DeepAllocator

        # Attach a 1-column macro feature ("index_implied_vol") alongside multi-asset features
        prices = synthetic_market_data.prices.iloc[:60].copy()
        idx_iv = pd.DataFrame(
            {"SPX": np.full(len(prices), 0.20)},
            index=prices.index,
        )
        md = MarketData(
            prices=prices,
            features={"index_implied_vol": idx_iv},
        )
        ds = MarketDataset(data=md, lookback_window=12, horizon=1)
        x_sample, _ = ds[0]
        assert x_sample.shape == (12, len(md.assets), 2)

        backbone = CRISPSparseSTGAT(
            num_assets=len(md.assets),
            num_features=2,
            hidden_dim=8,
            num_heads=2,
        )
        for mode in ("risk_parity", "cvar"):
            e2e_mod = DifferentiableKKTAllocatorModule(backbone=backbone, mode=mode)
            alloc = DeepAllocator(model=e2e_mod, lookback_window=12, tickers=md.assets)
            res = alloc.allocate(md)
            assert np.isclose(res.weights.sum(), 1.0, atol=1e-5)
            assert np.all(res.weights >= 0.0)

    def test_disconnected_block_diagonal_and_single_asset_edge_cases(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq import SpectralLaplacianHRPAllocator

        # 1. Disconnected block-diagonal covariance matrix (2 isolated clusters)
        tickers = ["A1", "A2", "B1", "B2"]
        block_cov = np.array(
            [
                [0.04, 0.03, 0.00, 0.00],
                [0.03, 0.04, 0.00, 0.00],
                [0.00, 0.00, 0.09, 0.07],
                [0.00, 0.00, 0.07, 0.09],
            ],
            dtype=np.float64,
        )
        cov_df = pd.DataFrame(block_cov, index=tickers, columns=tickers)
        spec_hrp = SpectralLaplacianHRPAllocator(tickers=tickers, fragility_penalty=0.3)
        res = spec_hrp.allocate(cov_matrix=cov_df)
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
        assert np.isclose(res.metadata["fiedler_value"], 0.0, atol=1e-6)
        # Higher-variance cluster B1/B2 should receive strictly smaller weight than A1/A2
        assert (res.weights["A1"] + res.weights["A2"]) > (
            res.weights["B1"] + res.weights["B2"]
        )

        # 2. Single-asset universe (N = 1) edge case
        single_prices = synthetic_market_data.prices.iloc[:25, :1].copy()
        single_md = MarketData(prices=single_prices)
        cov_1 = ReactionDiffusionCovarianceEstimator().estimate(single_md)
        assert cov_1.shape == (1, 1)
        assert float(cov_1.to_numpy(dtype=np.float64)[0, 0]) > 0.0
        imp_1 = ImpliedCovarianceEstimator().estimate(single_md)
        assert imp_1.shape == (1, 1)
        assert float(imp_1.to_numpy(dtype=np.float64)[0, 0]) > 0.0

    def test_property_scale_invariance_spectral_hrp_and_kkt_risk_parity(self) -> None:
        import torch
        from hypothesis import given, settings
        from hypothesis import strategies as st
        from pyalloq import SpectralLaplacianHRPAllocator
        from pyalloq.dl.layers import DifferentiableRiskParityLayer

        rp_layer = DifferentiableRiskParityLayer(max_iter=30, tol=1e-11)

        @settings(max_examples=12, deadline=None)
        @given(
            n_assets=st.integers(min_value=2, max_value=6),
            scale=st.floats(min_value=0.1, max_value=25.0),
            seed=st.integers(min_value=1, max_value=10_000),
        )
        def _check(n_assets: int, scale: float, seed: int) -> None:
            rng = np.random.default_rng(seed)
            a_mat = rng.normal(0.0, 1.0, size=(n_assets, n_assets))
            cov_np = a_mat @ a_mat.T + 0.25 * np.eye(n_assets)
            tickers = [f"T{i}" for i in range(n_assets)]
            cov_df = pd.DataFrame(cov_np, index=tickers, columns=tickers)
            cov_scaled_df = pd.DataFrame(scale * cov_np, index=tickers, columns=tickers)

            # 1. SpectralLaplacianHRPAllocator scale invariance
            hrp = SpectralLaplacianHRPAllocator(tickers=tickers, fragility_penalty=0.2)
            w1 = hrp.allocate(cov_matrix=cov_df).weights.to_numpy(dtype=np.float64)
            w2 = hrp.allocate(cov_matrix=cov_scaled_df).weights.to_numpy(
                dtype=np.float64
            )
            assert np.isclose(np.sum(w1), 1.0, atol=1e-6)
            assert np.all(w1 >= 0.0)
            assert np.allclose(w1, w2, atol=1e-6)

            # 2. DifferentiableRiskParityLayer scale invariance
            cov_t = torch.tensor(cov_np, dtype=torch.float64)
            w_rp1 = rp_layer(cov_t).detach().numpy()
            w_rp2 = rp_layer(scale * cov_t).detach().numpy()
            assert np.isclose(np.sum(w_rp1), 1.0, atol=1e-6)
            assert np.allclose(w_rp1, w_rp2, atol=1e-5)

        _check()

    def test_hawkes_self_exciting_cascade_recovery(self) -> None:
        from pyalloq.classical.estimators.covariance import WienerHopfHawkesSolver

        rng = np.random.default_rng(2026)
        n_steps, n_assets = 300, 3
        calm_returns = rng.normal(0.0, 0.01, size=(n_steps, n_assets))

        # Construct a self-exciting jump cascade where asset 0 jumps trigger lagged jumps in assets 1 & 2
        cascade_returns = calm_returns.copy()
        for t in range(10, n_steps - 3, 12):
            cascade_returns[t, 0] = -0.06
            cascade_returns[t + 1, 1] = -0.055
            cascade_returns[t + 2, 2] = -0.052

        solver = WienerHopfHawkesSolver(
            max_lag=4, jump_z_threshold=1.5, max_spectral_radius=0.95
        )
        _, rho_calm, _ = solver.fit_from_returns(calm_returns)
        branch_casc, rho_casc, _ = solver.fit_from_returns(cascade_returns)

        assert np.all(branch_casc >= 0.0)
        assert rho_casc > rho_calm
        # Cross-excitation from asset 0 -> asset 1 and asset 1 -> asset 2 should be strictly positive
        assert branch_casc[1, 0] > 0.0
        assert branch_casc[2, 1] > 0.0

    def test_cpcv_scrambling_on_differentiable_kkt_backtest(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq_backtest.costs import FlatBpsCostModel
        from pyalloq_backtest.engine import WalkForwardEngine
        from pyalloq_backtest.overfitting import CPCVSplitter
        from pyalloq_backtest.splitters import RollingWindowSplitter
        from pyalloq.dl import CRISPSparseSTGAT, DifferentiableKKTAllocatorModule
        from pyalloq.dl.estimators.e2e.deep_allocator import DeepAllocator

        md = MarketData(prices=synthetic_market_data.prices.iloc[:120].copy())
        backbone = CRISPSparseSTGAT(
            num_assets=len(md.assets),
            num_features=1,
            hidden_dim=8,
            num_heads=2,
        )
        kkt_mod = DifferentiableKKTAllocatorModule(
            backbone=backbone, mode="risk_parity"
        )
        alloc = DeepAllocator(model=kkt_mod, lookback_window=12, tickers=md.assets)
        pipe = StrategyPipeline(allocator=alloc)

        bt = WalkForwardEngine(
            pipeline=pipe,
            splitter=RollingWindowSplitter(lookback_window=40),
            cost_model=FlatBpsCostModel(bps=5.0),
            rebalance_freq="ME",
        ).run(md)

        cpcv = CPCVSplitter(n_groups=4, k_test_groups=2, max_paths=10, random_state=42)
        cpcv_res = cpcv.scramble_timelines(bt)
        assert cpcv_res.equity_curves.shape[1] > 0
        assert np.all(np.isfinite(cpcv_res.sharpe_ratios.to_numpy()))

    def test_bkm_higher_moments_cornish_fisher_and_vrp_estimator(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq import VarianceRiskPremiumEstimator

        dates = synthetic_market_data.prices.index
        target_date = dates[-1]
        ticker = synthetic_market_data.assets[0]
        spot = float(synthetic_market_data.prices.loc[target_date, ticker])
        exp_date = target_date + pd.Timedelta(days=30)
        strikes = np.linspace(0.80 * spot, 1.20 * spot, 15)

        # Left-skewed OTM option smile (puts more expensive than calls)
        rows = []
        for k in strikes:
            opt_type = "P" if k <= spot else "C"
            moneyness = (k - spot) / spot
            skew_bump = 1.65 if k < spot else 0.75
            mid_price = max(
                0.15,
                spot * 0.032 * skew_bump * float(np.exp(-5.0 * (moneyness**2))),
            )
            rows.append(
                {
                    "date": target_date,
                    "ticker": ticker,
                    "expiration": exp_date,
                    "strike": float(k),
                    "option_type": opt_type,
                    "bid": 0.95 * mid_price,
                    "ask": 1.05 * mid_price,
                    "quote_timestamp": target_date + pd.Timedelta(hours=15, minutes=45),
                }
            )
        chain_df = pd.DataFrame(rows)

        transformer = PITOptionsMFIVTransformer(option_chain_df=chain_df)
        md = MarketData(prices=synthetic_market_data.prices.copy())
        md = transformer.transform(md)

        assert "implied_vol" in md.features
        assert "implied_skew" in md.features
        assert "implied_kurtosis" in md.features
        asset_skew = float(md.features["implied_skew"].loc[target_date, ticker])
        asset_kurt = float(md.features["implied_kurtosis"].loc[target_date, ticker])
        assert np.isfinite(asset_skew)
        assert np.isfinite(asset_kurt)
        # Put-heavy smile produces negative risk-neutral skewness
        assert asset_skew < 0.0

        # Cornish-Fisher tail adjustment inflates diagonal variance for negatively skewed / fat-tailed assets
        cov_base = ImpliedCovarianceEstimator(use_cornish_fisher_tail=False).estimate(
            md
        )
        cov_cf = ImpliedCovarianceEstimator(
            use_cornish_fisher_tail=True, tail_confidence=0.95
        ).estimate(md)
        assert float(cov_cf.loc[ticker, ticker]) >= float(cov_base.loc[ticker, ticker])

        # Variance Risk Premium (VRP) return estimator produces finite expected returns
        vrp_est = VarianceRiskPremiumEstimator(realized_window=20, vrp_aversion=1.5)
        mu_vrp = vrp_est.estimate(md)
        assert len(mu_vrp) == len(md.assets)
        assert np.all(np.isfinite(mu_vrp.to_numpy(dtype=np.float64)))

    def test_signature_mmd_regime_gate_and_scenario_evaluator(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq.rl import SignatureMMDRegimeGate
        from pyalloq_simulation.core.evaluator import ScenarioEvaluator

        returns = synthetic_market_data.prices.pct_change().dropna()
        gate = SignatureMMDRegimeGate(
            subpath_window=6, fast_window=18, slow_window=50, mmd_threshold=0.04
        )
        diag_calm = gate.evaluate(returns)
        assert diag_calm.signature_mmd >= 0.0
        assert diag_calm.levy_area_divergence >= 0.0
        assert 0.0 <= diag_calm.damping_factor <= 1.0

        # Inject a sharp lead-lag / volatility regime shock in the last 18 bars
        shock_returns = returns.copy()
        shock_returns.iloc[-18:, 0] *= -3.5
        shock_returns.iloc[-18:, 1] = np.roll(
            shock_returns.iloc[-18:, 0].to_numpy() * 1.8, 1
        )
        diag_shock = gate.evaluate(shock_returns)
        assert diag_shock.signature_mmd >= 0.0
        assert diag_shock.levy_area_divergence > 0.0

        w_target = pd.Series(0.25, index=synthetic_market_data.assets)
        w_curr = pd.Series([0.7, 0.1, 0.1, 0.1], index=synthetic_market_data.assets)
        w_gated, _ = gate.gate_weights(w_target, w_curr, shock_returns)
        assert np.isclose(float(w_gated.sum()), 1.0, atol=1e-6)

        # ScenarioEvaluator.compute_signature_mmd
        gen = ReactionDiffusionGenerator(random_state=7)
        gen.fit(synthetic_market_data)
        scen_data = gen.generate(n_paths=25, horizon=15)
        hist_arr = returns.to_numpy(dtype=np.float64)
        mmd_val = ScenarioEvaluator.compute_signature_mmd(hist_arr, scen_data, window=8)
        assert np.isfinite(mmd_val)
        assert mmd_val >= 0.0

    def test_spectral_contagion_and_fragility_visualizations(
        self, synthetic_market_data: MarketData
    ) -> None:
        from pyalloq_viz import (
            plot_fragility_hawkes_regime_portrait,
            plot_spectral_contagion_network,
        )

        rd_est = ReactionDiffusionCovarianceEstimator(slow_window=40)
        _ = rd_est.estimate(synthetic_market_data)
        assert rd_est.last_adjacency_ is not None

        adj_df = pd.DataFrame(
            rd_est.last_adjacency_,
            index=synthetic_market_data.assets,
            columns=synthetic_market_data.assets,
        )
        branch_mat = rd_est.hawkes_branching_matrix_

        fig_net = plot_spectral_contagion_network(
            adjacency=adj_df,
            branching_matrix=branch_mat,
        )
        assert len(fig_net.data) >= 2

        diag_ts = pd.DataFrame(
            {
                "lambda_2": np.linspace(0.65, 0.18, 30),
                "hawkes_spectral_radius": np.linspace(0.22, 0.86, 30),
                "effective_hurst": np.linspace(0.45, 0.14, 30),
            },
            index=synthetic_market_data.prices.index[-30:],
        )
        fig_portrait = plot_fragility_hawkes_regime_portrait(diag_ts)
        assert len(fig_portrait.data) == 3
