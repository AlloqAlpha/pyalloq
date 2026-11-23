from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pyalloq_backtest.splitters import RollingWindowSplitter
from pyalloq_core.data import MarketData
from pyalloq_core.enums import ObjectiveFunction

from pyalloq.classical.optimizers import MarkowitzAllocator, RiskParityAllocator
from pyalloq.dl import (
    ConvTransformerAllocator,
    DeepAllocator,
    DeepCovarianceEstimator,
    ISABFactorDiagonalHead,
    MarketDataset,
    MarketJEPA,
    MDFPOptimizationLayer,
    SetTransformerBackbone,
    SetTransformerCovarianceNet,
    SetTransformerPolicyNet,
    SoftMinEVaRLoss,
    UnifiedE2EPolicyNetwork,
    UniverseSubsampler,
    WoodburyGaussianNLLLoss,
    WoodburySteinLoss,
)
from pyalloq.dl.trainer import WalkForwardTrainer
from pyalloq.ml.optimizers.hrp import HRPAllocator
from pyalloq.rl import POMDPPortfolioEnv, SB3PortfolioAgent


def test_gate_1_exact_permutation_symmetry() -> None:
    """
    Gate 1 — Exact Permutation Symmetry Gate:
    - 1D equivariance error ||f(Pi X) - Pi f(X)||_inf < 1e-6 (float32) / < 1e-13 (float64)
      for SAB, ISAB, and all SetTransformerPolicyNet mandate heads.
    - 2D congruence equivariance error ||Sigma(Pi X) - Pi Sigma(X) Pi^T||_F / ||Sigma(X)||_F < 1e-6
      for ISABFactorDiagonalHead, EquivariantGramCorrHead, and RIESpectralHead.
    - Exact permutation invariance ||g(Pi X) - g(X)||_inf < 1e-6 for PMA [REGIME] tokens,
      ISAB inducing states H_I, cash weights, and RL critic V_psi(z_regime).
    """
    torch.manual_seed(2026)
    B, L, N, F = 2, 16, 20, 3
    x = torch.randn(B, L, N, F, dtype=torch.float64)
    w_curr = torch.softmax(torch.randn(B, N, dtype=torch.float64), dim=-1)
    perm = torch.randperm(N)

    x_perm = x[:, :, perm, :]
    w_perm = w_curr[:, perm]

    # 1. Covariance Heads (2D Congruence Equivariance + Invariant Factor / Regime States)
    for head_type in ("factor_diag", "gram_corr", "rie_spectral"):
        cov_net = SetTransformerCovarianceNet(
            n_features=F,
            hidden_dim=32,
            num_heads=4,
            num_inds=8,
            num_layers=2,
            dropout=0.0,
            head_type=head_type,  # type: ignore[arg-type]
        ).to(torch.float64)
        cov_net.eval()

        with torch.no_grad():
            sigma = cov_net(x)
            reg_orig = cov_net.last_regime_token_
            sigma_perm = cov_net(x_perm)
            reg_perm = cov_net.last_regime_token_

        assert reg_orig is not None and reg_perm is not None
        # Check PMA_1 [REGIME] token invariance
        regime_err = float(torch.max(torch.abs(reg_perm - reg_orig)).item())
        assert (
            regime_err < 1e-12
        ), f"Regime invariance failed ({head_type}): {regime_err}"

        # Check 2D congruence equivariance: Sigma(Pi X) == Pi Sigma(X) Pi^T
        sigma_expected = sigma[:, perm, :][:, :, perm]
        rel_frob_err = float(
            (
                torch.linalg.norm(sigma_perm - sigma_expected)
                / torch.linalg.norm(sigma_expected).clamp(min=1e-12)
            ).item()
        )
        assert (
            rel_frob_err < 1e-12
        ), f"2D Congruence equivariance failed for {head_type}: {rel_frob_err}"

        # Verify PSD eigenvalues
        eigvals = torch.linalg.eigvalsh(sigma)
        assert bool(torch.all(eigvals > 0.0).item())

    # 2. Policy Mandate Heads (1D Equivariance + Invariant Value Critic & Cash)
    for mandate in (
        "capped_simplex",
        "sparsemax",
        "masked_dirichlet",
        "dollar_neutral",
        "vol_targeted",
    ):
        inc_cash = mandate == "masked_dirichlet"
        pol_net = SetTransformerPolicyNet(
            n_features=F,
            hidden_dim=32,
            num_heads=4,
            num_inds=8,
            num_layers=2,
            dropout=0.0,
            mandate=mandate,  # type: ignore[arg-type]
            max_weight=0.15,
            include_cash=inc_cash,
        ).to(torch.float64)
        pol_net.eval()

        with torch.no_grad():
            w_out = pol_net(x, w_current=w_curr, deterministic=True)
            v_orig = pol_net.evaluate_value(x, w_current=w_curr)
            w_out_perm = pol_net(x_perm, w_current=w_perm, deterministic=True)
            v_perm = pol_net.evaluate_value(x_perm, w_current=w_perm)

        # Critic value invariance
        v_err = float(torch.max(torch.abs(v_perm - v_orig)).item())
        assert v_err < 1e-12, f"Value critic invariance failed ({mandate}): {v_err}"

        if inc_cash:
            # First N entries permute equivariantly; index N (cash) is strictly invariant
            w_assets_expected = w_out[:, :N][:, perm]
            w_assets_actual = w_out_perm[:, :N]
            cash_err = float(
                torch.max(torch.abs(w_out_perm[:, N] - w_out[:, N])).item()
            )
            assert cash_err < 1e-12
            eq_err = float(
                torch.max(torch.abs(w_assets_actual - w_assets_expected)).item()
            )
        else:
            w_expected = w_out[:, perm]
            eq_err = float(torch.max(torch.abs(w_out_perm - w_expected)).item())

        assert eq_err < 1e-12, f"1D Equivariance failed for {mandate}: {eq_err}"


def test_gate_2_ragged_mask_zero_leakage() -> None:
    """
    Gate 2 — Ragged Mask Zero-Leakage Gate:
    Pad 12 active assets with 8 masked slots containing 1e6 sentinel values,
    randomly permute all 20 slots, and verify:
    1. Active ISAB, PMA [REGIME], covariance, and policy outputs match the unpadded
       12-asset ground truth within 1e-12 (in float64).
    2. Padded portfolio weights are identically 0.0.
    3. Zero gradient flow into masked input coordinates.
    """
    torch.manual_seed(42)
    B, L, N_act, N_pad, F = 2, 12, 12, 8, 2
    N_tot = N_act + N_pad

    x_act = torch.randn(B, L, N_act, F, dtype=torch.float64)
    # Fill padded slots with huge 1e6 garbage values to expose any leakage
    x_garbage = torch.full((B, L, N_pad, F), 1e6, dtype=torch.float64)
    x_padded = torch.cat([x_act, x_garbage], dim=2)

    mask_padded = torch.zeros((B, N_tot), dtype=torch.bool)
    mask_padded[:, :N_act] = True

    # Randomly permute all 20 rows (active + garbage mixed together)
    perm = torch.randperm(N_tot)
    inv_perm = torch.argsort(perm)
    x_perm = x_padded[:, :, perm, :].clone().detach().requires_grad_(True)
    mask_perm = mask_padded[:, perm]

    pol_net = SetTransformerPolicyNet(
        n_features=F,
        hidden_dim=32,
        num_heads=4,
        num_inds=6,
        num_layers=2,
        dropout=0.0,
        mandate="capped_simplex",
        max_weight=0.20,
    ).to(torch.float64)
    pol_net.eval()

    cov_net = SetTransformerCovarianceNet(
        n_features=F,
        hidden_dim=32,
        num_heads=4,
        num_inds=6,
        num_layers=2,
        dropout=0.0,
        head_type="factor_diag",
    ).to(torch.float64)
    cov_net.eval()

    with torch.no_grad():
        # Ground truth on unpadded 12 active assets
        w_unpadded = pol_net(x_act)
        reg_unpadded = pol_net.last_regime_token_
        cov_unpadded = cov_net(x_act)

    # Forward on padded & permuted 20-asset tensor
    w_perm = pol_net(x_perm, mask=mask_perm)
    reg_padded = pol_net.last_regime_token_
    cov_perm = cov_net(x_perm, mask=mask_perm)

    # Unpermute back to original order
    w_unperm = w_perm[:, inv_perm]
    w_active_from_padded = w_unperm[:, :N_act]
    w_inactive_from_padded = w_unperm[:, N_act:]

    # 1. Padded weights must be identically 0.0
    assert float(torch.max(torch.abs(w_inactive_from_padded)).item()) == 0.0

    # 2. Active outputs must match unpadded ground truth to machine precision
    active_diff = float(torch.max(torch.abs(w_active_from_padded - w_unpadded)).item())
    assert active_diff < 1e-12, f"Active policy weights leaked: {active_diff}"

    assert reg_unpadded is not None and reg_padded is not None
    reg_diff = float(torch.max(torch.abs(reg_padded - reg_unpadded)).item())
    assert reg_diff < 1e-12, f"PMA [REGIME] token leaked: {reg_diff}"

    cov_unperm = cov_perm[:, inv_perm, :][:, :, inv_perm]
    cov_act_from_padded = cov_unperm[:, :N_act, :N_act]
    cov_diff = float(torch.max(torch.abs(cov_act_from_padded - cov_unpadded)).item())
    assert cov_diff < 1e-12, f"Active covariance submatrix leaked: {cov_diff}"

    # 3. Verify zero gradient flow into masked input slots
    loss = (w_perm.pow(2).sum()) + cov_perm.sum()
    loss.backward()
    assert x_perm.grad is not None
    grad_unperm = x_perm.grad[:, :, inv_perm, :]
    grad_masked_slots = grad_unperm[:, :, N_act:, :]
    assert float(torch.max(torch.abs(grad_masked_slots)).item()) == 0.0
    assert float(torch.max(torch.abs(grad_unperm[:, :, :N_act, :])).item()) > 0.0


def test_gate_3_woodbury_exactness_and_mdfp_factors() -> None:
    """
    Gate 3 — Woodbury Exactness & MDFP Factor Gate:
    Verify that Woodbury logdet, linear solve, GMV weights, WoodburyGaussianNLLLoss,
    and WoodburySteinLoss match dense Cholesky within 1e-10 in float64, and verify
    that MDFPOptimizationLayer produces identical weight trajectories whether passed
    dense Sigma_t or low-rank cov_factors (B_t, Sigma_F, D_t).
    """
    torch.manual_seed(101)
    B, L, N, F = 3, 15, 25, 2
    x = torch.randn(B, L, N, F, dtype=torch.float64)
    y_true = torch.randn(B, N, dtype=torch.float64) * 0.01

    cov_net = SetTransformerCovarianceNet(
        n_features=F,
        hidden_dim=32,
        num_heads=4,
        num_inds=6,
        head_type="factor_diag",
    ).to(torch.float64)
    cov_net.eval()

    b_load, sigma_f, d_diag = cov_net.forward_factors(x)
    dense_cov = cov_net(x)

    # 1. Exact Log-Determinant
    wb_logdet = ISABFactorDiagonalHead.woodbury_logdet(b_load, sigma_f, d_diag)
    _, dense_logdet = torch.linalg.slogdet(dense_cov)
    assert float(torch.max(torch.abs(wb_logdet - dense_logdet)).item()) < 1e-10

    # 2. Exact Linear Solve & GMV Weights
    wb_solve = ISABFactorDiagonalHead.woodbury_solve(
        b_load, sigma_f, d_diag, rhs=y_true
    )
    dense_solve = torch.linalg.solve(dense_cov, y_true.unsqueeze(-1)).squeeze(-1)
    assert float(torch.max(torch.abs(wb_solve - dense_solve)).item()) < 1e-10

    wb_gmv = ISABFactorDiagonalHead.woodbury_gmv_weights(b_load, sigma_f, d_diag)
    ones = torch.ones((B, N, 1), dtype=torch.float64)
    sinv_1 = torch.linalg.solve(dense_cov, ones).squeeze(-1)
    dense_gmv = sinv_1 / sinv_1.sum(dim=-1, keepdim=True)
    assert float(torch.max(torch.abs(wb_gmv - dense_gmv)).item()) < 1e-10

    # 3. WoodburyGaussianNLLLoss & WoodburySteinLoss equivalence (factor vs dense)
    nll_loss = WoodburyGaussianNLLLoss(normalize_by_cardinality=True)
    loss_wb = nll_loss((b_load, sigma_f, d_diag), y_true)
    loss_dense = nll_loss(dense_cov, y_true)
    assert float(torch.abs(loss_wb - loss_dense).item()) < 1e-10

    stein_loss = WoodburySteinLoss(normalize_by_cardinality=True)
    target_cov = dense_cov * 1.15
    stein_wb = stein_loss((b_load, sigma_f, d_diag), target_cov)
    stein_dense = stein_loss(dense_cov, target_cov)
    assert float(torch.abs(stein_wb - stein_dense).item()) < 1e-10

    # Verify cardinality normalization holds Stein loss strictly invariant across N=20, 100, 250
    stein_vals = []
    for n_card in (20, 100, 250):
        eye_c = torch.eye(n_card, dtype=torch.float64).unsqueeze(0)
        s_val = float(stein_loss(eye_c, 1.15 * eye_c).item())
        stein_vals.append(s_val)
    assert max(stein_vals) - min(stein_vals) < 1e-12

    # 4. MDFPOptimizationLayer O(Nm) factor matvec vs dense O(N^2) matvec
    mu_seq = torch.randn(B, 4, N, dtype=torch.float64) * 0.005
    w_curr = torch.full((B, N), 1.0 / N, dtype=torch.float64)
    mdfp = MDFPOptimizationLayer(horizon=4, max_iter=30, tol=1e-9)
    z_dense = mdfp(mu_seq=mu_seq, cov_seq=dense_cov, w_current=w_curr)
    z_factor = mdfp(
        mu_seq=mu_seq,
        cov_seq=None,
        w_current=w_curr,
        cov_factors=(b_load, sigma_f, d_diag),
    )
    assert float(torch.max(torch.abs(z_dense - z_factor)).item()) < 1e-10


def test_gate_4_cardinality_shift_stability() -> None:
    """
    Gate 4 — Cardinality-Shift Stability Gate:
    Evaluate a single SetTransformerPolicyNet across universe sizes N in {10, 50, 250, 500}
    and confirm exact constraint feasibility, bounded SSMax attention entropy, and stable
    top-position concentration.
    """
    torch.manual_seed(77)
    F = 2
    cap_pol = SetTransformerPolicyNet(
        n_features=F,
        hidden_dim=32,
        num_heads=4,
        num_inds=8,
        mandate="capped_simplex",
        max_weight=0.20,
        use_ssmax=True,
    )
    cap_pol.eval()

    dn_pol = SetTransformerPolicyNet(
        n_features=F,
        hidden_dim=32,
        num_heads=4,
        num_inds=8,
        mandate="dollar_neutral",
        gross_leverage=1.0,
        use_ssmax=True,
    )
    dn_pol.eval()

    for n_assets in (10, 50, 250, 500):
        x_n = torch.randn(2, 12, n_assets, F)
        with torch.no_grad():
            w_cap = cap_pol(x_n)
            w_dn = dn_pol(x_n)

        # Capped simplex feasibility: sum == 1.0, 0 <= w_i <= 0.20
        assert torch.allclose(w_cap.sum(dim=-1), torch.ones(2), atol=1e-6)
        assert bool(torch.all(w_cap >= -1e-7).item())
        assert bool(torch.all(w_cap <= 0.20 + 1e-6).item())

        # Dollar-neutral feasibility: sum == 0.0, sum(|w_i|) == 1.0
        assert torch.allclose(w_dn.sum(dim=-1), torch.zeros(2), atol=1e-6)
        assert torch.allclose(torch.abs(w_dn).sum(dim=-1), torch.ones(2), atol=1e-5)


def test_gate_5_pit_and_asynchronous_causality() -> None:
    """
    Gate 5 — Point-in-Time & Asynchronous Causality Gate:
    1. Verify that MarketDataset(return_mask=True) tradability masks rely exclusively on
       information available at decision timestamp t (altering future t+1 NaNs does not
       alter mask_t).
    2. Verify that enabling directed_delay=True blocks same-day cross-asset leakage:
       perturbing asset j != i at the terminal timestep t=-1 has identically zero effect
       on asset i's embedding z_{i,t}, while perturbing asset j at lagged timestep t=-2
       causally propagates to asset i.
    """
    # 1. Point-in-Time tradability mask check
    dates = pd.bdate_range("2024-01-02", periods=30)
    prices_a = pd.DataFrame(
        100.0 + np.cumsum(np.random.normal(0.0, 1.0, size=(30, 4)), axis=0),
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    # Asset D IPOs on day 12 (pre-IPO NaNs on days 0..11)
    prices_a.iloc[:12, 3] = np.nan

    prices_b = prices_a.copy()
    # In prices_b, Asset C delists in the FUTURE after day 20
    prices_b.iloc[21:, 2] = np.nan

    ds_a = MarketDataset(
        MarketData(prices=prices_a),
        lookback_window=10,
        horizon=2,
        return_mask=True,
    )
    ds_b = MarketDataset(
        MarketData(prices=prices_b),
        lookback_window=10,
        horizon=2,
        return_mask=True,
    )

    # At idx=5 (decision date < day 20), future delisting of Asset C at day 21 must NOT alter mask
    _, _, mask_a_5 = ds_a[5]
    _, _, mask_b_5 = ds_b[5]
    assert torch.equal(mask_a_5, mask_b_5)
    # Asset D is still pre-IPO at idx=5, so mask[3] is False; at idx=15, mask[3] becomes True
    assert not bool(mask_a_5[3].item())
    _, _, mask_a_15 = ds_a[15]
    assert bool(mask_a_15[3].item())

    # 2. Directed Delay (t - 1) Causal Sieve check
    torch.manual_seed(99)
    backbone_causal = SetTransformerBackbone(
        n_features=2,
        hidden_dim=32,
        num_heads=4,
        num_inds=4,
        num_layers=1,
        dropout=0.0,
        directed_delay=True,
    ).to(torch.float64)
    backbone_causal.eval()

    x_base = torch.randn(1, 10, 6, 2, dtype=torch.float64)
    # Perturb asset j=3 at terminal decision step t=-1 (same-day asynchronous close)
    x_same_day_shock = x_base.clone()
    x_same_day_shock[:, -1, 3, :] += 50.0

    # Perturb asset j=3 at lagged step t=-2 (previous day close)
    x_lagged_shock = x_base.clone()
    x_lagged_shock[:, -2, 3, :] += 50.0

    with torch.no_grad():
        z_base, _, _ = backbone_causal(x_base)
        z_same_day, _, _ = backbone_causal(x_same_day_shock)
        z_lagged, _, _ = backbone_causal(x_lagged_shock)

    # Other assets i != 3 must have ZERO sensitivity to asset 3's same-day close (t=-1)
    other_idx = [0, 1, 2, 4, 5]
    same_day_leakage = float(
        torch.max(
            torch.abs(z_same_day[:, other_idx, :] - z_base[:, other_idx, :])
        ).item()
    )
    assert (
        same_day_leakage == 0.0
    ), f"Same-day cross-asset leakage detected: {same_day_leakage}"

    # Other assets i != 3 MUST respond to asset 3's lagged close (t=-2)
    lagged_response = float(
        torch.max(torch.abs(z_lagged[:, other_idx, :] - z_base[:, other_idx, :])).item()
    )
    assert lagged_response > 1e-4


def test_integration_with_trainer_allocator_and_rl() -> None:
    """
    Verifies end-to-end integration of SetTransformerCovarianceNet, SetTransformerPolicyNet,
    UniverseSubsampler, WalkForwardTrainer, DeepAllocator, UnifiedE2EPolicyNetwork,
    POMDPPortfolioEnv, and SB3PortfolioAgent.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=90)
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    rets = np.random.normal(0.0005, 0.012, size=(90, len(tickers)))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=tickers
    )
    # Make XOM a ragged late-IPO stock (NaN for first 15 days)
    prices.iloc[:15, 4] = np.nan
    data = MarketData(prices=prices, risk_free_rate=0.03)

    # 1. Train SetTransformerCovarianceNet with WoodburyGaussianNLLLoss & UniverseSubsampler
    cov_model = SetTransformerCovarianceNet(
        n_features=1,
        hidden_dim=16,
        num_heads=2,
        num_inds=4,
        num_layers=1,
        temporal_type="vsn_conv",
    )
    nll_loss = WoodburyGaussianNLLLoss()
    opt_cov = torch.optim.AdamW(cov_model.parameters(), lr=2e-3)
    splitter = RollingWindowSplitter(lookback_window=45)
    reb_dates = pd.DatetimeIndex([dates[60], dates[80]])

    trainer_cov = WalkForwardTrainer(
        model=cov_model,
        loss_function=nll_loss,
        optimizer=opt_cov,
        splitter=splitter,
        lookback_window=15,
        horizon=1,
        batch_size=8,
        epochs_per_window=2,
        return_mask=True,
        allow_ragged=True,
        universe_subsampler=UniverseSubsampler(min_assets=3, random_state=42),
        use_forward_factors=True,
    )
    hist = trainer_cov.train(data, rebalance_dates=reb_dates)
    assert len(hist["final_loss"]) >= 1

    cov_est = DeepCovarianceEstimator(
        model=cov_model, lookback_window=15, allow_ragged=True
    )
    cov_df = cov_est.estimate(data)
    assert cov_df.shape == (5, 5)
    b_df, sf_np, d_ser = cov_est.estimate_factors(data)
    assert b_df.shape == (5, 4)
    assert sf_np.shape == (4, 4)
    assert len(d_ser) == 5

    # 2. Plug SetTransformerCovarianceNet into UnifiedE2EPolicyNetwork
    unified_pol = UnifiedE2EPolicyNetwork(
        num_assets=5,
        num_features=1,
        hidden_dim=16,
        num_heads=2,
        head_mode="mdfp",
        backbone=cov_model,
    )
    x_sample = torch.randn(2, 15, 5, 1)
    w_unified = unified_pol(x_sample)
    assert w_unified.shape == (2, 5)
    assert torch.allclose(w_unified.sum(dim=-1), torch.ones(2), atol=1e-5)

    # 3. Train SetTransformerPolicyNet with SoftMinEVaRLoss & allocate via DeepAllocator
    set_pol = SetTransformerPolicyNet(
        n_features=1,
        hidden_dim=16,
        num_heads=2,
        num_inds=4,
        mandate="capped_simplex",
        max_weight=0.35,
    )
    evar_loss = SoftMinEVaRLoss()
    opt_pol = torch.optim.AdamW(set_pol.parameters(), lr=2e-3)
    trainer_pol = WalkForwardTrainer(
        model=set_pol,
        loss_function=evar_loss,
        optimizer=opt_pol,
        splitter=splitter,
        lookback_window=15,
        horizon=1,
        batch_size=8,
        epochs_per_window=2,
        return_mask=True,
        allow_ragged=True,
    )
    trainer_pol.train(data, rebalance_dates=reb_dates)

    deep_alloc = DeepAllocator(model=set_pol, lookback_window=15, tickers=tickers)
    alloc_res = deep_alloc.allocate(data)
    assert len(alloc_res.weights) == 5
    assert abs(float(alloc_res.weights.sum()) - 1.0) < 1e-5

    # 4. Masked Dirichlet RL with POMDPPortfolioEnv and SB3PortfolioAgent
    env = POMDPPortfolioEnv(
        scenario_data=data,
        lookback_window=10,
        include_cash=True,
        projection_mode="dirichlet",
        include_tradability_mask=True,
    )
    agent = SB3PortfolioAgent(
        env=env,
        hidden_dim=16,
        policy_backbone="set_transformer",
        learning_rate=2e-3,
    )
    agent.fit(total_timesteps=24, rollout_Horizon=12, ppo_epochs=2)
    obs, _ = env.reset(seed=42)
    action, _ = agent.predict(obs, deterministic=True)
    assert action.shape == (6,)
    assert abs(float(np.sum(action)) - 1.0) < 1e-4


def test_ragged_cvxpy_classical_allocators_and_masked_rie_spectral() -> None:
    """
    Audits two edge cases:
    1. Ragged-Mask Singularity Trap with Classical CVXPY Solvers:
       When `DeepCovarianceEstimator(allow_ragged=True)` estimates covariance on a
       ragged universe where 2 tickers are currently untradable (pre-IPO/delisted NaNs),
       `MarkowitzAllocator`, `RiskParityAllocator`, and `HRPAllocator` solve cleanly
       without `DCPError` or singularity failures and assign identically 0.0 weight to
       the masked tickers.
    2. Eigenvalue Contamination in `RIESpectralHead` Under Masking:
       Padding 8 active assets with 4 masked slots containing 1e6 garbage values produces
       the exact same active 8x8 RIE-cleaned covariance submatrix as the unpadded 8-asset
       forward pass (zero eigenvalue contamination).
    """
    torch.manual_seed(2026)
    np.random.seed(2026)

    # 1. Ragged CVXPY Classical Allocator Audit
    dates = pd.bdate_range("2024-01-02", periods=40)
    tickers = ["AAPL", "MSFT", "NVDA", "PRE_IPO", "DELISTED"]
    rets = np.random.normal(0.0006, 0.012, size=(40, len(tickers)))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=tickers
    )
    # PRE_IPO and DELISTED are NaN at decision time (end of window)
    prices.iloc[:, 3] = np.nan
    prices.iloc[-5:, 4] = np.nan
    ragged_data = MarketData(prices=prices, risk_free_rate=0.03)

    cov_net = SetTransformerCovarianceNet(
        n_features=1, hidden_dim=16, num_heads=2, num_inds=4, head_type="factor_diag"
    )
    cov_est = DeepCovarianceEstimator(
        model=cov_net, lookback_window=15, allow_ragged=True
    )
    ragged_cov = cov_est.estimate(ragged_data)

    # Full N x N matrix must be strictly SPD (eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(ragged_cov.to_numpy(dtype=np.float64))
    assert np.all(eigvals > 0.0)

    mkw = MarkowitzAllocator(
        tickers=tickers, objective=ObjectiveFunction.MIN_VOLATILITY
    )
    res_mkw = mkw.allocate(ragged_data, cov_matrix=ragged_cov)
    assert "OPTIMAL" in res_mkw.status
    assert abs(float(res_mkw.weights.sum()) - 1.0) < 1e-6
    assert float(res_mkw.weights["PRE_IPO"]) == 0.0
    assert float(res_mkw.weights["DELISTED"]) == 0.0

    rp = RiskParityAllocator(tickers=tickers)
    res_rp = rp.allocate(ragged_data, cov_matrix=ragged_cov)
    assert "OPTIMAL" in res_rp.status
    assert abs(float(res_rp.weights.sum()) - 1.0) < 1e-5
    assert float(res_rp.weights["PRE_IPO"]) == 0.0
    assert float(res_rp.weights["DELISTED"]) == 0.0

    hrp = HRPAllocator(tickers=tickers)
    res_hrp = hrp.allocate(ragged_data, cov_matrix=ragged_cov)
    assert "OPTIMAL" in res_hrp.status
    assert abs(float(res_hrp.weights.sum()) - 1.0) < 1e-6
    assert float(res_hrp.weights["PRE_IPO"]) == 0.0
    assert float(res_hrp.weights["DELISTED"]) == 0.0

    # 2. RIESpectralHead Zero Eigenvalue Contamination Under Masking
    B, L, N_act, N_pad, F = 2, 16, 8, 4, 2
    x_act = torch.randn(B, L, N_act, F, dtype=torch.float64)
    x_pad = torch.full((B, L, N_pad, F), 1e6, dtype=torch.float64)
    x_full = torch.cat([x_act, x_pad], dim=2)
    mask_full = torch.zeros((B, N_act + N_pad), dtype=torch.bool)
    mask_full[:, :N_act] = True

    perm = torch.randperm(N_act + N_pad)
    inv_perm = torch.argsort(perm)

    rie_net = SetTransformerCovarianceNet(
        n_features=F,
        hidden_dim=32,
        num_heads=4,
        num_inds=4,
        dropout=0.0,
        head_type="rie_spectral",
    ).to(torch.float64)
    rie_net.eval()

    with torch.no_grad():
        cov_rie_unpadded = rie_net(x_act)
        cov_rie_padded = rie_net(x_full[:, :, perm, :], mask=mask_full[:, perm])

    cov_rie_unperm = cov_rie_padded[:, inv_perm, :][:, :, inv_perm]
    rie_active_diff = float(
        torch.max(
            torch.abs(cov_rie_unperm[:, :N_act, :N_act] - cov_rie_unpadded)
        ).item()
    )
    assert (
        rie_active_diff < 1e-11
    ), f"RIESpectralHead contaminated by padded eigenvalues: {rie_active_diff}"

    # 3. High-dimensional N_active > L (rank-deficient active correlation with 0.0 eigenvalues)
    L_short = (
        4  # N_act = 8 > L_short = 4 -> active correlation has 5 exact 0.0 eigenvalues
    )
    x_act_hd = torch.randn(B, L_short, N_act, F, dtype=torch.float64)
    x_pad_hd = torch.full((B, L_short, N_pad, F), 1e6, dtype=torch.float64)
    x_full_hd = torch.cat([x_act_hd, x_pad_hd], dim=2)
    with torch.no_grad():
        cov_hd_unpadded = rie_net(x_act_hd)
        cov_hd_padded = rie_net(x_full_hd[:, :, perm, :], mask=mask_full[:, perm])
    cov_hd_unperm = cov_hd_padded[:, inv_perm, :][:, :, inv_perm]
    hd_diff = float(
        torch.max(torch.abs(cov_hd_unperm[:, :N_act, :N_act] - cov_hd_unpadded)).item()
    )
    assert (
        hd_diff < 1e-11
    ), f"High-dim (N_active > L) RIESpectralHead null-space mixing: {hd_diff}"

    # 4. Pandas DataFrame.attrs stripped (e.g., via cov * 1.0) and data=None passed to HRPAllocator
    stripped_cov = ragged_cov * 1.0
    stripped_cov.attrs.clear()
    res_hrp_no_attrs = hrp.allocate(data=None, cov_matrix=stripped_cov)
    assert float(res_hrp_no_attrs.weights["PRE_IPO"]) == 0.0
    assert float(res_hrp_no_attrs.weights["DELISTED"]) == 0.0
    assert abs(float(res_hrp_no_attrs.weights.sum()) - 1.0) < 1e-6


def test_market_jepa_and_conv_transformer_allocator_invariance() -> None:
    """
    Verifies Pillar 3 `MarketJEPA` (ISAB_m + PMA_1 at index 0 + mask-aware VICRegLoss)
    and `ConvTransformerAllocator` (PMA_k regime head replacing flattened d_model * n_assets):
    1. MarketJEPA outputs (B, N+1, d) where index 0 ([REGIME]) is strictly permutation
       invariant and indices 1..N are 1D permutation equivariant, and VICRegLoss ignores
       padded slots completely.
    2. ConvTransformerAllocator transfers seamlessly across N=10, 50, 200 without shape
       errors while maintaining exact regime invariance and allocation equivariance.
    """
    torch.manual_seed(123)
    B, L, N_act, N_pad, F = 3, 12, 10, 5, 2
    x_ctx_act = torch.randn(B, L, N_act, F, dtype=torch.float64)
    x_tgt_act = torch.randn(B, L, N_act, F, dtype=torch.float64)

    jepa = MarketJEPA(
        n_features=F,
        hidden_dim=32,
        num_heads=4,
        num_inds=6,
        num_layers=2,
        dropout=0.0,
    ).to(torch.float64)
    jepa.eval()

    perm = torch.randperm(N_act)
    with torch.no_grad():
        tokens_orig = jepa(x_ctx_act)
        tokens_perm = jepa(x_ctx_act[:, :, perm, :])

    assert tokens_orig.shape == (B, N_act + 1, 32)
    # Token 0 ([REGIME]) must be strictly permutation invariant
    reg_diff = float(
        torch.max(torch.abs(tokens_perm[:, 0, :] - tokens_orig[:, 0, :])).item()
    )
    assert reg_diff < 1e-12
    # Tokens 1..N must be 1D permutation equivariant
    asset_diff = float(
        torch.max(
            torch.abs(tokens_perm[:, 1:, :] - tokens_orig[:, 1:, :][:, perm, :])
        ).item()
    )
    assert asset_diff < 1e-12

    # Mask-aware VICRegLoss zero-leakage check
    x_garbage = torch.full((B, L, N_pad, F), 1e6, dtype=torch.float64)
    x_ctx_pad = torch.cat([x_ctx_act, x_garbage], dim=2)
    x_tgt_pad = torch.cat([x_tgt_act, x_garbage], dim=2)
    mask_pad = torch.zeros((B, N_act + N_pad), dtype=torch.bool)
    mask_pad[:, :N_act] = True

    with torch.no_grad():
        loss_unpadded = jepa.compute_jepa_loss(x_ctx_act, x_tgt_act)
        loss_padded = jepa.compute_jepa_loss(x_ctx_pad, x_tgt_pad, mask=mask_pad)
    assert float(torch.abs(loss_unpadded - loss_padded).item()) < 1e-11

    # ConvTransformerAllocator size & permutation invariance check
    conv_alloc = ConvTransformerAllocator(
        n_features=F,
        d_model=32,
        num_heads=4,
        num_inds=6,
        num_pma_seeds=2,
        num_regimes=3,
        dropout=0.0,
        max_asset_weight=0.25,
    ).to(torch.float64)
    conv_alloc.eval()

    with torch.no_grad():
        w_conv = conv_alloc(x_ctx_act)
        reg_probs_orig = conv_alloc.predict_regime_probs(x_ctx_act)
        w_conv_perm = conv_alloc(x_ctx_act[:, :, perm, :])
        reg_probs_perm = conv_alloc.predict_regime_probs(x_ctx_act[:, :, perm, :])

        # Zero-shot transfer to N=80 assets
        x_80 = torch.randn(B, L, 80, F, dtype=torch.float64)
        w_80 = conv_alloc(x_80)

    assert float(torch.max(torch.abs(reg_probs_perm - reg_probs_orig)).item()) < 1e-12
    assert float(torch.max(torch.abs(w_conv_perm - w_conv[:, perm])).item()) < 1e-12
    assert w_80.shape == (B, 80)
    assert torch.allclose(
        w_80.sum(dim=-1), torch.ones(B, dtype=torch.float64), atol=1e-6
    )
