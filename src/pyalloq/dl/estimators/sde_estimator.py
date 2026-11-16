from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import (
    BaseReturnEstimator,
    BaseScenarioEstimator,
)
from pyalloq_simulation.generators.neural_sde import NeuralSDEGenerator
from torch import nn

from pyalloq.dl.layers.neural_jump_rde import TorchLogSignatureExtractor


class ArbitrageFreeNeuralSDENet(nn.Module):
    """
    Differentiable Arbitrage-Free Coupled Jump-Diffusion Neural SDE Module.

    Parameterizes continuous-time drift `mu_theta`, lower-triangular Cholesky diffusion
    `L_theta`, and stochastic log-volatility dynamics `V_t` with:
    1. **Hansen-Jagannathan Market-Price-of-Risk Projection**:
       Smoothly bounds `theta_t = L_t^{-1} (mu_t - r_f 1)` so `||theta_t||_2 <= max_daily_sr`.
    2. **Strictly PSD Cholesky Diffusion Head**:
       Constructs instantaneous covariance `Sigma_t = L_t L_t^T + eps I`.
    """

    def __init__(
        self,
        n_assets: int,
        hidden_dim: int = 32,
        max_daily_sr: float = 0.12,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.hidden_dim = int(hidden_dim)
        self.max_daily_sr = float(max_daily_sr)
        self.eps = float(eps)

        self.state_encoder = nn.Sequential(
            nn.Linear(2 * self.n_assets, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.drift_head = nn.Linear(self.hidden_dim, self.n_assets)
        self.vol_drift_head = nn.Linear(self.hidden_dim, self.n_assets)
        self.chol_scale_head = nn.Linear(self.hidden_dim, self.n_assets)

        # Initialize near zero so initial dynamics stay anchored to physical scale
        nn.init.zeros_(self.drift_head.bias)
        nn.init.normal_(self.drift_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.vol_drift_head.bias)
        nn.init.normal_(self.vol_drift_head.weight, mean=0.0, std=0.02)

    def forward_step(
        self,
        x_state: torch.Tensor,
        v_state: torch.Tensor,
        base_drift: torch.Tensor,
        sigma_base: torch.Tensor,
        corr_chol: torch.Tensor,
        rf_daily: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates arbitrage-free drift `mu_t`, instantaneous volatility vector `sigma_t`,
        and log-volatility drift `dv_drift` for batch `(B, N)`.
        """
        h = self.state_encoder(torch.cat([x_state, v_state], dim=-1))
        vol_mod = 0.25 * torch.tanh(self.chol_scale_head(h))
        sigma_t = sigma_base.unsqueeze(0) * torch.exp(
            0.5 * torch.clamp(v_state + vol_mod, -2.5, 2.5)
        )
        sigma_t = torch.clamp(sigma_t, min=self.eps)

        raw_mu = base_drift.unsqueeze(0) + self.drift_head(h) * sigma_base.unsqueeze(0)
        excess = (raw_mu - rf_daily) / sigma_t

        # Solve L_corr @ theta^T = excess^T
        theta = torch.linalg.solve_triangular(
            corr_chol, excess.transpose(0, 1), upper=False
        ).transpose(0, 1)
        theta_norm = torch.linalg.norm(theta, dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(self.max_daily_sr / theta_norm, max=1.0)
        theta_bounded = theta * scale

        mu_arb_free = rf_daily + torch.matmul(theta_bounded, corr_chol.T) * sigma_t
        dv_drift = -0.20 * v_state + 0.15 * self.vol_drift_head(h)
        return mu_arb_free, sigma_t, dv_drift


class NeuralSDEEstimator(BaseScenarioEstimator, BaseReturnEstimator):
    """
    Unified Dual-Role Arbitrage-Free Neural SDE Estimator (`NeuralSDEEstimator`).

    Integrates continuous-time neural jump-diffusion dynamics from `t = 0` to `T = horizon`,
    supporting all three PyAlloq estimation paradigms:
    1. **3D Trajectory Synthesis (`BaseScenarioEstimator` / `BaseScenarioGenerator`)**:
       `.estimate_scenarios(data, n_paths, horizon)` and `.generate(n_paths, horizon)`
       return a `ScenarioMarketData` tensor of shape `(M_paths, T_horizon, N_assets)`.
    2. **Forward Return Estimation (`BaseReturnEstimator`)**:
       `.estimate(data)` integrates the arbitrage-free SDE trajectories to output an
       annualized `pd.Series` of expected returns across the horizon.
    3. **Instantaneous & Horizon Covariance Surface (`BaseCovarianceEstimator`)**:
       `.estimate_covariance(data)` returns the integrated forward `N x N` PSD covariance
       surface `pd.DataFrame` (blending `t = 0` Point-in-Time implied-vol-anchored
       instantaneous covariance with the horizon-integrated diffusion tensor).
    """

    def __init__(
        self,
        horizon: int = 20,
        n_paths: int = 200,
        hidden_dim: int = 32,
        epochs: int = 25,
        learning_rate: float = 5e-3,
        max_market_price_of_risk: float = 1.75,
        signature_mmd_weight: float = 0.5,
        martingale_reg_weight: float = 0.1,
        signature_window: int = 5,
        implied_vol_feature: str = "implied_vol",
        annualization_factor: float = 252.0,
        random_state: int | None = 42,
    ) -> None:
        self.horizon = max(1, int(horizon))
        self.n_paths = max(16, int(n_paths))
        self.hidden_dim = max(8, int(hidden_dim))
        self.epochs = max(1, int(epochs))
        self.learning_rate = float(learning_rate)
        self.max_market_price_of_risk = float(max(0.1, max_market_price_of_risk))
        self.signature_mmd_weight = float(max(0.0, signature_mmd_weight))
        self.martingale_reg_weight = float(max(0.0, martingale_reg_weight))
        self.signature_window = max(2, int(signature_window))
        self.implied_vol_feature = implied_vol_feature
        self.annualization_factor = float(annualization_factor)
        self.random_state = random_state

        self._generator = NeuralSDEGenerator(
            hidden_dim=self.hidden_dim,
            max_market_price_of_risk=self.max_market_price_of_risk,
            signature_window=self.signature_window,
            implied_vol_feature=self.implied_vol_feature,
            annualization_factor=self.annualization_factor,
            random_state=self.random_state,
        )
        self.net_: ArbitrageFreeNeuralSDENet | None = None
        self.is_fitted: bool = False
        self.assets: list[str] = []
        self.last_training_loss_: float = 0.0
        self.last_signature_mmd_: float = 0.0
        self.last_scenarios_: ScenarioMarketData | None = None

    def fit(self, data: MarketData, log_returns: bool = False) -> None:
        self._generator.fit(data, log_returns=log_returns)
        self.assets = list(self._generator.assets)
        n_assets = len(self.assets)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        max_daily_sr = self.max_market_price_of_risk / float(
            np.sqrt(self.annualization_factor)
        )
        self.net_ = ArbitrageFreeNeuralSDENet(
            n_assets=n_assets,
            hidden_dim=self.hidden_dim,
            max_daily_sr=max_daily_sr,
        )

        returns_df = data.get_returns(log_returns=log_returns).reindex(
            columns=self.assets
        )
        ret_np = returns_df.to_numpy(dtype=np.float32)
        t_obs = len(ret_np)

        assert (
            self._generator.base_drift_ is not None
            and self._generator.sigma_base_ is not None
            and self._generator.corr_cholesky_ is not None
            and self._generator.v0_ is not None
        )

        base_drift_t = torch.as_tensor(self._generator.base_drift_, dtype=torch.float32)
        sigma_base_t = torch.as_tensor(self._generator.sigma_base_, dtype=torch.float32)
        corr_chol_t = torch.as_tensor(
            self._generator.corr_cholesky_, dtype=torch.float32
        )
        ret_t = torch.as_tensor(ret_np, dtype=torch.float32)

        z_ret_t = (ret_t - base_drift_t.unsqueeze(0)) / sigma_base_t.unsqueeze(0).clamp(
            min=1e-6
        )
        v_seq_list: list[torch.Tensor] = []
        for idx in range(t_obs):
            w_slice = z_ret_t[max(0, idx - 5) : idx + 1]
            local_std = (
                torch.std(w_slice, dim=0, unbiased=False)
                if len(w_slice) > 1
                else torch.ones(n_assets)
            )
            v_seq_list.append(
                torch.clamp(2.0 * torch.log(local_std.clamp(min=0.2)), -2.0, 2.0)
            )
        v_seq_t = torch.stack(v_seq_list, dim=0)

        sig_extractor = TorchLogSignatureExtractor(
            n_channels=n_assets, window_size=min(self.signature_window, t_obs)
        )
        with torch.no_grad():
            emp_logsig_tokens, _ = sig_extractor(z_ret_t.unsqueeze(0))
            emp_sig_mean = emp_logsig_tokens.mean(dim=(0, 1))

        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)
        rf_daily = self._generator.risk_free_daily_

        for _ in range(self.epochs):
            optimizer.zero_grad()
            mu_pred, sigma_pred, _ = self.net_.forward_step(
                x_state=z_ret_t[:-1],
                v_state=v_seq_t[:-1],
                base_drift=base_drift_t,
                sigma_base=sigma_base_t,
                corr_chol=corr_chol_t,
                rf_daily=rf_daily,
            )
            res = (ret_t[1:] - mu_pred) / sigma_pred
            z_whitened = torch.linalg.solve_triangular(
                corr_chol_t, res.transpose(0, 1), upper=False
            ).transpose(0, 1)
            nll_loss = 0.5 * torch.mean(z_whitened**2) + torch.mean(
                torch.log(sigma_pred)
            )

            # Differentiable rollout for Rough-Path Signature-MMD calibration
            roll_steps = min(max(3, self.signature_window), t_obs)
            x_curr = torch.zeros((8, n_assets), dtype=torch.float32)
            v_curr = (
                torch.as_tensor(self._generator.v0_, dtype=torch.float32)
                .unsqueeze(0)
                .expand(8, n_assets)
            )
            sim_z_steps: list[torch.Tensor] = []
            for _s in range(roll_steps):
                mu_s, sig_s, dv_s = self.net_.forward_step(
                    x_curr, v_curr, base_drift_t, sigma_base_t, corr_chol_t, rf_daily
                )
                eps_x = torch.randn((8, n_assets)) @ corr_chol_t.T
                dx_s = mu_s + sig_s * eps_x
                z_s = (dx_s - base_drift_t.unsqueeze(0)) / sigma_base_t.unsqueeze(
                    0
                ).clamp(min=1e-6)
                sim_z_steps.append(z_s)
                x_curr = 0.8 * x_curr + z_s
                v_curr = torch.clamp(
                    v_curr + dv_s + 0.1 * torch.randn_like(v_curr), -2.5, 2.5
                )

            sim_z_path = torch.stack(sim_z_steps, dim=1)  # (8, roll_steps, N)
            sim_logsig_tokens, _ = sig_extractor(sim_z_path)
            sim_sig_mean = sim_logsig_tokens.mean(dim=(0, 1))
            sig_mmd = torch.mean((sim_sig_mean - emp_sig_mean) ** 2)

            martingale_pen = torch.mean((mu_pred - base_drift_t.unsqueeze(0)) ** 2)
            total_loss = (
                nll_loss
                + self.signature_mmd_weight * sig_mmd
                + self.martingale_reg_weight * martingale_pen
            )
            total_loss.backward()
            optimizer.step()

            self.last_training_loss_ = float(total_loss.detach().item())
            self.last_signature_mmd_ = float(torch.sqrt(sig_mmd.detach()).item())

        self.is_fitted = True

    def generate(
        self,
        n_paths: int | None = None,
        horizon: int | None = None,
    ) -> ScenarioMarketData:
        if not self.is_fitted:
            raise ValueError(
                "NeuralSDEEstimator needs to be fitted. Call .fit(data: MarketData) first."
            )
        m = self.n_paths if n_paths is None else int(n_paths)
        h = self.horizon if horizon is None else int(horizon)
        scenarios = self._generator.generate(n_paths=m, horizon=h)
        self.last_scenarios_ = scenarios
        return scenarios

    def estimate_scenarios(
        self,
        data: MarketData,
        n_paths: int | None = None,
        horizon: int | None = None,
        refit: bool = True,
    ) -> ScenarioMarketData:
        if refit or not self.is_fitted:
            self.fit(data)
        return self.generate(
            n_paths=self.n_paths if n_paths is None else n_paths,
            horizon=self.horizon if horizon is None else horizon,
        )

    def estimate(self, data: MarketData, **kwargs: Any) -> pd.Series:
        """
        Integrates the Arbitrage-Free Neural SDE over `horizon` to return an annualized
        `pd.Series` of forward expected asset returns (`BaseReturnEstimator` contract).
        """
        scenarios = self.estimate_scenarios(
            data,
            n_paths=kwargs.get("n_paths", self.n_paths),
            horizon=kwargs.get("horizon", self.horizon),
            refit=kwargs.get("refit", True),
        )
        mean_daily = np.mean(scenarios.returns, axis=(0, 1), dtype=np.float64)
        ann_ret = mean_daily * self.annualization_factor
        return pd.Series(ann_ret, index=self.assets, name="expected_returns")

    def estimate_covariance(self, data: MarketData, **kwargs: Any) -> pd.DataFrame:
        """
        Computes the forward-looking PSD covariance surface (`BaseCovarianceEstimator`
        compatible) by blending the `t = 0` Point-in-Time instantaneous covariance surface
        with the horizon-integrated trajectory covariance `E[int_0^H Sigma_t dt] / H`.
        """
        scenarios = self.estimate_scenarios(
            data,
            n_paths=kwargs.get("n_paths", self.n_paths),
            horizon=kwargs.get("horizon", self.horizon),
            refit=kwargs.get("refit", True),
        )
        cov_inst = self._generator.estimate_instantaneous_covariance().to_numpy(
            dtype=np.float64
        )
        flat_rets = scenarios.returns.reshape(-1, len(self.assets)).astype(np.float64)
        cov_traj = np.cov(flat_rets, rowvar=False) * self.annualization_factor
        if cov_traj.ndim == 0:
            cov_traj = np.array([[float(cov_traj)]], dtype=np.float64)

        cov_blend = 0.5 * cov_inst + 0.5 * cov_traj
        cov_blend = 0.5 * (cov_blend + cov_blend.T)
        eigvals, eigvecs = np.linalg.eigh(cov_blend)
        cov_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T
        return pd.DataFrame(cov_psd, index=self.assets, columns=self.assets)

    def estimate_instantaneous_covariance(
        self, data: MarketData | None = None
    ) -> pd.DataFrame:
        if data is not None and not self.is_fitted:
            self.fit(data)
        return self._generator.estimate_instantaneous_covariance()
