from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator
from pyalloq_core.results import OptimizationResult

from pyalloq.rl.envs.portfolio_env import PortfolioGymEnv
from pyalloq.rl.thermodynamics import WassersteinRegimeGate
from pyalloq.rl.trajectory import TrajectoryPolicyNet


class SB3CompatiblePolicy(Protocol):
    """Protocol matching Stable-Baselines3 (`PPO`, `SAC`, `DDPG`, `TD3`, `A2C`) models."""

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]: ...


class RLAllocator(BaseAllocator):
    """
    Reinforcement Learning & Deep Hedging Portfolio Allocator (Pillar 3).

    Wraps either:
    1. A PyTorch `TrajectoryPolicyNet` trained with `DeepHedgingTrajectoryTrainer`, OR
    2. Any Stable-Baselines3 / FinRL agent implementing `.predict(obs, deterministic=True)`.

    Optionally applies the `WassersteinRegimeGate` (Moroke et al. 2026) during inference to
    dampen or halt turnover when the Bures-Wasserstein distance between fast and slow market
    regimes indicates excessive frictional dissipation.
    """

    def __init__(
        self,
        policy: TrajectoryPolicyNet | SB3CompatiblePolicy,
        lookback_window: int = 20,
        regime_gate: WassersteinRegimeGate | None = None,
        device: str = "cpu",
        tickers: list[str] | None = None,
    ) -> None:
        super().__init__(tickers or [])
        self.policy = policy
        self.lookback_window = lookback_window
        self.regime_gate = regime_gate
        self.device = device

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        if not self.tickers and data.assets:
            self.tickers = list(data.assets)
        tickers = list(data.prices.columns)
        n_assets = len(tickers)

        returns_df = data.prices.pct_change().fillna(0.0)
        ret_window = returns_df.iloc[-self.lookback_window :]
        if len(ret_window) < self.lookback_window:
            pad_len = self.lookback_window - len(ret_window)
            pad_df = pd.DataFrame(0.0, index=range(pad_len), columns=ret_window.columns)
            ret_window = pd.concat([pad_df, ret_window], ignore_index=True)

        if current_weights is not None:
            w_curr_np = (
                current_weights.reindex(tickers)
                .fillna(1.0 / n_assets)
                .to_numpy(dtype=np.float32)
            )
            w_sum = float(w_curr_np.sum())
            if w_sum > 0:
                w_curr_np = w_curr_np / w_sum
        else:
            w_curr_np = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)

        if isinstance(self.policy, TrajectoryPolicyNet):
            self.policy.eval()
            with torch.no_grad():
                ret_t = torch.tensor(
                    ret_window.to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                w_t = torch.tensor(
                    w_curr_np, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                w_out = self.policy(ret_t, w_t).squeeze(0).cpu().numpy()
        else:
            # Build FinRL observation vector via PortfolioGymEnv helper
            env = PortfolioGymEnv(
                data=data,
                lookback_window=self.lookback_window,
                use_thermodynamic_gate=False,
            )
            env.current_step = len(data.prices)
            env.current_weights = w_curr_np.astype(np.float64)
            obs = env._get_observation()
            raw_action, _ = self.policy.predict(obs, deterministic=True)
            w_out = PortfolioGymEnv.project_action_to_simplex(raw_action)

        target_series = pd.Series(w_out, index=tickers, name="weights", dtype=float)
        metadata: dict[str, Any] = {"policy_type": type(self.policy).__name__}

        if self.regime_gate is not None:
            target_series, diag = self.regime_gate.gate_weights(
                target_weights=target_series,
                current_weights=current_weights,
                returns=returns_df,
            )
            metadata.update(
                {
                    "wasserstein_distance": diag.wasserstein_distance,
                    "carnot_efficiency": diag.carnot_efficiency,
                    "damping_factor": diag.damping_factor,
                    "should_halt_rebalance": diag.should_halt_rebalance,
                }
            )

        return OptimizationResult(
            weights=target_series,
            status="OPTIMAL_RL_POLICY",
            metadata=metadata,
        )
