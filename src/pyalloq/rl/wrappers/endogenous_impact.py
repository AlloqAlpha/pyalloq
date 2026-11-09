from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym

    _WRAPPER_BASE: type = gym.Wrapper
except ImportError:
    gym = None
    _WRAPPER_BASE = object


class EndogenousLiquidityImpactWrapper(_WRAPPER_BASE):  # type: ignore[misc]
    """
    Endogenous Liquidity & Graph Contagion Impact Wrapper (`EndogenousLiquidityImpactWrapper`).

    Elevates historical or synthetic RL environments (`PortfolioGymEnv`, `ScenarioPortfolioEnv`,
    `POMDPPortfolioEnv`) from a static price-taker tape into a reactive order-book resilience
    and graph contagion simulator:
      1. Transient Order-Book Impact State `q_t`:
             q_{t+1} = resilience_rho * q_t + impact_lambda * |Delta w_t|
                       + diffusion_kappa * (I - L_t) |Delta w_t|
         where `L_t = I - D_t^{-1/2} W_t D_t^{-1/2}` is the symmetric normalized Graph Laplacian
         of the asset network, modeling cross-asset liquidity withdrawal when an agent aggressively
         rebalances a hub asset.
      2. Endogenous Spread-Widening & Execution Cost:
             impact_cost_t = <q_{t+1}, |Delta w_t|>
         Deducts the non-linear transient market impact and cross-asset contagion cost from
         `reward` and updates `portfolio_value` and `info` diagnostics.
    """

    def __init__(
        self,
        env: Any,
        impact_lambda: float = 0.005,
        resilience_rho: float = 0.65,
        diffusion_kappa: float = 0.002,
        adjacency_matrix: np.ndarray | None = None,
    ) -> None:
        if gym is not None and isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        self.impact_lambda = float(max(0.0, impact_lambda))
        self.resilience_rho = float(np.clip(resilience_rho, 0.0, 0.999))
        self.diffusion_kappa = float(max(0.0, diffusion_kappa))
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.n_assets = int(getattr(env, "n_assets", env.action_space.shape[0]))
        self._transient_impact = np.zeros(self.n_assets, dtype=np.float64)
        self._cum_impact_cost: float = 0.0

        if adjacency_matrix is not None:
            adj = np.asarray(adjacency_matrix, dtype=np.float64)
            if adj.shape != (self.n_assets, self.n_assets):
                raise ValueError(
                    f"adjacency_matrix shape {adj.shape} must match ({self.n_assets}, {self.n_assets})."
                )
            self._base_adj = adj
        else:
            self._base_adj = self._infer_adjacency_from_env(env)

        self._normalized_adj = self._compute_normalized_adjacency(self._base_adj)

    def _infer_adjacency_from_env(self, env: Any) -> np.ndarray:
        """Infers an asset adjacency matrix from the wrapped environment's return history."""
        returns_df = getattr(env, "returns_df", None)
        if returns_df is not None and len(returns_df) > 5:
            corr = returns_df.corr().fillna(0.0).to_numpy(dtype=np.float64)
            w_mat = np.maximum(corr, 0.0)
            np.fill_diagonal(w_mat, 0.0)
            return w_mat

        scenario_data = getattr(env, "scenario_data", None)
        if scenario_data is not None and scenario_data.returns.shape[1] > 2:
            flat_ret = scenario_data.returns.reshape(-1, self.n_assets)
            if flat_ret.shape[0] > 3:
                std = np.std(flat_ret, axis=0)
                if np.all(std > 1e-10):
                    corr = np.corrcoef(flat_ret, rowvar=False)
                    corr = np.nan_to_num(corr, nan=0.0)
                    w_mat = np.maximum(corr, 0.0)
                    np.fill_diagonal(w_mat, 0.0)
                    return w_mat

        w_mat = np.ones((self.n_assets, self.n_assets), dtype=np.float64) - np.eye(
            self.n_assets, dtype=np.float64
        )
        return w_mat / max(1, self.n_assets - 1)

    @staticmethod
    def _compute_normalized_adjacency(w_mat: np.ndarray) -> np.ndarray:
        w = 0.5 * (w_mat + w_mat.T)
        w = np.maximum(w, 0.0)
        np.fill_diagonal(w, 0.0)
        deg = np.sum(w, axis=1)
        inv_sqrt_deg = np.where(deg > 1e-10, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
        d_inv_sqrt = np.diag(inv_sqrt_deg)
        return d_inv_sqrt @ w @ d_inv_sqrt

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._transient_impact = np.zeros(self.n_assets, dtype=np.float64)
        self._cum_impact_cost = 0.0
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["transient_impact_state"] = self._transient_impact.copy()
        info["endogenous_impact_cost"] = 0.0
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        prev_weights = getattr(self.env, "current_weights", None)
        if prev_weights is not None:
            prev_w = np.asarray(prev_weights, dtype=np.float64).copy()
        else:
            prev_w = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float64)

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        if "weights" in info and hasattr(info["weights"], "to_numpy"):
            new_w = info["weights"].to_numpy(dtype=np.float64)
        else:
            curr_w = getattr(self.env, "current_weights", prev_w)
            new_w = np.asarray(curr_w, dtype=np.float64)

        delta_abs = np.abs(new_w - prev_w)
        graph_spillover = self.diffusion_kappa * (self._normalized_adj @ delta_abs)

        self._transient_impact = (
            self.resilience_rho * self._transient_impact
            + self.impact_lambda * delta_abs
            + graph_spillover
        )

        impact_cost = float(np.dot(self._transient_impact, delta_abs))
        self._cum_impact_cost += impact_cost

        adjusted_reward = float(reward - impact_cost)
        if hasattr(self.env, "portfolio_value"):
            self.env.portfolio_value = float(
                self.env.portfolio_value * max(0.0, 1.0 - impact_cost)
            )
            info["portfolio_value"] = self.env.portfolio_value

        info["endogenous_impact_cost"] = impact_cost
        info["cumulative_impact_cost"] = self._cum_impact_cost
        info["transient_impact_state"] = self._transient_impact.copy()
        info["graph_contagion_spillover"] = float(np.sum(graph_spillover))

        return obs, adjusted_reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
