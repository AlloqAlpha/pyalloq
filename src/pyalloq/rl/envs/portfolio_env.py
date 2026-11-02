from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData, ScenarioMarketData

from pyalloq.rl.thermodynamics import WassersteinRegimeGate

try:
    import gymnasium as gym  # type: ignore[import-not-found]
    from gymnasium import spaces  # type: ignore[import-not-found]

    _GYM_BASE: type = gym.Env
except ImportError:
    gym = None
    spaces = None
    _GYM_BASE = object


class _BoxSpaceFallback:
    """Lightweight fallback space when `gymnasium` is not installed."""

    def __init__(
        self, low: float, high: float, shape: tuple[int, ...], dtype: Any = np.float32
    ) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=self.shape).astype(
            self.dtype
        )


def _make_box_space(
    low: float, high: float, shape: tuple[int, ...], dtype: Any = np.float32
) -> Any:
    if spaces is not None:
        return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _BoxSpaceFallback(low=low, high=high, shape=shape, dtype=dtype)


class PortfolioGymEnv(_GYM_BASE):  # type: ignore[misc]
    """
    FinRL-Compatible Multi-Period Portfolio Optimization Environment (Research Section 7.1).

    State Space s_t:
        Flattened vector containing:
        - Pre-trade drifted portfolio weights w_{t^-} (N,)
        - Lookback window of asset returns R_{t-L+1:t} (L * N,)
        - Optional stacked features F_{t} (K * N,)
        - Rolling volatility diagonal sigma_t (N,)
        - Mahalanobis turbulence index d_t and Bures-Wasserstein regime distance W_2 (2,)

    Action Space a_t:
        Continuous vector in [-1, 1]^N mapped onto the long-only simplex Delta^N
        (or temperature-scaled softmax), followed by optional thermodynamic regime gating
        and turbulence circuit-breaker.

    Reward Function r_t:
        Differential log-return / risk-adjusted net portfolio return:
            r_t = log(1 + r_{p,t}^{net}) - (gamma / 2) * (r_{p,t}^{net})^2 - eta_w2 * W_2 * turnover_t
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: MarketData,
        lookback_window: int = 20,
        transaction_cost_bps: float = 10.0,
        risk_aversion: float = 1.0,
        turbulence_threshold: float | None = None,
        use_thermodynamic_gate: bool = True,
        initial_capital: float = 1.0,
    ) -> None:
        super().__init__()
        self.data = data
        self.prices = data.prices
        self.returns_df = data.prices.pct_change().fillna(0.0)
        self.tickers = list(data.prices.columns)
        self.n_assets = len(self.tickers)
        self.n_steps = len(self.prices)
        self.lookback_window = min(lookback_window, max(2, self.n_steps - 2))
        self.tc_rate = transaction_cost_bps / 10000.0
        self.risk_aversion = risk_aversion
        self.turbulence_threshold = turbulence_threshold
        self.use_thermodynamic_gate = use_thermodynamic_gate
        self.initial_capital = initial_capital
        self.regime_gate = WassersteinRegimeGate(
            fast_window=max(5, self.lookback_window // 2),
            slow_window=max(10, self.lookback_window),
        )

        self.feature_names = list(data.features.keys()) if data.features else []
        self.n_extra_features = len(self.feature_names)

        # State dimension: N (drifted weights) + L*N (lookback returns) + K*N (extra features) + N (vols) + 2 (turbulence, W2)
        self.state_dim = (
            self.n_assets
            + self.lookback_window * self.n_assets
            + self.n_extra_features * self.n_assets
            + self.n_assets
            + 2
        )

        self.action_space = _make_box_space(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = _make_box_space(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.current_step: int = self.lookback_window
        self.current_weights: np.ndarray = np.full(
            self.n_assets, 1.0 / self.n_assets, dtype=np.float64
        )
        self.portfolio_value: float = self.initial_capital
        self.asset_memory: list[float] = [self.initial_capital]
        self.weights_memory: list[np.ndarray] = []
        self.return_memory: list[float] = []

    @staticmethod
    def project_action_to_simplex(action: np.ndarray) -> np.ndarray:
        """Map raw continuous action vector in R^N onto the probability simplex Delta^N."""
        arr = np.asarray(action, dtype=np.float64).flatten()
        if np.all(arr >= 0.0) and np.isclose(np.sum(arr), 1.0, atol=1e-4):
            return arr / np.sum(arr)
        shifted = arr - np.max(arr)
        exp_a = np.exp(np.clip(shifted, -20.0, 20.0))
        return exp_a / np.sum(exp_a)

    def _compute_turbulence(self, step_idx: int) -> float:
        """Compute FinRL Mahalanobis turbulence index d_t at `step_idx`."""
        if step_idx < max(10, self.lookback_window):
            return 0.0
        hist = self.returns_df.iloc[max(0, step_idx - 60) : step_idx]
        curr = self.returns_df.iloc[step_idx].to_numpy(dtype=np.float64)
        mu = hist.mean().to_numpy(dtype=np.float64)
        cov = (
            hist.cov().fillna(0.0).to_numpy(dtype=np.float64)
            + np.eye(self.n_assets) * 1e-6
        )
        diff = curr - mu
        try:
            inv_cov = np.linalg.pinv(cov)
            turb = float(diff.T @ inv_cov @ diff) / self.n_assets
            return max(0.0, turb)
        except np.linalg.LinAlgError:
            return 0.0

    def _get_observation(self) -> np.ndarray:
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        ret_window = self.returns_df.iloc[start_idx:end_idx]

        ret_flat = ret_window.to_numpy(dtype=np.float32).flatten()
        vols = ret_window.std().fillna(0.0).to_numpy(dtype=np.float32)

        feat_parts: list[np.ndarray] = []
        if self.data.features:
            for fname in self.feature_names:
                f_row = (
                    self.data.features[fname]
                    .iloc[end_idx - 1]
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                )
                feat_parts.append(f_row)

        turb = np.float32(self._compute_turbulence(end_idx - 1))
        diag = self.regime_gate.evaluate(self.returns_df.iloc[:end_idx])
        w2 = np.float32(diag.wasserstein_distance)

        obs_components = [
            self.current_weights.astype(np.float32),
            ret_flat,
            *feat_parts,
            vols,
            np.array([turb, w2], dtype=np.float32),
        ]
        return np.concatenate(obs_components, axis=0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self.current_step = self.lookback_window
        self.current_weights = np.full(
            self.n_assets, 1.0 / self.n_assets, dtype=np.float64
        )
        self.portfolio_value = self.initial_capital
        self.asset_memory = [self.initial_capital]
        self.weights_memory = [self.current_weights.copy()]
        self.return_memory = []
        obs = self._get_observation()
        return obs, {"portfolio_value": self.portfolio_value}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target_w = self.project_action_to_simplex(action)

        # Optional thermodynamic regime gating
        hist_returns = self.returns_df.iloc[: self.current_step]
        diag = self.regime_gate.evaluate(hist_returns)
        if self.use_thermodynamic_gate:
            lam = 0.0 if diag.should_halt_rebalance else diag.damping_factor
            target_w = lam * target_w + (1.0 - lam) * self.current_weights
            target_w /= np.sum(target_w)

        # FinRL turbulence circuit breaker: shift toward minimum-turnover hold if turbulence spikes
        turb = self._compute_turbulence(self.current_step)
        if self.turbulence_threshold is not None and turb > self.turbulence_threshold:
            target_w = self.current_weights.copy()

        # Compute turnover and transaction cost at current_step
        turnover = float(np.sum(np.abs(target_w - self.current_weights)))
        tc_cost = self.tc_rate * turnover

        # Realized asset returns over period current_step
        step_returns = self.returns_df.iloc[self.current_step].to_numpy(
            dtype=np.float64
        )
        gross_return = float(np.dot(target_w, step_returns))
        net_return = (1.0 - tc_cost) * (1.0 + gross_return) - 1.0

        self.portfolio_value *= 1.0 + net_return

        # Update drifted pre-trade weights w_{(t+1)^-}
        grown = target_w * (1.0 + step_returns)
        denom = float(np.sum(grown))
        if denom > 0:
            self.current_weights = grown / denom
        else:
            self.current_weights = target_w.copy()

        self.asset_memory.append(self.portfolio_value)
        self.weights_memory.append(target_w.copy())
        self.return_memory.append(net_return)

        # Differential risk-adjusted reward with thermodynamic friction penalty
        log_ret = float(np.log(max(1e-8, 1.0 + net_return)))
        risk_pen = 0.5 * self.risk_aversion * (net_return**2)
        thermo_pen = 0.1 * diag.wasserstein_distance * turnover
        reward = float(log_ret - risk_pen - thermo_pen)

        self.current_step += 1
        terminated = bool(self.current_step >= self.n_steps)
        truncated = False

        next_obs = (
            self._get_observation()
            if not terminated
            else np.zeros(self.state_dim, dtype=np.float32)
        )
        info = {
            "portfolio_value": self.portfolio_value,
            "net_return": net_return,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": tc_cost,
            "turbulence": turb,
            "wasserstein_distance": diag.wasserstein_distance,
            "carnot_efficiency": diag.carnot_efficiency,
            "weights": pd.Series(target_w, index=self.tickers),
        }
        return next_obs, reward, terminated, truncated, info


class ScenarioPortfolioEnv(_GYM_BASE):  # type: ignore[misc]
    """
    Generative Scenario Portfolio Environment wrapping `ScenarioMarketData` (M, T, N).

    Each `reset()` samples one of the M Monte Carlo / Bootstrap / Diffusion paths,
    enabling RL policies to train across synthetic forward trajectories without overfitting
    to a single historical realization.
    """

    def __init__(
        self,
        scenario_data: ScenarioMarketData,
        lookback_window: int = 10,
        transaction_cost_bps: float = 10.0,
        risk_aversion: float = 1.0,
    ) -> None:
        super().__init__()
        self.scenario_data = scenario_data
        self.n_scenarios, self.horizon, self.n_assets = scenario_data.returns.shape
        self.tickers = list(scenario_data.asset_names)
        self.lookback_window = min(lookback_window, max(1, self.horizon - 2))
        self.tc_rate = transaction_cost_bps / 10000.0
        self.risk_aversion = risk_aversion

        self.state_dim = self.n_assets + self.lookback_window * self.n_assets
        self.action_space = _make_box_space(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = _make_box_space(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.active_scenario_idx: int = 0
        self.current_step: int = self.lookback_window
        self.current_weights: np.ndarray = np.full(
            self.n_assets, 1.0 / self.n_assets, dtype=np.float64
        )
        self.portfolio_value: float = 1.0

    def _get_obs(self) -> np.ndarray:
        path = self.scenario_data.returns[self.active_scenario_idx]
        window = path[self.current_step - self.lookback_window : self.current_step]
        return np.concatenate(
            [
                self.current_weights.astype(np.float32),
                window.astype(np.float32).flatten(),
            ]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        if options and "scenario_idx" in options:
            self.active_scenario_idx = int(options["scenario_idx"]) % self.n_scenarios
        else:
            self.active_scenario_idx = int(np.random.randint(0, self.n_scenarios))

        self.current_step = self.lookback_window
        self.current_weights = np.full(
            self.n_assets, 1.0 / self.n_assets, dtype=np.float64
        )
        self.portfolio_value = 1.0
        return self._get_obs(), {"scenario_idx": self.active_scenario_idx}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target_w = PortfolioGymEnv.project_action_to_simplex(action)
        turnover = float(np.sum(np.abs(target_w - self.current_weights)))
        tc_cost = self.tc_rate * turnover

        step_returns = self.scenario_data.returns[
            self.active_scenario_idx, self.current_step
        ]
        gross_ret = float(np.dot(target_w, step_returns))
        net_ret = (1.0 - tc_cost) * (1.0 + gross_ret) - 1.0
        self.portfolio_value *= 1.0 + net_ret

        grown = target_w * (1.0 + step_returns)
        denom = float(np.sum(grown))
        self.current_weights = grown / denom if denom > 0 else target_w.copy()

        reward = float(
            np.log(max(1e-8, 1.0 + net_ret)) - 0.5 * self.risk_aversion * (net_ret**2)
        )
        self.current_step += 1
        terminated = bool(self.current_step >= self.horizon)
        next_obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.state_dim, dtype=np.float32)
        )
        return (
            next_obs,
            reward,
            terminated,
            False,
            {
                "portfolio_value": self.portfolio_value,
                "net_return": net_ret,
                "turnover": turnover,
                "scenario_idx": self.active_scenario_idx,
            },
        )
