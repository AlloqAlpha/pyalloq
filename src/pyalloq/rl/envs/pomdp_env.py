from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pyalloq_backtest.costs import BaseCostModel, CostData, FlatBpsCostModel
from pyalloq_core.data import MarketData, ScenarioMarketData

from pyalloq.rl.envs.portfolio_env import _GYM_BASE, _make_box_space
from pyalloq.rl.rewards.functions import BaseRewardFunction, DifferentialSharpeReward
from pyalloq.rl.thermodynamics import WassersteinRegimeGate
from pyalloq.rl.wrappers.dict_features import make_dict_space

ProjectionMode = Literal["softmax", "dirichlet", "relu_l1"]


class POMDPPortfolioEnv(_GYM_BASE):  # type: ignore[misc]
    """
    Multiverse Partially Observable Markov Decision Process Environment (`POMDPPortfolioEnv`).

    Key Capabilities:
    - **3D ScenarioMarketData Ingestion & Synthetic Spawning**: Accepts `ScenarioMarketData` (M, T, N)
      (or `MarketData` wrapped as M=1) and randomly spawns the agent into one of the M Monte Carlo
      realities on every `reset()`.
    - **Evolvable `spaces.Dict` Observation Space**:
        * `"market_state"`: `(lookback_window, n_assets, n_features)` dynamically sized to any
          `pyalloq-features` `FeaturePipeline` or `ScenarioMarketData.features`.
        * `"current_weights"`: `(n_assets,)` tracking pre-trade drifted asset allocations w_{t^-}.
        * `"portfolio_meta"`: `(3,)` tracking `[unrealized_drawdown, cash_reserve, w2_distance]`.
    - **Simplex Action Projection & Cash Reserve Sleeve**:
      Action space is `spaces.Box(0.0, 1.0, shape=(N + 1,))` when `include_cash=True` (or `(N,)`
      when `include_cash=False`), projected onto the simplex via `"softmax"`, `"dirichlet"`, or `"relu_l1"`.
    - **Institutional Cost & Risk-Adjusted Reward Layer**:
      Deducts `BaseCostModel` (e.g., `AlmgrenChrissCostModel`, `ProportionalSpreadCostModel`,
      `FlatBpsCostModel`) directly inside `step()` and computes step-wise risk-adjusted rewards via
      `BaseRewardFunction` (`DifferentialSharpeReward`, `StepwiseSortinoReward`, `CARAUtilReward`).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario_data: ScenarioMarketData | MarketData,
        feature_pipeline: Any | None = None,
        lookback_window: int = 20,
        include_cash: bool = True,
        projection_mode: ProjectionMode = "softmax",
        softmax_temperature: float = 1.0,
        cost_model: BaseCostModel | None = None,
        reward_function: BaseRewardFunction | None = None,
        scenario_probs: np.ndarray | None = None,
        risk_free_rate: float = 0.0,
        initial_capital: float = 1.0,
        use_thermodynamic_gate: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(scenario_data, MarketData):
            prices_np = scenario_data.prices.to_numpy(dtype=np.float64)[None, ...]
            rets_np = (
                scenario_data.prices.pct_change()
                .fillna(0.0)
                .to_numpy(dtype=np.float64)[None, ...]
            )
            feats_3d: dict[str, np.ndarray] = {}
            feat_cols: dict[str, list[str]] = {}
            if scenario_data.features:
                for k, df_f in scenario_data.features.items():
                    feats_3d[k] = df_f.to_numpy(dtype=np.float64)[None, ...]
                    feat_cols[k] = [str(c) for c in df_f.columns]
            self.scenario_data = ScenarioMarketData(
                prices=prices_np,
                returns=rets_np,
                asset_names=list(scenario_data.assets),
                timestamps=scenario_data.prices.index,
                features=feats_3d,
                feature_columns=feat_cols,
            )
        else:
            self.scenario_data = scenario_data

        self.feature_pipeline = feature_pipeline
        self.n_paths = self.scenario_data.n_paths
        self.horizon = self.scenario_data.horizon
        self.n_assets = self.scenario_data.n_assets
        self.tickers = list(self.scenario_data.asset_names)
        self.lookback_window = min(lookback_window, max(1, self.horizon - 2))
        self.include_cash = include_cash
        self.projection_mode: ProjectionMode = projection_mode
        self.softmax_temperature = max(float(softmax_temperature), 1e-4)
        self.cost_model = cost_model or FlatBpsCostModel(bps=10.0)
        self.reward_function = reward_function or DifferentialSharpeReward()
        self.risk_free_daily = float(risk_free_rate) / 252.0
        self.initial_capital = float(initial_capital)
        self.use_thermodynamic_gate = use_thermodynamic_gate
        self.regime_gate = WassersteinRegimeGate(
            fast_window=max(5, self.lookback_window // 2),
            slow_window=max(10, self.lookback_window),
        )

        if scenario_probs is not None and len(scenario_probs) == self.n_paths:
            p = np.asarray(scenario_probs, dtype=np.float64)
            self.scenario_probs: np.ndarray | None = p / np.sum(p)
        else:
            self.scenario_probs = None

        # Lazy per-path cache mapping path_idx -> (T, N, F) float32 tensor and volume array (T, N)
        self._path_tensor_cache: dict[int, np.ndarray] = {}
        self._path_volume_cache: dict[int, np.ndarray] = {}

        # Probe path 0 to determine exact dynamic feature channel count F
        sample_tensor = self._get_or_build_path_tensor(0)
        self.n_features = int(sample_tensor.shape[2])

        self.action_dim = self.n_assets + (1 if self.include_cash else 0)
        self.action_space = _make_box_space(
            low=0.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = make_dict_space(
            {
                "market_state": _make_box_space(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.lookback_window, self.n_assets, self.n_features),
                    dtype=np.float32,
                ),
                "current_weights": _make_box_space(
                    low=0.0,
                    high=1.0,
                    shape=(self.n_assets,),
                    dtype=np.float32,
                ),
                "portfolio_meta": _make_box_space(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )

        self.active_path_idx: int = 0
        self.current_step: int = self.lookback_window
        self.current_weights: np.ndarray = np.full(
            self.n_assets, 1.0 / (self.n_assets + (1 if self.include_cash else 0))
        )
        self.cash_reserve: float = float(
            max(0.0, 1.0 - float(np.sum(self.current_weights)))
        )
        self.portfolio_value: float = self.initial_capital
        self.peak_value: float = self.initial_capital
        self.unrealized_drawdown: float = 0.0

    def _get_or_build_path_tensor(self, path_idx: int) -> np.ndarray:
        """
        Builds and caches the (T, N, F) market feature tensor for scenario `path_idx`.
        Channel 0 is always asset simple returns; subsequent channels come from
        `feature_pipeline.run(path_md).features` or `scenario_data.features`.
        """
        if path_idx in self._path_tensor_cache:
            return self._path_tensor_cache[path_idx]

        base_rets = np.asarray(
            self.scenario_data.returns[path_idx], dtype=np.float32
        )  # (T, N)
        channels: list[np.ndarray] = [base_rets[..., None]]
        vol_arr: np.ndarray | None = None

        if self.feature_pipeline is not None:
            path_prices = pd.DataFrame(
                self.scenario_data.prices[path_idx],
                index=self.scenario_data.timestamps,
                columns=self.scenario_data.asset_names,
            )
            path_md = self.feature_pipeline.run(MarketData(prices=path_prices))
            for fname, df_feat in path_md.features.items():
                f_np = (
                    df_feat.reindex(index=path_md.prices.index)
                    .ffill()
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                )
                if f_np.ndim == 1:
                    f_np = np.repeat(f_np[:, None], self.n_assets, axis=1)
                elif f_np.shape[1] != self.n_assets:
                    # Broadcast or tile non-asset-aligned exogenous features across N assets
                    repeats = int(np.ceil(self.n_assets / max(1, f_np.shape[1])))
                    f_np = np.tile(f_np, (1, repeats))[:, : self.n_assets]
                channels.append(f_np[..., None])
                if fname == "volume":
                    vol_arr = np.maximum(f_np.astype(np.float64), 1.0)
        elif self.scenario_data.features:
            for fname, f_tensor in self.scenario_data.features.items():
                if f_tensor.ndim == 3 and f_tensor.shape[0] == self.n_paths:
                    f_slice = np.asarray(f_tensor[path_idx], dtype=np.float32)
                    if f_slice.shape[1] != self.n_assets:
                        repeats = int(np.ceil(self.n_assets / max(1, f_slice.shape[1])))
                        f_slice = np.tile(f_slice, (1, repeats))[:, : self.n_assets]
                    channels.append(f_slice[..., None])
                    if fname == "volume":
                        vol_arr = np.maximum(f_slice.astype(np.float64), 1.0)

        if vol_arr is None:
            # Synthesize realistic dollar volume proxy (scaled by asset price & inverse vol) for Almgren-Chriss
            prices_slice = np.asarray(
                self.scenario_data.prices[path_idx], dtype=np.float64
            )
            vol_arr = np.maximum(prices_slice * 1e5, 1e4)

        full_tensor = np.concatenate(channels, axis=-1).astype(np.float32)
        self._path_tensor_cache[path_idx] = full_tensor
        self._path_volume_cache[path_idx] = vol_arr
        return full_tensor

    def project_action(
        self,
        action: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, float]:
        """
        Projects raw action vector in R^{action_dim} onto the simplex Delta^{action_dim},
        returning `(risky_weights_N, cash_weight)`.
        """
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(raw) < self.action_dim:
            pad = np.zeros(self.action_dim - len(raw), dtype=np.float64)
            raw = np.concatenate([raw, pad])
        elif len(raw) > self.action_dim:
            raw = raw[: self.action_dim]

        if self.projection_mode == "relu_l1":
            clipped = np.maximum(raw, 0.0)
            total = float(np.sum(clipped))
            if total > 1e-12:
                simplex_w = clipped / total
            else:
                simplex_w = np.full(self.action_dim, 1.0 / self.action_dim)
        elif self.projection_mode == "dirichlet":
            # Softplus concentration alpha_i = log(1 + exp(a_i)) + 1.0
            clipped_raw = np.clip(raw, -20.0, 20.0)
            alpha = np.log1p(np.exp(clipped_raw)) + 1.0
            if deterministic:
                simplex_w = alpha / np.sum(alpha)
            else:
                simplex_w = np.random.dirichlet(alpha)
        else:
            # Default: temperature-scaled softmax
            scaled = (raw - np.max(raw)) / self.softmax_temperature
            exp_w = np.exp(np.clip(scaled, -25.0, 25.0))
            simplex_w = exp_w / np.sum(exp_w)

        if self.include_cash:
            risky_w = simplex_w[: self.n_assets]
            cash_w = float(simplex_w[self.n_assets])
        else:
            risky_w = simplex_w[: self.n_assets]
            r_sum = float(np.sum(risky_w))
            if r_sum > 0:
                risky_w = risky_w / r_sum
            cash_w = 0.0

        return risky_w, cash_w

    def _compute_w2(self, path_idx: int, step_idx: int) -> float:
        rets_slice = self.scenario_data.returns[path_idx, :step_idx]
        if len(rets_slice) < max(6, self.regime_gate.fast_window):
            return 0.0
        df_r = pd.DataFrame(rets_slice, columns=self.tickers)
        return float(self.regime_gate.evaluate(df_r).wasserstein_distance)

    def _get_obs_dict(self) -> dict[str, np.ndarray]:
        path_tensor = self._get_or_build_path_tensor(self.active_path_idx)
        start = self.current_step - self.lookback_window
        end = self.current_step
        market_window = path_tensor[start:end]  # (L, N, F)

        w2_val = self._compute_w2(self.active_path_idx, end)
        meta_vec = np.array(
            [self.unrealized_drawdown, self.cash_reserve, w2_val],
            dtype=np.float32,
        )
        return {
            "market_state": market_window.astype(np.float32),
            "current_weights": self.current_weights.astype(np.float32),
            "portfolio_meta": meta_vec,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        if options and "scenario_idx" in options:
            self.active_path_idx = int(options["scenario_idx"]) % self.n_paths
        elif self.scenario_probs is not None:
            self.active_path_idx = int(
                np.random.choice(self.n_paths, p=self.scenario_probs)
            )
        else:
            self.active_path_idx = int(np.random.randint(0, self.n_paths))

        self.current_step = self.lookback_window
        init_denom = self.n_assets + (1 if self.include_cash else 0)
        self.current_weights = np.full(
            self.n_assets, 1.0 / init_denom, dtype=np.float64
        )
        self.cash_reserve = float(max(0.0, 1.0 - float(np.sum(self.current_weights))))
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.unrealized_drawdown = 0.0
        self.reward_function.reset()

        obs = self._get_obs_dict()
        return obs, {
            "scenario_idx": self.active_path_idx,
            "portfolio_value": self.portfolio_value,
            "cash_reserve": self.cash_reserve,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        target_risky_w, target_cash_w = self.project_action(action, deterministic=True)

        # Optional thermodynamic regime gating
        w2_dist = self._compute_w2(self.active_path_idx, self.current_step)
        if self.use_thermodynamic_gate:
            rets_df = pd.DataFrame(
                self.scenario_data.returns[self.active_path_idx, : self.current_step],
                columns=self.tickers,
            )
            diag = self.regime_gate.evaluate(rets_df)
            lam = 0.0 if diag.should_halt_rebalance else diag.damping_factor
            target_risky_w = lam * target_risky_w + (1.0 - lam) * self.current_weights
            target_cash_w = float(max(0.0, 1.0 - float(np.sum(target_risky_w))))

        # Compute turnover and transaction costs via BaseCostModel (e.g., AlmgrenChrissCostModel)
        delta_w = target_risky_w - self.current_weights
        turnover = float(np.sum(np.abs(delta_w)))
        delta_series = pd.Series(delta_w, index=self.tickers)

        vol_matrix = self._path_volume_cache[self.active_path_idx]
        step_vol = pd.Series(vol_matrix[self.current_step], index=self.tickers)
        features_slice: dict[str, CostData] = {"volume": step_vol}

        cost_series = self.cost_model.calculate_costs(
            delta_series, features_slice=features_slice
        )
        tc_cost = float(np.clip(pd.Series(cost_series).sum(), 0.0, 0.95))

        # Realized step returns for scenario active_path_idx at current_step
        asset_returns = np.asarray(
            self.scenario_data.returns[self.active_path_idx, self.current_step],
            dtype=np.float64,
        )
        gross_return = (
            float(np.dot(target_risky_w, asset_returns))
            + target_cash_w * self.risk_free_daily
        )
        net_return = (1.0 - tc_cost) * (1.0 + gross_return) - 1.0

        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.unrealized_drawdown = float(
            (self.portfolio_value - self.peak_value) / max(self.peak_value, 1e-12)
        )

        # Self-financing post-return weight drift w_{(t+1)^-}
        grown_risky = target_risky_w * (1.0 + asset_returns)
        grown_cash = target_cash_w * (1.0 + self.risk_free_daily)
        total_grown = float(np.sum(grown_risky) + grown_cash)
        if total_grown > 1e-12:
            self.current_weights = np.clip(grown_risky / total_grown, 0.0, 1.0)
            self.cash_reserve = float(np.clip(grown_cash / total_grown, 0.0, 1.0))
        else:
            self.current_weights = target_risky_w.copy()
            self.cash_reserve = target_cash_w

        reward = self.reward_function.compute(
            net_return=net_return,
            gross_return=gross_return,
            turnover=turnover,
            transaction_cost=tc_cost,
            unrealized_drawdown=self.unrealized_drawdown,
            w2_distance=w2_dist,
        )

        self.current_step += 1
        terminated = bool(self.current_step >= self.horizon)
        truncated = False

        if not terminated:
            next_obs = self._get_obs_dict()
        else:
            next_obs = {
                "market_state": np.zeros(
                    (self.lookback_window, self.n_assets, self.n_features),
                    dtype=np.float32,
                ),
                "current_weights": self.current_weights.astype(np.float32),
                "portfolio_meta": np.array(
                    [self.unrealized_drawdown, self.cash_reserve, w2_dist],
                    dtype=np.float32,
                ),
            }

        info = {
            "scenario_idx": self.active_path_idx,
            "portfolio_value": self.portfolio_value,
            "net_return": net_return,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": tc_cost,
            "unrealized_drawdown": self.unrealized_drawdown,
            "cash_reserve": self.cash_reserve,
            "wasserstein_distance": w2_dist,
            "weights": pd.Series(target_risky_w, index=self.tickers),
        }
        return next_obs, float(reward), terminated, truncated, info
