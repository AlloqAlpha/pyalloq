from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pyalloq_backtest.costs import BaseCostModel, FlatBpsCostModel
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.interfaces import (
    BaseAllocator,
    BaseCovarianceEstimator,
    BaseReturnEstimator,
)

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.rl.envs.portfolio_env import _GYM_BASE, _make_box_space
from pyalloq.rl.rewards.functions import BaseRewardFunction, DifferentialSharpeReward
from pyalloq.rl.thermodynamics import WassersteinRegimeGate
from pyalloq.rl.wrappers.dict_features import make_dict_space


class MetaPortfolioEnv(_GYM_BASE):  # type: ignore[misc]
    """
    Hierarchical Meta-RL Environment (`MetaPortfolioEnv`).

    Instead of picking raw asset weights directly, the Meta-Agent observes macro/regime
    belief states (e.g., GMM regime posteriors, rolling cross-asset volatility, Bures-Wasserstein
    W_2 distance, and candidate sub-allocator proposals) and outputs simplex blending coefficients
    alpha_t in Delta^K across K base allocators (e.g., Markowitz, Risk Parity, HRP, MPC):
        w_t^{meta} = sum_{k=1}^K alpha_{t, k} * w_t^{(k)}.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario_data: ScenarioMarketData | MarketData,
        base_allocators: Sequence[BaseAllocator],
        returns_estimator: BaseReturnEstimator | None = None,
        cov_estimator: BaseCovarianceEstimator | None = None,
        lookback_window: int = 30,
        rebalance_stride: int = 5,
        cost_model: BaseCostModel | None = None,
        reward_function: BaseRewardFunction | None = None,
    ) -> None:
        super().__init__()
        if len(base_allocators) < 2:
            raise ValueError("MetaPortfolioEnv requires at least 2 base_allocators.")

        if isinstance(scenario_data, MarketData):
            prices_np = scenario_data.prices.to_numpy(dtype=np.float64)[None, ...]
            rets_np = (
                scenario_data.prices.pct_change()
                .fillna(0.0)
                .to_numpy(dtype=np.float64)[None, ...]
            )
            self.scenario_data = ScenarioMarketData(
                prices=prices_np,
                returns=rets_np,
                asset_names=list(scenario_data.assets),
                timestamps=scenario_data.prices.index,
            )
        else:
            self.scenario_data = scenario_data

        self.base_allocators = list(base_allocators)
        self.n_allocators = len(self.base_allocators)
        self.returns_estimator = returns_estimator or EWMAReturnEstimator(span=30)
        self.cov_estimator = cov_estimator or EmpiricalCovariance()
        self.n_paths = self.scenario_data.n_paths
        self.horizon = self.scenario_data.horizon
        self.n_assets = self.scenario_data.n_assets
        self.tickers = list(self.scenario_data.asset_names)
        self.lookback_window = min(lookback_window, max(5, self.horizon - 2))
        self.rebalance_stride = max(1, rebalance_stride)
        self.cost_model = cost_model or FlatBpsCostModel(bps=10.0)
        self.reward_function = reward_function or DifferentialSharpeReward()
        self.regime_gate = WassersteinRegimeGate(
            fast_window=max(5, self.lookback_window // 2),
            slow_window=max(10, self.lookback_window),
        )

        # market_state has shape (1, n_assets, K + 2): K candidate allocator weight vectors + [asset_vol, asset_mom]
        self.n_features = self.n_allocators + 2
        self.action_space = _make_box_space(
            low=0.0,
            high=1.0,
            shape=(self.n_allocators,),
            dtype=np.float32,
        )
        self.observation_space = make_dict_space(
            {
                "market_state": _make_box_space(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, self.n_assets, self.n_features),
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
            self.n_assets, 1.0 / self.n_assets, dtype=np.float64
        )
        self.portfolio_value: float = 1.0
        self.peak_value: float = 1.0
        self.unrealized_drawdown: float = 0.0
        self._cached_candidate_weights: np.ndarray = np.full(
            (self.n_allocators, self.n_assets),
            1.0 / self.n_assets,
            dtype=np.float64,
        )

    def _compute_candidate_allocations(self) -> np.ndarray:
        """Evaluates each of the K base allocators on the current lookback MarketData slice."""
        start = max(0, self.current_step - self.lookback_window)
        end = self.current_step
        prices_slice = self.scenario_data.prices[self.active_path_idx, start:end]
        dates = pd.bdate_range("2023-01-02", periods=len(prices_slice))
        md_window = MarketData(
            prices=pd.DataFrame(prices_slice, index=dates, columns=self.tickers)
        )

        cov_df = self.cov_estimator.estimate(md_window)
        mu_ser = self.returns_estimator.estimate(md_window)
        curr_w_ser = pd.Series(self.current_weights, index=self.tickers)

        candidates = np.zeros((self.n_allocators, self.n_assets), dtype=np.float64)
        for k, alloc in enumerate(self.base_allocators):
            try:
                res = alloc.allocate(
                    md_window,
                    cov_matrix=cov_df,
                    expected_returns=mu_ser,
                    current_weights=curr_w_ser,
                )
                w_k = (
                    res.weights.reindex(self.tickers)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
                w_sum = float(np.sum(w_k))
                if w_sum > 1e-12:
                    w_k = np.clip(w_k, 0.0, None) / w_sum
                else:
                    w_k = np.full(self.n_assets, 1.0 / self.n_assets)
            except Exception:
                w_k = np.full(self.n_assets, 1.0 / self.n_assets)
            candidates[k] = w_k

        self._cached_candidate_weights = candidates
        return candidates

    def _get_obs_dict(self) -> dict[str, np.ndarray]:
        candidates = self._compute_candidate_allocations()  # (K, N)
        start = max(0, self.current_step - self.lookback_window)
        end = self.current_step
        rets_win = self.scenario_data.returns[self.active_path_idx, start:end]

        asset_vol = np.std(rets_win, axis=0).astype(np.float32)
        asset_mom = np.mean(rets_win, axis=0).astype(np.float32)

        # Stack (N, K) candidate weights with (N, 2) [vol, mom] -> (1, N, K + 2)
        feat_mat = np.column_stack([candidates.T, asset_vol, asset_mom]).astype(
            np.float32
        )
        df_r = pd.DataFrame(rets_win, columns=self.tickers)
        diag = self.regime_gate.evaluate(df_r)

        meta_vec = np.array(
            [
                self.unrealized_drawdown,
                diag.carnot_efficiency,
                diag.wasserstein_distance,
            ],
            dtype=np.float32,
        )
        return {
            "market_state": feat_mat[None, ...],
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
        else:
            self.active_path_idx = int(np.random.randint(0, self.n_paths))

        self.current_step = self.lookback_window
        self.current_weights = np.full(
            self.n_assets, 1.0 / self.n_assets, dtype=np.float64
        )
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.unrealized_drawdown = 0.0
        self.reward_function.reset()
        return self._get_obs_dict(), {"scenario_idx": self.active_path_idx}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        raw = np.asarray(action, dtype=np.float64).reshape(-1)[: self.n_allocators]
        shifted = raw - np.max(raw)
        blend_alpha = np.exp(np.clip(shifted, -20.0, 20.0))
        blend_alpha /= np.sum(blend_alpha)

        # Blend candidate allocator weights: w_meta = blend_alpha^T @ candidates
        target_w = blend_alpha @ self._cached_candidate_weights
        target_w = np.clip(target_w, 0.0, None)
        target_w /= np.sum(target_w)

        delta_w = target_w - self.current_weights
        turnover = float(np.sum(np.abs(delta_w)))
        tc_cost = float(
            pd.Series(
                self.cost_model.calculate_costs(pd.Series(delta_w, index=self.tickers))
            ).sum()
        )

        # Advance up to rebalance_stride steps
        end_step = min(self.horizon, self.current_step + self.rebalance_stride)
        sub_rets = self.scenario_data.returns[
            self.active_path_idx, self.current_step : end_step
        ]
        w_hold = target_w.copy()
        cum_gross = 1.0
        for r_row in sub_rets:
            step_g = float(np.dot(w_hold, r_row))
            cum_gross *= 1.0 + step_g
            grown = w_hold * (1.0 + r_row)
            g_sum = float(np.sum(grown))
            if g_sum > 1e-12:
                w_hold = grown / g_sum

        gross_return = cum_gross - 1.0
        net_return = (1.0 - tc_cost) * (1.0 + gross_return) - 1.0
        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.unrealized_drawdown = float(
            (self.portfolio_value - self.peak_value) / max(self.peak_value, 1e-12)
        )
        self.current_weights = w_hold

        reward = self.reward_function.compute(
            net_return=net_return,
            gross_return=gross_return,
            turnover=turnover,
            transaction_cost=tc_cost,
            unrealized_drawdown=self.unrealized_drawdown,
        )

        self.current_step = end_step
        terminated = bool(self.current_step >= self.horizon)
        if not terminated:
            next_obs = self._get_obs_dict()
        else:
            next_obs = {
                "market_state": np.zeros(
                    (1, self.n_assets, self.n_features), dtype=np.float32
                ),
                "current_weights": self.current_weights.astype(np.float32),
                "portfolio_meta": np.array(
                    [self.unrealized_drawdown, 0.0, 0.0], dtype=np.float32
                ),
            }

        info = {
            "portfolio_value": self.portfolio_value,
            "net_return": net_return,
            "turnover": turnover,
            "blend_weights": pd.Series(
                blend_alpha,
                index=[a.__class__.__name__ for a in self.base_allocators],
            ),
            "weights": pd.Series(target_w, index=self.tickers),
        }
        return next_obs, float(reward), terminated, False, info
