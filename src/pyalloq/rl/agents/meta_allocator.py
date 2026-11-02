from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import (
    BaseAllocator,
    BaseCovarianceEstimator,
    BaseReturnEstimator,
)
from pyalloq_core.results import OptimizationResult

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.rl.agents.base import BaseRLAgent
from pyalloq.rl.thermodynamics import WassersteinRegimeGate


class HierarchicalMetaRLAllocator(BaseAllocator):
    """
    Hierarchical Meta-RL Allocator (`HierarchicalMetaRLAllocator`).

    Instead of selecting individual asset weights directly, observes macro regime features
    (volatility, Bures-Wasserstein W_2 regime distance, Carnot efficiency, and candidate
    sub-allocator proposals) and outputs blending coefficients alpha_t in Delta^K across
    K base allocators (e.g., `[MarkowitzAllocator, RiskParityAllocator, HRPAllocator, MultiPeriodMPCAllocator]`):
        w_t^{meta} = sum_{k=1}^K alpha_{t, k} * w_t^{(k)}.

    Can be driven by a trained `SB3PortfolioAgent` (trained on `MetaPortfolioEnv`) or by
    its built-in regime-adaptive Bayesian/Thermodynamic meta-policy when `meta_agent=None`.
    """

    def __init__(
        self,
        base_allocators: Sequence[BaseAllocator],
        meta_agent: BaseRLAgent | None = None,
        returns_estimator: BaseReturnEstimator | None = None,
        cov_estimator: BaseCovarianceEstimator | None = None,
        regime_gate: WassersteinRegimeGate | None = None,
        lookback_window: int = 30,
        tickers: list[str] | None = None,
    ) -> None:
        super().__init__(tickers or [])
        if len(base_allocators) < 2:
            raise ValueError(
                "HierarchicalMetaRLAllocator requires at least 2 base_allocators."
            )
        self.base_allocators = list(base_allocators)
        self.n_allocators = len(self.base_allocators)
        self.meta_agent = meta_agent
        self.returns_estimator = returns_estimator or EWMAReturnEstimator(span=30)
        self.cov_estimator = cov_estimator or EmpiricalCovariance()
        self.regime_gate = regime_gate or WassersteinRegimeGate(
            fast_window=max(5, lookback_window // 2),
            slow_window=max(15, lookback_window),
        )
        self.lookback_window = lookback_window

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

        cov_df = (
            cov_matrix if cov_matrix is not None else self.cov_estimator.estimate(data)
        )
        mu_ser = (
            expected_returns
            if expected_returns is not None
            else self.returns_estimator.estimate(data)
        )

        if current_weights is not None:
            w_curr_np = (
                current_weights.reindex(tickers)
                .fillna(1.0 / n_assets)
                .to_numpy(dtype=np.float64, copy=True)
            )
            w_sum = float(np.sum(w_curr_np))
            if w_sum > 1e-12:
                w_curr_np = w_curr_np / w_sum
        else:
            w_curr_np = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

        curr_w_ser = pd.Series(w_curr_np, index=tickers)

        # 1. Evaluate all K base allocators
        candidate_weights = np.zeros((self.n_allocators, n_assets), dtype=np.float64)
        allocator_names: list[str] = []
        for k, alloc in enumerate(self.base_allocators):
            allocator_names.append(f"{alloc.__class__.__name__}_{k}")
            try:
                res = alloc.allocate(
                    data,
                    cov_matrix=cov_df,
                    expected_returns=mu_ser,
                    current_weights=curr_w_ser,
                )
                w_k = (
                    res.weights.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float64)
                )
                s_k = float(np.sum(w_k))
                if s_k > 1e-12:
                    w_k = np.clip(w_k, 0.0, None) / s_k
                else:
                    w_k = np.full(n_assets, 1.0 / n_assets)
            except Exception:
                w_k = np.full(n_assets, 1.0 / n_assets)
            candidate_weights[k] = w_k

        # 2. Build Meta Observation Dict
        rets_df = data.prices.pct_change().fillna(0.0)
        rets_win = rets_df.iloc[-self.lookback_window :].to_numpy(dtype=np.float32)
        asset_vol = np.std(rets_win, axis=0).astype(np.float32)
        asset_mom = np.mean(rets_win, axis=0).astype(np.float32)

        feat_mat = np.column_stack([candidate_weights.T, asset_vol, asset_mom]).astype(
            np.float32
        )
        diag = self.regime_gate.evaluate(rets_df)

        obs_dict: dict[str, np.ndarray] = {
            "market_state": feat_mat[None, ...],
            "current_weights": w_curr_np.astype(np.float32),
            "portfolio_meta": np.array(
                [0.0, diag.carnot_efficiency, diag.wasserstein_distance],
                dtype=np.float32,
            ),
        }

        # 3. Compute Meta Blending Coefficients alpha_t in Delta^K
        if self.meta_agent is not None:
            raw_act, _ = self.meta_agent.predict(obs_dict, deterministic=True)
            raw_arr = np.asarray(raw_act, dtype=np.float64).reshape(-1)[
                : self.n_allocators
            ]
            shifted = raw_arr - np.max(raw_arr)
            blend_alpha = np.exp(np.clip(shifted, -20.0, 20.0))
            blend_alpha /= np.sum(blend_alpha)
        else:
            # Regime-adaptive ex-ante Sharpe minus turnover utility per candidate allocator
            cov_np = cov_df.reindex(index=tickers, columns=tickers).to_numpy(
                dtype=np.float64
            )
            mu_np = mu_ser.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float64)
            scores = np.zeros(self.n_allocators, dtype=np.float64)
            for k in range(self.n_allocators):
                w_k = candidate_weights[k]
                vol_k = float(np.sqrt(max(float(w_k.T @ cov_np @ w_k), 1e-8)))
                ret_k = float(w_k @ mu_np)
                to_k = float(np.sum(np.abs(w_k - w_curr_np)))
                # Penalize high-turnover allocators when W_2 regime distance is elevated
                scores[k] = (ret_k / vol_k) - (
                    1.0 + 15.0 * diag.wasserstein_distance
                ) * to_k
            shifted = scores - np.max(scores)
            blend_alpha = np.exp(np.clip(shifted, -15.0, 15.0))
            blend_alpha /= np.sum(blend_alpha)

        blended_w = blend_alpha @ candidate_weights
        blended_w = np.clip(blended_w, 0.0, None)
        blended_w /= np.sum(blended_w)

        blend_map = {
            name: float(alpha) for name, alpha in zip(allocator_names, blend_alpha)
        }
        return OptimizationResult(
            weights=pd.Series(blended_w, index=tickers, name="weights"),
            status="OPTIMAL_HIERARCHICAL_META_RL",
            metadata={
                "allocator_blend_weights": blend_map,
                "wasserstein_distance": diag.wasserstein_distance,
                "carnot_efficiency": diag.carnot_efficiency,
            },
        )
