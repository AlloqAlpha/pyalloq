from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import BaseAllocator, BaseRegimeDetector
from pyalloq_core.results import OptimizationResult
from pyalloq_features.regimes.jump_models import ContinuousJumpModelDetector
from sklearn.mixture import GaussianMixture

from pyalloq.classical.optimizers.naive import EqualWeightAllocator
from pyalloq.classical.optimizers.risk_parity import RiskParityAllocator


class RollingGaussianHMMDetector(BaseRegimeDetector):
    """
    Unregularized Rolling Gaussian HMM / GMM-Viterbi Baseline Detector
    (`RollingGaussianHMMDetector`, Hamilton, 1989; Supriadi & Siswoyo, Oct 2026,
    arXiv:2610.27751v4).

    Provided as an explicit empirical baseline comparator against `ContinuousJumpModelDetector`
    and `SparseJumpModelDetector` to reproduce how unpenalized discrete state partitioning
    and hard Viterbi decoding suffer from minority-regime sample starvation (`T_k < N`)
    and regime-switch churn in rolling walk-forward backtests.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        hard_viterbi: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = max(2, int(n_regimes))
        self.hard_viterbi = bool(hard_viterbi)
        self.random_state = int(random_state)
        self.is_fitted: bool = False
        self._gmm: GaussianMixture | None = None
        self._order: np.ndarray | None = None

    def fit(self, data: MarketData) -> RollingGaussianHMMDetector:
        returns = data.prices.pct_change().dropna()
        ret_np = returns.to_numpy(dtype=np.float64)
        k_eff = min(self.n_regimes, max(2, len(ret_np) // 4))
        gmm = GaussianMixture(
            n_components=k_eff,
            covariance_type="full",
            reg_covar=1e-5,
            random_state=self.random_state,
        )
        gmm.fit(ret_np)
        vols = np.array([np.trace(gmm.covariances_[k]) for k in range(k_eff)])
        self._order = np.argsort(vols)
        self._gmm = gmm
        self.is_fitted = True
        return self

    def detect(self, data: MarketData, refit: bool = True) -> Any:
        from pyalloq_core.results import RegimeState

        if refit or not self.is_fitted or self._gmm is None or self._order is None:
            self.fit(data)
        assert self._gmm is not None and self._order is not None

        returns = data.prices.pct_change().dropna()
        ret_np = returns.to_numpy(dtype=np.float64)
        k_eff = len(self._order)
        raw_probs = self._gmm.predict_proba(ret_np)[:, self._order]
        hard_states = np.argmax(raw_probs, axis=1).astype(np.int64)

        if self.hard_viterbi:
            probs_np = np.eye(k_eff, dtype=np.float64)[hard_states]
        else:
            probs_np = raw_probs

        cols = [f"regime_{k}" for k in range(k_eff)]
        n_switches = (
            int(np.sum(hard_states[1:] != hard_states[:-1]))
            if len(hard_states) > 1
            else 0
        )
        return RegimeState(
            probs=pd.DataFrame(probs_np, index=returns.index, columns=cols),
            hard_state=pd.Series(hard_states, index=returns.index, name="regime_state"),
            confidence=pd.Series(
                np.max(raw_probs, axis=1), index=returns.index, name="regime_confidence"
            ),
            metadata={
                "detector": "RollingGaussianHMMDetector",
                "n_regimes": k_eff,
                "hard_viterbi": self.hard_viterbi,
                "n_switches": n_switches,
            },
        )


class RegimeSwitchingAllocator(BaseAllocator):
    """
    Persistence-Regularized Regime-Switching Portfolio Allocator (`RegimeSwitchingAllocator`).

    Conditioned by default on `ContinuousJumpModelDetector` (or any `BaseRegimeDetector`),
    avoids the two failure modes of unregularized HMM-conditioned optimization
    (Supriadi & Siswoyo, Oct 2026, arXiv:2610.27751v4):
    1. **Eliminates Sample Starvation (`T_k < N`)**: Uses full-sample or relevance-weighted
       covariance matrices rather than slicing a short rolling window into tiny discrete
       regime buckets.
    2. **Eliminates Viterbi Boundary Churn**: Blends regime-specific child allocators
       (e.g., `[BullGrowthAllocator, NeutralHRPAllocator, BearDefensiveAllocator]`)
       smoothly across the continuous simplex vector `p_t in Delta^K`, with optional
       defensive cash scaling in the crisis regime.
    """

    def __init__(
        self,
        regime_allocators: Sequence[BaseAllocator] | None = None,
        regime_detector: BaseRegimeDetector | None = None,
        mode: Literal[
            "probability_blend", "hard_switch", "partitioned_hmm"
        ] = "probability_blend",
        bear_defensive_cash_weight: float = 0.0,
        tickers: list[str] | None = None,
    ) -> None:
        super().__init__(tickers or [])
        self.regime_allocators = (
            list(regime_allocators)
            if regime_allocators is not None
            else [RiskParityAllocator(), EqualWeightAllocator()]
        )
        n_reg = max(2, len(self.regime_allocators))
        self.regime_detector: BaseRegimeDetector = (
            regime_detector
            if regime_detector is not None
            else ContinuousJumpModelDetector(n_regimes=n_reg, jump_penalty=20.0)
        )
        self.mode = mode
        self.bear_defensive_cash_weight = float(
            np.clip(bear_defensive_cash_weight, 0.0, 0.90)
        )
        self.last_regime_state_: Any = None

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        assets = (
            list(cov_matrix.columns)
            if cov_matrix is not None
            else list(data.prices.columns)
        )
        n_assets = len(assets)
        returns = data.prices.reindex(columns=assets).pct_change().dropna()
        if cov_matrix is None:
            cov_matrix = (
                returns.cov().fillna(0.0) * 252.0
                if len(returns) >= 2
                else pd.DataFrame(np.eye(n_assets) * 0.04, index=assets, columns=assets)
            )
        if expected_returns is None:
            expected_returns = (
                returns.mean().fillna(0.0) * 252.0
                if len(returns) >= 1
                else pd.Series(0.0, index=assets)
            )

        state = self.regime_detector.detect(data, refit=True)
        self.last_regime_state_ = state
        p_t = state.current_probs.to_numpy(dtype=np.float64)
        k_reg = len(p_t)
        n_allocs = len(self.regime_allocators)

        # Align probability vector length with regime_allocators length
        if k_reg != n_allocs:
            p_interp = np.interp(
                np.linspace(0.0, 1.0, n_allocs),
                np.linspace(0.0, 1.0, k_reg),
                p_t,
            )
            p_t = p_interp / max(float(np.sum(p_interp)), 1e-12)
            k_reg = n_allocs

        if self.mode == "hard_switch" or self.mode == "partitioned_hmm":
            active_k = int(np.argmax(p_t))
            p_eff = np.zeros(k_reg, dtype=np.float64)
            p_eff[active_k] = 1.0
        else:
            p_eff = p_t

        w_blended = np.zeros(n_assets, dtype=np.float64)
        for k, alloc in enumerate(self.regime_allocators):
            if p_eff[k] <= 1e-6:
                continue

            if self.mode == "partitioned_hmm" and len(returns) >= 5:
                # Reproduces partitioned-window covariance starvation when explicitly requested
                hard_ser = state.hard_state.reindex(returns.index).fillna(0)
                sub_ret = returns[hard_ser == min(k, int(hard_ser.max()))]
                if len(sub_ret) >= 2:
                    sub_cov = sub_ret.cov().fillna(0.0) * 252.0 + 1e-4 * np.eye(
                        n_assets
                    )
                    sub_mu = sub_ret.mean().fillna(0.0) * 252.0
                else:
                    sub_cov = cov_matrix
                    sub_mu = expected_returns
            else:
                sub_cov = cov_matrix
                sub_mu = expected_returns

            try:
                res_k = alloc.allocate(
                    data,
                    cov_matrix=sub_cov,
                    expected_returns=sub_mu,
                    **kwargs,
                )
                w_k = (
                    res_k.weights.reindex(assets).fillna(0.0).to_numpy(dtype=np.float64)
                )
            except (ValueError, KeyError, RuntimeError, TypeError):
                w_k = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

            if k == k_reg - 1 and self.bear_defensive_cash_weight > 0.0:
                w_k = w_k * (1.0 - self.bear_defensive_cash_weight)

            w_blended += p_eff[k] * w_k

        w_sum = float(np.sum(np.abs(w_blended)))
        if w_sum > 1.0 + 1e-6:
            w_blended = w_blended / w_sum

        return OptimizationResult(
            name="RegimeSwitchingAllocator",
            weights=pd.Series(w_blended, index=assets, name="weights"),
            status="OPTIMAL_REGIME_SWITCHING",
            metadata={
                "regime_probs": p_t.tolist(),
                "active_regime": int(np.argmax(p_t)),
                "regime_confidence": state.current_confidence,
                "mode": self.mode,
                "detector_metadata": state.metadata,
            },
        )
