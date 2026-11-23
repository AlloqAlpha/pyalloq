from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.interfaces import (
    BaseAllocator,
    BaseCovarianceEstimator,
    BaseReturnEstimator,
)
from pyalloq_core.meta import (
    AdaHedgeStrategy,
    AdaNormalHedgeStrategy,
    BaseMetaWeightingStrategy,
    BPDSMetaStrategy,
    CADTSBanditStrategy,
    CBCEStrategy,
    DualScaledSwitchingCostStrategy,
    EllipticalShrinkageStrategy,
    FixedShareStrategy,
    compute_optimal_universe_size,
    estimate_elliptical_kurtosis_kappa,
)
from pyalloq_core.results import OptimizationResult

from pyalloq.classical.estimators.covariance.empirical import EmpiricalCovariance
from pyalloq.classical.estimators.returns.classical.ewma import EWMAReturnEstimator
from pyalloq.classical.optimizers.multi_period import (
    _project_capped_simplex,
    _project_l1_ball,
)
from pyalloq.classical.optimizers.naive import EqualWeightAllocator

MetaStrategyName = Literal[
    "dual_scaled_olo",
    "weighted_fractional",
    "fixed_share",
    "cbce",
    "adanormalhedge",
    "adahedge",
    "bpds",
    "elliptical_shrinkage",
    "cadts",
    "inverse_volatility",
    "equal",
]


class EnsembleAllocator(BaseAllocator):
    """
    Meta-Allocation & Portfolio Blending Engine (`EnsembleAllocator`).

    Combines `K` child `BaseAllocator` instances (e.g., `HRPAllocator`, `MeanCVaRAllocator`,
    `MarkowitzAllocator`, `EqualWeightAllocator`) via a pluggable `BaseMetaWeightingStrategy`
    with three institutional safeguards:

    1. **Elliptical Fat-Tail & 1/N Anchoring (Kan & Lassance, *JFQA* Dec 2025; Lassance et al., *JBES* 2024)**:
       Optionally appends `EqualWeightAllocator` (`1/N`) as a defensive anchor child and caps
       active universe cardinality at the finite-sample elliptical optimum `N^* < T / 2`.
    2. **Adaptive & Switching-Cost-Aware Online Meta-Weighting**:
       Supports `DualScaledSwitchingCostStrategy` (`"dual_scaled_olo"`), `FixedShareStrategy`
       (`"fixed_share"`), `CBCEStrategy` (`"cbce"`), `AdaNormalHedgeStrategy` (`"adanormalhedge"`),
       `BPDSMetaStrategy` (`"bpds"`), `EllipticalShrinkageStrategy` (`"elliptical_shrinkage"`),
       `CADTSBanditStrategy` (`"cadts"`), `AdaHedgeStrategy` (`"adahedge"`), `"inverse_volatility"`,
       or `"equal"`.
    3. **Pre-TCA Internal Trade Crossing & Top-Level Constraint Projection**:
       Computes `w_{net, t} = Pi_C(W_t alpha_t)` *before* `WalkForwardEngine` charges Almgren-Chriss
       transaction fees, netting opposing trades across child allocators and enforcing top-level
       gross leverage `||w||_1 <= max_gross_leverage` and box bounds `weight_bounds`.
    """

    def __init__(
        self,
        child_allocators: Sequence[BaseAllocator],
        meta_strategy: BaseMetaWeightingStrategy | MetaStrategyName = "dual_scaled_olo",
        include_1n_anchor: bool = True,
        optimal_cardinality_cap: bool = False,
        max_gross_leverage: float = 1.0,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
        enforce_full_investment: bool = True,
        evaluation_window: int = 60,
        switching_cost_bps: float = 10.0,
        returns_estimator: BaseReturnEstimator | None = None,
        cov_estimator: BaseCovarianceEstimator | None = None,
        tickers: list[str] | None = None,
    ) -> None:
        super().__init__(tickers or [])
        if len(child_allocators) < 1:
            raise ValueError("EnsembleAllocator requires at least 1 child allocator.")

        children = list(child_allocators)
        has_eq = any(isinstance(c, EqualWeightAllocator) for c in children)
        if include_1n_anchor and not has_eq:
            children.append(EqualWeightAllocator())

        self.child_allocators = children
        self.include_1n_anchor = bool(include_1n_anchor)
        self.optimal_cardinality_cap = bool(optimal_cardinality_cap)
        self.max_gross_leverage = float(max(0.05, max_gross_leverage))
        self.weight_bounds = weight_bounds
        self.enforce_full_investment = bool(enforce_full_investment)
        self.evaluation_window = max(5, int(evaluation_window))
        self.switching_cost_bps = float(max(0.0, switching_cost_bps))
        self.returns_estimator = returns_estimator or EWMAReturnEstimator(span=30)
        self.cov_estimator = cov_estimator or EmpiricalCovariance()

        self.meta_strategy_obj, self.meta_mode_name = self._resolve_meta_strategy(
            meta_strategy
        )
        self._last_meta_weights: np.ndarray | None = None
        self._prev_child_weights: np.ndarray | None = None
        self.last_internal_crossing_savings_: float = 0.0
        self.last_optimal_cardinality_: int | None = None

    @staticmethod
    def _resolve_meta_strategy(
        strategy: BaseMetaWeightingStrategy | MetaStrategyName,
    ) -> tuple[BaseMetaWeightingStrategy | None, str]:
        if isinstance(strategy, BaseMetaWeightingStrategy):
            return strategy, type(strategy).__name__
        name = str(strategy).lower()
        if name in ("dual_scaled_olo", "weighted_fractional"):
            return DualScaledSwitchingCostStrategy(), name
        if name == "fixed_share":
            return FixedShareStrategy(), name
        if name == "cbce":
            return CBCEStrategy(), name
        if name == "adanormalhedge":
            return AdaNormalHedgeStrategy(), name
        if name == "adahedge":
            return AdaHedgeStrategy(), name
        if name == "bpds":
            return BPDSMetaStrategy(), name
        if name == "elliptical_shrinkage":
            return EllipticalShrinkageStrategy(), name
        if name == "cadts":
            return CADTSBanditStrategy(), name
        if name in ("inverse_volatility", "equal"):
            return None, name
        raise ValueError(f"Unsupported meta_strategy: {strategy}")

    def _project_constraints(self, w_raw: np.ndarray) -> np.ndarray:
        """
        Projects raw blended weights `w_raw = W_t @ alpha_t` onto the feasible mandate set
        `Pi_C` enforcing box bounds `[low, high]`, gross leverage `||w||_1 <= L_max`,
        and full investment `sum(w) == 1` when `enforce_full_investment=True` and `low >= 0`.
        """
        low, high = float(self.weight_bounds[0]), float(self.weight_bounds[1])
        w = np.clip(w_raw, low, high)

        if self.enforce_full_investment and low >= 0.0:
            target_sum = min(1.0, self.max_gross_leverage)
            return _project_capped_simplex(w, low=low, high=high, target_sum=target_sum)

        l1_norm = float(np.sum(np.abs(w)))
        if l1_norm > self.max_gross_leverage and l1_norm > 1e-12:
            w = _project_l1_ball(w, radius=self.max_gross_leverage)
            w = np.clip(w, low, high)
        return w

    def allocate(
        self,
        data: MarketData,
        cov_matrix: pd.DataFrame | None = None,
        expected_returns: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        all_assets = (
            list(cov_matrix.columns)
            if cov_matrix is not None
            else list(data.prices.columns)
        )
        n_total = len(all_assets)
        returns_df = data.prices.reindex(columns=all_assets).pct_change().dropna()
        ret_np = returns_df.to_numpy(dtype=np.float64)

        cov_df = (
            cov_matrix if cov_matrix is not None else self.cov_estimator.estimate(data)
        )
        mu_ser = (
            expected_returns
            if expected_returns is not None
            else self.returns_estimator.estimate(data)
        )

        # Optional JFQA Dec 2025 finite-sample optimal universe cardinality cap N^*
        if self.optimal_cardinality_cap and len(ret_np) >= 6 and n_total > 2:
            kappa_est = estimate_elliptical_kurtosis_kappa(ret_np)
            corr_mat = returns_df.corr().fillna(0.0).to_numpy(dtype=np.float64)
            off_diag = corr_mat[~np.eye(n_total, dtype=bool)]
            avg_corr = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.25
            n_star = compute_optimal_universe_size(
                n_obs=len(ret_np),
                n_available_assets=n_total,
                kurtosis_kappa=kappa_est,
                avg_correlation=avg_corr,
            )
            self.last_optimal_cardinality_ = n_star
            # Select top N^* assets by trailing risk-adjusted Sharpe / inverse volatility
            asset_vols = np.maximum(np.std(ret_np, axis=0, ddof=1), 1e-6)
            asset_mus = (
                mu_ser.reindex(all_assets).fillna(0.0).to_numpy(dtype=np.float64)
            )
            asset_scores = asset_mus / (asset_vols * np.sqrt(252.0))
            active_idx = np.argsort(asset_scores)[::-1][:n_star]
            active_assets = [all_assets[i] for i in sorted(active_idx)]
        else:
            active_assets = all_assets
            self.last_optimal_cardinality_ = n_total

        sub_data = (
            MarketData(
                prices=data.prices[active_assets],
                features=data.features,
                cross_sectional=data.cross_sectional,
                risk_free_rate=data.risk_free_rate,
                risk_aversion=data.risk_aversion,
            )
            if len(active_assets) < n_total
            else data
        )
        sub_cov = cov_df.reindex(index=active_assets, columns=active_assets)
        sub_mu = mu_ser.reindex(active_assets)
        sub_curr = (
            current_weights.reindex(active_assets)
            if current_weights is not None
            else None
        )

        k_children = len(self.child_allocators)
        n_act = len(active_assets)
        child_w_mat = np.zeros((k_children, n_act), dtype=np.float64)
        child_names: list[str] = []
        anchor_idx = k_children - 1

        for k, child in enumerate(self.child_allocators):
            child_names.append(f"{child.__class__.__name__}_{k}")
            if isinstance(child, EqualWeightAllocator):
                anchor_idx = k
            try:
                res_k = child.allocate(
                    sub_data,
                    cov_matrix=sub_cov,
                    expected_returns=sub_mu,
                    current_weights=sub_curr,
                )
                w_k = (
                    res_k.weights.reindex(active_assets)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
                # Normalize gross leverage mismatches across long-only & dollar-neutral children
                gross_k = float(np.sum(np.abs(w_k)))
                if gross_k > self.max_gross_leverage and gross_k > 1e-12:
                    w_k = w_k * (self.max_gross_leverage / gross_k)
            except (ValueError, KeyError, RuntimeError, TypeError):
                w_k = np.full(n_act, 1.0 / max(1, n_act), dtype=np.float64)
            child_w_mat[k] = w_k

        # Evaluate historical hypothetical net return matrix (T_win, K) across children
        sub_ret_np = returns_df[active_assets].to_numpy(dtype=np.float64)
        w_eval = min(len(sub_ret_np), self.evaluation_window)
        if w_eval >= 2:
            win_rets = sub_ret_np[-w_eval:]  # (w_eval, n_act)
            expert_rets = win_rets @ child_w_mat.T  # (w_eval, K)
            if (
                self._prev_child_weights is not None
                and self._prev_child_weights.shape == child_w_mat.shape
            ):
                child_turnover = np.sum(
                    np.abs(child_w_mat - self._prev_child_weights), axis=1
                )
                cost_drag = (self.switching_cost_bps / 10000.0) * child_turnover
                expert_rets[-1] = expert_rets[-1] - cost_drag
        else:
            expert_rets = np.zeros((2, k_children), dtype=np.float64)

        # Compute meta-blending weights alpha_t in Delta^K
        if self.meta_mode_name == "equal" or k_children == 1:
            alpha_t = np.full(k_children, 1.0 / k_children, dtype=np.float64)
        elif self.meta_mode_name == "inverse_volatility":
            vols = np.maximum(np.std(expert_rets, axis=0, ddof=1), 1e-6)
            inv_v = 1.0 / vols
            alpha_t = inv_v / np.sum(inv_v)
        else:
            assert self.meta_strategy_obj is not None
            alpha_t = self.meta_strategy_obj.compute_meta_weights(
                expert_returns=expert_rets,
                current_meta_weights=self._last_meta_weights,
                asset_returns=sub_ret_np[-w_eval:] if w_eval >= 2 else sub_ret_np,
                baseline_index=anchor_idx,
                anchor_index=anchor_idx,
            )

        self._last_meta_weights = alpha_t.copy()
        self._prev_child_weights = child_w_mat.copy()

        # Pre-TCA Internal Trade Crossing Audit:
        # Un-netted sum of child turnovers vs. netted ensemble turnover
        if sub_curr is not None:
            w_prev_act = sub_curr.fillna(0.0).to_numpy(dtype=np.float64)
        else:
            w_prev_act = np.zeros(n_act, dtype=np.float64)

        unnetted_turnover = float(
            np.sum(alpha_t[:, None] * np.abs(child_w_mat - w_prev_act[None, :]))
        )
        w_blended_act = alpha_t @ child_w_mat
        w_projected_act = self._project_constraints(w_blended_act)
        netted_turnover = float(np.sum(np.abs(w_projected_act - w_prev_act)))
        self.last_internal_crossing_savings_ = max(
            0.0, unnetted_turnover - netted_turnover
        )

        w_full = pd.Series(0.0, index=all_assets, name="weights")
        w_full.loc[active_assets] = w_projected_act

        cov_full_np = (
            cov_df.reindex(index=all_assets, columns=all_assets)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        mu_full_np = mu_ser.reindex(all_assets).fillna(0.0).to_numpy(dtype=np.float64)
        w_full_np = w_full.to_numpy(dtype=np.float64)
        exp_ret = float(mu_full_np @ w_full_np)
        exp_vol = float(np.sqrt(max(0.0, float(w_full_np @ cov_full_np @ w_full_np))))
        sharpe = exp_ret / exp_vol if exp_vol > 1e-12 else 0.0

        meta_map = {
            name: float(a) for name, a in zip(child_names, alpha_t, strict=False)
        }
        return OptimizationResult(
            name=f"EnsembleAllocator_{self.meta_mode_name}",
            weights=w_full,
            status="OPTIMAL_ENSEMBLE",
            expected_return=exp_ret,
            volatility=exp_vol,
            sharpe_ratio=sharpe,
            metadata={
                "meta_strategy": self.meta_mode_name,
                "meta_weights": meta_map,
                "internal_crossing_turnover_saved": self.last_internal_crossing_savings_,
                "unnetted_turnover": unnetted_turnover,
                "netted_turnover": netted_turnover,
                "optimal_cardinality": self.last_optimal_cardinality_,
                "gross_leverage": float(np.sum(np.abs(w_full_np))),
            },
        )
