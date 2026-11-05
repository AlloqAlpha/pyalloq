from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.pipeline import StrategyPipeline
from pyalloq_core.results import BacktestResult, MonteCarloResult

from pyalloq_backtest.attribution import AttributionResult, PerformanceAttributor
from pyalloq_backtest.costs import BaseCostModel, FlatBpsCostModel
from pyalloq_backtest.metrics import (
    MetricsTearSheet,
    compute_deflated_sharpe_ratio,
    compute_expected_max_sharpe,
)
from pyalloq_backtest.splitters import BaseWindowSplitter, RollingWindowSplitter

if TYPE_CHECKING:
    from pyalloq_backtest.engine import WalkForwardEngine


@dataclass(frozen=True)
class BacktestTrialRecord:
    """Immutable record of a single executed backtest trial tracked by `BacktestTracker`."""

    trial_id: int
    strategy_name: str
    n_observations: int
    annualized_sharpe: float
    daily_sharpe: float
    skewness: float
    kurtosis: float
    timestamp: str


class BacktestTracker:
    """
    Thread-safe singleton ledger tracking all backtest trials executed across `WalkForwardEngine`
    to guard against the Multiple Testing Problem and Selection Bias (Bailey & Lopez de Prado, 2014).

    Automatically increments the trial counter `N` and records each trial's Sharpe ratio, track
    record length, skewness, and kurtosis so `Var[{SR_n}]`, `E[max_n SR_n]`, and the Deflated
    Sharpe Ratio (DSR) can be evaluated rigorously.
    """

    _instance: BacktestTracker | None = None
    _instance_lock: threading.Lock = threading.Lock()
    _lock: threading.RLock
    _trials: list[BacktestTrialRecord]
    _enabled: bool

    def __new__(cls) -> BacktestTracker:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._lock = threading.RLock()
                    inst._trials = []
                    inst._enabled = True
                    cls._instance = inst
        return cls._instance

    @classmethod
    def instance(cls) -> BacktestTracker:
        """Returns the global `BacktestTracker` singleton instance."""
        return cls()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = bool(value)

    @property
    def n_trials(self) -> int:
        """Total number of backtest trials recorded in the active session."""
        with self._lock:
            return len(self._trials)

    @property
    def trials(self) -> list[BacktestTrialRecord]:
        """Snapshot list of all recorded `BacktestTrialRecord` entries."""
        with self._lock:
            return list(self._trials)

    @property
    def trial_sharpes(self) -> list[float]:
        """List of annualized Sharpe ratios across all recorded trials."""
        with self._lock:
            return [t.annualized_sharpe for t in self._trials]

    def reset(self) -> None:
        """Clears all recorded trials and resets `N = 0`."""
        with self._lock:
            self._trials.clear()

    @classmethod
    @contextmanager
    def session(
        cls, reset: bool = True, restore_on_exit: bool = False
    ) -> Iterator[BacktestTracker]:
        """
        Context manager for scoped multi-strategy benchmarking experiments.
        If `reset=True`, clears the trial ledger on entry.
        If `restore_on_exit=True`, restores the previous trial ledger upon exiting the block.
        """
        tracker = cls.instance()
        with tracker._lock:
            saved_trials = list(tracker._trials)
            saved_enabled = tracker._enabled
            if reset:
                tracker._trials.clear()
        try:
            yield tracker
        finally:
            if restore_on_exit:
                with tracker._lock:
                    tracker._trials = saved_trials
                    tracker._enabled = saved_enabled

    def record(
        self,
        strategy_name: str,
        returns: pd.Series,
        risk_free_rate: float | pd.Series = 0.0,
        ann_factor: float = 252.0,
    ) -> BacktestTrialRecord:
        """
        Records an executed backtest return series into the singleton ledger and increments `N`.
        """
        clean = returns.dropna()
        n_obs = len(clean)

        if isinstance(risk_free_rate, pd.Series):
            aligned_rf = risk_free_rate.reindex(clean.index).ffill().fillna(0.0)
            rf_annual = float(aligned_rf.mean()) if len(aligned_rf) > 0 else 0.0
        else:
            rf_annual = float(risk_free_rate)

        if n_obs > 1:
            vals = clean.to_numpy(dtype=np.float64)
            cum_ret = float(np.prod(1.0 + vals) - 1.0)
            n_years = max(n_obs / ann_factor, 1e-4)
            ann_ret = (
                float((1.0 + cum_ret) ** (1.0 / n_years) - 1.0)
                if (1.0 + cum_ret) > 0.0
                else -1.0
            )
            ann_vol = float(np.std(vals, ddof=1) * np.sqrt(ann_factor))
            ann_sr = (ann_ret - rf_annual) / ann_vol if ann_vol > 1e-12 else 0.0
            daily_sr = ann_sr / float(np.sqrt(ann_factor))
            skew_val = float(cast(Any, clean.skew())) if n_obs > 2 else 0.0
            kurt_val = float(cast(Any, clean.kurtosis())) if n_obs > 3 else 0.0
            skew_val = 0.0 if np.isnan(skew_val) else skew_val
            kurt_val = 0.0 if np.isnan(kurt_val) else kurt_val
        else:
            ann_sr = 0.0
            daily_sr = 0.0
            skew_val = 0.0
            kurt_val = 0.0

        with self._lock:
            trial_id = len(self._trials) + 1
            rec = BacktestTrialRecord(
                trial_id=trial_id,
                strategy_name=strategy_name,
                n_observations=n_obs,
                annualized_sharpe=float(ann_sr),
                daily_sharpe=float(daily_sr),
                skewness=float(skew_val),
                kurtosis=float(kurt_val),
                timestamp=datetime.now(tz=UTC).isoformat(),
            )
            if self._enabled:
                self._trials.append(rec)
            return rec

    def sharpe_variance(self, annualized: bool = True) -> float:
        """
        Computes the empirical variance Var[{SR_n}] across all recorded trials.
        """
        with self._lock:
            if len(self._trials) < 2:
                return 0.0
            vals = np.array(
                [
                    t.annualized_sharpe if annualized else t.daily_sharpe
                    for t in self._trials
                ],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                return 0.0
            return float(np.var(vals, ddof=1))

    def expected_max_sharpe(
        self,
        n_trials: int | None = None,
        sharpe_std: float | None = None,
        mean_sharpe: float = 0.0,
        annualized: bool = True,
    ) -> float:
        """
        Computes E[max_n SR_n] under the False Strategy Theorem across `n_trials` (defaults to `self.n_trials`).
        """
        with self._lock:
            eff_n = n_trials if n_trials is not None else max(len(self._trials), 1)
            if sharpe_std is None:
                var_sr = self.sharpe_variance(annualized=annualized)
                sharpe_std = float(np.sqrt(max(var_sr, 0.0)))
                if sharpe_std <= 1e-12 and len(self._trials) > 0:
                    avg_t = max(
                        int(np.mean([t.n_observations for t in self._trials])), 2
                    )
                    se_daily = 1.0 / float(np.sqrt(avg_t - 1))
                    sharpe_std = (
                        se_daily * float(np.sqrt(252.0)) if annualized else se_daily
                    )
        return compute_expected_max_sharpe(
            n_trials=eff_n, sharpe_std=sharpe_std, mean_sharpe=mean_sharpe
        )

    def compute_dsr(
        self,
        source: BacktestResult | pd.Series | np.ndarray,
        risk_free_rate: float | pd.Series = 0.0,
        target_sharpe: float = 0.0,
        n_trials: int | None = None,
    ) -> tuple[float, float]:
        """
        Evaluates `(deflated_sharpe_ratio, expected_max_sharpe_ratio)` for a strategy using
        the tracker's accumulated trial count `N` and empirical trial Sharpe distribution.
        """
        returns = source.returns if isinstance(source, BacktestResult) else source
        with self._lock:
            eff_n = n_trials if n_trials is not None else max(len(self._trials), 1)
            sharpes = [t.annualized_sharpe for t in self._trials]
        return compute_deflated_sharpe_ratio(
            portfolio_returns=returns,
            n_trials=eff_n,
            trial_sharpes=sharpes if len(sharpes) >= 2 else None,
            risk_free_rate=risk_free_rate,
            target_sharpe=target_sharpe,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Exports the complete session ledger of executed backtest trials as a DataFrame."""
        with self._lock:
            if not self._trials:
                return pd.DataFrame(
                    columns=[
                        "trial_id",
                        "strategy_name",
                        "n_observations",
                        "annualized_sharpe",
                        "daily_sharpe",
                        "skewness",
                        "kurtosis",
                        "timestamp",
                    ]
                )
            return pd.DataFrame([t.__dict__ for t in self._trials]).set_index(
                "trial_id"
            )


@dataclass
class ComparisonResult:
    """
    Container output of `MultiStrategyComparator.run(...)` (`BenchmarkSuite`).
    Provides:
      - `summary_table`: Strategy-by-metric DataFrame including raw metrics, `deflated_sharpe_ratio`,
        `expected_max_sharpe_ratio`, `n_trials`, and attribution highlights.
      - `results`: Dict mapping strategy name -> `BacktestResult`.
      - `scenario_results`: Optional dict mapping strategy name -> `MonteCarloResult` across synthetic paths.
      - `attributions`: Dict mapping strategy name -> `AttributionResult` (Brinson-Fachler, Factor, Frictional Drag).
      - `n_trials`: Total backtest trials accounted for in the DSR multiple-testing penalty.
    """

    summary_table: pd.DataFrame
    results: dict[str, BacktestResult]
    attributions: dict[str, AttributionResult] = field(default_factory=dict)
    scenario_results: dict[str, MonteCarloResult] | None = None
    n_trials: int = 1
    expected_max_sharpe: float = 0.0


class MultiStrategyComparator:
    """
    Concurrent multi-strategy walk-forward benchmark suite with Deflated Sharpe Ratio (DSR)
    multiple-testing correction and automated Brinson-Fachler / Factor / Frictional attribution.
    """

    def __init__(
        self,
        pipelines: Mapping[str, StrategyPipeline | WalkForwardEngine],
        splitter: BaseWindowSplitter | None = None,
        cost_model: BaseCostModel | None = None,
        rebalance_freq: str = "ME",
        drift_weights: bool = True,
        rebalance_threshold: float | None = None,
        short_borrow_bps: float = 0.0,
        max_gross_leverage: float | None = None,
        benchmark: str | pd.Series | BacktestResult | None = None,
        sector_map: Mapping[str, str] | pd.Series | pd.DataFrame | None = None,
        factor_returns: pd.DataFrame | None = None,
        max_workers: int | None = None,
        reset_tracker: bool = False,
    ) -> None:
        if not pipelines:
            raise ValueError(
                "`MultiStrategyComparator` requires at least one pipeline or engine."
            )
        self.pipelines = dict(pipelines)
        self.splitter = splitter or RollingWindowSplitter(lookback_window=252)
        self.cost_model = cost_model or FlatBpsCostModel(bps=10.0)
        self.rebalance_freq = rebalance_freq
        self.drift_weights = drift_weights
        self.rebalance_threshold = rebalance_threshold
        self.short_borrow_bps = short_borrow_bps
        self.max_gross_leverage = max_gross_leverage
        self.benchmark = benchmark
        self.sector_map = sector_map
        self.factor_returns = factor_returns
        self.max_workers = max_workers
        self.reset_tracker = reset_tracker
        self.last_result: ComparisonResult | None = None

    def _build_engine(
        self, item: StrategyPipeline | WalkForwardEngine
    ) -> WalkForwardEngine:
        from pyalloq_backtest.engine import WalkForwardEngine

        if isinstance(item, WalkForwardEngine):
            return item
        return WalkForwardEngine(
            pipeline=item,
            splitter=self.splitter,
            cost_model=self.cost_model,
            rebalance_freq=self.rebalance_freq,
            drift_weights=self.drift_weights,
            rebalance_threshold=self.rebalance_threshold,
            short_borrow_bps=self.short_borrow_bps,
            max_gross_leverage=self.max_gross_leverage,
            benchmark_returns=self.benchmark
            if isinstance(self.benchmark, pd.Series)
            else (
                self.benchmark.returns
                if isinstance(self.benchmark, BacktestResult)
                else None
            ),
        )

    @staticmethod
    def _evaluate_on_scenarios(
        bt_res: BacktestResult,
        scenario_data: ScenarioMarketData,
        risk_free_rate: float = 0.0,
    ) -> MonteCarloResult:
        """
        Evaluates a strategy's terminal/average allocation weights across all M paths of
        `ScenarioMarketData` to produce a `MonteCarloResult` for multi-strategy fan charts.
        """
        valid_w = bt_res.weights.dropna(how="all")
        if len(valid_w) > 0:
            w_ser = valid_w.iloc[-1]
        else:
            n = len(scenario_data.asset_names)
            w_ser = pd.Series(1.0 / max(1, n), index=scenario_data.asset_names)

        w_arr = (
            w_ser.reindex(scenario_data.asset_names)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        w_sum = float(np.sum(np.abs(w_arr)))
        if w_sum > 1e-12:
            w_arr = w_arr / w_sum

        # scenario_data.returns has shape (M, T, N)
        path_rets_m_t = np.tensordot(
            scenario_data.returns.astype(np.float64), w_arr, axes=([2], [0])
        )  # (M, T)
        path_eq_m_t = np.cumprod(1.0 + path_rets_m_t, axis=1)

        cols = [f"Path_{m}" for m in range(scenario_data.n_paths)]
        returns_df = pd.DataFrame(
            path_rets_m_t.T, index=scenario_data.timestamps, columns=cols
        )
        equity_df = pd.DataFrame(
            path_eq_m_t.T, index=scenario_data.timestamps, columns=cols
        )

        ann_factor = 252.0
        means = returns_df.mean(axis=0) * ann_factor
        stds = returns_df.std(axis=0) * np.sqrt(ann_factor)
        sharpes = ((means - risk_free_rate) / stds.replace(0.0, np.nan)).fillna(0.0)
        tot_rets = equity_df.iloc[-1] - 1.0
        running_max = equity_df.cummax(axis=0).replace(0.0, np.nan)
        mdds = ((equity_df - running_max) / running_max).min(axis=0).fillna(0.0)

        metrics_dist = pd.DataFrame(
            {
                "total_return": tot_rets,
                "annualized_return": means,
                "annualized_volatility": stds,
                "sharpe_ratio": sharpes,
                "maximum_drawdown": mdds,
            },
            index=cols,
        )
        return MonteCarloResult(
            returns_matrix=returns_df,
            equity_curves=equity_df,
            metrics_distribution=metrics_dist,
        )

    def run(
        self,
        data: MarketData,
        scenario_data: ScenarioMarketData | None = None,
    ) -> ComparisonResult:
        """
        Executes all configured strategy pipelines concurrently over `data`, updates the
        `BacktestTracker` singleton, computes the rigorous Deflated Sharpe Ratio (DSR) across
        all trials, runs 3-pillar performance attribution, and returns a `ComparisonResult`.
        """
        tracker = BacktestTracker.instance()
        if self.reset_tracker:
            tracker.reset()

        names = list(self.pipelines.keys())
        engines = {name: self._build_engine(self.pipelines[name]) for name in names}

        def _exec_one(name: str) -> tuple[str, BacktestResult]:
            res = engines[name].run(data)
            res.name = name
            return name, res

        n_workers = self.max_workers or min(len(names), 8)
        raw_results: dict[str, BacktestResult] = {}
        if n_workers <= 1 or len(names) == 1:
            for name in names:
                _, r = _exec_one(name)
                raw_results[name] = r
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_exec_one, name) for name in names]
                for fut in futures:
                    k, r = fut.result()
                    raw_results[k] = r

        # Preserve user dictionary order
        ordered_results: dict[str, BacktestResult] = {
            name: raw_results[name] for name in names
        }

        # Resolve benchmark returns & weights for cross-strategy comparison & attribution
        bench_result: BacktestResult | None = None
        bench_returns: pd.Series | None = None
        if isinstance(self.benchmark, str) and self.benchmark in ordered_results:
            bench_result = ordered_results[self.benchmark]
            bench_returns = bench_result.returns
        elif isinstance(self.benchmark, BacktestResult):
            bench_result = self.benchmark
            bench_returns = bench_result.returns
        elif isinstance(self.benchmark, pd.Series):
            bench_returns = self.benchmark
        else:
            # Default benchmark: Equal-Weight daily return across assets in `data`
            asset_rets = data.prices.pct_change().dropna()
            bench_returns = asset_rets.mean(axis=1)

        # Synchronize DSR across all strategies using post-batch tracker state (N & trial_sharpes)
        total_n_trials = max(tracker.n_trials, len(ordered_results))
        all_trial_sharpes = tracker.trial_sharpes
        if len(all_trial_sharpes) < len(ordered_results):
            all_trial_sharpes = [
                float(
                    cast_val(r.tear_sheet.loc["sharpe_ratio", "Value"])
                    if "sharpe_ratio" in r.tear_sheet.index
                    else 0.0
                )
                for r in ordered_results.values()
            ]

        attributor = PerformanceAttributor()
        attributions: dict[str, AttributionResult] = {}
        summary_rows: dict[str, dict[str, float]] = {}

        for name, bt_res in ordered_results.items():
            updated_ts = MetricsTearSheet.generate(
                portfolio_returns=bt_res.returns,
                risk_free_rate=data.risk_free_rate,
                weights=bt_res.weights.reindex(bt_res.returns.index),
                turnover=bt_res.turnover.reindex(bt_res.returns.index)
                if bt_res.turnover is not None
                else None,
                benchmark_returns=bench_returns,
                n_trials=total_n_trials,
                trial_sharpes=all_trial_sharpes,
            )
            bt_res.tear_sheet = updated_ts

            attr_res = attributor.attribute(
                backtest_result=bt_res,
                market_data=data,
                benchmark_weights=bench_result,
                benchmark_returns=bench_returns,
                sector_map=self.sector_map,
                factor_returns=self.factor_returns,
            )
            attributions[name] = attr_res

            row_dict = {
                str(idx): float(cast_val(updated_ts.loc[idx, "Value"]))
                for idx in updated_ts.index
            }
            row_dict["n_trials"] = float(total_n_trials)
            row_dict["brinson_allocation"] = attr_res.brinson.total_allocation_effect
            row_dict["brinson_selection"] = attr_res.brinson.total_selection_effect
            row_dict["brinson_interaction"] = attr_res.brinson.total_interaction_effect
            row_dict["factor_alpha"] = attr_res.factor.annualized_alpha
            row_dict["factor_r_squared"] = attr_res.factor.r_squared
            row_dict["total_cost_drag"] = float(
                attr_res.frictional.waterfall_summary["total_explicit_cost_drag"]
            )
            row_dict["alpha_decay_drift_drag"] = float(
                attr_res.frictional.waterfall_summary["alpha_decay_and_drift_drag"]
            )
            summary_rows[name] = row_dict

        summary_df = pd.DataFrame.from_dict(summary_rows, orient="index")
        summary_df.index.name = "Strategy"

        scenario_results: dict[str, MonteCarloResult] | None = None
        if scenario_data is not None:
            rf_scalar = (
                float(data.risk_free_rate.mean())
                if isinstance(data.risk_free_rate, pd.Series)
                else float(data.risk_free_rate)
            )
            scenario_results = {
                name: self._evaluate_on_scenarios(
                    bt_res, scenario_data, risk_free_rate=rf_scalar
                )
                for name, bt_res in ordered_results.items()
            }

        sr0_val = (
            float(summary_df["expected_max_sharpe_ratio"].iloc[0])
            if "expected_max_sharpe_ratio" in summary_df.columns and len(summary_df) > 0
            else 0.0
        )

        comp_res = ComparisonResult(
            summary_table=summary_df,
            results=ordered_results,
            attributions=attributions,
            scenario_results=scenario_results,
            n_trials=total_n_trials,
            expected_max_sharpe=sr0_val,
        )
        self.last_result = comp_res
        return comp_res


def cast_val(v: Any) -> float:
    return float(v)


BenchmarkSuite = MultiStrategyComparator
