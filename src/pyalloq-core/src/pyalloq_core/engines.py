from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import Any
from pyalloq_core.data import MarketData, ScenarioMarketData
from pyalloq_core.results import BacktestResult, MonteCarloResult
import pandas as pd


class MonteCarloEngine:
    """
    Distributes PyAlloq pipeline execution across synthetic M paths using CPU multiprocessing.
    This effectively turns a single backtest into a probability map of drawdowns and returns.
    """

    def __init__(
        self,
        backtest_engine: Any,
        max_workers: int | None = None,
        feature_pipeline: Any | None = None,
    ) -> None:
        self.backtest_engine = backtest_engine
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.feature_pipeline = feature_pipeline

    def _worker(
        self, path_idx: int, path_data: MarketData
    ) -> tuple[int, BacktestResult]:
        if self.feature_pipeline is not None and hasattr(self.feature_pipeline, "run"):
            path_data = self.feature_pipeline.run(path_data)
        result = self.backtest_engine.run(path_data)
        return path_idx, result

    def run(self, scenario_data: ScenarioMarketData) -> MonteCarloResult:
        return self.run_batches([scenario_data])

    def run_batches(
        self, scenario_batches: Iterable[ScenarioMarketData]
    ) -> MonteCarloResult:
        results_list: list[tuple[int, BacktestResult]] = []
        global_path_offset = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for batch in scenario_batches:
                futures_map = {}
                for m in range(batch.n_paths):
                    global_idx = global_path_offset + m
                    path_data = batch.get_path(path_idx=m)
                    future = executor.submit(self._worker, global_idx, path_data)
                    futures_map[future] = global_idx

                for future in as_completed(futures_map):
                    m_idx = futures_map[future]
                    try:
                        res = future.result()
                        results_list.append(res)
                    except Exception as e:
                        print(
                            f"[Warning] Backtest failed on Scenario Path {m_idx}: {e}"
                        )

                global_path_offset += batch.n_paths

        if not results_list:
            raise RuntimeError("All Monte Carlo scenario backtest paths failed.")

        results_list.sort(
            key=lambda x: x[0]
        )  # Ensure results are ordered by path index [0 ... M-1]

        all_returns = []
        all_metrics = []

        for m_idx, backtest_result in results_list:
            ret = backtest_result.returns.rename(f"Path_{m_idx}")
            all_returns.append(ret)
            metrics_row = backtest_result.tear_sheet["Value"].rename(f"Path_{m_idx}")
            all_metrics.append(metrics_row)

        returns_matrix = pd.concat(all_returns, axis=1)
        equity_matrix = (1.0 + returns_matrix).cumprod()
        metrics_distribution = pd.concat(all_metrics, axis=1).T

        return MonteCarloResult(
            returns_matrix=returns_matrix,
            equity_curves=equity_matrix,
            metrics_distribution=metrics_distribution,
        )


ParallelMonteCarloEngine = MonteCarloEngine
