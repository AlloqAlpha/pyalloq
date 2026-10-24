import numpy as np
from dataclasses import dataclass

from pyalloq_core.data import ScenarioMarketData
from pyalloq_simulation.views.pooler import EntropyResult


@dataclass
class EvaluationMetrics:
    expected_return: float
    volatility: float
    sharpe_ratio: float
    cvar_95: float
    view_adjusted_mdd: float = 0.0
    probability_of_ruin: float = 0.0


class ScenarioEvaluator:
    def __init__(
        self, risk_free_rate: float = 0.0, ruin_threshold: float = -0.30
    ) -> None:
        self.rf = risk_free_rate
        # Normalize ruin threshold to a negative drawdown fraction (e.g., -0.30 for a 30% drop)
        self.ruin_threshold = -abs(ruin_threshold)

    def evaluate(
        self,
        scenario_data: ScenarioMarketData,
        asset_weights: np.ndarray,
        entropy_result: EntropyResult,
        ruin_threshold: float | None = None,
    ) -> EvaluationMetrics:
        """
        Calculates probability-weighted terminal and path-dependent portfolio metrics
        across the simulation horizon T.
        """
        weights = np.asarray(asset_weights, dtype=np.float64)
        probs = np.asarray(entropy_result.weights, dtype=np.float64)  # Shape: (M,)

        cumulative_asset_returns = scenario_data.get_cumulative_returns().astype(
            np.float64
        )  # Shape: (M, N)
        portfolio_returns = np.dot(cumulative_asset_returns, weights)

        expected_return = float(np.dot(portfolio_returns, probs))

        variance = float(np.dot(probs, (portfolio_returns - expected_return) ** 2))
        volatility = float(np.sqrt(max(variance, 0.0)))
        sharpe = (expected_return - self.rf) / volatility if volatility > 0 else 0.0

        sort_idx = np.argsort(portfolio_returns)
        sorted_rets = portfolio_returns[sort_idx]
        sorted_probs = probs[sort_idx]

        cum_probs = np.cumsum(sorted_probs)
        alpha = 0.05
        tail_idx = np.where(cum_probs <= alpha)[0]

        if len(tail_idx) == 0:
            # Fallback if the very first scenario carries > 5% probability mass
            cvar_95 = float(sorted_rets[0])
        else:
            # Normalize the tail probabilities so they sum to 1.0
            tail_probs = sorted_probs[tail_idx] / np.sum(sorted_probs[tail_idx])
            cvar_95 = float(np.dot(sorted_rets[tail_idx], tail_probs))

        # Path-dependent intra-horizon metrics along axis=1 (time steps 0..T)
        M = scenario_data.prices.shape[0]
        if scenario_data.initial_prices is not None:
            init_p = np.asarray(scenario_data.initial_prices, dtype=np.float64)[
                None, None, :
            ]
            rel_prices = scenario_data.prices.astype(np.float64) / init_p
            # Portfolio wealth at each step t=1..T: (M, T)
            step_wealth: np.ndarray = np.tensordot(rel_prices, weights, axes=([2], [0]))
        else:
            step_port_rets = np.tensordot(
                scenario_data.returns.astype(np.float64), weights, axes=([2], [0])
            )
            step_wealth = np.cumprod(1.0 + step_port_rets, axis=1).astype(np.float64)

        # Prepend t=0 initial portfolio equity = 1.0 -> shape (M, T + 1)
        wealth_curve = np.concatenate(
            [np.ones((M, 1), dtype=np.float64), step_wealth], axis=1
        )
        running_peaks = np.maximum.accumulate(wealth_curve, axis=1)
        drawdowns = (wealth_curve - running_peaks) / np.maximum(running_peaks, 1e-12)
        path_mdds = np.min(drawdowns, axis=1)  # Shape: (M,), values <= 0.0

        view_adjusted_mdd = float(np.dot(probs, path_mdds))

        eff_threshold = (
            -abs(ruin_threshold) if ruin_threshold is not None else self.ruin_threshold
        )
        # Ruin occurs if intra-horizon drawdown or equity drop breaches threshold
        min_equity_return = np.min(wealth_curve - 1.0, axis=1)
        ruin_mask = (path_mdds <= eff_threshold) | (min_equity_return <= eff_threshold)
        probability_of_ruin = float(np.dot(probs, ruin_mask.astype(np.float64)))

        return EvaluationMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            cvar_95=cvar_95,
            view_adjusted_mdd=view_adjusted_mdd,
            probability_of_ruin=probability_of_ruin,
        )
