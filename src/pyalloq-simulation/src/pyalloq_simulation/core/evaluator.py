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


class ScenarioEvaluator:
    def __init__(self, risk_free_rate: float = 0.0):
        self.rf = risk_free_rate

    def evaluate(
        self,
        scenario_data: ScenarioMarketData,
        asset_weights: np.ndarray,
        entropy_result: EntropyResult,
    ) -> EvaluationMetrics:
        """
        Calculates probability-weighted portfolio metrics at the end of the simulation horizon T.
        """
        # prices shape is (M, T, N). We compare T_final to T_initial.
        initial_prices = scenario_data.prices[:, 0, :]
        final_prices = scenario_data.prices[:, -1, :]
        cumulative_asset_returns = (
            final_prices - initial_prices
        ) / initial_prices  # Shape: (M, N)

        portfolio_returns = np.dot(cumulative_asset_returns, asset_weights)
        probs = entropy_result.weights  # Shape: (M,)

        expected_return = np.dot(portfolio_returns, probs)

        variance = np.dot(probs, (portfolio_returns - expected_return) ** 2)
        volatility = np.sqrt(variance)
        sharpe = (expected_return - self.rf) / volatility if volatility > 0 else 0.0

        sort_idx = np.argsort(portfolio_returns)
        sorted_rets = portfolio_returns[sort_idx]
        sorted_probs = probs[sort_idx]

        cum_probs = np.cumsum(sorted_probs)
        alpha = 0.05
        tail_idx = np.where(cum_probs <= alpha)[0]

        if len(tail_idx) == 0:
            # Fallback if the very first scenario carries > 5% probability mass
            cvar_95 = sorted_rets[0]
        else:
            # Normalize the tail probabilities so they sum to 1.0
            tail_probs = sorted_probs[tail_idx] / np.sum(sorted_probs[tail_idx])
            cvar_95 = np.dot(sorted_rets[tail_idx], tail_probs)

        return EvaluationMetrics(expected_return, volatility, sharpe, cvar_95)
