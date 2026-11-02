from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class BaseRewardFunction(ABC):
    """
    Abstract base class for pluggable Reinforcement Learning reward functions in `pyalloq[rl]`.
    """

    def reset(self) -> None:
        """Resets any internal episodic moving-average or return buffers."""
        return None

    @abstractmethod
    def compute(
        self,
        net_return: float,
        gross_return: float,
        turnover: float,
        transaction_cost: float,
        unrealized_drawdown: float = 0.0,
        w2_distance: float = 0.0,
    ) -> float:
        """
        Computes the scalar step reward given realized step portfolio mechanics.
        """
        ...


class DifferentialSharpeReward(BaseRewardFunction):
    """
    Moody & Saffell (2001) Online Differential Sharpe Ratio Reward:
        D_t = (B_{t-1} * dA_t - 0.5 * A_{t-1} * dB_t) / (B_{t-1} - A_{t-1}^2 + eps)^{3/2}
    where A_t and B_t are exponential moving averages of the 1st and 2nd moments
    of net portfolio returns with adaptation rate `eta`.
    """

    def __init__(
        self,
        eta: float = 0.05,
        drawdown_penalty: float = 0.1,
        w2_turnover_penalty: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        self.eta = eta
        self.drawdown_penalty = drawdown_penalty
        self.w2_turnover_penalty = w2_turnover_penalty
        self.eps = eps
        self._a: float = 0.0
        self._b: float = 1e-4
        self._step_count: int = 0

    def reset(self) -> None:
        self._a = 0.0
        self._b = 1e-4
        self._step_count = 0

    def compute(
        self,
        net_return: float,
        gross_return: float,
        turnover: float,
        transaction_cost: float,
        unrealized_drawdown: float = 0.0,
        w2_distance: float = 0.0,
    ) -> float:
        r = float(net_return)
        self._step_count += 1

        da = r - self._a
        db = (r**2) - self._b

        var_est = max(self._b - (self._a**2), self.eps)
        denom = var_est**1.5
        diff_sharpe = (self._b * da - 0.5 * self._a * db) / denom

        # Update moving moments
        self._a += self.eta * da
        self._b += self.eta * db

        dd_pen = self.drawdown_penalty * abs(min(0.0, float(unrealized_drawdown)))
        thermo_pen = self.w2_turnover_penalty * float(w2_distance) * float(turnover)

        # Clip extreme initial transient spikes for numerical stability in PPO/SAC
        reward = float(np.clip(diff_sharpe, -10.0, 10.0) - dd_pen - thermo_pen)
        return reward


class StepwiseSortinoReward(BaseRewardFunction):
    """
    Rolling Step-Wise Sortino Ratio Reward:
        Sortino_t = (r_{net, t} - MAR) / sqrt( E[ min(0, r_{net, tau} - MAR)^2 ] + floor^2 )
    Rewards upside net return while penalizing downside semi-deviation and unrealized drawdown.
    """

    def __init__(
        self,
        window: int = 20,
        mar: float = 0.0,
        downside_floor: float = 1e-3,
        drawdown_penalty: float = 0.2,
    ) -> None:
        self.window = max(2, window)
        self.mar = mar
        self.downside_floor = downside_floor
        self.drawdown_penalty = drawdown_penalty
        self._buffer: deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._buffer.clear()

    def compute(
        self,
        net_return: float,
        gross_return: float,
        turnover: float,
        transaction_cost: float,
        unrealized_drawdown: float = 0.0,
        w2_distance: float = 0.0,
    ) -> float:
        r = float(net_return)
        self._buffer.append(r)

        arr = np.asarray(self._buffer, dtype=np.float64)
        downside = np.minimum(0.0, arr - self.mar)
        downside_dev = float(
            np.sqrt(np.mean(np.square(downside)) + self.downside_floor**2)
        )

        step_sortino = (r - self.mar) / downside_dev
        dd_pen = self.drawdown_penalty * abs(min(0.0, float(unrealized_drawdown)))
        return float(np.clip(step_sortino, -10.0, 10.0) - dd_pen)


class CARAUtilReward(BaseRewardFunction):
    """
    Log-Wealth / Constant Absolute Risk Aversion (CARA) & Quadratic Variance Penalty Reward:
        r_t = log(1 + r_{net, t}) - (gamma / 2) * r_{net, t}^2
              - lambda_dd * |DD_t| - eta_{W2} * W_2(t) * ||Delta w_t||_1
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,
        drawdown_penalty: float = 0.25,
        w2_turnover_penalty: float = 0.1,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.drawdown_penalty = drawdown_penalty
        self.w2_turnover_penalty = w2_turnover_penalty

    def compute(
        self,
        net_return: float,
        gross_return: float,
        turnover: float,
        transaction_cost: float,
        unrealized_drawdown: float = 0.0,
        w2_distance: float = 0.0,
    ) -> float:
        r = float(net_return)
        log_ret = float(np.log(max(1e-8, 1.0 + r)))
        var_pen = 0.5 * self.risk_aversion * (r**2)
        dd_pen = self.drawdown_penalty * abs(min(0.0, float(unrealized_drawdown)))
        thermo_pen = self.w2_turnover_penalty * float(w2_distance) * float(turnover)
        return float(log_ret - var_pen - dd_pen - thermo_pen)
