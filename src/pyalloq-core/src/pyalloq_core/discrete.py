from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from .data import MarketData


@dataclass
class DiscreteAllocationResult:
    """
    Standardized output for discrete share allocation converting continuous
    target portfolio weights into integer share counts and residual cash.
    """

    shares: dict[str, int]
    leftover_cash: float
    total_portfolio_value: float
    realized_weights: pd.Series
    rmse_error: float


class DiscreteShareAllocator:
    """
    Converts continuous portfolio weights into executable integer share/lot counts
    given latest asset prices and a total portfolio cash value.

    Supports both long-only and long-short portfolios via:
    - `method="greedy"`: Two-stage floor + greedy residual cash allocation.
    - `method="lp"`: Mixed-Integer Linear Programming (with greedy fallback if no MILP solver is installed).
    """

    def __init__(
        self,
        total_portfolio_value: float = 100_000.0,
        lot_size: int = 1,
        short_ratio: float | None = None,
    ) -> None:
        if total_portfolio_value <= 0:
            raise ValueError("total_portfolio_value must be strictly positive.")
        if lot_size < 1:
            raise ValueError("lot_size must be an integer >= 1.")
        self.total_portfolio_value = float(total_portfolio_value)
        self.lot_size = int(lot_size)
        self.short_ratio = short_ratio

    @staticmethod
    def extract_latest_prices(
        prices_or_data: MarketData | pd.DataFrame | pd.Series,
    ) -> pd.Series:
        """Extracts the most recent price vector as a 1D pandas Series."""
        if isinstance(prices_or_data, MarketData):
            latest = prices_or_data.prices.iloc[-1]
        elif isinstance(prices_or_data, pd.DataFrame):
            latest = prices_or_data.iloc[-1]
        elif isinstance(prices_or_data, pd.Series):
            latest = prices_or_data.copy()
        else:
            raise TypeError(
                "prices_or_data must be a MarketData instance, DataFrame, or Series."
            )
        latest = latest.astype(float)
        if (latest <= 0).any() or latest.isna().any():
            raise ValueError(
                "Latest asset prices must be strictly positive and non-NaN."
            )
        return latest

    def _allocate_greedy_positive(
        self,
        weights: pd.Series,
        latest_prices: pd.Series,
        budget: float,
    ) -> tuple[dict[str, int], float]:
        """Greedy two-phase integer allocation for non-negative weights."""
        active = weights[weights > 0].copy()
        if active.empty or budget <= 0:
            return {str(k): 0 for k in weights.index}, float(budget)

        active = active / float(active.sum())
        prices = latest_prices.reindex(active.index)
        lot_costs = prices * self.lot_size

        target_dollars = active * budget
        initial_lots = pd.Series(
            np.floor((target_dollars / lot_costs).to_numpy(dtype=float)).astype(int),
            index=active.index,
        )
        spent = float((initial_lots * lot_costs).sum())
        available_cash = budget - spent

        # Second pass: greedily buy additional lots of whichever affordable asset
        # currently has the largest deficit relative to its target weight
        current_lots = initial_lots.copy()
        while True:
            affordable_mask = lot_costs <= available_cash + 1e-9
            if not affordable_mask.any():
                break

            current_dollars = current_lots * lot_costs
            total_invested = max(float(current_dollars.sum()), 1e-12)
            current_w = current_dollars / total_invested
            deficits = (active - current_w).where(affordable_mask, -np.inf)
            best_ticker = str(deficits.idxmax())
            if deficits[best_ticker] == -np.inf:
                break

            current_lots[best_ticker] += 1
            available_cash -= float(lot_costs[best_ticker])

        shares_out: dict[str, int] = {str(k): 0 for k in weights.index}
        for ticker, lots in current_lots.items():
            shares_out[str(ticker)] = int(lots * self.lot_size)

        return shares_out, max(0.0, float(available_cash))

    def _allocate_lp_positive(
        self,
        weights: pd.Series,
        latest_prices: pd.Series,
        budget: float,
    ) -> tuple[dict[str, int], float]:
        """MILP integer allocation minimizing L1 tracking error plus uninvested cash."""
        active = weights[weights > 0].copy()
        if active.empty or budget <= 0:
            return {str(k): 0 for k in weights.index}, float(budget)

        active = active / float(active.sum())
        tickers = list(active.index)
        n = len(tickers)
        lot_costs = (
            latest_prices.reindex(tickers).to_numpy(dtype=float)
        ) * self.lot_size
        target_dollars = active.to_numpy(dtype=float) * budget

        lots_var = cp.Variable(n, integer=True)
        u_var = cp.Variable(n, nonneg=True)
        cash_rem = budget - lot_costs @ lots_var

        constraints: list[Any] = [
            lots_var >= 0,
            cash_rem >= 0,
            target_dollars - cp.multiply(lot_costs, lots_var) <= u_var,
            cp.multiply(lot_costs, lots_var) - target_dollars <= u_var,
        ]
        obj = cp.Minimize(cp.sum(u_var) + cash_rem)
        prob = cp.Problem(obj, constraints)

        solved = False
        for solver_name in ("HIGHS", "SCIP", "CBC", "GLPK_MI", "ECOS_BB"):
            if solver_name in cp.installed_solvers():
                try:
                    prob.solve(solver=solver_name)
                    if (
                        prob.status in ("optimal", "optimal_inaccurate")
                        and lots_var.value is not None
                    ):
                        solved = True
                        break
                except Exception:
                    continue

        if not solved or lots_var.value is None:
            return self._allocate_greedy_positive(weights, latest_prices, budget)

        lots_rounded = np.maximum(np.rint(lots_var.value).astype(int), 0)
        spent = float(np.dot(lots_rounded, lot_costs))
        if spent > budget + 1e-6:
            return self._allocate_greedy_positive(weights, latest_prices, budget)

        shares_out: dict[str, int] = {str(k): 0 for k in weights.index}
        for idx, ticker in enumerate(tickers):
            shares_out[str(ticker)] = int(lots_rounded[idx] * self.lot_size)
        return shares_out, max(0.0, float(budget - spent))

    def allocate(
        self,
        weights: pd.Series | dict[str, float],
        latest_prices: MarketData | pd.DataFrame | pd.Series,
        method: str = "greedy",
    ) -> DiscreteAllocationResult:
        """
        Computes discrete share allocations for the given continuous weights and latest prices.
        """
        if method not in ("greedy", "lp"):
            raise ValueError("method must be 'greedy' or 'lp'.")

        w_series = (
            weights.copy().astype(float)
            if isinstance(weights, pd.Series)
            else pd.Series(weights, dtype=float)
        )
        prices = self.extract_latest_prices(latest_prices).reindex(w_series.index)
        if prices.isna().any():
            missing = list(prices[prices.isna()].index)
            raise ValueError(f"Missing latest prices for tickers: {missing}")

        solver_fn = (
            self._allocate_lp_positive
            if method == "lp"
            else self._allocate_greedy_positive
        )

        has_shorts = bool((w_series < -1e-8).any())
        if not has_shorts:
            w_clean = w_series.clip(lower=0.0)
            w_sum = float(w_clean.sum())
            investable_budget = (
                self.total_portfolio_value * min(w_sum, 1.0) if w_sum > 0 else 0.0
            )
            unallocated_cash = self.total_portfolio_value - investable_budget
            shares, rem_cash = solver_fn(w_clean, prices, investable_budget)
            leftover_cash = unallocated_cash + rem_cash
        else:
            long_w = w_series.clip(lower=0.0)
            short_w = (-w_series).clip(lower=0.0)

            short_ratio = (
                self.short_ratio
                if self.short_ratio is not None
                else float(short_w.sum())
            )
            long_sum = float(long_w.sum())
            long_budget = self.total_portfolio_value * long_sum if long_sum > 0 else 0.0
            short_budget = self.total_portfolio_value * short_ratio

            long_shares, long_rem = solver_fn(long_w, prices, long_budget)
            short_shares, short_rem = solver_fn(short_w, prices, short_budget)

            shares = {}
            for k in w_series.index:
                tk = str(k)
                shares[tk] = long_shares.get(tk, 0) - short_shares.get(tk, 0)
            leftover_cash = (
                self.total_portfolio_value - long_budget + long_rem + short_rem
            )

        share_series = (
            pd.Series(shares, dtype=float).reindex(w_series.index).fillna(0.0)
        )
        dollar_positions = share_series * prices
        realized_weights = dollar_positions / self.total_portfolio_value

        w_norm = (
            w_series / float(w_series.abs().sum())
            if float(w_series.abs().sum()) > 1e-12 and not has_shorts
            else w_series
        )
        rmse = float(
            np.sqrt(np.mean(np.square((realized_weights - w_norm).to_numpy())))
        )

        return DiscreteAllocationResult(
            shares=shares,
            leftover_cash=float(leftover_cash),
            total_portfolio_value=self.total_portfolio_value,
            realized_weights=realized_weights,
            rmse_error=rmse,
        )
