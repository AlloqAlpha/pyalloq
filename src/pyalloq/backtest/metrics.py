import pandas as pd
import numpy as np


class MetricsTearSheet:
    @staticmethod
    def generate(
        portfolio_returns: pd.Series, risk_free_rate: float = 0.0
    ) -> pd.DataFrame:
        metrics = {}

        ann_factor = 252

        cum_return = (1 + portfolio_returns).cumprod()
        metrics["total_return"] = cum_return.iloc[-1] - 1

        n_years = len(portfolio_returns) / ann_factor
        metrics["annualized_return"] = (1 + metrics["total_return"]) ** (
            1 / n_years
        ) - 1

        metrics["annualized_volatility"] = portfolio_returns.std() * np.sqrt(ann_factor)

        excess_return = metrics["annualized_return"] - risk_free_rate
        metrics["sharpe_ratio"] = (
            excess_return / metrics["annualized_volatility"]
            if metrics["annualized_volatility"] > 0
            else 0
        )

        rolling_max = cum_return.cummax()
        drawdown = (cum_return - rolling_max) / rolling_max
        metrics["maximum_drawdown"] = drawdown.min()

        metrics["calmar_ratio"] = (
            metrics["annualized_return"] / abs(metrics["maximum_drawdown"])
            if metrics["maximum_drawdown"] != 0
            else 0
        )

        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
