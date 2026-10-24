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
        metrics["total_return"] = cum_return.iloc[-1] - 1.0

        n_years = max(
            len(portfolio_returns) / ann_factor, 1e-4
        )  # Prevent division by zero
        metrics["annualized_return"] = (1 + metrics["total_return"]) ** (
            1 / n_years
        ) - 1.0
        metrics["annualized_volatility"] = portfolio_returns.std() * np.sqrt(ann_factor)

        excess_return = metrics["annualized_return"] - risk_free_rate
        metrics["sharpe_ratio"] = (
            excess_return / metrics["annualized_volatility"]
            if metrics["annualized_volatility"] > 0
            else 0.0
        )

        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(ann_factor)
        metrics["sortino_ratio"] = (
            excess_return / downside_vol
            if pd.notna(downside_vol) and downside_vol > 0
            else 0.0
        )

        rolling_max = cum_return.cummax()
        drawdown = (cum_return - rolling_max) / rolling_max
        metrics["maximum_drawdown"] = drawdown.min()

        metrics["calmar_ratio"] = (
            metrics["annualized_return"] / abs(metrics["maximum_drawdown"])
            if metrics["maximum_drawdown"] != 0
            else 0.0
        )

        metrics["skewness"] = portfolio_returns.skew()
        metrics["kurtosis"] = portfolio_returns.kurtosis()

        var_95 = portfolio_returns.quantile(0.05)
        metrics["var_95_daily"] = var_95

        metrics["cvar_95_daily"] = portfolio_returns[portfolio_returns <= var_95].mean()

        metrics["win_rate"] = (portfolio_returns > 0).mean()

        gross_profits = portfolio_returns[portfolio_returns > 0].sum()
        gross_losses = np.abs(portfolio_returns[portfolio_returns < 0].sum())
        metrics["profit_factor"] = (
            gross_profits / gross_losses if gross_losses > 0 else np.inf
        )

        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
