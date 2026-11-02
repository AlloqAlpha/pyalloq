import math
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import norm


class MetricsTearSheet:
    @staticmethod
    def generate(
        portfolio_returns: pd.Series,
        risk_free_rate: float | pd.Series = 0.0,
        weights: pd.DataFrame | None = None,
        turnover: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
        target_sharpe: float = 0.0,
    ) -> pd.DataFrame:
        clean_returns = portfolio_returns.dropna()
        ann_factor = 252.0

        daily_rf: pd.Series | float
        if isinstance(risk_free_rate, pd.Series):
            aligned_rf = risk_free_rate.reindex(clean_returns.index).ffill().fillna(0.0)
            rf_annual = float(aligned_rf.mean()) if len(aligned_rf) > 0 else 0.0
            daily_rf = aligned_rf / ann_factor
        else:
            rf_annual = float(risk_free_rate)
            daily_rf = rf_annual / ann_factor

        metrics: dict[str, float] = {}

        if len(clean_returns) == 0:
            for k in (
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "maximum_drawdown",
                "calmar_ratio",
                "ulcer_index",
                "omega_ratio",
                "tail_ratio",
                "probabilistic_sharpe_ratio",
                "skewness",
                "kurtosis",
                "var_95_daily",
                "cvar_95_daily",
                "win_rate",
                "profit_factor",
            ):
                metrics[k] = 0.0
            return pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])

        cum_return = (1.0 + clean_returns).cumprod()
        metrics["total_return"] = float(cum_return.iloc[-1] - 1.0)

        n_years = max(len(clean_returns) / ann_factor, 1e-4)
        metrics["annualized_return"] = float(
            (1.0 + metrics["total_return"]) ** (1.0 / n_years) - 1.0
        )
        metrics["annualized_volatility"] = float(
            clean_returns.std() * np.sqrt(ann_factor)
        )

        excess_return = metrics["annualized_return"] - rf_annual
        metrics["sharpe_ratio"] = float(
            excess_return / metrics["annualized_volatility"]
            if metrics["annualized_volatility"] > 0
            else 0.0
        )

        negative_returns = clean_returns[clean_returns < 0]
        downside_vol = float(negative_returns.std() * np.sqrt(ann_factor))
        metrics["sortino_ratio"] = float(
            excess_return / downside_vol
            if pd.notna(downside_vol) and downside_vol > 0
            else 0.0
        )

        rolling_max = cum_return.cummax()
        drawdown = (cum_return - rolling_max) / rolling_max.replace(0.0, np.nan)
        drawdown = drawdown.fillna(0.0)
        metrics["maximum_drawdown"] = float(drawdown.min())

        metrics["calmar_ratio"] = float(
            metrics["annualized_return"] / abs(metrics["maximum_drawdown"])
            if metrics["maximum_drawdown"] != 0
            else 0.0
        )

        # Ulcer Index: RMS of drawdown series
        metrics["ulcer_index"] = float(np.sqrt(np.mean(np.square(drawdown.to_numpy()))))

        # Omega Ratio above daily risk-free threshold
        excess_daily = clean_returns - daily_rf
        gains_above = float(excess_daily[excess_daily > 0].sum())
        losses_below = float(np.abs(excess_daily[excess_daily < 0].sum()))
        metrics["omega_ratio"] = (
            gains_above / losses_below
            if losses_below > 1e-12
            else (np.inf if gains_above > 0 else 0.0)
        )

        skew_val = (
            float(cast(Any, clean_returns.skew())) if len(clean_returns) > 2 else 0.0
        )
        kurt_val = (
            float(cast(Any, clean_returns.kurtosis()))
            if len(clean_returns) > 3
            else 0.0
        )
        metrics["skewness"] = 0.0 if np.isnan(skew_val) else skew_val
        metrics["kurtosis"] = 0.0 if np.isnan(kurt_val) else kurt_val

        var_95 = float(clean_returns.quantile(0.05))
        q_95 = float(clean_returns.quantile(0.95))
        metrics["var_95_daily"] = var_95

        tail_slice = clean_returns[clean_returns <= var_95]
        metrics["cvar_95_daily"] = (
            float(tail_slice.mean()) if len(tail_slice) > 0 else var_95
        )
        metrics["tail_ratio"] = abs(q_95 / var_95) if abs(var_95) > 1e-12 else 0.0

        # Probabilistic Sharpe Ratio (PSR, Bailey & Lopez de Prado)
        # Uses non-annualized sample Sharpe & target Sharpe over T observations
        n_obs = len(clean_returns)
        if n_obs > 2 and metrics["annualized_volatility"] > 1e-12:
            sr_daily = metrics["sharpe_ratio"] / math.sqrt(ann_factor)
            sr_star_daily = target_sharpe / math.sqrt(ann_factor)
            # Excess kurtosis is pandas .kurtosis(), Pearson kurtosis is kurt + 3
            pearson_kurt = metrics["kurtosis"] + 3.0
            denom_sq = (
                1.0
                - metrics["skewness"] * sr_daily
                + ((pearson_kurt - 1.0) / 4.0) * (sr_daily**2)
            )
            denom = math.sqrt(max(denom_sq, 1e-12))
            z_stat = (sr_daily - sr_star_daily) * math.sqrt(n_obs - 1) / denom
            metrics["probabilistic_sharpe_ratio"] = float(norm.cdf(z_stat))
        else:
            metrics["probabilistic_sharpe_ratio"] = (
                1.0 if metrics["total_return"] > 0 else 0.0
            )

        metrics["win_rate"] = float((clean_returns > 0).mean())

        gross_profits = float(clean_returns[clean_returns > 0].sum())
        gross_losses = float(np.abs(clean_returns[clean_returns < 0].sum()))
        metrics["profit_factor"] = (
            gross_profits / gross_losses if gross_losses > 0 else np.inf
        )

        if turnover is not None and len(turnover.dropna()) > 0:
            metrics["annualized_turnover"] = float(
                turnover.dropna().mean() * ann_factor
            )

        if weights is not None:
            valid_w = weights.dropna(how="all")
            if len(valid_w) > 0:
                herfindahl = np.sum(np.square(valid_w.to_numpy(dtype=float)), axis=1)
                valid_h = herfindahl[herfindahl > 1e-12]
                metrics["effective_number_of_bets"] = (
                    float(np.mean(1.0 / valid_h)) if len(valid_h) > 0 else 0.0
                )

        if benchmark_returns is not None:
            aligned = pd.concat(
                [clean_returns.rename("port"), benchmark_returns.rename("bench")],
                axis=1,
            ).dropna()
            if len(aligned) > 1:
                active_ret = aligned["port"] - aligned["bench"]
                te = float(active_ret.std() * np.sqrt(ann_factor))
                metrics["tracking_error"] = te
                metrics["information_ratio"] = (
                    float((active_ret.mean() * ann_factor) / te) if te > 1e-12 else 0.0
                )
                bench_var = float(aligned["bench"].var())
                if bench_var > 1e-12:
                    cov_pb = float(aligned["port"].cov(aligned["bench"]))
                    beta = cov_pb / bench_var
                    alpha = float(
                        (aligned["port"].mean() - beta * aligned["bench"].mean())
                        * ann_factor
                    )
                else:
                    beta = 0.0
                    alpha = 0.0
                metrics["beta"] = beta
                metrics["alpha"] = alpha

        return pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
