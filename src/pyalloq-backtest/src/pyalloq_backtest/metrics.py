import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI: float = 0.5772156649015329


def compute_expected_max_sharpe(
    n_trials: int,
    sharpe_std: float,
    mean_sharpe: float = 0.0,
) -> float:
    """
    Computes the Expected Maximum Sharpe Ratio E[max_n SR_n] across `n_trials` independent
    backtest configurations under the False Strategy Theorem (Bailey & Lopez de Prado, 2014):

        E[max SR] = E[SR] + sqrt(Var[SR]) * ((1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N * e)))

    where gamma is the Euler-Mascheroni constant (~0.5772).
    """
    if n_trials <= 1 or sharpe_std <= 0.0:
        return float(mean_sharpe)

    n_float = float(max(n_trials, 2))
    q1 = float(norm.ppf(1.0 - 1.0 / n_float))
    q2 = float(norm.ppf(1.0 - 1.0 / (n_float * math.e)))
    z_max = (1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2
    return float(mean_sharpe + sharpe_std * z_max)


def compute_deflated_sharpe_ratio(
    portfolio_returns: pd.Series | np.ndarray,
    n_trials: int = 1,
    trial_sharpes: Sequence[float] | np.ndarray | None = None,
    sharpe_variance: float | None = None,
    risk_free_rate: float | pd.Series = 0.0,
    target_sharpe: float = 0.0,
    ann_factor: float = 252.0,
) -> tuple[float, float]:
    """
    Computes the Deflated Sharpe Ratio (DSR) and annualized Expected Maximum Sharpe Ratio (SR_0)
    following Bailey & Lopez de Prado (2014), accounting for:
      1. Multiple testing penalty across `n_trials` (N) backtest variations,
      2. Track record length (T),
      3. Non-Normal return skewness (gamma_3) and Pearson kurtosis (gamma_4).

    Note: `trial_sharpes` and `sharpe_variance` (if provided) are expected in annualized Sharpe units.

    Returns:
        (dsr_probability, annualized_expected_max_sharpe)
    """
    if isinstance(portfolio_returns, pd.Series):
        clean = portfolio_returns.dropna()
    else:
        arr = np.asarray(portfolio_returns, dtype=np.float64)
        clean = pd.Series(arr[np.isfinite(arr)])

    n_obs = len(clean)
    if n_obs <= 2:
        return 0.0, float(target_sharpe)

    if isinstance(risk_free_rate, pd.Series):
        aligned_rf = risk_free_rate.reindex(clean.index).ffill().fillna(0.0)
        rf_annual = float(aligned_rf.mean()) if len(aligned_rf) > 0 else 0.0
    else:
        rf_annual = float(risk_free_rate)

    vals = clean.to_numpy(dtype=np.float64)
    cum_return = float(np.prod(1.0 + vals) - 1.0)
    n_years = max(n_obs / ann_factor, 1e-4)
    ann_ret = (
        float((1.0 + cum_return) ** (1.0 / n_years) - 1.0)
        if (1.0 + cum_return) > 0.0
        else -1.0
    )
    ann_vol = float(np.std(vals, ddof=1) * math.sqrt(ann_factor))
    if ann_vol <= 1e-12:
        return (1.0 if cum_return > 0.0 else 0.0), float(target_sharpe)

    sr_annual = (ann_ret - rf_annual) / ann_vol
    sr_daily = sr_annual / math.sqrt(ann_factor)

    skew_val = float(cast(Any, clean.skew())) if n_obs > 2 else 0.0
    kurt_val = float(cast(Any, clean.kurtosis())) if n_obs > 3 else 0.0
    skew_val = 0.0 if math.isnan(skew_val) else skew_val
    kurt_val = 0.0 if math.isnan(kurt_val) else kurt_val
    pearson_kurt = kurt_val + 3.0

    # Determine daily standard deviation of trial Sharpe ratios across N trials
    eff_n = max(int(n_trials), 1)
    if trial_sharpes is not None:
        ts_arr = np.asarray(list(trial_sharpes), dtype=np.float64)
        ts_arr = ts_arr[np.isfinite(ts_arr)]
        eff_n = max(eff_n, len(ts_arr))
        if len(ts_arr) >= 2:
            emp_std_ann = float(np.std(ts_arr, ddof=1))
            emp_std_daily = emp_std_ann / math.sqrt(ann_factor)
        else:
            emp_std_daily = 0.0
    elif sharpe_variance is not None and sharpe_variance > 0.0:
        emp_std_daily = math.sqrt(float(sharpe_variance)) / math.sqrt(ann_factor)
    else:
        emp_std_daily = 0.0

    # Under H_0, asymptotic standard error of daily Sharpe over T observations is 1 / sqrt(T - 1)
    null_se_daily = 1.0 / math.sqrt(max(n_obs - 1, 1))
    sr_std_daily = emp_std_daily if emp_std_daily > 1e-8 else null_se_daily

    base_sr0_daily = target_sharpe / math.sqrt(ann_factor)
    sr0_daily = compute_expected_max_sharpe(
        n_trials=eff_n,
        sharpe_std=sr_std_daily,
        mean_sharpe=base_sr0_daily,
    )
    sr0_annual = sr0_daily * math.sqrt(ann_factor)

    denom_sq = 1.0 - skew_val * sr_daily + ((pearson_kurt - 1.0) / 4.0) * (sr_daily**2)
    denom = math.sqrt(max(denom_sq, 1e-12))
    z_dsr = (sr_daily - sr0_daily) * math.sqrt(n_obs - 1) / denom
    dsr_val = float(norm.cdf(z_dsr))
    return dsr_val, float(sr0_annual)


class MetricsTearSheet:
    @staticmethod
    def generate(
        portfolio_returns: pd.Series,
        risk_free_rate: float | pd.Series = 0.0,
        weights: pd.DataFrame | None = None,
        turnover: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
        target_sharpe: float = 0.0,
        n_trials: int = 1,
        trial_sharpes: Sequence[float] | np.ndarray | None = None,
        sharpe_variance: float | None = None,
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
                "deflated_sharpe_ratio",
                "expected_max_sharpe_ratio",
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

        dsr_val, sr0_ann = compute_deflated_sharpe_ratio(
            clean_returns,
            n_trials=n_trials,
            trial_sharpes=trial_sharpes,
            sharpe_variance=sharpe_variance,
            risk_free_rate=risk_free_rate,
            target_sharpe=target_sharpe,
            ann_factor=ann_factor,
        )
        metrics["deflated_sharpe_ratio"] = dsr_val
        metrics["expected_max_sharpe_ratio"] = sr0_ann

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
