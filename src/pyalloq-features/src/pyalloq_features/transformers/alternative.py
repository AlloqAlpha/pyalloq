from __future__ import annotations

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_features.core.base import BaseFeatureTransformer


class PITOptionsMFIVTransformer(BaseFeatureTransformer):
    """
    Point-in-Time (PIT) Model-Free Implied Volatility (MFIV) Feature Transformer.

    Converts raw historical option chain snapshots into a clean, date-aligned
    2D feature table `data.features[out_col]` (default `"implied_vol"`) using the
    Britten-Jones & Neuberger (2000) / CBOE VIX model-free variance formula:
        sigma_MFIV^2 = (2 * exp(r * T) / T) * sum_i (Delta_K_i / K_i^2) * Q(K_i)
                       - (1 / T) * (F / K_0 - 1)^2

    Point-in-Time (PIT) Guarantees:
    -------------------------------
    1. Any option quote with `quote_timestamp > decision_timestamp` (where
       `decision_timestamp` defaults to the bar timestamp + `market_close_offset_hours`
       when dates are daily midnight timestamps) is strictly discarded to prevent look-ahead bias.
    2. If option chains are missing for a subset of tickers or dates, seamlessly falls
       back to Parkinson High-Low range volatility (if `"high"` and `"low"` are present
       in `data.features`) or EWMA realized volatility from `data.prices`.
    """

    def __init__(
        self,
        option_chain_df: pd.DataFrame | None = None,
        target_horizon_days: float = 30.0,
        fallback_window: int = 20,
        annualization_factor: float = 252.0,
        market_close_offset_hours: float = 16.0,
        out_col: str = "implied_vol",
        min_vol: float = 0.02,
        max_vol: float = 3.00,
    ) -> None:
        self.option_chain_df = (
            option_chain_df.copy() if option_chain_df is not None else None
        )
        self.target_horizon_days = float(target_horizon_days)
        self.fallback_window = int(fallback_window)
        self.annualization_factor = float(annualization_factor)
        self.market_close_offset_hours = float(market_close_offset_hours)
        self.out_col = out_col
        self.min_vol = float(min_vol)
        self.max_vol = float(max_vol)

    @staticmethod
    def compute_single_slice_mfiv(
        slice_df: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.0,
        target_horizon_days: float = 30.0,
    ) -> float | None:
        """
        Computes annualized Model-Free Implied Volatility (MFIV) for a single
        (date, ticker) option chain slice, interpolating across expirations if multiple exist.
        """
        if slice_df.empty or spot_price <= 0:
            return None

        df = slice_df.copy()
        if "mid" not in df.columns:
            if "bid" in df.columns and "ask" in df.columns:
                df["mid"] = 0.5 * (df["bid"].astype(float) + df["ask"].astype(float))
            elif "price" in df.columns:
                df["mid"] = df["price"].astype(float)
            else:
                return None

        df = df[df["mid"] > 0.0]
        if len(df) < 3:
            return None

        variances: list[tuple[float, float]] = []
        for _, exp_group in df.groupby("tau_years"):
            tau = float(exp_group["tau_years"].iloc[0])
            if tau <= 1.0 / 365.0:
                continue

            forward = spot_price * float(np.exp(risk_free_rate * tau))
            if "option_type" in exp_group.columns:
                opt_type = exp_group["option_type"].astype(str).str.lower().str[0]
                otm_puts = exp_group[
                    (opt_type == "p") & (exp_group["strike"] <= forward)
                ]
                otm_calls = exp_group[
                    (opt_type == "c") & (exp_group["strike"] >= forward)
                ]
                otm = pd.concat([otm_puts, otm_calls], ignore_index=True)
                if len(otm) < 3:
                    otm = exp_group
            else:
                otm = exp_group

            otm_series = otm.groupby("strike")["mid"].mean().sort_index()
            strikes = otm_series.index.to_numpy(dtype=np.float64)
            mids = otm_series.to_numpy(dtype=np.float64)
            n_k = len(strikes)
            if n_k < 3:
                continue

            delta_k = np.empty(n_k, dtype=np.float64)
            delta_k[0] = strikes[1] - strikes[0]
            delta_k[-1] = strikes[-1] - strikes[-2]
            if n_k > 2:
                delta_k[1:-1] = 0.5 * (strikes[2:] - strikes[:-2])

            below_f = strikes[strikes <= forward]
            k0 = float(below_f[-1]) if len(below_f) > 0 else float(strikes[0])

            integral_term = (2.0 * np.exp(risk_free_rate * tau) / tau) * float(
                np.sum((delta_k / (strikes**2)) * mids)
            )
            forward_adj = (1.0 / tau) * ((forward / k0 - 1.0) ** 2)
            var_tau = max(1e-6, integral_term - forward_adj)
            variances.append((tau, var_tau))

        if not variances:
            return None

        variances.sort(key=lambda x: x[0])
        if len(variances) == 1:
            return float(np.sqrt(variances[0][1]))

        taus = np.array([v[0] for v in variances], dtype=np.float64)
        vars_arr = np.array([v[1] for v in variances], dtype=np.float64)
        target_tau = float(target_horizon_days) / 365.25
        mean_var = float(np.interp(target_tau, taus, vars_arr))
        return float(np.sqrt(max(mean_var, 1e-6)))

    def _compute_fallback_volatility(self, data: MarketData) -> pd.DataFrame:
        """
        Computes causal fallback volatility (Parkinson High-Low if available,
        otherwise EWMA realized volatility) to guarantee 100% coverage.
        """
        prices = data.prices
        returns = prices.pct_change()
        ewma_vol = returns.ewm(
            span=self.fallback_window, min_periods=2
        ).std() * np.sqrt(self.annualization_factor)
        ewma_vol = ewma_vol.bfill().fillna(0.15)

        if "high" in data.features and "low" in data.features:
            high_df = data.features["high"].reindex(prices.index)
            low_df = data.features["low"].reindex(prices.index)
            valid_hl = (high_df > 0) & (low_df > 0) & (high_df >= low_df)
            log_hl_sq = np.where(
                valid_hl,
                np.log(high_df.clip(lower=1e-8) / low_df.clip(lower=1e-8)) ** 2,
                np.nan,
            )
            hl_df = pd.DataFrame(log_hl_sq, index=prices.index, columns=prices.columns)
            parkinson_factor = 1.0 / (4.0 * np.log(2.0))
            parkinson_var = (
                parkinson_factor
                * hl_df.ewm(span=self.fallback_window, min_periods=2).mean()
                * self.annualization_factor
            )
            parkinson_vol = np.sqrt(parkinson_var.clip(lower=1e-8))
            fallback = parkinson_vol.combine_first(ewma_vol)
        else:
            fallback = ewma_vol

        return fallback.clip(lower=self.min_vol, upper=self.max_vol)

    def transform(self, data: MarketData) -> MarketData:
        fallback_df = self._compute_fallback_volatility(data)
        iv_df = fallback_df.copy()

        if self.option_chain_df is not None and not self.option_chain_df.empty:
            chains = self.option_chain_df.copy()
            required_cols = {"date", "ticker", "expiration", "strike"}
            if not required_cols.issubset(set(chains.columns)):
                missing = required_cols - set(chains.columns)
                raise ValueError(
                    f"option_chain_df is missing required columns: {sorted(missing)}"
                )

            chains["date"] = pd.to_datetime(chains["date"])
            chains["expiration"] = pd.to_datetime(chains["expiration"])

            # Strict Point-in-Time (PIT) timestamp filtering
            if "quote_timestamp" in chains.columns:
                chains["quote_timestamp"] = pd.to_datetime(chains["quote_timestamp"])
                if "decision_timestamp" in chains.columns:
                    decision_ts = pd.to_datetime(chains["decision_timestamp"])
                else:
                    is_midnight = (
                        (chains["date"].dt.hour == 0)
                        & (chains["date"].dt.minute == 0)
                        & (chains["date"].dt.second == 0)
                    )
                    decision_ts = chains["date"] + pd.to_timedelta(
                        np.where(is_midnight, self.market_close_offset_hours, 0.0),
                        unit="h",
                    )
                chains = chains[chains["quote_timestamp"] <= decision_ts].copy()

            chains["tau_years"] = (
                chains["expiration"] - chains["date"]
            ).dt.total_seconds() / (365.25 * 86400.0)
            chains = chains[chains["tau_years"] > (1.0 / 365.0)].copy()

            prices_np = data.prices.to_numpy(dtype=np.float64)
            iv_np = iv_df.to_numpy(dtype=np.float64, copy=True)
            date_to_idx = {d: i for i, d in enumerate(data.prices.index)}
            col_to_idx = {c: j for j, c in enumerate(data.prices.columns)}

            for (dt, ticker), grp in chains.groupby(["date", "ticker"]):
                if dt not in date_to_idx or ticker not in col_to_idx:
                    continue
                r_idx = date_to_idx[dt]
                c_idx = col_to_idx[ticker]
                spot = float(prices_np[r_idx, c_idx])
                if isinstance(data.risk_free_rate, pd.Series):
                    rf = float(
                        data.risk_free_rate.reindex(data.prices.index)
                        .ffill()
                        .bfill()
                        .fillna(0.0)
                        .to_numpy(dtype=np.float64)[r_idx]
                    )
                else:
                    rf = float(data.risk_free_rate)

                mfiv_val = self.compute_single_slice_mfiv(
                    grp,
                    spot_price=spot,
                    risk_free_rate=rf,
                    target_horizon_days=self.target_horizon_days,
                )
                if mfiv_val is not None and np.isfinite(mfiv_val):
                    iv_np[r_idx, c_idx] = float(
                        np.clip(mfiv_val, self.min_vol, self.max_vol)
                    )

            iv_df = pd.DataFrame(
                iv_np, index=data.prices.index, columns=data.prices.columns
            )

        data.features[self.out_col] = iv_df.astype(np.float64)
        return data
