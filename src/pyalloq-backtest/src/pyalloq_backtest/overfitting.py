from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from pyalloq_core.data import MarketData
from pyalloq_core.results import BacktestResult, MonteCarloResult


@dataclass
class CPCVResult:
    """
    Holds the combinatorial timeline paths and distribution of performance metrics
    generated via Combinatorial Purged Cross-Validation (CPCV) or Block Scrambling.
    """

    returns_matrix: pd.DataFrame  # Shape (T, N_paths)
    equity_curves: pd.DataFrame  # Shape (T, N_paths)
    sharpe_ratios: pd.Series  # Shape (N_paths,)
    max_drawdowns: pd.Series  # Shape (N_paths,)
    path_block_orders: list[tuple[int, ...]] = field(default_factory=list)
    n_groups: int = 6
    k_test_groups: int = 2
    historical_sharpe: float | None = None
    historical_equity: pd.Series | None = None


@dataclass
class PBOResult:
    """
    Holds the results of Probability of Backtest Overfitting (PBO) evaluation.
    Supports both:
      - 'cscv' mode (Bailey & Lopez de Prado Combinatorial Symmetric Cross-Validation
        across M candidate strategy trials)
      - 'benchmark' mode (Single historical strategy vs. a distribution of synthetic/CPCV paths)
    """

    pbo: float  # Probability of Backtest Overfitting in [0, 1]
    logits: (
        np.ndarray
    )  # Rank logits lambda_c (CSCV) or standardized Sharpe diffs (benchmark)
    is_sharpes: np.ndarray  # In-Sample Sharpe ratios (or historical Sharpe repeated)
    oos_sharpes: np.ndarray  # Out-of-Sample / Synthetic Sharpe ratios
    prob_oos_loss: float  # P(OOS Sharpe < 0)
    degradation_slope: float  # OLS slope of OOS Sharpe vs IS Sharpe
    degradation_intercept: float
    historical_sharpe: float | None = None
    mode: str = "cscv"


class CPCVSplitter:
    """
    Combinatorial Purged Cross-Validation (CPCV) & Regime Block Scrambling engine.

    Supports two workflows:
    1. `.split(data)`: Yields `(train_data, test_data, test_group_indices)` across all
       `C(n_groups, k_test_groups)` splits with purge and embargo zones applied.
    2. `.scramble_returns(returns)` / `.scramble_backtest(backtest_result)`:
       Partitions a historical return series into `n_groups` chronological regime blocks
       and recombines them into `n_paths` combinatorial timelines (scrambling the
       chronological order of market regimes) to test path-dependent survival (drawdowns,
       compounding, ruin) when regime sequences are reordered.
    """

    def __init__(
        self,
        n_groups: int = 6,
        k_test_groups: int = 2,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01,
        max_paths: int = 250,
        random_state: int | None = 42,
    ) -> None:
        if n_groups < 2:
            raise ValueError("`n_groups` must be >= 2.")
        if not (1 <= k_test_groups < n_groups):
            raise ValueError("`k_test_groups` must be between 1 and `n_groups - 1`.")
        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.max_paths = max_paths
        self.random_state = random_state

    @property
    def n_splits(self) -> int:
        return math.comb(self.n_groups, self.k_test_groups)

    @property
    def n_backtest_paths(self) -> int:
        return math.comb(self.n_groups - 1, self.k_test_groups - 1)

    def split(
        self, data: MarketData
    ) -> list[tuple[MarketData, MarketData, tuple[int, ...]]]:
        """
        Generates all C(N, k) purged & embargoed train/test splits of MarketData.
        """
        total_length = len(data.prices)
        indices = np.arange(total_length)
        groups = np.array_split(indices, self.n_groups)
        purge_size = int(total_length * self.purge_pct)
        embargo_size = int(total_length * self.embargo_pct)
        dates = data.prices.index

        splits_out: list[tuple[MarketData, MarketData, tuple[int, ...]]] = []

        for test_group_ids in combinations(range(self.n_groups), self.k_test_groups):
            test_mask = np.zeros(total_length, dtype=bool)
            banned_mask = np.zeros(total_length, dtype=bool)

            for gid in test_group_ids:
                g_idx = groups[gid]
                if len(g_idx) == 0:
                    continue
                test_mask[g_idx] = True
                start_i = max(0, int(g_idx[0]) - purge_size)
                end_i = min(total_length, int(g_idx[-1]) + 1 + embargo_size)
                banned_mask[start_i:end_i] = True

            train_mask = ~banned_mask
            train_dates = dates[train_mask]
            test_dates = dates[test_mask]

            if len(train_dates) == 0 or len(test_dates) == 0:
                continue

            train_data = MarketData(
                prices=data.prices.loc[train_dates],
                features={k: v.loc[train_dates] for k, v in data.features.items()}
                if data.features
                else {},
                cross_sectional=data.cross_sectional,
                risk_free_rate=data.risk_free_rate.loc[train_dates]
                if isinstance(data.risk_free_rate, pd.Series)
                else data.risk_free_rate,
            )
            test_data = MarketData(
                prices=data.prices.loc[test_dates],
                features={k: v.loc[test_dates] for k, v in data.features.items()}
                if data.features
                else {},
                cross_sectional=data.cross_sectional,
                risk_free_rate=data.risk_free_rate.loc[test_dates]
                if isinstance(data.risk_free_rate, pd.Series)
                else data.risk_free_rate,
            )
            splits_out.append((train_data, test_data, test_group_ids))

        return splits_out

    def assemble_cpcv_paths(
        self,
        oos_split_returns: dict[tuple[int, ...], pd.Series],
        historical_returns: pd.Series | None = None,
        risk_free_rate: float = 0.0,
    ) -> CPCVResult:
        """
        Reconstructs the phi(N, k) = C(N - 1, k - 1) full-horizon out-of-sample backtest paths
        from the C(N, k) test split evaluations according to Lopez de Prado (2018).
        """
        group_appearances: dict[int, list[pd.Series]] = {
            g: [] for g in range(self.n_groups)
        }

        # Sort splits lexicographically so path assignment is deterministic
        for test_group_ids in sorted(oos_split_returns.keys()):
            split_ret = oos_split_returns[test_group_ids]
            # Split the test return series evenly across its k test groups
            group_slices = np.array_split(
                np.arange(len(split_ret)), len(test_group_ids)
            )
            for gid, idx_slice in zip(test_group_ids, group_slices):
                group_appearances[gid].append(split_ret.iloc[idx_slice])

        n_paths = min(len(v) for v in group_appearances.values() if len(v) > 0)
        path_series_list: list[pd.Series] = []
        block_orders: list[tuple[int, ...]] = []

        for p_idx in range(n_paths):
            segments = [group_appearances[g][p_idx] for g in range(self.n_groups)]
            combined = pd.concat(segments).sort_index()
            combined.name = f"CPCV_Path_{p_idx}"
            path_series_list.append(combined)
            block_orders.append(tuple(range(self.n_groups)))

        returns_df = pd.concat(path_series_list, axis=1).dropna(how="all").fillna(0.0)
        return self._build_cpcv_result(
            returns_df,
            block_orders=block_orders,
            historical_returns=historical_returns,
            risk_free_rate=risk_free_rate,
        )

    def scramble_timelines(
        self,
        source: BacktestResult | pd.Series,
        risk_free_rate: float = 0.0,
        block_bootstrap: bool = True,
    ) -> CPCVResult:
        """
        Partitions historical strategy returns into `n_groups` regime blocks and generates
        combinatorial scrambled timelines (permutations + block-resampled paths with subtle
        within-block jitter/subsampling) to evaluate strategy survival across scrambled
        chronological sequences of market regimes.
        """
        if isinstance(source, BacktestResult):
            hist_returns = source.returns.dropna()
        else:
            hist_returns = pd.Series(source).dropna()

        rng = np.random.default_rng(self.random_state)
        ret_vals = hist_returns.to_numpy(dtype=np.float64)
        blocks = np.array_split(ret_vals, self.n_groups)
        n_total = len(ret_vals)

        # Generate distinct block sequences (combinatorial permutations + bootstrap block draws)
        all_orders: list[tuple[int, ...]] = []
        if self.n_groups <= 6:
            from itertools import permutations

            perms = list(permutations(range(self.n_groups)))
            # Exclude identity as path 0 if we want pure scrambled, or include all
            if len(perms) > self.max_paths:
                chosen_idx = rng.choice(len(perms), size=self.max_paths, replace=False)
                all_orders = [perms[i] for i in chosen_idx]
            else:
                all_orders = perms
        else:
            seen: set[tuple[int, ...]] = set()
            for _ in range(self.max_paths * 3):
                if len(all_orders) >= self.max_paths:
                    break
                if block_bootstrap:
                    order = tuple(
                        int(x)
                        for x in rng.choice(
                            self.n_groups, size=self.n_groups, replace=True
                        )
                    )
                else:
                    order = tuple(int(x) for x in rng.permutation(self.n_groups))
                if order not in seen:
                    seen.add(order)
                    all_orders.append(order)

        scrambled_cols: dict[str, np.ndarray] = {}
        for idx, order in enumerate(all_orders):
            recombined = np.concatenate([blocks[b] for b in order])
            if block_bootstrap and idx > 0:
                # Apply circular shift within each block so paths aren't identical in terminal return
                shifted_blocks = []
                for b in order:
                    blk = blocks[b]
                    if len(blk) > 2:
                        shift = int(rng.integers(0, len(blk)))
                        # Sample with replacement 90% + permutation to vary compounding & Sharpe slightly
                        sub_idx = rng.choice(len(blk), size=len(blk), replace=True)
                        blk = np.roll(blk, shift) if idx % 2 == 0 else blk[sub_idx]
                    shifted_blocks.append(blk)
                recombined = np.concatenate(shifted_blocks)

            # Align length to n_total
            if len(recombined) >= n_total:
                recombined = recombined[:n_total]
            else:
                pad = np.zeros(n_total - len(recombined))
                recombined = np.concatenate([recombined, pad])

            scrambled_cols[f"Path_{idx}"] = recombined

        returns_df = pd.DataFrame(scrambled_cols, index=hist_returns.index)
        return self._build_cpcv_result(
            returns_df,
            block_orders=all_orders,
            historical_returns=hist_returns,
            risk_free_rate=risk_free_rate,
        )

    def _build_cpcv_result(
        self,
        returns_df: pd.DataFrame,
        block_orders: list[tuple[int, ...]],
        historical_returns: pd.Series | None = None,
        risk_free_rate: float = 0.0,
    ) -> CPCVResult:
        ann_factor = 252.0
        equity_curves = (1.0 + returns_df).cumprod()

        means = returns_df.mean(axis=0) * ann_factor
        stds = returns_df.std(axis=0) * np.sqrt(ann_factor)
        sharpes = ((means - risk_free_rate) / stds.replace(0.0, np.nan)).fillna(0.0)

        running_max = equity_curves.cummax(axis=0)
        drawdowns = (equity_curves - running_max) / running_max.replace(0.0, np.nan)
        max_dds = drawdowns.min(axis=0).fillna(0.0)

        hist_sharpe: float | None = None
        hist_eq: pd.Series | None = None
        if historical_returns is not None and len(historical_returns) > 1:
            hist_eq = (1.0 + historical_returns).cumprod()
            h_std = float(historical_returns.std() * np.sqrt(ann_factor))
            h_mean = float(historical_returns.mean() * ann_factor)
            hist_sharpe = (h_mean - risk_free_rate) / h_std if h_std > 0 else 0.0

        return CPCVResult(
            returns_matrix=returns_df,
            equity_curves=equity_curves,
            sharpe_ratios=sharpes,
            max_drawdowns=max_dds,
            path_block_orders=block_orders,
            n_groups=self.n_groups,
            k_test_groups=self.k_test_groups,
            historical_sharpe=hist_sharpe,
            historical_equity=hist_eq,
        )


class PBOEvaluator:
    """
    Computes the Probability of Backtest Overfitting (PBO) and out-of-sample degradation.

    Supports two modes:
    1. `evaluate_cscv(strategy_returns_matrix, n_partitions=8)`:
       Implements Bailey, Borwein, Lopez de Prado & Zhu (2017) Combinatorial Symmetric
       Cross-Validation (CSCV) on a (T, M) matrix of M strategy trial returns.
    2. `evaluate_benchmark(historical_result, synthetic_or_cpcv_result)`:
       Evaluates a single strategy's historical Sharpe ratio against a distribution of
       simulated (`MonteCarloResult`) or scrambled (`CPCVResult`) Sharpe ratios.
    """

    def __init__(self, n_partitions: int = 8, risk_free_rate: float = 0.0) -> None:
        if n_partitions < 2 or n_partitions % 2 != 0:
            raise ValueError("`n_partitions` must be an even integer >= 2.")
        self.n_partitions = n_partitions
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _annualized_sharpes(returns_2d: np.ndarray, rf: float = 0.0) -> np.ndarray:
        ann_factor = 252.0
        means = np.nanmean(returns_2d, axis=0) * ann_factor
        stds = np.nanstd(returns_2d, axis=0, ddof=1) * np.sqrt(ann_factor)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(stds > 1e-12, (means - rf) / stds, 0.0)
        return sr

    def evaluate_cscv(
        self,
        strategy_returns: pd.DataFrame | np.ndarray | Sequence[BacktestResult],
    ) -> PBOResult:
        """
        Calculates exact PBO via Combinatorial Symmetric Cross-Validation (CSCV) across M trials.
        """
        if isinstance(strategy_returns, pd.DataFrame):
            M_mat = (
                strategy_returns.dropna(how="all")
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        elif isinstance(strategy_returns, np.ndarray):
            M_mat = np.asarray(strategy_returns, dtype=np.float64)
        else:
            df_concat = pd.concat(
                [
                    res.returns.rename(f"{res.name}_{i}")
                    for i, res in enumerate(strategy_returns)
                ],
                axis=1,
            ).fillna(0.0)
            M_mat = df_concat.to_numpy(dtype=np.float64)

        if M_mat.ndim != 2 or M_mat.shape[1] < 2:
            raise ValueError(
                "CSCV requires a 2D return matrix of shape (T, M) with at least M >= 2 strategy trials."
            )

        T, M_trials = M_mat.shape
        S = min(self.n_partitions, (T // 5) * 2)
        S = max(2, S - (S % 2))

        submatrices = np.array_split(M_mat, S, axis=0)
        half_S = S // 2
        all_indices = set(range(S))

        logits_list: list[float] = []
        is_sharpes_list: list[float] = []
        oos_sharpes_list: list[float] = []

        for is_group_tuple in combinations(range(S), half_S):
            oos_group_tuple = tuple(sorted(all_indices - set(is_group_tuple)))
            J_is = np.vstack([submatrices[i] for i in is_group_tuple])
            J_oos = np.vstack([submatrices[i] for i in oos_group_tuple])

            R_is = self._annualized_sharpes(J_is, self.risk_free_rate)
            R_oos = self._annualized_sharpes(J_oos, self.risk_free_rate)

            best_m = int(np.argmax(R_is))
            is_best_sr = float(R_is[best_m])
            oos_of_best_sr = float(R_oos[best_m])

            # Relative rank of the IS-optimal strategy in OOS (in (0, 1))
            less_count = float(np.sum(R_oos < oos_of_best_sr))
            equal_count = float(np.sum(R_oos == oos_of_best_sr))
            omega_c = (less_count + 0.5 * equal_count) / float(M_trials)
            omega_c = float(
                np.clip(omega_c, 1.0 / (2.0 * M_trials), 1.0 - 1.0 / (2.0 * M_trials))
            )

            logit = math.log(omega_c / (1.0 - omega_c))
            logits_list.append(logit)
            is_sharpes_list.append(is_best_sr)
            oos_sharpes_list.append(oos_of_best_sr)

        logits_arr = np.asarray(logits_list, dtype=np.float64)
        is_arr = np.asarray(is_sharpes_list, dtype=np.float64)
        oos_arr = np.asarray(oos_sharpes_list, dtype=np.float64)

        pbo = float(np.mean(logits_arr <= 0.0))
        prob_loss = float(np.mean(oos_arr < 0.0))

        if len(is_arr) > 1 and np.std(is_arr) > 1e-12:
            slope, intercept = np.polyfit(is_arr, oos_arr, 1)
        else:
            slope, intercept = 0.0, float(np.mean(oos_arr))

        full_sharpes = self._annualized_sharpes(M_mat, self.risk_free_rate)
        best_overall_sharpe = float(np.max(full_sharpes))

        return PBOResult(
            pbo=pbo,
            logits=logits_arr,
            is_sharpes=is_arr,
            oos_sharpes=oos_arr,
            prob_oos_loss=prob_loss,
            degradation_slope=float(slope),
            degradation_intercept=float(intercept),
            historical_sharpe=best_overall_sharpe,
            mode="cscv",
        )

    def evaluate_benchmark(
        self,
        historical: BacktestResult | pd.Series | float,
        synthetic: MonteCarloResult | CPCVResult | pd.Series | np.ndarray,
    ) -> PBOResult:
        """
        Evaluates how a single strategy's historical Sharpe compares to a synthetic
        (`MonteCarloResult`) or combinatorial (`CPCVResult`) Sharpe distribution.
        """
        if isinstance(historical, BacktestResult):
            if "sharpe_ratio" in historical.tear_sheet.index:
                val = historical.tear_sheet.loc["sharpe_ratio", "Value"]
                hist_sr = float(val)  # type: ignore[arg-type]
            else:
                hist_sr = float(
                    self._annualized_sharpes(
                        historical.returns.to_numpy()[:, None], self.risk_free_rate
                    )[0]
                )
        elif isinstance(historical, pd.Series):
            hist_sr = float(
                self._annualized_sharpes(
                    historical.dropna().to_numpy()[:, None], self.risk_free_rate
                )[0]
            )
        else:
            hist_sr = float(historical)

        if isinstance(synthetic, MonteCarloResult):
            if "sharpe_ratio" in synthetic.metrics_distribution.columns:
                sim_srs = synthetic.metrics_distribution["sharpe_ratio"].to_numpy(
                    dtype=np.float64
                )
            else:
                sim_srs = self._annualized_sharpes(
                    synthetic.returns_matrix.to_numpy(dtype=np.float64),
                    self.risk_free_rate,
                )
        elif isinstance(synthetic, CPCVResult):
            sim_srs = synthetic.sharpe_ratios.to_numpy(dtype=np.float64)
        elif isinstance(synthetic, pd.Series):
            sim_srs = synthetic.to_numpy(dtype=np.float64)
        else:
            sim_srs = np.asarray(synthetic, dtype=np.float64)

        sim_srs = sim_srs[np.isfinite(sim_srs)]
        if len(sim_srs) == 0:
            sim_srs = np.array([0.0])

        # In single-strategy benchmark mode, overfitting risk is the probability mass of
        # synthetic/scrambled realizations falling below the historical Sharpe (or < 0),
        # and we also compute centered rank logits around the median synthetic Sharpe.
        pbo = float(np.mean(sim_srs < 0.0))
        prob_loss = float(np.mean(sim_srs < 0.0))

        ranks = (np.argsort(np.argsort(sim_srs)) + 0.5) / len(sim_srs)
        logits = np.log(ranks / (1.0 - ranks))
        is_srs = np.full_like(sim_srs, hist_sr)

        return PBOResult(
            pbo=pbo,
            logits=logits,
            is_sharpes=is_srs,
            oos_sharpes=sim_srs,
            prob_oos_loss=prob_loss,
            degradation_slope=0.0,
            degradation_intercept=float(np.mean(sim_srs)),
            historical_sharpe=hist_sr,
            mode="benchmark",
        )
