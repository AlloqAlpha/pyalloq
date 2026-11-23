from __future__ import annotations

import numpy as np
import torch
from pyalloq_core.data import MarketData
from torch.utils.data import Dataset


class UniverseSubsampler:
    r"""
    Multi-Cardinality Subsampling (Universe Dropout) Batch Transform.

    During training on an N-asset universe (e.g., 500 stocks), samples a random active
    sub-universe cardinality per minibatch:
        N_{\text{sub}} \sim \operatorname{LogUniform}(N_{\min}, N_{\text{active}})
    and masks unselected tickers to ``False`` (zeroing their input features). This
    explicitly calibrates Scalable-Softmax (SSMax) and portfolio output heads across
    variable universe sizes.
    """

    def __init__(
        self,
        min_assets: int = 4,
        max_assets: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.min_assets = max(2, int(min_assets))
        self.max_assets = int(max_assets) if max_assets is not None else None
        self.rng = np.random.default_rng(random_state)

    def __call__(
        self,
        batch_x: torch.Tensor,
        batch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch_x : torch.Tensor
            Input batch tensor of shape ``(B, L, N, F)`` or ``(B, L, N)``.
        batch_mask : torch.Tensor, optional
            Existing tradability mask of shape ``(B, N)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(subsampled_x, subsampled_mask)`` with ``N_sub`` active assets.
        """
        B = batch_x.shape[0]
        N = batch_x.shape[2]
        device = batch_x.device

        if batch_mask is None:
            base_mask = torch.ones((B, N), device=device, dtype=torch.bool)
        else:
            base_mask = batch_mask.to(device=device, dtype=torch.bool).clone()

        n_low = min(self.min_assets, N)
        n_high = min(self.max_assets, N) if self.max_assets is not None else N
        if n_high <= n_low:
            n_sub = n_low
        else:
            log_low = float(np.log(max(2, n_low)))
            log_high = float(np.log(max(2, n_high)))
            n_sub = int(
                np.clip(
                    round(float(np.exp(self.rng.uniform(log_low, log_high)))),
                    n_low,
                    n_high,
                )
            )

        out_mask = torch.zeros_like(base_mask)
        for b in range(B):
            active_idx = torch.nonzero(base_mask[b], as_tuple=False).view(-1)
            if len(active_idx) <= n_sub:
                out_mask[b] = base_mask[b]
            else:
                perm = torch.randperm(len(active_idx), device=device)[:n_sub]
                chosen = active_idx[perm]
                out_mask[b, chosen] = True

        if batch_x.dim() == 4:
            x_masked = torch.where(
                out_mask.view(B, 1, N, 1), batch_x, torch.zeros_like(batch_x)
            )
        else:
            x_masked = torch.where(
                out_mask.view(B, 1, N), batch_x, torch.zeros_like(batch_x)
            )
        return x_masked, out_mask


class MarketDataset(Dataset):
    """
    Point-in-Time Market Tensor Dataset generating 4D windows ``(Lookback, N_Assets, N_Features)``.

    Parameters
    ----------
    data : MarketData
        Source market data container.
    lookback_window : int, default=60
        Historical context window length ``L``.
    horizon : int, default=1
        Forward prediction target horizon ``H``.
    dtype : torch.dtype, default=torch.float32
        Output tensor dtype.
    return_mask : bool, default=False
        When ``True``, ``__getitem__`` returns a 3-tuple ``(X_tensor, y_tensor, mask_tensor)``
        where ``mask_tensor`` in ``{True, False}^N`` indicates assets with complete finite
        price observations strictly up to decision timestamp ``t`` (no look-ahead into
        future ``t+1..t+H`` prices).
        When ``False`` (default), returns ``(X_tensor, y_tensor)`` for backward compatibility.
    allow_ragged : bool, default=False
        When ``True``, preserves rows with partial asset histories (e.g., pre-IPO or
        delisted tickers) without synthetic back-casting or full-panel truncation.
    """

    def __init__(
        self,
        data: MarketData,
        lookback_window: int = 60,
        horizon: int = 1,
        dtype: torch.dtype = torch.float32,
        return_mask: bool = False,
        allow_ragged: bool = False,
    ) -> None:
        self.lookback_window = lookback_window
        self.horizon = horizon
        self._dtype = dtype
        self.return_mask = bool(return_mask)
        self.allow_ragged = bool(allow_ragged or return_mask)

        raw_prices = data.prices
        raw_returns = raw_prices.pct_change(fill_method=None)

        if self.allow_ragged:
            # Drop only leading all-NaN rows while keeping ragged IPO/delisting columns
            valid_rows = raw_returns.notna().any(axis=1)
            if valid_rows.any():
                first_valid = int(np.argmax(valid_rows.to_numpy()))
                df_returns_raw = raw_returns.iloc[first_valid:]
                df_prices_aligned = raw_prices.loc[df_returns_raw.index]
            else:
                df_returns_raw = raw_returns.iloc[1:]
                df_prices_aligned = raw_prices.iloc[1:]
            # Point-in-time finite check on both price and return at each timestamp
            self._finite_step_mask = (
                df_prices_aligned.notna().to_numpy(dtype=bool)
                & np.isfinite(df_prices_aligned.to_numpy(dtype=np.float64))
                & df_returns_raw.notna().to_numpy(dtype=bool)
                & np.isfinite(df_returns_raw.to_numpy(dtype=np.float64))
            )
            df_returns = df_returns_raw.fillna(0.0)
        else:
            df_returns = raw_returns.dropna()
            self._finite_step_mask = np.ones(df_returns.shape, dtype=bool)

        self.assets = df_returns.columns.tolist()
        self.dates = df_returns.index.tolist()

        features_list = [df_returns.to_numpy(dtype=np.float64)]

        if data.features is not None:
            for df_feature in data.features.values():
                if (
                    df_feature.shape[1] == 1
                    and len(self.assets) > 1
                    and df_feature.columns[0] not in self.assets
                ):
                    col_series = (
                        df_feature.iloc[:, 0]
                        .reindex(df_returns.index)
                        .ffill()
                        .bfill()
                        .fillna(0.0)
                    )
                    aligned_np = np.repeat(
                        col_series.to_numpy(dtype=np.float64)[:, None],
                        len(self.assets),
                        axis=1,
                    )
                else:
                    aligned_feat = df_feature.reindex(
                        index=df_returns.index, columns=self.assets
                    )
                    if self.allow_ragged:
                        # Avoid bfill (which leaks future post-IPO values backward into pre-IPO dates)
                        aligned_feat = aligned_feat.ffill().fillna(0.0)
                    else:
                        aligned_feat = aligned_feat.ffill().bfill().fillna(0.0)
                    aligned_np = aligned_feat.to_numpy(dtype=np.float64)
                features_list.append(aligned_np)

        stacked_features = np.nan_to_num(
            np.stack(features_list, axis=-1), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.X_data = torch.tensor(stacked_features, dtype=self._dtype)
        self.y_data = torch.tensor(
            np.nan_to_num(
                df_returns.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
            ),
            dtype=self._dtype,
        )
        self.mask_steps = torch.tensor(self._finite_step_mask, dtype=torch.bool)

        self.valid_length = len(df_returns) - self.lookback_window - self.horizon + 1

        if self.valid_length <= 0:
            raise ValueError(
                f"Data length ({len(df_returns)}) is too short for lookback"
                f"({self.lookback_window}) and horizon ({self.horizon})"
            )

    def __len__(self) -> int:
        return self.valid_length

    def get_tradability_mask(self, idx: int) -> torch.Tensor:
        """
        Returns the Point-in-Time boolean tradability mask ``M_t in {True, False}^N``
        for sample ``idx``, relying strictly on lookback observations ``[idx, idx + lookback_window)``
        available at decision time ``t`` (zero look-ahead into ``y``).
        """
        x_start = idx
        x_end = idx + self.lookback_window
        return self.mask_steps[x_start:x_end].all(dim=0)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Returns
        -------
        tuple
            - ``X``: Tensor of shape ``(Lookback, N_Assets, N_Features)``
            - ``y``: Tensor of shape ``(Horizon, N_Assets)``
            - ``mask`` (only when ``return_mask=True``): Boolean tensor of shape ``(N_Assets,)``
        """
        x_start = idx
        x_end = idx + self.lookback_window
        X_tensor = self.X_data[x_start:x_end]

        y_start = x_end
        y_end = y_start + self.horizon
        y_tensor = self.y_data[y_start:y_end]

        if self.return_mask:
            mask_tensor = self.get_tradability_mask(idx)
            X_masked = torch.where(
                mask_tensor.view(1, -1, 1), X_tensor, torch.zeros_like(X_tensor)
            )
            return (X_masked, y_tensor, mask_tensor)

        return (X_tensor, y_tensor)
