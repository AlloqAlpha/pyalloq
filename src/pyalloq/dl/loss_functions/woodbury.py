from __future__ import annotations

import torch
from torch import nn

from pyalloq.dl.estimators.covariance.models.set_covariance import (
    ISABFactorDiagonalHead,
)


class WoodburyGaussianNLLLoss(nn.Module):
    r"""
    Cardinality-Normalized Woodbury Multivariate Gaussian Negative Log-Likelihood Loss:
        \mathcal{L}_{\text{NLL}}(\hat{\Sigma}_t, r_{t+1}; M_t)
        = \frac{1}{N_{\text{active}}} \left( \log |\hat{\Sigma}_t| + r_{t+1}^T \hat{\Sigma}_t^{-1} r_{t+1} \right)

    - When ``pred`` is a factor tuple ``(B_load, Sigma_F, d_diag)`` from ``ISABFactorDiagonalHead``
      or ``SetTransformerCovarianceNet.forward_factors``, evaluates both ``\log |\hat{\Sigma}_t|``
      and ``r_{t+1}^T \hat{\Sigma}_t^{-1} r_{t+1}`` in ``O(N m^2 + m^3)`` time via the Woodbury
      Matrix Identity and Matrix Determinant Lemma without materializing an ``N x N`` matrix.
    - When ``pred`` is a dense ``(B, N, N)`` PSD covariance matrix, evaluates the loss via
      Cholesky factorization over active assets.
    - Dividing by ``N_active`` stabilizes loss scale and gradient norms across variable universe
      cardinalities (e.g., 50-to-500 stock subsampling).
    """

    def __init__(self, normalize_by_cardinality: bool = True) -> None:
        super().__init__()
        self.normalize_by_cardinality = bool(normalize_by_cardinality)

    def forward(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        y_true: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y_true.dim() == 3:
            # (B, H, N) -> take first horizon step or mean over horizon
            y_vec = y_true[:, 0, :]
        else:
            y_vec = y_true

        B, N = y_vec.shape
        device = y_vec.device
        dtype = y_vec.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            m_f = m_bool.to(dtype=dtype)
            y_clean = torch.where(
                m_bool, torch.nan_to_num(y_vec, nan=0.0), torch.zeros_like(y_vec)
            )
            n_active = m_f.sum(dim=-1).clamp(min=1.0)
        else:
            m_bool = None
            y_clean = torch.nan_to_num(y_vec, nan=0.0)
            n_active = torch.full((B,), max(1.0, float(N)), device=device, dtype=dtype)

        if isinstance(pred, tuple) and len(pred) == 3:
            b_load, sigma_f, d_diag = pred
            log_det = ISABFactorDiagonalHead.woodbury_logdet(
                b_load=b_load, sigma_f=sigma_f, d_diag=d_diag, mask=m_bool
            )
            sigma_inv_y = ISABFactorDiagonalHead.woodbury_solve(
                b_load=b_load,
                sigma_f=sigma_f,
                d_diag=d_diag,
                rhs=y_clean,
                mask=m_bool,
            )
            mahalanobis = (y_clean * sigma_inv_y).sum(dim=-1)
        else:
            assert isinstance(pred, torch.Tensor)
            cov = 0.5 * (pred + pred.transpose(-1, -2))
            eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
            if m_bool is not None:
                m_f = m_bool.to(dtype=dtype)
                pair_mask = m_f.unsqueeze(-1) * m_f.unsqueeze(-2)
                cov = cov * pair_mask + eye * (1.0 - pair_mask)
            L_chol, info = torch.linalg.cholesky_ex(cov)
            if bool(torch.any(info != 0).item()):
                L_chol = torch.linalg.cholesky(cov + 1e-8 * eye)
            diag_l = torch.diagonal(L_chol, dim1=-2, dim2=-1)
            log_det = 2.0 * torch.sum(torch.log(diag_l.clamp(min=1e-12)), dim=-1)
            sol = torch.linalg.solve_triangular(
                L_chol, y_clean.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahalanobis = torch.sum(sol.pow(2), dim=-1)

        nll = log_det + mahalanobis
        if self.normalize_by_cardinality:
            nll = nll / n_active
        return nll.mean()


class WoodburySteinLoss(nn.Module):
    r"""
    Cardinality-Normalized James-Stein / KL Divergence Covariance Loss:
        \mathcal{L}_{\text{Stein}}(\hat{\Sigma}, \Sigma_{\text{true}}; M_t)
        = \frac{1}{N_{\text{active}}} \left(
            \operatorname{tr}(\hat{\Sigma}^{-1} \Sigma_{\text{true}})
            - \log |\hat{\Sigma}^{-1} \Sigma_{\text{true}}| - N_{\text{active}}
          \right)

    Supports both low-rank Woodbury factor tuples ``(B_load, Sigma_F, d_diag)`` in
    ``O(N m^2 + m^3)`` time and dense ``(B, N, N)`` covariance tensors.
    """

    def __init__(self, normalize_by_cardinality: bool = True) -> None:
        super().__init__()
        self.normalize_by_cardinality = bool(normalize_by_cardinality)

    def forward(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_cov_or_returns: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = target_cov_or_returns.device
        dtype = target_cov_or_returns.dtype

        if (
            target_cov_or_returns.dim() == 3
            and target_cov_or_returns.shape[-1] == target_cov_or_returns.shape[-2]
        ):
            # Target is a batch of (B, N, N) SPD covariance matrices
            sigma_true = 0.5 * (
                target_cov_or_returns + target_cov_or_returns.transpose(-1, -2)
            )
            B, N, _ = sigma_true.shape
        else:
            # Target is forward returns (B, H, N) or (B, N) -> construct empirical covariance
            if target_cov_or_returns.dim() == 2:
                r = target_cov_or_returns.unsqueeze(1)
            else:
                r = target_cov_or_returns
            B, H, N = r.shape
            r_clean = torch.nan_to_num(r, nan=0.0)
            eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
            sigma_true = (
                torch.bmm(r_clean.transpose(1, 2), r_clean) / float(max(1, H))
                + 1e-4 * eye
            )

        eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            m_f = m_bool.to(dtype=dtype)
            pair_mask = m_f.unsqueeze(-1) * m_f.unsqueeze(-2)
            sigma_true = sigma_true * pair_mask + eye * (1.0 - pair_mask)
            n_active = m_f.sum(dim=-1).clamp(min=1.0)
        else:
            m_bool = None
            n_active = torch.full((B,), max(1.0, float(N)), device=device, dtype=dtype)

        _, logdet_true = torch.linalg.slogdet(sigma_true)

        if isinstance(pred, tuple) and len(pred) == 3:
            b_load, sigma_f, d_diag = pred
            logdet_pred = ISABFactorDiagonalHead.woodbury_logdet(
                b_load=b_load, sigma_f=sigma_f, d_diag=d_diag, mask=m_bool
            )
            sigma_inv_true = ISABFactorDiagonalHead.woodbury_solve(
                b_load=b_load,
                sigma_f=sigma_f,
                d_diag=d_diag,
                rhs=sigma_true,
                mask=m_bool,
            )
            trace_term = torch.diagonal(sigma_inv_true, dim1=-2, dim2=-1)
            if m_bool is not None:
                trace_term = trace_term * m_bool.to(dtype=dtype)
            trace_val = trace_term.sum(dim=-1)
        else:
            assert isinstance(pred, torch.Tensor)
            cov = 0.5 * (pred + pred.transpose(-1, -2))
            if m_bool is not None:
                m_f = m_bool.to(dtype=dtype)
                pair_mask = m_f.unsqueeze(-1) * m_f.unsqueeze(-2)
                cov = cov * pair_mask + eye * (1.0 - pair_mask)
            _, logdet_pred = torch.linalg.slogdet(cov)
            sigma_inv_true = torch.linalg.solve(cov, sigma_true)
            trace_term = torch.diagonal(sigma_inv_true, dim1=-2, dim2=-1)
            if m_bool is not None:
                trace_term = trace_term * m_bool.to(dtype=dtype)
            trace_val = trace_term.sum(dim=-1)

        stein = trace_val + (logdet_pred - logdet_true) - n_active
        if self.normalize_by_cardinality:
            stein = stein / n_active
        return stein.mean()


class SoftMinEVaRLoss(nn.Module):
    r"""
    Differentiable SoftMin / Entropic Value-at-Risk (EVaR) Tail-Risk Surrogate Loss
    (Wood, Roberts & Zohren, DeePM Jan 2026 / arXiv:2601.05975):
        \mathcal{L}_{\text{EVaR}}(w_t, r_{t+1})
        = \frac{1}{\beta} \log \left( \frac{1}{B} \sum_{b=1}^B \exp(-\beta \cdot r_{p,b}^{\text{net}}) \right)
          + \lambda_{\text{dd}} \cdot \operatorname{SoftMaxDrawdown}(r_p^{\text{net}})
    """

    def __init__(
        self,
        beta: float = 10.0,
        cost_bps: float = 5.0,
        drawdown_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.beta = float(max(0.1, beta))
        self.tc_rate = float(cost_bps) / 10000.0
        self.drawdown_weight = float(drawdown_weight)

    def forward(
        self,
        weights: torch.Tensor,
        y_true: torch.Tensor,
        w_current: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y_true.dim() == 3:
            r_next = y_true[:, 0, :]
        else:
            r_next = y_true

        if mask is not None:
            m_f = mask.to(device=weights.device, dtype=weights.dtype)
            w_eff = weights[:, : m_f.shape[1]] * m_f
            r_clean = torch.nan_to_num(r_next, nan=0.0) * m_f
        else:
            w_eff = weights[:, : r_next.shape[1]]
            r_clean = torch.nan_to_num(r_next, nan=0.0)

        gross_ret = (w_eff * r_clean).sum(dim=-1)
        if w_current is not None:
            w_prev = w_current.to(device=w_eff.device, dtype=w_eff.dtype)
            turnover = torch.abs(w_eff - w_prev[:, : w_eff.shape[1]]).sum(dim=-1)
            net_ret = gross_ret - self.tc_rate * turnover
        else:
            net_ret = gross_ret

        B = max(1, net_ret.shape[0])
        scaled = (-self.beta * net_ret).clamp(min=-30.0, max=30.0)
        evar_surrogate = (
            torch.logsumexp(scaled, dim=0)
            - torch.log(
                torch.tensor(float(B), device=net_ret.device, dtype=net_ret.dtype)
            )
        ) / self.beta

        # Smooth cumulative drawdown penalty across batch sequence
        cum_ret = torch.cumsum(net_ret, dim=0)
        running_peak = torch.cummax(cum_ret, dim=0).values
        dd = running_peak - cum_ret
        soft_max_dd = (
            torch.logsumexp((self.beta * dd).clamp(max=30.0), dim=0) / self.beta
        )

        return evar_surrogate + self.drawdown_weight * soft_max_dd
