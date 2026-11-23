from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from pyalloq.dl.layers.set_transformer import (
    SAB,
    NormStyle,
    SetBlockType,
    SetTransformerBackbone,
    TemporalEncoderType,
)

CovarianceHeadType = Literal["factor_diag", "gram_corr", "rie_spectral"]


class ISABFactorDiagonalHead(nn.Module):
    r"""
    ISAB Low-Rank Factor-Plus-Diagonal PSD Covariance Head (Lin et al., ACM ICAIF 2021):
        \hat{\Sigma}_t = B_t \Sigma_{F,t} B_t^T + D_t

    - ``B_t = \phi_B(Z_t) \in \mathbb{R}^{B \times N \times m}``: 1D permutation-equivariant
      dynamic factor loadings (zeroed on padded slots).
    - ``D_t = \operatorname{diag}(\operatorname{softplus}(\phi_D(Z_t)) + \epsilon_d)``:
      1D permutation-equivariant strictly positive idiosyncratic variances.
    - ``\Sigma_{F,t} = (1 / d) H_{I,t} H_{I,t}^T + \epsilon_F I_m``: strictly permutation-invariant
      ``m x m`` PSD factor covariance built from the ISAB inducing-point state ``H_{I,t}``.

    Guarantees exact 2D congruence equivariance ``\Sigma(\Pi X) = \Pi \Sigma(X) \Pi^T`` to
    machine precision and supports exact ``O(N m^2 + m^3)`` inversion and log-determinant
    evaluation via the Woodbury Matrix Identity and Matrix Determinant Lemma.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_factors: int = 16,
        eps_diag: float = 1e-3,
        eps_factor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_factors = int(num_factors)
        self.eps_diag = float(eps_diag)
        self.eps_factor = float(eps_factor)

        self.factor_loading_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_factors),
        )
        self.idio_var_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.factor_state_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward_factors(
        self,
        z_assets: torch.Tensor,
        h_inducing: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the low-rank factor tuple ``(B_load, Sigma_F, d_diag)``:
        - ``B_load``: shape ``(B, N, m)``
        - ``Sigma_F``: shape ``(B, m, m)``
        - ``d_diag``: shape ``(B, N)``
        """
        B, N, d = z_assets.shape
        m = h_inducing.shape[1]
        device = z_assets.device
        dtype = z_assets.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            m_f = m_bool.to(dtype=dtype)
            z_clean = torch.where(
                m_bool.unsqueeze(-1),
                torch.nan_to_num(z_assets, nan=0.0),
                torch.zeros_like(z_assets),
            )
        else:
            m_bool = None
            m_f = torch.ones((B, N), device=device, dtype=dtype)
            z_clean = z_assets

        # Scale factor loadings by 1 / sqrt(m) to maintain well-conditioned spectrum
        b_load = self.factor_loading_proj(z_clean) / (float(max(1, m)) ** 0.5)
        b_load = b_load * m_f.unsqueeze(-1)

        d_raw = self.idio_var_proj(z_clean).squeeze(-1)
        d_active = torch.nn.functional.softplus(d_raw) + self.eps_diag
        # Padded slots get unit diagonal variance 1.0 (and 0 factor loading) so the full N x N
        # matrix remains invertible while decoupled from active assets
        if m_bool is not None:
            d_diag = torch.where(m_bool, d_active, torch.ones_like(d_active))
        else:
            d_diag = d_active

        h_proj = self.factor_state_proj(h_inducing)
        eye_m = torch.eye(m, device=device, dtype=dtype).unsqueeze(0)
        sigma_f = (
            torch.bmm(h_proj, h_proj.transpose(1, 2)) / float(max(1, d))
            + self.eps_factor * eye_m
        )
        sigma_f = 0.5 * (sigma_f + sigma_f.transpose(1, 2))

        return b_load, sigma_f, d_diag

    def forward(
        self,
        z_assets: torch.Tensor,
        h_inducing: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Materializes the dense ``(B, N, N)`` PSD covariance matrix ``B_t Sigma_{F,t} B_t^T + D_t``.
        """
        b_load, sigma_f, d_diag = self.forward_factors(
            z_assets=z_assets, h_inducing=h_inducing, mask=mask
        )
        bf = torch.bmm(b_load, sigma_f)
        cov = torch.bmm(bf, b_load.transpose(1, 2)) + torch.diag_embed(d_diag)
        return 0.5 * (cov + cov.transpose(1, 2))

    @staticmethod
    def woodbury_capacitance(
        b_load: torch.Tensor,
        sigma_f: torch.Tensor,
        d_diag: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Computes the ``m x m`` Woodbury capacitance matrix in ``O(N m^2 + m^3)``:
            M_t = \Sigma_{F,t}^{-1} + B_t^T D_t^{-1} B_t
        Returns ``(M_t, sigma_f_inv, d_inv)``.
        """
        device = b_load.device
        dtype = b_load.dtype
        B, _N, m = b_load.shape

        if mask is not None:
            m_f = mask.to(device=device, dtype=dtype)
            b_eff = b_load * m_f.unsqueeze(-1)
            d_inv = torch.where(
                mask.to(device=device, dtype=torch.bool),
                1.0 / d_diag.clamp(min=1e-12),
                torch.zeros_like(d_diag),
            )
        else:
            b_eff = b_load
            d_inv = 1.0 / d_diag.clamp(min=1e-12)

        eye_m = torch.eye(m, device=device, dtype=dtype).unsqueeze(0).expand(B, m, m)
        sigma_f_sym = 0.5 * (sigma_f + sigma_f.transpose(1, 2))
        sigma_f_inv = torch.linalg.solve(sigma_f_sym, eye_m)

        # B_t^T D_t^{-1} B_t: shape (B, m, m)
        d_inv_b = d_inv.unsqueeze(-1) * b_eff  # (B, N, m)
        bt_dinv_b = torch.bmm(b_eff.transpose(1, 2), d_inv_b)
        cap_m = sigma_f_inv + bt_dinv_b
        cap_m = 0.5 * (cap_m + cap_m.transpose(1, 2))
        return cap_m, sigma_f_inv, d_inv

    @staticmethod
    def woodbury_logdet(
        b_load: torch.Tensor,
        sigma_f: torch.Tensor,
        d_diag: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Evaluates ``\log |\hat{\Sigma}_t|`` in ``O(N m^2 + m^3)`` via the Matrix Determinant Lemma:
            \log |\hat{\Sigma}_t| = \sum_{i \in \text{active}} \log d_{i,t}
                                    + \log |\Sigma_{F,t}| + \log |M_t|
        """
        cap_m, _, _ = ISABFactorDiagonalHead.woodbury_capacitance(
            b_load=b_load, sigma_f=sigma_f, d_diag=d_diag, mask=mask
        )
        if mask is not None:
            m_bool = mask.to(device=d_diag.device, dtype=torch.bool)
            log_d = torch.where(
                m_bool,
                torch.log(d_diag.clamp(min=1e-12)),
                torch.zeros_like(d_diag),
            ).sum(dim=-1)
        else:
            log_d = torch.log(d_diag.clamp(min=1e-12)).sum(dim=-1)

        _, logabsdet_f = torch.linalg.slogdet(0.5 * (sigma_f + sigma_f.transpose(1, 2)))
        _, logabsdet_m = torch.linalg.slogdet(cap_m)
        return log_d + logabsdet_f + logabsdet_m

    @staticmethod
    def woodbury_solve(
        b_load: torch.Tensor,
        sigma_f: torch.Tensor,
        d_diag: torch.Tensor,
        rhs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Solves ``\hat{\Sigma}_t x = v`` in ``O(N m^2 + m^3)`` via the Woodbury Matrix Identity:
            \hat{\Sigma}_t^{-1} v = D_t^{-1} v - D_t^{-1} B_t M_t^{-1} B_t^T D_t^{-1} v
        Supports ``rhs`` of shape ``(B, N)`` or ``(B, N, K)``.
        """
        squeezed = rhs.dim() == 2
        if squeezed:
            v = rhs.unsqueeze(-1)
        else:
            v = rhs

        cap_m, _, d_inv = ISABFactorDiagonalHead.woodbury_capacitance(
            b_load=b_load, sigma_f=sigma_f, d_diag=d_diag, mask=mask
        )
        if mask is not None:
            m_f = mask.to(device=b_load.device, dtype=b_load.dtype).unsqueeze(-1)
            b_eff = b_load * m_f
            v = v * m_f
        else:
            b_eff = b_load

        d_inv_v = d_inv.unsqueeze(-1) * v  # (B, N, K)
        bt_dinv_v = torch.bmm(b_eff.transpose(1, 2), d_inv_v)  # (B, m, K)
        m_inv_bt_v = torch.linalg.solve(cap_m, bt_dinv_v)  # (B, m, K)
        b_m_inv = torch.bmm(b_eff, m_inv_bt_v)  # (B, N, K)
        sol = d_inv_v - d_inv.unsqueeze(-1) * b_m_inv

        return sol.squeeze(-1) if squeezed else sol

    @staticmethod
    def woodbury_gmv_weights(
        b_load: torch.Tensor,
        sigma_f: torch.Tensor,
        d_diag: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Computes exact analytical Global Minimum Variance (GMV) weights:
            w_{\text{GMV}} = (\hat{\Sigma}_t^{-1} \mathbf{1}_{\text{active}})
                             / (\mathbf{1}_{\text{active}}^T \hat{\Sigma}_t^{-1} \mathbf{1}_{\text{active}})
        in ``O(N m^2 + m^3)`` time via ``woodbury_solve``.
        """
        B, N, _ = b_load.shape
        device = b_load.device
        dtype = b_load.dtype

        if mask is not None:
            ones = mask.to(device=device, dtype=dtype)
        else:
            ones = torch.ones((B, N), device=device, dtype=dtype)

        sigma_inv_ones = ISABFactorDiagonalHead.woodbury_solve(
            b_load=b_load,
            sigma_f=sigma_f,
            d_diag=d_diag,
            rhs=ones,
            mask=mask,
        )
        denom = (ones * sigma_inv_ones).sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return sigma_inv_ones / denom


class EquivariantGramCorrHead(nn.Module):
    r"""
    Volatility-Scaled Equivariant Gram Correlation Head:
        \hat{\Gamma}_t = (1 - \lambda_t) \tilde{U}_t \tilde{U}_t^T + \lambda_t I_N
        \hat{\Sigma}_t = D_{\sigma,t} \hat{\Gamma}_t D_{\sigma,t}

    - Each asset embedding ``z_{i,t}`` maps to ``u_{i,t} \in \mathbb{R}^r`` with L2-normalized
      rows ``\tilde{u}_{i,t} = u_{i,t} / (\|u_{i,t}\|_2 + \epsilon)``, producing a unit-diagonal
      PSD correlation matrix.
    - Shrinkage intensity ``\lambda_t \in (0, 1)`` is predicted from the permutation-invariant
      ``[REGIME]`` token ``z_{\text{regime}}``.
    - Marginal volatilities ``D_{\sigma,t}`` can optionally anchor to Point-in-Time implied
      volatilities ``implied_vol``.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        corr_rank: int = 16,
        eps_vol: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.corr_rank = int(corr_rank)
        self.eps_vol = float(eps_vol)

        self.corr_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.corr_rank),
        )
        self.vol_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.shrinkage_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_assets: torch.Tensor,
        z_regime: torch.Tensor,
        mask: torch.Tensor | None = None,
        implied_vol: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = z_assets.shape
        device = z_assets.device
        dtype = z_assets.dtype

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            m_f = m_bool.to(dtype=dtype)
            z_clean = torch.where(
                m_bool.unsqueeze(-1),
                torch.nan_to_num(z_assets, nan=0.0),
                torch.zeros_like(z_assets),
            )
        else:
            m_bool = None
            m_f = torch.ones((B, N), device=device, dtype=dtype)
            z_clean = z_assets

        u_raw = self.corr_proj(z_clean)
        u_norm = u_raw / (torch.linalg.norm(u_raw, dim=-1, keepdim=True) + 1e-8)
        u_norm = u_norm * m_f.unsqueeze(-1)

        gram = torch.bmm(u_norm, u_norm.transpose(1, 2))
        lam = self.shrinkage_head(z_regime).clamp(min=0.05, max=0.95).view(B, 1, 1)
        eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        corr = (1.0 - lam) * gram + lam * eye
        # Enforce exact unit diagonal on correlation matrix
        corr = corr * (1.0 - eye) + eye

        vol_pred = (
            torch.nn.functional.softplus(self.vol_proj(z_clean).squeeze(-1))
            + self.eps_vol
        )
        if implied_vol is not None:
            iv_t = torch.nan_to_num(
                implied_vol.to(device=device, dtype=dtype), nan=0.0
            ).clamp(min=self.eps_vol)
            if iv_t.dim() == 1:
                iv_t = iv_t.unsqueeze(0).expand(B, N)
            vol_pred = 0.5 * vol_pred + 0.5 * iv_t

        if m_bool is not None:
            vol_eff = torch.where(m_bool, vol_pred, torch.ones_like(vol_pred))
            pair_mask = m_f.unsqueeze(-1) * m_f.unsqueeze(-2)
            corr = corr * pair_mask + eye * (1.0 - pair_mask)
        else:
            vol_eff = vol_pred

        cov = corr * (vol_eff.unsqueeze(-1) * vol_eff.unsqueeze(-2))
        return 0.5 * (cov + cov.transpose(1, 2))


class RIESpectralHead(nn.Module):
    r"""
    Rotation- and Permutation-Invariant Spectral Cleaning Covariance Head
    (`RIESpectralHead`, Bongiorno, Manolakis & Mantegna, 2026 / arXiv:2507.01918).

    1. Computes the empirical sample correlation matrix ``C_t`` from lookback returns
       ``x[:, :, :, 0]`` and performs symmetric eigendecomposition ``C_t = V_t \Lambda_t V_t^T``.
    2. Feeds the unordered set of empirical eigenvalues conditioned on the aspect ratio
       ``q_{\text{in}} = N_{\text{active}} / L`` through a 1D permutation-equivariant
       Set Attention block to predict strictly positive cleaned eigenvalues ``\lambda_{k,t}^*``
       normalized so ``\sum_k \lambda_{k,t}^* = N_{\text{active}}``.
    3. Recombines ``\hat{C}_t = V_t \operatorname{diag}(\lambda_t^*) V_t^T`` with neural
       marginal volatilities ``D_{\sigma,t}`` predicted from ``z_assets`` to form
       ``\hat{\Sigma}_t = D_{\sigma,t} \hat{C}_t D_{\sigma,t}``.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        eps_eig: float = 1e-3,
        eps_vol: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.eps_eig = float(eps_eig)
        self.eps_vol = float(eps_vol)

        self.eig_in_proj = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.eig_sab = SAB(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            use_ssmax=True,
        )
        self.eig_out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.vol_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self.last_raw_eigenvalues_: torch.Tensor | None = None
        self.last_cleaned_eigenvalues_: torch.Tensor | None = None

    def forward(
        self,
        x_returns: torch.Tensor,
        z_assets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_returns : torch.Tensor
            Historical return tensor of shape ``(B, L, N)``.
        z_assets : torch.Tensor
            Asset embeddings of shape ``(B, N, hidden_dim)``.
        mask : torch.Tensor, optional
            Boolean tradability mask of shape ``(B, N)``.
        """
        B, L, N = x_returns.shape
        device = x_returns.device
        dtype = x_returns.dtype
        eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)

        if mask is not None:
            m_bool = mask.to(device=device, dtype=torch.bool)
            m_f = m_bool.to(dtype=dtype)
            r_clean = torch.where(
                m_bool.unsqueeze(1),
                torch.nan_to_num(x_returns, nan=0.0),
                torch.zeros_like(x_returns),
            )
            z_clean = torch.where(
                m_bool.unsqueeze(-1),
                torch.nan_to_num(z_assets, nan=0.0),
                torch.zeros_like(z_assets),
            )
            n_active = m_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            m_bool = None
            m_f = torch.ones((B, N), device=device, dtype=dtype)
            r_clean = torch.nan_to_num(x_returns, nan=0.0)
            z_clean = z_assets
            n_active = torch.full(
                (B, 1), max(1.0, float(N)), device=device, dtype=dtype
            )

        # Sample correlation matrix C_t in (B, N, N)
        r_centered = r_clean - r_clean.mean(dim=1, keepdim=True)
        r_centered = r_centered * m_f.unsqueeze(1)
        sample_cov = torch.bmm(r_centered.transpose(1, 2), r_centered) / float(
            max(1, L - 1)
        )
        std = torch.sqrt(torch.diagonal(sample_cov, dim1=-2, dim2=-1).clamp(min=1e-8))
        sample_corr = sample_cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-8)

        if m_bool is not None and not bool(torch.all(m_bool).item()):
            # Shift active diagonal by +2.0 while keeping masked rows/columns/diagonals at 0.0.
            # Even when N_active > L (where the active correlation block has 0.0 eigenvalues),
            # active eigenvalues are shifted to >= 2.0 while masked eigenvalues stay at 0.0,
            # completely preventing degenerate 0.0 null-space mixing between active and masked slots.
            pair_mask = m_f.unsqueeze(-1) * m_f.unsqueeze(-2)
            active_diag = torch.diag_embed(m_f)
            sample_corr = sample_corr * pair_mask * (1.0 - eye) + 3.0 * active_diag

            corr_f64 = 0.5 * (sample_corr + sample_corr.transpose(1, 2)).to(
                torch.float64
            )
            eigvals_f64, eigvecs_f64 = torch.linalg.eigh(corr_f64)
            eig_mask = eigvals_f64 > 1.0
            eig_mask_f = eig_mask.to(dtype=dtype)
            eigvals = torch.where(
                eig_mask,
                (eigvals_f64 - 2.0).clamp(min=0.0),
                torch.zeros_like(eigvals_f64),
            ).to(dtype=dtype)
            eigvecs = eigvecs_f64.to(dtype=dtype)
        else:
            sample_corr = sample_corr * (1.0 - eye) + eye
            sample_corr = 0.5 * (sample_corr + sample_corr.transpose(1, 2))
            corr_f64 = sample_corr.to(torch.float64)
            eigvals_f64, eigvecs_f64 = torch.linalg.eigh(corr_f64)
            eigvals = eigvals_f64.clamp(min=0.0).to(dtype=dtype)
            eigvecs = eigvecs_f64.to(dtype=dtype)
            eig_mask = None
            eig_mask_f = torch.ones((B, N), device=device, dtype=dtype)

        # Aspect ratio q_in = N_active / L
        q_in = (n_active / float(max(1, L))).unsqueeze(-1).expand(B, N, 1)
        eig_tokens = torch.cat([eigvals.unsqueeze(-1), q_in], dim=-1)  # (B, N, 2)
        if eig_mask is not None:
            eig_tokens = eig_tokens * eig_mask_f.unsqueeze(-1)
        eig_emb = self.eig_in_proj(eig_tokens)
        eig_attended = self.eig_sab(eig_emb, mask=eig_mask)
        lam_clean = (
            torch.nn.functional.softplus(self.eig_out_proj(eig_attended).squeeze(-1))
            + self.eps_eig
        ) * eig_mask_f
        # Normalize cleaned trace strictly to N_active over active eigenvalues
        lam_clean = lam_clean * (
            n_active / lam_clean.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        )

        with torch.no_grad():
            self.last_raw_eigenvalues_ = eigvals.detach()
            self.last_cleaned_eigenvalues_ = lam_clean.detach()

        # Reconstruct rotationally invariant correlation matrix V diag(lam^*) V^T
        v_scaled = eigvecs * lam_clean.unsqueeze(1)
        corr_clean = torch.bmm(v_scaled, eigvecs.transpose(1, 2))
        corr_clean = 0.5 * (corr_clean + corr_clean.transpose(1, 2))
        if m_bool is not None:
            pair_mask = m_f.unsqueeze(-1) * m_f.unsqueeze(-2)
            corr_clean = corr_clean * pair_mask + eye * (1.0 - pair_mask)
        # Rescale active entries to exact unit diagonal
        diag_c = torch.diagonal(corr_clean, dim1=-2, dim2=-1).clamp(min=1e-8)
        inv_sqrt_c = torch.rsqrt(diag_c)
        corr_clean = corr_clean * (inv_sqrt_c.unsqueeze(-1) * inv_sqrt_c.unsqueeze(-2))
        if m_bool is not None:
            pair_mask = m_f.unsqueeze(-1) * m_f.unsqueeze(-2)
            corr_clean = corr_clean * pair_mask + eye * (1.0 - pair_mask)

        vol_pred = (
            torch.nn.functional.softplus(self.vol_proj(z_clean).squeeze(-1))
            + self.eps_vol
        )
        if m_bool is not None:
            vol_eff = torch.where(m_bool, vol_pred, torch.ones_like(vol_pred))
        else:
            vol_eff = vol_pred

        cov = corr_clean * (vol_eff.unsqueeze(-1) * vol_eff.unsqueeze(-2))
        return 0.5 * (cov + cov.transpose(1, 2))


class SetTransformerCovarianceNet(nn.Module):
    r"""
    Size- and Permutation-Equivariant Set Transformer Deep Covariance Network.

    Unifies ``SetTransformerBackbone`` with three modular PSD covariance heads:
    - ``head_type="factor_diag"`` (Default): ``ISABFactorDiagonalHead`` with exact
      ``O(N m^2 + m^3)`` Woodbury inverse, log-det, and ``forward_factors`` pathway.
    - ``head_type="gram_corr"``: ``EquivariantGramCorrHead`` with ``[REGIME]``-conditioned
      correlation shrinkage and optional PIT implied-volatility anchoring.
    - ``head_type="rie_spectral"``: ``RIESpectralHead`` rotation-invariant spectral cleaning.

    Also implements ``predict_returns_and_cov(x)`` so it plugs directly into
    ``UnifiedE2EPolicyNetwork`` and ``DifferentiableKKTAllocatorModule`` as a drop-in
    replacement for ``CRISPSparseSTGAT``.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_inds: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_type: CovarianceHeadType = "factor_diag",
        use_ssmax: bool = True,
        block_type: SetBlockType = "isab",
        temporal_type: TemporalEncoderType = "lstm",
        temporal_encoder: nn.Module | None = None,
        norm_style: NormStyle = "pre",
        directed_delay: bool = False,
        static_metadata_dim: int = 0,
        eps_diag: float = 1e-3,
    ) -> None:
        super().__init__()
        self.n_features = int(max(1, n_features))
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_inds = int(num_inds)
        self.head_type: CovarianceHeadType = head_type

        self.backbone = SetTransformerBackbone(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_inds=self.num_inds,
            num_layers=num_layers,
            dropout=dropout,
            use_ssmax=use_ssmax,
            block_type=block_type,
            temporal_type=temporal_type,
            temporal_encoder=temporal_encoder,
            norm_style=norm_style,
            directed_delay=directed_delay,
            static_metadata_dim=static_metadata_dim,
        )

        self.factor_diag_head = ISABFactorDiagonalHead(
            hidden_dim=self.hidden_dim,
            num_factors=self.num_inds,
            eps_diag=eps_diag,
            eps_factor=eps_diag,
        )
        self.gram_corr_head = EquivariantGramCorrHead(
            hidden_dim=self.hidden_dim,
            corr_rank=self.num_inds,
            eps_vol=eps_diag,
        )
        self.rie_spectral_head = RIESpectralHead(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            eps_eig=eps_diag,
            eps_vol=eps_diag,
        )
        self.mu_head = nn.Linear(self.hidden_dim, 1)

        self.last_factors_: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (
            None
        )
        self.last_regime_token_: torch.Tensor | None = None

    def forward_factors(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        static_metadata: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the low-rank factor tuple ``(B_load, Sigma_F, d_diag)`` in ``O(N m)`` time.
        """
        z_assets, h_inducing, z_regime = self.backbone(
            x=x,
            mask=mask,
            attn_bias=attn_bias,
            static_metadata=static_metadata,
        )
        self.last_regime_token_ = z_regime.detach()
        factors = self.factor_diag_head.forward_factors(
            z_assets=z_assets,
            h_inducing=h_inducing,
            mask=mask,
        )
        self.last_factors_ = factors
        return factors

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        static_metadata: torch.Tensor | None = None,
        implied_vol: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns a strictly Positive Definite ``(B, N, N)`` covariance matrix satisfying
        ``Sigma(Pi X) = Pi Sigma(X) Pi^T``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        z_assets, h_inducing, z_regime = self.backbone(
            x=x,
            mask=mask,
            attn_bias=attn_bias,
            static_metadata=static_metadata,
        )
        self.last_regime_token_ = z_regime.detach()

        if self.head_type == "factor_diag":
            b_load, sigma_f, d_diag = self.factor_diag_head.forward_factors(
                z_assets=z_assets, h_inducing=h_inducing, mask=mask
            )
            self.last_factors_ = (b_load, sigma_f, d_diag)
            bf = torch.bmm(b_load, sigma_f)
            cov = torch.bmm(bf, b_load.transpose(1, 2)) + torch.diag_embed(d_diag)
            return 0.5 * (cov + cov.transpose(1, 2))

        if self.head_type == "gram_corr":
            return self.gram_corr_head(
                z_assets=z_assets,
                z_regime=z_regime,
                mask=mask,
                implied_vol=implied_vol,
            )

        if self.head_type == "rie_spectral":
            return self.rie_spectral_head(
                x_returns=x[:, :, :, 0],
                z_assets=z_assets,
                mask=mask,
            )

        raise ValueError(f"Unsupported head_type: {self.head_type}")

    def predict_returns_and_cov(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts expected return vector ``mu`` ``(B, N)``, PSD covariance matrix ``cov``
        ``(B, N, N)``, and normalized factor similarity adjacency ``(B, N, N)`` for
        drop-in compatibility with ``UnifiedE2EPolicyNetwork`` and ``DifferentiableKKTAllocatorModule``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        z_assets, h_inducing, z_regime = self.backbone(
            x=x, mask=mask, attn_bias=attn_bias
        )
        self.last_regime_token_ = z_regime.detach()
        mu = self.mu_head(z_assets).squeeze(-1)
        if mask is not None:
            mu = mu * mask.to(device=mu.device, dtype=mu.dtype)

        b_load, sigma_f, d_diag = self.factor_diag_head.forward_factors(
            z_assets=z_assets, h_inducing=h_inducing, mask=mask
        )
        self.last_factors_ = (b_load, sigma_f, d_diag)

        if self.head_type == "factor_diag":
            bf = torch.bmm(b_load, sigma_f)
            cov = torch.bmm(bf, b_load.transpose(1, 2)) + torch.diag_embed(d_diag)
            cov = 0.5 * (cov + cov.transpose(1, 2))
        elif self.head_type == "gram_corr":
            cov = self.gram_corr_head(z_assets=z_assets, z_regime=z_regime, mask=mask)
        else:
            cov = self.rie_spectral_head(
                x_returns=x[:, :, :, 0], z_assets=z_assets, mask=mask
            )

        std_vec = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp(min=1e-8))
        corr_adj = torch.relu(
            cov / (std_vec.unsqueeze(-1) * std_vec.unsqueeze(-2) + 1e-8)
        )
        return mu, cov, corr_adj
