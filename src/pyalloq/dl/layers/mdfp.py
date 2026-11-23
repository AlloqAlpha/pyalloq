from typing import Any

import torch
from torch import nn


def _factor_cov_matvec(
    z: torch.Tensor,
    b_load: torch.Tensor,
    sigma_f: torch.Tensor,
    d_diag: torch.Tensor,
) -> torch.Tensor:
    r"""
    Evaluates the quadratic risk matrix-vector product:
        \hat{\Sigma} z = B (\Sigma_F (B^T z)) + D \odot z
    in ``O(N m)`` time and memory without materializing an ``N x N`` matrix.

    Shapes:
        z: ``(B, H, N)``
        b_load: ``(B, N, m)`` or ``(B, H, N, m)``
        sigma_f: ``(B, m, m)`` or ``(B, H, m, m)``
        d_diag: ``(B, N)`` or ``(B, H, N)``
    """
    B, H, N = z.shape
    if b_load.dim() == 3:
        b_exp = b_load.unsqueeze(1).expand(B, H, N, -1)
    else:
        b_exp = b_load
    m = b_exp.shape[-1]

    if sigma_f.dim() == 3:
        sf_exp = sigma_f.unsqueeze(1).expand(B, H, m, m)
    else:
        sf_exp = sigma_f

    if d_diag.dim() == 2:
        d_exp = d_diag.unsqueeze(1).expand(B, H, N)
    else:
        d_exp = d_diag

    # B^T z: (B, H, m, 1)
    bt_z = torch.matmul(b_exp.transpose(-2, -1), z.unsqueeze(-1))
    # Sigma_F (B^T z): (B, H, m, 1)
    sf_bt_z = torch.matmul(sf_exp, bt_z)
    # B (Sigma_F (B^T z)): (B, H, N)
    low_rank_term = torch.matmul(b_exp, sf_bt_z).squeeze(-1)
    return low_rank_term + d_exp * z


def _mdfp_step(
    z: torch.Tensor,
    mu_seq: torch.Tensor,
    cov_seq: torch.Tensor | None,
    w_current: torch.Tensor,
    gamma: float,
    kappa: float,
    eta: float,
    discount: float,
    smooth_eps: float,
    cov_factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Single Entropic Mirror-Descent Fixed-Point (MDFP) operator evaluation z^{m+1} = F(z^m, theta).

    Shapes:
        z: (B, H, N) - multi-period weight trajectory on the probability simplex
        mu_seq: (B, H, N) - predicted expected return trajectory
        cov_seq: (B, H, N, N) - predicted or empirical covariance trajectory (optional if cov_factors given)
        w_current: (B, N) - pre-trade starting weights w_{t^-}
        cov_factors: optional low-rank factor tuple (B_load, Sigma_F, d_diag) for O(Nm) risk matvec
    """
    B, H, N = z.shape
    device = z.device
    dtype = z.dtype

    # Stage discount factors beta^s for s = 0, ..., H-1: shape (1, H, 1)
    stage_idx = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    discounts = (discount**stage_idx).clamp(min=1e-6)

    # Prepend w_current to form previous stage weights z_{s-1}: shape (B, H, N)
    z_prev = torch.cat([w_current.unsqueeze(1), z[:, :-1, :]], dim=1)
    diffs_prev = z - z_prev  # z_s - z_{s-1}

    # Smoothed L1 turnover gradient wrt z_s from current stage transition (s-1 -> s)
    grad_tc_curr = kappa * diffs_prev / torch.sqrt(diffs_prev.pow(2) + smooth_eps**2)

    # Smoothed L1 turnover gradient wrt z_s from next stage transition (s -> s+1) for s < H-1
    if H > 1:
        diffs_next = z[:, 1:, :] - z[:, :-1, :]  # z_{s+1} - z_s
        grad_tc_next_inner = (
            -discount
            * kappa
            * diffs_next
            / torch.sqrt(diffs_next.pow(2) + smooth_eps**2)
        )
        zeros_last = torch.zeros((B, 1, N), device=device, dtype=dtype)
        grad_tc_next = torch.cat([grad_tc_next_inner, zeros_last], dim=1)
    else:
        grad_tc_next = torch.zeros_like(z)

    # Risk gradient: gamma * Sigma_s z_s -> shape (B, H, N)
    if cov_factors is not None:
        b_load, sigma_f, d_diag = cov_factors
        grad_risk = gamma * _factor_cov_matvec(z, b_load, sigma_f, d_diag)
    else:
        assert cov_seq is not None
        grad_risk = gamma * torch.matmul(cov_seq, z.unsqueeze(-1)).squeeze(-1)

    # Total stage gradient of convex minimization objective J(z; theta):
    # min_z sum_{s=0}^{H-1} beta^s [ -mu_s^T z_s + (gamma/2) z_s^T Sigma_s z_s + kappa ||z_s - z_{s-1}||_1 ]
    g = discounts * (-mu_seq + grad_risk + grad_tc_curr + grad_tc_next)

    # Entropic mirror descent step (exponentiated gradient on probability simplex)
    log_z_unnorm = torch.log(z.clamp(min=1e-12)) - eta * g
    z_next = torch.softmax(log_z_unnorm, dim=-1)
    return z_next


class _ImplicitMDFPFunction(torch.autograd.Function):
    """
    Custom autograd function implementing Implicit Fixed-Point Differentiation
    at the converged trajectory z* = F(z*, mu_seq, cov_seq).

    Instead of unrolling M steps in backprop (O(M) memory), uses the Implicit Function Theorem:
        (I - J_F(z*))^{-T} v = sum_{k=0}^{K} (J_F(z*)^T)^k v
    via vector-Jacobian products (VJP) at the fixed point z*.
    """

    @staticmethod
    def forward(
        ctx: Any,
        mu_seq: torch.Tensor,
        cov_seq: torch.Tensor,
        w_current: torch.Tensor,
        gamma: float,
        kappa: float,
        eta: float,
        max_iter: int,
        tol: float,
        discount: float,
        smooth_eps: float,
        neumann_steps: int,
        neumann_damping: float,
    ) -> torch.Tensor:
        B, H, N = mu_seq.shape

        # Warm-start trajectory from w_current tiled across H stages (projected to simplex)
        w_init = w_current.clamp(min=1e-6)
        w_init = w_init / w_init.sum(dim=-1, keepdim=True)
        z = w_init.unsqueeze(1).expand(B, H, N).clone()

        with torch.no_grad():
            for _ in range(max_iter):
                z_next = _mdfp_step(
                    z=z,
                    mu_seq=mu_seq,
                    cov_seq=cov_seq,
                    w_current=w_current,
                    gamma=gamma,
                    kappa=kappa,
                    eta=eta,
                    discount=discount,
                    smooth_eps=smooth_eps,
                )
                max_diff = float(torch.max(torch.abs(z_next - z)).item())
                z = z_next
                if max_diff < tol:
                    break

        ctx.save_for_backward(z, mu_seq, cov_seq, w_current)
        ctx.gamma = gamma
        ctx.kappa = kappa
        ctx.eta = eta
        ctx.discount = discount
        ctx.smooth_eps = smooth_eps
        ctx.neumann_steps = neumann_steps
        ctx.neumann_damping = neumann_damping
        return z

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        z_star, mu_seq, cov_seq, w_current = ctx.saved_tensors
        gamma = ctx.gamma
        kappa = ctx.kappa
        eta = ctx.eta
        discount = ctx.discount
        smooth_eps = ctx.smooth_eps
        neumann_steps = ctx.neumann_steps
        damping = ctx.neumann_damping

        # Solve (I - damping * J_F(z*)^T) u = grad_output via truncated Neumann series
        u = grad_output.clone()
        v = grad_output.clone()

        with torch.enable_grad():
            z_var = z_star.detach().requires_grad_(True)
            f_z = _mdfp_step(
                z=z_var,
                mu_seq=mu_seq.detach(),
                cov_seq=cov_seq.detach(),
                w_current=w_current.detach(),
                gamma=gamma,
                kappa=kappa,
                eta=eta,
                discount=discount,
                smooth_eps=smooth_eps,
            )
            for _ in range(neumann_steps):
                vjp_z = torch.autograd.grad(
                    outputs=f_z,
                    inputs=z_var,
                    grad_outputs=v,
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                v = damping * vjp_z
                u = u + v
                if float(torch.max(torch.abs(v)).item()) < 1e-7:
                    break

            # Propagate resolved adjoint u into parameters (mu_seq, cov_seq, w_current)
            mu_var = mu_seq.detach().requires_grad_(mu_seq.requires_grad)
            cov_var = cov_seq.detach().requires_grad_(cov_seq.requires_grad)
            w_var = w_current.detach().requires_grad_(w_current.requires_grad)

            f_theta = _mdfp_step(
                z=z_star.detach(),
                mu_seq=mu_var,
                cov_seq=cov_var,
                w_current=w_var,
                gamma=gamma,
                kappa=kappa,
                eta=eta,
                discount=discount,
                smooth_eps=smooth_eps,
            )

            inputs_to_diff = []
            if mu_seq.requires_grad:
                inputs_to_diff.append(mu_var)
            if cov_seq.requires_grad:
                inputs_to_diff.append(cov_var)
            if w_current.requires_grad:
                inputs_to_diff.append(w_var)

            if inputs_to_diff:
                grads = torch.autograd.grad(
                    outputs=f_theta,
                    inputs=inputs_to_diff,
                    grad_outputs=u,
                    allow_unused=True,
                )
            else:
                grads = ()

        grad_idx = 0
        grad_mu = None
        grad_cov = None
        grad_w = None
        if mu_seq.requires_grad:
            grad_mu = grads[grad_idx]
            grad_idx += 1
        if cov_seq.requires_grad:
            grad_cov = grads[grad_idx]
            grad_idx += 1
        if w_current.requires_grad:
            grad_w = grads[grad_idx]

        return (
            grad_mu,
            grad_cov,
            grad_w,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MDFPOptimizationLayer(nn.Module):
    """
    Differentiable Multi-Period Optimization Layer via Entropic Mirror-Descent Fixed-Point (MDFP).

    Solves the coupled H-step simplex-constrained MPO problem:
        min_{z_0, ..., z_{H-1} in Delta^N} sum_{s=0}^{H-1} beta^s [
            -mu_s^T z_s + (gamma / 2) z_s^T Sigma_s z_s + kappa ||z_s - z_{s-1}||_1
        ]
    using entropic mirror descent (exponentiated gradient):
        z_{s,i}^{m+1} = (z_{s,i}^m * exp(-eta * g_{s,i}^m)) / sum_j (z_{s,j}^m * exp(-eta * g_{s,j}^m))
    and differentiates through the fixed point z* in O(1) memory via the Implicit Function Theorem.
    """

    def __init__(
        self,
        horizon: int = 5,
        risk_aversion: float = 2.0,
        cost_penalty: float = 0.001,
        step_size: float = 0.5,
        max_iter: int = 50,
        tol: float = 1e-5,
        discount: float = 0.98,
        smooth_eps: float = 1e-3,
        neumann_steps: int = 15,
        neumann_damping: float = 0.9,
        implicit_backward: bool = True,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.risk_aversion = risk_aversion
        self.cost_penalty = cost_penalty
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.discount = discount
        self.smooth_eps = smooth_eps
        self.neumann_steps = neumann_steps
        self.neumann_damping = neumann_damping
        self.implicit_backward = implicit_backward

    def forward(
        self,
        mu_seq: torch.Tensor,
        cov_seq: torch.Tensor | None = None,
        w_current: torch.Tensor | None = None,
        cov_factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mu_seq : torch.Tensor
            Predicted multi-horizon expected returns of shape (B, H, N) or (B, N).
        cov_seq : torch.Tensor, optional
            Covariance trajectory of shape (B, H, N, N) or (B, N, N). Defaults to identity * 0.01.
        w_current : torch.Tensor, optional
            Pre-trade starting weights of shape (B, N). Defaults to equal weights 1/N.
        cov_factors : tuple[torch.Tensor, torch.Tensor, torch.Tensor], optional
            Low-rank factor tuple ``(B_load, Sigma_F, d_diag)`` from ``ISABFactorDiagonalHead``
            enabling ``O(N m)`` mirror-descent risk matrix-vector products without materializing
            an ``N x N`` covariance tensor.

        Returns
        -------
        torch.Tensor
            Optimal multi-period weight trajectory z* of shape (B, H, N).
        """
        if mu_seq.dim() == 2:
            mu_seq = mu_seq.unsqueeze(1).expand(-1, self.horizon, -1)

        B, H, N = mu_seq.shape
        device = mu_seq.device
        dtype = mu_seq.dtype

        if w_current is None:
            w_current = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)

        if cov_factors is not None:
            # Fast O(Nm) factor matvec path matching exact max_iter fixed-point steps
            w_init = w_current.clamp(min=1e-6)
            w_init = w_init / w_init.sum(dim=-1, keepdim=True)
            z = w_init.unsqueeze(1).expand(B, H, N).clone()

            if self.max_iter > 1:
                with torch.no_grad():
                    for _ in range(self.max_iter - 1):
                        z_next = _mdfp_step(
                            z=z,
                            mu_seq=mu_seq,
                            cov_seq=None,
                            w_current=w_current,
                            gamma=self.risk_aversion,
                            kappa=self.cost_penalty,
                            eta=self.step_size,
                            discount=self.discount,
                            smooth_eps=self.smooth_eps,
                            cov_factors=cov_factors,
                        )
                        max_diff = float(torch.max(torch.abs(z_next - z)).item())
                        z = z_next
                        if max_diff < self.tol:
                            return z

            # Final differentiable step (step max_iter) for gradient propagation into mu_seq and cov_factors
            return _mdfp_step(
                z=z.detach(),
                mu_seq=mu_seq,
                cov_seq=None,
                w_current=w_current,
                gamma=self.risk_aversion,
                kappa=self.cost_penalty,
                eta=self.step_size,
                discount=self.discount,
                smooth_eps=self.smooth_eps,
                cov_factors=cov_factors,
            )

        if cov_seq is None:
            eye = torch.eye(N, device=device, dtype=dtype) * 0.01
            cov_seq = eye.view(1, 1, N, N).expand(B, H, N, N)
        elif cov_seq.dim() == 3:
            cov_seq = cov_seq.unsqueeze(1).expand(B, H, N, N)

        if self.implicit_backward:
            return _ImplicitMDFPFunction.apply(
                mu_seq,
                cov_seq,
                w_current,
                self.risk_aversion,
                self.cost_penalty,
                self.step_size,
                self.max_iter,
                self.tol,
                self.discount,
                self.smooth_eps,
                self.neumann_steps,
                self.neumann_damping,
            )

        # Unrolled fallback if implicit_backward=False
        w_init = w_current.clamp(min=1e-6)
        w_init = w_init / w_init.sum(dim=-1, keepdim=True)
        z = w_init.unsqueeze(1).expand(B, H, N).clone()
        for _ in range(self.max_iter):
            z = _mdfp_step(
                z=z,
                mu_seq=mu_seq,
                cov_seq=cov_seq,
                w_current=w_current,
                gamma=self.risk_aversion,
                kappa=self.cost_penalty,
                eta=self.step_size,
                discount=self.discount,
                smooth_eps=self.smooth_eps,
            )
        return z
