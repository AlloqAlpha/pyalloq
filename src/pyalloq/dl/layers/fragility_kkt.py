from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn


class BasePolicyNetwork(nn.Module, ABC):
    """
    Unified Abstract Base Class for End-to-End Portfolio Policy Networks (Pillar 4).

    Standardizes the controller contract:
        forward(x, w_current=None) -> w_target in Delta^N
    bridging supervised Decision-Focused Learning (DFL), pathwise Deep Hedging
    (`DeepHedgingTrajectoryTrainer`), `DeepAllocator` backtests, and continuous-control
    RL environments (`POMDPPortfolioEnv`, `PortfolioGymEnv`, `ScenarioPortfolioEnv`).
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Market state tensor of shape `(B, L, N, F)` or `(B, L, N)`.
        w_current : torch.Tensor, optional
            Pre-trade drifted portfolio weights `w_{t^-}` of shape `(B, N)` or `(N,)`.
            If `None`, defaults to uniform weights `1 / N`.

        Returns
        -------
        torch.Tensor
            Target portfolio weights `w_target` on the probability simplex `Delta^N`
            of shape `(B, N)`.
        """
        ...

    @staticmethod
    def _normalize_simplex(w: torch.Tensor) -> torch.Tensor:
        w_pos = torch.clamp(w, min=0.0)
        denom = w_pos.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return w_pos / denom

    @torch.no_grad()
    def predict_from_obs(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Evaluates the policy directly on a Gymnasium observation from `POMDPPortfolioEnv`,
        `PortfolioGymEnv`, or `ScenarioPortfolioEnv`.
        """
        self.eval()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if isinstance(observation, dict):
            raw_mkt = observation.get(
                "market_state",
                observation.get("market", observation.get("returns")),
            )
            mkt = np.asarray(raw_mkt)
            w_np = observation.get("current_weights", observation.get("weights"))
        else:
            mkt = np.asarray(observation)
            w_np = None

        if mkt.ndim == 2:
            x_t = (
                torch.tensor(mkt, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
            )
        elif mkt.ndim == 3:
            x_t = torch.tensor(mkt, device=device, dtype=dtype).unsqueeze(0)
        elif mkt.ndim == 4:
            x_t = torch.tensor(mkt, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported market observation shape: {mkt.shape}")

        w_t: torch.Tensor | None = None
        if w_np is not None:
            w_arr = np.asarray(w_np)
            if w_arr.ndim == 1:
                w_t = torch.tensor(w_arr, device=device, dtype=dtype).unsqueeze(0)
            else:
                w_t = torch.tensor(w_arr, device=device, dtype=dtype)

        w_out = self.forward(x_t, w_current=w_t)
        action = w_out.squeeze(0).cpu().numpy()
        return action, {"deterministic": deterministic}


class FragilityShrinkageCholeskyLayer(nn.Module):
    """
    Differentiable Cholesky Covariance Layer with Dynamic Network-Fragility Shrinkage.

    1. Projects node embeddings `H_node` of shape `(B, N, d)` into a lower-triangular
       matrix `L` with strictly positive `softplus` diagonal entries:
           Sigma_raw = L @ L^T + eps * I
    2. Dynamically shrinks `Sigma_raw` toward a structured Buss-Vilkov / equicorrelation
       target `Sigma_target` with shrinkage intensity `alpha(lambda_2, rho_G) in (0, 1)`
       governed by the Graph Laplacian's Fiedler connectivity `lambda_2` and the
       Hawkes branching spectral radius `rho_G`:
           Sigma_hat = (1 - alpha) * Sigma_raw + alpha * Sigma_target
    """

    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-4,
        default_equicorr: float = 0.35,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.default_equicorr = float(default_equicorr)

        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)

        # Learnable fragility shrinkage parameters: alpha = sigmoid(w_0 + w_lam * lambda_2 + w_rho * criticality)
        self.shrink_bias = nn.Parameter(torch.tensor(-1.2))
        self.shrink_fiedler_weight = nn.Parameter(torch.tensor(0.8))
        self.shrink_hawkes_weight = nn.Parameter(torch.tensor(1.0))

    def compute_shrinkage_alpha(
        self,
        fiedler_value: torch.Tensor | float = 0.2,
        hawkes_radius: torch.Tensor | float = 0.2,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        lam_t = torch.as_tensor(fiedler_value, device=device, dtype=dtype)
        rho_t = torch.as_tensor(hawkes_radius, device=device, dtype=dtype).clamp(
            0.0, 0.98
        )
        criticality = rho_t / (1.0 - rho_t + 0.02)
        logit = (
            self.shrink_bias
            + self.shrink_fiedler_weight * lam_t
            + self.shrink_hawkes_weight * criticality
        )
        return torch.sigmoid(logit).clamp(0.02, 0.95)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        buss_vilkov_target: torch.Tensor | None = None,
        fiedler_value: torch.Tensor | float = 0.2,
        hawkes_radius: torch.Tensor | float = 0.2,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_embeddings : torch.Tensor
            Spatially attended asset embeddings of shape `(B, N, d)`.
        buss_vilkov_target : torch.Tensor, optional
            Optional target covariance matrix of shape `(B, N, N)` or `(N, N)`.
            When None, constructs a variance-matched Buss-Vilkov equicorrelation target.
        """
        B, N, _ = node_embeddings.shape
        device = node_embeddings.device
        dtype = node_embeddings.dtype

        q = self.proj_q(node_embeddings)
        k = self.proj_k(node_embeddings)
        raw_mat = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim**0.5)

        L_strict = torch.tril(raw_mat, diagonal=-1)
        d_raw = torch.diagonal(raw_mat, dim1=-2, dim2=-1)
        d_pos = torch.nn.functional.softplus(d_raw) + self.eps
        L = L_strict + torch.diag_embed(d_pos)

        eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
        sigma_raw = torch.bmm(L, L.transpose(1, 2)) + self.eps * eye

        if buss_vilkov_target is not None:
            if buss_vilkov_target.dim() == 2:
                sigma_target = buss_vilkov_target.unsqueeze(0).expand(B, N, N)
            else:
                sigma_target = buss_vilkov_target
        else:
            # Construct PSD equicorrelation Buss-Vilkov prior from predicted marginal volatilities
            diag_var = torch.diagonal(sigma_raw, dim1=-2, dim2=-1).clamp(min=self.eps)
            std_vec = torch.sqrt(diag_var)
            outer_std = std_vec.unsqueeze(-1) * std_vec.unsqueeze(-2)
            rho = self.default_equicorr
            equi_corr = (1.0 - rho) * eye + rho * torch.ones(
                (1, N, N), device=device, dtype=dtype
            )
            sigma_target = outer_std * equi_corr

        alpha = self.compute_shrinkage_alpha(
            fiedler_value=fiedler_value,
            hawkes_radius=hawkes_radius,
            device=device,
            dtype=dtype,
        )
        while alpha.dim() < 3:
            alpha = alpha.unsqueeze(-1)

        sigma_shrunk = (1.0 - alpha) * sigma_raw + alpha * sigma_target
        return 0.5 * (sigma_shrunk + sigma_shrunk.transpose(1, 2))


class _ExactRiskParityKKTFunction(torch.autograd.Function):
    """
    Custom autograd function solving Spinu's (2013) convex Risk Budgeting / Risk Parity
    optimization problem and differentiating through its exact KKT conditions:

        min_{y > 0} 1/2 y^T Sigma y - sum_{i=1}^N b_i log(y_i)
        KKT Stationarity: F(y*, Sigma, b) = Sigma y* - b / y* = 0
        Normalized Portfolio Weights: w* = y* / (1^T y*)

    Exact KKT Implicit Jacobian:
        J_y = dF / dy* = Sigma + diag(b / (y*)^2)  (strictly SPD)
        dL / dSigma = -1/2 (u (y*)^T + y* u^T)     where J_y u = dL / dy*
        dL / db     = u / y*
    """

    @staticmethod
    def forward(
        ctx: Any,
        cov_matrix: torch.Tensor,
        risk_budgets: torch.Tensor,
        max_iter: int,
        tol: float,
    ) -> torch.Tensor:
        B, N, _ = cov_matrix.shape
        device = cov_matrix.device
        dtype = cov_matrix.dtype

        # Symmetrize & regularize Sigma in float64 for high-precision Newton-KKT solve
        eye = torch.eye(N, device=device, dtype=torch.float64).unsqueeze(0)
        sigma = (
            0.5
            * (
                cov_matrix.detach().to(torch.float64)
                + cov_matrix.detach().to(torch.float64).transpose(1, 2)
            )
            + 1e-6 * eye
        )
        b = risk_budgets.detach().to(torch.float64).clamp(min=1e-6)
        b = b / b.sum(dim=-1, keepdim=True)

        # Initialize y from inverse-volatility scaling
        diag_s = torch.diagonal(sigma, dim1=-2, dim2=-1).clamp(min=1e-6)
        y = torch.sqrt(b / diag_s)

        with torch.no_grad():
            for _ in range(max_iter):
                sigma_y = torch.bmm(sigma, y.unsqueeze(-1)).squeeze(-1)
                grad = sigma_y - (b / y)
                max_err = float(torch.max(torch.abs(grad)).item())
                if max_err < tol:
                    break

                hess = sigma + torch.diag_embed(b / (y.pow(2).clamp(min=1e-12)))
                delta = torch.linalg.solve(hess, grad.unsqueeze(-1)).squeeze(-1)

                # Backtracking line search to maintain strict positivity y > 0
                step = torch.ones((B, 1), device=device, dtype=torch.float64)
                for _ in range(8):
                    y_cand = y - step * delta
                    if bool(torch.all(y_cand > 1e-8).item()):
                        break
                    step = step * 0.5
                y = (y - step * delta).clamp(min=1e-8)

        y_sum = y.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        w = (y / y_sum).to(dtype)

        ctx.save_for_backward(
            sigma.to(dtype),
            b.to(dtype),
            y.to(dtype),
            w,
        )
        return w

    @staticmethod
    def backward(ctx: Any, grad_w: torch.Tensor) -> tuple[Any, ...]:
        sigma, b, y, w = ctx.saved_tensors
        # Quotient rule projection: dy_L = (grad_w - <grad_w, w> * 1) / (1^T y)
        y_sum = y.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        dot_gw = (grad_w * w).sum(dim=-1, keepdim=True)
        grad_y = (grad_w - dot_gw) / y_sum

        # Solve exact KKT linear system: (Sigma + diag(b / y^2)) u = grad_y
        hess = sigma + torch.diag_embed(b / (y.pow(2).clamp(min=1e-10)))
        u = torch.linalg.solve(hess, grad_y.unsqueeze(-1)).squeeze(-1)

        grad_cov = None
        grad_b = None
        if ctx.needs_input_grad[0]:
            outer_uy = torch.bmm(u.unsqueeze(-1), y.unsqueeze(-2))
            grad_cov = -0.5 * (outer_uy + outer_uy.transpose(1, 2))
        if ctx.needs_input_grad[1]:
            grad_b = u / y.clamp(min=1e-8)

        return grad_cov, grad_b, None, None


class DifferentiableRiskParityLayer(nn.Module):
    """
    End-to-End Differentiable Risk Parity / Risk Budgeting Layer via Exact KKT Implicit Differentiation.

    Interchangeable with classical `RiskParityAllocator` and `RiskBudgetingAllocator`,
    allowing `CRISPSparseSTGAT` or any covariance network to backpropagate downstream
    Sharpe/CVaR losses directly through the exact KKT stationarity conditions.
    """

    def __init__(
        self,
        max_iter: int = 25,
        tol: float = 1e-9,
    ) -> None:
        super().__init__()
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def forward(
        self,
        cov_matrix: torch.Tensor,
        risk_budgets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cov_matrix : torch.Tensor
            Positive-definite covariance tensor of shape `(B, N, N)` or `(N, N)`.
        risk_budgets : torch.Tensor, optional
            Target risk budget vector of shape `(B, N)` or `(N,)`. Defaults to equal risk `1/N`.

        Returns
        -------
        torch.Tensor
            Optimal risk-parity portfolio weights `w*` on the simplex of shape `(B, N)` or `(N,)`.
        """
        squeezed = cov_matrix.dim() == 2
        if squeezed:
            cov_matrix = cov_matrix.unsqueeze(0)

        B, N, _ = cov_matrix.shape
        if risk_budgets is None:
            b = torch.full(
                (B, N), 1.0 / N, device=cov_matrix.device, dtype=cov_matrix.dtype
            )
        elif risk_budgets.dim() == 1:
            b = risk_budgets.unsqueeze(0).expand(B, N)
        else:
            b = risk_budgets

        w = _ExactRiskParityKKTFunction.apply(cov_matrix, b, self.max_iter, self.tol)
        return w.squeeze(0) if squeezed else w


def _cvar_kkt_mirror_step(
    w: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    scenarios: torch.Tensor | None,
    risk_budgets: torch.Tensor,
    gamma: float,
    cvar_weight: float,
    beta: float,
    rb_barrier: float,
    step_size: float,
) -> torch.Tensor:
    """
    Single entropic mirror-descent step on the CVaR-tilted, risk-budgeted objective on Delta^N.
    """
    B, N = w.shape
    # Mean-variance gradient: -mu + gamma * Sigma w
    grad_mv = -mu + gamma * torch.bmm(cov, w.unsqueeze(-1)).squeeze(-1)

    # Logarithmic risk-budgeting barrier gradient: -rb_barrier * b / w
    grad_rb = -rb_barrier * (risk_budgets / w.clamp(min=1e-4))

    # Rockafellar-Uryasev smoothed CVaR gradient (or Gaussian/Cornish-Fisher tail gradient when scenarios=None)
    if scenarios is not None:
        # scenarios shape: (B, M, N) -> portfolio losses L = -scenarios @ w: shape (B, M)
        losses = -torch.bmm(scenarios, w.unsqueeze(-1)).squeeze(-1)
        q_alpha = torch.quantile(losses.detach(), q=beta, dim=-1, keepdim=True)
        tail_probs = torch.sigmoid(25.0 * (losses - q_alpha))
        tail_denom = ((1.0 - beta) * scenarios.shape[1]) + 1e-6
        grad_cvar = (
            -cvar_weight
            * torch.bmm(tail_probs.unsqueeze(1), scenarios).squeeze(1)
            / tail_denom
        )
    else:
        # Parametric CVaR normal tail factor phi(z_beta) / (1 - beta) * (Sigma w) / sqrt(w^T Sigma w)
        sigma_w = torch.bmm(cov, w.unsqueeze(-1)).squeeze(-1)
        port_var = (w * sigma_w).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        port_vol = torch.sqrt(port_var)
        # Standard normal 95% CVaR multiplier ~ 2.0627
        tail_mult = 2.0627
        grad_cvar = cvar_weight * tail_mult * (sigma_w / port_vol)

    total_grad = grad_mv + grad_cvar + grad_rb
    # Center gradient for numerical stability on simplex
    total_grad = total_grad - total_grad.mean(dim=-1, keepdim=True)
    log_w = torch.log(w.clamp(min=1e-8)) - step_size * total_grad
    return torch.softmax(log_w, dim=-1)


class _ImplicitCVaRKKTFunction(torch.autograd.Function):
    """
    Implicit fixed-point differentiation for the CVaR-tilted + Risk-Budgeted KKT allocation layer.
    """

    @staticmethod
    def forward(
        ctx: Any,
        mu: torch.Tensor,
        cov: torch.Tensor,
        scenarios: torch.Tensor | None,
        risk_budgets: torch.Tensor,
        gamma: float,
        cvar_weight: float,
        beta: float,
        rb_barrier: float,
        step_size: float,
        max_iter: int,
        tol: float,
        neumann_steps: int,
    ) -> torch.Tensor:
        B, N = mu.shape
        w = torch.full((B, N), 1.0 / N, device=mu.device, dtype=mu.dtype)

        with torch.no_grad():
            for _ in range(max_iter):
                w_next = _cvar_kkt_mirror_step(
                    w=w,
                    mu=mu,
                    cov=cov,
                    scenarios=scenarios,
                    risk_budgets=risk_budgets,
                    gamma=gamma,
                    cvar_weight=cvar_weight,
                    beta=beta,
                    rb_barrier=rb_barrier,
                    step_size=step_size,
                )
                diff = float(torch.max(torch.abs(w_next - w)).item())
                w = w_next
                if diff < tol:
                    break

        ctx.save_for_backward(w, mu, cov, scenarios, risk_budgets)
        ctx.gamma = gamma
        ctx.cvar_weight = cvar_weight
        ctx.beta = beta
        ctx.rb_barrier = rb_barrier
        ctx.step_size = step_size
        ctx.neumann_steps = neumann_steps
        return w

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        w_star, mu, cov, scenarios, risk_budgets = ctx.saved_tensors
        u = grad_output.clone()
        v = grad_output.clone()

        with torch.enable_grad():
            w_var = w_star.detach().requires_grad_(True)
            f_w = _cvar_kkt_mirror_step(
                w=w_var,
                mu=mu.detach(),
                cov=cov.detach(),
                scenarios=scenarios.detach() if scenarios is not None else None,
                risk_budgets=risk_budgets.detach(),
                gamma=ctx.gamma,
                cvar_weight=ctx.cvar_weight,
                beta=ctx.beta,
                rb_barrier=ctx.rb_barrier,
                step_size=ctx.step_size,
            )
            for _ in range(ctx.neumann_steps):
                vjp_w = torch.autograd.grad(
                    outputs=f_w,
                    inputs=w_var,
                    grad_outputs=v,
                    retain_graph=True,
                )[0]
                v = 0.9 * vjp_w
                u = u + v
                if float(torch.max(torch.abs(v)).item()) < 1e-7:
                    break

            mu_var = mu.detach().requires_grad_(mu.requires_grad)
            cov_var = cov.detach().requires_grad_(cov.requires_grad)
            scen_var = (
                scenarios.detach().requires_grad_(scenarios.requires_grad)
                if scenarios is not None
                else None
            )
            rb_var = risk_budgets.detach().requires_grad_(risk_budgets.requires_grad)

            f_theta = _cvar_kkt_mirror_step(
                w=w_star.detach(),
                mu=mu_var,
                cov=cov_var,
                scenarios=scen_var,
                risk_budgets=rb_var,
                gamma=ctx.gamma,
                cvar_weight=ctx.cvar_weight,
                beta=ctx.beta,
                rb_barrier=ctx.rb_barrier,
                step_size=ctx.step_size,
            )

            diff_inputs = []
            if mu.requires_grad:
                diff_inputs.append(mu_var)
            if cov.requires_grad:
                diff_inputs.append(cov_var)
            if scenarios is not None and scenarios.requires_grad:
                diff_inputs.append(scen_var)
            if risk_budgets.requires_grad:
                diff_inputs.append(rb_var)

            grads = (
                torch.autograd.grad(
                    outputs=f_theta,
                    inputs=diff_inputs,
                    grad_outputs=u,
                    allow_unused=True,
                )
                if diff_inputs
                else ()
            )

        idx = 0
        g_mu = g_cov = g_scen = g_rb = None
        if mu.requires_grad:
            g_mu = grads[idx]
            idx += 1
        if cov.requires_grad:
            g_cov = grads[idx]
            idx += 1
        if scenarios is not None and scenarios.requires_grad:
            g_scen = grads[idx]
            idx += 1
        if risk_budgets.requires_grad:
            g_rb = grads[idx]

        return (
            g_mu,
            g_cov,
            g_scen,
            g_rb,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DifferentiableCVaRLayer(nn.Module):
    """
    End-to-End Differentiable CVaR + Risk-Budgeted KKT Allocation Layer.

    Solves the Rockafellar-Uryasev CVaR-tilted & logarithmically risk-budgeted
    portfolio optimization problem on the probability simplex with O(1)-memory
    Implicit Function Theorem backpropagation.
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,
        cvar_weight: float = 1.0,
        beta: float = 0.95,
        risk_budget_barrier: float = 0.05,
        step_size: float = 0.25,
        max_iter: int = 40,
        tol: float = 1e-5,
        neumann_steps: int = 12,
    ) -> None:
        super().__init__()
        self.risk_aversion = float(risk_aversion)
        self.cvar_weight = float(cvar_weight)
        self.beta = float(beta)
        self.risk_budget_barrier = float(risk_budget_barrier)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.neumann_steps = int(neumann_steps)

    def forward(
        self,
        expected_returns: torch.Tensor,
        cov_matrix: torch.Tensor,
        scenarios: torch.Tensor | None = None,
        risk_budgets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeezed = expected_returns.dim() == 1
        if squeezed:
            expected_returns = expected_returns.unsqueeze(0)
        if cov_matrix.dim() == 2:
            cov_matrix = cov_matrix.unsqueeze(0)
        if scenarios is not None and scenarios.dim() == 2:
            scenarios = scenarios.unsqueeze(0)

        B, N = expected_returns.shape
        if risk_budgets is None:
            rb = torch.full(
                (B, N),
                1.0 / N,
                device=expected_returns.device,
                dtype=expected_returns.dtype,
            )
        elif risk_budgets.dim() == 1:
            rb = risk_budgets.unsqueeze(0).expand(B, N)
        else:
            rb = risk_budgets

        w = _ImplicitCVaRKKTFunction.apply(
            expected_returns,
            cov_matrix,
            scenarios,
            rb,
            self.risk_aversion,
            self.cvar_weight,
            self.beta,
            self.risk_budget_barrier,
            self.step_size,
            self.max_iter,
            self.tol,
            self.neumann_steps,
        )
        return w.squeeze(0) if squeezed else w


class DifferentiableKKTAllocatorModule(BasePolicyNetwork):
    """
    End-to-End Differentiable Portfolio Allocation Module chaining a Spatio-Temporal
    Graph Network (`CRISPSparseSTGAT`) directly into a Differentiable Classical KKT
    Layer (`DifferentiableRiskParityLayer` or `DifferentiableCVaRLayer`).

    Maps input tensors `x` of shape `(B, L, N, F)` directly to optimal simplex weights
    `w*` of shape `(B, N)`, making classical Risk Parity and CVaR optimization drop-in
    compatible with `WalkForwardTrainer`, `SharpeLoss`, and `DeepAllocator`.
    """

    backbone: nn.Module
    kkt_layer: nn.Module
    risk_budgets: torch.Tensor | None

    def __init__(
        self,
        backbone: nn.Module,
        kkt_layer: nn.Module | None = None,
        mode: str = "risk_parity",
        risk_budgets: torch.Tensor | None = None,
        no_trade_buffer: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.mode = mode.lower()
        self.no_trade_buffer = float(np.clip(no_trade_buffer, 0.0, 0.95))
        if kkt_layer is not None:
            self.kkt_layer = kkt_layer
        elif self.mode == "cvar":
            self.kkt_layer = DifferentiableCVaRLayer()
        else:
            self.kkt_layer = DifferentiableRiskParityLayer()
        if risk_budgets is not None:
            self.register_buffer("risk_budgets", risk_budgets)
        else:
            self.risk_budgets = None

    def forward(
        self,
        x: torch.Tensor,
        w_current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        pred_fn = getattr(self.backbone, "predict_returns_and_cov", None)
        if callable(pred_fn):
            mu, cov, _ = pred_fn(x)
        else:
            cov = self.backbone(x)
            mu = x[:, -1, :, 0]

        if isinstance(self.kkt_layer, DifferentiableCVaRLayer):
            # Use lookback return window x[:, :, :, 0] as empirical scenario distribution
            ret_scenarios = x[:, :, :, 0]
            w_opt = self.kkt_layer(
                expected_returns=mu,
                cov_matrix=cov,
                scenarios=ret_scenarios,
                risk_budgets=self.risk_budgets,
            )
        else:
            w_opt = self.kkt_layer(cov, risk_budgets=self.risk_budgets)

        if w_current is not None and self.no_trade_buffer > 0.0:
            w_c = (
                w_current.unsqueeze(0).expand_as(w_opt)
                if w_current.dim() == 1
                else w_current
            ).to(device=w_opt.device, dtype=w_opt.dtype)
            w_c = torch.clamp(w_c, min=0.0)
            w_c = w_c / w_c.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            w_opt = (1.0 - self.no_trade_buffer) * w_opt + self.no_trade_buffer * w_c
            w_opt = w_opt / w_opt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return w_opt
