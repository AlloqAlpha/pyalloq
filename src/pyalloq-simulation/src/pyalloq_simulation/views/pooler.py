import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class EntropyResult:
    """Holds posterior probability weights and optimization diagnostics."""

    weights: np.ndarray  # Shape: (M,) posterior probabilities x*
    prior: np.ndarray  # Shape: (M,) prior probabilities p
    kl_divergence: float  # Relative entropy achieved
    success: bool
    message: str


class EntropyPooler:
    """
    Meucci's Fully Flexible Views via Lagrange Dual Entropy Optimization.
    """

    def __init__(self, n_scenarios: int, prior: np.ndarray | None = None) -> None:
        self.M = n_scenarios
        # Uniform prior p_i = 1/M if none is provided
        self.p = np.full(self.M, 1.0 / self.M) if prior is None else prior

    def fit(
        self,
        A_ineq: np.ndarray | None = None,
        b_ineq: np.ndarray | None = None,
        C_eq: np.ndarray | None = None,
        d_eq: np.ndarray | None = None,
    ) -> EntropyResult:
        """
        Calculates posterior probability distribution over scenarios.

        Args:
            A_ineq: Shape (K_ineq, M) such that A_ineq @ x <= b_ineq
            b_ineq: Shape (K_ineq,)
            C_eq:   Shape (K_eq, M) such that C_eq @ x == d_eq
            d_eq:   Shape (K_eq,)
        """
        # Determine number of constraints
        k_ineq = len(b_ineq) if b_ineq is not None else 0
        k_eq = len(d_eq) if d_eq is not None else 0
        total_views = k_ineq + k_eq

        if total_views == 0:
            return EntropyResult(
                weights=self.p.copy(),
                prior=self.p.copy(),
                kl_divergence=0.0,
                success=True,
                message="No views supplied; returned prior.",
            )

        # Standardize constraints: stack into joint matrix G = [C; A]
        matrices: list[np.ndarray] = []
        bounds_targets: list[np.ndarray] = []

        if k_eq > 0 and C_eq is not None and d_eq is not None:
            matrices.append(C_eq)
            bounds_targets.append(d_eq)
        if k_ineq > 0 and A_ineq is not None and b_ineq is not None:
            matrices.append(A_ineq)
            bounds_targets.append(b_ineq)

        G = np.vstack(matrices)  # Shape: (V, M)
        targets: np.ndarray = np.concatenate(bounds_targets)  # Shape: (V,)

        # Dual bounds: lambda (equality) is unconstrained (-inf, inf),
        # nu (inequality) is constrained to [0, inf)
        bounds = [(None, None)] * k_eq + [(0.0, None)] * k_ineq
        init_dual = np.zeros(total_views)

        # Dual objective and analytical gradient
        def dual_objective(gamma):
            # gamma is [lambda, nu], shape: (V,)
            # exponent shape: (M,)
            exponent = -np.dot(gamma, G)

            # Log-sum-exp trick to prevent numerical overflow
            max_exp = np.max(exponent)
            weights_unnorm = self.p * np.exp(exponent - max_exp)
            partition_fn = np.sum(weights_unnorm)

            # Dual function value
            dual_val = np.log(partition_fn) + max_exp + np.dot(gamma, targets)

            # Analytical gradient w.r.t gamma: targets - (G @ posterior_probabilities)
            prob = weights_unnorm / partition_fn
            grad = targets - np.dot(G, prob)

            return dual_val, grad

        # Optimize dual problem using L-BFGS-B
        opt_res = minimize(
            fun=dual_objective,
            x0=init_dual,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 500},
        )

        # Recover posterior probabilities x* algebraically
        optimal_gamma = opt_res.x
        exponent = -np.dot(optimal_gamma, G)
        max_exp = np.max(exponent)
        unnorm_x = self.p * np.exp(exponent - max_exp)
        x_post = unnorm_x / np.sum(unnorm_x)

        kl = np.sum(np.where(x_post > 1e-16, x_post * np.log(x_post / self.p), 0.0))

        return EntropyResult(
            weights=x_post,
            prior=self.p,
            kl_divergence=kl,
            success=opt_res.success,
            message=opt_res.message,
        )
