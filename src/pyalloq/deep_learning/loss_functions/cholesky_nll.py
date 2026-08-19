import torch
import torch.nn as nn


class CholeskyNLLLoss(nn.Module):
    """
    Multivariate Gaussian Negative Log-Likelihood loss.
    Computes loss directly from the Cholesky factor (L) to guarantee
    numerical stability and avoid expensive matrix inversions.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        L: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            L: Predicted lower triangular Cholesky factor of shape (B, N, N).
            y_true: Realized forward returns r_{t+1} of shape (B,N).

        Returns:
            Scalar loss value averaged over batch.
        """

        # 1. Log-Determinant: 2 * sum(log(diag(L)))
        # Extracts the diagonal of L for each item in the batch
        diag_L = L.diagonal(dim1=-2, dim2=-1)

        # log(|Sigma|) = 2 * log(|L|)
        log_det = 2.0 * torch.sum(torch.log(diag_L), dim=-1)

        # 2. Mahalanobis Distance: y^T (L L^T)^-1 y
        # We solve the linear system Lx = y for x, which is mathematically
        # equivalent to x = L^-1 y.
        # reshape y_true to (B, N, 1) for the triangular solver
        y_true_col = y_true.unsqueeze(-1)

        # Fast, stable solver utilizing the lower-triangular property of L
        x = torch.linalg.solve_triangular(L, y_true_col, upper=False)

        # The Mahalanobis distance simplifies to x^T x (sum of squares)
        mahalanobis = torch.sum(x**2, dim=[-2, -1])

        # 3. Total NLL
        nll = log_det + mahalanobis

        # Return the mean NLL across the batch
        return torch.mean(nll)
