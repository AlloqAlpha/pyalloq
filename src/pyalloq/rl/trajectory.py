import numpy as np
import torch
from pyalloq_core.data import MarketData, ScenarioMarketData
from torch import nn

from pyalloq.dl.policy import BasePolicyNetwork


class TrajectoryPolicyNet(BasePolicyNetwork):
    """
    Deep Hedging Recurrent Policy Network pi_theta(s_t, w_{t^-}) -> w_t in Delta^N
    (Buehler et al. 2019; Research Section 8.1).

    Processes the sequential market state together with the pre-trade drifted weights w_{t^-}
    so the policy explicitly learns the multi-period no-trade zone around current holdings.
    """

    def __init__(
        self,
        n_assets: int,
        lookback_window: int = 20,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.lookback_window = lookback_window
        input_dim = n_assets + lookback_window * n_assets
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_assets),
        )

    def forward(
        self,
        return_window: torch.Tensor,
        w_current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        return_window : torch.Tensor
            Recent asset return window of shape (B, L, N, F), (B, L, N), or (B, L * N).
        w_current : torch.Tensor, optional
            Pre-trade drifted weights of shape (B, N). Defaults to uniform 1 / N.

        Returns
        -------
        torch.Tensor
            Target portfolio weights w_t on the simplex of shape (B, N).
        """
        if return_window.dim() == 4:
            return_window = return_window[:, :, :, 0]
        B = return_window.shape[0]
        if w_current is None:
            w_current = torch.full(
                (B, self.n_assets),
                1.0 / self.n_assets,
                device=return_window.device,
                dtype=return_window.dtype,
            )
        elif w_current.dim() == 1:
            w_current = w_current.unsqueeze(0).expand(B, self.n_assets)

        flat_ret = return_window.reshape(B, -1)
        x = torch.cat([w_current, flat_ret], dim=-1)
        logits = self.net(x)
        # Residual connection in log-weight space biases the policy toward holding w_current
        log_prior = torch.log(w_current.clamp(min=1e-6))
        return torch.softmax(log_prior + logits, dim=-1)


class DeepHedgingTrajectoryTrainer:
    """
    Pathwise Trajectory Trainer for Deep Hedging & Multi-Period Allocation (Research Section 8.1).

    Instead of noisy TD(0) bellman bootstrapping, unrolls the differentiable wealth & weight-drift
    dynamics over H-step sample paths omega_j and directly minimizes the entropic risk measure
    (or CVaR / multi-period negative Sharpe) of terminal net wealth:
        L(theta) = (1 / gamma) log E_{omega}[ exp(-gamma * sum_{s=0}^{H-1} r_{p,s}^{net}(omega; theta)) ]
    """

    def __init__(
        self,
        policy: BasePolicyNetwork | nn.Module,
        horizon: int = 10,
        lookback_window: int | None = None,
        transaction_cost_bps: float = 10.0,
        risk_aversion: float = 2.0,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.policy = policy.to(self.device)
        self.horizon = horizon
        self.lookback_window = int(
            lookback_window
            if lookback_window is not None
            else getattr(self.policy, "lookback_window", 20)
        )
        self.tc_rate = transaction_cost_bps / 10000.0
        self.risk_aversion = risk_aversion
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def compute_trajectory_loss(
        self,
        scenarios: torch.Tensor,
        w_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Unroll policy along a batch of return trajectories of shape (B, L + H, N)
        and compute the pathwise entropic + Sharpe trajectory loss.
        """
        B, T_total, N = scenarios.shape
        L = self.lookback_window
        H = T_total - L
        if H <= 0:
            raise ValueError(
                f"Trajectory length {T_total} must exceed lookback_window {L}."
            )

        if w_init is None:
            w_curr = torch.full(
                (B, N), 1.0 / N, device=scenarios.device, dtype=scenarios.dtype
            )
        else:
            w_curr = w_init

        net_returns_list: list[torch.Tensor] = []

        for s in range(H):
            ret_win = scenarios[:, s : s + L, :]  # (B, L, N)
            r_next = scenarios[:, s + L, :]  # (B, N)

            w_target = self.policy(ret_win, w_curr)
            turnover = torch.abs(w_target - w_curr).sum(dim=-1)  # (B,)
            tc = self.tc_rate * turnover

            gross_ret = (w_target * r_next).sum(dim=-1)  # (B,)
            net_ret = (1.0 - tc) * (1.0 + gross_ret) - 1.0
            net_returns_list.append(net_ret)

            # Differentiable post-return weight drift w_{(s+1)^-}
            grown = w_target * (1.0 + r_next).clamp(min=1e-4)
            w_curr = grown / grown.sum(dim=-1, keepdim=True)

        path_returns = torch.stack(net_returns_list, dim=1)  # (B, H)
        cum_return = path_returns.sum(dim=1)  # (B,)

        # Entropic risk measure (certainty equivalent) + path variance penalty
        gamma = self.risk_aversion
        scaled = torch.clamp(-gamma * cum_return, min=-20.0, max=20.0)
        entropic_loss = (torch.logsumexp(scaled, dim=0) - float(np.log(B))) / max(
            gamma, 1e-6
        )
        var_penalty = 0.5 * gamma * path_returns.var(dim=1, unbiased=False).mean()
        return entropic_loss + var_penalty

    def fit_on_market_data(
        self,
        data: MarketData,
        epochs: int = 20,
        batch_size: int = 16,
    ) -> list[float]:
        """
        Slice rolling trajectories of length (L + H) from `MarketData` and train `self.policy`.
        """
        returns_np = data.prices.pct_change().fillna(0.0).to_numpy(dtype=np.float32)
        L = self.lookback_window
        seq_len = L + self.horizon
        n_obs = len(returns_np)
        if n_obs <= seq_len:
            raise ValueError(
                f"Need more than {seq_len} observations in MarketData, got {n_obs}."
            )

        windows = np.stack(
            [returns_np[i : i + seq_len] for i in range(n_obs - seq_len + 1)],
            axis=0,
        )
        dataset_tensor = torch.tensor(windows, dtype=torch.float32, device=self.device)

        self.policy.train()
        loss_history: list[float] = []
        n_samples = dataset_tensor.shape[0]

        for _ in range(epochs):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_losses: list[float] = []
            for start in range(0, n_samples, batch_size):
                batch_idx = perm[start : start + batch_size]
                batch_paths = dataset_tensor[batch_idx]

                self.optimizer.zero_grad()
                loss = self.compute_trajectory_loss(batch_paths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_losses.append(float(loss.item()))

            loss_history.append(float(np.mean(epoch_losses)))

        self.policy.eval()
        return loss_history

    def fit_on_scenarios(
        self,
        scenario_data: ScenarioMarketData,
        epochs: int = 20,
        batch_size: int = 16,
    ) -> list[float]:
        """
        Train `self.policy` directly on synthetic paths in `ScenarioMarketData` (M, T, N).
        """
        paths_tensor = torch.tensor(
            scenario_data.returns, dtype=torch.float32, device=self.device
        )
        self.policy.train()
        loss_history: list[float] = []
        n_samples = paths_tensor.shape[0]

        for _ in range(epochs):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_losses: list[float] = []
            for start in range(0, n_samples, batch_size):
                batch_paths = paths_tensor[perm[start : start + batch_size]]
                self.optimizer.zero_grad()
                loss = self.compute_trajectory_loss(batch_paths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_losses.append(float(loss.item()))

            loss_history.append(float(np.mean(epoch_losses)))

        self.policy.eval()
        return loss_history
