from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from pyalloq.dl.estimators.e2e.models.set_policy import SetTransformerPolicyNet
from pyalloq.rl.agents.base import BaseRLAgent
from pyalloq.rl.wrappers.dict_features import DictFlattenWrapper

try:
    import stable_baselines3 as sb3

    _HAS_SB3 = True
except ImportError:
    sb3 = None
    _HAS_SB3 = False

AlgorithmName = Literal["PPO", "SAC", "DDPG", "TD3", "A2C"]
PolicyBackboneType = Literal["mlp", "set_transformer"]


def _masked_dirichlet_log_prob_and_entropy(
    actions: torch.Tensor,
    alpha: torch.Tensor,
    mask_full: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Evaluates the log-density and differential entropy of a Masked Dirichlet distribution
    over active simplex coordinates ``\{j : mask_full_j > 0\}``:
        \log p(w \mid \alpha, m) = \log \Gamma(\alpha_0) - \sum_{j \in \text{act}} \log \Gamma(\alpha_j)
                                   + \sum_{j \in \text{act}} (\alpha_j - 1) \log w_j
    """
    m_bool = mask_full > 0.5
    m_f = m_bool.to(dtype=alpha.dtype)
    alpha_act = torch.where(m_bool, alpha.clamp(min=1e-3), torch.ones_like(alpha))
    w_act = torch.where(
        m_bool, actions.clamp(min=1e-7, max=1.0), torch.ones_like(actions)
    )

    alpha_0 = (alpha_act * m_f).sum(dim=-1)
    log_norm = torch.lgamma(alpha_0) - (torch.lgamma(alpha_act) * m_f).sum(dim=-1)
    log_kernel = ((alpha_act - 1.0) * torch.log(w_act) * m_f).sum(dim=-1)
    log_prob = log_norm + log_kernel

    k_act = m_f.sum(dim=-1).clamp(min=1.0)
    digamma_0 = torch.digamma(alpha_0)
    entropy = (
        -log_norm
        + (alpha_0 - k_act) * digamma_0
        - ((alpha_act - 1.0) * torch.digamma(alpha_act) * m_f).sum(dim=-1)
    )
    return log_prob, entropy


class _TorchActorCriticFallback(nn.Module):
    """
    Lightweight PyTorch Actor-Critic PPO Policy used automatically when
    `stable-baselines3` is not installed in the current environment.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = torch.sigmoid(self.actor_mean(h))
        val = self.critic(h).squeeze(-1)
        return mean, val


class SB3PortfolioAgent(BaseRLAgent):
    """
    Stable-Baselines3 & Actor-Critic Orchestrator (`SB3PortfolioAgent`).

    Automatically wraps `POMDPPortfolioEnv` or `MetaPortfolioEnv` (`spaces.Dict` observations)
    with `DictFlattenWrapper` so MLP policies (`PPO`, `SAC`, `DDPG`, `TD3`, `A2C`) dynamically
    scale their input layer to match any `pyalloq-features` pipeline.

    When `stable-baselines3` is installed, delegates to the native SB3 algorithm;
    otherwise, uses the built-in PyTorch PPO Actor-Critic trainer so `pyalloq[rl]`
    works out-of-the-box in any PyTorch environment.
    """

    def __init__(
        self,
        env: Any,
        algorithm: AlgorithmName = "PPO",
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        policy_kwargs: dict[str, Any] | None = None,
        device: str = "cpu",
        seed: int = 42,
        use_sb3_if_available: bool = True,
        policy_backbone: PolicyBackboneType = "mlp",
        set_policy_net: SetTransformerPolicyNet | None = None,
    ) -> None:
        self.raw_env = env
        self.policy_backbone: PolicyBackboneType = (
            "set_transformer" if set_policy_net is not None else policy_backbone
        )
        if hasattr(env.observation_space, "spaces") or isinstance(
            env.observation_space, dict
        ):
            self.env: Any = DictFlattenWrapper(env)
        else:
            self.env = env

        self.algorithm: AlgorithmName = algorithm
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.policy_kwargs = policy_kwargs or {}
        self.device = device
        self.seed = seed

        obs_shape = tuple(int(d) for d in self.env.observation_space.shape)
        act_shape = tuple(int(d) for d in self.env.action_space.shape)
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(act_shape))

        self.using_sb3 = False
        self.sb3_model: Any = None
        if self.policy_backbone == "mlp" and _HAS_SB3 and use_sb3_if_available:
            try:
                algo_cls = getattr(sb3, self.algorithm)
                self.sb3_model = algo_cls(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    policy_kwargs=self.policy_kwargs,
                    device=self.device,
                    seed=self.seed,
                    verbose=0,
                )
                self.using_sb3 = True
            except (TypeError, ValueError, RuntimeError, AttributeError):
                self.using_sb3 = False

        torch.manual_seed(self.seed)
        self.set_policy_net: SetTransformerPolicyNet | None = None
        if self.policy_backbone == "set_transformer":
            if set_policy_net is not None:
                self.set_policy_net = set_policy_net.to(self.device)
            else:
                n_feats = int(getattr(self.raw_env, "n_features", 1))
                inc_cash = bool(getattr(self.raw_env, "include_cash", True))
                self.set_policy_net = SetTransformerPolicyNet(
                    n_features=n_feats,
                    hidden_dim=self.hidden_dim,
                    mandate="masked_dirichlet",
                    include_cash=inc_cash,
                ).to(self.device)

        self.fallback_net = _TorchActorCriticFallback(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        active_params = (
            self.set_policy_net.parameters()
            if self.set_policy_net is not None
            else self.fallback_net.parameters()
        )
        self.optimizer = torch.optim.Adam(active_params, lr=self.learning_rate)
        self.training_reward_history: list[float] = []

    def _unpack_structured_obs(
        self, observation: np.ndarray | dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Extracts ``(market_state (1, L, N, F), current_weights (1, N), mask (1, N))``
        from a raw dictionary observation or a ``DictFlattenWrapper`` vector.
        """
        if isinstance(observation, dict):
            mkt_np = np.asarray(
                observation.get(
                    "market_state",
                    observation.get("market", observation.get("returns")),
                ),
                dtype=np.float32,
            )
            w_np = observation.get("current_weights", observation.get("weights"))
            m_np = observation.get("tradability_mask")
        elif isinstance(self.env, DictFlattenWrapper):
            # Reconstruct dictionary slices from flattened vector
            flat = np.asarray(observation, dtype=np.float32).reshape(-1)
            reconstructed: dict[str, np.ndarray] = {}
            offset = 0
            for k in self.env.key_order:
                shp = self.env.key_shapes[k]
                sz = int(np.prod(shp))
                reconstructed[k] = flat[offset : offset + sz].reshape(shp)
                offset += sz
            mkt_np = reconstructed.get(
                "market_state", np.zeros((1, 1, 1), dtype=np.float32)
            )
            w_np = reconstructed.get("current_weights")
            m_np = reconstructed.get("tradability_mask")
        else:
            mkt_np = np.asarray(observation, dtype=np.float32)
            w_np = None
            m_np = None

        if mkt_np.ndim == 2:
            x_t = (
                torch.tensor(mkt_np, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
        elif mkt_np.ndim == 3:
            x_t = torch.tensor(
                mkt_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            x_t = torch.tensor(mkt_np, dtype=torch.float32, device=self.device)

        w_t = (
            torch.tensor(
                np.asarray(w_np, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1)
            if w_np is not None
            else None
        )
        if m_np is not None:
            mask_t = torch.tensor(
                np.asarray(m_np) > 0.5, dtype=torch.bool, device=self.device
            ).view(1, -1)
        elif hasattr(self.raw_env, "get_current_tradability_mask"):
            mask_arr = self.raw_env.get_current_tradability_mask()
            mask_t = torch.tensor(
                np.asarray(mask_arr, dtype=bool),
                dtype=torch.bool,
                device=self.device,
            ).view(1, -1)
        else:
            mask_t = None

        return x_t, w_t, mask_t

    def _flatten_obs(
        self, observation: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray:
        if isinstance(observation, dict):
            if isinstance(self.env, DictFlattenWrapper) and all(
                k in observation
                and tuple(observation[k].shape) == self.env.key_shapes[k]
                for k in self.env.key_order
            ):
                flat = self.env.observation(observation)
            else:
                parts = [
                    np.asarray(v, dtype=np.float32).reshape(-1)
                    for v in observation.values()
                ]
                flat = np.concatenate(parts, axis=0)
        else:
            flat = np.asarray(observation, dtype=np.float32).reshape(-1)

        if len(flat) < self.obs_dim:
            flat = np.pad(flat, (0, self.obs_dim - len(flat)), mode="constant")
        elif len(flat) > self.obs_dim:
            flat = flat[: self.obs_dim]
        return flat

    def _fit_set_transformer_ppo(
        self,
        total_timesteps: int,
        rollout_horizon: int,
        ppo_epochs: int,
        clip_range: float,
    ) -> SB3PortfolioAgent:
        assert self.set_policy_net is not None
        self.set_policy_net.train()
        steps_collected = 0
        obs, _ = self.raw_env.reset(seed=self.seed)

        while steps_collected < total_timesteps:
            x_buf: list[torch.Tensor] = []
            w_buf: list[torch.Tensor] = []
            m_buf: list[torch.Tensor] = []
            act_buf: list[torch.Tensor] = []
            logp_buf: list[float] = []
            rew_buf: list[float] = []
            val_buf: list[float] = []
            done_buf: list[bool] = []

            for _ in range(min(rollout_horizon, total_timesteps - steps_collected)):
                x_t, w_t, mask_t = self._unpack_structured_obs(obs)
                N = x_t.shape[2]
                if w_t is None:
                    w_t = torch.full(
                        (1, N), 1.0 / N, dtype=torch.float32, device=self.device
                    )
                if mask_t is None:
                    mask_t = torch.ones((1, N), dtype=torch.bool, device=self.device)

                with torch.no_grad():
                    alpha_full, mask_full, val_t = (
                        self.set_policy_net.get_dirichlet_concentrations(
                            x=x_t, w_current=w_t, mask=mask_t
                        )
                    )
                    alpha_safe = torch.where(
                        mask_full > 0.5,
                        alpha_full.clamp(min=1e-3),
                        torch.ones_like(alpha_full),
                    )
                    gamma_s = torch._standard_gamma(alpha_safe) * mask_full
                    act_t = gamma_s / gamma_s.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                    logp_t, _ = _masked_dirichlet_log_prob_and_entropy(
                        act_t, alpha_full, mask_full
                    )

                act_np = act_t.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, _ = self.raw_env.step(act_np)
                done = bool(terminated or truncated)

                x_buf.append(x_t.squeeze(0))
                w_buf.append(w_t.squeeze(0))
                m_buf.append(mask_t.squeeze(0))
                act_buf.append(act_t.squeeze(0))
                logp_buf.append(float(logp_t.item()))
                rew_buf.append(float(reward))
                val_buf.append(float(val_t.item()))
                done_buf.append(done)

                steps_collected += 1
                if done:
                    obs, _ = self.raw_env.reset()
                else:
                    obs = next_obs

            if not x_buf:
                break

            self.training_reward_history.append(float(np.mean(rew_buf)))

            returns = np.zeros(len(rew_buf), dtype=np.float32)
            running_ret = 0.0
            for t in reversed(range(len(rew_buf))):
                if done_buf[t]:
                    running_ret = 0.0
                running_ret = rew_buf[t] + self.gamma * running_ret
                returns[t] = running_ret

            values_arr = np.asarray(val_buf, dtype=np.float32)
            adv_arr = returns - values_arr
            adv_std = float(np.std(adv_arr))
            if adv_std > 1e-6:
                adv_arr = (adv_arr - float(np.mean(adv_arr))) / adv_std

            x_batch = torch.stack(x_buf, dim=0)
            w_batch = torch.stack(w_buf, dim=0)
            m_batch = torch.stack(m_buf, dim=0)
            act_batch = torch.stack(act_buf, dim=0)
            old_logp_batch = torch.tensor(
                logp_buf, dtype=torch.float32, device=self.device
            )
            ret_batch = torch.tensor(returns, dtype=torch.float32, device=self.device)
            adv_batch = torch.tensor(adv_arr, dtype=torch.float32, device=self.device)

            for _ in range(ppo_epochs):
                alpha_pred, mask_full_pred, val_pred = (
                    self.set_policy_net.get_dirichlet_concentrations(
                        x=x_batch, w_current=w_batch, mask=m_batch
                    )
                )
                new_logp, entropy = _masked_dirichlet_log_prob_and_entropy(
                    act_batch, alpha_pred, mask_full_pred
                )
                ratio = torch.exp(torch.clamp(new_logp - old_logp_batch, -10.0, 10.0))
                surr1 = ratio * adv_batch
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_batch
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((val_pred - ret_batch) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + value_loss - 0.01 * entropy_bonus
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.set_policy_net.parameters(), max_norm=1.0
                )
                self.optimizer.step()

        self.set_policy_net.eval()
        return self

    def fit(
        self,
        total_timesteps: int = 2000,
        rollout_Horizon: int = 64,
        ppo_epochs: int = 4,
        clip_range: float = 0.2,
        **kwargs: Any,
    ) -> SB3PortfolioAgent:
        if self.set_policy_net is not None:
            return self._fit_set_transformer_ppo(
                total_timesteps=total_timesteps,
                rollout_horizon=rollout_Horizon,
                ppo_epochs=ppo_epochs,
                clip_range=clip_range,
            )

        if self.using_sb3 and self.sb3_model is not None:
            self.sb3_model.learn(total_timesteps=total_timesteps, **kwargs)
            return self

        # PyTorch PPO Actor-Critic training loop over the wrapped environment
        self.fallback_net.train()
        steps_collected = 0
        obs, _ = self.env.reset(seed=self.seed)

        while steps_collected < total_timesteps:
            obs_buf: list[np.ndarray] = []
            act_buf: list[np.ndarray] = []
            logp_buf: list[float] = []
            rew_buf: list[float] = []
            val_buf: list[float] = []
            done_buf: list[bool] = []

            for _ in range(min(rollout_Horizon, total_timesteps - steps_collected)):
                obs_flat = self._flatten_obs(obs)
                obs_t = torch.tensor(
                    obs_flat, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    mean, val = self.fallback_net(obs_t)
                    std = torch.exp(self.fallback_net.actor_log_std).clamp(0.05, 1.0)
                    dist = torch.distributions.Normal(mean, std)
                    act_t = dist.sample()
                    logp_t = dist.log_prob(act_t).sum(dim=-1)

                act_np = act_t.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, _ = self.env.step(act_np)
                done = bool(terminated or truncated)

                obs_buf.append(obs_flat)
                act_buf.append(act_np)
                logp_buf.append(float(logp_t.item()))
                rew_buf.append(float(reward))
                val_buf.append(float(val.item()))
                done_buf.append(done)

                steps_collected += 1
                if done:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs

            if not obs_buf:
                break

            self.training_reward_history.append(float(np.mean(rew_buf)))

            # Compute discounted returns & advantages (GAE-lambda)
            returns = np.zeros(len(rew_buf), dtype=np.float32)
            running_ret = 0.0
            for t in reversed(range(len(rew_buf))):
                if done_buf[t]:
                    running_ret = 0.0
                running_ret = rew_buf[t] + self.gamma * running_ret
                returns[t] = running_ret

            values_arr = np.asarray(val_buf, dtype=np.float32)
            adv_arr = returns - values_arr
            adv_std = float(np.std(adv_arr))
            if adv_std > 1e-6:
                adv_arr = (adv_arr - float(np.mean(adv_arr))) / adv_std

            obs_batch = torch.tensor(
                np.stack(obs_buf, axis=0), dtype=torch.float32, device=self.device
            )
            act_batch = torch.tensor(
                np.stack(act_buf, axis=0), dtype=torch.float32, device=self.device
            )
            old_logp_batch = torch.tensor(
                logp_buf, dtype=torch.float32, device=self.device
            )
            ret_batch = torch.tensor(returns, dtype=torch.float32, device=self.device)
            adv_batch = torch.tensor(adv_arr, dtype=torch.float32, device=self.device)

            for _ in range(ppo_epochs):
                mean_pred, val_pred = self.fallback_net(obs_batch)
                std_pred = torch.exp(self.fallback_net.actor_log_std).clamp(0.05, 1.0)
                dist_pred = torch.distributions.Normal(mean_pred, std_pred)
                new_logp = dist_pred.log_prob(act_batch).sum(dim=-1)

                ratio = torch.exp(torch.clamp(new_logp - old_logp_batch, -10.0, 10.0))
                surr1 = ratio * adv_batch
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_batch
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((val_pred - ret_batch) ** 2).mean()
                entropy_bonus = dist_pred.entropy().mean()

                loss = policy_loss + value_loss - 0.01 * entropy_bonus
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.fallback_net.parameters(), max_norm=1.0
                )
                self.optimizer.step()

        self.fallback_net.eval()
        return self

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]:
        if self.set_policy_net is not None:
            self.set_policy_net.eval()
            with torch.no_grad():
                x_t, w_t, mask_t = self._unpack_structured_obs(observation)
                w_out = self.set_policy_net(
                    x=x_t,
                    w_current=w_t,
                    mask=mask_t,
                    deterministic=deterministic,
                )
            return w_out.squeeze(0).cpu().numpy().astype(np.float32), None

        obs_flat = self._flatten_obs(observation)
        if self.using_sb3 and self.sb3_model is not None:
            action, state = self.sb3_model.predict(
                obs_flat, deterministic=deterministic
            )
            return np.asarray(action, dtype=np.float32), state

        self.fallback_net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(
                obs_flat, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mean, _ = self.fallback_net(obs_t)
            if deterministic:
                act_t = mean
            else:
                std = torch.exp(self.fallback_net.actor_log_std).clamp(0.05, 1.0)
                act_t = torch.distributions.Normal(mean, std).sample()
        return act_t.squeeze(0).cpu().numpy().astype(np.float32), None

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.set_policy_net is not None:
            torch.save(self.set_policy_net.state_dict(), out_path)
        elif self.using_sb3 and self.sb3_model is not None:
            self.sb3_model.save(str(out_path))
        else:
            torch.save(self.fallback_net.state_dict(), out_path)
        return out_path

    def load(self, path: str | Path) -> SB3PortfolioAgent:
        in_path = Path(path)
        if self.set_policy_net is not None:
            state_dict = torch.load(in_path, map_location=self.device)
            self.set_policy_net.load_state_dict(state_dict)
            self.set_policy_net.eval()
        elif self.using_sb3 and _HAS_SB3 and self.sb3_model is not None:
            algo_cls = getattr(sb3, self.algorithm)
            self.sb3_model = algo_cls.load(str(in_path), env=self.env)
        else:
            state_dict = torch.load(in_path, map_location=self.device)
            self.fallback_net.load_state_dict(state_dict)
            self.fallback_net.eval()
        return self
