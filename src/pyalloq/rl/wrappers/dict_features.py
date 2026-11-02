from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _WRAPPER_BASE: type = gym.ObservationWrapper
except ImportError:
    gym = None
    spaces = None
    _WRAPPER_BASE = object


class _BoxSpaceFallback:
    """Lightweight fallback for `gymnasium.spaces.Box` when `gymnasium` is not installed."""

    def __init__(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
        dtype: Any = np.float32,
    ) -> None:
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        low_c = np.clip(self.low, -10.0, 10.0)
        high_c = np.clip(self.high, -10.0, 10.0)
        return np.random.uniform(low_c, high_c, size=self.shape).astype(self.dtype)


def _make_box_space(
    low: float,
    high: float,
    shape: tuple[int, ...],
    dtype: Any = np.float32,
) -> Any:
    if spaces is not None:
        return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _BoxSpaceFallback(low=low, high=high, shape=shape, dtype=dtype)


class _DictSpaceFallback:
    """Lightweight fallback for `gymnasium.spaces.Dict` when `gymnasium` is not installed."""

    def __init__(self, space_map: dict[str, Any]) -> None:
        self.spaces = dict(space_map)

    def __getitem__(self, key: str) -> Any:
        return self.spaces[key]

    def keys(self) -> list[str]:
        return list(self.spaces.keys())

    def sample(self) -> dict[str, np.ndarray]:
        return {k: v.sample() for k, v in self.spaces.items()}


def make_dict_space(space_map: dict[str, Any]) -> Any:
    if spaces is not None:
        return spaces.Dict(space_map)
    return _DictSpaceFallback(space_map)


class DictFlattenWrapper(_WRAPPER_BASE):  # type: ignore[misc]
    """
    Feature Bridge (`wrappers/dict_features.py`):
    Intercepts a dynamically sized `spaces.Dict` observation from `POMDPPortfolioEnv`
    or `MetaPortfolioEnv` (containing `"market_state"`, `"current_weights"`, and
    `"portfolio_meta"`) and flattens it into a single 1D continuous `np.float32` vector
    for Stable-Baselines3 MLP policies (`PPO`, `SAC`, `DDPG`, `TD3`).

    Automatically computes the exact flattened dimension `flat_dim` from the underlying
    `observation_space` and exposes `.unflatten(flat_obs)` to reconstruct the original
    structured dictionary tensor shapes.
    """

    DEFAULT_KEY_ORDER: tuple[str, ...] = (
        "market_state",
        "current_weights",
        "portfolio_meta",
    )

    def __init__(
        self,
        env: Any,
        key_order: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        if gym is not None and isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        obs_space = env.observation_space
        space_dict: dict[str, Any] = getattr(obs_space, "spaces", obs_space)

        if key_order is not None:
            self.key_order = tuple(k for k in key_order if k in space_dict)
        else:
            ordered = [k for k in self.DEFAULT_KEY_ORDER if k in space_dict]
            extras = [k for k in space_dict.keys() if k not in ordered]
            self.key_order = tuple(ordered + extras)

        self.key_shapes: dict[str, tuple[int, ...]] = {}
        self.key_slices: dict[str, slice] = {}
        offset = 0
        for k in self.key_order:
            sub_space = space_dict[k]
            shape = tuple(int(d) for d in sub_space.shape)
            size = int(np.prod(shape))
            self.key_shapes[k] = shape
            self.key_slices[k] = slice(offset, offset + size)
            offset += size

        self.flat_dim = offset
        self.observation_space = _make_box_space(
            low=-np.inf,
            high=np.inf,
            shape=(self.flat_dim,),
            dtype=np.float32,
        )
        self.action_space = env.action_space

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Flattens a dictionary observation into a 1D `np.float32` array."""
        parts = [
            np.asarray(observation[k], dtype=np.float32).reshape(-1)
            for k in self.key_order
        ]
        return np.concatenate(parts, axis=0)

    def unflatten(self, flat_obs: np.ndarray) -> dict[str, np.ndarray]:
        """Reconstructs the structured dictionary of arrays from a 1D flattened vector."""
        arr = np.asarray(flat_obs, dtype=np.float32).reshape(-1)
        if len(arr) != self.flat_dim:
            raise ValueError(
                f"Expected flattened observation of length {self.flat_dim}, got {len(arr)}."
            )
        out: dict[str, np.ndarray] = {}
        for k in self.key_order:
            sl = self.key_slices[k]
            out[k] = arr[sl].reshape(self.key_shapes[k])
        return out

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs_dict, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs_dict), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs_dict), reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class FeatureNormalizationWrapper(_WRAPPER_BASE):  # type: ignore[misc]
    """
    Online Welford running z-score normalizer for RL observations (supports both
    `spaces.Dict` environments—normalizing `"market_state"` while preserving simplex
    `"current_weights"`—and 1D `spaces.Box` environments).
    """

    def __init__(self, env: Any, clip: float = 10.0, eps: float = 1e-6) -> None:
        if gym is not None and isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env
        self.clip = clip
        self.eps = eps
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        space_dict = getattr(self.observation_space, "spaces", None)
        if space_dict is not None and "market_state" in space_dict:
            self._is_dict = True
            shape = tuple(int(d) for d in space_dict["market_state"].shape)
        else:
            self._is_dict = False
            shape = tuple(int(d) for d in self.observation_space.shape)

        self._count: float = 1e-4
        self._mean = np.zeros(shape, dtype=np.float64)
        self._m2 = np.ones(shape, dtype=np.float64)

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float64)
        self._count += 1.0
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2
        var = self._m2 / max(1.0, self._count)
        normed = (x - self._mean) / np.sqrt(var + self.eps)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def observation(
        self, observation: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        if isinstance(observation, dict):
            out = dict(observation)
            if "market_state" in out:
                out["market_state"] = self._normalize_array(out["market_state"])
            return out
        return self._normalize_array(observation)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
