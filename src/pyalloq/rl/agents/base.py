from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseRLAgent(ABC):
    """
    Abstract Base Class for Reinforcement Learning Portfolio Agents in `pyalloq[rl]`.
    """

    @abstractmethod
    def fit(self, total_timesteps: int = 5000, **kwargs: Any) -> BaseRLAgent:
        """Trains the RL policy on the attached environment for `total_timesteps`."""
        ...

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]:
        """Returns `(action, state)` for a given environment observation."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Serializes the trained policy parameters to `path`."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> BaseRLAgent:
        """Loads serialized policy parameters from `path`."""
        ...
