"""Baseline policy helpers for non-learning controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from env import EnvConfig


POS_SLICE = slice(0, 2)
CONNECTED_IDX = 4
DIST_TO_BASE_IDX = 3
TO_BASE_SLICE = slice(8, 10)


@dataclass
class BaselineState:
    step: int = 0


class BaselinePolicy:
    """Base class for hand-coded baselines.

    Provides utilities for parsing observations and emitting actions.
    """

    name = "baseline"

    def __init__(self, env_config: EnvConfig, seed: int = 0):
        self.cfg = env_config
        self.n_drones = int(env_config.n_drones)
        self.world_size = float(env_config.world_size)
        if env_config.base_pos is None:
            self.base_pos = np.array(
                [self.world_size / 2.0, self.world_size / 2.0], dtype=np.float32
            )
        else:
            self.base_pos = np.array(env_config.base_pos, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.state = BaselineState()
        self.reset()

    def reset(self, obs: Optional[np.ndarray] = None, seed: Optional[int] = None) -> None:
        """Reset per-episode state. Optionally reseed RNG and sync with obs."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = BaselineState(step=0)
        if obs is not None:
            self._sync_from_obs(obs)

    def _sync_from_obs(self, obs: np.ndarray) -> None:
        """Hook for subclasses to initialize state from observation."""
        return None

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Return actions with shape (n_drones, 3)."""
        raise NotImplementedError

    def get_action_and_value(self, obs: np.ndarray):
        """Mimic DroneSwarmPolicy interface with dummy logprob/entropy/value."""
        actions = self.get_action(obs)
        n = actions.shape[0]
        logprob = np.zeros(n, dtype=np.float32)
        entropy = np.zeros(n, dtype=np.float32)
        value = np.zeros((n, 1), dtype=np.float32)
        return actions, logprob, entropy, value

    def _positions_from_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs[:, POS_SLICE] * self.world_size

    def _connected_from_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs[:, CONNECTED_IDX] > 0.5

    def _direction_to_target(
        self,
        positions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        diff = targets - positions
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.where(norms > 1e-6, norms, 1.0)
        directions = diff / norms
        return np.clip(directions, -1.0, 1.0)

    def _action_array(self, directions: np.ndarray, scan_mask: np.ndarray) -> np.ndarray:
        actions = np.zeros((self.n_drones, 3), dtype=np.float32)
        actions[:, 0:2] = directions.astype(np.float32)
        actions[:, 2] = np.where(scan_mask, 1.0, -1.0).astype(np.float32)
        return actions
