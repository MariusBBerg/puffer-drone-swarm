"""Random walk baseline."""

from __future__ import annotations

import numpy as np

from env import EnvConfig
from baselines.base import BaselinePolicy


class RandomWalkBaseline(BaselinePolicy):
    name = "random_walk"

    def __init__(self, env_config: EnvConfig, seed: int = 0):
        self._waypoints = None
        super().__init__(env_config, seed=seed)

    def reset(self, obs: np.ndarray | None = None, seed: int | None = None) -> None:
        super().reset(obs=obs, seed=seed)
        self._waypoints = np.full((self.n_drones, 2), np.nan, dtype=np.float32)
        if obs is not None:
            self._update_waypoints(self._positions_from_obs(obs))

    def _update_waypoints(self, positions: np.ndarray) -> None:
        need_new = np.isnan(self._waypoints[:, 0])
        if not need_new.any():
            return
        self._waypoints[need_new] = self.rng.uniform(
            0.0, self.world_size, size=(need_new.sum(), 2)
        ).astype(np.float32)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        positions = self._positions_from_obs(obs)
        self._update_waypoints(positions)

        diff = self._waypoints - positions
        dist = np.linalg.norm(diff, axis=1)
        reached = dist < 1.5
        if reached.any():
            self._waypoints[reached] = self.rng.uniform(
                0.0, self.world_size, size=(reached.sum(), 2)
            ).astype(np.float32)
            diff = self._waypoints - positions

        directions = self._direction_to_target(positions, self._waypoints)
        scan_mask = self.rng.random(self.n_drones) > 0.7
        return self._action_array(directions, scan_mask)
