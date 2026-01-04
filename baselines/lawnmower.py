"""Lawnmower coverage baseline."""

from __future__ import annotations

import numpy as np

from env_config import EnvConfig
from baselines.base import BaselinePolicy


class LawnmowerBaseline(BaselinePolicy):
    name = "lawnmower"

    def __init__(self, env_config: EnvConfig, seed: int = 0):
        self._strip_bounds = None
        self._lane_x = None
        self._y_dir = None
        self._relay_idx = 0
        self._scan_width = None
        super().__init__(env_config, seed=seed)

    def reset(self, obs: np.ndarray | None = None, seed: int | None = None) -> None:
        super().reset(obs=obs, seed=seed)
        strip_width = self.world_size / max(self.n_drones, 1)
        self._strip_bounds = np.zeros((self.n_drones, 2), dtype=np.float32)
        for i in range(self.n_drones):
            self._strip_bounds[i, 0] = i * strip_width
            self._strip_bounds[i, 1] = (i + 1) * strip_width
        self._scan_width = min(strip_width * 0.8, max(2.0, self.cfg.r_sense * 0.5))
        self._lane_x = (
            self._strip_bounds[:, 0] + 0.5 * strip_width
        ).astype(np.float32)
        self._y_dir = np.ones(self.n_drones, dtype=np.int8)

        if obs is not None:
            positions = self._positions_from_obs(obs)
            dists = np.linalg.norm(positions - self.base_pos[None, :], axis=1)
            self._relay_idx = int(np.argmin(dists))
            self._y_dir = np.where(positions[:, 1] < self.world_size / 2.0, 1, -1)
        else:
            self._relay_idx = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        positions = self._positions_from_obs(obs)
        targets = np.zeros_like(positions)

        for i in range(self.n_drones):
            if i == self._relay_idx:
                targets[i] = self.base_pos
                continue

            strip_min, strip_max = self._strip_bounds[i]
            target_y = self.world_size if self._y_dir[i] > 0 else 0.0
            targets[i] = np.array([self._lane_x[i], target_y], dtype=np.float32)

            if np.linalg.norm(targets[i] - positions[i]) < 1.5:
                self._y_dir[i] *= -1
                self._lane_x[i] += self._scan_width
                if self._lane_x[i] > strip_max:
                    self._lane_x[i] = strip_min + 0.5 * self._scan_width
                if self._lane_x[i] < strip_min:
                    self._lane_x[i] = strip_min + 0.5 * self._scan_width
                target_y = self.world_size if self._y_dir[i] > 0 else 0.0
                targets[i] = np.array([self._lane_x[i], target_y], dtype=np.float32)

        directions = self._direction_to_target(positions, targets)
        scan_mask = np.ones(self.n_drones, dtype=bool)
        return self._action_array(directions, scan_mask)
