"""Voronoi partition baseline with relay drone."""

from __future__ import annotations

import numpy as np

from env import EnvConfig
from baselines.base import BaselinePolicy


class VoronoiBaseline(BaselinePolicy):
    name = "voronoi"

    def __init__(self, env_config: EnvConfig, seed: int = 0):
        self._centroids = None
        self._relay_idx = 0
        self._bounds = None
        self._angles = None
        self._radii = None
        self._grid_size = 20
        self._cell_size = None
        self._lloyd_iters = 3
        super().__init__(env_config, seed=seed)

    def reset(self, obs: np.ndarray | None = None, seed: int | None = None) -> None:
        super().reset(obs=obs, seed=seed)
        self._cell_size = self.world_size / self._grid_size
        if obs is not None:
            positions = self._positions_from_obs(obs)
        else:
            positions = np.tile(self.base_pos[None, :], (self.n_drones, 1))

        self._centroids = positions.copy()
        self._compute_voronoi(positions)
        dists = np.linalg.norm(positions - self.base_pos[None, :], axis=1)
        self._relay_idx = int(np.argmin(dists))
        self._angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.n_drones).astype(np.float32)
        self._radii = np.full(self.n_drones, self._cell_size, dtype=np.float32)

    def _compute_voronoi(self, positions: np.ndarray) -> None:
        grid_x = (np.arange(self._grid_size) + 0.5) * self._cell_size
        grid_y = (np.arange(self._grid_size) + 0.5) * self._cell_size
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_points = np.stack([gx, gy], axis=-1).reshape(-1, 2)

        centroids = positions.copy()
        for _ in range(self._lloyd_iters):
            dists = np.linalg.norm(grid_points[:, None, :] - centroids[None, :, :], axis=2)
            assignment = np.argmin(dists, axis=1)
            for i in range(self.n_drones):
                mask = assignment == i
                if np.any(mask):
                    centroids[i] = grid_points[mask].mean(axis=0)
        self._centroids = centroids

        self._bounds = np.zeros((self.n_drones, 4), dtype=np.float32)
        dists = np.linalg.norm(grid_points[:, None, :] - centroids[None, :, :], axis=2)
        assignment = np.argmin(dists, axis=1)
        for i in range(self.n_drones):
            mask = assignment == i
            if not np.any(mask):
                self._bounds[i] = np.array([0.0, 0.0, self.world_size, self.world_size])
                continue
            pts = grid_points[mask]
            minx = float(np.min(pts[:, 0]))
            maxx = float(np.max(pts[:, 0]))
            miny = float(np.min(pts[:, 1]))
            maxy = float(np.max(pts[:, 1]))
            self._bounds[i] = np.array([minx, miny, maxx, maxy], dtype=np.float32)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        positions = self._positions_from_obs(obs)
        targets = np.zeros_like(positions)

        angle_step = np.pi / 4.0
        radius_step = max(2.0, self._cell_size)

        for i in range(self.n_drones):
            if i == self._relay_idx:
                targets[i] = self.base_pos
                continue

            minx, miny, maxx, maxy = self._bounds[i]
            max_radius = 0.5 * max(maxx - minx, maxy - miny)
            angle = float(self._angles[i])
            radius = float(self._radii[i])

            target = self._centroids[i] + np.array(
                [np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32
            )
            target = np.clip(target, [minx, miny], [maxx, maxy])

            if np.linalg.norm(target - positions[i]) < 1.5:
                angle += angle_step
                if angle >= 2.0 * np.pi:
                    angle -= 2.0 * np.pi
                radius += radius_step
                if radius > max_radius:
                    radius = radius_step
                self._angles[i] = angle
                self._radii[i] = radius
                target = self._centroids[i] + np.array(
                    [np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32
                )
                target = np.clip(target, [minx, miny], [maxx, maxy])

            targets[i] = target

        directions = self._direction_to_target(positions, targets)
        scan_mask = np.ones(self.n_drones, dtype=bool)
        return self._action_array(directions, scan_mask)
