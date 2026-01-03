"""Minimal core environment for the PufferLib Drone Swarm SAR v1.

This is intentionally small: continuous 2D motion, comm connectivity,
scan/confirm/deliver loop, and a compact observation vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    world_size: float = 100.0
    n_drones: int = 8
    n_victims: int = 10
    r_comm: float = 15.0
    r_comm_min: float = 0.0
    r_comm_max: float = 0.0
    r_confirm_radius: float = 3.0
    r_sense: float = 15.0  # Slightly larger sense range
    t_confirm: int = 4
    t_confirm_values: Tuple[int, ...] = ()
    m_deliver: int = 15
    m_deliver_values: Tuple[int, ...] = ()
    v_max: float = 2.5
    dt: float = 1.0
    # Battery costs
    c_idle: float = 0.0003
    c_move: float = 0.0006
    c_scan: float = 0.0002
    # Reward shaping
    c_time: float = 0.005
    c_energy: float = 0.0
    r_found: float = 30.0    # Big reward for delivery
    r_found_divide_by_n: bool = True
    r_confirm_reward: float = 10.0  # Good reward for confirming
    r_approach: float = 0.0  # DISABLED - was causing herding
    r_explore: float = 0.1   # Individual exploration reward
    r_scan_near_victim: float = 0.3
    r_connectivity: float = 0.01
    r_dispersion: float = 0.05  # NEW: reward for spreading out
    r_owner_connected: float = 0.0  # Bonus when deliver-owner is connected for confirmed victim
    r_relay_bonus: float = 0.0  # Bonus for drones serving as relay nodes for disconnected owners
    r_chain_progress: float = 0.0  # Potential shaping: reward progress toward connecting owner to base
    # Detection model (used when scan=1)
    detect_prob_scale: float = 1.0
    detect_noise_std: float = 0.0
    false_positive_rate: float = 0.0
    false_positive_confidence: float = 0.3
    p_comm_drop: float = 0.0  # Probability of comm drop per link per step
    p_comm_drop_min: float = 0.0
    p_comm_drop_max: float = 0.0
    min_drone_separation: float = 8.0  # Drones should stay this far apart
    max_steps: int = 800
    base_pos: Optional[Tuple[float, float]] = None
    spawn_near_base: bool = True
    spawn_radius: Optional[float] = None
    obs_n_nearest: int = 3
    obs_n_obstacles: int = 0
    # New: victim spawn control
    victim_min_dist_from_base: float = 25.0  # Victims spawn away from base
    victim_max_dist_from_base: float = 90.0  # But not at edges
    victim_mix_prob: float = 0.0  # Probability of using alternate victim distance range
    victim_min_dist_from_base_alt: float = 0.0
    victim_max_dist_from_base_alt: float = 0.0
    # Obstacles (axis-aligned rectangles)
    obstacle_count: int = 0
    obstacle_min_size: float = 8.0
    obstacle_max_size: float = 25.0
    obstacle_margin: float = 5.0
    obstacle_base_clearance: float = 0.0
    obstacle_min_separation: float = 2.0
    obstacle_blocks_sensing: bool = True
    obstacles: Tuple[Tuple[float, float, float, float], ...] = ()

    @classmethod
    def from_dict(cls, data: dict) -> "EnvConfig":
        payload = dict(data)
        if "r_confirm" in payload and "r_confirm_radius" not in payload:
            payload["r_confirm_radius"] = payload.pop("r_confirm")
        if "r_confirm_reward" not in payload:
            payload["r_confirm_reward"] = cls().r_confirm_reward
        if "r_found_divide_by_n" not in payload:
            payload["r_found_divide_by_n"] = cls().r_found_divide_by_n
        if "r_owner_connected" not in payload:
            payload["r_owner_connected"] = cls().r_owner_connected
        if "r_relay_bonus" not in payload:
            payload["r_relay_bonus"] = cls().r_relay_bonus
        if "r_chain_progress" not in payload:
            payload["r_chain_progress"] = cls().r_chain_progress
        if "p_comm_drop" not in payload:
            payload["p_comm_drop"] = cls().p_comm_drop
        if "p_comm_drop_min" not in payload:
            payload["p_comm_drop_min"] = cls().p_comm_drop_min
        if "p_comm_drop_max" not in payload:
            payload["p_comm_drop_max"] = cls().p_comm_drop_max
        if "r_comm_min" not in payload:
            payload["r_comm_min"] = cls().r_comm_min
        if "r_comm_max" not in payload:
            payload["r_comm_max"] = cls().r_comm_max
        if "detect_prob_scale" not in payload:
            payload["detect_prob_scale"] = cls().detect_prob_scale
        if "detect_noise_std" not in payload:
            payload["detect_noise_std"] = cls().detect_noise_std
        if "false_positive_rate" not in payload:
            payload["false_positive_rate"] = cls().false_positive_rate
        if "false_positive_confidence" not in payload:
            payload["false_positive_confidence"] = cls().false_positive_confidence
        if "t_confirm_values" not in payload:
            payload["t_confirm_values"] = cls().t_confirm_values
        elif isinstance(payload["t_confirm_values"], list):
            payload["t_confirm_values"] = tuple(payload["t_confirm_values"])
        if "m_deliver_values" not in payload:
            payload["m_deliver_values"] = cls().m_deliver_values
        elif isinstance(payload["m_deliver_values"], list):
            payload["m_deliver_values"] = tuple(payload["m_deliver_values"])
        if "victim_mix_prob" not in payload:
            payload["victim_mix_prob"] = cls().victim_mix_prob
        if "victim_min_dist_from_base_alt" not in payload:
            payload["victim_min_dist_from_base_alt"] = cls().victim_min_dist_from_base_alt
        if "victim_max_dist_from_base_alt" not in payload:
            payload["victim_max_dist_from_base_alt"] = cls().victim_max_dist_from_base_alt
        if "obs_n_obstacles" not in payload:
            payload["obs_n_obstacles"] = cls().obs_n_obstacles
        if "obstacle_count" not in payload:
            payload["obstacle_count"] = cls().obstacle_count
        if "obstacle_min_size" not in payload:
            payload["obstacle_min_size"] = cls().obstacle_min_size
        if "obstacle_max_size" not in payload:
            payload["obstacle_max_size"] = cls().obstacle_max_size
        if "obstacle_margin" not in payload:
            payload["obstacle_margin"] = cls().obstacle_margin
        if "obstacle_base_clearance" not in payload:
            payload["obstacle_base_clearance"] = cls().obstacle_base_clearance
        if "obstacle_min_separation" not in payload:
            payload["obstacle_min_separation"] = cls().obstacle_min_separation
        if "obstacle_blocks_sensing" not in payload:
            payload["obstacle_blocks_sensing"] = cls().obstacle_blocks_sensing
        if "obstacles" not in payload:
            payload["obstacles"] = cls().obstacles
        elif isinstance(payload["obstacles"], list):
            payload["obstacles"] = tuple(payload["obstacles"])
        return cls(**payload)


class DroneSwarmEnv:
    """Small, fast core loop. No occupancy grid or sensor model yet."""

    def __init__(self, config: EnvConfig = EnvConfig()):
        self.cfg = config
        if self.cfg.base_pos is None:
            self.base_pos = np.array(
                [self.cfg.world_size / 2.0, self.cfg.world_size / 2.0], dtype=np.float32
            )
        else:
            self.base_pos = np.array(self.cfg.base_pos, dtype=np.float32)

        self.rng = np.random.default_rng()
        self.step_count = 0

        self.positions = np.zeros((self.cfg.n_drones, 2), dtype=np.float32) # (n_drones,2) each drone's (x,y) position
        self.battery = np.ones(self.cfg.n_drones, dtype=np.float32) # (n_drones,) each drone's battery level [0,1]
        self.last_comm_age = np.zeros(self.cfg.n_drones, dtype=np.float32) # (n_drones,) steps since last comm with base
        self.connected = np.zeros(self.cfg.n_drones, dtype=bool) # (n_drones,) whether each drone is connected to base (true,false)

        self.victim_pos = np.zeros((self.cfg.n_victims, 2), dtype=np.float32) # (n_victims,2) each victim's (x,y) position
        self.victim_status = np.zeros(self.cfg.n_victims, dtype=np.int32) # (n_victims,) each victim's status: 0=unknown, 1=confirmed, 2=delivered
        self.confirm_progress = np.zeros(self.cfg.n_victims, dtype=np.int32) # how many consecutive scan steps the current confirming drone has scanned this victim
        self.confirm_owner = -np.ones(self.cfg.n_victims, dtype=np.int32) # which drone is currently confirming this victim (used during scanning)
        self.deliver_owner = -np.ones(self.cfg.n_victims, dtype=np.int32) # which drone confirmed this victim (used during delivery)
        self.delivery_ttl = -np.ones(self.cfg.n_victims, dtype=np.int32) # how many steps remain to deliver before it expires
        self.detections = np.zeros((self.cfg.n_drones, self.cfg.obs_n_nearest, 3), dtype=np.float32)
        self.obstacles = np.zeros((0, 4), dtype=np.float32)
        self.obstacle_count = 0

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        L = self.cfg.world_size

        if self.cfg.r_comm_max > self.cfg.r_comm_min > 0.0:
            self.cfg.r_comm = float(self.rng.uniform(self.cfg.r_comm_min, self.cfg.r_comm_max))

        if self.cfg.p_comm_drop_max > self.cfg.p_comm_drop_min >= 0.0:
            self.cfg.p_comm_drop = float(
                self.rng.uniform(self.cfg.p_comm_drop_min, self.cfg.p_comm_drop_max)
            )

        if self.cfg.t_confirm_values:
            self.cfg.t_confirm = int(self.rng.choice(self.cfg.t_confirm_values))

        if self.cfg.m_deliver_values:
            self.cfg.m_deliver = int(self.rng.choice(self.cfg.m_deliver_values))

        self._init_obstacles()

        if self.cfg.spawn_near_base:
            radius = self.cfg.spawn_radius
            if radius is None:
                radius = self.cfg.r_comm
            if self.obstacle_count == 0:
                angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.cfg.n_drones)
                radii = np.sqrt(self.rng.uniform(0.0, 1.0, size=self.cfg.n_drones)) * radius
                offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None]
                self.positions = (self.base_pos[None, :] + offsets).astype(np.float32)
                self.positions = np.clip(self.positions, 0.0, L)
            else:
                for i in range(self.cfg.n_drones):
                    for _ in range(100):
                        angle = self.rng.uniform(0.0, 2.0 * np.pi)
                        r = np.sqrt(self.rng.uniform(0.0, 1.0)) * radius
                        pos = self.base_pos + np.array([np.cos(angle), np.sin(angle)]) * r
                        pos = np.clip(pos, 0.0, L)
                        if not self._point_in_any_obstacle(pos):
                            self.positions[i] = pos
                            break
                    else:
                        self.positions[i] = self.base_pos
        else:
            if self.obstacle_count == 0:
                self.positions = self.rng.uniform(0.0, L, size=(self.cfg.n_drones, 2)).astype(
                    np.float32
                )
            else:
                for i in range(self.cfg.n_drones):
                    for _ in range(100):
                        pos = self.rng.uniform(0.0, L, size=2)
                        if not self._point_in_any_obstacle(pos):
                            self.positions[i] = pos.astype(np.float32)
                            break
                    else:
                        self.positions[i] = self.base_pos
        self.battery.fill(1.0)
        self.last_comm_age.fill(0.0)

        # Spawn victims away from base (in an annulus)
        min_dist = self.cfg.victim_min_dist_from_base
        max_dist = self.cfg.victim_max_dist_from_base
        if (
            self.cfg.victim_mix_prob > 0.0
            and self.cfg.victim_max_dist_from_base_alt > self.cfg.victim_min_dist_from_base_alt
        ):
            if self.rng.uniform(0.0, 1.0) < self.cfg.victim_mix_prob:
                min_dist = self.cfg.victim_min_dist_from_base_alt
                max_dist = self.cfg.victim_max_dist_from_base_alt
        self.victim_pos = self._spawn_victims_in_annulus(min_dist, max_dist)
        self.victim_status.fill(0)  # 0=unknown, 1=confirmed, 2=delivered
        self.confirm_progress.fill(0)
        self.confirm_owner.fill(-1)
        self.deliver_owner.fill(-1)
        self.delivery_ttl.fill(-1)
        self.detections.fill(0.0)

        self.connected = self._compute_connectivity()
        self._update_comm_age()
        
        # Track exploration coverage - per drone
        self.explored_cells = set()
        self.drone_explored_cells = [set() for _ in range(self.cfg.n_drones)]
        self._update_explored_cells_per_drone()

        return self._get_obs()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.cfg.n_drones, 3):
            raise ValueError(
                f"actions must have shape (n_drones, 3), got {actions.shape}"
            )

        # Actions: (vx, vy, scan) where vx/vy are in [-1, 1]
        vel = actions[:, 0:2]
        scan = actions[:, 2] > 0.0

        dead = self.battery <= 0.0
        if np.any(dead):
            vel[dead] = 0.0
            scan[dead] = False

        # Normalize and scale velocity to v_max (euclidean norm)
        norms = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(norms > 1.0, vel / (norms + 1e-8), vel)
        vel = vel * self.cfg.v_max

        old_positions = self.positions.copy()
        self.positions += vel * self.cfg.dt
        self.positions = np.clip(self.positions, 0.0, self.cfg.world_size)
        if self.obstacle_count:
            for i in range(self.cfg.n_drones):
                if self._point_in_any_obstacle(self.positions[i]) or self._segment_intersects_any_obstacle(
                    old_positions[i], self.positions[i]
                ):
                    self.positions[i] = old_positions[i]

        speed = np.linalg.norm(vel, axis=1)
        speed_ratio = speed / max(self.cfg.v_max, 1e-6)
        self.battery -= (self.cfg.c_idle + self.cfg.c_move * (speed_ratio**2)) * self.cfg.dt
        self.battery = np.clip(self.battery, 0.0, 1.0)

        self.connected = self._compute_connectivity()
        self._update_comm_age()

        new_delivered, new_confirmed, new_expired, confirming_drone = self._update_confirm_and_delivery(scan)
        self.detections = self._compute_detections(scan)
        
        # INDIVIDUAL exploration tracking - which drone explored new cells
        drone_explored = self._update_explored_cells_per_drone()
        
        # INDIVIDUAL scan reward - only for drones actually scanning near victims
        scan_rewards = np.zeros(self.cfg.n_drones, dtype=np.float32)
        if scan.any():
            unconfirmed = self.victim_status == 0
            if unconfirmed.any():
                for d in range(self.cfg.n_drones):
                    if scan[d]:
                        dists = np.linalg.norm(self.victim_pos[unconfirmed] - self.positions[d], axis=1)
                        if (dists <= self.cfg.r_confirm_radius * 2).any():
                            scan_rewards[d] = self.cfg.r_scan_near_victim
        
        # DISPERSION reward - reward drones for maintaining distance from each other
        dispersion_rewards = self._compute_dispersion_rewards()
        
        # Build INDIVIDUAL rewards
        rewards = np.zeros(self.cfg.n_drones, dtype=np.float32)
        
        # Base time penalty for all
        rewards -= self.cfg.c_time
        
        # Energy penalty (per-drone)
        if self.cfg.c_energy != 0.0:
            rewards -= self.cfg.c_energy * (speed_ratio**2)
        
        # Team rewards (shared)
        team_reward = self.cfg.r_found * float(new_delivered)
        if self.cfg.r_found_divide_by_n and self.cfg.n_drones > 0:
            team_reward = team_reward / self.cfg.n_drones
        rewards += team_reward
        
        # Individual confirm reward - only the drone that confirmed gets it
        if new_confirmed > 0 and confirming_drone >= 0:
            rewards[confirming_drone] += self.cfg.r_confirm_reward
        
        # Individual exploration rewards
        rewards += self.cfg.r_explore * drone_explored
        
        # Scan cost (per-drone)
        if self.cfg.c_scan != 0.0:
            rewards -= self.cfg.c_scan * scan.astype(np.float32)
        
        # Individual scan rewards  
        rewards += scan_rewards
        
        # Individual dispersion rewards
        rewards += dispersion_rewards
        
        # Connectivity reward (individual - only connected drones)
        rewards += self.cfg.r_connectivity * self.connected.astype(np.float32)

        # Deliver-owner connected bonus for confirmed victims
        if self.cfg.r_owner_connected != 0.0:
            for v in range(self.cfg.n_victims):
                if self.victim_status[v] == 1:
                    owner = self.deliver_owner[v]
                    if owner >= 0 and self.connected[owner]:
                        rewards[owner] += self.cfg.r_owner_connected

        # Relay rewards: reward drones that form relay chains for disconnected owners
        if self.cfg.r_relay_bonus != 0.0 or self.cfg.r_chain_progress != 0.0:
            relay_rewards, chain_progress = self._compute_relay_rewards()
            rewards += relay_rewards
            rewards += chain_progress  # Distributed to all agents

        self.step_count += 1
        done = bool(
            (self.victim_status == 2).all() or self.step_count >= self.cfg.max_steps
        )

        info = {
            "step": self.step_count,
            "delivered": int((self.victim_status == 2).sum()),
            "confirmed": int((self.victim_status == 1).sum()),
            "connected_fraction": float(np.mean(self.connected)),
            "new_delivered": int(new_delivered),
            "new_confirmed": int(new_confirmed),
            "new_expired": int(new_expired),
            "explored_cells": len(self.explored_cells),
        }

        return self._get_obs(), rewards, done, info

    def _compute_connectivity(self):
        N = self.cfg.n_drones
        if N == 0:
            return np.zeros(0, dtype=bool)

        pos = self.positions
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        adj = dist <= self.cfg.r_comm
        np.fill_diagonal(adj, False)

        dist_base = np.linalg.norm(pos - self.base_pos[None, :], axis=1)
        connected = np.zeros(N, dtype=bool)
        base_links = dist_base <= self.cfg.r_comm
        if self.cfg.p_comm_drop > 0.0:
            base_drop = self.rng.uniform(0.0, 1.0, size=N) < self.cfg.p_comm_drop
            base_links = base_links & ~base_drop
        queue = list(np.where(base_links)[0])
        for i in queue:
            connected[i] = True

        if self.cfg.p_comm_drop > 0.0:
            rand = self.rng.uniform(0.0, 1.0, size=(N, N))
            rand = np.triu(rand, 1)
            rand = rand + rand.T
            adj = adj & (rand >= self.cfg.p_comm_drop)

        # BFS over comm graph
        while queue:
            i = queue.pop()
            neighbors = np.where(adj[i] & ~connected)[0]
            if neighbors.size:
                connected[neighbors] = True
                queue.extend(neighbors.tolist())

        return connected

    def _update_comm_age(self):
        self.last_comm_age = np.where(
            self.connected, 0.0, self.last_comm_age + 1.0
        ).astype(np.float32)

    def _point_in_rect(self, pos: np.ndarray, rect: np.ndarray) -> bool:
        return (rect[0] <= pos[0] <= rect[2]) and (rect[1] <= pos[1] <= rect[3])

    def _point_in_any_obstacle(self, pos: np.ndarray) -> bool:
        for rect in self.obstacles:
            if self._point_in_rect(pos, rect):
                return True
        return False

    def _distance_point_to_rect(self, pos: np.ndarray, rect: np.ndarray) -> float:
        cx = np.clip(pos[0], rect[0], rect[2])
        cy = np.clip(pos[1], rect[1], rect[3])
        dx = cx - pos[0]
        dy = cy - pos[1]
        return float(np.hypot(dx, dy))

    def _segment_intersects_rect(self, p0: np.ndarray, p1: np.ndarray, rect: np.ndarray) -> bool:
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        p = [-dx, dx, -dy, dy]
        q = [x0 - rect[0], rect[2] - x0, y0 - rect[1], rect[3] - y0]
        u1 = 0.0
        u2 = 1.0
        for pi, qi in zip(p, q):
            if pi == 0.0:
                if qi < 0.0:
                    return False
            else:
                t = qi / pi
                if pi < 0.0:
                    if t > u2:
                        return False
                    if t > u1:
                        u1 = t
                else:
                    if t < u1:
                        return False
                    if t < u2:
                        u2 = t
        return True

    def _segment_intersects_any_obstacle(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        for rect in self.obstacles:
            if self._segment_intersects_rect(p0, p1, rect):
                return True
        return False
    
    def _spawn_victims_in_annulus(self, min_dist: float, max_dist: float):
        """Spawn victims in an annulus around the base (not too close, not too far)."""
        L = self.cfg.world_size
        
        victims = np.zeros((self.cfg.n_victims, 2), dtype=np.float32)
        for i in range(self.cfg.n_victims):
            for _ in range(100):  # Max attempts
                # Sample in annulus
                angle = self.rng.uniform(0, 2 * np.pi)
                r = np.sqrt(self.rng.uniform(min_dist**2, max_dist**2))
                pos = self.base_pos + np.array([np.cos(angle), np.sin(angle)]) * r
                # Check bounds
                if 0 <= pos[0] <= L and 0 <= pos[1] <= L:
                    if self.obstacle_count and self._point_in_any_obstacle(pos):
                        continue
                    victims[i] = pos
                    break
            else:
                # Fallback to random position
                for _ in range(100):
                    pos = self.rng.uniform(0, L, size=2)
                    if not self.obstacle_count or not self._point_in_any_obstacle(pos):
                        victims[i] = pos
                        break
                else:
                    victims[i] = self.rng.uniform(0, L, size=2)
        return victims

    def _init_obstacles(self):
        """Initialize obstacles from config or generate random rectangles."""
        if self.cfg.obstacles:
            rects = []
            for entry in self.cfg.obstacles:
                if isinstance(entry, dict):
                    x = float(entry.get("x", 0.0))
                    y = float(entry.get("y", 0.0))
                    w = float(entry.get("w", 0.0))
                    h = float(entry.get("h", 0.0))
                else:
                    x, y, w, h = entry
                rects.append((x, y, x + w, y + h))
            self.obstacles = np.array(rects, dtype=np.float32)
            self.obstacle_count = self.obstacles.shape[0]
            return

        if self.cfg.obstacle_count <= 0:
            self.obstacles = np.zeros((0, 4), dtype=np.float32)
            self.obstacle_count = 0
            return

        self.obstacles = self._generate_obstacles(self.cfg.obstacle_count)
        self.obstacle_count = self.obstacles.shape[0]

    def _generate_obstacles(self, count: int) -> np.ndarray:
        """Generate axis-aligned rectangle obstacles."""
        L = self.cfg.world_size
        min_size = self.cfg.obstacle_min_size
        max_size = self.cfg.obstacle_max_size
        margin = self.cfg.obstacle_margin
        separation = self.cfg.obstacle_min_separation
        base_clearance = self.cfg.obstacle_base_clearance
        if base_clearance <= 0.0 and self.cfg.spawn_near_base:
            base_clearance = (self.cfg.spawn_radius or self.cfg.r_comm) + margin

        rects: list[tuple[float, float, float, float]] = []
        attempts = 0
        max_attempts = max(100, count * 200)
        while len(rects) < count and attempts < max_attempts:
            attempts += 1
            w = float(self.rng.uniform(min_size, max_size))
            h = float(self.rng.uniform(min_size, max_size))
            if w <= 0.0 or h <= 0.0:
                continue
            x = float(self.rng.uniform(margin, max(margin, L - margin - w)))
            y = float(self.rng.uniform(margin, max(margin, L - margin - h)))
            rect = np.array([x, y, x + w, y + h], dtype=np.float32)

            if base_clearance > 0.0:
                if self._distance_point_to_rect(self.base_pos, rect) < base_clearance:
                    continue

            if separation > 0.0:
                overlap = False
                for existing in rects:
                    ex = np.array(existing, dtype=np.float32)
                    if not (
                        rect[2] + separation < ex[0]
                        or rect[0] - separation > ex[2]
                        or rect[3] + separation < ex[1]
                        or rect[1] - separation > ex[3]
                    ):
                        overlap = True
                        break
                if overlap:
                    continue

            rects.append((float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])))

        if not rects:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(rects, dtype=np.float32)
    
    def _update_explored_cells_per_drone(self):
        """Track which grid cells have been explored by each drone. Returns new cells per drone."""
        cell_size = 10.0  # 10x10 grid cells
        new_cells = np.zeros(self.cfg.n_drones, dtype=np.float32)
        
        for d, pos in enumerate(self.positions):
            cell = (int(pos[0] / cell_size), int(pos[1] / cell_size))
            # Global exploration
            if cell not in self.explored_cells:
                self.explored_cells.add(cell)
                new_cells[d] += 1.0
            # Per-drone exploration (encourages individual exploration)
            if cell not in self.drone_explored_cells[d]:
                self.drone_explored_cells[d].add(cell)
                new_cells[d] += 0.5  # Bonus for personal new territory
        
        return new_cells
    
    def _compute_relay_rewards(self):
        """Reward drones that serve as relay nodes for disconnected confirmed-victim owners.
        
        For each confirmed victim with a disconnected owner:
        1. Find all drones on the shortest path from owner to base (relay nodes)
        2. Give r_relay_bonus to each relay drone (not the owner)
        
        Also compute chain progress: how many hops is the owner from a connected drone?
        """
        relay_rewards = np.zeros(self.cfg.n_drones, dtype=np.float32)
        
        # Find confirmed victims with disconnected owners
        confirmed_mask = self.victim_status == 1
        if not confirmed_mask.any():
            return relay_rewards, 0.0
        
        # Build adjacency matrix
        N = self.cfg.n_drones
        pos = self.positions
        diff = pos[:, None, :] - pos[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        adj = dist_matrix <= self.cfg.r_comm
        np.fill_diagonal(adj, False)
        
        # Distance to base for each drone
        dist_base = np.linalg.norm(pos - self.base_pos[None, :], axis=1)
        base_links = dist_base <= self.cfg.r_comm
        
        total_chain_progress = 0.0
        
        for v in np.where(confirmed_mask)[0]:
            owner = self.deliver_owner[v]
            if owner < 0 or self.connected[owner]:
                continue  # Owner connected, no relay needed
            
            # BFS from owner to find path to any connected drone or base
            visited = np.zeros(N, dtype=bool)
            parent = -np.ones(N, dtype=np.int32)
            queue = [owner]
            visited[owner] = True
            found_connected = -1
            
            while queue and found_connected < 0:
                current = queue.pop(0)
                
                # Check if this drone can reach base directly
                if base_links[current]:
                    found_connected = current
                    break
                
                # Check if connected to a connected drone
                for neighbor in range(N):
                    if adj[current, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        parent[neighbor] = current
                        queue.append(neighbor)
                        if self.connected[neighbor]:
                            found_connected = neighbor
                            break
            
            if found_connected >= 0 and found_connected != owner:
                # Trace back path and reward relay drones
                node = found_connected
                path_length = 0
                while node != owner and node >= 0:
                    if node != owner:
                        relay_rewards[node] += self.cfg.r_relay_bonus
                    path_length += 1
                    node = parent[node]
                
                # Chain progress: shorter path = better (invert for potential)
                # Normalize by max possible hops
                max_hops = N
                progress = 1.0 - (path_length / max_hops)
                total_chain_progress += progress * self.cfg.r_chain_progress
            else:
                # No path exists - compute distance to nearest connected drone
                if self.connected.any():
                    connected_positions = pos[self.connected]
                    owner_pos = pos[owner]
                    dists_to_connected = np.linalg.norm(connected_positions - owner_pos, axis=1)
                    min_dist = np.min(dists_to_connected)
                    # Potential based on distance (closer = better)
                    # Normalize by world size
                    potential = 1.0 - min(min_dist / self.cfg.world_size, 1.0)
                    total_chain_progress += potential * self.cfg.r_chain_progress * 0.5
        
        return relay_rewards, total_chain_progress
    
    def _compute_dispersion_rewards(self):
        """Reward drones for maintaining good separation from each other."""
        rewards = np.zeros(self.cfg.n_drones, dtype=np.float32)
        min_sep = self.cfg.min_drone_separation
        
        for i in range(self.cfg.n_drones):
            min_dist_to_other = float('inf')
            for j in range(self.cfg.n_drones):
                if i != j:
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
                    min_dist_to_other = min(min_dist_to_other, dist)
            
            # Reward if well-separated, penalty if too close
            if min_dist_to_other >= min_sep:
                rewards[i] = self.cfg.r_dispersion
            else:
                # Penalty proportional to how close they are
                rewards[i] = -self.cfg.r_dispersion * (1.0 - min_dist_to_other / min_sep)
        
        return rewards

    def _update_confirm_and_delivery(self, scan_mask):
        new_delivered = 0
        new_confirmed = 0
        new_expired = 0
        confirming_drone = -1
        if self.cfg.n_victims == 0:
            return 0, 0, 0, -1

        scan_idx = np.where(scan_mask)[0]
        if scan_idx.size == 0:
            unconfirmed = self.victim_status == 0
            self.confirm_progress[unconfirmed] = 0
            self.confirm_owner[unconfirmed] = -1
        else:
            scan_pos = self.positions[scan_idx]
            diff = self.victim_pos[:, None, :] - scan_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            within = dist <= self.cfg.r_confirm_radius

            for v in range(self.cfg.n_victims):
                if self.victim_status[v] == 2:
                    continue

                if self.victim_status[v] == 1:
                    continue

                if within[v].any():
                    nearest = int(scan_idx[np.argmin(dist[v])])
                    if self.confirm_owner[v] == nearest:
                        self.confirm_progress[v] += 1
                    else:
                        self.confirm_owner[v] = nearest
                        self.confirm_progress[v] = 1

                    if self.confirm_progress[v] >= self.cfg.t_confirm:
                        self.victim_status[v] = 1  # confirmed
                        self.deliver_owner[v] = self.confirm_owner[v]
                        self.delivery_ttl[v] = self.cfg.m_deliver
                        new_confirmed += 1
                        confirming_drone = nearest  # Track who confirmed
                else:
                    self.confirm_progress[v] = 0
                    self.confirm_owner[v] = -1

        # Delivery window check
        for v in range(self.cfg.n_victims):
            if self.victim_status[v] != 1:
                continue

            owner = self.deliver_owner[v]
            if owner >= 0 and self.connected[owner]:
                self.victim_status[v] = 2
                self.delivery_ttl[v] = -1
                self.deliver_owner[v] = -1
                new_delivered += 1
                continue

            if self.delivery_ttl[v] > 0:
                self.delivery_ttl[v] -= 1
            elif self.delivery_ttl[v] == 0:
                self.delivery_ttl[v] = -1
                self.victim_status[v] = 0
                self.confirm_progress[v] = 0
                self.confirm_owner[v] = -1
                self.deliver_owner[v] = -1

        return new_delivered, new_confirmed, new_expired, confirming_drone

    def _get_obs(self):
        """Observation per drone: position, battery, connectivity, and scan detections."""
        L = self.cfg.world_size
        pos_norm = self.positions / max(L, 1e-6)
        dist_to_base = (
            np.linalg.norm(self.positions - self.base_pos[None, :], axis=1) / max(L, 1e-6)
        )
        last_comm_age_norm = self.last_comm_age / max(self.cfg.max_steps, 1)
        connected_flag = self.connected.astype(np.float32)

        # Neighbor features
        if self.cfg.n_drones > 1:
            diff = self.positions[:, None, :] - self.positions[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            neighbor_count = (dist <= self.cfg.r_comm).sum(axis=1).astype(np.float32)
            neighbor_count_norm = neighbor_count / max(self.cfg.n_drones - 1, 1)
            min_neighbor_dist = np.min(dist, axis=1)
            min_neighbor_dist_norm = np.clip(
                min_neighbor_dist / max(self.cfg.r_comm, 1e-6), 0.0, 1.0
            ).astype(np.float32)
        else:
            neighbor_count_norm = np.zeros(self.cfg.n_drones, dtype=np.float32)
            min_neighbor_dist_norm = np.ones(self.cfg.n_drones, dtype=np.float32)
        
        # Direction to base (normalized)
        to_base = self.base_pos[None, :] - self.positions
        to_base_norm = to_base / (np.linalg.norm(to_base, axis=1, keepdims=True) + 1e-8)
        
        detection_features = self.detections.reshape(self.cfg.n_drones, -1)
        obstacle_features = self._compute_obstacle_features().reshape(self.cfg.n_drones, -1)

        obs = np.concatenate([
            pos_norm,  # (n_drones, 2)
            self.battery[:, None],  # (n_drones, 1)
            dist_to_base[:, None],  # (n_drones, 1)
            connected_flag[:, None],  # (n_drones, 1)
            last_comm_age_norm[:, None],  # (n_drones, 1)
            neighbor_count_norm[:, None],  # (n_drones, 1)
            min_neighbor_dist_norm[:, None],  # (n_drones, 1)
            to_base_norm,  # (n_drones, 2)
            detection_features,  # (n_drones, obs_n_nearest * 3)
            obstacle_features,   # (n_drones, obs_n_obstacles * 3)
        ], axis=1).astype(np.float32)
        
        return obs

    def _compute_detections(self, scan_mask: np.ndarray) -> np.ndarray:
        """Compute noisy local detections for scanning drones."""
        n = self.cfg.n_drones
        k = self.cfg.obs_n_nearest
        detections = np.zeros((n, k, 3), dtype=np.float32)
        if k == 0:
            return detections

        r_sense = float(self.cfg.r_sense)
        if r_sense <= 0.0:
            return detections
        inv_L = 1.0 / max(self.cfg.world_size, 1e-6)
        detect_scale = float(self.cfg.detect_prob_scale)
        noise_std = float(self.cfg.detect_noise_std)
        fp_rate = float(self.cfg.false_positive_rate)
        fp_conf = float(self.cfg.false_positive_confidence)

        for d in range(n):
            if not scan_mask[d]:
                continue
            drone_pos = self.positions[d]
            candidates = []
            for vid in range(self.cfg.n_victims):
                if self.victim_status[vid] >= 2:
                    continue
                dx = self.victim_pos[vid, 0] - drone_pos[0]
                dy = self.victim_pos[vid, 1] - drone_pos[1]
                dist = float(np.hypot(dx, dy))
                if dist > r_sense:
                    continue
                if self.cfg.obstacle_blocks_sensing and self.obstacle_count:
                    if self._segment_intersects_any_obstacle(drone_pos, self.victim_pos[vid]):
                        continue
                p = max(0.0, 1.0 - dist / r_sense) * detect_scale
                if p <= 0.0:
                    continue
                if p > 1.0:
                    p = 1.0
                if self.rng.uniform(0.0, 1.0) > p:
                    continue
                if noise_std > 0.0:
                    dx += self.rng.normal(0.0, noise_std)
                    dy += self.rng.normal(0.0, noise_std)
                candidates.append((dist, dx, dy, p))

            if fp_rate > 0.0 and self.rng.uniform(0.0, 1.0) < fp_rate:
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                r = np.sqrt(self.rng.uniform(0.0, 1.0)) * r_sense
                dx = np.cos(angle) * r
                dy = np.sin(angle) * r
                if noise_std > 0.0:
                    dx += self.rng.normal(0.0, noise_std)
                    dy += self.rng.normal(0.0, noise_std)
                conf = self.rng.uniform(0.0, fp_conf) if fp_conf > 0.0 else 0.0
                candidates.append((r, dx, dy, conf))

            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0])
            for i, (_, dx, dy, conf) in enumerate(candidates[:k]):
                detections[d, i, 0] = np.clip(dx * inv_L, -1.0, 1.0)
                detections[d, i, 1] = np.clip(dy * inv_L, -1.0, 1.0)
                detections[d, i, 2] = np.clip(conf, 0.0, 1.0)

        return detections

    def _compute_obstacle_features(self) -> np.ndarray:
        """Return nearest obstacle features per drone (dx, dy, dist_norm)."""
        n = self.cfg.n_drones
        k = self.cfg.obs_n_obstacles
        feats = np.zeros((n, k, 3), dtype=np.float32)
        if k == 0 or self.obstacle_count == 0:
            return feats
        inv_L = 1.0 / max(self.cfg.world_size, 1e-6)

        for d in range(n):
            best = []
            px, py = self.positions[d]
            for rect in self.obstacles:
                cx = np.clip(px, rect[0], rect[2])
                cy = np.clip(py, rect[1], rect[3])
                dx = cx - px
                dy = cy - py
                dist = float(np.hypot(dx, dy))
                best.append((dist, dx, dy))
            best.sort(key=lambda item: item[0])
            for i, (dist, dx, dy) in enumerate(best[:k]):
                feats[d, i, 0] = np.clip(dx * inv_L, -1.0, 1.0)
                feats[d, i, 1] = np.clip(dy * inv_L, -1.0, 1.0)
                feats[d, i, 2] = np.clip(dist * inv_L, 0.0, 1.0)
        return feats


if __name__ == "__main__":
    env = DroneSwarmEnv()
    obs = env.reset(seed=0)
    for _ in range(5):
        actions = env.rng.uniform(-1.0, 1.0, size=(env.cfg.n_drones, 3))
        actions[:, 2] = (actions[:, 2] > 0).astype(np.float32)
        obs, rewards, done, info = env.step(actions)
        print(info)
        if done:
            break
