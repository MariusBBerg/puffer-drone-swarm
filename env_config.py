"""Environment configuration for DroneSwarm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


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
