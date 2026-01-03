"""Environment presets for DroneSwarm.

Edit or extend these presets instead of passing CLI args.
"""

from __future__ import annotations

from env import EnvConfig


def _base_wilderness() -> EnvConfig:
    """Base wilderness SAR config (no obstacles)."""
    return EnvConfig(
        world_size=120.0,
        n_drones=10,
        n_victims=8,
        r_comm=15.0,
        r_comm_min=15.0,
        r_comm_max=15.0,
        p_comm_drop=0.0,
        p_comm_drop_min=0.0,
        p_comm_drop_max=0.02,
        r_sense=30.0,
        r_confirm_radius=5.0,
        detect_prob_scale=1.8,
        detect_noise_std=0.06,
        max_steps=1000,
        t_confirm=2,
        t_confirm_values=(),
        m_deliver=240,
        m_deliver_values=(),
        r_found=6.0,
        r_found_divide_by_n=False,
        r_confirm_reward=0.2,
        r_explore=0.01,
        r_scan_near_victim=0.02,
        r_dispersion=0.01,
        r_connectivity=0.0,
        r_owner_connected=0.15,
        r_relay_bonus=0.08,
        r_chain_progress=0.04,
        c_time=0.003,
        c_energy=0.0,
        c_scan=0.0003,
        victim_min_dist_from_base=35.0,
        victim_max_dist_from_base=55.0,
        victim_mix_prob=0.0,
        victim_min_dist_from_base_alt=25.0,
        victim_max_dist_from_base_alt=45.0,
        spawn_near_base=True,
        spawn_radius=12.0,
        obs_n_nearest=3,
        false_positive_rate=0.0,
        false_positive_confidence=0.3,
        obstacle_count=0,
        obs_n_obstacles=0,
    )


def _with_obstacles(cfg: EnvConfig, count: int, max_size: float, obs_n: int) -> EnvConfig:
    payload = cfg.__dict__.copy()
    payload.update(
        obstacle_count=count,
        obstacle_min_size=8.0,
        obstacle_max_size=max_size,
        obstacle_margin=5.0,
        obstacle_base_clearance=18.0,
        obstacle_min_separation=3.0,
        obstacle_blocks_sensing=True,
        obs_n_obstacles=obs_n,
    )
    return EnvConfig.from_dict(payload)


PRESETS = {
    "wilderness_hard": _base_wilderness(),
    "wilderness_hard_obstacles_stage1": _with_obstacles(_base_wilderness(), count=3, max_size=18.0, obs_n=2),
    "wilderness_hard_obstacles_stage2": _with_obstacles(_base_wilderness(), count=6, max_size=25.0, obs_n=3),
}


def get_env_config(name: str) -> EnvConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {', '.join(PRESETS)}")
    return PRESETS[name]
