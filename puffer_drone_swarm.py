"""Native PufferLib environment wrapper for the DroneSwarm core."""

from __future__ import annotations

from typing import Optional
import numpy as np
import gymnasium
import pufferlib

from env import EnvConfig

try:
    from drone_swarm import binding
except Exception as exc:
    binding = None
    _BINDING_ERROR = exc


class PufferDroneSwarm(pufferlib.PufferEnv):
    """PufferLib-native multi-agent drone swarm environment.
    
    Each drone is an agent with its own observation and action.
    Actions are continuous: (vx, vy, scan) where vx/vy in [-1,1] and scan > 0 means scanning.
    """
    
    def __init__(self, config: EnvConfig = None, buf=None, seed: int = 0,
                 log_interval: int = 32, num_envs: int = 1, render_mode: str = None):
        # Handle config - allow passing as dict or EnvConfig
        if config is None:
            config = EnvConfig()
        elif isinstance(config, dict):
            config = EnvConfig.from_dict(config)
        
        self.cfg = config
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.tick = 0
        
        # Observation size: 2 (pos) + 1 (battery) + 1 (dist_base) + 1 (connected) + 1 (comm_age)
        # + 1 (neighbor_count) + 1 (min_neighbor_dist) + 2 (to_base_dir)
        # + 3*n_nearest (detections) + 3*n_obstacles (nearest obstacles)
        obs_size = 10 + 3 * config.obs_n_nearest + 3 * config.obs_n_obstacles
        
        # Define spaces BEFORE calling super().__init__
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),  # [vx, vy, scan]
            dtype=np.float32,
        )
        self.num_agents = self.cfg.n_drones * max(1, int(num_envs))
        
        # Initialize PufferEnv (sets up buffers)
        super().__init__(buf)
        
        if binding is None:
            raise ImportError(
                "PufferLib C binding (drone_swarm.binding) is required. Build it with "
                "`uv run python setup.py build_ext --inplace`."
            ) from _BINDING_ERROR

        self._seed = seed
        self.num_envs = max(1, int(num_envs))

        cfg_kwargs = _config_kwargs(self.cfg)

        c_envs = []
        offset = 0
        for i in range(self.num_envs):
            obs_slice = self.observations[offset:offset + self.cfg.n_drones]
            act_slice = self.actions[offset:offset + self.cfg.n_drones]
            rew_slice = self.rewards[offset:offset + self.cfg.n_drones]
            term_slice = self.terminals[offset:offset + self.cfg.n_drones]
            trunc_slice = self.truncations[offset:offset + self.cfg.n_drones]
            env_seed = i + seed * self.num_envs
            env_id = binding.env_init(
                obs_slice,
                act_slice,
                rew_slice,
                term_slice,
                trunc_slice,
                env_seed,
                **cfg_kwargs,
            )
            c_envs.append(env_id)
            offset += self.cfg.n_drones

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed: Optional[int] = None):
        if seed is None:
            seed = self._seed or 0
        self._seed = None
        self.tick = 0
        binding.vec_reset(self.c_envs, int(seed))
        return self.observations, []

    def step(self, actions):
        action_arr = np.asarray(actions, dtype=np.float32)
        if action_arr.ndim == 1:
            action_arr = action_arr.reshape(self.num_agents, 3)
        self.actions[:] = action_arr
        self.tick += 1
        binding.vec_step(self.c_envs)

        infos = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log.get("n", 0) > 0:
                infos.append(log)
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def render(self):
        if self.render_mode is None:
            return None
        binding.vec_render(self.c_envs, 0)
        return None

    def close(self):
        if binding is not None and hasattr(self, "c_envs"):
            binding.vec_close(self.c_envs)


def env_creator(name='drone_swarm'):
    """Factory function for PufferLib compatibility."""
    import functools
    return functools.partial(PufferDroneSwarm)


def _config_kwargs(cfg: EnvConfig) -> dict:
    base_pos = cfg.base_pos
    if base_pos is None:
        base_pos = (cfg.world_size * 0.5, cfg.world_size * 0.5)
    spawn_radius = cfg.spawn_radius if cfg.spawn_radius is not None else 0.0
    payload = {
        "world_size": cfg.world_size,
        "n_drones": cfg.n_drones,
        "n_victims": cfg.n_victims,
        "r_comm": cfg.r_comm,
        "r_comm_min": cfg.r_comm_min,
        "r_comm_max": cfg.r_comm_max,
        "r_confirm_radius": cfg.r_confirm_radius,
        "r_sense": cfg.r_sense,
        "t_confirm": cfg.t_confirm,
        "t_confirm_values": cfg.t_confirm_values,
        "m_deliver": cfg.m_deliver,
        "m_deliver_values": cfg.m_deliver_values,
        "v_max": cfg.v_max,
        "dt": cfg.dt,
        "c_idle": cfg.c_idle,
        "c_move": cfg.c_move,
        "c_scan": cfg.c_scan,
        "c_time": cfg.c_time,
        "c_energy": cfg.c_energy,
        "r_found": cfg.r_found,
        "r_found_divide_by_n": int(cfg.r_found_divide_by_n),
        "r_confirm_reward": cfg.r_confirm_reward,
        "r_approach": cfg.r_approach,
        "r_explore": cfg.r_explore,
        "r_scan_near_victim": cfg.r_scan_near_victim,
        "r_connectivity": cfg.r_connectivity,
        "r_dispersion": cfg.r_dispersion,
        "min_drone_separation": cfg.min_drone_separation,
        "r_owner_connected": cfg.r_owner_connected,
        "r_relay_bonus": cfg.r_relay_bonus,
        "r_chain_progress": cfg.r_chain_progress,
        "detect_prob_scale": cfg.detect_prob_scale,
        "detect_noise_std": cfg.detect_noise_std,
        "false_positive_rate": cfg.false_positive_rate,
        "false_positive_confidence": cfg.false_positive_confidence,
        "p_comm_drop": cfg.p_comm_drop,
        "p_comm_drop_min": cfg.p_comm_drop_min,
        "p_comm_drop_max": cfg.p_comm_drop_max,
        "max_steps": cfg.max_steps,
        "base_pos": base_pos,
        "spawn_near_base": int(cfg.spawn_near_base),
        "spawn_radius": float(spawn_radius),
        "obs_n_nearest": cfg.obs_n_nearest,
        "obs_n_obstacles": cfg.obs_n_obstacles,
        "victim_min_dist_from_base": cfg.victim_min_dist_from_base,
        "victim_max_dist_from_base": cfg.victim_max_dist_from_base,
        "victim_mix_prob": cfg.victim_mix_prob,
        "victim_min_dist_from_base_alt": cfg.victim_min_dist_from_base_alt,
        "victim_max_dist_from_base_alt": cfg.victim_max_dist_from_base_alt,
        "obstacle_count": cfg.obstacle_count,
        "obstacle_min_size": cfg.obstacle_min_size,
        "obstacle_max_size": cfg.obstacle_max_size,
        "obstacle_margin": cfg.obstacle_margin,
        "obstacle_base_clearance": cfg.obstacle_base_clearance,
        "obstacle_min_separation": cfg.obstacle_min_separation,
        "obstacle_blocks_sensing": int(cfg.obstacle_blocks_sensing),
        "obstacles": cfg.obstacles if cfg.obstacles else None,
    }
    return payload


if __name__ == "__main__":
    print("Testing PufferDroneSwarm environment...")
    
    # Test basic functionality
    env = PufferDroneSwarm()
    print(f"Observation space: {env.single_observation_space}")
    print(f"Action space: {env.single_action_space}")
    print(f"Num agents: {env.num_agents}")
    
    obs, infos = env.reset(seed=0)
    print(f"\nAfter reset:")
    print(f"  obs shape: {obs.shape}")
    print(f"  obs sample: {obs[0]}")
    
    # Run a few steps
    total_reward = 0
    for i in range(10):
        actions = env.action_space.sample()
        obs, rewards, terminals, truncations, infos = env.step(actions)
        total_reward += rewards.sum()
        if infos:
            print(f"Step {i+1} info: {infos}")
    
    print(f"\nAfter 10 steps:")
    print(f"  obs shape: {obs.shape}")
    print(f"  rewards sample: {rewards[:4]}")
    print(f"  total_reward: {total_reward}")
    print(f"  terminals: {terminals.any()}")
    print(f"  truncations: {truncations.any()}")
    
    # Test vectorization
    print("\n--- Testing vectorization ---")
    import pufferlib.vector
    
    vecenv = pufferlib.vector.make(
        PufferDroneSwarm,
        num_envs=2,
        backend=pufferlib.vector.Serial,
    )
    obs, infos = vecenv.reset()
    print(f"VecEnv obs shape: {obs.shape}")
    
    actions = vecenv.action_space.sample()
    obs, rewards, terminals, truncations, infos = vecenv.step(actions)
    print(f"VecEnv step obs shape: {obs.shape}")
    print(f"VecEnv rewards shape: {rewards.shape}")
    
    vecenv.close()
    print("\nAll tests passed!")
