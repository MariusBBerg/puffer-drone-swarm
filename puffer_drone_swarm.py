"""Native PufferLib environment wrapper for the DroneSwarm core."""

from __future__ import annotations

from typing import Optional
from collections import defaultdict

import numpy as np
import gymnasium
import pufferlib

from env import EnvConfig

try:
    from drone_swarm_c import CyDroneSwarm
except Exception as exc:
    CyDroneSwarm = None
    _C_BACKEND_ERROR = exc


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
        # + 1 (neighbor_count) + 1 (min_neighbor_dist) + 2 (to_base_dir) + 3*n_nearest (detections)
        obs_size = 10 + 3 * config.obs_n_nearest
        
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
        self.num_agents = self.cfg.n_drones
        
        # Initialize PufferEnv (sets up buffers)
        super().__init__(buf)
        
        # Create underlying environment (C backend required)
        if CyDroneSwarm is None:
            raise ImportError(
                "C backend (drone_swarm_c) is required. Build it with "
                "`uv run python setup.py build_ext --inplace`."
            ) from _C_BACKEND_ERROR
        self.env = CyDroneSwarm(self.cfg)
        self._using_c_backend = True
        self._seed = seed
        
        # Episode tracking for logging
        self._episode_rewards = np.zeros(self.num_agents, dtype=np.float32)
        self._episode_length = 0
        self._logs = defaultdict(list)

    def reset(self, seed: Optional[int] = None):
        if seed is None:
            seed = self._seed
        self._seed = None
        
        obs = self.env.reset(seed=seed)
        np.copyto(self.observations, obs)
        self.rewards.fill(0.0)
        self.terminals.fill(False)
        self.truncations.fill(False)
        self.masks.fill(True)
        
        # Reset episode tracking
        self._episode_rewards.fill(0.0)
        self._episode_length = 0
        self.tick = 0
        
        return self.observations, []

    def step(self, actions):
        # Actions come in as (num_agents, 3) from PufferLib
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_agents, 3)
        
        obs, rewards, done, info = self.env.step(actions)
        np.copyto(self.observations, obs)
        np.copyto(self.rewards, rewards)
        
        # Track episode stats
        self._episode_rewards += rewards
        self._episode_length += 1
        self.tick += 1
        
        # Check termination conditions
        delivered = self._get_delivered_count()
        confirmed_total = self._get_confirmed_total()
        success = bool(delivered == self.cfg.n_victims)
        truncated = bool(self._get_step_count() >= self.cfg.max_steps)
        terminated = success
        
        # Set terminal/truncation flags
        self.terminals[:] = terminated
        self.truncations[:] = truncated
        self.masks[:] = True
        
        infos = []
        
        # Episode ended - report stats and auto-reset
        if done:
            episode_info = {
                'episode_return': float(self._episode_rewards.mean()),
                'episode_length': self._episode_length,
                'delivered': delivered,
                'confirmed': confirmed_total,
                'success': success,
                'connected_fraction': self._get_connected_fraction(),
            }
            self._logs['episode_return'].append(episode_info['episode_return'])
            self._logs['episode_length'].append(episode_info['episode_length'])
            self._logs['delivered'].append(episode_info['delivered'])
            self._logs['success'].append(float(episode_info['success']))
            
            infos.append(episode_info)
            
            # Auto-reset for continuous training
            obs = self.env.reset(seed=None)
            np.copyto(self.observations, obs)
            self._episode_rewards.fill(0.0)
            self._episode_length = 0
        
        # Periodic logging
        elif self.tick % self.log_interval == 0 and self._logs['episode_return']:
            log_info = {
                'mean_episode_return': np.mean(self._logs['episode_return']),
                'mean_episode_length': np.mean(self._logs['episode_length']),
                'mean_delivered': np.mean(self._logs['delivered']),
                'mean_success': np.mean(self._logs['success']),
            }
            infos.append(log_info)
            self._logs.clear()
        
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def render(self):
        """Optional rendering - returns None for now."""
        return None

    def close(self):
        pass

    def _get_delivered_count(self) -> int:
        if hasattr(self.env, "get_delivered_count"):
            return int(self.env.get_delivered_count())
        return int((self.env.victim_status == 2).sum())

    def _get_confirmed_total(self) -> int:
        if hasattr(self.env, "get_confirmed_count") and hasattr(self.env, "get_delivered_count"):
            return int(self.env.get_confirmed_count() + self.env.get_delivered_count())
        return int((self.env.victim_status >= 1).sum())

    def _get_connected_fraction(self) -> float:
        if hasattr(self.env, "get_connected_fraction"):
            return float(self.env.get_connected_fraction())
        return float(np.mean(self.env.connected))

    def _get_step_count(self) -> int:
        if hasattr(self.env, "get_step_count"):
            return int(self.env.get_step_count())
        return int(self.env.step_count)


def env_creator(name='drone_swarm'):
    """Factory function for PufferLib compatibility."""
    import functools
    return functools.partial(PufferDroneSwarm)


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
