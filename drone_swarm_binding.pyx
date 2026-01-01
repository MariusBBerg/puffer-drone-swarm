# distutils: language=c
# distutils: sources = c_src/drone_swarm.c
# distutils: include_dirs = c_src
# cython: language_level=3

from libc.stdint cimport uint64_t
from libc.string cimport memcpy

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

DEF MAX_DRONES = 64
DEF MAX_VICTIMS = 128
DEF MAX_NEAREST = 8

cdef extern from "drone_swarm.h":
    ctypedef struct DroneSwarmConfig:
        float world_size
        int n_drones
        int n_victims
        float r_comm
        float r_comm_min
        float r_comm_max
        float r_confirm_radius
        float r_sense
        int t_confirm
        int m_deliver
        float v_max
        float dt
        float c_idle
        float c_move
        float c_scan
        float c_time
        float c_energy
        float r_found
        int r_found_divide_by_n
        float r_confirm_reward
        float r_approach
        float r_explore
        float r_scan_near_victim
        float r_connectivity
        float r_dispersion
        float min_drone_separation
        float r_owner_connected
        float p_comm_drop
        float p_comm_drop_min
        float p_comm_drop_max
        int max_steps
        float base_pos[2]
        int spawn_near_base
        float spawn_radius
        int obs_n_nearest
        float victim_min_dist_from_base
        float victim_max_dist_from_base
        float victim_mix_prob
        float victim_min_dist_from_base_alt
        float victim_max_dist_from_base_alt

    ctypedef struct DroneSwarm:
        float observations[MAX_DRONES * (10 + 3 * MAX_NEAREST)]
        float rewards[MAX_DRONES]
        int step_count
        int delivered_count
        int confirmed_count
        int connected_count
        int explored_global_count
        int last_new_delivered
        int last_new_confirmed
        bint done
        DroneSwarmConfig cfg

    void drone_swarm_init(DroneSwarm* env, const DroneSwarmConfig* cfg)
    void drone_swarm_seed(DroneSwarm* env, uint64_t seed)
    void drone_swarm_reset(DroneSwarm* env)
    void drone_swarm_step(DroneSwarm* env, const float* actions)


@cython.final
cdef class CyDroneSwarm:
    cdef DroneSwarm env
    cdef DroneSwarmConfig cfg
    cdef int obs_size
    cdef int n_drones

    def __init__(self, config=None):
        if config is None:
            from env import EnvConfig
            config = EnvConfig()
        elif isinstance(config, dict):
            from env import EnvConfig
            config = EnvConfig.from_dict(config)

        self._init_cfg(config)
        drone_swarm_init(&self.env, &self.cfg)
        self.n_drones = self.cfg.n_drones
        self.obs_size = 10 + 3 * self.cfg.obs_n_nearest

    cdef void _init_cfg(self, config):
        cdef float world_size = float(config.world_size)
        if world_size <= 0.0:
            raise ValueError("world_size must be positive")

        cdef int n_drones = int(config.n_drones)
        cdef int n_victims = int(config.n_victims)
        cdef int obs_n_nearest = int(config.obs_n_nearest)

        if n_drones < 0 or n_drones > MAX_DRONES:
            raise ValueError(f"n_drones must be in [0, {MAX_DRONES}]")
        if n_victims < 0 or n_victims > MAX_VICTIMS:
            raise ValueError(f"n_victims must be in [0, {MAX_VICTIMS}]")
        if obs_n_nearest < 0 or obs_n_nearest > MAX_NEAREST:
            raise ValueError(f"obs_n_nearest must be in [0, {MAX_NEAREST}]")

        base_pos = config.base_pos
        if base_pos is None:
            base_x = world_size / 2.0
            base_y = world_size / 2.0
        else:
            base_x = float(base_pos[0])
            base_y = float(base_pos[1])

        spawn_radius = config.spawn_radius
        if spawn_radius is None:
            spawn_radius = float(config.r_comm)
        else:
            spawn_radius = float(spawn_radius)

        self.cfg.world_size = world_size
        self.cfg.n_drones = n_drones
        self.cfg.n_victims = n_victims
        self.cfg.r_comm = float(config.r_comm)
        self.cfg.r_comm_min = float(config.r_comm_min)
        self.cfg.r_comm_max = float(config.r_comm_max)
        self.cfg.r_confirm_radius = float(config.r_confirm_radius)
        self.cfg.r_sense = float(config.r_sense)
        self.cfg.t_confirm = int(config.t_confirm)
        self.cfg.m_deliver = int(config.m_deliver)
        self.cfg.v_max = float(config.v_max)
        self.cfg.dt = float(config.dt)
        self.cfg.c_idle = float(config.c_idle)
        self.cfg.c_move = float(config.c_move)
        self.cfg.c_scan = float(config.c_scan)
        self.cfg.c_time = float(config.c_time)
        self.cfg.c_energy = float(config.c_energy)
        self.cfg.r_found = float(config.r_found)
        self.cfg.r_found_divide_by_n = 1 if bool(config.r_found_divide_by_n) else 0
        self.cfg.r_confirm_reward = float(config.r_confirm_reward)
        self.cfg.r_approach = float(config.r_approach)
        self.cfg.r_explore = float(config.r_explore)
        self.cfg.r_scan_near_victim = float(config.r_scan_near_victim)
        self.cfg.r_connectivity = float(config.r_connectivity)
        self.cfg.r_dispersion = float(config.r_dispersion)
        self.cfg.min_drone_separation = float(config.min_drone_separation)
        self.cfg.r_owner_connected = float(config.r_owner_connected)
        self.cfg.p_comm_drop = float(config.p_comm_drop)
        self.cfg.p_comm_drop_min = float(config.p_comm_drop_min)
        self.cfg.p_comm_drop_max = float(config.p_comm_drop_max)
        self.cfg.max_steps = int(config.max_steps)
        self.cfg.base_pos[0] = base_x
        self.cfg.base_pos[1] = base_y
        self.cfg.spawn_near_base = 1 if bool(config.spawn_near_base) else 0
        self.cfg.spawn_radius = spawn_radius
        self.cfg.obs_n_nearest = obs_n_nearest
        self.cfg.victim_min_dist_from_base = float(config.victim_min_dist_from_base)
        self.cfg.victim_max_dist_from_base = float(config.victim_max_dist_from_base)
        self.cfg.victim_mix_prob = float(config.victim_mix_prob)
        self.cfg.victim_min_dist_from_base_alt = float(config.victim_min_dist_from_base_alt)
        self.cfg.victim_max_dist_from_base_alt = float(config.victim_max_dist_from_base_alt)

    def reset(self, seed=None):
        if seed is not None:
            drone_swarm_seed(&self.env, <uint64_t>seed)
        drone_swarm_reset(&self.env)
        return self._get_obs()

    def step(self, actions):
        cdef np.ndarray[np.float32_t, ndim=2, mode="c"] actions_c
        actions_c = np.asarray(actions, dtype=np.float32)
        if actions_c.shape[0] != self.n_drones or actions_c.shape[1] != 3:
            raise ValueError(
                "actions must have shape (%d, 3), got (%d, %d)"
                % (self.n_drones, actions_c.shape[0], actions_c.shape[1])
            )
        if not actions_c.flags['C_CONTIGUOUS']:
            actions_c = np.ascontiguousarray(actions_c, dtype=np.float32)

        drone_swarm_step(&self.env, <float*>actions_c.data)
        obs = self._get_obs()
        rewards = self._get_rewards()

        info = {
            "step": int(self.env.step_count),
            "delivered": int(self.env.delivered_count),
            "confirmed": int(self.env.confirmed_count),
            "connected_fraction": float(self.connected_fraction),
            "new_delivered": int(self.env.last_new_delivered),
            "new_confirmed": int(self.env.last_new_confirmed),
            "explored_cells": int(self.env.explored_global_count),
        }

        return obs, rewards, bool(self.env.done), info

    @property
    def connected_fraction(self):
        if self.env.cfg.n_drones == 0:
            return 0.0
        return float(self.env.connected_count) / float(self.env.cfg.n_drones)

    def get_delivered_count(self):
        return int(self.env.delivered_count)

    def get_confirmed_count(self):
        return int(self.env.confirmed_count)

    def get_connected_fraction(self):
        return float(self.connected_fraction)

    def get_step_count(self):
        return int(self.env.step_count)

    def get_explored_cells(self):
        return int(self.env.explored_global_count)

    def get_new_delivered(self):
        return int(self.env.last_new_delivered)

    def get_new_confirmed(self):
        return int(self.env.last_new_confirmed)

    def get_obs_size(self):
        return int(self.obs_size)

    def get_num_drones(self):
        return int(self.n_drones)

    cdef np.ndarray _get_obs(self):
        cdef np.ndarray[np.float32_t, ndim=2] obs = np.empty(
            (self.n_drones, self.obs_size), dtype=np.float32
        )
        cdef Py_ssize_t count = self.n_drones * self.obs_size
        memcpy(obs.data, self.env.observations, count * sizeof(float))
        return obs

    cdef np.ndarray _get_rewards(self):
        cdef np.ndarray[np.float32_t, ndim=1] rewards = np.empty(
            self.n_drones, dtype=np.float32
        )
        cdef Py_ssize_t count = self.n_drones
        memcpy(rewards.data, self.env.rewards, count * sizeof(float))
        return rewards
