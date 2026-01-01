#ifndef DRONE_SWARM_H
#define DRONE_SWARM_H

#include <stdbool.h>
#include <stdint.h>

// Fixed-size limits for fast stack allocation
enum {
    MAX_DRONES = 64,
    MAX_VICTIMS = 128,
    MAX_NEAREST = 8,
    MAX_GRID = 64,
    MAX_M_DELIVER_VALUES = 8,
    MAX_T_CONFIRM_VALUES = 8
};

typedef struct {
    float world_size;
    int n_drones;
    int n_victims;
    float r_comm;
    float r_comm_min;
    float r_comm_max;
    float r_confirm_radius;
    float r_sense;
    int t_confirm;
    int t_confirm_values_count;
    int t_confirm_values[MAX_T_CONFIRM_VALUES];
    int m_deliver;
    int m_deliver_values_count;
    int m_deliver_values[MAX_M_DELIVER_VALUES];
    float v_max;
    float dt;
    // Battery costs
    float c_idle;
    float c_move;
    float c_scan;
    // Reward shaping
    float c_time;
    float c_energy;
    float r_found;
    int r_found_divide_by_n;
    float r_confirm_reward;
    float r_approach;
    float r_explore;
    float r_scan_near_victim;
    float r_connectivity;
    float r_dispersion;
    float min_drone_separation;
    float r_owner_connected;
    float p_comm_drop;
    float p_comm_drop_min;
    float p_comm_drop_max;
    int max_steps;
    float base_pos[2];
    int spawn_near_base;
    float spawn_radius;
    int obs_n_nearest;
    float victim_min_dist_from_base;
    float victim_max_dist_from_base;
    float victim_mix_prob;
    float victim_min_dist_from_base_alt;
    float victim_max_dist_from_base_alt;
} DroneSwarmConfig;

typedef struct {
    // Drone state
    float positions[MAX_DRONES][2];
    float battery[MAX_DRONES];
    bool connected[MAX_DRONES];
    float last_comm_age[MAX_DRONES];

    // Victim state
    float victim_pos[MAX_VICTIMS][2];
    int victim_status[MAX_VICTIMS];
    int confirm_progress[MAX_VICTIMS];
    int confirm_owner[MAX_VICTIMS];
    int deliver_owner[MAX_VICTIMS];
    int delivery_ttl[MAX_VICTIMS];

    // Environment state
    int step_count;
    uint64_t rng_state;

    // Stats
    int delivered_count;
    int confirmed_count;
    int connected_count;
    int explored_global_count;
    int last_new_delivered;
    int last_new_confirmed;

    // Exploration tracking
    int grid_w;
    bool explored_global[MAX_GRID * MAX_GRID];
    bool explored_per_drone[MAX_DRONES][MAX_GRID * MAX_GRID];

    // Buffers
    float observations[MAX_DRONES * (10 + 3 * MAX_NEAREST)];
    float rewards[MAX_DRONES];
    bool done;

    // Config
    DroneSwarmConfig cfg;
} DroneSwarm;

void drone_swarm_init(DroneSwarm* env, const DroneSwarmConfig* cfg);
void drone_swarm_seed(DroneSwarm* env, uint64_t seed);
void drone_swarm_reset(DroneSwarm* env);
void drone_swarm_step(DroneSwarm* env, const float* actions);

#endif
