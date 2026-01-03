#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "drone_swarm.h"
#include "drone_swarm_render.h"

static void set_default_config(DroneSwarmConfig *cfg) {
    cfg->world_size = 120.0f;
    cfg->n_drones = 8;
    cfg->n_victims = 8;
    cfg->r_comm = 15.0f;
    cfg->r_comm_min = 0.0f;
    cfg->r_comm_max = 0.0f;
    cfg->r_confirm_radius = 5.0f;
    cfg->r_sense = 30.0f;
    cfg->t_confirm = 2;
    cfg->t_confirm_values_count = 0;
    cfg->m_deliver = 120;
    cfg->m_deliver_values_count = 0;
    cfg->v_max = 2.5f;
    cfg->dt = 1.0f;
    cfg->c_idle = 0.0003f;
    cfg->c_move = 0.0006f;
    cfg->c_scan = 0.0002f;
    cfg->c_time = 0.005f;
    cfg->c_energy = 0.0f;
    cfg->r_found = 3.0f;
    cfg->r_found_divide_by_n = 1;
    cfg->r_confirm_reward = 0.4f;
    cfg->r_approach = 0.0f;
    cfg->r_explore = 0.02f;
    cfg->r_scan_near_victim = 0.02f;
    cfg->r_connectivity = 0.0f;
    cfg->r_dispersion = 0.02f;
    cfg->min_drone_separation = 8.0f;
    cfg->r_owner_connected = 0.05f;
    cfg->r_relay_bonus = 0.06f;
    cfg->r_chain_progress = 0.04f;
    cfg->detect_prob_scale = 1.8f;
    cfg->detect_noise_std = 0.06f;
    cfg->false_positive_rate = 0.0f;
    cfg->false_positive_confidence = 0.3f;
    cfg->p_comm_drop = 0.0f;
    cfg->p_comm_drop_min = 0.0f;
    cfg->p_comm_drop_max = 0.0f;
    cfg->max_steps = 1000;
    cfg->base_pos[0] = cfg->world_size / 2.0f;
    cfg->base_pos[1] = cfg->world_size / 2.0f;
    cfg->spawn_near_base = 1;
    cfg->spawn_radius = 10.0f;
    cfg->obs_n_nearest = 3;
    cfg->obs_n_obstacles = 3;
    cfg->victim_min_dist_from_base = 35.0f;
    cfg->victim_max_dist_from_base = 55.0f;
    cfg->victim_mix_prob = 0.0f;
    cfg->victim_min_dist_from_base_alt = 25.0f;
    cfg->victim_max_dist_from_base_alt = 45.0f;
    cfg->obstacle_count = 6;
    cfg->obstacle_min_size = 8.0f;
    cfg->obstacle_max_size = 25.0f;
    cfg->obstacle_margin = 5.0f;
    cfg->obstacle_base_clearance = 0.0f;
    cfg->obstacle_min_separation = 2.0f;
    cfg->obstacle_random = 1;
    cfg->obstacle_blocks_sensing = 1;
}

int main(void) {
    DroneSwarm env;
    DroneSwarmConfig cfg = {0};
    set_default_config(&cfg);
    drone_swarm_init(&env, &cfg);
    drone_swarm_seed(&env, 1);
    drone_swarm_reset(&env);

    if (!init_render_client(cfg.world_size)) {
        fprintf(stderr, "raylib: failed to initialize window (check GUI permissions).\n");
        return 1;
    }
    printf("Raylib window initialized. Close the window or press ESC to exit.\n");
    fflush(stdout);

    float actions[MAX_DRONES * 3];
    for (int i = 0; i < cfg.n_drones * 3; i++) {
        actions[i] = 0.0f;
    }

    if (WindowShouldClose()) {
        fprintf(stderr, "raylib: window closed immediately. Are you running headless?\n");
        return 0;
    }

    while (!WindowShouldClose()) {
        for (int i = 0; i < cfg.n_drones; i++) {
            float rx = (float)rand() / (float)RAND_MAX;
            float ry = (float)rand() / (float)RAND_MAX;
            float rs = (float)rand() / (float)RAND_MAX;
            actions[i * 3 + 0] = rx * 2.0f - 1.0f;
            actions[i * 3 + 1] = ry * 2.0f - 1.0f;
            actions[i * 3 + 2] = rs > 0.7f ? 1.0f : -1.0f;
        }
        drone_swarm_step(&env, actions);
        c_render(&env);
        if (env.done) {
            drone_swarm_reset(&env);
        }
    }
    return 0;
}
