#include "drone_swarm.h"

#include <math.h>
#include <string.h>

#define CELL_SIZE 10.0f
#define PI_F 3.14159265358979323846f

static inline uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    if (x == 0)
    {
        x = 88172645463325252ull;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline float rand_float(uint64_t *state)
{
    return (float)((xorshift64(state) >> 11) * (1.0 / 9007199254740992.0));
}

static inline float rand_uniform(uint64_t *state, float lo, float hi)
{
    return lo + (hi - lo) * rand_float(state);
}

static inline float rand_normal(uint64_t *state)
{
    float u1 = fmaxf(rand_float(state), 1e-6f);
    float u2 = rand_float(state);
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * PI_F * u2;
    return r * cosf(theta);
}

static inline float clampf(float x, float lo, float hi)
{
    if (x < lo)
        return lo;
    if (x > hi)
        return hi;
    return x;
}

static void compute_connectivity(DroneSwarm *env)
{
    int n = env->cfg.n_drones;
    float r_comm = env->cfg.r_comm;
    float r_comm_sq = r_comm * r_comm;
    float base_x = env->cfg.base_pos[0];
    float base_y = env->cfg.base_pos[1];
    float p_drop = env->cfg.p_comm_drop;

    for (int i = 0; i < n; i++)
    {
        env->connected[i] = false;
    }

    int queue[MAX_DRONES];
    int queue_start = 0;
    int queue_end = 0;

    bool adj[MAX_DRONES][MAX_DRONES];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            adj[i][j] = false;
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            float dx = env->positions[i][0] - env->positions[j][0];
            float dy = env->positions[i][1] - env->positions[j][1];
            if (dx * dx + dy * dy <= r_comm_sq)
            {
                if (p_drop > 0.0f && rand_float(&env->rng_state) < p_drop)
                {
                    continue;
                }
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        float dx = env->positions[i][0] - base_x;
        float dy = env->positions[i][1] - base_y;
        if (dx * dx + dy * dy <= r_comm_sq)
        {
            if (p_drop > 0.0f && rand_float(&env->rng_state) < p_drop)
            {
                continue;
            }
            env->connected[i] = true;
            queue[queue_end++] = i;
        }
    }

    while (queue_start < queue_end)
    {
        int i = queue[queue_start++];
        for (int j = 0; j < n; j++)
        {
            if (env->connected[j])
            {
                continue;
            }
            if (adj[i][j])
            {
                env->connected[j] = true;
                queue[queue_end++] = j;
            }
        }
    }

    env->connected_count = 0;
    for (int i = 0; i < n; i++)
    {
        if (env->connected[i])
        {
            env->last_comm_age[i] = 0.0f;
            env->connected_count++;
        }
        else
        {
            env->last_comm_age[i] += 1.0f;
        }
    }
}

static void update_explored_cells(DroneSwarm *env, float *out_new_cells)
{
    int n = env->cfg.n_drones;
    int grid_w = env->grid_w;

    if (env->cfg.r_explore == 0.0f)
    {
        for (int i = 0; i < n; i++)
        {
            out_new_cells[i] = 0.0f;
        }
        return;
    }

    for (int i = 0; i < n; i++)
    {
        out_new_cells[i] = 0.0f;
        int cell_x = (int)(env->positions[i][0] / CELL_SIZE);
        int cell_y = (int)(env->positions[i][1] / CELL_SIZE);
        if (cell_x < 0)
            cell_x = 0;
        if (cell_y < 0)
            cell_y = 0;
        if (cell_x >= grid_w)
            cell_x = grid_w - 1;
        if (cell_y >= grid_w)
            cell_y = grid_w - 1;
        int idx = cell_y * grid_w + cell_x;
        if (idx < 0)
            idx = 0;
        if (idx >= MAX_GRID * MAX_GRID)
            idx = MAX_GRID * MAX_GRID - 1;

        if (!env->explored_global[idx])
        {
            env->explored_global[idx] = true;
            env->explored_global_count += 1;
            out_new_cells[i] += 1.0f;
        }
        if (!env->explored_per_drone[i][idx])
        {
            env->explored_per_drone[i][idx] = true;
            out_new_cells[i] += 0.5f;
        }
    }
}

static void compute_scan_rewards(DroneSwarm *env, const bool *scan_mask, float *out_scan_rewards)
{
    int n = env->cfg.n_drones;
    int v = env->cfg.n_victims;
    float range = env->cfg.r_confirm_radius * 2.0f;
    float range_sq = range * range;

    if (env->cfg.r_scan_near_victim == 0.0f)
    {
        for (int i = 0; i < n; i++)
        {
            out_scan_rewards[i] = 0.0f;
        }
        return;
    }

    for (int i = 0; i < n; i++)
    {
        out_scan_rewards[i] = 0.0f;
    }

    bool any_scan = false;
    for (int i = 0; i < n; i++)
    {
        if (scan_mask[i])
        {
            any_scan = true;
            break;
        }
    }
    if (!any_scan)
    {
        return;
    }

    bool any_unconfirmed = false;
    for (int k = 0; k < v; k++)
    {
        if (env->victim_status[k] == 0)
        {
            any_unconfirmed = true;
            break;
        }
    }
    if (!any_unconfirmed)
    {
        return;
    }

    for (int d = 0; d < n; d++)
    {
        if (!scan_mask[d])
        {
            continue;
        }
        for (int k = 0; k < v; k++)
        {
            if (env->victim_status[k] != 0)
            {
                continue;
            }
            float dx = env->victim_pos[k][0] - env->positions[d][0];
            float dy = env->victim_pos[k][1] - env->positions[d][1];
            if (dx * dx + dy * dy <= range_sq)
            {
                out_scan_rewards[d] = env->cfg.r_scan_near_victim;
                break;
            }
        }
    }
}

static void compute_dispersion_rewards(DroneSwarm *env, float *out_dispersion_rewards)
{
    int n = env->cfg.n_drones;
    float min_sep = env->cfg.min_drone_separation;

    if (env->cfg.r_dispersion == 0.0f || n < 2)
    {
        for (int i = 0; i < n; i++)
        {
            out_dispersion_rewards[i] = 0.0f;
        }
        return;
    }

    for (int i = 0; i < n; i++)
    {
        float min_dist_sq = 1e18f;
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                continue;
            }
            float dx = env->positions[i][0] - env->positions[j][0];
            float dy = env->positions[i][1] - env->positions[j][1];
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
            }
        }
        float min_dist = sqrtf(min_dist_sq);
        if (min_dist >= min_sep)
        {
            out_dispersion_rewards[i] = env->cfg.r_dispersion;
        }
        else
        {
            out_dispersion_rewards[i] = -env->cfg.r_dispersion * (1.0f - min_dist / min_sep);
        }
    }
}

static void compute_relay_rewards(DroneSwarm *env, float *out_rewards, float *out_chain_progress)
{
    /* Reward drones that serve as relay nodes for disconnected confirmed-victim owners.
     * Also compute chain progress: how close is the owner to becoming connected?
     */
    int n = env->cfg.n_drones;
    int v_count = env->cfg.n_victims;
    float r_comm = env->cfg.r_comm;
    float r_comm_sq = r_comm * r_comm;
    float world_size = env->cfg.world_size;

    *out_chain_progress = 0.0f;
    for (int i = 0; i < n; i++)
    {
        out_rewards[i] = 0.0f;
    }

    if (env->cfg.r_relay_bonus == 0.0f && env->cfg.r_chain_progress == 0.0f)
    {
        return;
    }

    // Check if any confirmed victims exist
    bool any_confirmed = false;
    for (int v = 0; v < v_count; v++)
    {
        if (env->victim_status[v] == 1)
        {
            any_confirmed = true;
            break;
        }
    }
    if (!any_confirmed)
    {
        return;
    }

    // Build adjacency matrix
    bool adj[MAX_DRONES][MAX_DRONES];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            adj[i][j] = false;
        }
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            float dx = env->positions[i][0] - env->positions[j][0];
            float dy = env->positions[i][1] - env->positions[j][1];
            if (dx * dx + dy * dy <= r_comm_sq)
            {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
    }

    // Distance to base for each drone
    float base_x = env->cfg.base_pos[0];
    float base_y = env->cfg.base_pos[1];
    bool base_links[MAX_DRONES];
    for (int i = 0; i < n; i++)
    {
        float dx = env->positions[i][0] - base_x;
        float dy = env->positions[i][1] - base_y;
        base_links[i] = (dx * dx + dy * dy <= r_comm_sq);
    }

    // For each confirmed victim with disconnected owner
    for (int v = 0; v < v_count; v++)
    {
        if (env->victim_status[v] != 1)
        {
            continue;
        }
        int owner = env->deliver_owner[v];
        if (owner < 0 || owner >= n || env->connected[owner])
        {
            continue; // Owner connected, no relay needed
        }

        // BFS from owner to find path to any connected drone or base
        bool visited[MAX_DRONES];
        int parent[MAX_DRONES];
        for (int i = 0; i < n; i++)
        {
            visited[i] = false;
            parent[i] = -1;
        }

        int queue[MAX_DRONES];
        int queue_start = 0;
        int queue_end = 0;

        queue[queue_end++] = owner;
        visited[owner] = true;
        int found_connected = -1;

        while (queue_start < queue_end && found_connected < 0)
        {
            int current = queue[queue_start++];

            // Check if this drone can reach base directly
            if (base_links[current])
            {
                found_connected = current;
                break;
            }

            // Check neighbors
            for (int neighbor = 0; neighbor < n; neighbor++)
            {
                if (adj[current][neighbor] && !visited[neighbor])
                {
                    visited[neighbor] = true;
                    parent[neighbor] = current;
                    queue[queue_end++] = neighbor;
                    if (env->connected[neighbor])
                    {
                        found_connected = neighbor;
                        break;
                    }
                }
            }
        }

        if (found_connected >= 0 && found_connected != owner)
        {
            // Trace back path and reward relay drones
            int node = found_connected;
            int path_length = 0;
            while (node != owner && node >= 0)
            {
                if (node != owner)
                {
                    out_rewards[node] += env->cfg.r_relay_bonus;
                }
                path_length++;
                node = parent[node];
            }

            // Chain progress: shorter path = better
            float max_hops = (float)n;
            float progress = 1.0f - ((float)path_length / max_hops);
            *out_chain_progress += progress * env->cfg.r_chain_progress;
        }
        else
        {
            // No path exists - compute distance to nearest connected drone
            float min_dist = 1e18f;
            for (int i = 0; i < n; i++)
            {
                if (env->connected[i])
                {
                    float dx = env->positions[owner][0] - env->positions[i][0];
                    float dy = env->positions[owner][1] - env->positions[i][1];
                    float dist = sqrtf(dx * dx + dy * dy);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                    }
                }
            }
            if (min_dist < 1e17f)
            {
                // Potential based on distance (closer = better)
                float potential = 1.0f - fminf(min_dist / world_size, 1.0f);
                *out_chain_progress += potential * env->cfg.r_chain_progress * 0.5f;
            }
        }
    }
}

static void update_confirm_and_delivery(
    DroneSwarm *env,
    const bool *scan_mask,
    int *out_new_delivered,
    int *out_new_confirmed,
    int *out_confirming_drone)
{
    int n_victims = env->cfg.n_victims;
    int n_drones = env->cfg.n_drones;
    float r_confirm_sq = env->cfg.r_confirm_radius * env->cfg.r_confirm_radius;

    *out_new_delivered = 0;
    *out_new_confirmed = 0;
    *out_confirming_drone = -1;

    int scan_count = 0;
    for (int i = 0; i < n_drones; i++)
    {
        if (scan_mask[i])
        {
            scan_count++;
        }
    }

    if (scan_count == 0)
    {
        for (int v = 0; v < n_victims; v++)
        {
            if (env->victim_status[v] == 0)
            {
                env->confirm_progress[v] = 0;
                env->confirm_owner[v] = -1;
            }
        }
    }
    else
    {
        for (int v = 0; v < n_victims; v++)
        {
            if (env->victim_status[v] != 0)
            {
                continue;
            }

            float best_dist = 1e9f;
            int best_drone = -1;
            for (int d = 0; d < n_drones; d++)
            {
                if (!scan_mask[d])
                {
                    continue;
                }
                float dx = env->victim_pos[v][0] - env->positions[d][0];
                float dy = env->victim_pos[v][1] - env->positions[d][1];
                float dist_sq = dx * dx + dy * dy;
                if (dist_sq <= r_confirm_sq && dist_sq < best_dist)
                {
                    best_dist = dist_sq;
                    best_drone = d;
                }
            }

            if (best_drone >= 0)
            {
                if (env->confirm_owner[v] == best_drone)
                {
                    env->confirm_progress[v] += 1;
                }
                else
                {
                    env->confirm_owner[v] = best_drone;
                    env->confirm_progress[v] = 1;
                }

                if (env->confirm_progress[v] >= env->cfg.t_confirm)
                {
                    env->victim_status[v] = 1;
                    env->deliver_owner[v] = env->confirm_owner[v];
                    env->delivery_ttl[v] = env->cfg.m_deliver;
                    (*out_new_confirmed)++;
                    *out_confirming_drone = best_drone;
                }
            }
            else
            {
                env->confirm_progress[v] = 0;
                env->confirm_owner[v] = -1;
            }
        }
    }

    for (int v = 0; v < n_victims; v++)
    {
        if (env->victim_status[v] != 1)
        {
            continue;
        }

        int owner = env->deliver_owner[v];
        if (owner >= 0 && env->connected[owner])
        {
            env->victim_status[v] = 2;
            env->delivery_ttl[v] = -1;
            env->deliver_owner[v] = -1;
            (*out_new_delivered)++;
            continue;
        }

        if (env->delivery_ttl[v] > 0)
        {
            env->delivery_ttl[v] -= 1;
        }
        else if (env->delivery_ttl[v] == 0)
        {
            env->delivery_ttl[v] = -1;
            env->victim_status[v] = 0;
            env->confirm_progress[v] = 0;
            env->confirm_owner[v] = -1;
            env->deliver_owner[v] = -1;
        }
    }
}

static void compute_observations(DroneSwarm *env)
{
    int n = env->cfg.n_drones;
    int k = env->cfg.obs_n_nearest;
    int obs_size = 10 + 3 * k;
    float L = env->cfg.world_size;
    float inv_L = 1.0f / fmaxf(L, 1e-6f);
    float base_x = env->cfg.base_pos[0];
    float base_y = env->cfg.base_pos[1];
    float r_comm = env->cfg.r_comm;
    float r_comm_sq = r_comm * r_comm;

    for (int d = 0; d < n; d++)
    {
        float *out = env->observations + d * obs_size;
        float px = env->positions[d][0];
        float py = env->positions[d][1];

        float dx_base = base_x - px;
        float dy_base = base_y - py;
        float dist_base = sqrtf(dx_base * dx_base + dy_base * dy_base);
        float dist_base_norm = dist_base * inv_L;
        float to_base_norm_x = 0.0f;
        float to_base_norm_y = 0.0f;
        if (dist_base > 1e-8f)
        {
            to_base_norm_x = dx_base / dist_base;
            to_base_norm_y = dy_base / dist_base;
        }

        out[0] = px * inv_L;
        out[1] = py * inv_L;
        out[2] = env->battery[d];
        out[3] = dist_base_norm;
        out[4] = env->connected[d] ? 1.0f : 0.0f;
        out[5] = env->last_comm_age[d] / fmaxf((float)env->cfg.max_steps, 1.0f);
        float neighbor_count_norm = 0.0f;
        float min_neighbor_dist_norm = 1.0f;
        if (n > 1)
        {
            int neighbor_count = 0;
            float min_dist_sq = 1e18f;
            for (int j = 0; j < n; j++)
            {
                if (j == d)
                {
                    continue;
                }
                float dx = env->positions[j][0] - px;
                float dy = env->positions[j][1] - py;
                float dist_sq = dx * dx + dy * dy;
                if (dist_sq <= r_comm_sq)
                {
                    neighbor_count++;
                }
                if (dist_sq < min_dist_sq)
                {
                    min_dist_sq = dist_sq;
                }
            }
            neighbor_count_norm = (float)neighbor_count / fmaxf((float)(n - 1), 1.0f);
            float min_dist = sqrtf(min_dist_sq);
            min_neighbor_dist_norm = min_dist / fmaxf(r_comm, 1e-6f);
            if (min_neighbor_dist_norm > 1.0f)
            {
                min_neighbor_dist_norm = 1.0f;
            }
        }
        out[6] = neighbor_count_norm;
        out[7] = min_neighbor_dist_norm;
        out[8] = to_base_norm_x;
        out[9] = to_base_norm_y;

        for (int i = 0; i < k; i++)
        {
            out[10 + i * 3] = env->detections[d][i][0];
            out[10 + i * 3 + 1] = env->detections[d][i][1];
            out[10 + i * 3 + 2] = env->detections[d][i][2];
        }
    }
}

static void compute_detections(DroneSwarm *env, const bool *scan_mask)
{
    int n = env->cfg.n_drones;
    int v = env->cfg.n_victims;
    int k = env->cfg.obs_n_nearest;
    float r_sense = env->cfg.r_sense;
    float r_sense_sq = r_sense * r_sense;
    float inv_L = 1.0f / fmaxf(env->cfg.world_size, 1e-6f);
    float detect_scale = env->cfg.detect_prob_scale;
    float noise_std = env->cfg.detect_noise_std;
    float fp_rate = env->cfg.false_positive_rate;
    float fp_conf = env->cfg.false_positive_confidence;

    for (int d = 0; d < n; d++)
    {
        for (int i = 0; i < k; i++)
        {
            env->detections[d][i][0] = 0.0f;
            env->detections[d][i][1] = 0.0f;
            env->detections[d][i][2] = 0.0f;
        }
        if (!scan_mask[d] || k <= 0 || r_sense <= 0.0f)
        {
            continue;
        }

        float best_dist[MAX_NEAREST];
        float best_dx[MAX_NEAREST];
        float best_dy[MAX_NEAREST];
        float best_conf[MAX_NEAREST];
        for (int i = 0; i < k; i++)
        {
            best_dist[i] = 1e18f;
            best_dx[i] = 0.0f;
            best_dy[i] = 0.0f;
            best_conf[i] = 0.0f;
        }

        float px = env->positions[d][0];
        float py = env->positions[d][1];

        for (int vid = 0; vid < v; vid++)
        {
            if (env->victim_status[vid] >= 2)
            {
                continue;
            }
            float dx = env->victim_pos[vid][0] - px;
            float dy = env->victim_pos[vid][1] - py;
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq > r_sense_sq)
            {
                continue;
            }
            float dist = sqrtf(dist_sq);
            float p = (1.0f - dist / r_sense) * detect_scale;
            if (p <= 0.0f)
            {
                continue;
            }
            if (p > 1.0f)
            {
                p = 1.0f;
            }
            if (rand_float(&env->rng_state) > p)
            {
                continue;
            }
            if (noise_std > 0.0f)
            {
                dx += rand_normal(&env->rng_state) * noise_std;
                dy += rand_normal(&env->rng_state) * noise_std;
            }

            if (dist_sq < best_dist[k - 1])
            {
                int insert = k - 1;
                while (insert > 0 && dist_sq < best_dist[insert - 1])
                {
                    best_dist[insert] = best_dist[insert - 1];
                    best_dx[insert] = best_dx[insert - 1];
                    best_dy[insert] = best_dy[insert - 1];
                    best_conf[insert] = best_conf[insert - 1];
                    insert--;
                }
                best_dist[insert] = dist_sq;
                best_dx[insert] = dx;
                best_dy[insert] = dy;
                best_conf[insert] = p;
            }
        }

        if (fp_rate > 0.0f && rand_float(&env->rng_state) < fp_rate)
        {
            float angle = rand_uniform(&env->rng_state, 0.0f, 2.0f * PI_F);
            float r = sqrtf(rand_float(&env->rng_state)) * r_sense;
            float dx = cosf(angle) * r;
            float dy = sinf(angle) * r;
            if (noise_std > 0.0f)
            {
                dx += rand_normal(&env->rng_state) * noise_std;
                dy += rand_normal(&env->rng_state) * noise_std;
            }
            float conf = fp_conf > 0.0f ? rand_uniform(&env->rng_state, 0.0f, fp_conf) : 0.0f;
            float dist_sq = r * r;

            if (dist_sq < best_dist[k - 1])
            {
                int insert = k - 1;
                while (insert > 0 && dist_sq < best_dist[insert - 1])
                {
                    best_dist[insert] = best_dist[insert - 1];
                    best_dx[insert] = best_dx[insert - 1];
                    best_dy[insert] = best_dy[insert - 1];
                    best_conf[insert] = best_conf[insert - 1];
                    insert--;
                }
                best_dist[insert] = dist_sq;
                best_dx[insert] = dx;
                best_dy[insert] = dy;
                best_conf[insert] = conf;
            }
        }

        for (int i = 0; i < k; i++)
        {
            if (best_dist[i] >= 1e17f)
            {
                continue;
            }
            env->detections[d][i][0] = clampf(best_dx[i] * inv_L, -1.0f, 1.0f);
            env->detections[d][i][1] = clampf(best_dy[i] * inv_L, -1.0f, 1.0f);
            env->detections[d][i][2] = clampf(best_conf[i], 0.0f, 1.0f);
        }
    }
}

void drone_swarm_init(DroneSwarm *env, const DroneSwarmConfig *cfg)
{
    memset(env, 0, sizeof(DroneSwarm));
    env->cfg = *cfg;
    if (env->cfg.obs_n_nearest > MAX_NEAREST)
    {
        env->cfg.obs_n_nearest = MAX_NEAREST;
    }
    if (env->cfg.n_drones > MAX_DRONES)
    {
        env->cfg.n_drones = MAX_DRONES;
    }
    if (env->cfg.n_victims > MAX_VICTIMS)
    {
        env->cfg.n_victims = MAX_VICTIMS;
    }
    int grid_w = (int)ceilf(env->cfg.world_size / CELL_SIZE);
    if (grid_w < 1)
        grid_w = 1;
    if (grid_w > MAX_GRID)
        grid_w = MAX_GRID;
    env->grid_w = grid_w;
}

void drone_swarm_seed(DroneSwarm *env, uint64_t seed)
{
    if (seed == 0)
    {
        env->rng_state = 88172645463325252ull;
    }
    else
    {
        env->rng_state = seed;
    }
}

void drone_swarm_reset(DroneSwarm *env)
{
    env->step_count = 0;
    env->done = false;

    float L = env->cfg.world_size;
    float base_x = env->cfg.base_pos[0];
    float base_y = env->cfg.base_pos[1];

    if (env->cfg.r_comm_max > env->cfg.r_comm_min && env->cfg.r_comm_min > 0.0f)
    {
        env->cfg.r_comm = rand_uniform(&env->rng_state, env->cfg.r_comm_min, env->cfg.r_comm_max);
    }

    if (env->cfg.p_comm_drop_max > env->cfg.p_comm_drop_min && env->cfg.p_comm_drop_min >= 0.0f)
    {
        env->cfg.p_comm_drop = rand_uniform(
            &env->rng_state, env->cfg.p_comm_drop_min, env->cfg.p_comm_drop_max);
    }

    if (env->cfg.t_confirm_values_count > 0)
    {
        int count = env->cfg.t_confirm_values_count;
        int idx = (int)(rand_float(&env->rng_state) * (float)count);
        if (idx < 0)
            idx = 0;
        if (idx >= count)
            idx = count - 1;
        int value = env->cfg.t_confirm_values[idx];
        if (value > 0)
        {
            env->cfg.t_confirm = value;
        }
    }

    if (env->cfg.m_deliver_values_count > 0)
    {
        int count = env->cfg.m_deliver_values_count;
        int idx = (int)(rand_float(&env->rng_state) * (float)count);
        if (idx < 0)
            idx = 0;
        if (idx >= count)
            idx = count - 1;
        int value = env->cfg.m_deliver_values[idx];
        if (value > 0)
        {
            env->cfg.m_deliver = value;
        }
    }

    if (env->cfg.spawn_near_base)
    {
        float radius = env->cfg.spawn_radius;
        if (radius <= 0.0f)
        {
            radius = env->cfg.r_comm;
        }
        for (int i = 0; i < env->cfg.n_drones; i++)
        {
            float angle = rand_uniform(&env->rng_state, 0.0f, 2.0f * PI_F);
            float r = sqrtf(rand_float(&env->rng_state)) * radius;
            float x = base_x + cosf(angle) * r;
            float y = base_y + sinf(angle) * r;
            env->positions[i][0] = clampf(x, 0.0f, L);
            env->positions[i][1] = clampf(y, 0.0f, L);
        }
    }
    else
    {
        for (int i = 0; i < env->cfg.n_drones; i++)
        {
            env->positions[i][0] = rand_uniform(&env->rng_state, 0.0f, L);
            env->positions[i][1] = rand_uniform(&env->rng_state, 0.0f, L);
        }
    }

    for (int i = 0; i < env->cfg.n_drones; i++)
    {
        env->battery[i] = 1.0f;
        env->last_comm_age[i] = 0.0f;
    }

    float min_dist = env->cfg.victim_min_dist_from_base;
    float max_dist = env->cfg.victim_max_dist_from_base;
    if (env->cfg.victim_mix_prob > 0.0f &&
        env->cfg.victim_max_dist_from_base_alt > env->cfg.victim_min_dist_from_base_alt)
    {
        if (rand_float(&env->rng_state) < env->cfg.victim_mix_prob)
        {
            min_dist = env->cfg.victim_min_dist_from_base_alt;
            max_dist = env->cfg.victim_max_dist_from_base_alt;
        }
    }
    float min_dist_sq = min_dist * min_dist;
    float max_dist_sq = max_dist * max_dist;

    for (int i = 0; i < env->cfg.n_victims; i++)
    {
        bool placed = false;
        for (int attempt = 0; attempt < 100; attempt++)
        {
            float angle = rand_uniform(&env->rng_state, 0.0f, 2.0f * PI_F);
            float r_sq = rand_uniform(&env->rng_state, min_dist_sq, max_dist_sq);
            float r = sqrtf(r_sq);
            float x = base_x + cosf(angle) * r;
            float y = base_y + sinf(angle) * r;
            if (x >= 0.0f && x <= L && y >= 0.0f && y <= L)
            {
                env->victim_pos[i][0] = x;
                env->victim_pos[i][1] = y;
                placed = true;
                break;
            }
        }
        if (!placed)
        {
            env->victim_pos[i][0] = rand_uniform(&env->rng_state, 0.0f, L);
            env->victim_pos[i][1] = rand_uniform(&env->rng_state, 0.0f, L);
        }
    }

    for (int i = 0; i < env->cfg.n_victims; i++)
    {
        env->victim_status[i] = 0;
        env->confirm_progress[i] = 0;
        env->confirm_owner[i] = -1;
        env->deliver_owner[i] = -1;
        env->delivery_ttl[i] = -1;
    }

    memset(env->explored_global, 0, sizeof(bool) * MAX_GRID * MAX_GRID);
    memset(env->explored_per_drone, 0, sizeof(bool) * MAX_DRONES * MAX_GRID * MAX_GRID);
    env->explored_global_count = 0;

    compute_connectivity(env);

    float tmp_cells[MAX_DRONES];
    update_explored_cells(env, tmp_cells);
    memset(env->detections, 0, sizeof(env->detections));
    compute_observations(env);

    env->delivered_count = 0;
    env->confirmed_count = 0;
    env->last_new_delivered = 0;
    env->last_new_confirmed = 0;
}

void drone_swarm_step(DroneSwarm *env, const float *actions)
{
    int n = env->cfg.n_drones;
    float v_max = env->cfg.v_max;
    float dt = env->cfg.dt;
    float L = env->cfg.world_size;

    bool scan_mask[MAX_DRONES];

    for (int i = 0; i < n; i++)
    {
        float vx = actions[i * 3 + 0];
        float vy = actions[i * 3 + 1];
        bool scan = actions[i * 3 + 2] > 0.0f;

        if (env->battery[i] <= 0.0f)
        {
            vx = 0.0f;
            vy = 0.0f;
            scan = false;
        }

        float norm = sqrtf(vx * vx + vy * vy);
        float speed_ratio = norm;
        if (norm > 1.0f)
        {
            vx /= norm;
            vy /= norm;
            speed_ratio = 1.0f;
        }
        vx *= v_max;
        vy *= v_max;

        env->positions[i][0] = clampf(env->positions[i][0] + vx * dt, 0.0f, L);
        env->positions[i][1] = clampf(env->positions[i][1] + vy * dt, 0.0f, L);

        env->battery[i] -= (env->cfg.c_idle + env->cfg.c_move * (speed_ratio * speed_ratio)) * dt;
        if (env->battery[i] < 0.0f)
        {
            env->battery[i] = 0.0f;
        }

        scan_mask[i] = scan;
    }

    compute_connectivity(env);

    int new_delivered = 0;
    int new_confirmed = 0;
    int confirming_drone = -1;
    update_confirm_and_delivery(env, scan_mask, &new_delivered, &new_confirmed, &confirming_drone);
    env->last_new_delivered = new_delivered;
    env->last_new_confirmed = new_confirmed;

    float explored_cells[MAX_DRONES];
    update_explored_cells(env, explored_cells);

    float scan_rewards[MAX_DRONES];
    compute_scan_rewards(env, scan_mask, scan_rewards);

    float dispersion_rewards[MAX_DRONES];
    compute_dispersion_rewards(env, dispersion_rewards);

    for (int i = 0; i < n; i++)
    {
        env->rewards[i] = -env->cfg.c_time;
    }

    if (n > 0 && new_delivered > 0)
    {
        float team_reward = env->cfg.r_found * (float)new_delivered;
        if (env->cfg.r_found_divide_by_n)
        {
            team_reward /= (float)n;
        }
        for (int i = 0; i < n; i++)
        {
            env->rewards[i] += team_reward;
        }
    }

    if (new_confirmed > 0 && confirming_drone >= 0 && confirming_drone < n)
    {
        env->rewards[confirming_drone] += env->cfg.r_confirm_reward;
    }

    float c_energy = env->cfg.c_energy;
    float c_scan = env->cfg.c_scan;
    for (int i = 0; i < n; i++)
    {
        env->rewards[i] += env->cfg.r_explore * explored_cells[i];
        env->rewards[i] += scan_rewards[i];
        env->rewards[i] += dispersion_rewards[i];
        if (env->connected[i])
        {
            env->rewards[i] += env->cfg.r_connectivity;
        }
        if (c_energy != 0.0f)
        {
            float vx = actions[i * 3 + 0];
            float vy = actions[i * 3 + 1];
            float norm = sqrtf(vx * vx + vy * vy);
            float speed_ratio = norm > 1.0f ? 1.0f : norm;
            env->rewards[i] -= c_energy * (speed_ratio * speed_ratio);
        }
        if (c_scan != 0.0f && scan_mask[i])
        {
            env->rewards[i] -= c_scan;
        }
    }

    if (env->cfg.r_owner_connected != 0.0f)
    {
        for (int v = 0; v < env->cfg.n_victims; v++)
        {
            if (env->victim_status[v] == 1)
            {
                int owner = env->deliver_owner[v];
                if (owner >= 0 && owner < n && env->connected[owner])
                {
                    env->rewards[owner] += env->cfg.r_owner_connected;
                }
            }
        }
    }

    // Relay rewards: reward drones that form relay chains for disconnected owners
    if (env->cfg.r_relay_bonus != 0.0f || env->cfg.r_chain_progress != 0.0f)
    {
        float relay_rewards[MAX_DRONES];
        float chain_progress = 0.0f;
        compute_relay_rewards(env, relay_rewards, &chain_progress);
        for (int i = 0; i < n; i++)
        {
            env->rewards[i] += relay_rewards[i];
            env->rewards[i] += chain_progress; // Distributed to all agents
        }
    }

    compute_detections(env, scan_mask);

    env->step_count++;

    env->delivered_count = 0;
    env->confirmed_count = 0;
    for (int v = 0; v < env->cfg.n_victims; v++)
    {
        if (env->victim_status[v] == 2)
        {
            env->delivered_count++;
        }
        else if (env->victim_status[v] == 1)
        {
            env->confirmed_count++;
        }
    }

    env->done = (env->delivered_count == env->cfg.n_victims) || (env->step_count >= env->cfg.max_steps);

    compute_observations(env);
}
