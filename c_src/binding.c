#include "drone_swarm.h"

#define c_render drone_swarm_render
#include "drone_swarm_render.h"
#undef c_render

#include <string.h>

typedef struct {
    float episode_return;
    float episode_length;
    float delivered;
    float confirmed;
    float success;
    float connected_fraction;
    float n;
} Log;

typedef struct {
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    DroneSwarm core;
    int num_agents;
    int obs_size;
    int action_size;
    float episode_return;
    int episode_length;
    Log log;
} DroneSwarmEnv;

static void c_reset(DroneSwarmEnv* env);
static void c_step(DroneSwarmEnv* env);
static void c_render(DroneSwarmEnv* env);
static void c_close(DroneSwarmEnv* env);

#define Env DroneSwarmEnv
#include "env_binding.h"

static int parse_int_list(PyObject* obj, int* out, int max_count, int* out_count) {
    if (!obj || obj == Py_None) {
        *out_count = 0;
        return 0;
    }
    if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected list/tuple for int list");
        return -1;
    }
    Py_ssize_t count = PySequence_Length(obj);
    if (count < 0) {
        return -1;
    }
    int n = (int)count;
    if (n > max_count) {
        n = max_count;
    }
    for (int i = 0; i < n; i++) {
        PyObject* item = PySequence_GetItem(obj, i);
        if (!item) {
            return -1;
        }
        long value = PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            return -1;
        }
        out[i] = (int)value;
    }
    *out_count = n;
    return 0;
}

static int parse_base_pos(PyObject* kwargs, float world_size, float* out_x, float* out_y) {
    PyObject* obj = PyDict_GetItemString(kwargs, "base_pos");
    if (!obj || obj == Py_None) {
        *out_x = world_size * 0.5f;
        *out_y = world_size * 0.5f;
        return 0;
    }
    if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "base_pos must be a list/tuple of length 2");
        return -1;
    }
    if (PySequence_Length(obj) < 2) {
        PyErr_SetString(PyExc_TypeError, "base_pos must have 2 elements");
        return -1;
    }
    PyObject* x_obj = PySequence_GetItem(obj, 0);
    PyObject* y_obj = PySequence_GetItem(obj, 1);
    if (!x_obj || !y_obj) {
        Py_XDECREF(x_obj);
        Py_XDECREF(y_obj);
        return -1;
    }
    double x = PyFloat_AsDouble(x_obj);
    double y = PyFloat_AsDouble(y_obj);
    Py_DECREF(x_obj);
    Py_DECREF(y_obj);
    if (PyErr_Occurred()) {
        return -1;
    }
    *out_x = (float)x;
    *out_y = (float)y;
    return 0;
}

static int parse_obstacles(PyObject* kwargs, DroneSwarmConfig* cfg) {
    PyObject* obj = PyDict_GetItemString(kwargs, "obstacles");
    if (!obj || obj == Py_None) {
        return 0;
    }
    if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "obstacles must be a list/tuple");
        return -1;
    }
    Py_ssize_t count = PySequence_Length(obj);
    if (count < 0) {
        return -1;
    }
    int n = (int)count;
    if (n <= 0) {
        return 0;
    }
    if (n > MAX_OBSTACLES) {
        n = MAX_OBSTACLES;
    }
    for (int i = 0; i < n; i++) {
        PyObject* entry = PySequence_GetItem(obj, i);
        if (!entry) {
            return -1;
        }
        double x = 0.0;
        double y = 0.0;
        double w = 0.0;
        double h = 0.0;
        if (PyDict_Check(entry)) {
            PyObject* x_obj = PyDict_GetItemString(entry, "x");
            PyObject* y_obj = PyDict_GetItemString(entry, "y");
            PyObject* w_obj = PyDict_GetItemString(entry, "w");
            PyObject* h_obj = PyDict_GetItemString(entry, "h");
            if (x_obj) x = PyFloat_AsDouble(x_obj);
            if (y_obj) y = PyFloat_AsDouble(y_obj);
            if (w_obj) w = PyFloat_AsDouble(w_obj);
            if (h_obj) h = PyFloat_AsDouble(h_obj);
            if (PyErr_Occurred()) {
                Py_DECREF(entry);
                return -1;
            }
        } else if (PyList_Check(entry) || PyTuple_Check(entry)) {
            if (PySequence_Length(entry) < 4) {
                Py_DECREF(entry);
                PyErr_SetString(PyExc_TypeError, "Obstacle tuple must have 4 elements");
                return -1;
            }
            PyObject* x_obj = PySequence_GetItem(entry, 0);
            PyObject* y_obj = PySequence_GetItem(entry, 1);
            PyObject* w_obj = PySequence_GetItem(entry, 2);
            PyObject* h_obj = PySequence_GetItem(entry, 3);
            if (!x_obj || !y_obj || !w_obj || !h_obj) {
                Py_XDECREF(x_obj);
                Py_XDECREF(y_obj);
                Py_XDECREF(w_obj);
                Py_XDECREF(h_obj);
                Py_DECREF(entry);
                return -1;
            }
            x = PyFloat_AsDouble(x_obj);
            y = PyFloat_AsDouble(y_obj);
            w = PyFloat_AsDouble(w_obj);
            h = PyFloat_AsDouble(h_obj);
            Py_DECREF(x_obj);
            Py_DECREF(y_obj);
            Py_DECREF(w_obj);
            Py_DECREF(h_obj);
            if (PyErr_Occurred()) {
                Py_DECREF(entry);
                return -1;
            }
        } else {
            Py_DECREF(entry);
            PyErr_SetString(PyExc_TypeError, "Obstacle must be dict or tuple");
            return -1;
        }
        Py_DECREF(entry);
        cfg->obstacle_rects[i][0] = (float)x;
        cfg->obstacle_rects[i][1] = (float)y;
        cfg->obstacle_rects[i][2] = (float)(x + w);
        cfg->obstacle_rects[i][3] = (float)(y + h);
    }
    cfg->obstacle_count = n;
    cfg->obstacle_random = 0;
    return 0;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    DroneSwarmConfig cfg = {0};
    cfg.world_size = (float)unpack(kwargs, "world_size");
    cfg.n_drones = (int)unpack(kwargs, "n_drones");
    cfg.n_victims = (int)unpack(kwargs, "n_victims");
    cfg.r_comm = (float)unpack(kwargs, "r_comm");
    cfg.r_comm_min = (float)unpack(kwargs, "r_comm_min");
    cfg.r_comm_max = (float)unpack(kwargs, "r_comm_max");
    cfg.r_confirm_radius = (float)unpack(kwargs, "r_confirm_radius");
    cfg.r_sense = (float)unpack(kwargs, "r_sense");
    cfg.t_confirm = (int)unpack(kwargs, "t_confirm");
    cfg.m_deliver = (int)unpack(kwargs, "m_deliver");
    cfg.v_max = (float)unpack(kwargs, "v_max");
    cfg.dt = (float)unpack(kwargs, "dt");
    cfg.c_idle = (float)unpack(kwargs, "c_idle");
    cfg.c_move = (float)unpack(kwargs, "c_move");
    cfg.c_scan = (float)unpack(kwargs, "c_scan");
    cfg.c_time = (float)unpack(kwargs, "c_time");
    cfg.c_energy = (float)unpack(kwargs, "c_energy");
    cfg.r_found = (float)unpack(kwargs, "r_found");
    cfg.r_found_divide_by_n = (int)unpack(kwargs, "r_found_divide_by_n");
    cfg.r_confirm_reward = (float)unpack(kwargs, "r_confirm_reward");
    cfg.r_approach = (float)unpack(kwargs, "r_approach");
    cfg.r_explore = (float)unpack(kwargs, "r_explore");
    cfg.r_scan_near_victim = (float)unpack(kwargs, "r_scan_near_victim");
    cfg.r_connectivity = (float)unpack(kwargs, "r_connectivity");
    cfg.r_dispersion = (float)unpack(kwargs, "r_dispersion");
    cfg.min_drone_separation = (float)unpack(kwargs, "min_drone_separation");
    cfg.r_owner_connected = (float)unpack(kwargs, "r_owner_connected");
    cfg.r_relay_bonus = (float)unpack(kwargs, "r_relay_bonus");
    cfg.r_chain_progress = (float)unpack(kwargs, "r_chain_progress");
    cfg.detect_prob_scale = (float)unpack(kwargs, "detect_prob_scale");
    cfg.detect_noise_std = (float)unpack(kwargs, "detect_noise_std");
    cfg.false_positive_rate = (float)unpack(kwargs, "false_positive_rate");
    cfg.false_positive_confidence = (float)unpack(kwargs, "false_positive_confidence");
    cfg.p_comm_drop = (float)unpack(kwargs, "p_comm_drop");
    cfg.p_comm_drop_min = (float)unpack(kwargs, "p_comm_drop_min");
    cfg.p_comm_drop_max = (float)unpack(kwargs, "p_comm_drop_max");
    cfg.max_steps = (int)unpack(kwargs, "max_steps");
    cfg.spawn_near_base = (int)unpack(kwargs, "spawn_near_base");
    cfg.spawn_radius = (float)unpack(kwargs, "spawn_radius");
    cfg.obs_n_nearest = (int)unpack(kwargs, "obs_n_nearest");
    cfg.obs_n_obstacles = (int)unpack(kwargs, "obs_n_obstacles");
    cfg.victim_min_dist_from_base = (float)unpack(kwargs, "victim_min_dist_from_base");
    cfg.victim_max_dist_from_base = (float)unpack(kwargs, "victim_max_dist_from_base");
    cfg.victim_mix_prob = (float)unpack(kwargs, "victim_mix_prob");
    cfg.victim_min_dist_from_base_alt = (float)unpack(kwargs, "victim_min_dist_from_base_alt");
    cfg.victim_max_dist_from_base_alt = (float)unpack(kwargs, "victim_max_dist_from_base_alt");
    cfg.obstacle_count = (int)unpack(kwargs, "obstacle_count");
    cfg.obstacle_min_size = (float)unpack(kwargs, "obstacle_min_size");
    cfg.obstacle_max_size = (float)unpack(kwargs, "obstacle_max_size");
    cfg.obstacle_margin = (float)unpack(kwargs, "obstacle_margin");
    cfg.obstacle_base_clearance = (float)unpack(kwargs, "obstacle_base_clearance");
    cfg.obstacle_min_separation = (float)unpack(kwargs, "obstacle_min_separation");
    cfg.obstacle_blocks_sensing = (int)unpack(kwargs, "obstacle_blocks_sensing");
    if (PyErr_Occurred()) {
        return -1;
    }
    cfg.obstacle_random = cfg.obstacle_count > 0 ? 1 : 0;

    if (parse_base_pos(kwargs, cfg.world_size, &cfg.base_pos[0], &cfg.base_pos[1]) != 0) {
        return -1;
    }

    PyObject* t_confirm_values = PyDict_GetItemString(kwargs, "t_confirm_values");
    if (parse_int_list(t_confirm_values, cfg.t_confirm_values, MAX_T_CONFIRM_VALUES,
                       &cfg.t_confirm_values_count) != 0) {
        return -1;
    }

    PyObject* m_deliver_values = PyDict_GetItemString(kwargs, "m_deliver_values");
    if (parse_int_list(m_deliver_values, cfg.m_deliver_values, MAX_M_DELIVER_VALUES,
                       &cfg.m_deliver_values_count) != 0) {
        return -1;
    }

    if (parse_obstacles(kwargs, &cfg) != 0) {
        return -1;
    }

    drone_swarm_init(&env->core, &cfg);

    env->num_agents = env->core.cfg.n_drones;
    env->obs_size = 10 + 3 * env->core.cfg.obs_n_nearest + 3 * env->core.cfg.obs_n_obstacles;
    env->action_size = 3;

    PyObject* seed_obj = PyDict_GetItemString(kwargs, "seed");
    uint64_t seed = 0;
    if (seed_obj && PyLong_Check(seed_obj)) {
        seed = (uint64_t)PyLong_AsUnsignedLong(seed_obj);
    }
    drone_swarm_seed(&env->core, seed);
    drone_swarm_reset(&env->core);

    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->log = (Log){0};

    int obs_count = env->num_agents * env->obs_size;
    memcpy(env->observations, env->core.observations, obs_count * sizeof(float));
    memcpy(env->rewards, env->core.rewards, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    float denom = log->n > 0.0f ? log->n : 1.0f;
    assign_to_dict(dict, "episode_return", log->episode_return / denom);
    assign_to_dict(dict, "episode_length", log->episode_length / denom);
    assign_to_dict(dict, "delivered", log->delivered / denom);
    assign_to_dict(dict, "confirmed", log->confirmed / denom);
    assign_to_dict(dict, "success", log->success / denom);
    assign_to_dict(dict, "connected_fraction", log->connected_fraction / denom);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

static void c_reset(DroneSwarmEnv* env) {
    drone_swarm_reset(&env->core);
    env->episode_return = 0.0f;
    env->episode_length = 0;
    int obs_count = env->num_agents * env->obs_size;
    memcpy(env->observations, env->core.observations, obs_count * sizeof(float));
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
}

static void c_step(DroneSwarmEnv* env) {
    drone_swarm_step(&env->core, env->actions);

    int obs_count = env->num_agents * env->obs_size;
    memcpy(env->observations, env->core.observations, obs_count * sizeof(float));
    memcpy(env->rewards, env->core.rewards, env->num_agents * sizeof(float));

    env->episode_length += 1;
    float reward_sum = 0.0f;
    for (int i = 0; i < env->num_agents; i++) {
        reward_sum += env->core.rewards[i];
    }
    env->episode_return += reward_sum / (float)env->num_agents;

    if (env->core.done) {
        float delivered = (float)env->core.delivered_count;
        float confirmed = (float)(env->core.confirmed_count + env->core.delivered_count);
        float success = delivered >= env->core.cfg.n_victims ? 1.0f : 0.0f;
        float connected_fraction = env->num_agents > 0
            ? (float)env->core.connected_count / (float)env->num_agents
            : 0.0f;

        env->log.episode_return += env->episode_return;
        env->log.episode_length += (float)env->episode_length;
        env->log.delivered += delivered;
        env->log.confirmed += confirmed;
        env->log.success += success;
        env->log.connected_fraction += connected_fraction;
        env->log.n += 1.0f;

        memset(env->terminals, 1, env->num_agents * sizeof(unsigned char));
        drone_swarm_reset(&env->core);
        env->episode_return = 0.0f;
        env->episode_length = 0;
        memcpy(env->observations, env->core.observations, obs_count * sizeof(float));
    } else {
        memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
    }
}

static void c_render(DroneSwarmEnv* env) {
    drone_swarm_render(&env->core);
}

static void c_close(DroneSwarmEnv* env) {
    c_close_render();
}
