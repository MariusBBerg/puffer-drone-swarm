# Optimizing PufferDroneSwarm for 2M+ Steps Per Second

## Current Performance

- **Current SPS**: ~15,000 steps/second (Python/NumPy)
- **Target SPS**: 2,000,000+ steps/second
- **Gap**: ~130x speedup needed

## Why Python is Slow

1. **Python interpreter overhead** - function calls, attribute lookups
2. **NumPy small array inefficiency** - overhead dominates for small arrays (8 drones, 10 victims)
3. **Python loops** - BFS connectivity, per-victim/per-drone iteration
4. **Memory allocation** - creating new arrays each step
5. **GIL** - can't truly parallelize across CPU cores

## Architecture Options

### Option 1: Cython (Recommended for Quick Wins)

**Effort**: Medium | **Speedup**: 10-50x | **Result**: 150k-750k SPS

```
puffer_drone_swarm/
├── env_cy.pyx          # Cython environment
├── env_cy.pxd          # Cython declarations
├── setup.py            # Build script
└── env.py              # Python fallback
```

**Key changes:**

- Type all variables with `cdef`
- Use `@cython.boundscheck(False)` and `@cython.wraparound(False)`
- Replace Python lists with C arrays or typed memoryviews
- Inline small functions

**Example transformation:**

```cython
# Before (Python)
def _compute_connectivity(self):
    connected = np.zeros(N, dtype=bool)
    queue = list(np.where(dist_base <= self.cfg.r_comm)[0])
    ...

# After (Cython)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_connectivity(self, bint[:] connected) nogil:
    cdef int i, j, queue_start, queue_end
    cdef int[64] queue  # Fixed-size stack array
    cdef float dist
    ...
```

### Option 2: Pure C with Python Bindings (PufferLib Ocean Style)

**Effort**: High | **Speedup**: 100-200x | **Result**: 1.5M-3M SPS

This is what PufferLib's Ocean environments (Snake, Breakout, etc.) use.

**Structure:**

```
puffer_drone_swarm/
├── c_src/
│   ├── drone_swarm.h       # Core environment in C
│   ├── drone_swarm.c       # Implementation
│   └── render.h            # Raylib rendering (optional)
├── drone_swarm_binding.pyx # Cython wrapper for Python
├── setup.py
└── puffer_drone_swarm.py   # PufferLib interface
```

**C Header (drone_swarm.h):**

```c
#ifndef DRONE_SWARM_H
#define DRONE_SWARM_H

#include <stdint.h>
#include <stdbool.h>

#define MAX_DRONES 16
#define MAX_VICTIMS 32

typedef struct {
    float world_size;
    int n_drones;
    int n_victims;
    float r_comm;
    float r_confirm_radius;
    float r_confirm_reward;
    float r_sense;
    int t_confirm;
    int m_deliver;
    float v_max;
    int max_steps;
    // ... other config
} DroneSwarmConfig;

typedef struct {
    // Drone state
    float positions[MAX_DRONES][2];
    float battery[MAX_DRONES];
    bool connected[MAX_DRONES];
    float last_comm_age[MAX_DRONES];

    // Victim state
    float victim_pos[MAX_VICTIMS][2];
    int victim_status[MAX_VICTIMS];  // 0=unknown, 1=confirmed, 2=delivered
    int confirm_progress[MAX_VICTIMS];
    int confirm_owner[MAX_VICTIMS];
    int deliver_owner[MAX_VICTIMS];
    int delivery_ttl[MAX_VICTIMS];

    // Environment state
    float base_pos[2];
    int step_count;
    uint64_t rng_state;

    // Config
    DroneSwarmConfig cfg;

    // Pre-allocated buffers
    float observations[MAX_DRONES * 20];
    float rewards[MAX_DRONES];
    bool done;
} DroneSwarm;

// Core functions
void drone_swarm_init(DroneSwarm* env, DroneSwarmConfig* cfg);
void drone_swarm_reset(DroneSwarm* env);
void drone_swarm_step(DroneSwarm* env, float* actions);

// Helpers
void compute_connectivity(DroneSwarm* env);
void update_confirm_delivery(DroneSwarm* env, bool* scan_mask);
void compute_observations(DroneSwarm* env);

#endif
```

**C Implementation (drone_swarm.c):**

```c
#include "drone_swarm.h"
#include <math.h>
#include <string.h>

// Fast xorshift RNG
static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

static inline float rand_float(uint64_t* state) {
    return (xorshift64(state) >> 11) * (1.0f / 9007199254740992.0f);
}

void drone_swarm_reset(DroneSwarm* env) {
    env->step_count = 0;

    // Spawn drones near base
    float radius = env->cfg.r_comm;
    for (int i = 0; i < env->cfg.n_drones; i++) {
        float angle = rand_float(&env->rng_state) * 6.283185f;
        float r = sqrtf(rand_float(&env->rng_state)) * radius;
        env->positions[i][0] = env->base_pos[0] + cosf(angle) * r;
        env->positions[i][1] = env->base_pos[1] + sinf(angle) * r;
        env->battery[i] = 1.0f;
        env->last_comm_age[i] = 0.0f;
    }

    // Spawn victims in annulus
    for (int i = 0; i < env->cfg.n_victims; i++) {
        // ... spawn logic
        env->victim_status[i] = 0;
        env->confirm_progress[i] = 0;
        env->confirm_owner[i] = -1;
    }

    compute_connectivity(env);
    compute_observations(env);
}

void drone_swarm_step(DroneSwarm* env, float* actions) {
    int n = env->cfg.n_drones;
    float v_max = env->cfg.v_max;
    float dt = 1.0f;
    float world_size = env->cfg.world_size;

    bool scan_mask[MAX_DRONES];

    // Process actions and update positions
    for (int i = 0; i < n; i++) {
        float vx = actions[i * 3 + 0];
        float vy = actions[i * 3 + 1];
        scan_mask[i] = actions[i * 3 + 2] > 0.0f;

        // Skip dead drones
        if (env->battery[i] <= 0.0f) {
            vx = vy = 0.0f;
            scan_mask[i] = false;
        }

        // Normalize velocity
        float norm = sqrtf(vx * vx + vy * vy);
        if (norm > 1.0f) {
            vx /= norm;
            vy /= norm;
        }
        vx *= v_max;
        vy *= v_max;

        // Update position
        env->positions[i][0] += vx * dt;
        env->positions[i][1] += vy * dt;

        // Clamp to world bounds
        if (env->positions[i][0] < 0) env->positions[i][0] = 0;
        if (env->positions[i][0] > world_size) env->positions[i][0] = world_size;
        if (env->positions[i][1] < 0) env->positions[i][1] = 0;
        if (env->positions[i][1] > world_size) env->positions[i][1] = world_size;

        // Battery drain
        float speed_ratio = sqrtf(vx * vx + vy * vy) / v_max;
        env->battery[i] -= (0.0003f + 0.0006f * speed_ratio * speed_ratio) * dt;
        if (env->battery[i] < 0) env->battery[i] = 0;
    }

    compute_connectivity(env);
    update_confirm_delivery(env, scan_mask);
    compute_observations(env);

    // Compute rewards (simplified - expand as needed)
    for (int i = 0; i < n; i++) {
        env->rewards[i] = -0.005f;  // Time penalty
        // ... add other reward components
    }

    env->step_count++;

    // Check done
    int delivered = 0;
    for (int i = 0; i < env->cfg.n_victims; i++) {
        if (env->victim_status[i] == 2) delivered++;
    }
    env->done = (delivered == env->cfg.n_victims) ||
                (env->step_count >= env->cfg.max_steps);
}

// BFS connectivity - O(n²) but very fast in C
void compute_connectivity(DroneSwarm* env) {
    int n = env->cfg.n_drones;
    float r_comm = env->cfg.r_comm;
    float r_comm_sq = r_comm * r_comm;

    // Reset connectivity
    memset(env->connected, 0, sizeof(bool) * n);

    // Find drones connected to base
    int queue[MAX_DRONES];
    int queue_start = 0, queue_end = 0;

    for (int i = 0; i < n; i++) {
        float dx = env->positions[i][0] - env->base_pos[0];
        float dy = env->positions[i][1] - env->base_pos[1];
        if (dx * dx + dy * dy <= r_comm_sq) {
            env->connected[i] = true;
            queue[queue_end++] = i;
        }
    }

    // BFS
    while (queue_start < queue_end) {
        int i = queue[queue_start++];
        for (int j = 0; j < n; j++) {
            if (env->connected[j]) continue;
            float dx = env->positions[i][0] - env->positions[j][0];
            float dy = env->positions[i][1] - env->positions[j][1];
            if (dx * dx + dy * dy <= r_comm_sq) {
                env->connected[j] = true;
                queue[queue_end++] = j;
            }
        }
    }

    // Update comm age
    for (int i = 0; i < n; i++) {
        env->last_comm_age[i] = env->connected[i] ? 0.0f : env->last_comm_age[i] + 1.0f;
    }
}
```

**Cython Binding (drone_swarm_binding.pyx):**

```cython
# distutils: sources = c_src/drone_swarm.c
# distutils: include_dirs = c_src

cimport cython
import numpy as np
cimport numpy as np

cdef extern from "drone_swarm.h":
    ctypedef struct DroneSwarmConfig:
        float world_size
        int n_drones
        int n_victims
        # ...

    ctypedef struct DroneSwarm:
        float observations[320]  # MAX_DRONES * 20
        float rewards[16]
        bint done
        int step_count
        # ...

    void drone_swarm_init(DroneSwarm* env, DroneSwarmConfig* cfg)
    void drone_swarm_reset(DroneSwarm* env)
    void drone_swarm_step(DroneSwarm* env, float* actions)

cdef class CyDroneSwarm:
    cdef DroneSwarm env
    cdef DroneSwarmConfig cfg
    cdef int obs_size

    def __init__(self, config=None):
        # Initialize config from Python dict
        self.cfg.world_size = 100.0
        self.cfg.n_drones = 8
        self.cfg.n_victims = 10
        # ...
        drone_swarm_init(&self.env, &self.cfg)
        self.obs_size = 19  # Adjust based on obs structure

    def reset(self):
        drone_swarm_reset(&self.env)
        return self._get_obs()

    def step(self, np.ndarray[np.float32_t, ndim=2] actions):
        cdef float* action_ptr = <float*>actions.data
        drone_swarm_step(&self.env, action_ptr)
        return self._get_obs(), self._get_rewards(), self.env.done, {}

    cdef np.ndarray _get_obs(self):
        cdef np.ndarray[np.float32_t, ndim=2] obs = np.empty(
            (self.cfg.n_drones, self.obs_size), dtype=np.float32)
        cdef int i, j
        for i in range(self.cfg.n_drones):
            for j in range(self.obs_size):
                obs[i, j] = self.env.observations[i * self.obs_size + j]
        return obs

    cdef np.ndarray _get_rewards(self):
        cdef np.ndarray[np.float32_t, ndim=1] rewards = np.empty(
            self.cfg.n_drones, dtype=np.float32)
        cdef int i
        for i in range(self.cfg.n_drones):
            rewards[i] = self.env.rewards[i]
        return rewards
```

### Option 3: JAX/XLA (GPU Acceleration)

**Effort**: High | **Speedup**: 1000x+ on GPU | **Result**: 10M+ SPS on GPU

Use JAX to JIT-compile the environment and run on GPU.

```python
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def step_env(cfg, state, actions):
    # All NumPy operations become JAX operations
    positions = state['positions']
    vel = actions[:, :2] * cfg.v_max
    new_positions = jnp.clip(positions + vel, 0, cfg.world_size)
    # ... rest of step logic
    return new_state, obs, rewards, done

# Vectorize across many environments
batched_step = jax.vmap(step_env, in_axes=(None, 0, 0))
```

## Implementation Roadmap

### Phase 1: Profile Current Bottlenecks (1 day)

```bash
python -m cProfile -s cumtime train.py --total-timesteps 100000
```

Identify which functions consume the most time.

### Phase 2: Quick Cython Wins (2-3 days)

1. Convert `_compute_connectivity()` to Cython with typed memoryviews
2. Convert `_update_confirm_and_delivery()` to Cython
3. Convert `_compute_dispersion_rewards()` to Cython
4. Pre-allocate all arrays in `__init__`

Expected result: 50k-100k SPS

### Phase 3: Full C Implementation (1-2 weeks)

1. Port entire environment to C
2. Write Cython bindings
3. Integrate with PufferLib

Expected result: 1M-2M SPS

### Phase 4: Vectorized Environments (Optional)

1. Run multiple environments in a single C struct
2. Use SIMD intrinsics for parallel computation
3. Consider GPU with JAX for massive parallelism

Expected result: 5M+ SPS

## Build System (setup.py)

```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "drone_swarm_c",
        sources=["drone_swarm_binding.pyx", "c_src/drone_swarm.c"],
        include_dirs=[np.get_include(), "c_src"],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    )
]

setup(
    name="pufferdroneswarm",
    ext_modules=cythonize(extensions, language_level=3),
)
```

## References

- [PufferLib Ocean environments](https://github.com/PufferAI/PufferLib/tree/main/pufferlib/ocean) - C implementations
- [Cython documentation](https://cython.readthedocs.io/)
- [JAX documentation](https://jax.readthedocs.io/)
- [NumPy C-API](https://numpy.org/doc/stable/reference/c-api/)

## Quick Profiling Commands

```bash
# Profile Python
python -m cProfile -s cumtime -o profile.prof train.py --total-timesteps 50000
python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumtime').print_stats(30)"

# Benchmark raw env speed
python -c "
import time
import numpy as np
from env import DroneSwarmEnv, EnvConfig

env = DroneSwarmEnv(EnvConfig())
env.reset()
start = time.time()
for _ in range(10000):
    env.step(np.random.uniform(-1, 1, (8, 3)).astype(np.float32))
    if env.step_count >= 800: env.reset()
print(f'{10000/(time.time()-start):.0f} env steps/sec')
"
```
