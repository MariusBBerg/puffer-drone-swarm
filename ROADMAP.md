# PufferDroneSwarm Roadmap

> Implementation guide for future improvements. Each section is self-contained with enough detail for an AI agent or contributor to implement.

---

## What This Project Is (And Isn't)

### ✅ This IS:

- **A fast MARL benchmark** for communication-constrained coordination
- **A research tool** for testing algorithms on coverage + connectivity problems
- **An educational environment** for learning multi-agent RL
- **A pre-training sandbox** for high-level swarm policies (potentially transferable to realistic sims)

### ❌ This is NOT:

- A realistic SAR simulator (no real physics, terrain, or sensor models)
- A deployable drone policy (needs sim-to-real transfer via Gazebo/AirSim)
- A validated SAR operations tool

### Target Users:

1. **MARL researchers** - benchmarking new algorithms on coordination tasks
2. **Students** - learning multi-agent RL on an interesting problem
3. **Robotics researchers** - pre-training swarm behaviors before fine-tuning in high-fidelity sims

---

## Current State (v1.0)

✅ **Working features:**

- High-performance C backend (~2M+ SPS)
- 4 drones, 6 victims default config
- Scan/confirm/deliver mechanics
- Multi-hop BFS connectivity to base
- Relay reward shaping (r_relay_bonus, r_chain_progress)
- Dense reward structure for from-scratch training
- Basic pygame visualization
- PPO training with PufferLib integration
- 97-100% success rate on distance ranges 10-55

---

## Training Findings: How We Trained Relay Behavior

This section documents our journey from a failing policy to 97%+ success. Useful for anyone trying to train similar coordination behaviors.

### The Problem

We had a policy that could **find and confirm victims** (~95% success at short range) but **collapsed at far distances** (25-55 range → ~20% success). The policy would confirm victims but fail to deliver because:

- Confirming drone wandered away after confirmation
- No other drones positioned to form relay chain to base
- Policy had no incentive to "help" disconnected teammates

### What Didn't Work

| Approach                         | Result                  | Why It Failed                                     |
| -------------------------------- | ----------------------- | ------------------------------------------------- |
| Just increase `r_found`          | No improvement          | Sparse signal, credit assignment unclear          |
| Add `r_connectivity` penalty     | Slight improvement      | Penalized all disconnection equally, not targeted |
| Aggressive curriculum (50% hard) | Catastrophic forgetting | Lost performance on easy cases                    |
| Large learning rate              | Unstable                | Policy oscillated                                 |

### What Worked: Relay Reward Shaping

We added two new reward components that directly credit drones for relay behavior:

```python
# r_relay_bonus: Reward for being a relay node
# If drone D is on the shortest path connecting a disconnected owner to base,
# D gets r_relay_bonus per step
r_relay_bonus = 0.03  # Small but consistent

# r_chain_progress: Potential-based shaping
# Reward = r_chain_progress * (prev_hops - curr_hops)
# When the chain to a disconnected owner gets shorter, reward the drones involved
r_chain_progress = 0.01
```

**Implementation:** BFS from each disconnected owner to find drones on potential relay paths. Drones on shortest path get bonus.

### The Curriculum That Worked

Starting from a model trained on easy cases (10-35 range):

| Stage         | Config                               | Result                          |
| ------------- | ------------------------------------ | ------------------------------- |
| C2 (baseline) | 10-35 range, no relay shaping        | 95% success on easy, 20% on far |
| C3a           | 20-45 range, add relay rewards       | 92% success after ~15M steps    |
| C3b           | 25-55 range, same rewards            | 89% success                     |
| C4            | Mix 80% easy + 20% far               | 97-100% on all ranges           |
| C5            | Add comm range randomization (14-22) | Robust to r_comm variation      |

**Key insight:** Gentle curriculum (10% → 20% → 30% hard mix) prevented catastrophic forgetting. Aggressive mixing (50% hard) caused regression.

### Hyperparameters That Mattered

| Parameter         | Value | Notes                                          |
| ----------------- | ----- | ---------------------------------------------- |
| `gamma`           | 0.995 | High discount for long-horizon relay behavior  |
| `ent_coef`        | 0.002 | Low entropy once behavior is learned           |
| `learning_rate`   | 1e-4  | Conservative, avoid forgetting                 |
| `m_deliver`       | 120   | Long delivery window (steps to stay connected) |
| `reset_optimizer` | True  | Fresh optimizer when changing curriculum stage |

### Observed Behaviors

**Good:**

- Drones form chains to reach far victims
- One drone often stays near base as "anchor"
- Policy adapts to different comm ranges

**Bad (room for improvement):**

- Drones tend to "cling together" (relay rewards too strong vs exploration)
- Not optimal coverage patterns (no explicit exploration reward balance)
- No explicit role differentiation (all drones use same policy)

### Recommendations for Future Training

1. **Balance relay vs exploration rewards** - Current policy over-prioritizes connectivity
2. **Add dispersion reward** - Penalize drones for being too close together
3. **Consider attention architecture** - For true multi-agent coordination
4. **Larger world size** - Current 100x100 with r_sense=80 is too easy to cover

### The Random Walk Problem

**Critical insight from baselines:** A simple random walk achieves 89-99% success! This means:

1. **The task is easier than expected** - connectivity constraint is not as hard with random movement
2. **RL must beat 90%+ to be useful** - just learning connectivity isn't enough
3. **Speed matters more** - random walk takes 215-398 steps, RL should be faster

**What makes random walk so effective:**

- Large sense radius (r_sense=80) on small world (100x100) means coverage is easy
- Random movement naturally keeps drones loosely together
- No pathological behaviors (unlike lawnmower/voronoi that disconnect)

**Implication:** To make RL impressive, we need:

- **Larger world** (200x200+) or **smaller sense radius** (r_sense=15-25)
- **Time pressure** (lower max_steps to force efficiency)
- **Obstacles** (break random walk's easy connectivity)
- **More victims** (force better coverage strategies)

Current config is a **connectivity benchmark** (easy coverage, hard delivery), not a **coverage benchmark**.

---

## Priority 0: Academic Positioning (For Real Usefulness)

### 0.1 PettingZoo Wrapper

**Why:** Standard interface lets MARL researchers plug in their algorithms.

**File:** `pettingzoo_wrapper.py`

```python
from pettingzoo import ParallelEnv

class DroneSwarmPettingZoo(ParallelEnv):
    """PettingZoo parallel environment wrapper."""

    metadata = {"name": "drone_swarm_v1"}

    def __init__(self, config=None):
        self.env = DroneSwarmEnv(config or EnvConfig())
        self.possible_agents = [f"drone_{i}" for i in range(self.env.cfg.n_drones)]
        self.agents = self.possible_agents.copy()

    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed)
        return {f"drone_{i}": obs[i] for i in range(len(obs))}, {}

    def step(self, actions):
        action_array = np.array([actions[f"drone_{i}"] for i in range(len(self.agents))])
        obs, rewards, done, info = self.env.step(action_array)

        obs_dict = {f"drone_{i}": obs[i] for i in range(len(obs))}
        rewards_dict = {f"drone_{i}": rewards[i] for i in range(len(rewards))}
        terms = {f"drone_{i}": done for i in range(len(self.agents))}
        truncs = {f"drone_{i}": False for i in range(len(self.agents))}

        return obs_dict, rewards_dict, terms, truncs, info
```

**Note:** This is for compatibility only. Use PufferLib native for training (10-100x faster).

### 0.2 Benchmark Task Variants

Define standardized tasks for reproducible comparison:

| Task ID          | Description                      | Config                                      |
| ---------------- | -------------------------------- | ------------------------------------------- |
| `coverage-easy`  | Basic coverage, large comm range | 4 drones, 6 victims, r_comm=22, range 10-35 |
| `coverage-hard`  | Tight comm, far victims          | 4 drones, 6 victims, r_comm=15, range 25-55 |
| `relay-required` | Must form relay chains           | 4 drones, 6 victims, r_comm=12, range 30-60 |
| `scale-test`     | Many agents                      | 8 drones, 12 victims, r_comm=18             |
| `robust`         | Variable conditions              | r_comm randomized 12-22, p_comm_drop 0-0.1  |

### 0.3 Multi-Algorithm Comparison

Run and publish results for:

- **PPO** (current)
- **MAPPO** (centralized critic)
- **IPPO** (independent PPO)
- **QMIX** (value decomposition)
- **Random baseline**
- **Heuristic baselines** (lawnmower, frontier)

---

## Priority 1: Baseline Policies

**Why:** Prove RL actually learns something useful. Baselines provide a lower bound for comparison.

### Baseline Results (Actual Performance)

**Critical Finding:** Random exploration with connectivity-aware returns dominates structured coverage!

| Baseline        | Easy (10-35) | Medium (20-45) | Hard (25-55) | Key Insight                                          |
| --------------- | ------------ | -------------- | ------------ | ---------------------------------------------------- |
| **Random Walk** | **99.0%** ✅ | **98.0%** ✅   | **89.5%** ✅ | Stochastic movement naturally maintains connectivity |
| Voronoi         | 30.5%        | 21.0%          | 14.5%        | Partitions break comm graph, poor delivery           |
| Lawnmower       | 4.0%         | 0.5%           | 0.0% ❌      | Deterministic paths disconnect from base             |

**Why Random Walk Wins:**

1. **Natural connectivity preservation** - random movement keeps drones near each other
2. **Efficient scanning** - 30% scan rate is enough with r_sense=80 on world_size=100
3. **No pathological patterns** - doesn't get stuck in bad local behaviors

**Why Structured Methods Fail:**

1. **Lawnmower** - drones spread to far corners, break comm graph, can't deliver
2. **Voronoi** - similar problem, rigid partitions ignore connectivity constraint

**Implication:** The environment is **connectivity-hard but coverage-easy** (large sense radius). A simple policy that stays loosely connected while wandering randomly solves most cases. This makes it a good benchmark for **coordination under comm constraints**, not coverage optimization.

### 1.1 Random Walk Baseline ✅ (Implemented)

**File:** `baselines/random_walk.py`

**Algorithm:**

```python
for each drone:
    if not currently moving to a waypoint:
        pick random point within world bounds
        set as waypoint
    move toward waypoint at v_max
    if close to waypoint: clear waypoint
    scan = random(0, 1) > 0.7  # 30% scan rate
```

**Performance:**

- Easy: 99.0% success, 215 steps avg
- Medium: 98.0% success, 286 steps avg
- Hard: 89.5% success, 398 steps avg

### 1.2 Lawnmower Coverage Baseline ✅ (Implemented)

**File:** `baselines/lawnmower.py`

**Algorithm:**

```python
# Divide world into N vertical strips (one per drone)
strip_width = world_size / n_drones

for drone i:
    assigned_strip = [i * strip_width, (i+1) * strip_width]

    # Lawnmower pattern within strip
    y_direction = 1 if (current_row % 2 == 0) else -1
    move in y_direction until hitting boundary
    then shift x by scan_width and reverse y_direction

    scan = True (always scanning)
```

**Performance:**

- Easy: 4.0% success (confirms 3.2/6 but delivers only 2.8/6)
- Medium: 0.5% success
- Hard: 0.0% success (complete failure)

**Why it fails:** Excellent coverage (100% scan rate, 95%+ detections) but drones disconnect during systematic coverage, can't deliver confirmations.

### 1.3 Voronoi Partition Baseline ✅ (Implemented)

**File:** `baselines/voronoi.py`

**Algorithm:**

```python
# Assign regions via Voronoi partition
for each drone:
    # Spiral outward from centroid while staying connected
    target = next_spiral_point(assigned_region)
    if can_reach_base_from(target):
        move_toward(target)
    else:
        move_toward_base()  # Return if getting disconnected
    scan = True (always scanning)
```

**Performance:**

- Easy: 30.5% success (delivers 4.7/6)
- Medium: 21.0% success (delivers 4.4/6)
- Hard: 14.5% success (delivers 4.1/6)

**Why it's mediocre:** Balanced tradeoff - tries to maintain connectivity while covering regions, but rigid partitions cause issues.

### 1.4 Frontier Exploration Baseline (NOT YET IMPLEMENTED)

**File:** `baselines/frontier.py`

**Algorithm:**

```python
# Track explored cells in a grid
explored = np.zeros((grid_size, grid_size), dtype=bool)

for each drone:
    # Mark current area as explored
    mark_explored(position, r_sense)

    # Find frontier cells (unexplored adjacent to explored)
    frontiers = find_frontier_cells(explored)

    # Pick nearest frontier that keeps connectivity
    valid_frontiers = [f for f in frontiers
                       if can_reach_base_from(f, other_drone_positions)]

    target = nearest(valid_frontiers, position)
    move_toward(target)
    scan = True
```

**Expected performance:** 50-70% success (better than Voronoi, worse than random walk)

### 1.5 Implementation Status & Next Steps

✅ **Implemented:**

- Random walk (99%/98%/89%)
- Lawnmower (4%/0.5%/0%)
- Voronoi (30%/21%/14%)

❌ **Not yet implemented:**

- Frontier exploration
- Potential field methods
- Greedy nearest-victim

### 1.6 Key Takeaways for RL Training

**The bar is HIGH:** Random walk at 89-99% means RL needs to beat ~90% just to be competitive!

**What RL should learn:**

1. **Better than random** - structured search patterns (lawnmower/voronoi prove this alone doesn't work)
2. **Connectivity-aware** - like random walk but intentional
3. **Efficient delivery** - get victims faster than 215-398 steps

**Target:** PPO should achieve 95%+ at 150-250 steps (faster than random walk).

**File:** `baselines/voronoi.py`

**Algorithm:**

```
# Assign regions via Voronoi partition
centroids = initial_drone_positions
for iteration in range(lloyd_iterations):
    # Assign cells to nearest drone
    regions = voronoi_partition(centroids, world_grid)
    # Move centroids to region centers
    centroids = [region_centroid(r) for r in regions]

# One drone designated as relay (stays near base)
relay_drone = argmin(distance_to_base)

for each non-relay drone:
    # Spiral outward from centroid
    spiral_search(centroid, region_boundary)
    scan = True
```

**Key details:**

- Lloyd's algorithm for balanced partitions
- Explicit relay drone role
- Should achieve ~70-85% success rate

### 1.5 Implementation Checklist

- [ ] Create `baselines/` directory
- [ ] Implement `BaselinePolicy` base class with same interface as `DroneSwarmPolicy`
- [ ] Each baseline outputs actions in same format: `(n_drones, 3)` array
- [ ] Add `--baseline` flag to `eval.py` to run baselines
- [ ] Create `baselines/benchmark.py` to run all baselines and output comparison table

**Expected output format:**

```
| Policy          | Success % | Avg Steps | Delivered/Ep |
|-----------------|-----------|-----------|--------------|
| Random Walk     | 12.3%     | 800       | 1.2          |
| Lawnmower       | 45.6%     | 650       | 3.8          |
| Frontier        | 72.1%     | 520       | 5.2          |
| Voronoi         | 78.4%     | 480       | 5.5          |
| PPO (ours)      | 97.2%     | 380       | 5.9          |
```

---

## Priority 2: Config Presets System

**Why:** Users should be able to quickly test different difficulty levels without editing code.

### 2.1 Preset Definitions

**File:** `presets.py`

```python
from dataclasses import asdict
from env import EnvConfig

PRESETS = {
    # Training presets
    "easy": EnvConfig(
        n_drones=4,
        n_victims=4,
        r_comm=22.0,
        victim_min_dist_from_base=10.0,
        victim_max_dist_from_base=30.0,
        max_steps=600,
    ),
    "medium": EnvConfig(
        n_drones=4,
        n_victims=6,
        r_comm=18.0,
        victim_min_dist_from_base=15.0,
        victim_max_dist_from_base=45.0,
        max_steps=800,
    ),
    "hard": EnvConfig(
        n_drones=4,
        n_victims=6,
        r_comm=15.0,
        victim_min_dist_from_base=25.0,
        victim_max_dist_from_base=55.0,
        max_steps=1000,
    ),
    "extreme": EnvConfig(
        n_drones=4,
        n_victims=8,
        r_comm=12.0,
        victim_min_dist_from_base=30.0,
        victim_max_dist_from_base=70.0,
        max_steps=1200,
    ),

    # Scaling presets
    "swarm-small": EnvConfig(n_drones=4, n_victims=6),
    "swarm-medium": EnvConfig(n_drones=8, n_victims=12),
    "swarm-large": EnvConfig(n_drones=16, n_victims=24),
    "swarm-massive": EnvConfig(n_drones=32, n_victims=48, world_size=200.0),

    # Robustness presets (for eval)
    "tight-comm": EnvConfig(r_comm=12.0, r_comm_min=10.0, r_comm_max=14.0),
    "noisy-comm": EnvConfig(p_comm_drop=0.1, p_comm_drop_min=0.05, p_comm_drop_max=0.15),
}

def make(preset_name: str, **overrides) -> EnvConfig:
    """Factory function to create config from preset with optional overrides."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    base = PRESETS[preset_name]
    if overrides:
        payload = asdict(base)
        payload.update(overrides)
        return EnvConfig(**payload)
    return base

def list_presets() -> list[str]:
    """List all available preset names."""
    return list(PRESETS.keys())
```

### 2.2 Usage Integration

Update `train.py`:

```python
# Add CLI argument
parser.add_argument("--preset", type=str, default=None, help="Config preset name")

# In main():
if args.preset:
    from presets import make
    env_config = make(args.preset)
```

Update `eval.py`:

```python
# Add preset support
if args.preset:
    from presets import make
    base_config = make(args.preset)
```

### 2.3 Implementation Checklist

- [ ] Create `presets.py` with preset definitions
- [ ] Add `--preset` flag to `train.py`
- [ ] Add `--preset` flag to `eval.py`
- [ ] Add `--preset` flag to `visualize.py`
- [ ] Document presets in README

---

## Priority 3: Eval Suite with Export

**Why:** Reproducible benchmarking requires deterministic seeds, multiple runs, and exportable results.

### 3.1 Eval Protocol

**File:** `eval.py` (refactor existing)

**Features needed:**

1. Deterministic seed control (train seeds vs test seeds)
2. Multiple runs with different seeds
3. CSV/JSON export of results
4. Distribution shift tests
5. Statistical significance (mean ± std over N seeds)

### 3.2 Seed Management

```python
# Train seeds: 0-999
# Test seeds: 1000-1999 (never seen during training)
TRAIN_SEEDS = range(0, 1000)
TEST_SEEDS = range(1000, 2000)

def get_eval_seeds(n_episodes: int, split: str = "test") -> list[int]:
    """Get deterministic seeds for evaluation."""
    seed_range = TEST_SEEDS if split == "test" else TRAIN_SEEDS
    rng = np.random.default_rng(42)  # Fixed RNG for reproducibility
    return rng.choice(list(seed_range), size=n_episodes, replace=False).tolist()
```

### 3.3 Distribution Shift Tests

```python
DISTRIBUTION_SHIFTS = {
    "comm_range_shift": {
        "description": "Train r_comm=18, test r_comm=12-15",
        "train": {"r_comm": 18.0},
        "test": {"r_comm_min": 12.0, "r_comm_max": 15.0},
    },
    "victim_distance_shift": {
        "description": "Train 15-45, test 30-60",
        "train": {"victim_min_dist_from_base": 15.0, "victim_max_dist_from_base": 45.0},
        "test": {"victim_min_dist_from_base": 30.0, "victim_max_dist_from_base": 60.0},
    },
    "scale_shift": {
        "description": "Train 4 drones, test 8 drones",
        "train": {"n_drones": 4, "n_victims": 6},
        "test": {"n_drones": 8, "n_victims": 12},
    },
    "comm_noise_shift": {
        "description": "Train p_drop=0, test p_drop=0.1",
        "train": {"p_comm_drop": 0.0},
        "test": {"p_comm_drop": 0.1},
    },
}
```

### 3.4 Export Format

**CSV output (`results/eval_TIMESTAMP.csv`):**

```csv
model,preset,variant,seed,success,steps,delivered,confirmed,explored_frac,avg_connected
policy_v1.pt,medium,default,1000,1,423,6,6,0.82,0.91
policy_v1.pt,medium,default,1001,1,512,6,6,0.78,0.88
policy_v1.pt,medium,tight_comm,1000,0,800,4,5,0.71,0.72
...
```

**JSON summary (`results/eval_TIMESTAMP.json`):**

```json
{
  "model": "policy_v1.pt",
  "timestamp": "2026-01-02T12:00:00",
  "results": {
    "medium/default": {
      "success_rate": {"mean": 0.97, "std": 0.02},
      "avg_steps": {"mean": 456, "std": 45},
      "avg_delivered": {"mean": 5.9, "std": 0.3}
    },
    "medium/tight_comm": {
      "success_rate": {"mean": 0.72, "std": 0.05},
      ...
    }
  }
}
```

### 3.5 CLI Interface

```bash
# Basic eval
uv run python eval.py --model checkpoints/policy.pt --preset medium --episodes 100

# With distribution shift
uv run python eval.py --model checkpoints/policy.pt --preset medium --shift comm_range_shift

# Full benchmark (all presets, all shifts)
uv run python eval.py --model checkpoints/policy.pt --full-benchmark --output results/

# Compare multiple models
uv run python eval.py --models policy_v1.pt,policy_v2.pt --preset medium --compare
```

### 3.6 Implementation Checklist

- [ ] Add seed management (train/test split)
- [ ] Add distribution shift configs
- [ ] Add CSV export
- [ ] Add JSON summary export
- [ ] Add `--full-benchmark` mode
- [ ] Add `--compare` mode for multiple models
- [ ] Create `results/` directory structure

---

## Priority 4: Obstacles & Terrain

**Why:** More realistic SAR scenarios, interesting emergent behavior around obstacles.

### 4.1 Obstacle Types

| Type         | Shape     | Effect on Drones       | Effect on Sensing       | Effect on Comm     |
| ------------ | --------- | ---------------------- | ----------------------- | ------------------ |
| Rock         | Circle    | Collision (bounce)     | Blocks detection        | No effect          |
| Cabin        | Rectangle | Collision              | Blocks detection        | Partial block      |
| Tree cluster | Circle    | Slow zone (0.5x speed) | Reduces detection prob  | No effect          |
| Water        | Polygon   | No-fly zone            | Clear sensing           | No effect          |
| Hill         | Circle    | No collision           | Blocks detection behind | Blocks comm behind |

### 4.2 Data Structures

**Add to `EnvConfig`:**

```python
@dataclass
class Obstacle:
    type: str  # "rock", "cabin", "trees", "water", "hill"
    x: float
    y: float
    radius: float  # For circles
    width: float = 0.0  # For rectangles
    height: float = 0.0  # For rectangles

@dataclass
class EnvConfig:
    # ... existing fields ...

    # Obstacle settings
    obstacles: list[Obstacle] = field(default_factory=list)
    n_random_obstacles: int = 0  # Auto-generate this many
    obstacle_density: float = 0.0  # Alternative: fraction of world covered
    obstacle_types: tuple[str, ...] = ("rock", "cabin", "trees")
```

### 4.3 Collision Detection

**In C backend (`c_src/drone_swarm.c`):**

```c
typedef struct {
    int type;  // 0=rock, 1=cabin, 2=trees, 3=water, 4=hill
    float x, y;
    float radius;
    float width, height;  // For rectangles
} Obstacle;

// Add to DroneSwarmState
Obstacle* obstacles;
int n_obstacles;

// Collision check (called in step)
int check_collision(float x, float y, Obstacle* obs) {
    if (obs->type == 1) {  // Rectangle (cabin)
        return (x >= obs->x - obs->width/2 && x <= obs->x + obs->width/2 &&
                y >= obs->y - obs->height/2 && y <= obs->y + obs->height/2);
    } else {  // Circle
        float dx = x - obs->x;
        float dy = y - obs->y;
        return (dx*dx + dy*dy) <= obs->radius * obs->radius;
    }
}

// In step(): handle collision response
for (int i = 0; i < n_drones; i++) {
    float new_x = positions[i*2] + vx * dt;
    float new_y = positions[i*2+1] + vy * dt;

    for (int j = 0; j < n_obstacles; j++) {
        if (check_collision(new_x, new_y, &obstacles[j])) {
            // Bounce: reflect velocity
            // Or: stop at boundary
            new_x = positions[i*2];  // Simple: don't move
            new_y = positions[i*2+1];
            break;
        }
    }

    positions[i*2] = new_x;
    positions[i*2+1] = new_y;
}
```

### 4.4 Line-of-Sight for Sensing

```c
// Check if line from (x1,y1) to (x2,y2) intersects obstacle
int line_intersects_obstacle(float x1, float y1, float x2, float y2, Obstacle* obs) {
    if (obs->type == 0 || obs->type == 4) {  // Rock or hill blocks LOS
        // Circle-line intersection
        float dx = x2 - x1;
        float dy = y2 - y1;
        float fx = x1 - obs->x;
        float fy = y1 - obs->y;

        float a = dx*dx + dy*dy;
        float b = 2*(fx*dx + fy*dy);
        float c = fx*fx + fy*fy - obs->radius*obs->radius;

        float discriminant = b*b - 4*a*c;
        if (discriminant >= 0) {
            float t = (-b - sqrt(discriminant)) / (2*a);
            if (t >= 0 && t <= 1) return 1;
        }
    }
    return 0;
}

// Modified detection: check LOS to victim
float detection_prob = base_prob;
for (int j = 0; j < n_obstacles; j++) {
    if (line_intersects_obstacle(drone_x, drone_y, victim_x, victim_y, &obstacles[j])) {
        if (obstacles[j].type == 0) detection_prob = 0;  // Rock: full block
        else if (obstacles[j].type == 2) detection_prob *= 0.5;  // Trees: partial
    }
}
```

### 4.5 Line-of-Sight for Comms

```c
// Hills block comm, others don't (radio goes through trees/cabins)
int comm_blocked(float x1, float y1, float x2, float y2) {
    for (int j = 0; j < n_obstacles; j++) {
        if (obstacles[j].type == 4) {  // Hill
            if (line_intersects_obstacle(x1, y1, x2, y2, &obstacles[j])) {
                return 1;
            }
        }
    }
    return 0;
}

// Modified connectivity check
int can_communicate(int drone_i, int drone_j) {
    float dist = distance(drone_i, drone_j);
    if (dist > r_comm) return 0;
    if (comm_blocked(pos[i], pos[j])) return 0;
    return 1;
}
```

### 4.6 Random Obstacle Generation

```python
def generate_random_obstacles(
    n_obstacles: int,
    world_size: float,
    base_pos: np.ndarray,
    min_dist_from_base: float = 15.0,
    obstacle_types: tuple[str, ...] = ("rock", "cabin", "trees"),
    rng: np.random.Generator = None,
) -> list[Obstacle]:
    """Generate random obstacles avoiding the base area."""
    if rng is None:
        rng = np.random.default_rng()

    obstacles = []
    for _ in range(n_obstacles):
        # Keep trying until we find valid position
        for attempt in range(100):
            obs_type = rng.choice(obstacle_types)
            x = rng.uniform(5, world_size - 5)
            y = rng.uniform(5, world_size - 5)

            # Size based on type
            if obs_type == "rock":
                radius = rng.uniform(2, 5)
            elif obs_type == "cabin":
                radius = 0
                width = rng.uniform(4, 8)
                height = rng.uniform(4, 8)
            elif obs_type == "trees":
                radius = rng.uniform(3, 8)
            elif obs_type == "hill":
                radius = rng.uniform(5, 12)

            # Check distance from base
            dist_to_base = np.sqrt((x - base_pos[0])**2 + (y - base_pos[1])**2)
            if dist_to_base < min_dist_from_base:
                continue

            # Check overlap with existing obstacles
            overlaps = False
            for existing in obstacles:
                dist = np.sqrt((x - existing.x)**2 + (y - existing.y)**2)
                if dist < radius + existing.radius + 2:
                    overlaps = True
                    break

            if not overlaps:
                obstacles.append(Obstacle(
                    type=obs_type, x=x, y=y, radius=radius,
                    width=width if obs_type == "cabin" else 0,
                    height=height if obs_type == "cabin" else 0,
                ))
                break

    return obstacles
```

### 4.7 Observation Updates

Add obstacle info to observation:

```python
# Option A: Add nearest obstacle features (minimal obs change)
# [dist_to_nearest_obstacle, angle_to_nearest_obstacle, obstacle_type]

# Option B: Local obstacle grid (more expensive)
# (P, P) grid around drone with obstacle presence

# Option C: Raycast features
# [distance_to_obstacle_in_8_directions]  # 8 floats
```

Recommend **Option A** for minimal impact:

```python
# Add to observation (3 new features)
nearest_obs_dist = min(dist(pos, obs) for obs in obstacles) / world_size
nearest_obs_angle = angle_to_nearest_obstacle / (2 * pi)
nearest_obs_type = obstacle_type_encoding  # 0-1 normalized
```

### 4.8 Visualization Updates

```python
def draw_obstacles(self):
    """Draw obstacles on the map."""
    for obs in self.env.obstacles:
        screen_pos = self.world_to_screen(obs.x, obs.y)

        if obs.type == "rock":
            color = (128, 128, 128)  # Gray
            radius = int(obs.radius * self.scale)
            pygame.draw.circle(self.screen, color, screen_pos, radius)

        elif obs.type == "cabin":
            color = (139, 90, 43)  # Brown
            rect = pygame.Rect(
                screen_pos[0] - int(obs.width * self.scale / 2),
                screen_pos[1] - int(obs.height * self.scale / 2),
                int(obs.width * self.scale),
                int(obs.height * self.scale),
            )
            pygame.draw.rect(self.screen, color, rect)

        elif obs.type == "trees":
            color = (34, 139, 34)  # Forest green
            radius = int(obs.radius * self.scale)
            pygame.draw.circle(self.screen, color, screen_pos, radius)
            # Add some texture
            for _ in range(5):
                offset = np.random.randint(-radius//2, radius//2, 2)
                pygame.draw.circle(self.screen, (0, 100, 0),
                                   (screen_pos[0]+offset[0], screen_pos[1]+offset[1]),
                                   radius//3)

        elif obs.type == "hill":
            color = (160, 82, 45)  # Sienna
            radius = int(obs.radius * self.scale)
            pygame.draw.circle(self.screen, color, screen_pos, radius)
            # Contour lines
            for r in range(radius//4, radius, radius//4):
                pygame.draw.circle(self.screen, (139, 69, 19), screen_pos, r, 1)
```

### 4.9 Presets with Obstacles

```python
PRESETS["obstacles-light"] = EnvConfig(
    n_drones=4,
    n_victims=6,
    n_random_obstacles=5,
    obstacle_types=("rock", "trees"),
)

PRESETS["obstacles-heavy"] = EnvConfig(
    n_drones=4,
    n_victims=6,
    n_random_obstacles=15,
    obstacle_types=("rock", "cabin", "trees", "hill"),
)

PRESETS["sar-realistic"] = EnvConfig(
    n_drones=6,
    n_victims=8,
    n_random_obstacles=10,
    obstacle_types=("rock", "cabin", "trees", "hill"),
    r_comm=16.0,
    victim_min_dist_from_base=20.0,
    victim_max_dist_from_base=60.0,
)
```

### 4.10 Implementation Checklist

- [ ] Add `Obstacle` dataclass to `env.py`
- [ ] Add obstacle fields to `EnvConfig`
- [ ] Implement `generate_random_obstacles()` in Python
- [ ] Add obstacle structs to C header (`drone_swarm.h`)
- [ ] Implement collision detection in C
- [ ] Implement LOS checks for sensing in C
- [ ] Implement LOS checks for comm in C
- [ ] Update Cython bindings
- [ ] Add obstacle features to observation
- [ ] Update visualizer to draw obstacles
- [ ] Add obstacle presets
- [ ] Test that SPS impact is acceptable (<20% slowdown)

---

## Priority 5: Recording & GIF Generation

**Why:** README appeal, debugging, sharing results.

### 5.1 Add Recording to Visualizer

**Update `visualize.py`:**

```python
import os
from PIL import Image

class DroneSwarmVisualizer:
    def __init__(self, ..., record: bool = False, record_dir: str = "recordings"):
        self.record = record
        self.record_dir = record_dir
        self.frames = []

        if record:
            os.makedirs(record_dir, exist_ok=True)

    def capture_frame(self):
        """Capture current frame for recording."""
        if self.record:
            # Get pygame surface as string buffer
            frame_str = pygame.image.tostring(self.screen, "RGB")
            frame = Image.frombytes("RGB", (self.width, self.height), frame_str)
            self.frames.append(frame)

    def save_gif(self, filename: str = "episode.gif", fps: int = 15):
        """Save recorded frames as GIF."""
        if not self.frames:
            print("No frames recorded!")
            return

        filepath = os.path.join(self.record_dir, filename)
        self.frames[0].save(
            filepath,
            save_all=True,
            append_images=self.frames[1:],
            duration=1000 // fps,
            loop=0,
        )
        print(f"Saved GIF to {filepath} ({len(self.frames)} frames)")
        self.frames = []

    def save_mp4(self, filename: str = "episode.mp4", fps: int = 30):
        """Save recorded frames as MP4 (requires imageio)."""
        import imageio

        filepath = os.path.join(self.record_dir, filename)
        writer = imageio.get_writer(filepath, fps=fps)
        for frame in self.frames:
            writer.append_data(np.array(frame))
        writer.close()
        print(f"Saved MP4 to {filepath}")
        self.frames = []
```

### 5.2 CLI Flags

```bash
# Record a single episode
uv run python visualize.py --checkpoint model.pt --record --output episode.gif

# Record multiple episodes
uv run python visualize.py --checkpoint model.pt --record --episodes 5 --output demo.gif

# Record as MP4
uv run python visualize.py --checkpoint model.pt --record --format mp4
```

### 5.3 Batch Recording Script

**File:** `record_demos.py`

```python
"""Record demo GIFs for README and documentation."""

import argparse
from visualize import DroneSwarmVisualizer
from presets import make
from train import load_policy

DEMO_CONFIGS = [
    ("easy_success", "easy", 42),
    ("medium_relay", "medium", 123),  # Seed that shows relay behavior
    ("hard_challenge", "hard", 456),
    ("swarm_large", "swarm-large", 789),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="demos")
    args = parser.parse_args()

    policy = load_policy(args.checkpoint)

    for name, preset, seed in DEMO_CONFIGS:
        config = make(preset)
        viz = DroneSwarmVisualizer(config, record=True, record_dir=args.output_dir)

        obs = viz.env.reset(seed=seed)
        done = False
        while not done:
            action = policy.get_action(obs)
            obs, _, done, _ = viz.env.step(action)
            viz.render()
            viz.capture_frame()

        viz.save_gif(f"{name}.gif")
        print(f"Recorded {name}")

if __name__ == "__main__":
    main()
```

### 5.4 Implementation Checklist

- [ ] Add `record` mode to `DroneSwarmVisualizer`
- [ ] Implement `capture_frame()` method
- [ ] Implement `save_gif()` method
- [ ] Implement `save_mp4()` method (optional, needs imageio)
- [ ] Add CLI flags to `visualize.py`
- [ ] Create `record_demos.py` script
- [ ] Add Pillow to dependencies in `pyproject.toml`
- [ ] Record demo GIFs for README

---

## Priority 6: README & Documentation Update

**Why:** First impression for users, determines adoption.

### 6.1 README Structure

```markdown
# PufferDroneSwarm 🚁

A high-throughput multi-agent SAR benchmark with communication constraints.

![Demo GIF](demos/medium_relay.gif)

## Highlights

- ⚡ **2M+ steps/second** with C backend
- 🔗 **Emergent relay behavior** from reward shaping
- 📊 **Benchmark-ready** with baselines and eval suite
- 🎮 **PufferLib native** for vectorized training

## Quick Start

### Install

\`\`\`bash
git clone https://github.com/...
cd PufferDroneSwarm
uv sync
uv run python setup.py build_ext --inplace
\`\`\`

### Train

\`\`\`bash
uv run python train.py --preset medium --total-timesteps 50000000
\`\`\`

### Evaluate

\`\`\`bash
uv run python eval.py --checkpoint checkpoints/policy.pt --preset medium
\`\`\`

### Visualize

\`\`\`bash
uv run python visualize.py --checkpoint checkpoints/policy.pt
\`\`\`

## Benchmark Results

| Policy         | Easy    | Medium  | Hard    | Extreme |
| -------------- | ------- | ------- | ------- | ------- |
| Random Walk    | 15%     | 8%      | 3%      | 1%      |
| Lawnmower      | 52%     | 38%     | 22%     | 12%     |
| Frontier       | 78%     | 65%     | 48%     | 31%     |
| **PPO (ours)** | **99%** | **97%** | **89%** | **72%** |

## Config Presets

| Preset  | Drones | Victims | Comm Range | Difficulty |
| ------- | ------ | ------- | ---------- | ---------- |
| easy    | 4      | 4       | 22         | ⭐         |
| medium  | 4      | 6       | 18         | ⭐⭐       |
| hard    | 4      | 6       | 15         | ⭐⭐⭐     |
| extreme | 4      | 8       | 12         | ⭐⭐⭐⭐   |

## Environment Details

[... observation/action space tables ...]

## Citation

\`\`\`bibtex
@software{pufferdronesarm2026,
...
}
\`\`\`
```

### 6.2 Implementation Checklist

- [ ] Update README with new structure
- [ ] Add demo GIFs
- [ ] Add benchmark results table
- [ ] Add config presets table
- [ ] Add citation block
- [ ] Add architecture diagram (optional)
- [ ] Add "Why this environment?" section
- [ ] Add troubleshooting section

---

## Future Ideas (v2+)

### Information-Theoretic Rewards

- Reward based on entropy reduction in belief map
- Encourages "checking behind rocks"

### Heterogeneous Drones

- Different drone types: scout (fast, short range), relay (slow, long comm range), carrier (for delivery)
- Role specialization emerges from training

### Dynamic Victims

- Victims that move (injured person crawling)
- Victims that call for help (audio beacon)

### Weather Effects

- Wind affecting drone movement
- Fog reducing sensor range
- Rain affecting comm reliability

### Multi-Base Scenarios

- Multiple base stations
- Drones must choose which base to relay through

### Adversarial Settings

- Comm jamming zones
- Limited-time rescue (victim health declining)

---

## Development Notes

### Performance Targets

- Maintain >1M SPS with obstacles
- Keep observation size <30 dims
- Episode length <1000 steps for hard preset

### Testing Checklist

- [ ] Unit tests for collision detection
- [ ] Unit tests for LOS checks
- [ ] Integration test: train for 1M steps, check >50% success on easy
- [ ] Benchmark test: verify SPS after changes

### Code Style

- Black formatting
- Type hints for all public functions
- Docstrings for classes and public methods
