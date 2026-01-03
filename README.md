# PufferLib Drone Swarm SAR

A high-throughput multi-agent Search and Rescue (SAR) benchmark where drones must **explore efficiently** and **deliver confirmations to base through a limited-range comm network**, producing emergent relay behavior.

## Features

- **Multi-agent coordination**: 8+ drones working together
- **Communication constraints**: Limited-range comm network requiring relay behavior
- **Scan/confirm/deliver mechanics**: Realistic SAR workflow
- **High throughput**: Vectorized NumPy core, PufferLib integration
- **Continuous actions**: (vx, vy, scan) control per drone
- **Optional obstacles**: Axis-aligned rectangles with collision + sensor occlusion

## Quick Start

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd PufferDroneSwarm

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Test the Environment

```bash
# Quick test with random actions
uv run python examples/main.py test

# Or directly test the env module
uv run python puffer_drone_swarm.py
```

### Raylib Demo (C)

```bash
# Build and run the C demo (requires raylib installed)
cc c_src/drone_swarm_demo.c c_src/drone_swarm.c -I c_src -o drone_swarm_demo -lraylib -lm
./drone_swarm_demo
```

Note: `visualize.py` is a legacy pygame-based viewer and is no longer part of the default dependencies.

## Model Log (GRU)

**Architecture (DroneSwarmPolicy, GRU):**

- Encoder: 3x `Linear` + `LayerNorm` + `Tanh`
- Memory: single-layer `GRU` (hidden_size=192)
- Actor: `Linear(hidden_size -> 3)` + tanh-squashed Normal policy
- Critic: `Linear(hidden_size -> 1)`
- `actor_logstd` init: -0.5

**Training config (v2.7 wilderness GRU):**

```
total_timesteps=120_000_000
num_envs=32
num_steps=128
num_minibatches=4
learning_rate=1.5e-4
gamma=0.995
gae_lambda=0.95
clip_coef=0.1
ent_coef=0.01
update_epochs=4
hidden_size=192
device=cpu
checkpoint_dir=checkpoints_v2.7_wilderness_gru
```

**Env config (hard-only wilderness):**

```
world_size=120
n_drones=10
n_victims=8
r_comm=15 (fixed)
r_sense=30
r_confirm_radius=5
t_confirm=2
m_deliver=240
victim_min/max=35-55
```

**Reward shaping:**

```
r_found=6.0
r_confirm_reward=0.2
r_explore=0.01
r_scan_near_victim=0.02
r_dispersion=0.01
r_owner_connected=0.15
r_relay_bonus=0.08
r_chain_progress=0.04
c_time=0.003
```

**Result (checkpoint_880.pt):**

```
hard (35-55): success=94.0% delivered=7.91 confirmed=7.93 len=410.3
scan_rate=87.8% detections=41.3%
```

### Train a Policy

Edit the config block at the top of `examples/train.py`, then run:

```bash
uv run python examples/train.py
```

### Evaluate a Trained Policy

Edit the config block in `examples/eval.py`, then run:

```bash
uv run python examples/eval.py
```

### Render a Policy (Raylib)

Edit the config block in `examples/render_policy.py`, then run:

```bash
uv run python examples/render_policy.py
```

## Environment Details

### Observation Space (per drone)

| Index | Feature          | Range  | Description                                       |
| ----- | ---------------- | ------ | ------------------------------------------------- |
| 0     | x position       | [0, 1] | Normalized x coordinate                           |
| 1     | y position       | [0, 1] | Normalized y coordinate                           |
| 2     | battery          | [0, 1] | Remaining battery level                           |
| 3     | distance to base | [0, 1] | Normalized distance to base station               |
| 4     | connected        | {0, 1} | Whether connected to base (directly or via relay) |
| 5     | comm age         | [0, 1] | Normalized time since last connected              |
| 6     | neighbor count   | [0, 1] | Neighbors within comm radius (normalized)         |
| 7     | min neighbor dist| [0, 1] | Distance to closest neighbor / r_comm             |

Following the base features, observations append:

- `3 * obs_n_nearest` victim detection features (dx, dy, confidence)
- `3 * obs_n_obstacles` obstacle features (dx, dy, distance) if enabled

### Action Space (per drone)

| Index | Action | Range   | Description                     |
| ----- | ------ | ------- | ------------------------------- |
| 0     | vx     | [-1, 1] | Velocity in x direction         |
| 1     | vy     | [-1, 1] | Velocity in y direction         |
| 2     | scan   | [-1, 1] | Scan action (>0 means scanning) |

### Rewards

- `+10.0` for each newly delivered victim
- `-0.1` per timestep (encourages speed)
- `-0.02 * mean(speed²)` energy cost
- `-0.01 * mean(scanning)` scan cost

### Game Mechanics

1. **Victims**: Randomly placed, must be found and confirmed
2. **Confirmation**: Drone must scan within `r_confirm_radius` for `t_confirm` consecutive steps
3. **Delivery**: Confirmed victim info delivered when confirming drone connects to base
4. **Connectivity**: Multi-hop BFS connectivity through comm network to base
5. **Battery**: Drains based on movement speed, idle cost

## Default Configuration

```python
EnvConfig(
    world_size=100.0,      # World dimensions
    n_drones=8,            # Number of drones
    n_victims=10,          # Number of victims to find
    r_comm=18.0,           # Communication radius
    r_confirm_radius=3.0,  # Confirmation radius
    r_confirm_reward=10.0, # Reward for confirming
    t_confirm=3,           # Steps to confirm
    m_deliver=10,          # Delivery window (steps)
    v_max=3.0,             # Max velocity
    max_steps=600,         # Episode length
    obstacle_count=0,      # Set >0 to generate random rectangles
    obs_n_obstacles=0,     # Set >0 to include nearest obstacles in obs
)
```

### Obstacles (optional)

You can either generate random rectangles or provide them explicitly:

```python
EnvConfig(
    obstacle_count=6,          # Random obstacles
    obstacle_min_size=8.0,
    obstacle_max_size=25.0,
    obs_n_obstacles=3,         # Include nearest obstacles in obs
)

EnvConfig(
    obstacles=[                # Explicit rectangles (x, y, w, h)
        {"x": 20.0, "y": 30.0, "w": 12.0, "h": 8.0},
        {"x": 60.0, "y": 55.0, "w": 18.0, "h": 10.0},
    ],
    obs_n_obstacles=3,
)
```

## Project Structure

```
PufferDroneSwarm/
├── env.py                 # Core environment logic
├── puffer_drone_swarm.py  # PufferLib wrapper
├── policy.py              # Model definition (GRU policy)
├── examples/              # Training/eval/render scripts
├── project.md             # Detailed project spec
└── pyproject.toml         # Dependencies
```

## Roadmap

- [x] Core environment with connectivity mechanics
- [x] PufferLib integration
- [x] PPO training script
- [ ] Visualization (matplotlib trajectories)
- [ ] Baseline algorithms (lawnmower, frontier, Voronoi)
- [ ] Evaluation suite with distribution shift tests
- [ ] Occupancy grid / belief map observations

## License

MIT
