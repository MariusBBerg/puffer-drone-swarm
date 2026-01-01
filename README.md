# PufferLib Drone Swarm SAR

A high-throughput multi-agent Search and Rescue (SAR) benchmark where drones must **explore efficiently** and **deliver confirmations to base through a limited-range comm network**, producing emergent relay behavior.

## Features

- **Multi-agent coordination**: 8+ drones working together
- **Communication constraints**: Limited-range comm network requiring relay behavior
- **Scan/confirm/deliver mechanics**: Realistic SAR workflow
- **High throughput**: Vectorized NumPy core, PufferLib integration
- **Continuous actions**: (vx, vy, scan) control per drone

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
uv run python main.py test --steps 100

# Or directly test the env module
uv run python puffer_drone_swarm.py
```

### Train a Policy

```bash
# Default training (10M steps)
uv run python main.py train

# Quick training run
uv run python train.py --total-timesteps 1000000 --num-envs 4

# With custom settings
uv run python train.py \
    --n-drones 8 \
    --n-victims 10 \
    --total-timesteps 10000000 \
    --num-envs 8 \
    --learning-rate 3e-4 \
    --checkpoint-dir checkpoints
```

### Evaluate a Trained Policy

```bash
uv run python main.py eval --checkpoint checkpoints/policy_final.pt --episodes 10
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
)
```

## Project Structure

```
PufferDroneSwarm/
├── env.py                 # Core environment logic
├── puffer_drone_swarm.py  # PufferLib wrapper
├── train.py               # PPO training script
├── main.py                # CLI entry point
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
