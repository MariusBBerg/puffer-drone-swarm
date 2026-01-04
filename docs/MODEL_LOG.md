# Model Log

## v2.7 wilderness GRU (checkpoint_880.pt)

**Architecture**

- Encoder: 3x `Linear` + `LayerNorm` + `Tanh`
- Memory: single-layer `GRU` (hidden_size=192)
- Actor: `Linear(hidden_size -> 3)` with tanh-squashed Normal policy
- Critic: `Linear(hidden_size -> 1)`
- `actor_logstd` init: -0.5

**Training config**

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

**Env config (hard-only wilderness)**

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

**Reward shaping**

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

**Result (checkpoint_880.pt)**

```
hard (35-55): success=94.0% delivered=7.91 confirmed=7.93 len=410.3
scan_rate=87.8% detections=41.3%
```
