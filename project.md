# PufferLib Drone Swarm SAR (SCCC)

## Technical + Product Outline (v1 → v3)

> **Goal:** A fast, reproducible PufferLib environment where a swarm learns **coverage/search coordination under comm constraints** (connectivity/relay behavior). Designed as a **benchmark-quality** repo (baselines + eval suite) but still fun for learning.

---

## 1) One‑sentence pitch

A high-throughput multi-agent SAR benchmark where drones must **explore efficiently** and **deliver confirmations to base through a limited-range comm network**, producing emergent relay behavior.

---

## 2) What “brain” this trains

- **Swarm dispatcher brain:** partition search space, reduce overlap, allocate roles (explore vs relay), decide when to scan/confirm, manage battery/time.
- **Connectivity-aware planning:** keep an information pipeline to base (multi-hop comm graph) while pushing the frontier.

---

## 3) Product principles

1. **Speed-first:** state-based, vectorized NumPy core, optional C acceleration later.
2. **Benchmark-first:** crisp spec, deterministic seeds, baselines, train/test distribution shift.
3. **No info leakage:** partial observability is real; belief maps are delayed/noisy.
4. **Knobs not forks:** variants are config presets; one env core.
5. **Debuggability:** simple visualization + rich logging.

---

## 4) Scope

### In-scope (v1)

- 2D continuous motion (fast kinematics) + hidden occupancy grid for mapping/exploration metrics.
- N drones, K victims, obstacles.
- Partial observability + **connectivity constraint** (comm graph to base).
- Scan/confirm mechanics.
- Baselines + eval suite.

### Out-of-scope (v1)

- Photorealism, CV images, full 3D flight dynamics.
- Full sim-to-real claims.

---

## 5) Core environment spec (v1)

### 5.1 World

- Continuous square: `world_size = L` (default 100).
- Obstacles: axis-aligned rectangles or circles.
- Hidden occupancy grid for bookkeeping: `G x G` (default 128×128).

### 5.2 Agents

- `N` drones (default 8; range 5–30).
- State per drone:

  - position `(x, y)` normalized to [0,1]
  - heading `θ` (optional if using vx/vy actions)
  - speed `v`
  - battery `b ∈ [0,1]`
  - last_comm_age (time since last connected-to-base state)

### 5.3 Victims

- `K` victims (default 10; range 5–20).
- Each victim has:

  - position `(x, y)`
  - status ∈ {hidden/unknown, detected_local, confirmed_local, delivered_global}

- **Confirm rule:** a victim is confirmed only if a drone performs `scan` while within `r_confirm_radius` for `T_confirm` consecutive steps (default 3).

### 5.4 Sensing (partial observability)

- Local sensor radius `r_sense`.
- Detection probability `p_detect(d)` decreasing with distance.
- Optional false positives rate `p_fp`.
- Obstacles affect sensing via **v1 proxy** (cheap): detection probability is reduced if local obstacle density around the ray direction is high (no exact raycasts in v1).

### 5.5 Connectivity / comms (the defining mechanic)

- Comms radius `r_comm`.
- Build comm graph each step: edge(i,j) if distance(i,j) < r_comm.
- Base station is a fixed node.
- Define `connected_to_base(i)` if i has a path to base.
- **Delivery rule (v1):** a _confirmed_ victim counts globally only if the confirming drone is connected-to-base at the moment confirmation completes **OR** becomes connected within `M_deliver` steps (store-and-forward window).
- Optional soft penalty: fraction of agents disconnected.

### 5.6 Actions

Two viable action APIs (pick one for v1 and stick with it):

- **A (simple):** continuous `(vx, vy)` clipped to `v_max` + `scan ∈ {0,1}`
- **B (more drone-like):** `(Δspeed, Δturn)` + scan

### 5.7 Step dynamics

- `pos += vel * dt`, clip/collide with obstacles and world bounds.
- Battery drain: `b -= (c_idle + c_move * (speed/v_max)^2)`.
- Termination if:

  - all victims delivered_global OR
  - max_steps reached OR
  - (optional) battery-depletion failure mode.

### 5.8 Observations (fast + learnable)

Per drone obs is a dict (StructuredPufferEnv):

- `self_state`: `[x, y, battery, (heading/speed)]`
- `local_occupancy_patch`: `(P,P)` around agent (unknown/free/obstacle_prob)
- `local_victim_belief_patch`: `(P,P)`
- `connectivity_features`: `[dist_to_base, neighbors_count, min_neighbor_dist, connected_to_base_flag, last_comm_age_norm]`
- Optional global shared map (low-res) with delay/noise: `(S,S)` occupancy + victim belief.

> **No global victim ground truth** in obs. Only belief.

### 5.9 Rewards (guardrail style)

- `+R_found` for each _new delivered_ victim (default +10).
- `-c_time` per step (default -0.1).
- `-c_energy * (speed/v_max)^2`.
- Potential-based shaping (safe): `+c_explore * Δ(explored_cells)`.
- Optional penalty: disconnected fraction each step.

### 5.10 Metrics (logged per episode)

- success (all delivered)
- time_to_first_delivered, time_to_all_delivered
- explored_area_fraction
- overlap_ratio (revisits / unique visits)
- avg_connected_fraction
- energy_per_delivered
- false_positive_count

---

## 6) Baselines (must-ship to be “useful”)

1. **Random walk** + periodic return-to-base.
2. **Lawnmower coverage** per region + fixed relay chain.
3. **Frontier exploration** with connectivity heuristic (don’t exceed hop budget).
4. **Voronoi partition** + coverage + 1–2 relay drones near base.

Provide baseline scripts that output the same metrics as RL.

---

## 7) Evaluation protocol (benchmark quality)

- Deterministic seeds.
- Train/test split on seeds.
- Distribution shift tests:

  - comm range shift (train 18–22, test 12–18)
  - obstacle density shift
  - sensor noise shift

- Report mean ± std over 5 seeds.
- Include learning curves + final tables.

---

## 8) Implementation plan

### Phase 0 (1–2 days): Repo skeleton

- Minimal package structure.
- `env.py` (core), `configs/` (yaml), `baselines/`, `train/`, `eval/`, `viz/`.
- Deterministic seeding + unit tests.

### Phase 1 (1 week): v1 env + PPO training

- Implement dynamics + occupancy bookkeeping.
- Implement victim confirm + delivery rule.
- Implement observation dict + wrappers.
- Provide CleanRL PPO example.
- Add Matplotlib viz: trajectories + explored heatmap.

### Phase 2 (1 week): Baselines + eval suite

- Implement 3–4 baselines.
- `eval.py` to run N seeds and dump metrics (CSV/JSON).
- README with quickstart.

### Phase 3 (optional): Performance pass

- Profile; optimize hotspots.
- Optional C extension for step() or occupancy updates.

---

## 9) Roadmap to “check behind the rock” (v2/v3 knobs)

### v2: Occlusion + Information Gain (no semantics)

- Add occlusion regions with cheap grid-based raycasts OR probabilistic occlusion.
- Add potential-based reward on **uncertainty reduction** (entropy over belief map).

### v3: Clue priors (semantic-ish)

- Add “terrain features” that act as **priors** (victim likely near feature).
- Keep it honest: clues only shift belief, not reveal truth.

---

## 10) README essentials (for adoption)

- One command to train PPO.
- One command to run eval suite.
- Clear config knobs table.
- Baseline results table.
- GIFs/videos from viz.

---

## 11) Clear positioning (honest + attractive)

- **Not** “deployable SAR policies.”
- **Yes:** “high-throughput benchmark for multi-agent search + comm constraints; trains a swarm coordination ‘dispatcher’ that can later be validated in PX4/Gazebo.”

---

## 12) Suggested defaults (so it works out-of-the-box)

- `L=100`, `G=128`, `N=8`, `K=10`
- `r_sense=10`, `r_confirm_radius=3`, `T_confirm=3`
- `r_comm=18`, `M_deliver=10`
- `max_steps=600`
- rewards: `R_found=10`, `c_time=0.1`, `c_energy=0.02`, `c_explore=0.5`

---

## 13) Open questions to lock quickly (no bikeshedding)

- Action API: (vx,vy) vs (Δspeed,Δturn)
- Delivery rule: must be connected at confirm-time vs within window
- Observation: include low-res shared map in v1 or keep purely local + features

(You can pick the simplest option for each; upgrade later.)
