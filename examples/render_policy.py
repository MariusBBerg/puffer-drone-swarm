"""Raylib visualization for a trained policy.

Edit the CONFIG section below instead of passing CLI flags.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env_config import EnvConfig
from puffer_drone_swarm import PufferDroneSwarm
from policy import DroneSwarmPolicy


# =============================
# CONFIG (edit these)
# =============================
CHECKPOINT_PATH = "checkpoints_v2.7_wilderness_gru/checkpoint_160.pt"
DEVICE = "cpu"
SEED = 0
EPISODES = None  # None = run forever
FORCE_NO_OBSTACLES = False
RENDER_EVERY_STEP = True
SLEEP_SECONDS = 0.05  # Set >0 for slower playback


def _load_env_config(checkpoint: dict) -> EnvConfig:
    cfg_dict = checkpoint.get("env_config", {})
    cfg = EnvConfig.from_dict(cfg_dict) if cfg_dict else EnvConfig()
    if FORCE_NO_OBSTACLES:
        cfg.obstacle_count = 0
        cfg.obs_n_obstacles = 0
        cfg.obstacles = ()
    return cfg


def _align_obs(obs: np.ndarray, target_size: int) -> np.ndarray:
    if obs.shape[1] == target_size:
        return obs
    if obs.shape[1] < target_size:
        pad = target_size - obs.shape[1]
        return np.pad(obs, ((0, 0), (0, pad)), mode="constant")
    return obs[:, :target_size]


def main() -> None:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    weight = checkpoint.get("model_state_dict", {}).get("encoder.0.weight")
    if weight is None:
        raise RuntimeError("Checkpoint missing encoder.0.weight (unexpected model format).")
    hidden_size = int(weight.shape[0])
    obs_size = int(weight.shape[1])

    cfg = _load_env_config(checkpoint)
    env = PufferDroneSwarm(config=cfg, render_mode="human", num_envs=1)

    policy = DroneSwarmPolicy(obs_size, 3, hidden_size=hidden_size).to(DEVICE)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    episode = 0
    obs, _ = env.reset(seed=SEED)
    obs_t = torch.tensor(_align_obs(obs, obs_size), dtype=torch.float32, device=DEVICE)
    h = policy.init_hidden(env.num_agents, DEVICE)
    done_mask = torch.zeros(env.num_agents, device=DEVICE)

    while EPISODES is None or episode < EPISODES:
        with torch.no_grad():
            action, _, _, _, h = policy.get_action_and_value(obs_t, h, done_mask)
        action_np = action.cpu().numpy()

        obs, rewards, terminals, truncations, infos = env.step(action_np)
        obs_t = torch.tensor(_align_obs(obs, obs_size), dtype=torch.float32, device=DEVICE)

        if RENDER_EVERY_STEP:
            env.render()
        if SLEEP_SECONDS > 0.0:
            time.sleep(SLEEP_SECONDS)

        done = (terminals | truncations).any()
        done_mask = torch.full((env.num_agents,), float(done), device=DEVICE)
        if done:
            episode += 1
            obs, _ = env.reset()
            obs_t = torch.tensor(_align_obs(obs, obs_size), dtype=torch.float32, device=DEVICE)
            h = policy.init_hidden(env.num_agents, DEVICE)
            done_mask.zero_()


if __name__ == "__main__":
    main()
