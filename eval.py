"""Evaluate trained policies on alternate victim distance distributions.

Edit the config blocks below to point at models and env settings.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from env import EnvConfig
from train import DroneSwarmPolicy, ENV_CONFIG

try:
    from drone_swarm_c import CyDroneSwarm
except Exception as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "C backend (drone_swarm_c) is required. Build it with "
        "`uv run python setup.py build_ext --inplace`."
    ) from exc

# Edit these configs directly instead of passing CLI flags.
MODEL_PATHS = [
    "checkpoints/policy_final_step12.pt",
    "checkpoints/policy_final.pt",
]

EVAL_DEVICE = "cpu"
EVAL_EPISODES = 200
EVAL_SEED = 0

BASE_ENV_CONFIG = EnvConfig.from_dict({
      **ENV_CONFIG.__dict__,
      "t_confirm_values": (),
      "m_deliver_values": (),
      "obs_n_nearest": 1,
      "r_sense": 40.0,
      "r_comm_min": 0.0,
      "r_comm_max": 0.0,
      "p_comm_drop_min": 0.0,
      "p_comm_drop_max": 0.0,
  })
EVAL_VARIANTS = {
    "r_comm=16 p_drop=0.10 m_deliver=20 t_confirm=5": {
        "r_comm": 16.0,
        "p_comm_drop": 0.10,
        "m_deliver": 20,
        "t_confirm": 5,
        "victim_min_dist_from_base": 0.0,
        "victim_max_dist_from_base": 55.0,
    },
    "r_comm=16 p_drop=0.10 m_deliver=30 t_confirm=5": {
        "r_comm": 16.0,
        "p_comm_drop": 0.10,
        "m_deliver": 30,
        "t_confirm": 5,
        "victim_min_dist_from_base": 0.0,
        "victim_max_dist_from_base": 55.0,
    },
    "r_comm=16 p_drop=0.10 m_deliver=45 t_confirm=5": {
        "r_comm": 16.0,
        "p_comm_drop": 0.10,
        "m_deliver": 45,
        "t_confirm": 5,
        "victim_min_dist_from_base": 0.0,
        "victim_max_dist_from_base": 55.0,
    },
    "r_comm=20 p_drop=0.0 m_deliver=20": {
        "r_comm": 20.0,
        "p_comm_drop": 0.0,
        "m_deliver": 20,
        "victim_min_dist_from_base": 0.0,
        "victim_max_dist_from_base": 55.0,
    },
    "r_comm=20 p_drop=0.0 m_deliver=30": {
        "r_comm": 20.0,
        "p_comm_drop": 0.0,
        "m_deliver": 30,
        "victim_min_dist_from_base": 0.0,
        "victim_max_dist_from_base": 55.0,
    },
    "r_comm=20 p_drop=0.0 m_deliver=45": {
        "r_comm": 20.0,
        "p_comm_drop": 0.0,
        "m_deliver": 45,
        "victim_min_dist_from_base": 0.0,
        "victim_max_dist_from_base": 55.0,
    },
}


def with_overrides(config: EnvConfig, **overrides) -> EnvConfig:
    payload = asdict(config)
    payload.update(overrides)
    return EnvConfig(**payload)


def load_policy(checkpoint_path: str, obs_size: int, act_size: int, device: torch.device) -> DroneSwarmPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy = DroneSwarmPolicy(obs_size, act_size).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def infer_checkpoint_obs_size(checkpoint: dict) -> int | None:
    weight = checkpoint.get("model_state_dict", {}).get("encoder.0.weight")
    if weight is None:
        return None
    return int(weight.shape[1])


def evaluate(policy: DroneSwarmPolicy, env_config: EnvConfig, episodes: int, seed: int, device: torch.device, obs_size: int):
    env = CyDroneSwarm(env_config)
    delivered = []
    lengths = []
    successes = []

    rng = np.random.default_rng(seed)

    for _ in range(episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs = env.reset(seed=ep_seed)
        done = False
        steps = 0
        last_info = None

        while not done:
            obs_in = obs
            if obs.shape[1] < obs_size:
                pad_width = obs_size - obs.shape[1]
                obs_in = np.pad(obs, ((0, 0), (0, pad_width)), mode="constant")
            obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device)
            with torch.no_grad():
                hidden = policy.encoder(obs_t)
                mean = policy.actor_mean(hidden)
                logstd = policy.actor_logstd.expand_as(mean)
                std = torch.exp(logstd)
                pre_tanh = mean + std * torch.randn_like(mean)
                action = torch.tanh(pre_tanh)
            action_np = action.cpu().numpy()

            obs, _, done, info = env.step(action_np)
            steps += 1
            last_info = info

        if last_info is None:
            continue
        delivered.append(last_info.get("delivered", 0))
        lengths.append(steps)
        successes.append(1 if last_info.get("delivered", 0) == env_config.n_victims else 0)

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_delivered": float(np.mean(delivered)) if delivered else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def main() -> None:
    device = torch.device(EVAL_DEVICE)

    for model_path in MODEL_PATHS:
        path = Path(model_path)
        if not path.exists():
            print(f"Missing checkpoint: {model_path}")
            continue

        print(f"\nModel: {model_path}")

        for name, overrides in EVAL_VARIANTS.items():
            cfg = with_overrides(BASE_ENV_CONFIG, **overrides)
            checkpoint = torch.load(model_path, map_location=device)
            checkpoint_obs = infer_checkpoint_obs_size(checkpoint)
            obs_size = 10 + 3 * cfg.obs_n_nearest
            if checkpoint_obs is not None:
                obs_size = checkpoint_obs
            act_size = 3
            policy = DroneSwarmPolicy(obs_size, act_size).to(device)
            policy.load_state_dict(checkpoint["model_state_dict"])
            policy.eval()
            results = evaluate(policy, cfg, EVAL_EPISODES, EVAL_SEED, device, obs_size)

            print(
                f"{name}: success={results['success_rate']*100:.1f}% "
                f"delivered={results['mean_delivered']:.2f} "
                f"len={results['mean_episode_length']:.1f}"
            )


if __name__ == "__main__":
    main()
