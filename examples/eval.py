"""Evaluate trained policies on alternate victim distance distributions.

Edit the config blocks below to point at models and env settings.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env import EnvConfig
from policy import DroneSwarmPolicy
from configs import get_env_config

try:
    from drone_swarm_c import CyDroneSwarm
except Exception as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "C backend (drone_swarm_c) is required. Build it with "
        "`uv run python setup.py build_ext --inplace`."
    ) from exc

# Edit these configs directly instead of passing CLI flags.
MODEL_PATHS = [
    "checkpoints_v2.7_wilderness_gru/checkpoint_160.pt",
]

EVAL_DEVICE = "cpu"
EVAL_EPISODES = 100
EVAL_SEED = 0
BASELINE_NAME = None  # Run all baselines to test v2 difficulty

PRESET_NAME = "wilderness_hard_obstacles_stage2"

BASE_ENV_CONFIG = EnvConfig.from_dict({
    **get_env_config(PRESET_NAME).__dict__,
    # Ensure no randomization during eval
    "r_comm_min": 0.0,
    "r_comm_max": 0.0,
    "p_comm_drop_min": 0.0,
    "p_comm_drop_max": 0.0,
    "t_confirm_values": (),
    "m_deliver_values": (),
    "victim_mix_prob": 0.0,
})

EVAL_VARIANTS = {
    "hard (35-55)": {},
}


def with_overrides(config: EnvConfig, **overrides) -> EnvConfig:
    payload = asdict(config)
    payload.update(overrides)
    return EnvConfig(**payload)


def infer_checkpoint_hidden_size(checkpoint: dict) -> int | None:
    weight = checkpoint.get("model_state_dict", {}).get("encoder.0.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


def load_policy(checkpoint_path: str, obs_size: int, act_size: int, device: torch.device) -> DroneSwarmPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_size = infer_checkpoint_hidden_size(checkpoint) or 256
    policy = DroneSwarmPolicy(obs_size, act_size, hidden_size=hidden_size).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def infer_checkpoint_obs_size(checkpoint: dict) -> int | None:
    weight = checkpoint.get("model_state_dict", {}).get("encoder.0.weight")
    if weight is None:
        return None
    return int(weight.shape[1])


def evaluate(
    policy: DroneSwarmPolicy,
    env_config: EnvConfig,
    episodes: int,
    seed: int,
    device: torch.device,
    obs_size: int,
):
    env = CyDroneSwarm(env_config)
    delivered = []
    lengths = []
    successes = []
    confirmed = []
    scan_rates = []
    detection_rates = []

    rng = np.random.default_rng(seed)

    for _ in range(episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs = env.reset(seed=ep_seed)
        done = False
        steps = 0
        last_info = None
        total_agent_steps = 0
        scan_steps = 0
        detect_steps = 0
        h = policy.init_hidden(env_config.n_drones, device)
        done_mask = torch.zeros(env_config.n_drones, device=device)

        while not done:
            obs_in = obs
            if obs.shape[1] < obs_size:
                pad_width = obs_size - obs.shape[1]
                obs_in = np.pad(obs, ((0, 0), (0, pad_width)), mode="constant")
            elif obs.shape[1] > obs_size:
                obs_in = obs[:, :obs_size]
            obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _, _, h = policy.get_action_and_value(obs_t, h, done_mask)
            action_np = action.cpu().numpy()

            scan_mask = action_np[:, 2] > 0.0
            scan_steps += int(np.sum(scan_mask))
            total_agent_steps += scan_mask.size

            obs, _, done, info = env.step(action_np)
            done_mask = torch.full((env_config.n_drones,), float(done), device=device)
            if env_config.obs_n_nearest > 0:
                detect_block = obs[:, 10:10 + 3 * env_config.obs_n_nearest]
                detect_nonzero = np.any(detect_block != 0.0, axis=1)
                detect_steps += int(np.sum(detect_nonzero))
            steps += 1
            last_info = info

        if last_info is None:
            continue
        delivered.append(last_info.get("delivered", 0))
        confirmed_total = last_info.get("confirmed", 0) + last_info.get("delivered", 0)
        confirmed.append(confirmed_total)
        lengths.append(steps)
        successes.append(1 if last_info.get("delivered", 0) == env_config.n_victims else 0)
        if total_agent_steps > 0:
            scan_rates.append(scan_steps / total_agent_steps)
            detection_rates.append(detect_steps / total_agent_steps)
        else:
            scan_rates.append(0.0)
            detection_rates.append(0.0)

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_delivered": float(np.mean(delivered)) if delivered else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_confirmed": float(np.mean(confirmed)) if confirmed else 0.0,
        "scan_rate": float(np.mean(scan_rates)) if scan_rates else 0.0,
        "detections_nonzero_rate": float(np.mean(detection_rates)) if detection_rates else 0.0,
    }


def evaluate_baseline(
    policy,
    env_config: EnvConfig,
    episodes: int,
    seed: int,
):
    env = CyDroneSwarm(env_config)
    delivered = []
    lengths = []
    successes = []
    confirmed = []
    scan_rates = []
    detection_rates = []

    rng = np.random.default_rng(seed)

    for _ in range(episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs = env.reset(seed=ep_seed)
        policy.reset(obs=obs, seed=ep_seed)
        done = False
        steps = 0
        last_info = None
        total_agent_steps = 0
        scan_steps = 0
        detect_steps = 0

        while not done:
            action_np = policy.get_action(obs)

            scan_mask = action_np[:, 2] > 0.0
            scan_steps += int(np.sum(scan_mask))
            total_agent_steps += scan_mask.size

            obs, _, done, info = env.step(action_np)
            if env_config.obs_n_nearest > 0:
                detect_block = obs[:, 10:10 + 3 * env_config.obs_n_nearest]
                detect_nonzero = np.any(detect_block != 0.0, axis=1)
                detect_steps += int(np.sum(detect_nonzero))
            steps += 1
            last_info = info

        if last_info is None:
            continue
        delivered.append(last_info.get("delivered", 0))
        confirmed_total = last_info.get("confirmed", 0) + last_info.get("delivered", 0)
        confirmed.append(confirmed_total)
        lengths.append(steps)
        successes.append(1 if last_info.get("delivered", 0) == env_config.n_victims else 0)
        if total_agent_steps > 0:
            scan_rates.append(scan_steps / total_agent_steps)
            detection_rates.append(detect_steps / total_agent_steps)
        else:
            scan_rates.append(0.0)
            detection_rates.append(0.0)

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_delivered": float(np.mean(delivered)) if delivered else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_confirmed": float(np.mean(confirmed)) if confirmed else 0.0,
        "scan_rate": float(np.mean(scan_rates)) if scan_rates else 0.0,
        "detections_nonzero_rate": float(np.mean(detection_rates)) if detection_rates else 0.0,
    }


def main() -> None:
    if BASELINE_NAME:
        from baselines import list_baselines, make_baseline

        base_cfg = BASE_ENV_CONFIG
        baseline_names = list_baselines() if BASELINE_NAME == "all" else [BASELINE_NAME]

        for baseline_name in baseline_names:
            print(f"\nBaseline: {baseline_name}")
            for name, overrides in EVAL_VARIANTS.items():
                cfg = with_overrides(base_cfg, **overrides)
                policy = make_baseline(baseline_name, cfg, seed=EVAL_SEED)
                results = evaluate_baseline(policy, cfg, EVAL_EPISODES, EVAL_SEED)
                print(
                    f"{name}: success={results['success_rate']*100:.1f}% "
                    f"delivered={results['mean_delivered']:.2f} "
                    f"confirmed={results['mean_confirmed']:.2f} "
                    f"len={results['mean_episode_length']:.1f} "
                    f"scan_rate={results['scan_rate']*100:.1f}% "
                    f"detections={results['detections_nonzero_rate']*100:.1f}%"
                )
        return

    device = torch.device(EVAL_DEVICE)

    for model_path in MODEL_PATHS:
        path = Path(model_path)
        if not path.exists():
            print(f"Missing checkpoint: {model_path}")
            continue

        print(f"\nModel: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)
        checkpoint_cfg = checkpoint.get("env_config") or {}
        base_cfg = BASE_ENV_CONFIG
        if checkpoint_cfg:
            base_cfg = EnvConfig.from_dict(checkpoint_cfg)
            base_cfg = with_overrides(
                base_cfg,
                t_confirm_values=(),
                m_deliver_values=(),
                r_comm_min=0.0,
                r_comm_max=0.0,
                p_comm_drop_min=0.0,
                p_comm_drop_max=0.0,
            )

        checkpoint_obs = infer_checkpoint_obs_size(checkpoint)
        obs_size = 10 + 3 * base_cfg.obs_n_nearest + 3 * base_cfg.obs_n_obstacles
        if checkpoint_obs is not None:
            obs_size = checkpoint_obs
        act_size = 3
        hidden_size = infer_checkpoint_hidden_size(checkpoint) or 256
        policy = DroneSwarmPolicy(obs_size, act_size, hidden_size=hidden_size).to(device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()

        for name, overrides in EVAL_VARIANTS.items():
            cfg = with_overrides(base_cfg, **overrides)
            results = evaluate(policy, cfg, EVAL_EPISODES, EVAL_SEED, device, obs_size)

            print(
                f"{name}: success={results['success_rate']*100:.1f}% "
                f"delivered={results['mean_delivered']:.2f} "
                f"confirmed={results['mean_confirmed']:.2f} "
                f"len={results['mean_episode_length']:.1f} "
                f"scan_rate={results['scan_rate']*100:.1f}% "
                f"detections={results['detections_nonzero_rate']*100:.1f}%"
            )


if __name__ == "__main__":
    main()
