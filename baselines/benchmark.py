"""Benchmark all baseline policies.

Edit BENCHMARK_* variables instead of passing CLI args.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from env import EnvConfig
from train import ENV_CONFIG
from baselines import list_baselines, make_baseline

try:
    from drone_swarm_c import CyDroneSwarm
except Exception as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "C backend (drone_swarm_c) is required. Build it with "
        "`uv run python setup.py build_ext --inplace`."
    ) from exc

# Edit these configs directly instead of passing CLI flags.
BENCHMARK_EPISODES = 100
BENCHMARK_SEED = 0
BASELINE_NAMES = list_baselines()

BASE_ENV_CONFIG = EnvConfig.from_dict({
    **ENV_CONFIG.__dict__,
    "n_drones": 4,
    "n_victims": 6,
    "r_comm": 18.0,
    "r_comm_min": 0.0,
    "r_comm_max": 0.0,
    "p_comm_drop": 0.0,
    "p_comm_drop_min": 0.0,
    "p_comm_drop_max": 0.0,
    "t_confirm": 1,
    "t_confirm_values": (),
    "m_deliver": 120,
    "m_deliver_values": (),
    "r_sense": 80.0,
    "detect_prob_scale": 2.0,
    "detect_noise_std": 0.0,
    "false_positive_rate": 0.0,
    "false_positive_confidence": 0.3,
    "victim_min_dist_from_base": 10.0,
    "victim_max_dist_from_base": 35.0,
})


def with_overrides(config: EnvConfig, **overrides) -> EnvConfig:
    payload = asdict(config)
    payload.update(overrides)
    return EnvConfig(**payload)


def evaluate_baseline(policy, env_config: EnvConfig, episodes: int, seed: int):
    env = CyDroneSwarm(env_config)
    rng = np.random.default_rng(seed)

    delivered = []
    lengths = []
    successes = []

    for _ in range(episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs = env.reset(seed=ep_seed)
        policy.reset(obs=obs, seed=ep_seed)
        done = False
        steps = 0
        last_info = None

        while not done:
            actions = policy.get_action(obs)
            obs, _, done, info = env.step(actions)
            steps += 1
            last_info = info

        if last_info is None:
            continue
        delivered_count = last_info.get("delivered", 0)
        delivered.append(delivered_count)
        lengths.append(steps)
        successes.append(1 if delivered_count == env_config.n_victims else 0)

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_delivered": float(np.mean(delivered)) if delivered else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def format_name(name: str) -> str:
    return name.replace("_", " ").title()


def main() -> None:
    results = []

    for name in BASELINE_NAMES:
        policy = make_baseline(name, BASE_ENV_CONFIG, seed=BENCHMARK_SEED)
        stats = evaluate_baseline(policy, BASE_ENV_CONFIG, BENCHMARK_EPISODES, BENCHMARK_SEED)
        results.append((name, stats))

    print("| Policy | Success % | Avg Steps | Delivered/Ep |")
    print("|---|---|---|---|")
    for name, stats in results:
        print(
            f"| {format_name(name)} | {stats['success_rate']*100:.1f}% | "
            f"{stats['mean_episode_length']:.0f} | {stats['mean_delivered']:.2f} |"
        )


if __name__ == "__main__":
    main()
