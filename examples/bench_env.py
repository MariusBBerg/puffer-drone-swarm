"""Benchmark raw environment throughput (env-steps/sec).

Runs the C backend if available, otherwise falls back to the Python env.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env_config import EnvConfig


def make_env(config: EnvConfig):
    try:
        from drone_swarm_c import CyDroneSwarm
    except Exception as exc:
        raise ImportError(
            "C backend (drone_swarm_c) is required. Build it with "
            "`uv run python setup.py build_ext --inplace`."
        ) from exc
    return CyDroneSwarm(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DroneSwarm env steps/sec")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--n-drones", type=int, default=8)
    parser.add_argument("--n-victims", type=int, default=10)
    parser.add_argument("--obs-n-nearest", type=int, default=3)
    parser.add_argument("--fast", action="store_true", help="Disable expensive reward/obs terms")
    args = parser.parse_args()

    config = EnvConfig(
        n_drones=args.n_drones,
        n_victims=args.n_victims,
        obs_n_nearest=args.obs_n_nearest,
    )

    if args.fast:
        config = EnvConfig(
            n_drones=args.n_drones,
            n_victims=args.n_victims,
            obs_n_nearest=0,
            r_explore=0.0,
            r_scan_near_victim=0.0,
            r_dispersion=0.0,
            r_connectivity=0.0,
            c_scan=0.0,
            c_energy=0.0,
            r_owner_connected=0.0,
        )

    env = make_env(config)
    obs = env.reset(seed=0)
    actions = np.random.uniform(-1.0, 1.0, size=(config.n_drones, 3)).astype(np.float32)

    start = time.time()
    steps = int(args.steps)
    for _ in range(steps):
        obs, rewards, done, info = env.step(actions)
        if done:
            env.reset(seed=None)
    elapsed = time.time() - start

    sps = int(steps / elapsed) if elapsed > 0 else 0
    backend = "C" if env.__class__.__name__ == "CyDroneSwarm" else "Python"
    print(f"Backend: {backend}")
    print(f"Env steps/sec: {sps}")


if __name__ == "__main__":
    main()
