"""Baseline policy registry."""

from __future__ import annotations

from baselines.random_walk import RandomWalkBaseline
from baselines.lawnmower import LawnmowerBaseline
from baselines.voronoi import VoronoiBaseline


BASELINES = {
    "random_walk": RandomWalkBaseline,
    "lawnmower": LawnmowerBaseline,
    "voronoi": VoronoiBaseline,
}


def list_baselines() -> list[str]:
    return list(BASELINES.keys())


def make_baseline(name: str, env_config, seed: int = 0):
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list_baselines()}")
    return BASELINES[name](env_config, seed=seed)
