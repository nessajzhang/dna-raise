#!/usr/bin/env python3
"""Synthetic seed benchmark for uncertainty-aware DNA archive maintenance.

This model is a policy-logic test, not an empirical DNA longevity model.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from scipy.stats import binom


@dataclass(frozen=True)
class Config:
    horizon_years: int = 60
    inspection_interval: int = 5
    fixed_refresh_interval: int = 15
    sentinel_size: int = 300
    archive_oligos: int = 10_000
    erasure_tolerance: float = 0.10
    recoverability_target: float = 0.99
    posterior_quantile: float = 0.95
    episodes: int = 400
    seed: int = 20260711


HAZARDS = (0.002, 0.006, 0.012)
POLICIES = (
    "fixed_schedule",
    "oracle_true_hazard",
    "posterior_mean",
    "posterior_conservative",
)


def is_action_year(year: int, horizon_years: int, interval: int) -> bool:
    """Return whether a policy action is scheduled strictly before the horizon."""
    return year % interval == 0 and year < horizon_years


def dropout_probability(hazard: float, age: float) -> float:
    return 1.0 - math.exp(-hazard * age)


@lru_cache(maxsize=65_536)
def exact_binomial_recoverability(
    archive_oligos: int, erasure_tolerance: float, dropout: float
) -> float:
    cutoff = math.floor(archive_oligos * erasure_tolerance)
    return float(binom.cdf(cutoff, archive_oligos, dropout))


def recoverability_probability(config: Config, hazard: float, age: float) -> float:
    """Exact P[dropouts <= outer-code tolerance] under the seed model."""
    q = dropout_probability(hazard, age)
    return exact_binomial_recoverability(
        config.archive_oligos, config.erasure_tolerance, q
    )


def recoverability_normal_approximation(
    config: Config, hazard: float, age: float
) -> float:
    """Continuity-corrected approximation retained for an explicit error audit."""
    q = dropout_probability(hazard, age)
    mean = config.archive_oligos * q
    variance = config.archive_oligos * q * (1.0 - q)
    cutoff = math.floor(config.archive_oligos * config.erasure_tolerance)
    if variance <= 0.0:
        return 1.0 if mean <= cutoff else 0.0
    z = (cutoff + 0.5 - mean) / math.sqrt(variance)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def binomial_draw(rng: random.Random, n: int, p: float) -> int:
    return sum(rng.random() < p for _ in range(n))


def normalize_log_weights(log_weights: list[float]) -> list[float]:
    peak = max(log_weights)
    weights = [math.exp(value - peak) for value in log_weights]
    total = sum(weights)
    return [value / total for value in weights]


def posterior_update(
    grid: list[float], weights: list[float], age: float, unavailable: int, total: int
) -> list[float]:
    updated = []
    for hazard, weight in zip(grid, weights):
        q = min(max(dropout_probability(hazard, age), 1e-12), 1.0 - 1e-12)
        updated.append(
            math.log(max(weight, 1e-300))
            + unavailable * math.log(q)
            + (total - unavailable) * math.log1p(-q)
        )
    return normalize_log_weights(updated)


def weighted_quantile(grid: list[float], weights: list[float], quantile: float) -> float:
    cumulative = 0.0
    for value, weight in zip(grid, weights):
        cumulative += weight
        if cumulative >= quantile:
            return value
    return grid[-1]


def policy_hazard(policy: str, grid: list[float], weights: list[float], config: Config) -> float:
    if policy == "posterior_mean":
        return sum(value * weight for value, weight in zip(grid, weights))
    if policy == "posterior_conservative":
        return weighted_quantile(grid, weights, config.posterior_quantile)
    raise ValueError(f"No posterior hazard for policy {policy}")


def run_episode(config: Config, true_hazard: float, policy: str, episode: int) -> dict[str, float | int | str]:
    rng = random.Random(config.seed + episode * 10_007 + int(true_hazard * 1_000_000))
    grid = [0.0005 + index * (0.0195 / 195) for index in range(196)]
    weights = [1.0 / len(grid)] * len(grid)
    age = 0
    refreshes = 0
    inspections = 0
    violation_years = 0
    any_violation = False

    for year in range(1, config.horizon_years + 1):
        age += 1
        reliability = recoverability_probability(config, true_hazard, age)
        if reliability < config.recoverability_target:
            violation_years += 1
            any_violation = True

        if not is_action_year(
            year, config.horizon_years, config.inspection_interval
        ):
            continue

        inspections += 1
        refresh = False
        if policy == "fixed_schedule":
            refresh = age >= config.fixed_refresh_interval
        elif policy == "oracle_true_hazard":
            next_age = age + config.inspection_interval
            next_reliability = recoverability_probability(
                config, true_hazard, next_age
            )
            refresh = next_reliability < config.recoverability_target
        else:
            q = dropout_probability(true_hazard, age)
            unavailable = binomial_draw(rng, config.sentinel_size, q)
            if age > 0:
                weights = posterior_update(
                    grid, weights, age, unavailable, config.sentinel_size
                )
            decision_hazard = policy_hazard(policy, grid, weights, config)
            next_age = age + config.inspection_interval
            next_reliability = recoverability_probability(config, decision_hazard, next_age)
            refresh = next_reliability < config.recoverability_target

        if refresh:
            refreshes += 1
            age = 0

    return {
        "policy": policy,
        "true_hazard": true_hazard,
        "episode": episode,
        "refreshes": refreshes,
        "inspections": inspections,
        "violation_years": violation_years,
        "any_violation": int(any_violation),
    }


def summarize(rows: list[dict[str, float | int | str]], config: Config) -> list[dict[str, float | int | str]]:
    summaries = []
    for hazard in HAZARDS:
        for policy in POLICIES:
            group = [row for row in rows if row["true_hazard"] == hazard and row["policy"] == policy]
            summaries.append(
                {
                    "policy": policy,
                    "true_hazard": hazard,
                    "episodes": len(group),
                    "mean_refreshes": sum(int(row["refreshes"]) for row in group) / len(group),
                    "mean_inspections": sum(int(row["inspections"]) for row in group) / len(group),
                    "violation_rate": sum(int(row["violation_years"]) for row in group)
                    / (len(group) * config.horizon_years),
                    "episode_failure_rate": sum(int(row["any_violation"]) for row in group) / len(group),
                }
            )
    return summaries


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    config = Config(episodes=2)
    assert dropout_probability(0.01, 20) > dropout_probability(0.01, 10)
    assert recoverability_probability(config, 0.01, 5) > recoverability_probability(config, 0.01, 15)
    for hazard in HAZARDS:
        for age in range(1, config.horizon_years + 1):
            exact = recoverability_probability(config, hazard, age)
            approximate = recoverability_normal_approximation(config, hazard, age)
            assert abs(exact - approximate) < 0.006
    grid = [0.001, 0.005, 0.01]
    weights = posterior_update(grid, [1 / 3] * 3, age=10, unavailable=20, total=300)
    assert abs(sum(weights) - 1.0) < 1e-12
    quantile = weighted_quantile(grid, weights, config.posterior_quantile)
    covered_mass = sum(weight for value, weight in zip(grid, weights) if value <= quantile)
    threshold = recoverability_probability(config, quantile, age=15)
    safe_mass = sum(
        weight
        for value, weight in zip(grid, weights)
        if recoverability_probability(config, value, age=15) >= threshold
    )
    assert covered_mass >= config.posterior_quantile
    assert safe_mass >= config.posterior_quantile
    first = run_episode(config, 0.006, "posterior_conservative", 0)
    second = run_episode(config, 0.006, "posterior_conservative", 0)
    assert first == second
    assert sum(
        is_action_year(year, config.horizon_years, config.inspection_interval)
        for year in range(1, config.horizon_years + 1)
    ) == 11
    for hazard in HAZARDS:
        oracle = run_episode(config, hazard, "oracle_true_hazard", 0)
        assert oracle["violation_years"] == 0
    print("PASS_PAPER4_ARCHIVE_MAINTENANCE_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    config = Config(episodes=args.episodes)

    rows = [
        run_episode(config, hazard, policy, episode)
        for hazard in HAZARDS
        for policy in POLICIES
        for episode in range(config.episodes)
    ]
    summaries = summarize(rows, config)
    write_csv(args.output_dir / "paper4_archive_maintenance_seed_episodes.csv", rows)
    write_csv(args.output_dir / "paper4_archive_maintenance_seed_summary.csv", summaries)

    print("PASS_PAPER4_ARCHIVE_MAINTENANCE_SEED")
    for row in summaries:
        print(
            f"{row['true_hazard']:.3f} {row['policy']}: "
            f"refreshes={row['mean_refreshes']:.3f}, "
            f"violation_rate={row['violation_rate']:.6f}, "
            f"episode_failure_rate={row['episode_failure_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
