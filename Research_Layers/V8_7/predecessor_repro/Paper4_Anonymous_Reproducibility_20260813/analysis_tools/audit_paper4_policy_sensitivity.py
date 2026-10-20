#!/usr/bin/env python3
"""Robustness audit for the Paper 4 archive-maintenance policy.

The experiments are synthetic policy tests. They do not estimate physical DNA
lifetimes or demonstrate maintenance benefit on a molecular archive.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from paper4_archive_maintenance_seed import (
    Config,
    exact_binomial_recoverability,
    is_action_year,
    normalize_log_weights,
    recoverability_normal_approximation,
    recoverability_probability,
    weighted_quantile,
)


@dataclass(frozen=True)
class Scenario:
    sweep: str
    policy: str
    true_hazard: float
    prior: str
    sentinel_size: int
    inspection_interval: int
    hazard_profile: str = "constant"
    observation_factor: float = 1.0


PRIORS = {
    "broad": (0.0005, 0.0200),
    "low_biased": (0.0005, 0.0080),
    "high_biased": (0.0080, 0.0200),
}


def true_dropout_probability(hazard: float, age: int, profile: str) -> float:
    if profile == "constant":
        cumulative_hazard = hazard * age
    elif profile == "accelerating":
        cumulative_hazard = hazard * age * (1.0 + 0.03 * max(age - 1, 0))
    else:
        raise ValueError(f"unknown hazard profile: {profile}")
    return 1.0 - math.exp(-cumulative_hazard)


def posterior_update_from_dropout(
    grid: np.ndarray,
    weights: list[float],
    age: int,
    unavailable: int,
    total: int,
) -> list[float]:
    log_weights = []
    for hazard, weight in zip(grid, weights):
        q = min(max(1.0 - math.exp(-float(hazard) * age), 1e-12), 1.0 - 1e-12)
        log_weights.append(
            math.log(max(weight, 1e-300))
            + unavailable * math.log(q)
            + (total - unavailable) * math.log1p(-q)
        )
    return normalize_log_weights(log_weights)


def scenario_seed(scenario_index: int, episode: int) -> int:
    return 20260713 + scenario_index * 1_000_003 + episode * 10_007


def run_episode(task: tuple[int, Scenario, int, int]) -> dict[str, object]:
    scenario_index, scenario, episode, horizon_years = task
    rng = np.random.default_rng(scenario_seed(scenario_index, episode))
    prior_low, prior_high = PRIORS[scenario.prior]
    grid = np.linspace(prior_low, prior_high, 196)
    weights = [1.0 / len(grid)] * len(grid)
    config = Config(
        horizon_years=horizon_years,
        inspection_interval=scenario.inspection_interval,
        sentinel_size=scenario.sentinel_size,
        episodes=1,
    )
    age = 0
    refreshes = 0
    inspections = 0
    violation_years = 0
    any_violation = False

    for year in range(1, horizon_years + 1):
        age += 1
        true_q = true_dropout_probability(
            scenario.true_hazard, age, scenario.hazard_profile
        )
        reliability = exact_binomial_recoverability(
            config.archive_oligos, config.erasure_tolerance, true_q
        )
        if reliability < config.recoverability_target:
            violation_years += 1
            any_violation = True

        if not is_action_year(year, horizon_years, scenario.inspection_interval):
            continue

        inspections += 1
        observed_q = min(max(true_q * scenario.observation_factor, 0.0), 1.0)
        unavailable = int(rng.binomial(scenario.sentinel_size, observed_q))
        weights = posterior_update_from_dropout(
            grid, weights, age, unavailable, scenario.sentinel_size
        )

        if scenario.policy == "posterior_mean":
            decision_hazard = sum(
                float(value) * weight for value, weight in zip(grid, weights)
            )
        elif scenario.policy == "posterior_conservative":
            decision_hazard = weighted_quantile(
                [float(value) for value in grid], weights, config.posterior_quantile
            )
        else:
            raise ValueError(f"unknown policy: {scenario.policy}")

        next_age = age + scenario.inspection_interval
        next_reliability = recoverability_probability(
            config, decision_hazard, next_age
        )
        if next_reliability < config.recoverability_target:
            refreshes += 1
            age = 0

    return {
        "sweep": scenario.sweep,
        "policy": scenario.policy,
        "true_hazard": scenario.true_hazard,
        "prior": scenario.prior,
        "sentinel_size": scenario.sentinel_size,
        "inspection_interval": scenario.inspection_interval,
        "hazard_profile": scenario.hazard_profile,
        "observation_factor": scenario.observation_factor,
        "episode": episode,
        "refreshes": refreshes,
        "inspections": inspections,
        "violation_years": violation_years,
        "any_violation": int(any_violation),
    }


def summarize(rows: list[dict[str, object]], horizon_years: int) -> list[dict[str, object]]:
    keys = (
        "sweep",
        "policy",
        "true_hazard",
        "prior",
        "sentinel_size",
        "inspection_interval",
        "hazard_profile",
        "observation_factor",
    )
    rows.sort(key=lambda row: tuple(row[key] for key in keys) + (row["episode"],))
    summaries = []
    for group_key, group_iter in itertools.groupby(rows, key=lambda row: tuple(row[key] for key in keys)):
        group = list(group_iter)
        summaries.append(
            {
                **dict(zip(keys, group_key)),
                "episodes": len(group),
                "mean_refreshes": sum(int(row["refreshes"]) for row in group) / len(group),
                "violation_rate": sum(int(row["violation_years"]) for row in group)
                / (len(group) * horizon_years),
                "episode_failure_rate": sum(int(row["any_violation"]) for row in group)
                / len(group),
            }
        )
    return summaries


def approximation_audit() -> list[dict[str, object]]:
    config = Config()
    rows = []
    for hazard in (0.0005, 0.002, 0.006, 0.012, 0.020):
        for age in range(1, config.horizon_years + 1):
            exact = recoverability_probability(config, hazard, age)
            approximate = recoverability_normal_approximation(config, hazard, age)
            rows.append(
                {
                    "hazard": hazard,
                    "age": age,
                    "exact": exact,
                    "normal_approximation": approximate,
                    "absolute_error": abs(exact - approximate),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_scenarios() -> list[Scenario]:
    design = [
        Scenario("design", "posterior_conservative", hazard, prior, sentinel, interval)
        for hazard, prior, sentinel, interval in itertools.product(
            (0.006, 0.012), PRIORS, (100, 300, 1000), (2, 5, 10)
        )
    ]
    misspecification = [
        Scenario(
            "misspecification",
            policy,
            hazard,
            "broad",
            300,
            5,
            profile,
            observation_factor,
        )
        for policy, hazard, profile, observation_factor in itertools.product(
            ("posterior_mean", "posterior_conservative"),
            (0.006, 0.012),
            ("constant", "accelerating"),
            (0.8, 1.0, 1.2),
        )
    ]
    return design + misspecification


def self_test() -> None:
    scenarios = build_scenarios()
    assert len(scenarios) == 78
    first = run_episode((0, scenarios[0], 0, 20))
    second = run_episode((0, scenarios[0], 0, 20))
    assert first == second
    assert true_dropout_probability(0.006, 10, "accelerating") > true_dropout_probability(0.006, 10, "constant")
    audit = approximation_audit()
    assert len(audit) == 300
    assert max(float(row["absolute_error"]) for row in audit) < 0.006
    print("PASS_PAPER4_POLICY_SENSITIVITY_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--horizon-years", type=int, default=60)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    scenarios = build_scenarios()
    tasks = [
        (index, scenario, episode, args.horizon_years)
        for index, scenario in enumerate(scenarios)
        for episode in range(args.episodes)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_episode, tasks, chunksize=32))

    summaries = summarize(rows, args.horizon_years)
    approximation_rows = approximation_audit()
    write_csv(args.output_dir / "paper4_policy_sensitivity_episodes.csv", rows)
    write_csv(args.output_dir / "paper4_policy_sensitivity_summary.csv", summaries)
    write_csv(args.output_dir / "paper4_binomial_approximation_audit.csv", approximation_rows)

    max_error = max(float(row["absolute_error"]) for row in approximation_rows)
    print("PASS_PAPER4_POLICY_SENSITIVITY")
    print(f"scenarios={len(scenarios)} episodes_per_scenario={args.episodes} rows={len(rows)}")
    print(f"exact_vs_normal_max_absolute_error={max_error:.9f}")


if __name__ == "__main__":
    main()
