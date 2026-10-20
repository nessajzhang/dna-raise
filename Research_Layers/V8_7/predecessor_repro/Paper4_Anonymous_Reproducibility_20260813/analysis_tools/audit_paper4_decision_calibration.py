#!/usr/bin/env python3
"""Audit the model-internal certificate and empirical false-safe decisions.

The calibrated scenario is prior-predictive: the true hazard is sampled from
the same discrete prior used by the controller. Misspecified scenarios are
stress tests, not fitted descriptions of molecular degradation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paper4_archive_maintenance_seed import (
    Config,
    exact_binomial_recoverability,
    is_action_year,
    recoverability_probability,
)


ROOT = Path(__file__).resolve().parents[1]
PRIORS = {
    "broad": (0.0005, 0.0200),
    "low_biased": (0.0005, 0.0080),
}
Z_975 = 1.959963984540054
CLUSTER_BOOTSTRAP_REPLICATES = 10_000
CLUSTER_BOOTSTRAP_BASE_SEED = 20260824


@dataclass(frozen=True)
class CalibrationScenario:
    name: str
    prior: str
    true_hazard: float | None
    sentinel_size: int = 300
    inspection_interval: int = 5
    hazard_profile: str = "constant"
    observation_factor: float = 1.0


SCENARIOS = (
    CalibrationScenario("calibrated_prior_predictive", "broad", None),
    CalibrationScenario("calibrated_fixed_hazard", "broad", 0.006),
    CalibrationScenario("prior_support_exclusion", "low_biased", 0.012),
    CalibrationScenario(
        "biased_accelerating_hazard",
        "broad",
        0.006,
        hazard_profile="accelerating",
        observation_factor=0.8,
    ),
    CalibrationScenario(
        "ten_year_inspection",
        "broad",
        0.012,
        inspection_interval=10,
    ),
)


def cumulative_dropout(hazard: float, age: int, profile: str) -> float:
    multiplier = 1.0 if profile == "constant" else 1.0 + 0.03 * max(age - 1, 0)
    return 1.0 - math.exp(-hazard * age * multiplier)


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / weights.sum()


def posterior_update(
    grid: np.ndarray,
    weights: np.ndarray,
    age: int,
    unavailable: int,
    total: int,
) -> np.ndarray:
    q = np.clip(1.0 - np.exp(-grid * age), 1e-12, 1.0 - 1e-12)
    log_weights = (
        np.log(np.clip(weights, 1e-300, None))
        + unavailable * np.log(q)
        + (total - unavailable) * np.log1p(-q)
    )
    return normalize_log_weights(log_weights)


def quantile_index(weights: np.ndarray, quantile: float) -> int:
    return int(np.searchsorted(np.cumsum(weights), quantile, side="left"))


def scenario_seed(scenario_index: int, episode: int) -> int:
    return 20260718 + scenario_index * 10_000_019 + episode * 10_007


def run_episode(task: tuple[int, CalibrationScenario, int, int]) -> dict[str, object]:
    scenario_index, scenario, episode, horizon_years = task
    rng = np.random.default_rng(scenario_seed(scenario_index, episode))
    prior_low, prior_high = PRIORS[scenario.prior]
    grid = np.linspace(prior_low, prior_high, 196)
    weights = np.full(grid.size, 1.0 / grid.size)
    true_hazard = (
        float(grid[int(rng.integers(0, grid.size))])
        if scenario.true_hazard is None
        else scenario.true_hazard
    )
    config = Config(
        horizon_years=horizon_years,
        inspection_interval=scenario.inspection_interval,
        sentinel_size=scenario.sentinel_size,
        episodes=1,
    )
    age = 0
    refreshes = 0
    violations = 0
    decisions = 0
    no_refresh_decisions = 0
    false_safe_decisions = 0
    minimum_certificate_mass = 1.0

    for year in range(1, horizon_years + 1):
        age += 1
        true_q = cumulative_dropout(true_hazard, age, scenario.hazard_profile)
        if (
            exact_binomial_recoverability(
                config.archive_oligos, config.erasure_tolerance, true_q
            )
            < config.recoverability_target
        ):
            violations += 1
        if not is_action_year(year, horizon_years, scenario.inspection_interval):
            continue

        decisions += 1
        observed_q = float(np.clip(true_q * scenario.observation_factor, 0.0, 1.0))
        unavailable = int(rng.binomial(scenario.sentinel_size, observed_q))
        weights = posterior_update(
            grid, weights, age, unavailable, scenario.sentinel_size
        )
        index = quantile_index(weights, config.posterior_quantile)
        decision_hazard = float(grid[index])
        next_age = age + scenario.inspection_interval
        refresh = (
            recoverability_probability(config, decision_hazard, next_age)
            < config.recoverability_target
        )
        if refresh:
            refreshes += 1
            age = 0
            continue

        no_refresh_decisions += 1
        certificate_mass = float(weights[: index + 1].sum())
        minimum_certificate_mass = min(minimum_certificate_mass, certificate_mass)
        true_next_q = cumulative_dropout(true_hazard, next_age, scenario.hazard_profile)
        true_next_recoverability = exact_binomial_recoverability(
            config.archive_oligos, config.erasure_tolerance, true_next_q
        )
        if true_next_recoverability < config.recoverability_target:
            false_safe_decisions += 1

    return {
        "scenario": scenario.name,
        "episode": episode,
        "true_hazard": true_hazard,
        "prior": scenario.prior,
        "inspection_interval": scenario.inspection_interval,
        "sentinel_size": scenario.sentinel_size,
        "hazard_profile": scenario.hazard_profile,
        "observation_factor": scenario.observation_factor,
        "decisions": decisions,
        "no_refresh_decisions": no_refresh_decisions,
        "false_safe_decisions": false_safe_decisions,
        "any_false_safe": int(false_safe_decisions > 0),
        "minimum_model_certificate_mass": minimum_certificate_mass,
        "refreshes": refreshes,
        "violation_years": violations,
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    if successes == 0:
        return 0.0, Z_975**2 / (total + Z_975**2)
    if successes == total:
        return total / (total + Z_975**2), 1.0
    estimate = successes / total
    denominator = 1.0 + Z_975**2 / total
    centre = (estimate + Z_975**2 / (2.0 * total)) / denominator
    half = (
        Z_975
        * math.sqrt(estimate * (1.0 - estimate) / total + Z_975**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def stable_cluster_seed(label: str) -> int:
    """Derive a reproducible seed without depending on row or process order."""
    payload = f"{CLUSTER_BOOTSTRAP_BASE_SEED}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def cluster_count_ratio_interval(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    seed: int,
    replicates: int = CLUSTER_BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Percentile CI for a pooled decision count ratio resampling missions."""
    if numerators.size != denominators.size:
        raise ValueError("numerator and denominator arrays must have equal length")
    if numerators.size == 0 or float(denominators.sum()) <= 0.0:
        return math.nan, math.nan
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    ratios = np.full(replicates, np.nan, dtype=float)
    chunk = min(128, replicates)
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, numerators.size, size=(stop - start, numerators.size))
        numerator_sum = numerators[indices].sum(axis=1)
        denominator_sum = denominators[indices].sum(axis=1)
        valid = denominator_sum > 0.0
        ratios[start:stop][valid] = numerator_sum[valid] / denominator_sum[valid]
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return math.nan, math.nan
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    rows: list[dict[str, object]], horizon_years: int
) -> list[dict[str, object]]:
    output = []
    for scenario in SCENARIOS:
        group = [row for row in rows if row["scenario"] == scenario.name]
        no_refresh = sum(int(row["no_refresh_decisions"]) for row in group)
        false_safe = sum(int(row["false_safe_decisions"]) for row in group)
        episode_false_safe = sum(int(row["any_false_safe"]) for row in group)
        numerator_clusters = np.asarray(
            [float(row["false_safe_decisions"]) for row in group], dtype=float
        )
        denominator_clusters = np.asarray(
            [float(row["no_refresh_decisions"]) for row in group], dtype=float
        )
        decision_low, decision_high = cluster_count_ratio_interval(
            numerator_clusters,
            denominator_clusters,
            seed=stable_cluster_seed(f"decision-calibration|{scenario.name}"),
        )
        episode_low, episode_high = wilson_interval(episode_false_safe, len(group))
        output.append(
            {
                "scenario": scenario.name,
                "episodes": len(group),
                "decisions": sum(int(row["decisions"]) for row in group),
                "no_refresh_decisions": no_refresh,
                "false_safe_decisions": false_safe,
                "false_safe_rate": false_safe / no_refresh if no_refresh else math.nan,
                "false_safe_rate_ci95_low": decision_low,
                "false_safe_rate_ci95_high": decision_high,
                "decision_count_ratio": false_safe / no_refresh if no_refresh else math.nan,
                "decision_count_ratio_ci95_low": decision_low,
                "decision_count_ratio_ci95_high": decision_high,
                "decision_count_ratio_ci_method": (
                    "mission-cluster nonparametric percentile bootstrap; "
                    f"B={CLUSTER_BOOTSTRAP_REPLICATES}; "
                    f"seed={stable_cluster_seed(f'decision-calibration|{scenario.name}')}"
                ),
                "episode_false_safe_rate": episode_false_safe / len(group),
                "episode_false_safe_rate_ci95_low": episode_low,
                "episode_false_safe_rate_ci95_high": episode_high,
                "minimum_model_certificate_mass": min(
                    float(row["minimum_model_certificate_mass"]) for row in group
                ),
                "mean_refreshes": sum(int(row["refreshes"]) for row in group)
                / len(group),
                "violation_rate": sum(int(row["violation_years"]) for row in group)
                / (len(group) * horizon_years),
                "calibration_scope": (
                    "prior-predictive joint calibration"
                    if scenario.true_hazard is None
                    else "fixed-hazard stress evaluation"
                ),
                "claim_upgrade_allowed": False,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    task = (0, SCENARIOS[0], 0, 20)
    assert run_episode(task) == run_episode(task)
    rows = [run_episode((0, SCENARIOS[0], episode, 20)) for episode in range(20)]
    assert all(float(row["minimum_model_certificate_mass"]) >= 0.95 for row in rows)
    first_ratio = cluster_count_ratio_interval(
        np.asarray([0.0, 1.0, 0.0, 1.0]),
        np.asarray([2.0, 1.0, 3.0, 1.0]),
        seed=17,
        replicates=1_000,
    )
    second_ratio = cluster_count_ratio_interval(
        np.asarray([0.0, 1.0, 0.0, 1.0]),
        np.asarray([2.0, 1.0, 3.0, 1.0]),
        seed=17,
        replicates=1_000,
    )
    assert first_ratio == second_ratio
    assert 0.0 <= first_ratio[0] <= first_ratio[1] <= 1.0
    print("PASS_PAPER4_DECISION_CALIBRATION_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--horizon-years", type=int, default=60)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--summarize-existing",
        type=Path,
        help="recompute summary/cluster CIs from existing mission rows; do not simulate",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.summarize_existing:
        with args.summarize_existing.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
        summaries = summarize(existing_rows, args.horizon_years)
        write_csv(args.output_dir / "paper4_decision_calibration_summary.csv", summaries)
        print("PASS_PAPER4_DECISION_CALIBRATION_EXISTING_ROWS")
        print(
            f"scenarios={len(SCENARIOS)} rows={len(existing_rows)} "
            f"bootstrap_replicates={CLUSTER_BOOTSTRAP_REPLICATES}"
        )
        return
    tasks = [
        (index, scenario, episode, args.horizon_years)
        for index, scenario in enumerate(SCENARIOS)
        for episode in range(args.episodes)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_episode, tasks, chunksize=32))
    summaries = summarize(rows, args.horizon_years)
    write_csv(args.output_dir / "paper4_decision_calibration_episodes.csv", rows)
    write_csv(args.output_dir / "paper4_decision_calibration_summary.csv", summaries)
    print("PASS_PAPER4_DECISION_CALIBRATION")
    print(
        f"scenarios={len(SCENARIOS)} episodes_per_scenario={args.episodes} "
        f"rows={len(rows)}"
    )


if __name__ == "__main__":
    main()
