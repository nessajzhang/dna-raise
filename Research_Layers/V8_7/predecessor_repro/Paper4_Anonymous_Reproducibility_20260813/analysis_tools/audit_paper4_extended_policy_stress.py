#!/usr/bin/env python3
"""Extended model-risk and common-design policy audits for Paper 4.

This script evaluates only declared computational models.  It does not model
file decoding, demonstrate a successful physical refresh, or validate a DNA
archive lifetime.  Its purpose is to expose how service dependence, imperfect
refresh, and monitoring burden alter a finite-horizon maintenance decision.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta, betabinom, binom

from audit_paper4_finite_horizon_certificate import (
    certified_next_dropout,
    clopper_pearson_upper,
    exposure,
    interval_violation_duration,
)


ROOT = Path(__file__).resolve().parents[1]
HORIZON_YEARS = 60
INSPECTION_INTERVAL = 5
MAX_LOOKS = (HORIZON_YEARS - 1) // INSPECTION_INTERVAL
ALPHA_TOTAL = 0.05
SENTINEL_SIZE = 3_000
HAZARD = 0.006
ACCELERATION = 0.03
OBSERVATION_FACTOR = 0.80
DETECTION_FLOOR = 0.80
ARCHIVE_OLIGOS = 10_000
ERASURE_TOLERANCE = 0.10
RECOVERABILITY_TARGET = 0.99
Z_975 = 1.959963984540054

POLICIES = (
    "fixed_5y",
    "fixed_10y",
    "naive_plugin",
    "beta_sentinel_q95",
    "certificate_independent",
    "certificate_robust",
    "oracle_service_model",
    "posterior_hazard_q95",
    "certificate_observation_robust",
)

HAZARD_GRID = np.linspace(0.0005, 0.0200, 196)


@dataclass(frozen=True)
class AuditScenario:
    family: str
    label: str
    hazard: float = HAZARD
    actual_rho: float = 1e-4
    declared_rho: float = 1e-4
    actual_observation_rho: float = 0.0
    declared_observation_rho: float = 0.0
    residual_dropout: float = 0.0
    refresh_failure_probability: float = 0.0


@lru_cache(maxsize=256)
def service_recoverability(dropout: float, rho: float) -> float:
    cutoff = math.floor(ARCHIVE_OLIGOS * ERASURE_TOLERANCE)
    dropout = float(np.clip(dropout, 0.0, 1.0))
    if rho == 0.0:
        return float(binom.cdf(cutoff, ARCHIVE_OLIGOS, dropout))
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if dropout <= 0.0:
        return 1.0
    if dropout >= 1.0:
        return 0.0
    concentration = 1.0 / rho - 1.0
    return float(
        betabinom.cdf(
            cutoff,
            ARCHIVE_OLIGOS,
            dropout * concentration,
            (1.0 - dropout) * concentration,
        )
    )


@lru_cache(maxsize=64)
def service_critical_dropout(rho: float) -> float:
    low, high = 0.0, ERASURE_TOLERANCE
    for _ in range(80):
        middle = (low + high) / 2.0
        if service_recoverability(middle, rho) >= RECOVERABILITY_TARGET:
            low = middle
        else:
            high = middle
    return low


def physical_dropout(age: float, residual_dropout: float, hazard: float) -> float:
    return 1.0 - (1.0 - residual_dropout) * math.exp(
        -hazard * exposure(age, ACCELERATION)
    )


def physical_critical_age(residual_dropout: float, rho: float, hazard: float) -> float:
    qcrit = service_critical_dropout(rho)
    if residual_dropout >= qcrit:
        return 0.0
    target_exposure = math.log((1.0 - residual_dropout) / (1.0 - qcrit)) / hazard
    low, high = 0.0, max(1.0, target_exposure)
    while exposure(high, ACCELERATION) < target_exposure:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if exposure(middle, ACCELERATION) <= target_exposure:
            low = middle
        else:
            high = middle
    return low


@lru_cache(maxsize=8192)
def cp_cached(unavailable: int) -> float:
    return clopper_pearson_upper(
        unavailable, SENTINEL_SIZE, ALPHA_TOTAL / MAX_LOOKS
    )


@lru_cache(maxsize=8192)
def beta_sentinel_q95_cached(unavailable: int) -> float:
    return float(
        beta.ppf(0.95, unavailable + 1, SENTINEL_SIZE - unavailable + 1)
    )


@lru_cache(maxsize=32768)
def beta_binomial_observation_upper(
    unavailable: int, total: int, alpha: float, rho: float
) -> float:
    """Exact one-sided upper bound for a beta-binomial mean at fixed rho.

    The bound inverts the lower-tail test.  For rho=0 it reduces to the
    Clopper--Pearson bound.  A declared rho is part of the observation model;
    this function does not claim validity for an unknown or misspecified rho.
    """
    if not 0 <= unavailable <= total:
        raise ValueError("unavailable must lie in [0, total]")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if alpha == 0.0 or unavailable == total:
        return 1.0
    if rho == 0.0:
        return clopper_pearson_upper(unavailable, total, alpha)

    concentration = 1.0 / rho - 1.0

    def lower_tail_minus_alpha(mean: float) -> float:
        if mean <= 0.0:
            return 1.0 - alpha
        if mean >= 1.0:
            return -alpha
        return float(
            betabinom.cdf(
                unavailable,
                total,
                mean * concentration,
                (1.0 - mean) * concentration,
            )
            - alpha
        )

    return float(
        brentq(
            lower_tail_minus_alpha,
            1e-10,
            1.0 - 1e-10,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=200,
        )
    )


def normalized_log_weights(log_weights: np.ndarray) -> np.ndarray:
    peak = float(np.max(log_weights))
    weights = np.exp(log_weights - peak)
    return weights / np.sum(weights)


def update_hazard_posterior(
    weights: np.ndarray, age: int, unavailable: int
) -> np.ndarray:
    modeled_dropout = 1.0 - np.exp(
        -HAZARD_GRID * exposure(age, ACCELERATION)
    )
    observed_probability = np.clip(
        DETECTION_FLOOR * modeled_dropout, 1e-12, 1.0 - 1e-12
    )
    log_likelihood = (
        unavailable * np.log(observed_probability)
        + (SENTINEL_SIZE - unavailable) * np.log1p(-observed_probability)
    )
    return normalized_log_weights(np.log(np.maximum(weights, 1e-300)) + log_likelihood)


def posterior_quantile(weights: np.ndarray, quantile: float = 0.95) -> float:
    index = int(np.searchsorted(np.cumsum(weights), quantile, side="left"))
    return float(HAZARD_GRID[min(index, len(HAZARD_GRID) - 1)])


def draw_observation_count(
    rng: np.random.Generator, mean: float, rho: float
) -> int:
    mean = float(np.clip(mean, 0.0, 1.0))
    if rho == 0.0 or mean in {0.0, 1.0}:
        return int(rng.binomial(SENTINEL_SIZE, mean))
    if not 0.0 < rho < 1.0:
        raise ValueError("observation rho must lie in [0, 1)")
    concentration = 1.0 / rho - 1.0
    latent_probability = float(
        rng.beta(mean * concentration, (1.0 - mean) * concentration)
    )
    return int(rng.binomial(SENTINEL_SIZE, latent_probability))


def propagate_current_bound(current_upper: float, age: int) -> float:
    current_upper = min(1.0, current_upper)
    if current_upper >= 1.0:
        return 1.0
    ratio = exposure(age + INSPECTION_INTERVAL, ACCELERATION) / exposure(
        age, ACCELERATION
    )
    return 1.0 - (1.0 - current_upper) ** ratio


def seed_for(scenario_index: int, episode: int, policy: str) -> int:
    policy_index = POLICIES.index(policy)
    return 20260721 + scenario_index * 10_000_019 + episode * 10_007 + policy_index


def decide_refresh(
    policy: str,
    year: int,
    controller_age: int,
    unavailable: int | None,
    true_next_dropout: float,
    scenario: AuditScenario,
    posterior_hazard: float | None = None,
) -> tuple[bool, bool]:
    """Return (refresh, failed_to_certify_deferral)."""
    if policy == "fixed_5y":
        return True, False
    if policy == "fixed_10y":
        return year % 10 == 0, False
    if policy == "oracle_service_model":
        return (
            service_recoverability(true_next_dropout, scenario.actual_rho)
            < RECOVERABILITY_TARGET,
            False,
        )
    if unavailable is None:
        raise AssertionError("observation-based policy requires a sentinel count")

    if policy == "naive_plugin":
        current_upper = min(
            1.0, unavailable / SENTINEL_SIZE / DETECTION_FLOOR
        )
        next_upper = propagate_current_bound(current_upper, controller_age)
        rho_for_decision = 0.0
    elif policy == "beta_sentinel_q95":
        current_upper = min(
            1.0, beta_sentinel_q95_cached(unavailable) / DETECTION_FLOOR
        )
        next_upper = propagate_current_bound(current_upper, controller_age)
        rho_for_decision = 0.0
    elif policy == "posterior_hazard_q95":
        if posterior_hazard is None:
            raise AssertionError("posterior hazard is required")
        next_upper = 1.0 - math.exp(
            -posterior_hazard
            * exposure(controller_age + INSPECTION_INTERVAL, ACCELERATION)
        )
        rho_for_decision = scenario.declared_rho
    elif policy in {
        "certificate_independent",
        "certificate_robust",
        "certificate_observation_robust",
    }:
        if policy == "certificate_observation_robust":
            observed_upper = beta_binomial_observation_upper(
                unavailable,
                SENTINEL_SIZE,
                ALPHA_TOTAL / MAX_LOOKS,
                scenario.declared_observation_rho,
            )
            current_upper = min(1.0, observed_upper / DETECTION_FLOOR)
            next_upper = propagate_current_bound(current_upper, controller_age)
        else:
            _, _, next_upper = certified_next_dropout(
                unavailable,
                SENTINEL_SIZE,
                controller_age,
                controller_age + INSPECTION_INTERVAL,
                ALPHA_TOTAL / MAX_LOOKS,
                DETECTION_FLOOR,
                ACCELERATION,
            )
        rho_for_decision = (
            0.0 if policy == "certificate_independent" else scenario.declared_rho
        )
    else:
        raise ValueError(f"unknown policy: {policy}")

    refresh = next_upper > service_critical_dropout(rho_for_decision)
    return refresh, bool(refresh and policy.startswith("certificate"))


def run_episode(
    task: tuple[int, AuditScenario, str, int]
) -> dict[str, object]:
    scenario_index, scenario, policy, episode = task
    rng = np.random.default_rng(seed_for(scenario_index, episode, policy))
    physical_age = 0
    controller_age = 0
    refresh_attempts = 0
    refresh_failures = 0
    false_refreshes = 0
    unsafe_deferral_decisions = 0
    failed_to_certify_deferral = 0
    decisions = 0
    sentinel_assays = 0
    inspection_panels = 0
    hazard_posterior = np.full(len(HAZARD_GRID), 1.0 / len(HAZARD_GRID))
    violation_duration = 0.0
    critical_age = physical_critical_age(
        scenario.residual_dropout, scenario.actual_rho, scenario.hazard
    )

    for year in range(1, HORIZON_YEARS + 1):
        prior_physical_age = physical_age
        physical_age += 1
        controller_age += 1
        violation_duration += interval_violation_duration(
            prior_physical_age, physical_age, critical_age
        )
        if year % INSPECTION_INTERVAL != 0 or year >= HORIZON_YEARS:
            continue

        decisions += 1
        true_q = physical_dropout(
            physical_age, scenario.residual_dropout, scenario.hazard
        )
        uses_observation = policy not in {
            "fixed_5y",
            "fixed_10y",
            "oracle_service_model",
        }
        unavailable: int | None = None
        if uses_observation:
            observed_q = float(np.clip(true_q * OBSERVATION_FACTOR, 0.0, 1.0))
            unavailable = draw_observation_count(
                rng, observed_q, scenario.actual_observation_rho
            )
            sentinel_assays += SENTINEL_SIZE
            inspection_panels += 1
            if policy == "posterior_hazard_q95":
                hazard_posterior = update_hazard_posterior(
                    hazard_posterior, controller_age, unavailable
                )

        true_next_q = physical_dropout(
            physical_age + INSPECTION_INTERVAL,
            scenario.residual_dropout,
            scenario.hazard,
        )
        true_next_recovery = service_recoverability(
            true_next_q, scenario.actual_rho
        )
        refresh, failed_certificate = decide_refresh(
            policy,
            year,
            controller_age,
            unavailable,
            true_next_q,
            scenario,
            posterior_hazard=(
                posterior_quantile(hazard_posterior)
                if policy == "posterior_hazard_q95"
                else None
            ),
        )
        failed_to_certify_deferral += int(failed_certificate)
        if not refresh:
            if true_next_recovery < RECOVERABILITY_TARGET:
                unsafe_deferral_decisions += 1
            continue

        refresh_attempts += 1
        if true_next_recovery >= RECOVERABILITY_TARGET:
            false_refreshes += 1
        controller_age = 0
        if rng.random() < scenario.refresh_failure_probability:
            refresh_failures += 1
        else:
            physical_age = 0

    if policy == "certificate_independent":
        policy_declared_rho: float | str = 0.0
    elif policy in {"certificate_robust", "certificate_observation_robust"}:
        policy_declared_rho = scenario.declared_rho
    else:
        policy_declared_rho = "not applicable"

    return {
        "family": scenario.family,
        "scenario": scenario.label,
        "policy": policy,
        "episode": episode,
        "actual_rho": scenario.actual_rho,
        "actual_observation_rho": scenario.actual_observation_rho,
        "true_hazard_per_year": scenario.hazard,
        "policy_declared_rho": policy_declared_rho,
        "policy_declared_observation_rho": (
            scenario.declared_observation_rho
            if policy == "certificate_observation_robust"
            else (0.0 if policy.startswith("certificate") else "not applicable")
        ),
        "post_refresh_residual_dropout": scenario.residual_dropout,
        "refresh_failure_probability": scenario.refresh_failure_probability,
        "decisions": decisions,
        "refresh_attempts": refresh_attempts,
        "refresh_failures": refresh_failures,
        "false_refreshes": false_refreshes,
        "unsafe_deferral_decisions": unsafe_deferral_decisions,
        "any_unsafe_deferral": int(unsafe_deferral_decisions > 0),
        "failed_to_certify_deferral_decisions": failed_to_certify_deferral,
        "inspection_panels": inspection_panels,
        "sentinel_trials": sentinel_assays,
        "continuous_time_violation_years": violation_duration,
        "continuous_time_violation_fraction": violation_duration / HORIZON_YEARS,
        "normalized_burden_w1": refresh_attempts + inspection_panels,
        "normalized_burden_w5": 5 * refresh_attempts + inspection_panels,
        "normalized_burden_w20": 20 * refresh_attempts + inspection_panels,
    }


def wilson(successes: int, total: int) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + Z_975**2 / total
    center = (p + Z_975**2 / (2.0 * total)) / denominator
    radius = Z_975 * math.sqrt(
        p * (1.0 - p) / total + Z_975**2 / (4.0 * total**2)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["scenario"]), str(row["policy"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (family, scenario, policy), group in sorted(groups.items()):
        episode_unsafe_deferral = sum(
            int(row["any_unsafe_deferral"]) for row in group
        )
        low, high = wilson(episode_unsafe_deferral, len(group))
        certificate_policy = policy.startswith("certificate")
        if certificate_policy:
            observation_envelope = (
                "yes"
                if math.isclose(
                    float(group[0]["policy_declared_observation_rho"]),
                    float(group[0]["actual_observation_rho"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                else "no"
            )
            aging_envelope = "yes"
            service_envelope = (
                "yes"
                if float(group[0]["policy_declared_rho"])
                >= float(group[0]["actual_rho"])
                else "no"
            )
            theorem_envelope = (
                "yes"
                if observation_envelope == "yes" and service_envelope == "yes"
                else "no"
            )
            event_label = "false-safe decision"
        else:
            observation_envelope = "not applicable"
            aging_envelope = "not applicable"
            service_envelope = "not applicable"
            theorem_envelope = "not applicable"
            event_label = "unsafe deferral"
        output.append(
            {
                "family": family,
                "scenario": scenario,
                "policy": policy,
                "episodes": len(group),
                "actual_rho": group[0]["actual_rho"],
                "actual_observation_rho": group[0]["actual_observation_rho"],
                "true_hazard_per_year": group[0]["true_hazard_per_year"],
                "policy_declared_rho": group[0]["policy_declared_rho"],
                "policy_declared_observation_rho": group[0]["policy_declared_observation_rho"],
                "post_refresh_residual_dropout": group[0]["post_refresh_residual_dropout"],
                "refresh_failure_probability": group[0]["refresh_failure_probability"],
                "deferral_event_label": event_label,
                "observation_envelope_valid": observation_envelope,
                "aging_envelope_valid": aging_envelope,
                "service_envelope_valid": service_envelope,
                "theorem_envelope_valid": theorem_envelope,
                "randomization_design": "policy-specific deterministic seeds; cohorts are not trajectory-paired",
                "episode_unsafe_deferral_rate": episode_unsafe_deferral / len(group),
                "episode_unsafe_deferral_rate_ci95_low": low,
                "episode_unsafe_deferral_rate_ci95_high": high,
                "mean_refresh_attempts": float(
                    np.mean([int(row["refresh_attempts"]) for row in group])
                ),
                "se_mean_refresh_attempts": float(
                    np.std([int(row["refresh_attempts"]) for row in group], ddof=1)
                    / math.sqrt(len(group))
                ),
                "mean_refresh_failures": float(
                    np.mean([int(row["refresh_failures"]) for row in group])
                ),
                "mean_false_refreshes": float(
                    np.mean([int(row["false_refreshes"]) for row in group])
                ),
                "mean_failed_to_certify_deferral_decisions": float(
                    np.mean(
                        [
                            int(row["failed_to_certify_deferral_decisions"])
                            for row in group
                        ]
                    )
                ),
                "mean_inspection_panels": float(
                    np.mean([int(row["inspection_panels"]) for row in group])
                ),
                "mean_sentinel_trials": float(
                    np.mean([int(row["sentinel_trials"]) for row in group])
                ),
                "mean_continuous_time_violation_fraction": float(
                    np.mean(
                        [float(row["continuous_time_violation_fraction"]) for row in group]
                    )
                ),
                "mean_normalized_burden_w1": float(
                    np.mean([float(row["normalized_burden_w1"]) for row in group])
                ),
                "mean_normalized_burden_w5": float(
                    np.mean([float(row["normalized_burden_w5"]) for row in group])
                ),
                "mean_normalized_burden_w20": float(
                    np.mean([float(row["normalized_burden_w20"]) for row in group])
                ),
                "scope": "synthetic computational audit; normalized burden weights are not monetary costs",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_scenarios() -> tuple[
    list[AuditScenario],
    list[AuditScenario],
    list[AuditScenario],
    list[AuditScenario],
]:
    baseline = [
        AuditScenario(
            "common_design_policy",
            f"adverse_common_design_h{hazard:.3f}_rho1e-4",
            hazard=hazard,
        )
        for hazard in (0.006, 0.009, 0.012)
    ]
    dependence = [
        AuditScenario(
            "dependence",
            f"rho_{rho:g}",
            hazard=0.006,
            actual_rho=rho,
            declared_rho=rho,
        )
        for rho in (0.0, 1e-5, 1e-4, 1e-3)
    ]
    refresh = [
        AuditScenario(
            "nonideal_refresh",
            f"q0_{q0:g}_eps_{epsilon:g}",
            residual_dropout=q0,
            refresh_failure_probability=epsilon,
        )
        for q0 in (0.0, 0.005, 0.01, 0.02)
        for epsilon in (0.0, 1e-4, 1e-3, 1e-2)
    ]
    observation_dependence = [
        AuditScenario(
            "observation_dependence",
            f"sentinel_rho_{rho:g}",
            hazard=0.006,
            actual_observation_rho=rho,
            declared_observation_rho=rho,
        )
        for rho in (0.0, 1e-5, 1e-4, 1e-3)
    ]
    return baseline, dependence, refresh, observation_dependence


def self_test() -> None:
    assert MAX_LOOKS == 11
    assert service_critical_dropout(1e-4) < service_critical_dropout(0.0)
    assert math.isclose(
        physical_dropout(0.0, 0.01, HAZARD), 0.01, abs_tol=1e-15
    )
    for rho in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        qcrit = service_critical_dropout(rho)
        assert abs(service_recoverability(qcrit, rho) - RECOVERABILITY_TARGET) < 1e-8
        scores = [
            service_recoverability(float(q), rho)
            for q in np.linspace(0.001, 0.20, 200)
        ]
        # SciPy's extreme beta-binomial tails show sub-nanounit roundoff near 1.
        assert all(left + 1e-8 >= right for left, right in zip(scores, scores[1:]))
    baseline, dependence, refresh, observation_dependence = build_scenarios()
    assert (len(baseline), len(dependence), len(refresh), len(observation_dependence)) == (3, 4, 16, 4)
    for rho in (0.0, 1e-4, 1e-3):
        upper_values = [
            beta_binomial_observation_upper(y, 100, 0.05, rho)
            for y in range(101)
        ]
        assert all(left <= right for left, right in zip(upper_values, upper_values[1:]))
        for mean in np.linspace(0.01, 0.30, 30):
            undercoverage = sum(
                (
                    binom.pmf(y, 100, mean)
                    if rho == 0.0
                    else betabinom.pmf(
                        y,
                        100,
                        mean * (1.0 / rho - 1.0),
                        (1.0 - mean) * (1.0 / rho - 1.0),
                    )
                )
                for y, upper in enumerate(upper_values)
                if upper < mean
            )
            assert undercoverage <= 0.05 + 1e-10
    task = (0, baseline[0], "certificate_robust", 0)
    assert run_episode(task) == run_episode(task)
    print("PASS_PAPER4_EXTENDED_POLICY_STRESS_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--baseline-episodes", type=int, default=5_000)
    parser.add_argument("--dependence-episodes", type=int, default=2_000)
    parser.add_argument("--refresh-episodes", type=int, default=2_000)
    parser.add_argument("--observation-dependence-episodes", type=int, default=5_000)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(14, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if min(
        args.baseline_episodes,
        args.dependence_episodes,
        args.refresh_episodes,
        args.observation_dependence_episodes,
    ) <= 0:
        raise SystemExit("episode counts must be positive")

    baseline, dependence, refresh, observation_dependence = build_scenarios()
    task_groups: list[tuple[str, list[tuple[int, AuditScenario, str, int]]]] = []
    task_groups.append(
        (
            "common_design_policy",
            [
                (index, scenario, policy, episode)
                for index, scenario in enumerate(baseline)
                for policy in (
                    "fixed_5y",
                    "fixed_10y",
                    "naive_plugin",
                    "beta_sentinel_q95",
                    "certificate_independent",
                    "certificate_robust",
                    "oracle_service_model",
                    "posterior_hazard_q95",
                )
                for episode in range(args.baseline_episodes)
            ],
        )
    )
    task_groups.append(
        (
            "dependence",
            [
                (index, scenario, policy, episode)
                for index, scenario in enumerate(dependence, start=10)
                for policy in ("certificate_independent", "certificate_robust")
                for episode in range(args.dependence_episodes)
            ],
        )
    )
    task_groups.append(
        (
            "observation_dependence",
            [
                (index, scenario, policy, episode)
                for index, scenario in enumerate(observation_dependence, start=40)
                for policy in (
                    "certificate_robust",
                    "certificate_observation_robust",
                )
                for episode in range(args.observation_dependence_episodes)
            ],
        )
    )
    task_groups.append(
        (
            "nonideal_refresh",
            [
                (index, scenario, "certificate_robust", episode)
                for index, scenario in enumerate(refresh, start=20)
                for episode in range(args.refresh_episodes)
            ],
        )
    )

    all_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for family, tasks in task_groups:
            rows = list(pool.map(run_episode, tasks, chunksize=32))
            write_csv(
                args.output_dir / f"paper4_extended_{family}_episodes.csv", rows
            )
            all_rows.extend(rows)
    summaries = summarize(all_rows)
    write_csv(args.output_dir / "paper4_extended_policy_summary.csv", summaries)
    print("PASS_PAPER4_EXTENDED_POLICY_STRESS")
    print(
        f"rows={len(all_rows)} baseline={args.baseline_episodes} "
        f"dependence={args.dependence_episodes} refresh={args.refresh_episodes} "
        f"observation_dependence={args.observation_dependence_episodes}"
    )


if __name__ == "__main__":
    main()
