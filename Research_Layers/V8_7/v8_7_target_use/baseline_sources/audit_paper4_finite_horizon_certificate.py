#!/usr/bin/env python3
"""Audit a finite-horizon simultaneous no-refresh certificate for Paper 4.

The certificate is a model-conditional finite-sample result, not evidence that
DNA archives actually follow the declared ageing and observation envelopes.
At each inspection it combines a one-sided Clopper--Pearson bound, a lower
bound on sentinel detection, an upper envelope on ageing acceleration, and the
exact outer-code binomial tail.  A prespecified alpha budget is split over the
maximum number of inspections, so a union bound controls the probability of
any false-safe no-refresh decision over the finite archive horizon.
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
from scipy.stats import beta, binom

from paper4_archive_maintenance_seed import (
    Config,
    exact_binomial_recoverability,
    is_action_year,
)


ROOT = Path(__file__).resolve().parents[1]
Z_975 = 1.959963984540054
A_REF_YEARS = 1.0
CLUSTER_BOOTSTRAP_REPLICATES = 10_000
CLUSTER_BOOTSTRAP_BASE_SEED = 20260824


@dataclass(frozen=True)
class CertificateScenario:
    name: str
    true_hazard: float
    sentinel_size: int = 300
    inspection_interval: int = 5
    true_acceleration: float = 0.0
    observation_factor: float = 1.0
    declared_detection_floor: float = 1.0
    declared_acceleration_ceiling: float = 0.0
    propagation_bound: str = "tight"

    @property
    def envelope_valid(self) -> bool:
        return (
            self.observation_factor >= self.declared_detection_floor
            and self.true_acceleration <= self.declared_acceleration_ceiling
        )


SCENARIOS = (
    CertificateScenario("exact_constant_h0002", 0.002),
    CertificateScenario("exact_constant_h0006", 0.006),
    CertificateScenario("exact_constant_h0012", 0.012),
    CertificateScenario(
        "naive_acceleration_and_underdetection_high_information",
        0.006,
        sentinel_size=3_000,
        true_acceleration=0.03,
        observation_factor=0.80,
    ),
    CertificateScenario(
        "enveloped_acceleration_and_underdetection_high_information",
        0.006,
        sentinel_size=3_000,
        true_acceleration=0.03,
        observation_factor=0.80,
        declared_detection_floor=0.80,
        declared_acceleration_ceiling=0.03,
    ),
    CertificateScenario(
        "enveloped_acceleration_and_underdetection_high_information_coarse",
        0.006,
        sentinel_size=3_000,
        true_acceleration=0.03,
        observation_factor=0.80,
        declared_detection_floor=0.80,
        declared_acceleration_ceiling=0.03,
        propagation_bound="coarse",
    ),
    CertificateScenario(
        "envelope_cost_under_constant_hazard_high_information",
        0.006,
        sentinel_size=3_000,
        declared_detection_floor=0.80,
        declared_acceleration_ceiling=0.03,
    ),
    CertificateScenario(
        "envelope_cost_under_constant_hazard_high_information_coarse",
        0.006,
        sentinel_size=3_000,
        declared_detection_floor=0.80,
        declared_acceleration_ceiling=0.03,
        propagation_bound="coarse",
    ),
    CertificateScenario(
        "ten_year_schedule_gap",
        0.012,
        inspection_interval=10,
    ),
)


def exposure(age: float, acceleration: float) -> float:
    """Cumulative exposure in years under the declared aging stress envelope.

    ``age`` and ``A_REF_YEARS`` are measured in years and ``acceleration`` has
    units year^-1.  The reference-age form avoids subtracting a dimensionless
    constant from a dimensional age.  The envelope is linear below the
    reference age and quadratically accelerated thereafter.
    """
    if age < 0.0:
        raise ValueError("age must be nonnegative")
    if acceleration < 0.0:
        raise ValueError("acceleration must be nonnegative")
    return age * (
        1.0 + acceleration * max(age - A_REF_YEARS, 0.0)
    )


def cumulative_dropout(hazard: float, age: float, acceleration: float) -> float:
    return 1.0 - math.exp(-hazard * exposure(age, acceleration))


def clopper_pearson_upper(unavailable: int, total: int, alpha: float) -> float:
    """One-sided exact upper confidence bound for a binomial proportion."""
    if not 0 <= unavailable <= total:
        raise ValueError("unavailable must lie in [0, total]")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    # A look assigned no type-I-error budget can only return the vacuous bound.
    if alpha == 0.0:
        return 1.0
    if unavailable == total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, unavailable + 1, total - unavailable))


def certified_next_dropout(
    unavailable: int,
    total: int,
    age: int,
    next_age: int,
    alpha: float,
    detection_floor: float,
    acceleration_ceiling: float,
    propagation_bound: str = "tight",
) -> tuple[float, float, float]:
    """Return observed, current-true, and next-age upper dropout bounds.

    The tight bound assumes a fixed acceleration parameter within a refresh
    cycle and uses sup_c h_c(next_age)/h_c(age), attained at c_max.  The
    optional coarse comparator relaxes h_c(age) to age.
    """
    if age <= 0 or next_age <= age:
        raise ValueError("certificate requires 0 < age < next_age")
    if not 0.0 < detection_floor <= 1.0:
        raise ValueError("detection_floor must lie in (0, 1]")
    if acceleration_ceiling < 0.0:
        raise ValueError("acceleration_ceiling must be nonnegative")
    if propagation_bound not in {"tight", "coarse"}:
        raise ValueError("propagation_bound must be 'tight' or 'coarse'")

    observed_upper = clopper_pearson_upper(unavailable, total, alpha)
    current_true_upper = min(1.0, observed_upper / detection_floor)
    if current_true_upper >= 1.0:
        return observed_upper, current_true_upper, 1.0
    denominator = (
        exposure(age, acceleration_ceiling)
        if propagation_bound == "tight"
        else float(age)
    )
    exposure_ratio = exposure(next_age, acceleration_ceiling) / denominator
    next_upper = 1.0 - (1.0 - current_true_upper) ** exposure_ratio
    return observed_upper, current_true_upper, next_upper


def critical_dropout(
    archive_oligos: int, erasure_tolerance: float, recoverability_target: float
) -> float:
    """Largest dropout probability meeting the exact service target."""
    if archive_oligos <= 0:
        raise ValueError("archive_oligos must be positive")
    if not 0.0 <= erasure_tolerance <= 1.0:
        raise ValueError("erasure_tolerance must lie in [0, 1]")
    if not 0.0 < recoverability_target < 1.0:
        raise ValueError("recoverability_target must lie in (0, 1)")
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if exact_binomial_recoverability(
            archive_oligos, erasure_tolerance, middle
        ) >= recoverability_target:
            low = middle
        else:
            high = middle
    return low


def critical_age(
    hazard: float, acceleration: float, dropout_threshold: float
) -> float:
    """Continuous cycle age at which the service target is first crossed."""
    if hazard <= 0.0:
        return math.inf
    target_exposure = -math.log1p(-dropout_threshold) / hazard
    low, high = 0.0, max(A_REF_YEARS, target_exposure)
    while exposure(high, acceleration) < target_exposure:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if exposure(middle, acceleration) <= target_exposure:
            low = middle
        else:
            high = middle
    return low


def interval_violation_duration(
    start_age: float, end_age: float, service_critical_age: float
) -> float:
    """Lebesgue duration below target on a monotone within-cycle interval."""
    if not 0.0 <= start_age <= end_age:
        raise ValueError("require 0 <= start_age <= end_age")
    return max(0.0, end_age - max(start_age, service_critical_age))


def scenario_seed(scenario_index: int, episode: int) -> int:
    return 20260718 + scenario_index * 10_000_019 + episode * 10_007


def run_episode(
    task: tuple[int, CertificateScenario, int, int, float]
) -> dict[str, object]:
    scenario_index, scenario, episode, horizon_years, alpha_total = task
    rng = np.random.default_rng(scenario_seed(scenario_index, episode))
    config = Config(
        horizon_years=horizon_years,
        inspection_interval=scenario.inspection_interval,
        sentinel_size=scenario.sentinel_size,
        episodes=1,
    )
    max_looks = (horizon_years - 1) // scenario.inspection_interval
    per_look_alpha = alpha_total / max_looks
    age = 0
    refreshes = 0
    decisions = 0
    no_refresh_decisions = 0
    false_safe_decisions = 0
    look_bound_failures = 0
    next_dropout_bound_failures = 0
    violation_years = 0
    uncertified_violation_years = 0
    certified_interval_violation_years = 0
    violation_duration = 0.0
    uncertified_violation_duration = 0.0
    certified_interval_violation_duration = 0.0
    interval_protected_by_no_refresh_certificate = False
    q_critical = critical_dropout(
        config.archive_oligos,
        config.erasure_tolerance,
        config.recoverability_target,
    )
    service_critical_age = critical_age(
        scenario.true_hazard, scenario.true_acceleration, q_critical
    )

    for year in range(1, horizon_years + 1):
        previous_age = age
        age += 1
        interval_duration = interval_violation_duration(
            previous_age, age, service_critical_age
        )
        violation_duration += interval_duration
        if interval_protected_by_no_refresh_certificate:
            certified_interval_violation_duration += interval_duration
        else:
            uncertified_violation_duration += interval_duration
        true_q = cumulative_dropout(
            scenario.true_hazard, age, scenario.true_acceleration
        )
        true_recoverability = exact_binomial_recoverability(
            config.archive_oligos, config.erasure_tolerance, true_q
        )
        if true_recoverability < config.recoverability_target:
            violation_years += 1
            if interval_protected_by_no_refresh_certificate:
                certified_interval_violation_years += 1
            else:
                uncertified_violation_years += 1

        if not is_action_year(year, horizon_years, scenario.inspection_interval):
            continue

        decisions += 1
        observed_q = float(np.clip(true_q * scenario.observation_factor, 0.0, 1.0))
        unavailable = int(rng.binomial(scenario.sentinel_size, observed_q))
        observed_upper, _, next_dropout_upper = certified_next_dropout(
            unavailable,
            scenario.sentinel_size,
            age,
            age + scenario.inspection_interval,
            per_look_alpha,
            scenario.declared_detection_floor,
            scenario.declared_acceleration_ceiling,
            scenario.propagation_bound,
        )
        if observed_upper + 1e-15 < observed_q:
            look_bound_failures += 1

        next_age = age + scenario.inspection_interval
        true_next_q = cumulative_dropout(
            scenario.true_hazard, next_age, scenario.true_acceleration
        )
        if next_dropout_upper + 1e-15 < true_next_q:
            next_dropout_bound_failures += 1
        certified_recoverability = exact_binomial_recoverability(
            config.archive_oligos,
            config.erasure_tolerance,
            next_dropout_upper,
        )
        refresh = certified_recoverability < config.recoverability_target
        if refresh:
            refreshes += 1
            age = 0
            interval_protected_by_no_refresh_certificate = False
            continue

        no_refresh_decisions += 1
        interval_protected_by_no_refresh_certificate = True
        true_next_recoverability = exact_binomial_recoverability(
            config.archive_oligos,
            config.erasure_tolerance,
            true_next_q,
        )
        if true_next_recoverability < config.recoverability_target:
            false_safe_decisions += 1

    return {
        "scenario": scenario.name,
        "episode": episode,
        "true_hazard": scenario.true_hazard,
        "true_acceleration": scenario.true_acceleration,
        "observation_factor": scenario.observation_factor,
        "declared_detection_floor": scenario.declared_detection_floor,
        "declared_acceleration_ceiling": scenario.declared_acceleration_ceiling,
        "propagation_bound": scenario.propagation_bound,
        "model_envelope_valid": int(scenario.envelope_valid),
        "sentinel_size": scenario.sentinel_size,
        "inspection_interval": scenario.inspection_interval,
        "max_looks": max_looks,
        "alpha_total": alpha_total,
        "per_look_alpha": per_look_alpha,
        "decisions": decisions,
        "no_refresh_decisions": no_refresh_decisions,
        "false_safe_decisions": false_safe_decisions,
        "any_false_safe": int(false_safe_decisions > 0),
        "look_bound_failures": look_bound_failures,
        "any_look_bound_failure": int(look_bound_failures > 0),
        "next_dropout_bound_failures": next_dropout_bound_failures,
        "any_next_dropout_bound_failure": int(next_dropout_bound_failures > 0),
        "refreshes": refreshes,
        "violation_years": violation_years,
        "any_violation": int(violation_years > 0),
        "uncertified_violation_years": uncertified_violation_years,
        "certified_interval_violation_years": certified_interval_violation_years,
        "violation_duration_years": violation_duration,
        "uncertified_violation_duration_years": uncertified_violation_duration,
        "certified_interval_violation_duration_years": (
            certified_interval_violation_duration
        ),
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
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
    """Percentile CI for a pooled count ratio after resampling missions.

    Each row is one complete mission/episode cluster.  A replicate resamples
    complete rows with replacement, then forms sum(numerator)/sum(denominator);
    it never treats individual decision looks as independent Bernoulli draws.
    """
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
    output: list[dict[str, object]] = []
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
            seed=stable_cluster_seed(f"finite-horizon|{scenario.name}"),
        )
        episode_low, episode_high = wilson_interval(episode_false_safe, len(group))
        episode_bound_failure = sum(
            int(row["any_look_bound_failure"]) for row in group
        )
        bound_low, bound_high = wilson_interval(episode_bound_failure, len(group))
        episode_violation = sum(int(row["any_violation"]) for row in group)
        output.append(
            {
                "scenario": scenario.name,
                "episodes": len(group),
                "model_envelope_valid": scenario.envelope_valid,
                "alpha_total": float(group[0]["alpha_total"]),
                "per_look_alpha": float(group[0]["per_look_alpha"]),
                "max_looks": int(group[0]["max_looks"]),
                "sentinel_size": scenario.sentinel_size,
                "inspection_interval": scenario.inspection_interval,
                "propagation_bound": scenario.propagation_bound,
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
                    f"seed={stable_cluster_seed(f'finite-horizon|{scenario.name}')}"
                ),
                "episode_false_safe_rate": episode_false_safe / len(group),
                "episode_false_safe_rate_ci95_low": episode_low,
                "episode_false_safe_rate_ci95_high": episode_high,
                "episode_look_bound_failure_rate": episode_bound_failure / len(group),
                "episode_look_bound_failure_rate_ci95_low": bound_low,
                "episode_look_bound_failure_rate_ci95_high": bound_high,
                "mean_refreshes": sum(int(row["refreshes"]) for row in group)
                / len(group),
                "mean_no_refresh_decisions": no_refresh / len(group),
                "violation_rate": sum(int(row["violation_years"]) for row in group)
                / (len(group) * horizon_years),
                "continuous_time_violation_fraction": sum(
                    float(row["violation_duration_years"]) for row in group
                )
                / (len(group) * horizon_years),
                "episode_violation_rate": episode_violation / len(group),
                "uncertified_violation_rate": sum(
                    int(row["uncertified_violation_years"]) for row in group
                )
                / (len(group) * horizon_years),
                "continuous_time_uncertified_violation_fraction": sum(
                    float(row["uncertified_violation_duration_years"])
                    for row in group
                )
                / (len(group) * horizon_years),
                "certified_interval_violation_rate": sum(
                    int(row["certified_interval_violation_years"]) for row in group
                )
                / (len(group) * horizon_years),
                "continuous_time_certified_interval_violation_fraction": sum(
                    float(row["certified_interval_violation_duration_years"])
                    for row in group
                )
                / (len(group) * horizon_years),
                "theorem_applicable": scenario.envelope_valid,
                "finite_horizon_false_safe_upper_bound": (
                    float(group[0]["alpha_total"])
                    if scenario.envelope_valid
                    else math.nan
                ),
                "guarantee_scope": (
                    "simultaneous no-refresh decisions under declared envelopes"
                    if scenario.envelope_valid
                    else "stress test outside declared envelope; no guarantee"
                ),
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
    alpha = 0.05
    n = 30
    bounds = [clopper_pearson_upper(y, n, alpha) for y in range(n + 1)]
    assert all(left <= right for left, right in zip(bounds, bounds[1:]))
    assert abs(bounds[0] - (1.0 - alpha ** (1.0 / n))) < 1e-12
    assert clopper_pearson_upper(0, n, 0.0) == 1.0
    assert clopper_pearson_upper(n, n, 0.0) == 1.0
    for p in np.linspace(0.0, 1.0, 401):
        failure = sum(
            float(binom.pmf(y, n, p))
            for y, upper in enumerate(bounds)
            if upper + 1e-14 < p
        )
        assert failure <= alpha + 1e-11
    threshold_candidates = [
        float(np.nextafter(upper, 1.0)) for upper in bounds[:-1] if upper < 1.0
    ]
    for p in threshold_candidates:
        failure = sum(
            float(binom.pmf(y, n, p))
            for y, upper in enumerate(bounds)
            if upper < p
        )
        assert failure <= alpha + 1e-10

    for hazard in (0.001, 0.006, 0.020):
        for acceleration in (0.0, 0.01, 0.03):
            for detection in (0.8, 0.9, 1.0):
                for age in (1, 5, 10, 20):
                    next_age = age + 5
                    current = cumulative_dropout(hazard, age, acceleration)
                    observed = detection * current
                    current_upper = min(1.0, observed / 0.8)
                    next_upper = 1.0 - (1.0 - current_upper) ** (
                        exposure(next_age, 0.03) / exposure(age, 0.03)
                    )
                    truth = cumulative_dropout(hazard, next_age, acceleration)
                    assert next_upper + 1e-14 >= truth

                    coarse_upper = 1.0 - (1.0 - current_upper) ** (
                        exposure(next_age, 0.03) / age
                    )
                    assert coarse_upper + 1e-14 >= next_upper

    q_star = critical_dropout(10_000, 0.10, 0.99)
    assert 0.0 < q_star < 0.10
    age_star = critical_age(0.012, 0.03, q_star)
    assert math.isclose(
        cumulative_dropout(0.012, age_star, 0.03), q_star, abs_tol=1e-12
    )
    assert interval_violation_duration(0.0, age_star, age_star) == 0.0
    assert math.isclose(
        interval_violation_duration(age_star - 1.0, age_star + 2.0, age_star),
        2.0,
    )

    task = (0, SCENARIOS[0], 0, 20, 0.05)
    assert run_episode(task) == run_episode(task)
    cluster_numerators = np.asarray([0.0, 1.0, 0.0, 1.0])
    cluster_denominators = np.asarray([2.0, 1.0, 3.0, 1.0])
    first_ratio = cluster_count_ratio_interval(
        cluster_numerators, cluster_denominators, seed=17, replicates=1_000
    )
    second_ratio = cluster_count_ratio_interval(
        cluster_numerators, cluster_denominators, seed=17, replicates=1_000
    )
    assert first_ratio == second_ratio
    assert 0.0 <= first_ratio[0] <= first_ratio[1] <= 1.0
    empty_ratio = cluster_count_ratio_interval(
        np.asarray([0.0]), np.asarray([0.0]), seed=17
    )
    assert all(math.isnan(value) for value in empty_ratio)
    assert SCENARIOS[0].envelope_valid
    assert not SCENARIOS[3].envelope_valid
    assert len(SCENARIOS) == 9
    print("PASS_PAPER4_FINITE_HORIZON_CERTIFICATE_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--horizon-years", type=int, default=60)
    parser.add_argument("--alpha-total", type=float, default=0.05)
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
        write_csv(args.output_dir / "paper4_finite_horizon_certificate_summary.csv", summaries)
        print("PASS_PAPER4_FINITE_HORIZON_CERTIFICATE_EXISTING_ROWS")
        print(
            f"scenarios={len(SCENARIOS)} rows={len(existing_rows)} "
            f"bootstrap_replicates={CLUSTER_BOOTSTRAP_REPLICATES}"
        )
        return
    if not 0.0 < args.alpha_total < 1.0:
        raise SystemExit("--alpha-total must lie in (0, 1)")
    tasks = [
        (index, scenario, episode, args.horizon_years, args.alpha_total)
        for index, scenario in enumerate(SCENARIOS)
        for episode in range(args.episodes)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_episode, tasks, chunksize=32))
    summaries = summarize(rows, args.horizon_years)
    write_csv(args.output_dir / "paper4_finite_horizon_certificate_episodes.csv", rows)
    write_csv(args.output_dir / "paper4_finite_horizon_certificate_summary.csv", summaries)
    print("PASS_PAPER4_FINITE_HORIZON_CERTIFICATE")
    print(
        f"scenarios={len(SCENARIOS)} episodes_per_scenario={args.episodes} "
        f"rows={len(rows)} alpha_total={args.alpha_total}"
    )


if __name__ == "__main__":
    main()
