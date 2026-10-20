#!/usr/bin/env python3
"""Append-only Paper 4 V8.8 independent structural challenge.

This program never reads or rewrites V8.7 episode records. It generates fresh
synthetic missions under a separately declared beta-binomial observation and
service model, compares prespecified theorem-valid alpha-spending profiles,
and keeps tuning, test, and truth-aware offline-control namespaces disjoint.

The result is model-internal. It is not physical qualification of a DNA
archive, an estimate of an empirical intraclass correlation, or evidence that
unknown misspecification can be detected at runtime.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import beta, betabinom, binom


ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "paper4_nc_v8_8_independent_structural_challenge_20260905"
PREDECESSOR_SHA256 = (
    "6BAECA7E25E618946F7BA067C5861AE6F67CBA7A9D6BB4BB1AC4EC132BC3BC30"
)
REVIEW_SYNTHESIS_SHA256 = (
    "93662C7724EB59020D976EDE3198A6AF4A7B658B3B799F5CEF31BFC9ED8CBA2E"
)

HAZARDS = (0.002, 0.006, 0.012)
GAMMAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
TUNE_MISSIONS_PER_HAZARD = 5_000
TEST_MISSIONS_PER_HAZARD = 20_000
CONTROL_MISSIONS = 5_000
BOOTSTRAP_REPLICATES = 20_000
PRIMARY_FAMILY_ALPHA = 0.05
PRIMARY_GATE_COUNT = 6
ALPHA_GATE = PRIMARY_FAMILY_ALPHA / PRIMARY_GATE_COUNT
ELIGIBLE_RELATIVE_MARGIN = 0.50
ELIGIBLE_PASSAGE_FLOOR = 0.25
OVERALL_AUTHORIZATION_FLOOR = 0.05
A_REF_YEARS = 1.0

TUNE_NAMESPACE = "paper4-v8.8|tune|20260905"
TEST_NAMESPACE = "paper4-v8.8|test|20260905"
CONTROL_NAMESPACE = "paper4-v8.8|offline-control|20260905"


@dataclass(frozen=True)
class StudyConfig:
    horizon_years: int = 60
    inspection_interval: int = 5
    sentinel_size: int = 3_000
    archive_oligos: int = 10_000
    erasure_tolerance: float = 0.10
    recoverability_target: float = 0.99
    alpha_total: float = 0.05
    true_detection_factor: float = 0.80
    declared_detection_floor: float = 0.80
    true_acceleration: float = 0.03
    declared_acceleration_ceiling: float = 0.03
    actual_observation_rho: float = 0.001
    declared_observation_rho: float = 0.001
    actual_service_rho: float = 0.0001
    declared_service_rho: float = 0.0001

    @property
    def max_looks(self) -> int:
        return (self.horizon_years - 1) // self.inspection_interval

    @property
    def q_critical(self) -> float:
        return service_critical_dropout(
            self.archive_oligos,
            self.erasure_tolerance,
            self.recoverability_target,
            self.declared_service_rho,
        )

    @property
    def envelope_valid(self) -> bool:
        return (
            self.true_detection_factor >= self.declared_detection_floor
            and self.true_acceleration <= self.declared_acceleration_ceiling
            and math.isclose(
                self.actual_observation_rho,
                self.declared_observation_rho,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                self.actual_service_rho,
                self.declared_service_rho,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )


CONFIG = StudyConfig()
_AUTHORIZATION_CUTOFFS: np.ndarray | None = None
_UNDERCOVERAGE_CUTOFFS: np.ndarray | None = None
_BOUND_MODEL = "unset"
_Q_CRITICAL: float | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_seed(namespace: str, hazard: float, episode: int) -> int:
    payload = f"{STUDY_ID}|{namespace}|{hazard:.6f}|{episode}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_seed(label: str) -> int:
    payload = f"{STUDY_ID}|bootstrap|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def gamma_label(gamma: float) -> str:
    sign = "m" if gamma < 0 else "p"
    magnitude = str(abs(gamma)).replace(".", "p")
    return f"gamma_{sign}{magnitude}"


def gamma_index(gamma: float) -> int:
    return GAMMAS.index(float(gamma))


def alpha_schedule(gamma: float, looks: int, alpha_total: float) -> np.ndarray:
    if looks <= 0:
        raise ValueError("looks must be positive")
    if not 0.0 < alpha_total < 1.0:
        raise ValueError("alpha_total must lie in (0,1)")
    indices = np.arange(1, looks + 1, dtype=float)
    weights = indices**gamma
    schedule = alpha_total * weights / weights.sum()
    if not np.all(schedule > 0.0):
        raise AssertionError("every look requires positive alpha")
    if not math.isclose(
        float(schedule.sum()), alpha_total, rel_tol=0.0, abs_tol=1e-15
    ):
        raise AssertionError("alpha schedule does not exhaust the declared budget")
    return schedule


def exposure(age: float, acceleration: float) -> float:
    if age < 0.0 or acceleration < 0.0:
        raise ValueError("age and acceleration must be nonnegative")
    return age * (1.0 + acceleration * max(age - A_REF_YEARS, 0.0))


def cumulative_dropout(hazard: float, age: float, acceleration: float) -> float:
    return 1.0 - math.exp(-hazard * exposure(age, acceleration))


def propagate_current_bound(
    current_upper: float, age: int, next_age: int, acceleration_ceiling: float
) -> float:
    if age <= 0 or next_age <= age:
        raise ValueError("propagation requires 0 < age < next_age")
    current_upper = float(np.clip(current_upper, 0.0, 1.0))
    if current_upper >= 1.0:
        return 1.0
    ratio = exposure(next_age, acceleration_ceiling) / exposure(
        age, acceleration_ceiling
    )
    return 1.0 - (1.0 - current_upper) ** ratio


def service_recoverability(
    dropout: float, archive_oligos: int, erasure_tolerance: float, rho: float
) -> float:
    cutoff = math.floor(archive_oligos * erasure_tolerance)
    dropout = float(np.clip(dropout, 0.0, 1.0))
    if rho == 0.0:
        return float(binom.cdf(cutoff, archive_oligos, dropout))
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie in [0,1)")
    if dropout <= 0.0:
        return 1.0
    if dropout >= 1.0:
        return 0.0
    concentration = 1.0 / rho - 1.0
    return float(
        betabinom.cdf(
            cutoff,
            archive_oligos,
            dropout * concentration,
            (1.0 - dropout) * concentration,
        )
    )


@lru_cache(maxsize=None)
def service_critical_dropout(
    archive_oligos: int,
    erasure_tolerance: float,
    recoverability_target: float,
    rho: float,
) -> float:
    low, high = 0.0, erasure_tolerance
    for _ in range(80):
        middle = (low + high) / 2.0
        if (
            service_recoverability(
                middle, archive_oligos, erasure_tolerance, rho
            )
            >= recoverability_target
        ):
            low = middle
        else:
            high = middle
    return low


def exact_upper(successes: int, total: int, alpha: float = ALPHA_GATE) -> float:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial count")
    if successes == total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, total - successes))


def beta_binomial_upper_table(total: int, alpha: float, rho: float) -> np.ndarray:
    """Invert the beta-binomial lower tail for every count at fixed ICC."""
    if total <= 0 or not 0.0 < alpha < 1.0 or not 0.0 < rho < 1.0:
        raise ValueError("invalid beta-binomial table parameters")
    counts = np.arange(total, dtype=int)
    low = np.full(total, 1e-12, dtype=float)
    high = np.full(total, 1.0 - 1e-12, dtype=float)
    concentration = 1.0 / rho - 1.0
    for _ in range(64):
        middle = (low + high) / 2.0
        cdf = betabinom.cdf(
            counts,
            total,
            middle * concentration,
            (1.0 - middle) * concentration,
        )
        move_low = cdf > alpha
        low = np.where(move_low, middle, low)
        high = np.where(move_low, high, middle)
    output = np.empty(total + 1, dtype=float)
    output[:-1] = high
    output[-1] = 1.0
    if np.any(np.diff(output) < -1e-12):
        raise AssertionError("beta-binomial upper table is not monotone")
    return output


def binomial_upper_table(total: int, alpha: float) -> np.ndarray:
    counts = np.arange(total + 1, dtype=int)
    output = np.ones(total + 1, dtype=float)
    output[:-1] = beta.ppf(
        1.0 - alpha, counts[:-1] + 1, total - counts[:-1]
    )
    if np.any(np.diff(output) < -1e-12):
        raise AssertionError("binomial upper table is not monotone")
    return output


def model_count_cdf(
    count: int, total: int, mean: float, rho: float, bound_model: str
) -> float:
    """CDF used by the declared observation model at a fixed candidate mean."""
    if not -1 <= count <= total:
        raise ValueError("count outside support")
    if count < 0:
        return 0.0
    if count >= total:
        return 1.0
    mean = float(np.clip(mean, 0.0, 1.0))
    if bound_model == "nominal_binomial":
        return float(binom.cdf(count, total, mean))
    if bound_model != "robust_beta_binomial":
        raise ValueError("unsupported bound model")
    if not 0.0 < rho < 1.0:
        raise ValueError("robust beta-binomial model requires rho in (0,1)")
    if mean <= 0.0:
        return 1.0
    if mean >= 1.0:
        return 0.0
    concentration = 1.0 / rho - 1.0
    return float(
        betabinom.cdf(
            count,
            total,
            mean * concentration,
            (1.0 - mean) * concentration,
        )
    )


def maximum_count_below_cdf_level(
    *,
    total: int,
    mean: float,
    rho: float,
    alpha: float,
    bound_model: str,
    inclusive: bool,
) -> int:
    """Largest count whose declared-model lower tail is below alpha."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")

    def qualifies(count: int) -> bool:
        value = model_count_cdf(count, total, mean, rho, bound_model)
        return value <= alpha if inclusive else value < alpha

    if not qualifies(0):
        return -1
    low, high = 0, total
    while low < high:
        middle = (low + high + 1) // 2
        if qualifies(middle):
            low = middle
        else:
            high = middle - 1
    return low


def build_decision_cutoffs(bound_model: str) -> tuple[np.ndarray, np.ndarray]:
    """Precompute exact discrete tests needed by the simulator.

    Inverting a confidence bound at all 3,001 possible counts is unnecessary.
    Monotonicity makes both the authorization rule and the undercoverage audit
    integer-cutoff tests. This routine computes those cutoffs directly from the
    same beta-binomial or binomial CDF used by the full inversion.
    """
    if bound_model not in {"robust_beta_binomial", "nominal_binomial"}:
        raise ValueError("unsupported bound model")
    authorization = np.full(
        (len(GAMMAS), CONFIG.max_looks, CONFIG.max_looks), -2, dtype=np.int16
    )
    undercoverage = np.full(
        (
            len(GAMMAS),
            CONFIG.max_looks,
            len(HAZARDS),
            CONFIG.max_looks,
        ),
        -2,
        dtype=np.int16,
    )
    q_critical = CONFIG.q_critical
    for g_index, gamma in enumerate(GAMMAS):
        schedule = alpha_schedule(gamma, CONFIG.max_looks, CONFIG.alpha_total)
        for look_index, alpha_value in enumerate(schedule):
            # At look j, only ages 5, ..., 5(j+1) are reachable after resets.
            for age_index in range(look_index + 1):
                age = (age_index + 1) * CONFIG.inspection_interval
                next_age = age + CONFIG.inspection_interval
                ratio = exposure(
                    next_age, CONFIG.declared_acceleration_ceiling
                ) / exposure(age, CONFIG.declared_acceleration_ceiling)
                current_q_threshold = 1.0 - (1.0 - q_critical) ** (1.0 / ratio)
                observed_threshold = (
                    CONFIG.declared_detection_floor * current_q_threshold
                )
                authorization[g_index, look_index, age_index] = (
                    maximum_count_below_cdf_level(
                        total=CONFIG.sentinel_size,
                        mean=observed_threshold,
                        rho=CONFIG.declared_observation_rho,
                        alpha=float(alpha_value),
                        bound_model=bound_model,
                        inclusive=True,
                    )
                )
                for hazard_index, hazard in enumerate(HAZARDS):
                    observed_mean = CONFIG.true_detection_factor * cumulative_dropout(
                        hazard, age, CONFIG.true_acceleration
                    )
                    undercoverage[g_index, look_index, hazard_index, age_index] = (
                        maximum_count_below_cdf_level(
                            total=CONFIG.sentinel_size,
                            mean=observed_mean,
                            rho=CONFIG.declared_observation_rho,
                            alpha=float(alpha_value),
                            bound_model=bound_model,
                            inclusive=False,
                        )
                    )
    return authorization, undercoverage


def init_worker(
    authorization_cutoffs: np.ndarray,
    undercoverage_cutoffs: np.ndarray,
    bound_model: str,
    q_critical: float,
) -> None:
    global _AUTHORIZATION_CUTOFFS, _UNDERCOVERAGE_CUTOFFS
    global _BOUND_MODEL, _Q_CRITICAL
    _AUTHORIZATION_CUTOFFS = authorization_cutoffs
    _UNDERCOVERAGE_CUTOFFS = undercoverage_cutoffs
    _BOUND_MODEL = bound_model
    _Q_CRITICAL = q_critical


def draw_observation_count(
    rng: np.random.Generator, mean: float, rho: float
) -> int:
    mean = float(np.clip(mean, 0.0, 1.0))
    if rho == 0.0 or mean in {0.0, 1.0}:
        return int(rng.binomial(CONFIG.sentinel_size, mean))
    concentration = 1.0 / rho - 1.0
    latent = float(
        rng.beta(mean * concentration, (1.0 - mean) * concentration)
    )
    return int(rng.binomial(CONFIG.sentinel_size, latent))


def simulate_profile(
    *,
    cohort: str,
    namespace: str,
    hazard: float,
    episode: int,
    gamma: float,
) -> dict[str, object]:
    if (
        _AUTHORIZATION_CUTOFFS is None
        or _UNDERCOVERAGE_CUTOFFS is None
        or _Q_CRITICAL is None
    ):
        raise RuntimeError("worker state was not initialized")
    root_seed = stable_seed(namespace, hazard, episode)
    # Reinitializing with the same root seed for each profile creates a fixed,
    # auditable coupling for paired mission comparisons.
    rng = np.random.default_rng(root_seed)
    age = 0
    refreshes = 0
    authorizations = 0
    false_safe_decisions = 0
    eligible_candidates = 0
    eligible_passes = 0
    sampling_undercoverage_looks = 0
    current_state_upper_failures = 0
    next_state_upper_failures = 0
    service_lower_failures = 0
    realized_chain_failure_looks = 0
    q_critical = _Q_CRITICAL
    q_eligible = ELIGIBLE_RELATIVE_MARGIN * q_critical
    g_index = gamma_index(gamma)
    hazard_index = HAZARDS.index(float(hazard))

    for look_index in range(CONFIG.max_looks):
        age += CONFIG.inspection_interval
        true_current_q = cumulative_dropout(
            hazard, age, CONFIG.true_acceleration
        )
        observed_mean = float(
            np.clip(CONFIG.true_detection_factor * true_current_q, 0.0, 1.0)
        )
        unavailable = draw_observation_count(
            rng, observed_mean, CONFIG.actual_observation_rho
        )
        age_index = age // CONFIG.inspection_interval - 1
        authorization_cutoff = int(
            _AUTHORIZATION_CUTOFFS[g_index, look_index, age_index]
        )
        undercoverage_cutoff = int(
            _UNDERCOVERAGE_CUTOFFS[
                g_index, look_index, hazard_index, age_index
            ]
        )
        if authorization_cutoff == -2 or undercoverage_cutoff == -2:
            raise AssertionError("encountered an unreachable cutoff-table state")
        next_age = age + CONFIG.inspection_interval
        true_next_q = cumulative_dropout(
            hazard, next_age, CONFIG.true_acceleration
        )
        # These discrete cutoffs are exactly equivalent to inverting the
        # one-sided confidence bound and then applying the monotone transition
        # and service maps; see the exhaustive small-n self-test.
        certified_safe = unavailable <= authorization_cutoff
        under_target = true_next_q > q_critical
        eligible = true_next_q <= q_eligible
        eligible_candidates += int(eligible)

        sampling_failure = unavailable <= undercoverage_cutoff
        # In this locked primary generator, true and declared detection,
        # acceleration, and service ICC match exactly. Therefore a sampling
        # undercoverage propagates to the current-state, next-state, and
        # service lower-bound audits, and the converse also holds.
        current_failure = sampling_failure
        next_failure = sampling_failure
        service_failure = sampling_failure
        chain_failure = current_failure or next_failure or service_failure
        sampling_undercoverage_looks += int(sampling_failure)
        current_state_upper_failures += int(current_failure)
        next_state_upper_failures += int(next_failure)
        service_lower_failures += int(service_failure)
        realized_chain_failure_looks += int(chain_failure)

        if certified_safe:
            authorizations += 1
            eligible_passes += int(eligible)
            false_safe_decisions += int(under_target)
        else:
            refreshes += 1
            age = 0

    if authorizations + refreshes != CONFIG.max_looks:
        raise AssertionError("action accounting failed")
    if eligible_passes > eligible_candidates:
        raise AssertionError("eligible accounting failed")

    return {
        "study_id": STUDY_ID,
        "cohort": cohort,
        "seed_namespace": namespace,
        "root_seed": root_seed,
        "true_hazard": hazard,
        "episode": episode,
        "gamma": gamma,
        "profile": gamma_label(gamma),
        "bound_model": _BOUND_MODEL,
        "declared_envelope_valid": int(
            CONFIG.envelope_valid and _BOUND_MODEL == "robust_beta_binomial"
        ),
        "decisions": CONFIG.max_looks,
        "refreshes": refreshes,
        "authorizations": authorizations,
        "authorization_rate": authorizations / CONFIG.max_looks,
        "false_safe_decisions": false_safe_decisions,
        "any_false_safe": int(false_safe_decisions > 0),
        "eligible_candidates": eligible_candidates,
        "eligible_passes": eligible_passes,
        "sampling_undercoverage_looks": sampling_undercoverage_looks,
        "current_state_upper_failures": current_state_upper_failures,
        "next_state_upper_failures": next_state_upper_failures,
        "service_lower_failures": service_lower_failures,
        "realized_chain_failure_looks": realized_chain_failure_looks,
        "any_realized_chain_failure": int(realized_chain_failure_looks > 0),
    }


def simulate_all_profiles_task(
    task: tuple[str, str, float, int]
) -> list[dict[str, object]]:
    cohort, namespace, hazard, episode = task
    return [
        simulate_profile(
            cohort=cohort,
            namespace=namespace,
            hazard=hazard,
            episode=episode,
            gamma=gamma,
        )
        for gamma in GAMMAS
    ]


def simulate_one_profile_task(
    task: tuple[str, str, float, int, float]
) -> dict[str, object]:
    cohort, namespace, hazard, episode, gamma = task
    return simulate_profile(
        cohort=cohort,
        namespace=namespace,
        hazard=hazard,
        episode=episode,
        gamma=gamma,
    )


def run_all_profiles(
    *,
    cohort: str,
    namespace: str,
    missions_per_hazard: int,
    workers: int,
    bound_model: str,
) -> list[dict[str, object]]:
    authorization_cutoffs, undercoverage_cutoffs = build_decision_cutoffs(
        bound_model
    )
    q_critical = CONFIG.q_critical
    tasks = [
        (cohort, namespace, hazard, episode)
        for hazard in HAZARDS
        for episode in range(missions_per_hazard)
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(
            authorization_cutoffs,
            undercoverage_cutoffs,
            bound_model,
            q_critical,
        ),
    ) as pool:
        for task_rows in pool.map(simulate_all_profiles_task, tasks, chunksize=16):
            rows.extend(task_rows)
    return rows


def run_one_profile(
    *,
    cohort: str,
    namespace: str,
    hazard: float,
    missions: int,
    gamma: float,
    workers: int,
    bound_model: str,
) -> list[dict[str, object]]:
    authorization_cutoffs, undercoverage_cutoffs = build_decision_cutoffs(
        bound_model
    )
    q_critical = CONFIG.q_critical
    tasks = [
        (cohort, namespace, hazard, episode, gamma)
        for episode in range(missions)
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(
            authorization_cutoffs,
            undercoverage_cutoffs,
            bound_model,
            q_critical,
        ),
    ) as pool:
        return list(pool.map(simulate_one_profile_task, tasks, chunksize=16))


def cluster_ratio_bound(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    strata: np.ndarray,
    label: str,
    side: str,
    alpha: float = ALPHA_GATE,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> float:
    if side not in {"lower", "upper"}:
        raise ValueError("side must be lower or upper")
    if numerators.ndim != 1 or denominators.ndim != 1:
        raise ValueError("cluster arrays must be vectors")
    if numerators.size == 0 or numerators.size != denominators.size:
        raise ValueError("cluster arrays must have equal nonzero length")
    if strata.ndim != 1 or strata.size != numerators.size:
        raise ValueError("strata must align with cluster arrays")
    if denominators.sum() <= 0:
        return math.nan
    groups: list[tuple[int, np.ndarray, np.ndarray]] = []
    for stratum in np.unique(strata):
        mask = strata == stratum
        patterns, pattern_counts = np.unique(
            np.column_stack((numerators[mask], denominators[mask])),
            axis=0,
            return_counts=True,
        )
        groups.append(
            (
                int(mask.sum()),
                patterns,
                pattern_counts.astype(float) / mask.sum(),
            )
        )
    rng = np.random.default_rng(bootstrap_seed(label))
    estimates = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 1024):
        stop = min(start + 1024, replicates)
        num = np.zeros(stop - start, dtype=float)
        den = np.zeros(stop - start, dtype=float)
        for size, patterns, probabilities in groups:
            resampled = rng.multinomial(
                size, probabilities, size=stop - start
            )
            num += resampled @ patterns[:, 0]
            den += resampled @ patterns[:, 1]
        estimates[start:stop] = np.divide(
            num, den, out=np.zeros_like(num, dtype=float), where=den > 0
        )
    quantile = alpha if side == "lower" else 1.0 - alpha
    return float(np.quantile(estimates, quantile))


def paired_mean_bound(
    values: np.ndarray,
    *,
    strata: np.ndarray,
    label: str,
    side: str,
    alpha: float = ALPHA_GATE,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> float:
    if side not in {"lower", "upper"}:
        raise ValueError("side must be lower or upper")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired values must be a nonempty vector")
    if strata.ndim != 1 or strata.size != values.size:
        raise ValueError("strata must align with paired values")
    groups: list[tuple[int, np.ndarray, np.ndarray]] = []
    for stratum in np.unique(strata):
        mask = strata == stratum
        unique, counts = np.unique(values[mask], return_counts=True)
        groups.append(
            (int(mask.sum()), unique, counts.astype(float) / mask.sum())
        )
    rng = np.random.default_rng(bootstrap_seed(label))
    estimates = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 1024):
        stop = min(start + 1024, replicates)
        total = np.zeros(stop - start, dtype=float)
        for size, unique, probabilities in groups:
            resampled = rng.multinomial(
                size, probabilities, size=stop - start
            )
            total += resampled @ unique
        estimates[start:stop] = total / values.size
    quantile = alpha if side == "lower" else 1.0 - alpha
    return float(np.quantile(estimates, quantile))


def summarize_profiles(
    rows: Sequence[dict[str, object]], *, label_prefix: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for gamma in GAMMAS:
        group = sorted(
            [row for row in rows if float(row["gamma"]) == gamma],
            key=lambda row: (float(row["true_hazard"]), int(row["episode"])),
        )
        if not group:
            raise ValueError(f"missing profile {gamma}")
        risk_estimates: dict[str, float] = {}
        risk_uppers: dict[str, float] = {}
        refresh_means: dict[str, float] = {}
        authorization_rates: dict[str, float] = {}
        for hazard in HAZARDS:
            stratum = [row for row in group if float(row["true_hazard"]) == hazard]
            events = sum(int(row["any_false_safe"]) for row in stratum)
            key = f"h{int(round(hazard * 10000)):04d}"
            risk_estimates[key] = events / len(stratum)
            risk_uppers[key] = exact_upper(events, len(stratum))
            refresh_means[key] = float(
                np.mean([int(row["refreshes"]) for row in stratum])
            )
            authorization_rates[key] = sum(
                int(row["authorizations"]) for row in stratum
            ) / sum(int(row["decisions"]) for row in stratum)
        auth_num = np.asarray(
            [float(row["authorizations"]) for row in group], dtype=float
        )
        decisions = np.asarray(
            [float(row["decisions"]) for row in group], dtype=float
        )
        eligible_num = np.asarray(
            [float(row["eligible_passes"]) for row in group], dtype=float
        )
        eligible_den = np.asarray(
            [float(row["eligible_candidates"]) for row in group], dtype=float
        )
        mission_strata = np.asarray(
            [float(row["true_hazard"]) for row in group], dtype=float
        )
        output.append(
            {
                "study_id": STUDY_ID,
                "profile": gamma_label(gamma),
                "gamma": gamma,
                "missions": len(group),
                "risk_estimate_json": json.dumps(risk_estimates, sort_keys=True),
                "risk_upper_json": json.dumps(risk_uppers, sort_keys=True),
                "mean_refreshes_equal_hazard": float(
                    np.mean([int(row["refreshes"]) for row in group])
                ),
                "mean_refreshes_by_hazard_json": json.dumps(
                    refresh_means, sort_keys=True
                ),
                "authorization_rate": float(auth_num.sum() / decisions.sum()),
                "authorization_rate_by_hazard_json": json.dumps(
                    authorization_rates, sort_keys=True
                ),
                "authorization_lower": cluster_ratio_bound(
                    auth_num,
                    decisions,
                    strata=mission_strata,
                    label=f"{label_prefix}|{gamma}|authorization",
                    side="lower",
                ),
                "eligible_passage_rate": float(
                    eligible_num.sum() / eligible_den.sum()
                ),
                "eligible_passage_lower": cluster_ratio_bound(
                    eligible_num,
                    eligible_den,
                    strata=mission_strata,
                    label=f"{label_prefix}|{gamma}|eligible",
                    side="lower",
                ),
                "mission_sampling_undercoverage_rate": float(
                    np.mean(
                        [
                            int(row["sampling_undercoverage_looks"]) > 0
                            for row in group
                        ]
                    )
                ),
                "mission_chain_failure_rate": float(
                    np.mean(
                        [int(row["any_realized_chain_failure"]) for row in group]
                    )
                ),
            }
        )
    return output


def select_tuning(summary: Sequence[dict[str, object]]) -> dict[str, object]:
    eligible = [
        row
        for row in summary
        if float(row["gamma"]) != 0.0
        and float(row["authorization_rate"]) >= OVERALL_AUTHORIZATION_FLOOR
        and float(row["eligible_passage_rate"]) >= ELIGIBLE_PASSAGE_FLOOR
    ]
    eligible.sort(
        key=lambda row: (
            float(row["mean_refreshes_equal_hazard"]),
            abs(float(row["gamma"])),
            float(row["gamma"]),
        )
    )
    selected = eligible[0] if eligible else None
    return {
        "study_id": STUDY_ID,
        "created_at": utc_now(),
        "selection_source": "tuning cohort only",
        "selection_rule": (
            "among nonzero gamma profiles with authorization >=0.05 and "
            "eligible passage >=0.25, minimize equal-hazard mean refreshes; "
            "ties favor smaller absolute gamma then smaller gamma"
        ),
        "selected_gamma": (
            float(selected["gamma"])
            if selected is not None
            else "NO_QUALIFIED_NONZERO_PROFILE"
        ),
        "selected_profile": (
            str(selected["profile"])
            if selected is not None
            else "NO_QUALIFIED_NONZERO_PROFILE"
        ),
        "eligible_profiles": [str(row["profile"]) for row in eligible],
        "thresholds": {
            "authorization_rate_point": OVERALL_AUTHORIZATION_FLOOR,
            "eligible_passage_rate_point": ELIGIBLE_PASSAGE_FLOOR,
        },
    }


def paired_refresh_differences(
    rows: Sequence[dict[str, object]], candidate_gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    candidate = {
        (float(row["true_hazard"]), int(row["episode"])): int(row["refreshes"])
        for row in rows
        if float(row["gamma"]) == candidate_gamma
    }
    equal = {
        (float(row["true_hazard"]), int(row["episode"])): int(row["refreshes"])
        for row in rows
        if float(row["gamma"]) == 0.0
    }
    if candidate.keys() != equal.keys() or not candidate:
        raise ValueError("paired mission keys do not match")
    keys = sorted(candidate)
    differences = np.asarray(
        [candidate[key] - equal[key] for key in keys], dtype=float
    )
    strata = np.asarray([key[0] for key in keys], dtype=float)
    return differences, strata


def all_profile_comparisons(
    rows: Sequence[dict[str, object]], *, label_prefix: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for gamma in GAMMAS:
        if gamma == 0.0:
            continue
        differences, strata = paired_refresh_differences(rows, gamma)
        output.append(
            {
                "study_id": STUDY_ID,
                "candidate_profile": gamma_label(gamma),
                "candidate_gamma": gamma,
                "reference_profile": gamma_label(0.0),
                "missions": differences.size,
                "mean_refresh_difference_candidate_minus_equal": float(
                    differences.mean()
                ),
                "two_sided_95_lower": paired_mean_bound(
                    differences,
                    strata=strata,
                    label=f"{label_prefix}|{gamma}|vs-equal|lower95",
                    side="lower",
                    alpha=0.025,
                ),
                "two_sided_95_upper": paired_mean_bound(
                    differences,
                    strata=strata,
                    label=f"{label_prefix}|{gamma}|vs-equal|upper95",
                    side="upper",
                    alpha=0.025,
                ),
            }
        )
    return output


def test_gates(
    rows: Sequence[dict[str, object]],
    summary: Sequence[dict[str, object]],
    selection: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected_gamma = selection.get("selected_gamma")
    if not isinstance(selected_gamma, (int, float)):
        return [], {
            "study_id": STUDY_ID,
            "status": "NO_QUALIFIED_NONZERO_PROFILE_FROM_TUNING",
            "all_primary_gates_pass": False,
            "claim_ceiling": "model-internal structural challenge; physical qualification NOT_RUN",
        }
    selected_gamma = float(selected_gamma)
    selected = next(
        row for row in summary if float(row["gamma"]) == selected_gamma
    )
    risk_uppers = json.loads(str(selected["risk_upper_json"]))
    gates: list[dict[str, object]] = []
    for hazard in HAZARDS:
        key = f"h{int(round(hazard * 10000)):04d}"
        value = float(risk_uppers[key])
        gates.append(
            {
                "gate": f"risk_{key}",
                "estimate_or_bound": value,
                "operator": "<=",
                "threshold": 0.05,
                "pass": int(value <= 0.05),
                "method": "one-sided Clopper-Pearson upper bound on complete missions",
            }
        )
    eligible = float(selected["eligible_passage_lower"])
    gates.append(
        {
            "gate": "eligible_passage",
            "estimate_or_bound": eligible,
            "operator": ">=",
            "threshold": ELIGIBLE_PASSAGE_FLOOR,
            "pass": int(eligible >= ELIGIBLE_PASSAGE_FLOOR),
            "method": "complete-mission cluster-bootstrap lower bound",
        }
    )
    authorization = float(selected["authorization_lower"])
    gates.append(
        {
            "gate": "overall_authorization",
            "estimate_or_bound": authorization,
            "operator": ">=",
            "threshold": OVERALL_AUTHORIZATION_FLOOR,
            "pass": int(authorization >= OVERALL_AUTHORIZATION_FLOOR),
            "method": "complete-mission cluster-bootstrap lower bound",
        }
    )
    differences, strata = paired_refresh_differences(rows, selected_gamma)
    refresh_upper = paired_mean_bound(
        differences,
        strata=strata,
        label=f"test|{selected_gamma}|vs-equal|primary",
        side="upper",
    )
    gates.append(
        {
            "gate": "refresh_gain_vs_equal_spending",
            "estimate_or_bound": refresh_upper,
            "point_estimate": float(differences.mean()),
            "operator": "<",
            "threshold": 0.0,
            "pass": int(refresh_upper < 0.0),
            "method": "paired complete-mission bootstrap upper bound",
        }
    )
    all_pass = all(int(row["pass"]) == 1 for row in gates)
    return gates, {
        "study_id": STUDY_ID,
        "status": (
            "STRUCTURAL_CHALLENGE_GATES_PASS"
            if all_pass
            else "PRIMARY_GATES_FAIL"
        ),
        "all_primary_gates_pass": all_pass,
        "selected_gamma": selected_gamma,
        "selected_profile": gamma_label(selected_gamma),
        "reference_gamma": 0.0,
        "reference_profile": gamma_label(0.0),
        "refresh_difference_point_estimate": float(differences.mean()),
        "refresh_difference_upper": refresh_upper,
        "claim_ceiling": (
            "model-internal independent structural challenge only; physical "
            "qualification, empirical ICC estimation, and external validation NOT_RUN"
        ),
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def complete_run_lock(path: Path, **fields: object) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "STARTED":
        raise ValueError(f"run lock is not STARTED: {path}")
    value.update(fields)
    value["status"] = "COMPLETED"
    value["completed_at"] = utc_now()
    write_json(path, value)


def write_output_manifest(
    output_dir: Path, paths: Sequence[Path], *, command: str, extra: dict[str, object]
) -> None:
    manifest = {
        "study_id": STUDY_ID,
        "created_at": utc_now(),
        "command": command,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in paths
        },
        **extra,
    }
    write_json(output_dir / "OUTPUT_MANIFEST.json", manifest)


def verify_output_manifest(output_dir: Path) -> dict[str, object]:
    path = output_dir / "OUTPUT_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("study_id") != STUDY_ID:
        raise ValueError(f"wrong study id in {path}")
    for name, record in manifest["files"].items():
        target = output_dir / name
        if not target.is_file():
            raise FileNotFoundError(target)
        if sha256_file(target) != record["sha256"]:
            raise ValueError(f"hash mismatch: {target}")
        if target.stat().st_size != int(record["bytes"]):
            raise ValueError(f"size mismatch: {target}")
    return manifest


def load_design_lock(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("study_id") != STUDY_ID or value.get("status") != "LOCKED":
        raise ValueError("invalid design lock")
    if value.get("script_sha256") != sha256_file(Path(__file__)):
        raise ValueError("script differs from locked design")
    if value.get("predecessor_sha256") != PREDECESSOR_SHA256:
        raise ValueError("wrong predecessor authority")
    return value


def load_tuning(tune_dir: Path, design_lock: Path) -> dict[str, object]:
    manifest = verify_output_manifest(tune_dir)
    if (
        manifest.get("command") != "tune"
        or manifest.get("design_lock_sha256") != sha256_file(design_lock)
        or manifest.get("script_sha256") != sha256_file(Path(__file__))
    ):
        raise ValueError("tuning output does not match the current locked design")
    selection_path = tune_dir / "SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("study_id") != STUDY_ID:
        raise ValueError("selection belongs to another study")
    if manifest.get("selection_sha256") != sha256_file(selection_path):
        raise ValueError("selection hash mismatch")
    return selection


def command_lock_design(args: argparse.Namespace) -> None:
    expected = {
        args.predecessor_zip: PREDECESSOR_SHA256,
        args.review_synthesis: REVIEW_SYNTHESIS_SHA256,
    }
    for path, expected_hash in expected.items():
        observed = sha256_file(path)
        if observed != expected_hash:
            raise SystemExit(
                f"authority hash mismatch: {path} expected={expected_hash} observed={observed}"
            )
    required = [
        args.sap,
        args.method_record,
        args.method_evidence,
        args.search_strategy,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    script_sha256 = sha256_file(Path(__file__))
    self_test_stdout = args.self_test_stdout.read_text(encoding="utf-8")
    self_test_stderr = args.self_test_stderr.read_text(encoding="utf-8")
    if "PASS_V8_8_STRUCTURAL_CHALLENGE_SELF_TEST" not in self_test_stdout:
        raise ValueError("self-test PASS marker is missing")
    if script_sha256.lower() not in self_test_stdout.lower():
        raise ValueError("self-test did not exercise the current script hash")
    if self_test_stderr.strip():
        raise ValueError("self-test stderr is not empty")
    smoke_manifest = verify_output_manifest(args.smoke_dir)
    if (
        smoke_manifest.get("command") != "smoke"
        or smoke_manifest.get("evidence_status") != "NOT_EVIDENCE"
        or smoke_manifest.get("script_sha256") != script_sha256
    ):
        raise ValueError("smoke validation does not match the current script")
    lock = {
        "study_id": STUDY_ID,
        "status": "LOCKED",
        "locked_at": utc_now(),
        "predecessor_sha256": PREDECESSOR_SHA256,
        "review_synthesis_sha256": REVIEW_SYNTHESIS_SHA256,
        "script_sha256": script_sha256,
        "supporting_file_hashes": {
            path.name: sha256_file(path) for path in required
        },
        "prelock_validation": {
            "self_test_stdout_sha256": sha256_file(args.self_test_stdout),
            "self_test_stderr_sha256": sha256_file(args.self_test_stderr),
            "smoke_manifest_sha256": sha256_file(
                args.smoke_dir / "OUTPUT_MANIFEST.json"
            ),
            "smoke_rows": smoke_manifest["rows"],
            "smoke_evidence_status": smoke_manifest["evidence_status"],
        },
        "environment": {
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
        },
        "config": asdict(CONFIG),
        "hazards": HAZARDS,
        "gammas": GAMMAS,
        "sample_sizes": {
            "tune_missions_per_hazard": TUNE_MISSIONS_PER_HAZARD,
            "test_missions_per_hazard": TEST_MISSIONS_PER_HAZARD,
            "offline_control_missions": CONTROL_MISSIONS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "seed_namespaces": {
            "tune": TUNE_NAMESPACE,
            "test": TEST_NAMESPACE,
            "offline_control": CONTROL_NAMESPACE,
        },
        "selection_rule": (
            "nonzero gamma; tuning authorization point estimate >=0.05; "
            "eligible-passage point estimate >=0.25; minimum equal-hazard "
            "mean refreshes; ties by abs(gamma), then gamma"
        ),
        "primary_gates": {
            "family_alpha": PRIMARY_FAMILY_ALPHA,
            "gate_count": PRIMARY_GATE_COUNT,
            "alpha_per_gate": ALPHA_GATE,
            "risk_upper_each_hazard": 0.05,
            "eligible_passage_lower": ELIGIBLE_PASSAGE_FLOOR,
            "overall_authorization_lower": OVERALL_AUTHORIZATION_FLOOR,
            "paired_refresh_difference_upper_selected_minus_equal": 0.0,
        },
        "novelty": {"L0": True, "L1": True, "L2": True, "L3": False},
        "failure_statuses": [
            "NO_QUALIFIED_NONZERO_PROFILE",
            "PRIMARY_GATES_FAIL",
            "OFFLINE_CONTROL_INCONCLUSIVE",
        ],
        "claim_ceiling": (
            "synthetic model-internal structural challenge; physical and "
            "external validation NOT_RUN"
        ),
    }
    write_json(args.output, lock, exclusive=True)
    digest_path = args.output.with_suffix(args.output.suffix + ".sha256")
    digest_path.write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(f"PASS_V8_8_DESIGN_LOCK {sha256_file(args.output)}")


def command_self_test(_: argparse.Namespace) -> None:
    if CONFIG.max_looks != 11 or not CONFIG.envelope_valid:
        raise AssertionError("configuration invariant failed")
    for gamma in GAMMAS:
        schedule = alpha_schedule(gamma, CONFIG.max_looks, CONFIG.alpha_total)
        if len(schedule) != 11 or not np.all(schedule > 0.0):
            raise AssertionError("alpha schedule invariant failed")
    if not np.allclose(
        alpha_schedule(0.0, 11, 0.05), 0.05 / 11, rtol=0, atol=1e-15
    ):
        raise AssertionError("equal spending is wrong")
    # Exhaustively validate the cutoff shortcut against full confidence-bound
    # inversion on a smaller support for both declared observation models.
    for bound_model, rho in (
        ("robust_beta_binomial", 0.001),
        ("nominal_binomial", 0.0),
    ):
        for alpha_value in (0.001, 0.01, 0.05):
            if bound_model == "robust_beta_binomial":
                small_table = beta_binomial_upper_table(100, alpha_value, rho)
            else:
                small_table = binomial_upper_table(100, alpha_value)
            for mean in (0.01, 0.05, 0.20, 0.60):
                authorization_cutoff = maximum_count_below_cdf_level(
                    total=100,
                    mean=mean,
                    rho=rho,
                    alpha=alpha_value,
                    bound_model=bound_model,
                    inclusive=True,
                )
                expected_authorized = np.flatnonzero(small_table <= mean)
                expected_authorization_cutoff = (
                    int(expected_authorized[-1])
                    if expected_authorized.size
                    else -1
                )
                if authorization_cutoff != expected_authorization_cutoff:
                    raise AssertionError("authorization cutoff is not equivalent")
                undercoverage_cutoff = maximum_count_below_cdf_level(
                    total=100,
                    mean=mean,
                    rho=rho,
                    alpha=alpha_value,
                    bound_model=bound_model,
                    inclusive=False,
                )
                expected_undercovered = np.flatnonzero(small_table < mean)
                expected_undercoverage_cutoff = (
                    int(expected_undercovered[-1])
                    if expected_undercovered.size
                    else -1
                )
                if undercoverage_cutoff != expected_undercoverage_cutoff:
                    raise AssertionError("undercoverage cutoff is not equivalent")
        if bound_model == "robust_beta_binomial":
            small_table = beta_binomial_upper_table(100, 0.05, rho)
            for mean in np.linspace(0.01, 0.40, 20):
                concentration = 1.0 / rho - 1.0
                pmf = betabinom.pmf(
                    np.arange(101),
                    100,
                    mean * concentration,
                    (1.0 - mean) * concentration,
                )
                undercoverage = float(pmf[small_table < mean].sum())
                if undercoverage > 0.05 + 1e-9:
                    raise AssertionError("beta-binomial upper bound undercovers")
    qcrit = CONFIG.q_critical
    if not 0.0 < qcrit < CONFIG.erasure_tolerance:
        raise AssertionError("critical dropout out of range")
    if not math.isclose(
        service_recoverability(
            qcrit,
            CONFIG.archive_oligos,
            CONFIG.erasure_tolerance,
            CONFIG.declared_service_rho,
        ),
        CONFIG.recoverability_target,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise AssertionError("critical dropout inversion failed")
    if stable_seed(TUNE_NAMESPACE, 0.006, 1) == stable_seed(
        TEST_NAMESPACE, 0.006, 1
    ):
        raise AssertionError("seed namespaces collide")
    service_grid = np.asarray(
        [
            service_recoverability(
                q,
                CONFIG.archive_oligos,
                CONFIG.erasure_tolerance,
                CONFIG.declared_service_rho,
            )
            for q in np.linspace(0.0, CONFIG.erasure_tolerance, 25)
        ]
    )
    if np.any(np.diff(service_grid) > 1e-9):
        raise AssertionError("service recoverability is not nonincreasing")
    toy_strata = np.asarray([0, 0, 1, 1], dtype=int)
    toy_num = np.asarray([0, 1, 1, 1], dtype=float)
    toy_den = np.ones(4, dtype=float)
    toy_ratio_first = cluster_ratio_bound(
        toy_num,
        toy_den,
        strata=toy_strata,
        label="self-test-ratio",
        side="lower",
        alpha=0.05,
        replicates=1_000,
    )
    toy_ratio_second = cluster_ratio_bound(
        toy_num,
        toy_den,
        strata=toy_strata,
        label="self-test-ratio",
        side="lower",
        alpha=0.05,
        replicates=1_000,
    )
    toy_mean = paired_mean_bound(
        np.asarray([-1, 0, 0, 1], dtype=float),
        strata=toy_strata,
        label="self-test-paired",
        side="upper",
        alpha=0.05,
        replicates=1_000,
    )
    if (
        toy_ratio_first != toy_ratio_second
        or not 0.0 <= toy_ratio_first <= 1.0
        or not -1.0 <= toy_mean <= 1.0
    ):
        raise AssertionError("stratified bootstrap invariant failed")
    authorization_cutoffs, undercoverage_cutoffs = build_decision_cutoffs(
        "robust_beta_binomial"
    )
    if (
        authorization_cutoffs.shape
        != (len(GAMMAS), CONFIG.max_looks, CONFIG.max_looks)
        or undercoverage_cutoffs.shape
        != (
            len(GAMMAS),
            CONFIG.max_looks,
            len(HAZARDS),
            CONFIG.max_looks,
        )
    ):
        raise AssertionError("cutoff-table shape is wrong")
    init_worker(
        authorization_cutoffs,
        undercoverage_cutoffs,
        "robust_beta_binomial",
        qcrit,
    )
    first = simulate_profile(
        cohort="self_test",
        namespace="paper4-v8.8|self-test",
        hazard=0.006,
        episode=0,
        gamma=0.0,
    )
    second = simulate_profile(
        cohort="self_test",
        namespace="paper4-v8.8|self-test",
        hazard=0.006,
        episode=0,
        gamma=0.0,
    )
    if first != second or int(first["decisions"]) != 11:
        raise AssertionError("deterministic replay failed")
    print("PASS_V8_8_STRUCTURAL_CHALLENGE_SELF_TEST")


def run_and_write_all(
    *,
    cohort: str,
    namespace: str,
    missions_per_hazard: int,
    workers: int,
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = run_all_profiles(
        cohort=cohort,
        namespace=namespace,
        missions_per_hazard=missions_per_hazard,
        workers=workers,
        bound_model="robust_beta_binomial",
    )
    expected = len(HAZARDS) * missions_per_hazard * len(GAMMAS)
    if len(rows) != expected:
        raise AssertionError(f"wrong row count: {len(rows)} != {expected}")
    summary = summarize_profiles(rows, label_prefix=cohort)
    comparisons = all_profile_comparisons(rows, label_prefix=cohort)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "structural_challenge_episodes.csv.gz"
    summary_path = output_dir / "profile_summary.csv"
    comparisons_path = output_dir / "profile_vs_equal_comparisons.csv"
    write_csv_gz(episodes_path, rows)
    write_csv(summary_path, summary)
    write_csv(comparisons_path, comparisons)
    return rows, summary


def command_smoke(args: argparse.Namespace) -> None:
    marker = args.output_dir / "SMOKE_NOT_EVIDENCE.json"
    write_json(
        marker,
        {
            "study_id": STUDY_ID,
            "status": "SMOKE_NOT_EVIDENCE",
            "created_at": utc_now(),
            "missions_per_hazard": args.missions_per_hazard,
        },
        exclusive=True,
    )
    rows, summary = run_and_write_all(
        cohort="smoke",
        namespace="paper4-v8.8|smoke-not-evidence",
        missions_per_hazard=args.missions_per_hazard,
        workers=args.workers,
        output_dir=args.output_dir,
    )
    selection = select_tuning(summary)
    write_json(args.output_dir / "SMOKE_SELECTION_NOT_EVIDENCE.json", selection)
    paths = [
        marker,
        args.output_dir / "structural_challenge_episodes.csv.gz",
        args.output_dir / "profile_summary.csv",
        args.output_dir / "profile_vs_equal_comparisons.csv",
        args.output_dir / "SMOKE_SELECTION_NOT_EVIDENCE.json",
    ]
    write_output_manifest(
        args.output_dir,
        paths,
        command="smoke",
        extra={
            "rows": len(rows),
            "evidence_status": "NOT_EVIDENCE",
            "script_sha256": sha256_file(Path(__file__)),
        },
    )
    print(f"PASS_V8_8_SMOKE_NOT_EVIDENCE rows={len(rows)}")


def command_tune(args: argparse.Namespace) -> None:
    design = load_design_lock(args.design_lock)
    run_lock = args.output_dir / "TUNE_RUN_LOCK.json"
    write_json(
        run_lock,
        {
            "study_id": STUDY_ID,
            "status": "STARTED",
            "started_at": utc_now(),
            "namespace": TUNE_NAMESPACE,
            "missions_per_hazard": TUNE_MISSIONS_PER_HAZARD,
            "script_sha256": sha256_file(Path(__file__)),
            "design_lock_sha256": sha256_file(args.design_lock),
        },
        exclusive=True,
    )
    rows, summary = run_and_write_all(
        cohort="tune",
        namespace=TUNE_NAMESPACE,
        missions_per_hazard=TUNE_MISSIONS_PER_HAZARD,
        workers=args.workers,
        output_dir=args.output_dir,
    )
    selection = select_tuning(summary)
    selection_path = args.output_dir / "SELECTION.json"
    write_json(selection_path, selection)
    complete_run_lock(
        run_lock,
        selection_sha256=sha256_file(selection_path),
        selected_gamma=selection["selected_gamma"],
    )
    paths = [
        run_lock,
        args.output_dir / "structural_challenge_episodes.csv.gz",
        args.output_dir / "profile_summary.csv",
        args.output_dir / "profile_vs_equal_comparisons.csv",
        selection_path,
    ]
    write_output_manifest(
        args.output_dir,
        paths,
        command="tune",
        extra={
            "rows": len(rows),
            "script_sha256": sha256_file(Path(__file__)),
            "design_lock_sha256": sha256_file(args.design_lock),
            "selection_sha256": sha256_file(selection_path),
            "design_status": design["status"],
        },
    )
    print("PASS_V8_8_TUNING")
    print(json.dumps(selection, sort_keys=True))


def command_test(args: argparse.Namespace) -> None:
    load_design_lock(args.design_lock)
    selection = load_tuning(args.tune_dir, args.design_lock)
    if not isinstance(selection.get("selected_gamma"), (int, float)):
        raise SystemExit("NO_QUALIFIED_NONZERO_PROFILE_FROM_TUNING")
    run_lock = args.output_dir / "TEST_RUN_LOCK.json"
    write_json(
        run_lock,
        {
            "study_id": STUDY_ID,
            "status": "STARTED",
            "started_at": utc_now(),
            "namespace": TEST_NAMESPACE,
            "missions_per_hazard": TEST_MISSIONS_PER_HAZARD,
            "script_sha256": sha256_file(Path(__file__)),
            "design_lock_sha256": sha256_file(args.design_lock),
            "selection_sha256": sha256_file(args.tune_dir / "SELECTION.json"),
        },
        exclusive=True,
    )
    rows, summary = run_and_write_all(
        cohort="test",
        namespace=TEST_NAMESPACE,
        missions_per_hazard=TEST_MISSIONS_PER_HAZARD,
        workers=args.workers,
        output_dir=args.output_dir,
    )
    gates, result = test_gates(rows, summary, selection)
    gate_path = args.output_dir / "PRIMARY_GATE_TABLE.csv"
    result_path = args.output_dir / "PRIMARY_RESULT.json"
    selection_used_path = args.output_dir / "TUNING_SELECTION_USED.json"
    if gates:
        write_csv(gate_path, gates)
    else:
        write_json(
            args.output_dir / "PRIMARY_GATE_TABLE_NOT_APPLICABLE.json", result
        )
    write_json(result_path, result)
    write_json(selection_used_path, selection)
    complete_run_lock(
        run_lock,
        primary_status=result["status"],
        all_primary_gates_pass=result["all_primary_gates_pass"],
        primary_result_sha256=sha256_file(result_path),
    )
    paths = [
        run_lock,
        args.output_dir / "structural_challenge_episodes.csv.gz",
        args.output_dir / "profile_summary.csv",
        args.output_dir / "profile_vs_equal_comparisons.csv",
        result_path,
        selection_used_path,
    ]
    if gate_path.is_file():
        paths.append(gate_path)
    else:
        paths.append(args.output_dir / "PRIMARY_GATE_TABLE_NOT_APPLICABLE.json")
    write_output_manifest(
        args.output_dir,
        paths,
        command="test",
        extra={
            "rows": len(rows),
            "script_sha256": sha256_file(Path(__file__)),
            "design_lock_sha256": sha256_file(args.design_lock),
            "selection_sha256": sha256_file(args.tune_dir / "SELECTION.json"),
            "primary_result_sha256": sha256_file(result_path),
        },
    )
    print("PASS_V8_8_TEST_EXECUTION")
    print(json.dumps(result, sort_keys=True))


def command_control(args: argparse.Namespace) -> None:
    load_design_lock(args.design_lock)
    selection = load_tuning(args.tune_dir, args.design_lock)
    selected_gamma = selection.get("selected_gamma")
    if not isinstance(selected_gamma, (int, float)):
        raise SystemExit("offline control requires a selected profile")
    selected_gamma = float(selected_gamma)
    run_lock = args.output_dir / "OFFLINE_CONTROL_RUN_LOCK.json"
    write_json(
        run_lock,
        {
            "study_id": STUDY_ID,
            "status": "STARTED",
            "started_at": utc_now(),
            "namespace": CONTROL_NAMESPACE,
            "missions": CONTROL_MISSIONS,
            "true_observation_rho": CONFIG.actual_observation_rho,
            "declared_observation_rho": 0.0,
            "selected_gamma": selected_gamma,
            "design_lock_sha256": sha256_file(args.design_lock),
        },
        exclusive=True,
    )
    rows = run_one_profile(
        cohort="truth_aware_offline_control",
        namespace=CONTROL_NAMESPACE,
        hazard=0.006,
        missions=CONTROL_MISSIONS,
        gamma=selected_gamma,
        workers=args.workers,
        bound_model="nominal_binomial",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "offline_control_episodes.csv.gz"
    write_csv_gz(episodes_path, rows)
    sampling_failures = sum(
        int(row["sampling_undercoverage_looks"]) for row in rows
    )
    chain_failures = sum(int(row["realized_chain_failure_looks"]) for row in rows)
    false_safe = sum(int(row["false_safe_decisions"]) for row in rows)
    result = {
        "study_id": STUDY_ID,
        "status": (
            "OFFLINE_CONTROL_EXPOSED_MISSPECIFICATION"
            if sampling_failures > 0 and chain_failures > 0
            else "OFFLINE_CONTROL_INCONCLUSIVE"
        ),
        "selected_gamma": selected_gamma,
        "missions": len(rows),
        "sampling_undercoverage_looks": sampling_failures,
        "realized_chain_failure_looks": chain_failures,
        "under_target_authorizations": false_safe,
        "missions_with_under_target_authorization": sum(
            int(row["any_false_safe"]) for row in rows
        ),
        "pooled_into_primary_evidence": False,
        "runtime_detector": False,
        "interpretation": (
            "truth-aware offline verification control only; it does not detect "
            "unknown observation dependence during deployment"
        ),
    }
    result_path = args.output_dir / "OFFLINE_CONTROL_RESULT.json"
    write_json(result_path, result)
    complete_run_lock(
        run_lock,
        control_status=result["status"],
        control_result_sha256=sha256_file(result_path),
    )
    paths = [run_lock, episodes_path, result_path]
    write_output_manifest(
        args.output_dir,
        paths,
        command="offline-control",
        extra={
            "rows": len(rows),
            "script_sha256": sha256_file(Path(__file__)),
            "design_lock_sha256": sha256_file(args.design_lock),
            "selection_sha256": sha256_file(args.tune_dir / "SELECTION.json"),
            "control_result_sha256": sha256_file(result_path),
        },
    )
    print("PASS_V8_8_OFFLINE_CONTROL_EXECUTION")
    print(json.dumps(result, sort_keys=True))


def count_gzip_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in handle) - 1


def command_verify(args: argparse.Namespace) -> None:
    design = load_design_lock(args.design_lock)
    design_lock_sha256 = sha256_file(args.design_lock)
    tune_manifest = verify_output_manifest(args.tune_dir)
    test_manifest = verify_output_manifest(args.test_dir)
    control_manifest = verify_output_manifest(args.control_dir)
    manifests = {
        "tune": tune_manifest,
        "test": test_manifest,
        "offline-control": control_manifest,
    }
    for expected_command, manifest in manifests.items():
        if manifest.get("command") != expected_command:
            raise ValueError(f"wrong manifest command: {expected_command}")
        if manifest.get("script_sha256") != design["script_sha256"]:
            raise ValueError(f"script hash mismatch: {expected_command}")
        if manifest.get("design_lock_sha256") != design_lock_sha256:
            raise ValueError(f"design-lock mismatch: {expected_command}")
    run_lock_specs = (
        (args.tune_dir / "TUNE_RUN_LOCK.json", TUNE_NAMESPACE),
        (args.test_dir / "TEST_RUN_LOCK.json", TEST_NAMESPACE),
        (args.control_dir / "OFFLINE_CONTROL_RUN_LOCK.json", CONTROL_NAMESPACE),
    )
    for run_lock_path, expected_namespace in run_lock_specs:
        run_lock = json.loads(run_lock_path.read_text(encoding="utf-8"))
        if (
            run_lock.get("status") != "COMPLETED"
            or run_lock.get("namespace") != expected_namespace
            or run_lock.get("design_lock_sha256") != design_lock_sha256
        ):
            raise ValueError(f"invalid completed run lock: {run_lock_path}")
    selection = load_tuning(args.tune_dir, args.design_lock)
    test_selection = json.loads(
        (args.test_dir / "TUNING_SELECTION_USED.json").read_text(encoding="utf-8")
    )
    if selection != test_selection:
        raise ValueError("test did not use the locked tuning selection")
    expected_tune = len(HAZARDS) * TUNE_MISSIONS_PER_HAZARD * len(GAMMAS)
    expected_test = len(HAZARDS) * TEST_MISSIONS_PER_HAZARD * len(GAMMAS)
    observed_tune = count_gzip_rows(
        args.tune_dir / "structural_challenge_episodes.csv.gz"
    )
    observed_test = count_gzip_rows(
        args.test_dir / "structural_challenge_episodes.csv.gz"
    )
    observed_control = count_gzip_rows(
        args.control_dir / "offline_control_episodes.csv.gz"
    )
    if (observed_tune, observed_test, observed_control) != (
        expected_tune,
        expected_test,
        CONTROL_MISSIONS,
    ):
        raise ValueError("episode row count mismatch")
    primary = json.loads(
        (args.test_dir / "PRIMARY_RESULT.json").read_text(encoding="utf-8")
    )
    control = json.loads(
        (args.control_dir / "OFFLINE_CONTROL_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "study_id": STUDY_ID,
        "status": "PASS_V8_8_STRUCTURAL_CHALLENGE_CONFORMANCE",
        "verified_at": utc_now(),
        "design_lock_sha256": design_lock_sha256,
        "script_sha256": design["script_sha256"],
        "predecessor_sha256": design["predecessor_sha256"],
        "tune_rows": observed_tune,
        "test_rows": observed_test,
        "control_rows": observed_control,
        "selected_gamma": selection["selected_gamma"],
        "primary_status": primary["status"],
        "all_primary_gates_pass": primary["all_primary_gates_pass"],
        "offline_control_status": control["status"],
        "manifest_commands": [
            tune_manifest["command"],
            test_manifest["command"],
            control_manifest["command"],
        ],
        "frozen_v8_7_episode_records_read": False,
        "frozen_v8_7_episode_records_changed": False,
        "claim_ceiling": primary["claim_ceiling"],
    }
    write_json(args.output, summary)
    print("PASS_V8_8_STRUCTURAL_CHALLENGE_CONFORMANCE")
    print(json.dumps(summary, sort_keys=True))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def add_workers(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=max(1, min(24, (os.cpu_count() or 2) - 1)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(func=command_self_test)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--missions-per-hazard", type=positive_int, default=10)
    add_workers(smoke)
    smoke.set_defaults(func=command_smoke)

    lock = subparsers.add_parser("lock-design")
    lock.add_argument("--predecessor-zip", type=Path, required=True)
    lock.add_argument("--review-synthesis", type=Path, required=True)
    lock.add_argument("--sap", type=Path, required=True)
    lock.add_argument("--method-record", type=Path, required=True)
    lock.add_argument("--method-evidence", type=Path, required=True)
    lock.add_argument("--search-strategy", type=Path, required=True)
    lock.add_argument("--self-test-stdout", type=Path, required=True)
    lock.add_argument("--self-test-stderr", type=Path, required=True)
    lock.add_argument("--smoke-dir", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    lock.set_defaults(func=command_lock_design)

    tune = subparsers.add_parser("tune")
    tune.add_argument("--design-lock", type=Path, required=True)
    tune.add_argument("--output-dir", type=Path, required=True)
    add_workers(tune)
    tune.set_defaults(func=command_tune)

    test = subparsers.add_parser("test")
    test.add_argument("--design-lock", type=Path, required=True)
    test.add_argument("--tune-dir", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    add_workers(test)
    test.set_defaults(func=command_test)

    control = subparsers.add_parser("offline-control")
    control.add_argument("--design-lock", type=Path, required=True)
    control.add_argument("--tune-dir", type=Path, required=True)
    control.add_argument("--output-dir", type=Path, required=True)
    add_workers(control)
    control.set_defaults(func=command_control)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--design-lock", type=Path, required=True)
    verify.add_argument("--tune-dir", type=Path, required=True)
    verify.add_argument("--test-dir", type=Path, required=True)
    verify.add_argument("--control-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
