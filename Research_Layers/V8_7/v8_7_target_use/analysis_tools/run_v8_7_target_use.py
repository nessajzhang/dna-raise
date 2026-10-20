#!/usr/bin/env python3
"""Append-only Paper 4 V8.7 target-use and comparator study.

The study is a synthetic, model-internal qualification.  It imports unchanged
V8.6.4.1 functions from ``baseline_sources`` and never writes to the predecessor.
Complete 60-year missions are the independent units.  Tune and test namespaces
are disjoint and the final test can only start from a locked tuning selection.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline_sources"
sys.path.insert(0, str(BASELINE))

from audit_paper4_finite_horizon_certificate import (  # noqa: E402
    certified_next_dropout,
    critical_dropout,
    cumulative_dropout,
)
from paper4_archive_maintenance_seed import (  # noqa: E402
    Config,
    exact_binomial_recoverability,
    policy_hazard,
    posterior_update,
)


STUDY_ID = "paper4_nc_v8_7_target_use_comparator_20260905"
HAZARDS = (0.002, 0.006, 0.012)
BASE_POLICIES = ("posterior_mean", "posterior_conservative")
PERIODIC_INTERVALS = tuple(range(5, 61, 5))
TUNE_MISSIONS_PER_HAZARD = 5_000
TEST_MISSIONS_PER_HAZARD = 20_000
NEGATIVE_CONTROL_MISSIONS = 5_000
BOOTSTRAP_REPLICATES = 20_000
PRIMARY_FAMILY_ALPHA = 0.05
PRIMARY_GATE_COUNT = 6
ALPHA_GATE = PRIMARY_FAMILY_ALPHA / PRIMARY_GATE_COUNT
ELIGIBLE_RELATIVE_MARGIN = 0.50
ELIGIBLE_PASSAGE_FLOOR = 0.25
OVERALL_AUTHORIZATION_FLOOR = 0.05
REFRESH_GAIN_THRESHOLD = -0.50


@dataclass(frozen=True)
class StudyConfig:
    horizon_years: int = 60
    inspection_interval: int = 5
    sentinel_size: int = 300
    archive_oligos: int = 10_000
    erasure_tolerance: float = 0.10
    recoverability_target: float = 0.99
    posterior_quantile: float = 0.95
    alpha_total: float = 0.05

    @property
    def max_looks(self) -> int:
        return (self.horizon_years - 1) // self.inspection_interval

    @property
    def per_look_alpha(self) -> float:
        return self.alpha_total / self.max_looks

    @property
    def q_critical(self) -> float:
        return critical_dropout(
            self.archive_oligos,
            self.erasure_tolerance,
            self.recoverability_target,
        )

    def seed_config(self) -> Config:
        return Config(
            horizon_years=self.horizon_years,
            inspection_interval=self.inspection_interval,
            sentinel_size=self.sentinel_size,
            archive_oligos=self.archive_oligos,
            erasure_tolerance=self.erasure_tolerance,
            recoverability_target=self.recoverability_target,
            posterior_quantile=self.posterior_quantile,
            episodes=1,
        )


@dataclass(frozen=True)
class TruthEnvelope:
    true_detection_factor: float = 1.0
    declared_detection_floor: float = 1.0
    true_acceleration: float = 0.0
    declared_acceleration_ceiling: float = 0.0

    @property
    def declared_envelope_valid(self) -> bool:
        return (
            self.true_detection_factor >= self.declared_detection_floor
            and self.true_acceleration <= self.declared_acceleration_ceiling
        )


TARGET_ENVELOPE = TruthEnvelope()
NEGATIVE_ENVELOPE = TruthEnvelope(
    true_detection_factor=0.8,
    declared_detection_floor=1.0,
    true_acceleration=0.03,
    declared_acceleration_ceiling=0.0,
)


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


def exact_upper(successes: int, total: int, alpha: float = ALPHA_GATE) -> float:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial count")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if successes == total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, total - successes))


def cluster_ratio_bound(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    label: str,
    side: str,
    alpha: float = ALPHA_GATE,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> float:
    if numerators.ndim != 1 or denominators.ndim != 1:
        raise ValueError("cluster arrays must be one-dimensional")
    if numerators.size != denominators.size or numerators.size == 0:
        raise ValueError("cluster arrays must have equal nonzero length")
    if float(denominators.sum()) <= 0.0:
        return math.nan
    if side not in {"lower", "upper"}:
        raise ValueError("side must be lower or upper")
    # A nonparametric cluster bootstrap is exactly multinomial over the
    # observed cluster types.  Compressing identical (numerator, denominator)
    # pairs preserves that bootstrap law, removes any dependence on row order,
    # and avoids allocating an O(replicates * missions) index matrix.
    patterns, pattern_counts = np.unique(
        np.column_stack((numerators, denominators)),
        axis=0,
        return_counts=True,
    )
    probabilities = pattern_counts.astype(float) / numerators.size
    rng = np.random.default_rng(bootstrap_seed(label))
    values = np.empty(replicates, dtype=float)
    chunk = min(1_024, replicates)
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        resampled_counts = rng.multinomial(
            numerators.size, probabilities, size=stop - start
        )
        num = resampled_counts @ patterns[:, 0]
        den = resampled_counts @ patterns[:, 1]
        ratio = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        values[start:stop] = ratio
    quantile = alpha if side == "lower" else 1.0 - alpha
    return float(np.quantile(values, quantile))


def paired_mean_bound(
    values: np.ndarray,
    *,
    label: str,
    side: str,
    alpha: float = ALPHA_GATE,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> float:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired values must be a nonempty vector")
    if side not in {"lower", "upper"}:
        raise ValueError("side must be lower or upper")
    unique_values, value_counts = np.unique(values, return_counts=True)
    probabilities = value_counts.astype(float) / values.size
    rng = np.random.default_rng(bootstrap_seed(label))
    estimates = np.empty(replicates, dtype=float)
    chunk = min(1_024, replicates)
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        resampled_counts = rng.multinomial(
            values.size, probabilities, size=stop - start
        )
        estimates[start:stop] = (
            resampled_counts @ unique_values
        ) / values.size
    quantile = alpha if side == "lower" else 1.0 - alpha
    return float(np.quantile(estimates, quantile))


def service_success(config: StudyConfig, dropout: float) -> float:
    return exact_binomial_recoverability(
        config.archive_oligos, config.erasure_tolerance, dropout
    )


def common_uniforms(
    namespace: str, hazard: float, episode: int, config: StudyConfig
) -> tuple[int, np.ndarray]:
    seed = stable_seed(namespace, hazard, episode)
    rng = np.random.default_rng(seed)
    return seed, rng.random((config.max_looks, config.sentinel_size))


def simulate_adaptive_path(
    *,
    cohort: str,
    namespace: str,
    hazard: float,
    episode: int,
    root_seed: int,
    uniforms: np.ndarray,
    base_policy: str,
    screened: bool,
    config: StudyConfig,
    envelope: TruthEnvelope,
) -> dict[str, object]:
    if base_policy not in BASE_POLICIES:
        raise ValueError(f"unsupported base policy {base_policy}")
    if uniforms.shape != (config.max_looks, config.sentinel_size):
        raise ValueError("uniform array has wrong shape")

    grid = [0.0005 + index * (0.0195 / 195) for index in range(196)]
    weights = [1.0 / len(grid)] * len(grid)
    seed_config = config.seed_config()
    age = 0

    decisions = 0
    candidate_no_refresh = 0
    candidate_refresh = 0
    final_no_refresh = 0
    screen_overrides = 0
    refreshes = 0
    candidate_under_target = 0
    authorized_under_target = 0
    eligible_candidates = 0
    eligible_passes = 0
    sampling_undercoverage_looks = 0
    current_state_upper_failures = 0
    next_state_upper_failures = 0
    service_lower_failures = 0
    candidate_declared_envelope_invalid_looks = 0
    authorized_declared_envelope_invalid_looks = 0
    candidate_realized_chain_failure_looks = 0
    authorized_realized_chain_failure_looks = 0

    q_eligible = (1.0 - ELIGIBLE_RELATIVE_MARGIN) * config.q_critical

    for look_index in range(config.max_looks):
        age += config.inspection_interval
        decisions += 1
        true_current_q = cumulative_dropout(
            hazard, age, envelope.true_acceleration
        )
        observation_probability = float(
            np.clip(envelope.true_detection_factor * true_current_q, 0.0, 1.0)
        )
        unavailable = int(
            np.count_nonzero(uniforms[look_index] < observation_probability)
        )
        weights = posterior_update(
            grid, weights, age, unavailable, config.sentinel_size
        )
        decision_hazard = policy_hazard(
            base_policy, grid, weights, seed_config
        )
        next_age = age + config.inspection_interval
        base_requests_refresh = (
            service_success(
                config,
                1.0 - math.exp(-decision_hazard * next_age),
            )
            < config.recoverability_target
        )

        if base_requests_refresh:
            candidate_refresh += 1
            refreshes += 1
            age = 0
            continue

        candidate_no_refresh += 1
        true_next_q = cumulative_dropout(
            hazard, next_age, envelope.true_acceleration
        )
        true_next_service = service_success(config, true_next_q)
        true_next_under_target = true_next_service < config.recoverability_target
        if true_next_under_target:
            candidate_under_target += 1

        eligible = true_next_q <= q_eligible
        if eligible:
            eligible_candidates += 1

        observed_upper = math.nan
        current_true_upper = math.nan
        next_upper = math.nan
        certified_service = math.nan
        screen_passes = True
        sampling_failure = False
        current_upper_failure = False
        next_upper_failure = False
        service_failure = False
        declared_invalid = not envelope.declared_envelope_valid

        if screened:
            observed_upper, current_true_upper, next_upper = certified_next_dropout(
                unavailable,
                config.sentinel_size,
                age,
                next_age,
                config.per_look_alpha,
                envelope.declared_detection_floor,
                envelope.declared_acceleration_ceiling,
                "tight",
            )
            certified_service = service_success(config, next_upper)
            screen_passes = certified_service >= config.recoverability_target
            sampling_failure = observed_upper + 1e-15 < observation_probability
            current_upper_failure = current_true_upper + 1e-15 < true_current_q
            next_upper_failure = next_upper + 1e-15 < true_next_q
            service_failure = certified_service > true_next_service + 1e-15
            sampling_undercoverage_looks += int(sampling_failure)
            current_state_upper_failures += int(current_upper_failure)
            next_state_upper_failures += int(next_upper_failure)
            service_lower_failures += int(service_failure)
            candidate_declared_envelope_invalid_looks += int(declared_invalid)
            candidate_realized_chain_failure_looks += int(
                current_upper_failure or next_upper_failure or service_failure
            )

        if screen_passes:
            final_no_refresh += 1
            if eligible:
                eligible_passes += 1
            if true_next_under_target:
                authorized_under_target += 1
            if screened:
                authorized_declared_envelope_invalid_looks += int(declared_invalid)
                authorized_realized_chain_failure_looks += int(
                    current_upper_failure or next_upper_failure or service_failure
                )
        else:
            screen_overrides += 1
            refreshes += 1
            age = 0

    if decisions != config.max_looks:
        raise AssertionError("wrong decision count")
    if candidate_no_refresh + candidate_refresh != decisions:
        raise AssertionError("candidate accounting failed")
    if final_no_refresh + refreshes != decisions:
        raise AssertionError("final action accounting failed")
    if final_no_refresh > candidate_no_refresh:
        raise AssertionError("wrapper created an authorization")
    if authorized_under_target > candidate_under_target:
        raise AssertionError("wrapper increased under-target count")
    if eligible_passes > eligible_candidates:
        raise AssertionError("eligible passage accounting failed")
    if screened and final_no_refresh + screen_overrides != candidate_no_refresh:
        raise AssertionError("screen accounting failed")
    if not screened and screen_overrides != 0:
        raise AssertionError("unscreened path cannot override")

    policy = base_policy + ("+screen" if screened else "")
    return {
        "study_id": STUDY_ID,
        "cohort": cohort,
        "seed_namespace": namespace,
        "true_hazard": hazard,
        "episode": episode,
        "root_seed": root_seed,
        "policy": policy,
        "base_policy": base_policy,
        "screened": int(screened),
        "true_detection_factor": envelope.true_detection_factor,
        "declared_detection_floor": envelope.declared_detection_floor,
        "true_acceleration": envelope.true_acceleration,
        "declared_acceleration_ceiling": envelope.declared_acceleration_ceiling,
        "declared_envelope_valid": int(envelope.declared_envelope_valid),
        "decisions": decisions,
        "candidate_no_refresh": candidate_no_refresh,
        "candidate_refresh": candidate_refresh,
        "final_no_refresh": final_no_refresh,
        "screen_overrides": screen_overrides,
        "refreshes": refreshes,
        "candidate_under_target": candidate_under_target,
        "authorized_under_target": authorized_under_target,
        "any_authorized_under_target": int(authorized_under_target > 0),
        "eligible_candidates": eligible_candidates,
        "eligible_passes": eligible_passes,
        "sampling_undercoverage_looks": sampling_undercoverage_looks,
        "any_sampling_undercoverage": int(sampling_undercoverage_looks > 0),
        "current_state_upper_failures": current_state_upper_failures,
        "next_state_upper_failures": next_state_upper_failures,
        "service_lower_failures": service_lower_failures,
        "candidate_declared_envelope_invalid_looks": (
            candidate_declared_envelope_invalid_looks
        ),
        "authorized_declared_envelope_invalid_looks": (
            authorized_declared_envelope_invalid_looks
        ),
        "candidate_realized_chain_failure_looks": (
            candidate_realized_chain_failure_looks
        ),
        "authorized_realized_chain_failure_looks": (
            authorized_realized_chain_failure_looks
        ),
        "any_candidate_declared_envelope_invalid": int(
            candidate_declared_envelope_invalid_looks > 0
        ),
        "any_authorized_declared_envelope_invalid": int(
            authorized_declared_envelope_invalid_looks > 0
        ),
        "any_candidate_realized_chain_failure": int(
            candidate_realized_chain_failure_looks > 0
        ),
        "any_authorized_realized_chain_failure": int(
            authorized_realized_chain_failure_looks > 0
        ),
    }


def simulate_task(
    task: tuple[str, str, float, int, StudyConfig, TruthEnvelope]
) -> list[dict[str, object]]:
    cohort, namespace, hazard, episode, config, envelope = task
    root_seed, uniforms = common_uniforms(namespace, hazard, episode, config)
    rows: list[dict[str, object]] = []
    for base_policy in BASE_POLICIES:
        for screened in (False, True):
            rows.append(
                simulate_adaptive_path(
                    cohort=cohort,
                    namespace=namespace,
                    hazard=hazard,
                    episode=episode,
                    root_seed=root_seed,
                    uniforms=uniforms,
                    base_policy=base_policy,
                    screened=screened,
                    config=config,
                    envelope=envelope,
                )
            )
    return rows


def run_adaptive_cohort(
    *,
    cohort: str,
    namespace: str,
    hazards: Sequence[float],
    missions_per_hazard: int,
    workers: int,
    envelope: TruthEnvelope,
    config: StudyConfig,
) -> list[dict[str, object]]:
    tasks = [
        (cohort, namespace, hazard, episode, config, envelope)
        for hazard in hazards
        for episode in range(missions_per_hazard)
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task_rows in pool.map(simulate_task, tasks, chunksize=16):
            rows.extend(task_rows)
    return rows


def periodic_path(
    config: StudyConfig, hazard: float, interval: int
) -> dict[str, object]:
    age = 0
    refreshes = 0
    final_no_refresh = 0
    authorized_under_target = 0
    for _ in range(config.max_looks):
        age += config.inspection_interval
        if age >= interval:
            refreshes += 1
            age = 0
            continue
        final_no_refresh += 1
        true_next_q = cumulative_dropout(
            hazard, age + config.inspection_interval, 0.0
        )
        if service_success(config, true_next_q) < config.recoverability_target:
            authorized_under_target += 1
    if final_no_refresh + refreshes != config.max_looks:
        raise AssertionError("periodic action accounting failed")
    return {
        "true_hazard": hazard,
        "periodic_interval_years": interval,
        "decisions": config.max_looks,
        "final_no_refresh": final_no_refresh,
        "refreshes": refreshes,
        "authorized_under_target": authorized_under_target,
        "any_authorized_under_target": int(authorized_under_target > 0),
    }


def periodic_frontier(config: StudyConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for interval in PERIODIC_INTERVALS:
        strata = [periodic_path(config, hazard, interval) for hazard in HAZARDS]
        mean_refreshes = sum(float(row["refreshes"]) for row in strata) / len(strata)
        authorization_rate = sum(
            float(row["final_no_refresh"]) for row in strata
        ) / (len(strata) * config.max_looks)
        risk_pass_all = all(
            int(row["any_authorized_under_target"]) == 0 for row in strata
        )
        for row in strata:
            row.update(
                {
                    "mean_refreshes_equal_weight_domain": mean_refreshes,
                    "authorization_rate_equal_weight_domain": authorization_rate,
                    "risk_pass_all_strata": int(risk_pass_all),
                    "joint_nonvacuity_pass": int(
                        risk_pass_all
                        and authorization_rate >= OVERALL_AUTHORIZATION_FLOOR
                    ),
                    "risk_evaluation": "exact under declared deterministic periodic path",
                }
            )
            rows.append(row)
    return rows


def write_csv_gz(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as gzip_handle:
            with io.TextIOWrapper(
                gzip_handle, newline="", encoding="utf-8"
            ) as text_handle:
                writer = csv.DictWriter(text_handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_hash_list(path: Path, paths: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for target in paths:
        resolved = target.resolve()
        relative = os.path.relpath(resolved, path.parent.resolve())
        lines.append(f"{sha256_file(resolved)}  {relative}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def adaptive_summary(
    rows: Sequence[dict[str, object]], *, label_prefix: str
) -> list[dict[str, object]]:
    policies = tuple(
        base + suffix for base in BASE_POLICIES for suffix in ("", "+screen")
    )
    output: list[dict[str, object]] = []
    for policy in policies:
        group = sorted(
            [row for row in rows if str(row["policy"]) == policy],
            key=lambda row: (float(row["true_hazard"]), int(row["episode"])),
        )
        if not group:
            raise ValueError(f"missing policy rows: {policy}")
        by_hazard: dict[float, list[dict[str, object]]] = {
            hazard: [row for row in group if float(row["true_hazard"]) == hazard]
            for hazard in sorted({float(row["true_hazard"]) for row in group})
        }
        risk_upper: dict[str, float] = {}
        risk_rate: dict[str, float] = {}
        risk_events: dict[str, int] = {}
        risk_passes = []
        for hazard, stratum in by_hazard.items():
            events = sum(int(row["any_authorized_under_target"]) for row in stratum)
            key = f"h{int(round(hazard * 10000)):04d}"
            upper = exact_upper(events, len(stratum))
            risk_events[key] = events
            risk_rate[key] = events / len(stratum)
            risk_upper[key] = upper
            risk_passes.append(upper <= 0.05)

        auth_num = np.asarray(
            [float(row["final_no_refresh"]) for row in group], dtype=float
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
        overall_auth = float(auth_num.sum() / decisions.sum())
        overall_auth_lower = cluster_ratio_bound(
            auth_num,
            decisions,
            label=f"{label_prefix}|{policy}|overall-auth",
            side="lower",
        )
        eligible_pass = (
            float(eligible_num.sum() / eligible_den.sum())
            if float(eligible_den.sum()) > 0.0
            else math.nan
        )
        eligible_pass_lower = cluster_ratio_bound(
            eligible_num,
            eligible_den,
            label=f"{label_prefix}|{policy}|eligible-pass",
            side="lower",
        )
        screened = policy.endswith("+screen")
        preburden_qualified = (
            screened
            and all(risk_passes)
            and not math.isnan(eligible_pass_lower)
            and eligible_pass_lower >= ELIGIBLE_PASSAGE_FLOOR
            and overall_auth_lower >= OVERALL_AUTHORIZATION_FLOOR
        )
        output.append(
            {
                "policy": policy,
                "base_policy": str(group[0]["base_policy"]),
                "screened": int(screened),
                "missions": len(group),
                "hazards": len(by_hazard),
                "mean_refreshes": sum(float(row["refreshes"]) for row in group)
                / len(group),
                "mean_final_no_refresh": sum(
                    float(row["final_no_refresh"]) for row in group
                )
                / len(group),
                "overall_authorization_rate": overall_auth,
                "overall_authorization_lower": overall_auth_lower,
                "eligible_candidates": int(eligible_den.sum()),
                "eligible_passes": int(eligible_num.sum()),
                "eligible_passage_rate": eligible_pass,
                "eligible_passage_lower": eligible_pass_lower,
                "risk_events_json": json.dumps(risk_events, sort_keys=True),
                "risk_rates_json": json.dumps(risk_rate, sort_keys=True),
                "risk_upper_json": json.dumps(risk_upper, sort_keys=True),
                "risk_pass_all_strata": int(all(risk_passes)),
                "eligible_passage_gate": int(
                    screened
                    and not math.isnan(eligible_pass_lower)
                    and eligible_pass_lower >= ELIGIBLE_PASSAGE_FLOOR
                ),
                "overall_authorization_gate": int(
                    overall_auth_lower >= OVERALL_AUTHORIZATION_FLOOR
                ),
                "preburden_qualified": int(preburden_qualified),
                "candidate_declared_envelope_invalid_missions": sum(
                    int(row["any_candidate_declared_envelope_invalid"])
                    for row in group
                ),
                "authorized_declared_envelope_invalid_missions": sum(
                    int(row["any_authorized_declared_envelope_invalid"])
                    for row in group
                ),
                "candidate_realized_chain_failure_missions": sum(
                    int(row["any_candidate_realized_chain_failure"])
                    for row in group
                ),
                "authorized_realized_chain_failure_missions": sum(
                    int(row["any_authorized_realized_chain_failure"])
                    for row in group
                ),
                "sampling_undercoverage_missions": sum(
                    int(row["any_sampling_undercoverage"]) for row in group
                ),
                "alpha_gate": ALPHA_GATE,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "analysis_unit": "complete 60-year mission",
            }
        )
    return output


def periodic_aggregate(frontier: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for interval in PERIODIC_INTERVALS:
        group = [
            row
            for row in frontier
            if int(row["periodic_interval_years"]) == interval
        ]
        if len(group) != len(HAZARDS):
            raise ValueError("periodic frontier is incomplete")
        output.append(
            {
                "periodic_interval_years": interval,
                "mean_refreshes_equal_weight_domain": float(
                    group[0]["mean_refreshes_equal_weight_domain"]
                ),
                "authorization_rate_equal_weight_domain": float(
                    group[0]["authorization_rate_equal_weight_domain"]
                ),
                "risk_pass_all_strata": int(group[0]["risk_pass_all_strata"]),
                "joint_nonvacuity_pass": int(group[0]["joint_nonvacuity_pass"]),
                "stratum_events_json": json.dumps(
                    {
                        f"h{int(round(float(row['true_hazard']) * 10000)):04d}": int(
                            row["any_authorized_under_target"]
                        )
                        for row in group
                    },
                    sort_keys=True,
                ),
            }
        )
    return output


def select_tuning(
    summary: Sequence[dict[str, object]],
    periodic: Sequence[dict[str, object]],
) -> dict[str, object]:
    adaptive_candidates = [
        row
        for row in summary
        if int(row["screened"]) == 1 and int(row["preburden_qualified"]) == 1
    ]
    adaptive_candidates.sort(
        key=lambda row: (
            float(row["mean_refreshes"]),
            -float(row["overall_authorization_rate"]),
            str(row["policy"]),
        )
    )
    selected_adaptive = (
        str(adaptive_candidates[0]["policy"])
        if adaptive_candidates
        else "NO_QUALIFIED_ADAPTIVE_POLICY"
    )

    periodic_rows = periodic_aggregate(periodic)
    joint = [row for row in periodic_rows if int(row["joint_nonvacuity_pass"]) == 1]
    risk_only = [row for row in periodic_rows if int(row["risk_pass_all_strata"]) == 1]
    joint.sort(
        key=lambda row: (
            float(row["mean_refreshes_equal_weight_domain"]),
            int(row["periodic_interval_years"]),
        )
    )
    risk_only.sort(
        key=lambda row: (
            float(row["mean_refreshes_equal_weight_domain"]),
            int(row["periodic_interval_years"]),
        )
    )
    return {
        "study_id": STUDY_ID,
        "created_at": utc_now(),
        "selection_source": "tuning cohort only",
        "selected_adaptive_policy": selected_adaptive,
        "selected_periodic_joint_interval": (
            int(joint[0]["periodic_interval_years"])
            if joint
            else "NO_QUALIFIED_PERIODIC_JOINT"
        ),
        "selected_periodic_risk_only_interval": (
            int(risk_only[0]["periodic_interval_years"])
            if risk_only
            else "NO_QUALIFIED_PERIODIC_RISK_ONLY"
        ),
        "adaptive_qualified_candidates": [
            str(row["policy"]) for row in adaptive_candidates
        ],
        "periodic_joint_candidates": [
            int(row["periodic_interval_years"]) for row in joint
        ],
        "periodic_risk_only_candidates": [
            int(row["periodic_interval_years"]) for row in risk_only
        ],
        "thresholds": {
            "risk_upper": 0.05,
            "eligible_passage_lower": ELIGIBLE_PASSAGE_FLOOR,
            "overall_authorization_lower": OVERALL_AUTHORIZATION_FLOOR,
            "alpha_gate": ALPHA_GATE,
        },
    }


def periodic_refresh_for_hazard(
    config: StudyConfig, hazard: float, interval: int
) -> int:
    return int(periodic_path(config, hazard, interval)["refreshes"])


def test_gate_table(
    rows: Sequence[dict[str, object]],
    summary: Sequence[dict[str, object]],
    selection: dict[str, object],
    config: StudyConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected = str(selection["selected_adaptive_policy"])
    interval_value = selection["selected_periodic_risk_only_interval"]
    if selected == "NO_QUALIFIED_ADAPTIVE_POLICY":
        return [], {
            "study_id": STUDY_ID,
            "status": "NO_QUALIFIED_ADAPTIVE_POLICY_FROM_TUNING",
            "all_primary_gates_pass": False,
        }
    if not isinstance(interval_value, int):
        return [], {
            "study_id": STUDY_ID,
            "status": "NO_QUALIFIED_PERIODIC_RISK_ONLY_FROM_TUNING",
            "all_primary_gates_pass": False,
        }

    selected_summary = next(row for row in summary if row["policy"] == selected)
    risk_upper = json.loads(str(selected_summary["risk_upper_json"]))
    gates: list[dict[str, object]] = []
    for hazard in HAZARDS:
        key = f"h{int(round(hazard * 10000)):04d}"
        value = float(risk_upper[key])
        gates.append(
            {
                "gate": f"risk_{key}",
                "estimate_or_bound": value,
                "operator": "<=",
                "threshold": 0.05,
                "pass": int(value <= 0.05),
                "method": "one-sided Clopper-Pearson upper bound",
            }
        )

    eligible_value = float(selected_summary["eligible_passage_lower"])
    gates.append(
        {
            "gate": "eligible_passage",
            "estimate_or_bound": eligible_value,
            "operator": ">=",
            "threshold": ELIGIBLE_PASSAGE_FLOOR,
            "pass": int(eligible_value >= ELIGIBLE_PASSAGE_FLOOR),
            "method": "complete-mission cluster-bootstrap lower bound",
        }
    )
    auth_value = float(selected_summary["overall_authorization_lower"])
    gates.append(
        {
            "gate": "overall_authorization",
            "estimate_or_bound": auth_value,
            "operator": ">=",
            "threshold": OVERALL_AUTHORIZATION_FLOOR,
            "pass": int(auth_value >= OVERALL_AUTHORIZATION_FLOOR),
            "method": "complete-mission cluster-bootstrap lower bound",
        }
    )

    selected_rows = sorted(
        [row for row in rows if str(row["policy"]) == selected],
        key=lambda row: (float(row["true_hazard"]), int(row["episode"])),
    )
    differences = np.asarray(
        [
            float(row["refreshes"])
            - periodic_refresh_for_hazard(
                config, float(row["true_hazard"]), interval_value
            )
            for row in selected_rows
        ],
        dtype=float,
    )
    mean_difference = float(differences.mean())
    upper = paired_mean_bound(
        differences,
        label=f"test|{selected}|vs-periodic-{interval_value}|refresh",
        side="upper",
    )
    gates.append(
        {
            "gate": "refresh_gain",
            "estimate_or_bound": upper,
            "point_estimate": mean_difference,
            "operator": "<",
            "threshold": REFRESH_GAIN_THRESHOLD,
            "pass": int(upper < REFRESH_GAIN_THRESHOLD),
            "method": "paired complete-mission bootstrap upper bound",
        }
    )
    all_pass = all(int(row["pass"]) == 1 for row in gates)
    joint_interval = selection["selected_periodic_joint_interval"]
    capability_status = (
        "QUALIFIED_ADAPTIVE_SCREEN_WITH_NO_JOINT_PERIODIC_COMPARATOR"
        if all_pass and joint_interval == "NO_QUALIFIED_PERIODIC_JOINT"
        else (
            "PRIMARY_GATES_PASS_BUT_JOINT_PERIODIC_EXISTS"
            if all_pass
            else "PRIMARY_GATES_FAIL"
        )
    )
    return gates, {
        "study_id": STUDY_ID,
        "status": capability_status,
        "all_primary_gates_pass": all_pass,
        "selected_adaptive_policy": selected,
        "selected_periodic_risk_only_interval": interval_value,
        "selected_periodic_joint_interval": joint_interval,
        "refresh_difference_point_estimate": mean_difference,
        "refresh_difference_upper": upper,
        "claim_ceiling": (
            "model-internal target-use comparison only; physical qualification NOT_RUN"
        ),
    }


def key_audit(
    rows: Sequence[dict[str, object]],
    *,
    hazards: Sequence[float],
    missions_per_hazard: int,
) -> None:
    expected = {
        (hazard, episode, base + suffix)
        for hazard in hazards
        for episode in range(missions_per_hazard)
        for base in BASE_POLICIES
        for suffix in ("", "+screen")
    }
    actual = {
        (float(row["true_hazard"]), int(row["episode"]), str(row["policy"]))
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise AssertionError("adaptive cohort key set is incomplete or duplicated")
    for row in rows:
        if int(row["decisions"]) != StudyConfig().max_looks:
            raise AssertionError("row has wrong opportunity count")
        if int(row["final_no_refresh"]) + int(row["refreshes"]) != int(
            row["decisions"]
        ):
            raise AssertionError("row action counts do not reconcile")


def seed_ledger(
    *, namespace: str, hazards: Sequence[float], missions_per_hazard: int
) -> dict[str, object]:
    first_last = {}
    for hazard in hazards:
        first_last[f"{hazard:.3f}"] = {
            "first_episode": 0,
            "first_seed": stable_seed(namespace, hazard, 0),
            "last_episode": missions_per_hazard - 1,
            "last_seed": stable_seed(namespace, hazard, missions_per_hazard - 1),
        }
    return {
        "study_id": STUDY_ID,
        "namespace": namespace,
        "missions_per_hazard": missions_per_hazard,
        "hazards": list(hazards),
        "derivation": "uint64(first 8 bytes SHA256(study|namespace|hazard|episode))",
        "first_last": first_last,
    }


def write_run_manifest(
    output_dir: Path,
    *,
    command: str,
    rows_path: Path,
    summary_path: Path,
    extra_paths: Iterable[Path],
    config: StudyConfig,
) -> None:
    paths = [rows_path, summary_path, *extra_paths]
    write_json(
        output_dir / "RUN_MANIFEST.json",
        {
            "study_id": STUDY_ID,
            "created_at": utc_now(),
            "command": command,
            "config": asdict(config),
            "alpha_gate": ALPHA_GATE,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "runtime": {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "numpy": package_version("numpy"),
                "scipy": package_version("scipy"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NOT_SLURM"),
                "slurm_cpus_per_task": os.environ.get(
                    "SLURM_CPUS_PER_TASK", "NOT_SLURM"
                ),
            },
            "script_sha256": sha256_file(Path(__file__)),
            "files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
        },
    )


def verify_hash_list(path: Path) -> None:
    base = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        target = (base / rel.strip().lstrip("*")).resolve()
        if sha256_file(target) != expected.upper():
            raise ValueError(f"hash mismatch: {target}")


def load_locked_selection(
    selection_path: Path, tuning_lock: Path
) -> dict[str, object]:
    verify_hash_list(tuning_lock)
    expected_path = (tuning_lock.parent / "SELECTION.json").resolve()
    if selection_path.resolve() != expected_path:
        raise ValueError("selection is not the SELECTION.json covered by tuning lock")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("study_id") != STUDY_ID:
        raise ValueError("selection belongs to another study")
    return selection


def create_exclusive_lock(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_and_write(
    *,
    cohort: str,
    namespace: str,
    hazards: Sequence[float],
    missions_per_hazard: int,
    envelope: TruthEnvelope,
    workers: int,
    output_dir: Path,
    command: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    config = StudyConfig()
    rows = run_adaptive_cohort(
        cohort=cohort,
        namespace=namespace,
        hazards=hazards,
        missions_per_hazard=missions_per_hazard,
        workers=workers,
        envelope=envelope,
        config=config,
    )
    key_audit(rows, hazards=hazards, missions_per_hazard=missions_per_hazard)
    summary = adaptive_summary(rows, label_prefix=cohort)
    periodic = periodic_frontier(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "adaptive_episodes.csv.gz"
    summary_path = output_dir / "adaptive_summary.csv"
    periodic_path_out = output_dir / "periodic_frontier.csv"
    seed_path = output_dir / "SEED_LEDGER.json"
    write_csv_gz(rows_path, rows)
    write_csv(summary_path, summary)
    write_csv(periodic_path_out, periodic)
    write_json(
        seed_path,
        seed_ledger(
            namespace=namespace,
            hazards=hazards,
            missions_per_hazard=missions_per_hazard,
        ),
    )
    write_run_manifest(
        output_dir,
        command=command,
        rows_path=rows_path,
        summary_path=summary_path,
        extra_paths=[periodic_path_out, seed_path],
        config=config,
    )
    return rows, summary, periodic


def command_tune(args: argparse.Namespace) -> None:
    lock = args.output_dir / "TUNE_RUN_LOCK.json"
    create_exclusive_lock(
        lock,
        {
            "study_id": STUDY_ID,
            "started_at": utc_now(),
            "namespace": "paper4-v8.7|tune|20260905",
            "missions_per_hazard": TUNE_MISSIONS_PER_HAZARD,
            "script_sha256": sha256_file(Path(__file__)),
        },
    )
    rows, summary, periodic = run_and_write(
        cohort="tune",
        namespace="paper4-v8.7|tune|20260905",
        hazards=HAZARDS,
        missions_per_hazard=TUNE_MISSIONS_PER_HAZARD,
        envelope=TARGET_ENVELOPE,
        workers=args.workers,
        output_dir=args.output_dir,
        command="tune",
    )
    del rows
    selection = select_tuning(summary, periodic)
    selection["adaptive_summary_sha256"] = sha256_file(
        args.output_dir / "adaptive_summary.csv"
    )
    selection["adaptive_episodes_sha256"] = sha256_file(
        args.output_dir / "adaptive_episodes.csv.gz"
    )
    selection["periodic_frontier_sha256"] = sha256_file(
        args.output_dir / "periodic_frontier.csv"
    )
    write_json(args.output_dir / "SELECTION.json", selection)
    write_hash_list(
        args.output_dir / "TUNING_LOCK.sha256",
        [
            args.output_dir / "TUNE_RUN_LOCK.json",
            args.output_dir / "adaptive_episodes.csv.gz",
            args.output_dir / "adaptive_summary.csv",
            args.output_dir / "periodic_frontier.csv",
            args.output_dir / "SEED_LEDGER.json",
            args.output_dir / "RUN_MANIFEST.json",
            args.output_dir / "SELECTION.json",
        ],
    )
    print("PASS_V8_7_TUNING")
    print(json.dumps(selection, sort_keys=True))


def command_test(args: argparse.Namespace) -> None:
    verify_hash_list(args.design_lock)
    selection = load_locked_selection(args.selection, args.tuning_lock)
    verify_hash_list(args.negative_control_lock)
    negative_result_path = (
        args.negative_control_lock.parent / "NEGATIVE_CONTROL_RESULT.json"
    ).resolve()
    negative_result = json.loads(negative_result_path.read_text(encoding="utf-8"))
    if negative_result.get("study_id") != STUDY_ID:
        raise SystemExit("negative control belongs to another study")
    if negative_result.get("selected_policy") != selection.get(
        "selected_adaptive_policy"
    ):
        raise SystemExit("negative control used a different selected policy")
    if bool(negative_result.get("pooled_into_target_use")):
        raise SystemExit("negative control cannot be pooled into target-use evidence")
    if not bool(negative_result.get("negative_control_detected")):
        raise SystemExit("test requires a detected, locked negative control")
    lock = args.output_dir / "TEST_RUN_LOCK.json"
    create_exclusive_lock(
        lock,
        {
            "study_id": STUDY_ID,
            "started_at": utc_now(),
            "namespace": "paper4-v8.7|test|20260905",
            "missions_per_hazard": TEST_MISSIONS_PER_HAZARD,
            "script_sha256": sha256_file(Path(__file__)),
            "selection_sha256": sha256_file(args.selection),
            "design_lock_sha256": sha256_file(args.design_lock),
            "tuning_lock_sha256": sha256_file(args.tuning_lock),
            "negative_control_lock_sha256": sha256_file(
                args.negative_control_lock
            ),
            "negative_control_result_sha256": sha256_file(
                negative_result_path
            ),
        },
    )
    rows, summary, periodic = run_and_write(
        cohort="test",
        namespace="paper4-v8.7|test|20260905",
        hazards=HAZARDS,
        missions_per_hazard=TEST_MISSIONS_PER_HAZARD,
        envelope=TARGET_ENVELOPE,
        workers=args.workers,
        output_dir=args.output_dir,
        command="test",
    )
    gates, result = test_gate_table(rows, summary, selection, StudyConfig())
    if gates:
        write_csv(args.output_dir / "PRIMARY_GATE_TABLE.csv", gates)
    else:
        write_json(args.output_dir / "PRIMARY_GATE_TABLE_NOT_APPLICABLE.json", result)
    write_json(args.output_dir / "PRIMARY_RESULT.json", result)
    write_json(args.output_dir / "TUNING_SELECTION_USED.json", selection)
    write_csv(args.output_dir / "periodic_frontier.csv", periodic)
    test_outputs = [
        args.output_dir / "TEST_RUN_LOCK.json",
        args.output_dir / "adaptive_episodes.csv.gz",
        args.output_dir / "adaptive_summary.csv",
        args.output_dir / "periodic_frontier.csv",
        args.output_dir / "SEED_LEDGER.json",
        args.output_dir / "RUN_MANIFEST.json",
        args.output_dir / "PRIMARY_RESULT.json",
        args.output_dir / "TUNING_SELECTION_USED.json",
    ]
    gate_path = args.output_dir / "PRIMARY_GATE_TABLE.csv"
    if gate_path.is_file():
        test_outputs.append(gate_path)
    else:
        test_outputs.append(
            args.output_dir / "PRIMARY_GATE_TABLE_NOT_APPLICABLE.json"
        )
    write_hash_list(args.output_dir / "TEST_OUTPUT_LOCK.sha256", test_outputs)
    print("PASS_V8_7_TEST_EXECUTION")
    print(json.dumps(result, sort_keys=True))


def command_negative_control(args: argparse.Namespace) -> None:
    verify_hash_list(args.design_lock)
    selection = load_locked_selection(args.selection, args.tuning_lock)
    selected = str(selection.get("selected_adaptive_policy"))
    if selected == "NO_QUALIFIED_ADAPTIVE_POLICY":
        raise SystemExit("negative control requires a tuning-selected adaptive policy")
    lock = args.output_dir / "NEGATIVE_CONTROL_RUN_LOCK.json"
    create_exclusive_lock(
        lock,
        {
            "study_id": STUDY_ID,
            "started_at": utc_now(),
            "namespace": "paper4-v8.7|negative-control|20260905",
            "missions": NEGATIVE_CONTROL_MISSIONS,
            "script_sha256": sha256_file(Path(__file__)),
            "selection_sha256": sha256_file(args.selection),
            "design_lock_sha256": sha256_file(args.design_lock),
            "tuning_lock_sha256": sha256_file(args.tuning_lock),
        },
    )
    rows, summary, _ = run_and_write(
        cohort="negative_control",
        namespace="paper4-v8.7|negative-control|20260905",
        hazards=(0.006,),
        missions_per_hazard=NEGATIVE_CONTROL_MISSIONS,
        envelope=NEGATIVE_ENVELOPE,
        workers=args.workers,
        output_dir=args.output_dir,
        command="negative-control",
    )
    selected_rows = [row for row in rows if str(row["policy"]) == selected]
    candidate_invalid = sum(
        int(row["candidate_declared_envelope_invalid_looks"])
        for row in selected_rows
    )
    realized_failures = sum(
        int(row["candidate_realized_chain_failure_looks"])
        for row in selected_rows
    )
    result = {
        "study_id": STUDY_ID,
        "selected_policy": selected,
        "missions": len(selected_rows),
        "candidate_declared_envelope_invalid_looks": candidate_invalid,
        "candidate_realized_chain_failure_looks": realized_failures,
        "negative_control_detected": bool(
            candidate_invalid > 0 and realized_failures > 0
        ),
        "pooled_into_target_use": False,
    }
    write_json(args.output_dir / "NEGATIVE_CONTROL_RESULT.json", result)
    write_hash_list(
        args.output_dir / "NEGATIVE_CONTROL_OUTPUT_LOCK.sha256",
        [
            args.output_dir / "NEGATIVE_CONTROL_RUN_LOCK.json",
            args.output_dir / "adaptive_episodes.csv.gz",
            args.output_dir / "adaptive_summary.csv",
            args.output_dir / "periodic_frontier.csv",
            args.output_dir / "SEED_LEDGER.json",
            args.output_dir / "RUN_MANIFEST.json",
            args.output_dir / "NEGATIVE_CONTROL_RESULT.json",
        ],
    )
    if not result["negative_control_detected"]:
        raise SystemExit("NEGATIVE_CONTROL_NOT_DETECTED")
    print("PASS_V8_7_NEGATIVE_CONTROL")
    print(json.dumps(result, sort_keys=True))


def command_smoke(args: argparse.Namespace) -> None:
    marker = args.output_dir / "SMOKE_NOT_EVIDENCE.json"
    create_exclusive_lock(
        marker,
        {
            "study_id": STUDY_ID,
            "created_at": utc_now(),
            "status": "SMOKE_NOT_EVIDENCE",
            "missions_per_hazard": args.missions_per_hazard,
        },
    )
    rows, summary, periodic = run_and_write(
        cohort="smoke",
        namespace="paper4-v8.7|smoke|20260905",
        hazards=HAZARDS,
        missions_per_hazard=args.missions_per_hazard,
        envelope=TARGET_ENVELOPE,
        workers=args.workers,
        output_dir=args.output_dir,
        command="smoke",
    )
    selection = select_tuning(summary, periodic)
    write_json(args.output_dir / "SMOKE_SELECTION_NOT_EVIDENCE.json", selection)
    print(f"PASS_V8_7_SMOKE rows={len(rows)}")


def command_self_test(_: argparse.Namespace) -> None:
    config = StudyConfig()
    assert config.max_looks == 11
    assert math.isclose(config.per_look_alpha, 0.05 / 11.0)
    assert 0.0 < config.q_critical < config.erasure_tolerance
    assert stable_seed("a", 0.002, 1) == stable_seed("a", 0.002, 1)
    assert stable_seed("a", 0.002, 1) != stable_seed("b", 0.002, 1)
    assert stable_seed("paper4-v8.7|tune|20260905", 0.002, 0) != stable_seed(
        "paper4-v8.7|test|20260905", 0.002, 0
    )
    root_seed, uniforms = common_uniforms("self-test", 0.006, 0, config)
    first = simulate_adaptive_path(
        cohort="self-test",
        namespace="self-test",
        hazard=0.006,
        episode=0,
        root_seed=root_seed,
        uniforms=uniforms,
        base_policy="posterior_mean",
        screened=True,
        config=config,
        envelope=TARGET_ENVELOPE,
    )
    second = simulate_adaptive_path(
        cohort="self-test",
        namespace="self-test",
        hazard=0.006,
        episode=0,
        root_seed=root_seed,
        uniforms=uniforms,
        base_policy="posterior_mean",
        screened=True,
        config=config,
        envelope=TARGET_ENVELOPE,
    )
    assert first == second
    assert int(first["declared_envelope_valid"]) == 1
    negative_candidate_invalid = 0
    negative_realized_failures = 0
    for episode in range(64):
        negative_seed, negative_uniforms = common_uniforms(
            "self-test-negative", 0.006, episode, config
        )
        negative = simulate_adaptive_path(
            cohort="self-test-negative",
            namespace="self-test-negative",
            hazard=0.006,
            episode=episode,
            root_seed=negative_seed,
            uniforms=negative_uniforms,
            base_policy="posterior_mean",
            screened=True,
            config=config,
            envelope=NEGATIVE_ENVELOPE,
        )
        assert int(negative["declared_envelope_valid"]) == 0
        negative_candidate_invalid += int(
            negative["candidate_declared_envelope_invalid_looks"]
        )
        negative_realized_failures += int(
            negative["candidate_realized_chain_failure_looks"]
        )
    assert negative_candidate_invalid > 0
    assert negative_realized_failures > 0
    frontier = periodic_frontier(config)
    aggregate = periodic_aggregate(frontier)
    assert len(aggregate) == len(PERIODIC_INTERVALS)
    five = next(row for row in aggregate if row["periodic_interval_years"] == 5)
    assert int(five["risk_pass_all_strata"]) == 1
    assert float(five["authorization_rate_equal_weight_domain"]) == 0.0
    assert exact_upper(0, 20_000) < 0.001
    num = np.asarray([0.0, 1.0, 1.0, 0.0])
    den = np.ones(4)
    a = cluster_ratio_bound(num, den, label="test", side="lower", replicates=500)
    b = cluster_ratio_bound(num, den, label="test", side="lower", replicates=500)
    assert a == b
    assert a == cluster_ratio_bound(
        num[::-1], den[::-1], label="test", side="lower", replicates=500
    )
    values = np.asarray([-1.0, 0.0, -2.0, -1.0])
    c = paired_mean_bound(values, label="test", side="upper", replicates=500)
    d = paired_mean_bound(values, label="test", side="upper", replicates=500)
    assert c == d
    assert c == paired_mean_bound(
        values[::-1], label="test", side="upper", replicates=500
    )
    print("PASS_V8_7_TARGET_USE_SELF_TEST")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("self-test").set_defaults(func=command_self_test)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--missions-per-hazard", type=int, default=10)
    smoke.add_argument("--workers", type=int, default=2)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.set_defaults(func=command_smoke)

    tune = sub.add_parser("tune")
    tune.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    tune.add_argument("--output-dir", type=Path, required=True)
    tune.set_defaults(func=command_tune)

    test = sub.add_parser("test")
    test.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    test.add_argument("--output-dir", type=Path, required=True)
    test.add_argument("--selection", type=Path, required=True)
    test.add_argument("--design-lock", type=Path, required=True)
    test.add_argument("--tuning-lock", type=Path, required=True)
    test.add_argument("--negative-control-lock", type=Path, required=True)
    test.set_defaults(func=command_test)

    negative = sub.add_parser("negative-control")
    negative.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    negative.add_argument("--output-dir", type=Path, required=True)
    negative.add_argument("--selection", type=Path, required=True)
    negative.add_argument("--design-lock", type=Path, required=True)
    negative.add_argument("--tuning-lock", type=Path, required=True)
    negative.set_defaults(func=command_negative_control)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "missions_per_hazard", 1) <= 0:
        raise SystemExit("missions per hazard must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
