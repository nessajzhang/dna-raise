#!/usr/bin/env python3
"""Paired audit of the finite-horizon screen around existing policies.

This is a synthetic policy-logic experiment. It does not estimate physical DNA
degradation, observation-law defect, or action-envelope validity. The complete
60-year mission is the independent analysis unit.
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

from audit_paper4_finite_horizon_certificate import certified_next_dropout
from paper4_archive_maintenance_seed import (
    Config,
    HAZARDS,
    dropout_probability,
    exact_binomial_recoverability,
    is_action_year,
    policy_hazard,
    posterior_update,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_POLICIES = ("posterior_mean", "posterior_conservative")
Z_975 = 1.959963984540054
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BASE_SEED = 20260902
ROOT_RANDOM_SEED = 2026090201


@dataclass(frozen=True)
class WrapperConfig:
    horizon_years: int = 60
    inspection_interval: int = 5
    sentinel_size: int = 300
    archive_oligos: int = 10_000
    erasure_tolerance: float = 0.10
    recoverability_target: float = 0.99
    posterior_quantile: float = 0.95
    alpha_total: float = 0.05
    episodes: int = 5_000

    @property
    def max_looks(self) -> int:
        return (self.horizon_years - 1) // self.inspection_interval

    @property
    def per_look_alpha(self) -> float:
        return self.alpha_total / self.max_looks

    def seed_config(self) -> Config:
        return Config(
            horizon_years=self.horizon_years,
            inspection_interval=self.inspection_interval,
            sentinel_size=self.sentinel_size,
            archive_oligos=self.archive_oligos,
            erasure_tolerance=self.erasure_tolerance,
            recoverability_target=self.recoverability_target,
            posterior_quantile=self.posterior_quantile,
            episodes=self.episodes,
        )


def stable_seed(label: str, base: int = ROOT_RANDOM_SEED) -> int:
    payload = f"{base}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def common_uniforms(
    true_hazard: float, episode: int, config: WrapperConfig
) -> np.ndarray:
    """One root stream shared across base policies and screened paths."""
    seed = stable_seed(f"uniforms|{true_hazard:.6f}|{episode}")
    rng = np.random.default_rng(seed)
    return rng.random((config.max_looks, config.sentinel_size))


def recoverability(config: WrapperConfig, hazard: float, age: float) -> float:
    return exact_binomial_recoverability(
        config.archive_oligos,
        config.erasure_tolerance,
        dropout_probability(hazard, age),
    )


def simulate_path(
    config: WrapperConfig,
    true_hazard: float,
    policy: str,
    uniforms: np.ndarray,
    *,
    screened: bool,
) -> dict[str, int]:
    if policy not in BASE_POLICIES:
        raise ValueError(f"unsupported base policy: {policy}")
    if uniforms.shape != (config.max_looks, config.sentinel_size):
        raise ValueError("uniform array does not match the declared look/sample design")

    seed_config = config.seed_config()
    grid = [0.0005 + index * (0.0195 / 195) for index in range(196)]
    weights = [1.0 / len(grid)] * len(grid)
    age = 0
    look_index = 0
    decisions = 0
    candidate_no_refresh = 0
    candidate_refresh = 0
    final_no_refresh = 0
    screen_overrides = 0
    refreshes = 0
    candidate_under_target = 0
    authorized_under_target = 0
    any_candidate_under_target = False
    any_authorized_under_target = False
    violation_years = 0
    any_violation = False

    for year in range(1, config.horizon_years + 1):
        age += 1
        if recoverability(config, true_hazard, age) < config.recoverability_target:
            violation_years += 1
            any_violation = True

        if not is_action_year(year, config.horizon_years, config.inspection_interval):
            continue

        decisions += 1
        q = dropout_probability(true_hazard, age)
        unavailable = int(np.count_nonzero(uniforms[look_index] < q))
        look_index += 1
        weights = posterior_update(
            grid, weights, age, unavailable, config.sentinel_size
        )
        decision_hazard = policy_hazard(policy, grid, weights, seed_config)
        next_age = age + config.inspection_interval
        base_requests_refresh = (
            recoverability(config, decision_hazard, next_age)
            < config.recoverability_target
        )

        if base_requests_refresh:
            candidate_refresh += 1
            refresh = True
        else:
            candidate_no_refresh += 1
            true_next_under_target = (
                recoverability(config, true_hazard, next_age)
                < config.recoverability_target
            )
            if true_next_under_target:
                candidate_under_target += 1
                any_candidate_under_target = True

            screen_passes = True
            if screened:
                _, _, next_dropout_upper = certified_next_dropout(
                    unavailable,
                    config.sentinel_size,
                    age,
                    next_age,
                    config.per_look_alpha,
                    detection_floor=1.0,
                    acceleration_ceiling=0.0,
                )
                screen_passes = (
                    exact_binomial_recoverability(
                        config.archive_oligos,
                        config.erasure_tolerance,
                        next_dropout_upper,
                    )
                    >= config.recoverability_target
                )

            refresh = not screen_passes
            if refresh:
                screen_overrides += 1
            else:
                final_no_refresh += 1
                if true_next_under_target:
                    authorized_under_target += 1
                    any_authorized_under_target = True

        if refresh:
            refreshes += 1
            age = 0

    assert look_index == config.max_looks
    assert decisions == config.max_looks
    assert candidate_no_refresh + candidate_refresh == decisions
    assert final_no_refresh + refreshes == decisions
    assert final_no_refresh <= candidate_no_refresh
    assert screen_overrides <= candidate_no_refresh
    assert authorized_under_target <= candidate_under_target
    if screened:
        assert refreshes == candidate_refresh + screen_overrides
        assert final_no_refresh == candidate_no_refresh - screen_overrides
    else:
        assert screen_overrides == 0
        assert final_no_refresh == candidate_no_refresh

    return {
        "decisions": decisions,
        "candidate_no_refresh": candidate_no_refresh,
        "candidate_refresh": candidate_refresh,
        "final_no_refresh": final_no_refresh,
        "screen_overrides": screen_overrides,
        "refreshes": refreshes,
        "candidate_under_target": candidate_under_target,
        "authorized_under_target": authorized_under_target,
        "any_candidate_under_target": int(any_candidate_under_target),
        "any_authorized_under_target": int(any_authorized_under_target),
        "violation_years": violation_years,
        "any_violation": int(any_violation),
    }


def run_task(
    task: tuple[float, int, WrapperConfig]
) -> list[dict[str, float | int | str]]:
    true_hazard, episode, config = task
    uniforms = common_uniforms(true_hazard, episode, config)
    rows: list[dict[str, float | int | str]] = []
    for policy in BASE_POLICIES:
        base = simulate_path(
            config, true_hazard, policy, uniforms, screened=False
        )
        wrapped = simulate_path(
            config, true_hazard, policy, uniforms, screened=True
        )
        row: dict[str, float | int | str] = {
            "true_hazard": true_hazard,
            "policy": policy,
            "episode": episode,
            "root_seed": stable_seed(f"uniforms|{true_hazard:.6f}|{episode}"),
            "alpha_total": config.alpha_total,
            "per_look_alpha": config.per_look_alpha,
            "max_looks": config.max_looks,
        }
        row.update({f"base_{key}": value for key, value in base.items()})
        row.update({f"screened_{key}": value for key, value in wrapped.items()})
        row["paired_refresh_difference"] = (
            wrapped["refreshes"] - base["refreshes"]
        )
        row["paired_authorized_under_target_difference"] = (
            wrapped["any_authorized_under_target"]
            - base["any_authorized_under_target"]
        )
        rows.append(row)
    return rows


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    estimate = successes / total
    denominator = 1.0 + Z_975**2 / total
    centre = (estimate + Z_975**2 / (2.0 * total)) / denominator
    half = (
        Z_975
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + Z_975**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_mean_interval(values: np.ndarray, label: str) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap requires a nonempty one-dimensional array")
    rng = np.random.default_rng(stable_seed(label, BOOTSTRAP_BASE_SEED))
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    chunk = 128
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        estimates[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    rows: list[dict[str, float | int | str]], config: WrapperConfig
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    expected_keys = {
        (hazard, policy, episode)
        for hazard in HAZARDS
        for policy in BASE_POLICIES
        for episode in range(config.episodes)
    }
    actual_keys = {
        (float(row["true_hazard"]), str(row["policy"]), int(row["episode"]))
        for row in rows
    }
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("wrapper rows do not form the prespecified one-to-one key set")

    for hazard in HAZARDS:
        for policy in BASE_POLICIES:
            group = sorted(
                (
                    row
                    for row in rows
                    if float(row["true_hazard"]) == hazard
                    and str(row["policy"]) == policy
                ),
                key=lambda row: int(row["episode"]),
            )
            episodes = len(group)
            decisions = sum(int(row["base_decisions"]) for row in group)
            screened_decisions = sum(
                int(row["screened_decisions"]) for row in group
            )
            if decisions != screened_decisions:
                raise AssertionError("base and screened paths must have equal look counts")
            base_authorizations = sum(
                int(row["base_final_no_refresh"]) for row in group
            )
            screened_candidates = sum(
                int(row["screened_candidate_no_refresh"]) for row in group
            )
            screened_authorizations = sum(
                int(row["screened_final_no_refresh"]) for row in group
            )
            overrides = sum(int(row["screened_screen_overrides"]) for row in group)
            if screened_authorizations + overrides != screened_candidates:
                raise AssertionError("screen candidate accounting does not reconcile")

            base_events = sum(
                int(row["base_any_authorized_under_target"]) for row in group
            )
            screened_events = sum(
                int(row["screened_any_authorized_under_target"]) for row in group
            )
            base_low, base_high = wilson_interval(base_events, episodes)
            screened_low, screened_high = wilson_interval(screened_events, episodes)
            refresh_differences = np.asarray(
                [float(row["paired_refresh_difference"]) for row in group],
                dtype=float,
            )
            event_differences = np.asarray(
                [
                    float(row["paired_authorized_under_target_difference"])
                    for row in group
                ],
                dtype=float,
            )
            refresh_low, refresh_high = paired_mean_interval(
                refresh_differences, f"refresh|{hazard:.6f}|{policy}"
            )
            event_low, event_high = paired_mean_interval(
                event_differences, f"event|{hazard:.6f}|{policy}"
            )
            summaries.append(
                {
                    "true_hazard": hazard,
                    "policy": policy,
                    "episodes": episodes,
                    "decisions_per_episode": config.max_looks,
                    "alpha_total": config.alpha_total,
                    "per_look_alpha": config.per_look_alpha,
                    "base_candidate_no_refresh_rate": base_authorizations / decisions,
                    "screened_candidate_no_refresh_rate": screened_candidates / decisions,
                    "screen_pass_given_candidate": (
                        screened_authorizations / screened_candidates
                        if screened_candidates
                        else math.nan
                    ),
                    "screen_override_given_candidate": (
                        overrides / screened_candidates
                        if screened_candidates
                        else math.nan
                    ),
                    "base_mean_refreshes": sum(
                        int(row["base_refreshes"]) for row in group
                    )
                    / episodes,
                    "screened_mean_refreshes": sum(
                        int(row["screened_refreshes"]) for row in group
                    )
                    / episodes,
                    "paired_mean_refresh_difference": float(
                        refresh_differences.mean()
                    ),
                    "paired_mean_refresh_difference_ci95_low": refresh_low,
                    "paired_mean_refresh_difference_ci95_high": refresh_high,
                    "base_episode_under_target_authorization_rate": base_events
                    / episodes,
                    "base_episode_under_target_authorization_ci95_low": base_low,
                    "base_episode_under_target_authorization_ci95_high": base_high,
                    "screened_episode_under_target_authorization_rate": screened_events
                    / episodes,
                    "screened_episode_under_target_authorization_ci95_low": screened_low,
                    "screened_episode_under_target_authorization_ci95_high": screened_high,
                    "paired_episode_event_difference": float(event_differences.mean()),
                    "paired_episode_event_difference_ci95_low": event_low,
                    "paired_episode_event_difference_ci95_high": event_high,
                    "base_total_under_target_authorizations": sum(
                        int(row["base_authorized_under_target"]) for row in group
                    ),
                    "screened_total_under_target_authorizations": sum(
                        int(row["screened_authorized_under_target"]) for row in group
                    ),
                    "screened_total_candidates": screened_candidates,
                    "screened_total_overrides": overrides,
                    "screened_total_authorizations": screened_authorizations,
                    "analysis_unit": "complete 60-year mission",
                    "interval_method": (
                        "Wilson for individual episode rates; paired mission-level "
                        f"percentile bootstrap B={BOOTSTRAP_REPLICATES} for differences"
                    ),
                    "evidence_scope": "synthetic model-internal policy-logic stress test",
                }
            )
    return summaries


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    config = WrapperConfig(episodes=3)
    rows: list[dict[str, float | int | str]] = []
    for hazard in HAZARDS:
        for episode in range(config.episodes):
            first = run_task((hazard, episode, config))
            second = run_task((hazard, episode, config))
            assert first == second
            rows.extend(first)
    summaries = summarize(rows, config)
    assert len(summaries) == len(HAZARDS) * len(BASE_POLICIES)
    shuffled = list(reversed(rows))
    assert summarize(shuffled, config) == summaries
    for row in rows:
        assert int(row["screened_final_no_refresh"]) <= int(
            row["screened_candidate_no_refresh"]
        )
        assert int(row["screened_authorized_under_target"]) <= int(
            row["screened_candidate_under_target"]
        )
    print("PASS_PAPER4_POLICY_WRAPPER_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired base-policy versus base-policy-plus-screen audit."
    )
    parser.add_argument("--episodes", type=int, default=WrapperConfig.episodes)
    parser.add_argument("--alpha-total", type=float, default=WrapperConfig.alpha_total)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if not 0.0 < args.alpha_total < 1.0:
        raise SystemExit("--alpha-total must lie in (0, 1)")
    config = WrapperConfig(episodes=args.episodes, alpha_total=args.alpha_total)
    tasks = [
        (hazard, episode, config)
        for hazard in HAZARDS
        for episode in range(config.episodes)
    ]
    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task_rows in pool.map(run_task, tasks, chunksize=16):
            rows.extend(task_rows)
    summaries = summarize(rows, config)
    write_csv(args.output_dir / "paper4_policy_wrapper_episodes.csv", rows)
    write_csv(args.output_dir / "paper4_policy_wrapper_summary.csv", summaries)
    print("PASS_PAPER4_POLICY_WRAPPER")
    print(
        f"episodes_per_hazard={config.episodes} policies={len(BASE_POLICIES)} "
        f"rows={len(rows)} alpha_total={config.alpha_total}"
    )


if __name__ == "__main__":
    main()
