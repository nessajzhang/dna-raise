#!/usr/bin/env python3
"""Audit prespecified deterministic alpha-spending profiles for Paper 4.

The experiment is deliberately model-internal.  It tunes a one-parameter
calendar-look profile on one fixed, envelope-valid adverse scenario and then
compares the selected profile with equal spending on independent evaluation
episodes.  Every candidate is deterministic and exhausts the same total alpha
budget, so the finite-horizon familywise guarantee is unchanged.  The exercise
does not establish economic optimality or physical DNA-archive performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from audit_paper4_finite_horizon_certificate import (
    CertificateScenario,
    certified_next_dropout,
    cumulative_dropout,
)
from paper4_archive_maintenance_seed import (
    Config,
    exact_binomial_recoverability,
    is_action_year,
)


ROOT = Path(__file__).resolve().parents[1]
GAMMAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
ALPHA_TOTAL = 0.05
HORIZON_YEARS = 60
SCENARIO = CertificateScenario(
    "alpha_spending_design_adverse_valid",
    0.006,
    sentinel_size=3_000,
    inspection_interval=5,
    true_acceleration=0.03,
    observation_factor=0.80,
    declared_detection_floor=0.80,
    declared_acceleration_ceiling=0.03,
)
Z_975 = 1.959963984540054


def alpha_schedule(gamma: float, looks: int, alpha_total: float) -> np.ndarray:
    """Return alpha_j proportional to j**gamma for j=1,...,looks."""
    if looks <= 0:
        raise ValueError("looks must be positive")
    if not 0.0 < alpha_total < 1.0:
        raise ValueError("alpha_total must lie in (0, 1)")
    indices = np.arange(1, looks + 1, dtype=float)
    weights = indices**gamma
    schedule = alpha_total * weights / weights.sum()
    if not np.all(schedule > 0.0):
        raise AssertionError("all per-look alpha allocations must be positive")
    if not math.isclose(float(schedule.sum()), alpha_total, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError("alpha schedule does not sum to the declared budget")
    return schedule


def episode_seed(phase: str, episode: int) -> int:
    base = 20260719 if phase == "design" else 20260720
    return base + episode * 100_003


def run_episode(task: tuple[str, float, int]) -> dict[str, object]:
    phase, gamma, episode = task
    rng = np.random.default_rng(episode_seed(phase, episode))
    config = Config(
        horizon_years=HORIZON_YEARS,
        inspection_interval=SCENARIO.inspection_interval,
        sentinel_size=SCENARIO.sentinel_size,
        episodes=1,
    )
    max_looks = (HORIZON_YEARS - 1) // SCENARIO.inspection_interval
    spending = alpha_schedule(gamma, max_looks, ALPHA_TOTAL)
    age = 0
    decisions = 0
    refreshes = 0
    no_refresh_decisions = 0
    false_safe_decisions = 0
    false_refresh_decisions = 0
    look_bound_failures = 0

    for year in range(1, HORIZON_YEARS + 1):
        age += 1
        if not is_action_year(year, HORIZON_YEARS, SCENARIO.inspection_interval):
            continue

        alpha_j = float(spending[decisions])
        decisions += 1
        true_q = cumulative_dropout(
            SCENARIO.true_hazard, age, SCENARIO.true_acceleration
        )
        observed_q = float(np.clip(true_q * SCENARIO.observation_factor, 0.0, 1.0))
        unavailable = int(rng.binomial(SCENARIO.sentinel_size, observed_q))
        observed_upper, _, next_dropout_upper = certified_next_dropout(
            unavailable,
            SCENARIO.sentinel_size,
            age,
            age + SCENARIO.inspection_interval,
            alpha_j,
            SCENARIO.declared_detection_floor,
            SCENARIO.declared_acceleration_ceiling,
            SCENARIO.propagation_bound,
        )
        if observed_upper + 1e-15 < observed_q:
            look_bound_failures += 1

        certified_recoverability = exact_binomial_recoverability(
            config.archive_oligos,
            config.erasure_tolerance,
            next_dropout_upper,
        )
        true_next_q = cumulative_dropout(
            SCENARIO.true_hazard,
            age + SCENARIO.inspection_interval,
            SCENARIO.true_acceleration,
        )
        true_next_recoverability = exact_binomial_recoverability(
            config.archive_oligos,
            config.erasure_tolerance,
            true_next_q,
        )
        if certified_recoverability < config.recoverability_target:
            refreshes += 1
            if true_next_recoverability >= config.recoverability_target:
                false_refresh_decisions += 1
            age = 0
            continue

        no_refresh_decisions += 1
        if true_next_recoverability < config.recoverability_target:
            false_safe_decisions += 1

    return {
        "phase": phase,
        "gamma": gamma,
        "episode": episode,
        "alpha_total": ALPHA_TOTAL,
        "alpha_min": float(spending.min()),
        "alpha_max": float(spending.max()),
        "decisions": decisions,
        "refreshes": refreshes,
        "false_refresh_decisions": false_refresh_decisions,
        "sentinel_assays": decisions * SCENARIO.sentinel_size,
        "no_refresh_decisions": no_refresh_decisions,
        "false_safe_decisions": false_safe_decisions,
        "any_false_safe": int(false_safe_decisions > 0),
        "look_bound_failures": look_bound_failures,
        "any_look_bound_failure": int(look_bound_failures > 0),
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + Z_975**2 / total
    centre = (p + Z_975**2 / (2.0 * total)) / denominator
    radius = (
        Z_975
        * math.sqrt(p * (1.0 - p) / total + Z_975**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def summarize(rows: list[dict[str, object]], selected_gamma: float) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["phase"]), float(row["gamma"])), []).append(row)

    summaries: list[dict[str, object]] = []
    for (phase, gamma), group in sorted(grouped.items()):
        false_safe_episodes = sum(int(row["any_false_safe"]) for row in group)
        low, high = wilson_interval(false_safe_episodes, len(group))
        summaries.append(
            {
                "phase": phase,
                "gamma": gamma,
                "profile": "equal" if gamma == 0.0 else "power_law_by_calendar_look",
                "selected_on_design": int(gamma == selected_gamma),
                "episodes": len(group),
                "alpha_total": ALPHA_TOTAL,
                "alpha_min": float(group[0]["alpha_min"]),
                "alpha_max": float(group[0]["alpha_max"]),
                "mean_refreshes": float(np.mean([int(row["refreshes"]) for row in group])),
                "se_mean_refreshes": float(
                    np.std([int(row["refreshes"]) for row in group], ddof=1)
                    / math.sqrt(len(group))
                ),
                "mean_false_refreshes": float(
                    np.mean([int(row["false_refresh_decisions"]) for row in group])
                ),
                "mean_sentinel_assays": float(
                    np.mean([int(row["sentinel_assays"]) for row in group])
                ),
                "mean_no_refresh_decisions": float(
                    np.mean([int(row["no_refresh_decisions"]) for row in group])
                ),
                "false_safe_decisions": sum(
                    int(row["false_safe_decisions"]) for row in group
                ),
                "false_safe_episodes": false_safe_episodes,
                "episode_false_safe_rate": false_safe_episodes / len(group),
                "episode_false_safe_rate_ci95_low": low,
                "episode_false_safe_rate_ci95_high": high,
                "look_bound_failure_episodes": sum(
                    int(row["any_look_bound_failure"]) for row in group
                ),
            }
        )
    return summaries


def paired_difference(
    evaluation_rows: list[dict[str, object]], candidate_gamma: float
) -> dict[str, object]:
    equal = {
        int(row["episode"]): int(row["refreshes"])
        for row in evaluation_rows
        if float(row["gamma"]) == 0.0
    }
    candidate = {
        int(row["episode"]): int(row["refreshes"])
        for row in evaluation_rows
        if float(row["gamma"]) == candidate_gamma
    }
    common = sorted(set(equal) & set(candidate))
    differences = np.asarray([candidate[i] - equal[i] for i in common], dtype=float)
    rng = np.random.default_rng(2026071901 + int((candidate_gamma + 2.0) * 100))
    bootstrap = np.empty(10_000, dtype=float)
    for index in range(10_000):
        draw = rng.integers(0, len(differences), size=len(differences))
        bootstrap[index] = float(np.mean(differences[draw]))
    familywise_tail = 0.05 / (2.0 * max(1, len(GAMMAS) - 1))
    return {
        "candidate_gamma": candidate_gamma,
        "evaluation_episodes": len(common),
        "mean_refresh_difference_candidate_minus_equal": float(np.mean(differences)),
        "se_paired_difference": float(np.std(differences, ddof=1) / math.sqrt(len(common))),
        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "familywise_bootstrap_ci95_low": float(np.quantile(bootstrap, familywise_tail)),
        "familywise_bootstrap_ci95_high": float(
            np.quantile(bootstrap, 1.0 - familywise_tail)
        ),
        "fraction_candidate_fewer_refreshes": float(np.mean(differences < 0.0)),
        "fraction_equal": float(np.mean(differences == 0.0)),
        "fraction_candidate_more_refreshes": float(np.mean(differences > 0.0)),
    }


def design_rank_stability(
    design_rows: list[dict[str, object]], replicates: int = 10_000
) -> dict[str, float]:
    """Paired episode bootstrap frequency with which each profile ranks first."""
    by_gamma = {
        gamma: np.asarray(
            [
                int(row["refreshes"])
                for row in design_rows
                if float(row["gamma"]) == gamma
            ],
            dtype=float,
        )
        for gamma in GAMMAS
    }
    episode_count = len(next(iter(by_gamma.values())))
    rng = np.random.default_rng(2026071902)
    wins = {gamma: 0 for gamma in GAMMAS}
    for _ in range(replicates):
        draw = rng.integers(0, episode_count, size=episode_count)
        means = {gamma: float(values[draw].mean()) for gamma, values in by_gamma.items()}
        winner = min(GAMMAS, key=lambda gamma: (means[gamma], abs(gamma)))
        wins[winner] += 1
    return {str(gamma): wins[gamma] / replicates for gamma in GAMMAS}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--design-episodes", type=int, default=1_000)
    parser.add_argument("--evaluation-episodes", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.self_test:
        for gamma in GAMMAS:
            schedule = alpha_schedule(gamma, 11, ALPHA_TOTAL)
            assert len(schedule) == 11
            assert np.all(schedule > 0.0)
        equal = alpha_schedule(0.0, 11, ALPHA_TOTAL)
        assert np.allclose(equal, ALPHA_TOTAL / 11.0, rtol=0.0, atol=1e-15)
        assert SCENARIO.envelope_valid
        print("PASS_PAPER4_ALPHA_SPENDING_DESIGN_SELF_TEST")
        return
    if args.design_episodes <= 0 or args.evaluation_episodes <= 0:
        raise SystemExit("episode counts must be positive")

    design_tasks = [
        ("design", gamma, episode)
        for gamma in GAMMAS
        for episode in range(args.design_episodes)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        design_rows = list(pool.map(run_episode, design_tasks, chunksize=32))

    design_means = {
        gamma: float(
            np.mean(
                [
                    int(row["refreshes"])
                    for row in design_rows
                    if float(row["gamma"]) == gamma
                ]
            )
        )
        for gamma in GAMMAS
    }
    selected_gamma = min(GAMMAS, key=lambda gamma: (design_means[gamma], abs(gamma)))

    # Evaluate every prespecified profile on the same held-out episode seeds;
    # restricting evaluation to the selected profile would hide rank instability.
    evaluation_gammas = GAMMAS
    evaluation_tasks = [
        ("evaluation", gamma, episode)
        for gamma in evaluation_gammas
        for episode in range(args.evaluation_episodes)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        evaluation_rows = list(pool.map(run_episode, evaluation_tasks, chunksize=32))

    all_rows = design_rows + evaluation_rows
    summaries = summarize(all_rows, selected_gamma)
    comparisons = [
        paired_difference(evaluation_rows, gamma)
        for gamma in GAMMAS
        if gamma != 0.0
    ]
    rank_stability = design_rank_stability(design_rows)

    write_csv(args.output_dir / "paper4_alpha_spending_design_episodes.csv", all_rows)
    write_csv(args.output_dir / "paper4_alpha_spending_design_summary.csv", summaries)
    write_csv(
        args.output_dir / "paper4_alpha_spending_design_paired_comparisons.csv",
        comparisons,
    )
    with (args.output_dir / "paper4_alpha_spending_design_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "scenario": SCENARIO.name,
                "candidate_gammas": GAMMAS,
                "design_means": design_means,
                "selection_rule": "minimum mean refreshes; ties favour smaller absolute gamma",
                "selected_profile_comparison": next(
                    (
                        row
                        for row in comparisons
                        if float(row["candidate_gamma"]) == selected_gamma
                    ),
                    {
                        "candidate_gamma": 0.0,
                        "evaluation_episodes": args.evaluation_episodes,
                        "mean_refresh_difference_candidate_minus_equal": 0.0,
                    },
                ),
                "all_held_out_comparisons": comparisons,
                "design_bootstrap_selection_frequency": rank_stability,
                "scope": "model-internal deterministic spending design; not economic or physical validation",
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(
        "PASS_PAPER4_ALPHA_SPENDING_DESIGN "
        f"selected_gamma={selected_gamma:g} "
        f"design_rows={len(design_rows)} evaluation_rows={len(evaluation_rows)}"
    )


if __name__ == "__main__":
    main()
