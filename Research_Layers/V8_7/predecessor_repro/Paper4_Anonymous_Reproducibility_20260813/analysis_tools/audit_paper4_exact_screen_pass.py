#!/usr/bin/env python3
"""Deterministic exact-binomial screen-pass diagnostic for Paper 4 V8.3.

The calculation evaluates the controller's declared service screen directly.
It does not rerun episode simulations and does not equate screen passage with
the authorization rate of an arbitrary wrapped policy.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import beta, binom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "paper4_exact_screen_pass_diagnostic.csv"

N_SENTINELS = 300
ALPHA_PER_LOOK = 0.05 / 11.0
ARCHIVE_SIZE = 10_000
ERASURE_TOLERANCE = 0.10
SERVICE_TARGET = 0.99
GAP_YEARS = 5

SCENARIOS = (
    (0.002, 5),
    (0.002, 20),
    (0.002, 40),
    (0.006, 5),
    (0.006, 10),
    (0.012, 5),
)

EXPECTED = (
    (0.002, 5, True, 5, 0.9185884207590301),
    (0.002, 20, True, 11, 0.48715384692353136),
    (0.002, 40, True, 12, 0.006981163362795158),
    (0.006, 5, True, 5, 0.12034252781067852),
    (0.006, 10, True, 8, 0.008133337617047215),
    (0.012, 5, False, 5, 0.00036226371029259),
)


def dropout(hazard: float, age_years: float) -> float:
    return 1.0 - math.exp(-hazard * age_years)


def service_success(q: float) -> float:
    cutoff = math.floor(ARCHIVE_SIZE * ERASURE_TOLERANCE)
    return float(binom.cdf(cutoff, ARCHIVE_SIZE, q))


def clopper_pearson_upper(y: int) -> float:
    if y == N_SENTINELS:
        return 1.0
    return float(beta.ppf(1.0 - ALPHA_PER_LOOK, y + 1, N_SENTINELS - y))


def propagated_upper(y: int, age_years: int) -> float:
    upper = clopper_pearson_upper(y)
    kappa = (age_years + GAP_YEARS) / age_years
    return 1.0 - (1.0 - upper) ** kappa


def screen_cutoff(age_years: int) -> int:
    passing = [
        y
        for y in range(N_SENTINELS + 1)
        if service_success(propagated_upper(y, age_years)) >= SERVICE_TARGET
    ]
    return max(passing) if passing else -1


def build_rows() -> list[dict[str, object]]:
    q_crit = brentq(
        lambda q: service_success(q) - SERVICE_TARGET,
        1e-12,
        ERASURE_TOLERANCE,
    )
    rows: list[dict[str, object]] = []
    for hazard, age_years in SCENARIOS:
        observation_probability = dropout(hazard, age_years)
        next_state_dropout = dropout(hazard, age_years + GAP_YEARS)
        y_star = screen_cutoff(age_years)
        screen_pass = (
            float(binom.cdf(y_star, N_SENTINELS, observation_probability))
            if y_star >= 0
            else 0.0
        )
        rows.append(
            {
                "true_hazard": hazard,
                "cycle_age_years": age_years,
                "inspection_gap_years": GAP_YEARS,
                "n_sentinels": N_SENTINELS,
                "alpha_per_look": ALPHA_PER_LOOK,
                "observation_probability": observation_probability,
                "candidate_next_state_dropout": next_state_dropout,
                "q_crit": q_crit,
                "candidate_next_state_model_safe": next_state_dropout <= q_crit,
                "y_star": y_star,
                "screen_pass_probability": screen_pass,
                "screen_pass_percent": 100.0 * screen_pass,
            }
        )
    return rows


def self_test(rows: list[dict[str, object]]) -> None:
    assert len(rows) == len(EXPECTED)
    for row, expected in zip(rows, EXPECTED, strict=True):
        hazard, age, safe, y_star, probability = expected
        assert math.isclose(float(row["true_hazard"]), hazard, rel_tol=0.0, abs_tol=0.0)
        assert int(row["cycle_age_years"]) == age
        assert bool(row["candidate_next_state_model_safe"]) is safe
        assert int(row["y_star"]) == y_star
        assert math.isclose(
            float(row["screen_pass_probability"]),
            probability,
            rel_tol=0.0,
            abs_tol=2e-14,
        )
    assert sum(bool(row["candidate_next_state_model_safe"]) for row in rows) == 5
    print("PASS_PAPER4_EXACT_SCREEN_PASS_DIAGNOSTIC rows=6 safe=5 unsafe=1")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path} rows={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_rows()
    self_test(rows)
    if not args.self_test:
        write_rows(args.output, rows)


if __name__ == "__main__":
    main()
