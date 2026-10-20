#!/usr/bin/env python3
"""Exact margin-domain screen-passage sensitivity for Paper 4 V8.5.

This deterministic diagnostic extends the existing fixed-history calculation.
It does not rerun episode simulations and does not establish a deployment-wide
screen-passage floor.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import beta, binom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "paper4_margin_safe_screen_pass.csv"
DEFAULT_TABLE = (
    ROOT
    / "papers"
    / "paper4_adaptive_archive_maintenance"
    / "generated_tables"
    / "si_margin_safe_screen_rows.tex"
)
REFERENCE = ROOT / "data" / "paper4_exact_screen_pass_diagnostic.csv"

N_SENTINELS = 300
ALPHA_PER_LOOK = 0.05 / 11.0
ARCHIVE_SIZE = 10_000
ERASURE_TOLERANCE = 0.10
SERVICE_TARGET = 0.99
GAP_YEARS = 5
SCHEDULED_AGES = tuple(range(5, 56, 5))
RELATIVE_STATE_MARGINS = (0.0, 0.10, 0.25, 0.50, 0.75)

EXPECTED_REFERENCE = (
    (0.002, 5, 5, 0.9185884207590301),
    (0.002, 20, 11, 0.48715384692353136),
    (0.002, 40, 12, 0.006981163362795158),
    (0.006, 5, 5, 0.12034252781067852),
    (0.006, 10, 8, 0.008133337617047215),
    (0.012, 5, 5, 0.00036226371029259),
)

EXPECTED_FLOORS = (
    (0.00, 40, 0.0022936466912880536),
    (0.10, 40, 0.009617867818524795),
    (0.25, 5, 0.042714642363616856),
    (0.50, 5, 0.28807759299586433),
    (0.75, 5, 0.8565761952795712),
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


def reproduce_reference() -> None:
    calculated = []
    for hazard, age, expected_cutoff, expected_probability in EXPECTED_REFERENCE:
        cutoff = screen_cutoff(age)
        probability = float(
            binom.cdf(cutoff, N_SENTINELS, dropout(hazard, age))
        )
        assert cutoff == expected_cutoff
        assert math.isclose(
            probability, expected_probability, rel_tol=0.0, abs_tol=2e-14
        )
        calculated.append((hazard, age, cutoff, probability))

    if REFERENCE.is_file():
        with REFERENCE.open(newline="", encoding="utf-8-sig") as handle:
            frozen = list(csv.DictReader(handle))
        assert len(frozen) == len(calculated)
        for row, values in zip(frozen, calculated, strict=True):
            hazard, age, cutoff, probability = values
            assert math.isclose(float(row["true_hazard"]), hazard, abs_tol=0.0)
            assert int(row["cycle_age_years"]) == age
            assert int(row["y_star"]) == cutoff
            assert math.isclose(
                float(row["screen_pass_probability"]),
                probability,
                rel_tol=0.0,
                abs_tol=2e-14,
            )


def build_rows() -> list[dict[str, object]]:
    q_crit = brentq(
        lambda q: service_success(q) - SERVICE_TARGET,
        1e-12,
        ERASURE_TOLERANCE,
    )
    rows: list[dict[str, object]] = []
    for margin in RELATIVE_STATE_MARGINS:
        q_boundary = (1.0 - margin) * q_crit
        block: list[dict[str, object]] = []
        for age in SCHEDULED_AGES:
            current_boundary = 1.0 - (1.0 - q_boundary) ** (
                age / (age + GAP_YEARS)
            )
            cutoff = screen_cutoff(age)
            probability = (
                float(binom.cdf(cutoff, N_SENTINELS, current_boundary))
                if cutoff >= 0
                else 0.0
            )
            block.append(
                {
                    "relative_state_margin": margin,
                    "cycle_age_years": age,
                    "inspection_gap_years": GAP_YEARS,
                    "n_sentinels": N_SENTINELS,
                    "alpha_per_look": ALPHA_PER_LOOK,
                    "q_crit": q_crit,
                    "candidate_next_state_dropout_boundary": q_boundary,
                    "current_observation_probability_boundary": current_boundary,
                    "y_star": cutoff,
                    "screen_pass_probability_at_boundary": probability,
                    "screen_pass_percent_at_boundary": 100.0 * probability,
                    "is_margin_floor": False,
                }
            )
        floor = min(block, key=lambda row: row["screen_pass_probability_at_boundary"])
        floor["is_margin_floor"] = True
        rows.extend(block)
    return rows


def floor_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if bool(row["is_margin_floor"])]


def self_test(rows: list[dict[str, object]]) -> None:
    reproduce_reference()
    assert len(rows) == len(RELATIVE_STATE_MARGINS) * len(SCHEDULED_AGES)
    floors = floor_rows(rows)
    assert len(floors) == len(EXPECTED_FLOORS)
    for row, expected in zip(floors, EXPECTED_FLOORS, strict=True):
        margin, age, probability = expected
        assert math.isclose(
            float(row["relative_state_margin"]), margin, abs_tol=0.0
        )
        assert int(row["cycle_age_years"]) == age
        assert math.isclose(
            float(row["screen_pass_probability_at_boundary"]),
            probability,
            rel_tol=0.0,
            abs_tol=2e-14,
        )
    probabilities = [
        float(row["screen_pass_probability_at_boundary"]) for row in floors
    ]
    assert all(left < right for left, right in zip(probabilities, probabilities[1:]))
    print(
        "PASS_PAPER4_MARGIN_SAFE_SCREEN_PASS "
        f"rows={len(rows)} margins={len(floors)} scheduled_ages={len(SCHEDULED_AGES)}"
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path} rows={len(rows)}")


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated from audited CSV: data/paper4_margin_safe_screen_pass.csv"]
    for row in floor_rows(rows):
        margin = float(row["relative_state_margin"])
        q_boundary = float(row["candidate_next_state_dropout_boundary"])
        percentage = float(row["screen_pass_percent_at_boundary"])
        age = int(row["cycle_age_years"])
        lines.append(
            f"{margin:.2f} & {q_boundary:.6f} & {percentage:.4f} & {age} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {path} rows={len(lines) - 1}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    self_test(rows)
    if not args.self_test:
        write_csv(args.output, rows)
        write_table(args.table, rows)


if __name__ == "__main__":
    main()
