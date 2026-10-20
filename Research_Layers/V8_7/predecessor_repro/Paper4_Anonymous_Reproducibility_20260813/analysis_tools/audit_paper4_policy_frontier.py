#!/usr/bin/env python3
"""Enumerate periodic refresh schedules for the Paper 4 synthetic model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper4_archive_maintenance_seed import (
    Config,
    HAZARDS,
    is_action_year,
    recoverability_probability,
)


ROOT = Path(__file__).resolve().parents[1]


def run_schedule(
    config: Config, hazard: float, refresh_interval: int
) -> dict[str, object]:
    age = 0
    refreshes = 0
    violation_years = 0
    for year in range(1, config.horizon_years + 1):
        age += 1
        if (
            recoverability_probability(config, hazard, age)
            < config.recoverability_target
        ):
            violation_years += 1
        if (
            is_action_year(year, config.horizon_years, config.inspection_interval)
            and age >= refresh_interval
        ):
            refreshes += 1
            age = 0
    return {
        "true_hazard": hazard,
        "refresh_interval_years": refresh_interval,
        "refreshes": refreshes,
        "violation_years": violation_years,
        "violation_rate": violation_years / config.horizon_years,
        "zero_violation": int(violation_years == 0),
    }


def build(config: Config) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    intervals = range(
        config.inspection_interval, config.horizon_years + 1, config.inspection_interval
    )
    rows = [
        run_schedule(config, hazard, interval)
        for hazard in HAZARDS
        for interval in intervals
    ]
    summary = []
    for hazard in HAZARDS:
        safe = [
            row
            for row in rows
            if row["true_hazard"] == hazard and row["zero_violation"]
        ]
        minimum_refreshes = min(int(row["refreshes"]) for row in safe)
        tied = [row for row in safe if int(row["refreshes"]) == minimum_refreshes]
        summary.append(
            {
                "true_hazard": hazard,
                "minimum_refreshes_among_enumerated_periodic_schedules": minimum_refreshes,
                "largest_interval_at_minimum_refreshes": max(
                    int(row["refresh_interval_years"]) for row in tied
                ),
                "enumerated_intervals": len(list(intervals)),
                "comparison_class": "periodic schedules at the declared inspection grid",
                "claim_upgrade_allowed": False,
            }
        )
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    rows, summary = build(Config())
    assert len(rows) == 36
    assert [
        row["minimum_refreshes_among_enumerated_periodic_schedules"] for row in summary
    ] == [1, 3, 11]
    assert all(row["comparison_class"].startswith("periodic") for row in summary)
    print("PASS_PAPER4_POLICY_FRONTIER_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    rows, summary = build(Config())
    write_csv(args.output_dir / "paper4_periodic_schedule_frontier.csv", rows)
    write_csv(
        args.output_dir / "paper4_periodic_schedule_frontier_summary.csv", summary
    )
    print("PASS_PAPER4_POLICY_FRONTIER")
    print("hazards=3 intervals_per_hazard=12 rows=36")


if __name__ == "__main__":
    main()
