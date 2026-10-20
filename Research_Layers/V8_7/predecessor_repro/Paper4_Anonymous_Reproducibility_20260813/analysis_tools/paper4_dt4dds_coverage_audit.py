#!/usr/bin/env python3
"""Audit the bounded DT4DDS coverage-evolution summary used by Paper 4."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external_data/paper4_dt4dds/coverage_evolution_source_summary.csv"
TIMEPOINTS = ROOT / "data/paper4_dt4dds_coverage_detection_timepoints.csv"
SUMMARY = ROOT / "data/paper4_dt4dds_coverage_detection_summary.csv"

EXPECTED_MISSING = {
    "Genscript_GCall": [110, 136, 113, 100, 100],
    "Twist_GCall": [9, 12, 16, 17, 11],
}
EXPECTED_REFERENCE_COUNTS = {"Genscript_GCall": 12472, "Twist_GCall": 12000}
EXPECTED_SAMPLES = ["0a", "0b", "2d", "4d", "7d"]


def load_source() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(rows: list[dict[str, str]]) -> None:
    if len(rows) != 10:
        raise AssertionError(f"expected 10 source rows, observed {len(rows)}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["pool"]].append(row)
        reference_count = int(row["reference_count"])
        detected = int(row["detected_reference_count"])
        missing = int(row["missing_reference_count"])
        if detected + missing != reference_count:
            raise AssertionError(f"reference accounting failed for {row['pool']} {row['sample']}")
        if int(row["unexpected_name_count"]) != 0:
            raise AssertionError(f"unexpected names present for {row['pool']} {row['sample']}")
        for key in ("source_scafstats_blob_sha", "source_design_blob_sha"):
            value = row[key]
            if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
                raise AssertionError(f"invalid Git blob SHA in {key}")
    if set(grouped) != set(EXPECTED_MISSING):
        raise AssertionError("pool set mismatch")
    for pool, pool_rows in grouped.items():
        if [row["sample"] for row in pool_rows] != EXPECTED_SAMPLES:
            raise AssertionError(f"sample order mismatch for {pool}")
        if [int(row["missing_reference_count"]) for row in pool_rows] != EXPECTED_MISSING[pool]:
            raise AssertionError(f"missing-count vector mismatch for {pool}")
        if {int(row["reference_count"]) for row in pool_rows} != {EXPECTED_REFERENCE_COUNTS[pool]}:
            raise AssertionError(f"reference-count mismatch for {pool}")


def write_outputs(rows: list[dict[str, str]]) -> None:
    TIMEPOINTS.parent.mkdir(parents=True, exist_ok=True)
    timepoint_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        reference_count = int(row["reference_count"])
        missing = int(row["missing_reference_count"])
        total_reads = int(row["total_assigned_reads"])
        output = {
            **row,
            "missing_fraction": f"{missing / reference_count:.12f}",
            "detected_fraction": f"{1 - missing / reference_count:.12f}",
            "mean_assigned_reads_all_references": f"{total_reads / reference_count:.9f}",
            "claim_boundary": (
                "post-workflow reference detection; not molecular survival, independent aging validation, "
                "outer-code recovery, file recovery, or maintenance benefit"
            ),
        }
        timepoint_rows.append(output)
        grouped[row["pool"]].append(output)

    with TIMEPOINTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timepoint_rows[0]))
        writer.writeheader()
        writer.writerows(timepoint_rows)

    summary_rows: list[dict[str, object]] = []
    for pool, pool_rows in grouped.items():
        fractions = [float(row["missing_fraction"]) for row in pool_rows]
        day_zero = [float(row["missing_fraction"]) for row in pool_rows if int(row["day"]) == 0]
        aged = [float(row["missing_fraction"]) for row in pool_rows if int(row["day"]) > 0]
        day7 = next(float(row["missing_fraction"]) for row in pool_rows if int(row["day"]) == 7)
        day_zero_mean = sum(day_zero) / len(day_zero)
        reads = [int(row["total_assigned_reads"]) for row in pool_rows]
        summary_rows.append(
            {
                "pool": pool,
                "reference_count": int(pool_rows[0]["reference_count"]),
                "time_zero_missing_fraction_mean": f"{day_zero_mean:.12f}",
                "day7_missing_fraction": f"{day7:.12f}",
                "day7_minus_time_zero_percentage_points": f"{100 * (day7 - day_zero_mean):.9f}",
                "minimum_missing_fraction": f"{min(fractions):.12f}",
                "maximum_missing_fraction": f"{max(fractions):.12f}",
                "missing_fraction_range_percentage_points": f"{100 * (max(fractions) - min(fractions)):.9f}",
                "aged_series_monotonic_nondecreasing": str(all(a <= b for a, b in zip(aged, aged[1:]))),
                "total_assigned_reads_min": min(reads),
                "total_assigned_reads_max": max(reads),
                "interpretation": "no monotonic increase in post-workflow missing-reference fraction over 2-7 days",
                "claim_boundary": (
                    "descriptive replay of two GCall pools; sample preparation, PCR, and read depth confound "
                    "material-aging interpretation"
                ),
            }
        )
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    rows = load_source()
    validate(rows)
    if args.self_test:
        print("PASS_PAPER4_DT4DDS_COVERAGE_AUDIT_SELF_TEST")
        return
    write_outputs(rows)
    print(TIMEPOINTS)
    print(SUMMARY)


if __name__ == "__main__":
    main()
