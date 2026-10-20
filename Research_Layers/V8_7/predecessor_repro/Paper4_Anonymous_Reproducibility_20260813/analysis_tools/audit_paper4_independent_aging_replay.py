#!/usr/bin/env python3
"""Replay independent Meiser and Song aging coverage intermediates.

This audit measures post-workflow reference detection. It does not estimate
pre-amplification molecular survival, maintenance benefit, or file recovery.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external_data/paper4_dt4dds_challenges_notebooks"

CONDITIONS = [
    ("Meiser", "unaged", "unaged"),
    ("Meiser", "aged", "aged"),
    ("Meiser", "repaired", "repaired"),
    ("Song", "0d", "0d"),
    ("Song", "28d", "28d"),
    ("Song", "56d", "56d"),
    ("Song", "70d", "70d"),
]

PAIRS = [
    ("Meiser", "unaged", "aged"),
    ("Meiser", "aged", "repaired"),
    ("Meiser", "unaged", "repaired"),
    ("Song", "0d", "28d"),
    ("Song", "0d", "56d"),
    ("Song", "0d", "70d"),
    ("Song", "28d", "56d"),
    ("Song", "56d", "70d"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_dir(study: str, condition: str) -> Path:
    return SOURCE / "data_experimental" / f"Aging_{study}" / condition


def fasta_names(path: Path) -> list[str]:
    names = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                names.append(line[1:].strip().split()[0])
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate FASTA identifiers: {path}")
    return names


def coverage(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = row["#name"]
            if name in values:
                raise ValueError(f"duplicate coverage identifier: {name} in {path}")
            values[name] = int(row["assignedReads"])
    return values


def quantile(values: list[int], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else 0.0


def build() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    designs: dict[tuple[str, str], list[str]] = {}
    coverages: dict[tuple[str, str], dict[str, int]] = {}
    summary_rows: list[dict[str, object]] = []

    for study, condition, label in CONDITIONS:
        directory = data_dir(study, condition)
        fasta = directory / "design_files.fasta"
        coverage_path = directory / "analysis" / "coverage.tsv"
        names = fasta_names(fasta)
        counts = coverage(coverage_path)
        name_set = set(names)
        unexpected = set(counts) - name_set
        if unexpected:
            raise ValueError(f"unexpected coverage identifiers in {study}/{condition}")
        detected = set(counts)
        positives = list(counts.values())
        total = sum(positives)
        designs[(study, condition)] = names
        coverages[(study, condition)] = counts
        summary_rows.append(
            {
                "study": study,
                "condition": label,
                "design_count": len(names),
                "detected_count": len(detected),
                "missing_count": len(name_set - detected),
                "missing_fraction": len(name_set - detected) / len(names),
                "total_assigned_reads": total,
                "median_positive_assigned_reads": quantile(positives, 0.5),
                "q10_positive_assigned_reads": quantile(positives, 0.1),
                "q90_positive_assigned_reads": quantile(positives, 0.9),
                "design_sha256": sha256(fasta),
                "coverage_sha256": sha256(coverage_path),
            }
        )

    for study in ("Meiser", "Song"):
        study_conditions = [condition for s, condition, _ in CONDITIONS if s == study]
        reference = designs[(study, study_conditions[0])]
        for condition in study_conditions[1:]:
            if designs[(study, condition)] != reference:
                raise ValueError(f"design FASTA mismatch within {study}: {condition}")

    pair_rows: list[dict[str, object]] = []
    for study, first, second in PAIRS:
        names = designs[(study, first)]
        first_counts = coverages[(study, first)]
        second_counts = coverages[(study, second)]
        first_detected = set(first_counts)
        second_detected = set(second_counts)
        union = first_detected | second_detected
        first_total = sum(first_counts.values())
        second_total = sum(second_counts.values())
        first_log_cpm = [math.log2(first_counts.get(name, 0) * 1_000_000 / first_total + 1.0) for name in names]
        second_log_cpm = [math.log2(second_counts.get(name, 0) * 1_000_000 / second_total + 1.0) for name in names]
        rho = float(spearmanr(first_log_cpm, second_log_cpm).statistic)
        pair_rows.append(
            {
                "study": study,
                "first_condition": first,
                "second_condition": second,
                "detected_both": len(first_detected & second_detected),
                "detected_first_only": len(first_detected - second_detected),
                "detected_second_only": len(second_detected - first_detected),
                "detection_jaccard": len(first_detected & second_detected) / len(union),
                "log2_cpm_spearman": rho,
            }
        )
    return summary_rows, pair_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    summaries, pairs = build()
    assert len(summaries) == 7
    assert len(pairs) == 8
    by_key = {(row["study"], row["condition"]): row for row in summaries}
    assert by_key[("Meiser", "unaged")]["design_count"] == 7373
    assert by_key[("Song", "0d")]["design_count"] == 210000
    assert by_key[("Meiser", "repaired")]["detected_count"] > by_key[("Meiser", "aged")]["detected_count"]
    assert by_key[("Song", "70d")]["missing_fraction"] < by_key[("Song", "0d")]["missing_fraction"]
    print("PASS_PAPER4_INDEPENDENT_AGING_REPLAY_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    summaries, pairs = build()
    write_csv(args.output_dir / "paper4_independent_aging_detection_summary.csv", summaries)
    write_csv(args.output_dir / "paper4_independent_aging_pairwise.csv", pairs)
    print("PASS_PAPER4_INDEPENDENT_AGING_REPLAY")
    for row in summaries:
        print(
            f"{row['study']} {row['condition']}: detected={row['detected_count']}/{row['design_count']} "
            f"missing_fraction={row['missing_fraction']:.6f} total_reads={row['total_assigned_reads']}"
        )


if __name__ == "__main__":
    main()
