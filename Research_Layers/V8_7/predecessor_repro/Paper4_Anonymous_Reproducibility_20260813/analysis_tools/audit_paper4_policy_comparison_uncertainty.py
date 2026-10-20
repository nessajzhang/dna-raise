#!/usr/bin/env python3
"""Independent-cohort uncertainty for the Paper 4 policy comparison.

The common-design policy cohorts use policy-specific deterministic seeds and
are not trajectory-paired.  Risk differences therefore use an independent
Newcombe score interval, while refresh and monitoring-burden differences use
an unpaired two-sample bootstrap.  These are Monte Carlo comparison intervals,
not physical-system uncertainty intervals.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_POLICY = "certificate_robust"
Z_975 = 1.959963984540054
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026072201


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, total: int) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + Z_975**2 / total
    center = (proportion + Z_975**2 / (2.0 * total)) / denominator
    radius = (
        Z_975
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + Z_975**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def newcombe_difference(
    successes_a: int, total_a: int, successes_b: int, total_b: int
) -> tuple[float, float, float]:
    """Newcombe hybrid-score interval for p_a - p_b (method 10)."""
    p_a = successes_a / total_a
    p_b = successes_b / total_b
    low_a, high_a = wilson(successes_a, total_a)
    low_b, high_b = wilson(successes_b, total_b)
    difference = p_a - p_b
    lower = difference - math.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2)
    upper = difference + math.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2)
    return difference, max(-1.0, lower), min(1.0, upper)


def unpaired_bootstrap_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float, float]:
    observed = float(np.mean(values_a) - np.mean(values_b))
    differences = np.empty(replicates, dtype=float)
    chunk = 250
    for start in range(0, replicates, chunk):
        stop = min(replicates, start + chunk)
        count = stop - start
        sample_a = rng.integers(0, len(values_a), size=(count, len(values_a)))
        sample_b = rng.integers(0, len(values_b), size=(count, len(values_b)))
        differences[start:stop] = (
            np.mean(values_a[sample_a], axis=1)
            - np.mean(values_b[sample_b], axis=1)
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return observed, float(low), float(high)


def build(rows: list[dict[str, str]], replicates: int) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row["family"] == "common_design_policy"
        and math.isclose(float(row["true_hazard_per_year"]), 0.009)
    ]
    groups: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        groups.setdefault(row["policy"], []).append(row)
    if REFERENCE_POLICY not in groups:
        raise AssertionError("reference policy is absent")
    reference = groups[REFERENCE_POLICY]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: list[dict[str, object]] = []
    for policy in sorted(groups):
        if policy == REFERENCE_POLICY:
            continue
        group = groups[policy]
        risk = newcombe_difference(
            sum(int(row["any_unsafe_deferral"]) for row in group),
            len(group),
            sum(int(row["any_unsafe_deferral"]) for row in reference),
            len(reference),
        )
        refresh = unpaired_bootstrap_difference(
            np.asarray([float(row["refresh_attempts"]) for row in group]),
            np.asarray([float(row["refresh_attempts"]) for row in reference]),
            rng,
            replicates,
        )
        panels = unpaired_bootstrap_difference(
            np.asarray([float(row["inspection_panels"]) for row in group]),
            np.asarray([float(row["inspection_panels"]) for row in reference]),
            rng,
            replicates,
        )
        output.append(
            {
                "policy": policy,
                "reference_policy": REFERENCE_POLICY,
                "hazard_per_year": 0.009,
                "episodes_policy": len(group),
                "episodes_reference": len(reference),
                "risk_difference": risk[0],
                "risk_difference_newcombe95_low": risk[1],
                "risk_difference_newcombe95_high": risk[2],
                "mean_refresh_difference": refresh[0],
                "mean_refresh_difference_bootstrap95_low": refresh[1],
                "mean_refresh_difference_bootstrap95_high": refresh[2],
                "mean_inspection_panel_difference": panels[0],
                "mean_inspection_panel_difference_bootstrap95_low": panels[1],
                "mean_inspection_panel_difference_bootstrap95_high": panels[2],
                "bootstrap_replicates": replicates,
                "randomization_design": (
                    "common-design independent cohorts; unpaired inference"
                ),
                "scope": "Monte Carlo comparison uncertainty only",
            }
        )
    return output


def self_test() -> None:
    difference, low, high = newcombe_difference(0, 5_000, 0, 5_000)
    assert difference == 0.0 and low < 0.0 < high
    rng = np.random.default_rng(1)
    observed, boot_low, boot_high = unpaired_bootstrap_difference(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0.0, 1.0, 2.0]),
        rng,
        1_000,
    )
    assert observed == 1.0 and boot_low <= observed <= boot_high
    print("PASS_PAPER4_POLICY_COMPARISON_UNCERTAINTY_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.replicates < 1_000:
        raise SystemExit("at least 1,000 bootstrap replicates are required")
    rows = read_csv(args.data_dir / "paper4_extended_common_design_policy_episodes.csv")
    output = build(rows, args.replicates)
    write_csv(args.data_dir / "paper4_policy_comparison_uncertainty.csv", output)
    print("PASS_PAPER4_POLICY_COMPARISON_UNCERTAINTY")
    print(f"comparisons={len(output)} bootstrap_replicates={args.replicates}")


if __name__ == "__main__":
    main()
