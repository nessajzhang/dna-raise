#!/usr/bin/env python3
"""Episode-level Monte Carlo uncertainty audit for Paper 4.

The intervals quantify finite-simulation uncertainty only. They do not measure
physical archive variability or uncertainty in a molecular intervention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HORIZON_YEARS = 60
Z_975 = 1.959963984540054


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    if successes == 0:
        return 0.0, Z_975**2 / (total + Z_975**2)
    if successes == total:
        return total / (total + Z_975**2), 1.0
    estimate = successes / total
    denominator = 1.0 + Z_975**2 / total
    centre = (estimate + Z_975**2 / (2.0 * total)) / denominator
    half_width = (
        Z_975
        * math.sqrt(estimate * (1.0 - estimate) / total + Z_975**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def stable_seed(key: tuple[object, ...]) -> int:
    payload = "|".join(str(value) for value in key).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_mean_interval(
    values: np.ndarray, seed: int, replicates: int
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if np.all(values == values[0]):
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    chunk = min(500, replicates)
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_groups(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        refreshes = np.asarray([float(row["refreshes"]) for row in group])
        violation_fraction = np.asarray(
            [float(row["violation_years"]) / HORIZON_YEARS for row in group]
        )
        failures = int(sum(int(row["any_violation"]) for row in group))
        failure_low, failure_high = wilson_interval(failures, len(group))
        refresh_low, refresh_high = bootstrap_mean_interval(
            refreshes, stable_seed(key + ("refreshes",)), bootstrap_replicates
        )
        violation_low, violation_high = bootstrap_mean_interval(
            violation_fraction,
            stable_seed(key + ("violation",)),
            bootstrap_replicates,
        )
        output.append(
            {
                **dict(zip(keys, key)),
                "episodes": len(group),
                "mean_refreshes": float(refreshes.mean()),
                "mean_refreshes_ci95_low": refresh_low,
                "mean_refreshes_ci95_high": refresh_high,
                "violation_rate": float(violation_fraction.mean()),
                "violation_rate_ci95_low": violation_low,
                "violation_rate_ci95_high": violation_high,
                "episode_failure_rate": failures / len(group),
                "episode_failure_rate_ci95_low": failure_low,
                "episode_failure_rate_ci95_high": failure_high,
                "interval_scope": "finite-simulation uncertainty at the episode unit",
                "claim_upgrade_allowed": False,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    low, high = wilson_interval(0, 400)
    assert low == 0.0
    assert 0.0 < high < 0.02
    values = np.asarray([0.0, 0.0, 1.0, 1.0])
    first = bootstrap_mean_interval(values, 7, 1_000)
    second = bootstrap_mean_interval(values, 7, 1_000)
    assert first == second
    assert first[0] <= 0.5 <= first[1]
    print("PASS_PAPER4_MONTE_CARLO_UNCERTAINTY_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    seed_rows = read_csv(ROOT / "data/paper4_archive_maintenance_seed_episodes.csv")
    sensitivity_rows = read_csv(ROOT / "data/paper4_policy_sensitivity_episodes.csv")
    seed_summary = summarize_groups(
        seed_rows, ("policy", "true_hazard"), args.bootstrap_replicates
    )
    sensitivity_summary = summarize_groups(
        sensitivity_rows,
        (
            "sweep",
            "policy",
            "true_hazard",
            "prior",
            "sentinel_size",
            "inspection_interval",
            "hazard_profile",
            "observation_factor",
        ),
        args.bootstrap_replicates,
    )
    write_csv(args.output_dir / "paper4_seed_uncertainty_summary.csv", seed_summary)
    write_csv(
        args.output_dir / "paper4_sensitivity_uncertainty_summary.csv",
        sensitivity_summary,
    )
    print("PASS_PAPER4_MONTE_CARLO_UNCERTAINTY")
    print(
        f"seed_groups={len(seed_summary)} sensitivity_groups={len(sensitivity_summary)} "
        f"bootstrap_replicates={args.bootstrap_replicates}"
    )


if __name__ == "__main__":
    main()
