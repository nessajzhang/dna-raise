#!/usr/bin/env python3
"""Illustrative sampling-law sensitivity for the exact-binomial screen.

The calculation evaluates the declared Clopper--Pearson upper bound under a
prespecified beta-binomial alternative. It is not an estimate of the physical
archive's observation process and does not repair misspecification.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import betabinom, binom

from audit_paper4_finite_horizon_certificate import (
    clopper_pearson_upper,
    critical_dropout,
)


ROOT = Path(__file__).resolve().parents[1]
RHO_VALUES = (0.0, 0.001, 0.005, 0.010)


def noncoverage_probability(
    p: float,
    upper_bounds: np.ndarray,
    sample_size: int,
    rho: float,
) -> float:
    failing = np.flatnonzero(upper_bounds < p)
    if failing.size == 0:
        return 0.0
    cutoff = int(failing[-1])
    if rho == 0.0:
        return float(binom.cdf(cutoff, sample_size, p))
    concentration = 1.0 / rho - 1.0
    shape_a = p * concentration
    shape_b = (1.0 - p) * concentration
    return float(betabinom.cdf(cutoff, sample_size, shape_a, shape_b))


def candidate_probabilities(upper_bounds: np.ndarray, q_limit: float) -> np.ndarray:
    dense = np.linspace(1e-5, q_limit, 4_000)
    thresholds = np.asarray(
        [
            np.nextafter(value, 1.0)
            for value in upper_bounds
            if 0.0 < value < q_limit
        ],
        dtype=float,
    )
    return np.unique(np.concatenate((dense, thresholds)))


def calculate(
    sample_size: int = 300,
    max_looks: int = 11,
    alpha_total: float = 0.05,
) -> list[dict[str, float | int | str]]:
    per_look_alpha = alpha_total / max_looks
    upper_bounds = np.asarray(
        [
            clopper_pearson_upper(y, sample_size, per_look_alpha)
            for y in range(sample_size + 1)
        ],
        dtype=float,
    )
    q_limit = critical_dropout(10_000, 0.10, 0.99)
    probabilities = candidate_probabilities(upper_bounds, q_limit)
    rows: list[dict[str, float | int | str]] = []
    for rho in RHO_VALUES:
        failures = np.asarray(
            [
                noncoverage_probability(p, upper_bounds, sample_size, rho)
                for p in probabilities
            ],
            dtype=float,
        )
        index = int(np.argmax(failures))
        maximum = float(failures[index])
        defect = max(0.0, maximum - per_look_alpha)
        rows.append(
            {
                "sample_size": sample_size,
                "max_looks": max_looks,
                "alpha_total": alpha_total,
                "per_look_alpha": per_look_alpha,
                "beta_binomial_intraclass_correlation": rho,
                "p_search_upper": q_limit,
                "p_at_max_noncoverage": float(probabilities[index]),
                "max_per_look_noncoverage": maximum,
                "excess_per_look_noncoverage": defect,
                "illustrative_delta_samp_upper": max_looks * defect,
                "illustrative_horizon_bound": min(
                    1.0, alpha_total + max_looks * defect
                ),
                "search_candidates": int(probabilities.size),
                "alternative_family": (
                    "binomial" if rho == 0.0 else "beta-binomial"
                ),
                "evidence_scope": (
                    "illustrative model-family sensitivity; not an empirical "
                    "observation-law estimate"
                ),
            }
        )
    return rows


def self_test() -> None:
    rows = calculate(sample_size=30, max_looks=3, alpha_total=0.05)
    assert len(rows) == len(RHO_VALUES)
    assert math.isclose(
        float(rows[0]["beta_binomial_intraclass_correlation"]), 0.0
    )
    assert float(rows[0]["max_per_look_noncoverage"]) <= 0.05 / 3 + 1e-10
    assert float(rows[0]["excess_per_look_noncoverage"]) <= 1e-10
    assert all(
        0.0 <= float(row["illustrative_horizon_bound"]) <= 1.0
        for row in rows
    )
    assert all(
        float(right["max_per_look_noncoverage"])
        + 1e-12
        >= float(left["max_per_look_noncoverage"])
        for left, right in zip(rows, rows[1:])
    )
    print("PASS_PAPER4_SAMPLING_LAW_SENSITIVITY_SELF_TEST")


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Illustrative beta-binomial sensitivity of exact-binomial coverage."
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--max-looks", type=int, default=11)
    parser.add_argument("--alpha-total", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.sample_size <= 0 or args.max_looks <= 0:
        raise SystemExit("sample size and maximum looks must be positive")
    if not 0.0 < args.alpha_total < 1.0:
        raise SystemExit("--alpha-total must lie in (0, 1)")
    rows = calculate(args.sample_size, args.max_looks, args.alpha_total)
    write_csv(args.output_dir / "paper4_sampling_law_sensitivity.csv", rows)
    print("PASS_PAPER4_SAMPLING_LAW_SENSITIVITY")
    print(f"rows={len(rows)} sample_size={args.sample_size} max_looks={args.max_looks}")


if __name__ == "__main__":
    main()
