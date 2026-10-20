#!/usr/bin/env python3
"""Analytical design audits for the Paper 4 maintenance certificate.

The outputs separate three issues that the false-safe theorem alone does not
resolve: residual stochastic erasure risk, the cycle-entry inspection gap, and
dependence among archive erasures.  They are model-design calculations, not
measurements of physical DNA archives.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import betabinom, binom


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_OLIGOS = 10_000
ERASURE_TOLERANCE = 0.10
RECOVERABILITY_TARGET = 0.99
HORIZON_YEARS = 60
INSPECTION_INTERVAL = 5
ALPHA_TOTAL = 0.05
A_REF_YEARS = 1.0


def critical_dropout(
    intraclass_correlation: float,
    archive_oligos: int = ARCHIVE_OLIGOS,
    erasure_tolerance: float = ERASURE_TOLERANCE,
    recoverability_target: float = RECOVERABILITY_TARGET,
) -> float:
    """Largest mean dropout with recovery probability equal to the target."""
    cutoff = math.floor(archive_oligos * erasure_tolerance)

    def recovery(dropout: float) -> float:
        if intraclass_correlation == 0.0:
            return float(binom.cdf(cutoff, archive_oligos, dropout))
        concentration = 1.0 / intraclass_correlation - 1.0
        alpha = dropout * concentration
        beta = (1.0 - dropout) * concentration
        return float(betabinom.cdf(cutoff, archive_oligos, alpha, beta))

    return float(
        brentq(
            lambda dropout: recovery(dropout) - recoverability_target,
            1e-10,
            erasure_tolerance,
        )
    )


def exposure(age: float, acceleration: float) -> float:
    """Exposure in years; acceleration is expressed in year^-1."""
    return age * (
        1.0 + acceleration * max(age - A_REF_YEARS, 0.0)
    )


def inverse_exposure(target: float, acceleration: float) -> float:
    if target <= 0.0:
        return 0.0
    if acceleration == 0.0:
        return target
    linear = 1.0 - acceleration
    return (-linear + math.sqrt(linear**2 + 4.0 * acceleration * target)) / (
        2.0 * acceleration
    )


def erasure_dependence_rows() -> list[dict[str, object]]:
    independent_qcrit = critical_dropout(0.0)
    rows = []
    for rho in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        qcrit = critical_dropout(rho)
        rows.append(
            {
                "archive_oligos": ARCHIVE_OLIGOS,
                "erasure_tolerance": ERASURE_TOLERANCE,
                "recoverability_target": RECOVERABILITY_TARGET,
                "intraclass_correlation": rho,
                "critical_mean_dropout": qcrit,
                "critical_dropout_relative_to_independent": qcrit
                / independent_qcrit,
                "model_scope": "beta-binomial exchangeable-erasure stress",
            }
        )
    return rows


def service_model_rows() -> list[dict[str, object]]:
    """Sensitivity of the service threshold to code and target assumptions."""
    rows: list[dict[str, object]] = []
    for archive_oligos in (1_000, 10_000, 100_000):
        for erasure_tolerance in (0.05, 0.10, 0.15, 0.20):
            for recoverability_target in (0.95, 0.99, 0.999):
                independent = critical_dropout(
                    0.0,
                    archive_oligos,
                    erasure_tolerance,
                    recoverability_target,
                )
                dependent = critical_dropout(
                    1e-4,
                    archive_oligos,
                    erasure_tolerance,
                    recoverability_target,
                )
                rows.append(
                    {
                        "archive_oligos": archive_oligos,
                        "erasure_tolerance": erasure_tolerance,
                        "recoverability_target": recoverability_target,
                        "independent_critical_dropout": independent,
                        "beta_binomial_rho": 1e-4,
                        "robust_critical_dropout": dependent,
                        "robust_to_independent_threshold_ratio": dependent
                        / independent,
                        "maximum_cycle_entry_years_h006_c003_independent": (
                            inverse_exposure(
                                -math.log1p(-independent) / 0.006, 0.03
                            )
                        ),
                        "maximum_cycle_entry_years_h006_c003_robust": (
                            inverse_exposure(
                                -math.log1p(-dependent) / 0.006, 0.03
                            )
                        ),
                        "scope": "model-design sensitivity; not a physical DNA calibration",
                    }
                )
    return rows


def cycle_entry_rows() -> list[dict[str, object]]:
    qcrit = critical_dropout(0.0)
    rows = []
    for residual_dropout in (0.0, 0.01):
        for hazard in (0.006, 0.012):
            for acceleration in (0.0, 0.03):
                if residual_dropout >= qcrit:
                    hazard_budget = 0.0
                else:
                    hazard_budget = -math.log(
                        (1.0 - qcrit) / (1.0 - residual_dropout)
                    )
                maximum_interval = inverse_exposure(
                    hazard_budget / hazard, acceleration
                )
                rows.append(
                    {
                        "archive_oligos": ARCHIVE_OLIGOS,
                        "erasure_tolerance": ERASURE_TOLERANCE,
                        "recoverability_target": RECOVERABILITY_TARGET,
                        "critical_dropout": qcrit,
                        "post_refresh_residual_dropout": residual_dropout,
                        "hazard_ceiling": hazard,
                        "acceleration_ceiling": acceleration,
                        "maximum_cycle_entry_interval_years": maximum_interval,
                        "five_year_schedule_feasible": maximum_interval
                        >= INSPECTION_INTERVAL,
                        "ten_year_schedule_feasible": maximum_interval >= 10.0,
                    }
                )
    return rows


def mission_risk_rows() -> list[dict[str, object]]:
    max_actions = (HORIZON_YEARS - 1) // INSPECTION_INTERVAL
    designs = (
        (
            "current_per_endpoint_target",
            ALPHA_TOTAL,
            max_actions * (1.0 - RECOVERABILITY_TARGET),
            RECOVERABILITY_TARGET,
        ),
        (
            "alpha_0.05_plus_beta_0.05",
            0.05,
            0.05,
            1.0 - 0.05 / max_actions,
        ),
        (
            "balanced_total_mission_bound_0.05",
            0.025,
            0.025,
            1.0 - 0.025 / max_actions,
        ),
    )
    return [
        {
            "design": name,
            "horizon_years": HORIZON_YEARS,
            "inspection_interval_years": INSPECTION_INTERVAL,
            "maximum_no_refresh_endpoints": max_actions,
            "false_safe_budget_alpha": alpha,
            "residual_random_failure_budget_beta": beta,
            "required_per_endpoint_recoverability": endpoint_target,
            "coarse_ideal_model_mission_failure_upper_bound": min(
                1.0, alpha + beta
            ),
            "delta_env_status": "uncalibrated_not_assumed_zero",
            "delta_svc_status": "uncalibrated_not_assumed_zero",
            "risk_decomposition": (
                "under the explicit event-cover assumption: P(mission failure) "
                "<= alpha_false_safe_within_valid_envelope + "
                "beta_joint_residual_service + delta_env + delta_svc + "
                "delta_entry + delta_refresh + delta_other"
            ),
            "included_terms": (
                "alpha false-safe decision; beta residual service; delta_env "
                "observation/transition-envelope failure; delta_svc "
                "service-envelope failure; delta_entry cycle-entry failure; "
                "delta_refresh refresh failure; delta_other event-cover remainder"
            ),
            "event_cover_assumption": (
                "every mission failure is a protected no-refresh endpoint "
                "failure or belongs to cycle-entry, refresh, or other events"
            ),
            "excluded_terms": (
                "the coarse alpha-plus-beta ideal-model subtotal excludes "
                "uncalibrated delta_env state-envelope failure, delta_svc "
                "service-envelope failure, delta_entry schedule gaps, "
                "delta_refresh refresh failure, and delta_other mechanisms"
            ),
        }
        for name, alpha, beta, endpoint_target in designs
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    dependence = erasure_dependence_rows()
    qcrits = [float(row["critical_mean_dropout"]) for row in dependence]
    assert all(left >= right for left, right in zip(qcrits, qcrits[1:]))
    assert abs(qcrits[0] - 0.09322709318657661) < 1e-12
    cycle = cycle_entry_rows()
    assert len(cycle) == 8
    assert all(float(row["maximum_cycle_entry_interval_years"]) >= 0 for row in cycle)
    mission = mission_risk_rows()
    assert int(mission[0]["maximum_no_refresh_endpoints"]) == 11
    assert abs(
        float(mission[0]["coarse_ideal_model_mission_failure_upper_bound"])
        - 0.16
    ) < 1e-12
    assert all(
        "delta_svc" in str(row["risk_decomposition"])
        and "service-envelope failure" in str(row["included_terms"])
        and row["delta_svc_status"] == "uncalibrated_not_assumed_zero"
        for row in mission
    )
    service = service_model_rows()
    assert len(service) == 36
    assert all(
        float(row["robust_critical_dropout"])
        <= float(row["independent_critical_dropout"])
        for row in service
    )
    print("PASS_PAPER4_MODEL_RISK_DESIGN_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    write_csv(
        args.output_dir / "paper4_erasure_dependence_stress.csv",
        erasure_dependence_rows(),
    )
    write_csv(
        args.output_dir / "paper4_cycle_entry_design.csv", cycle_entry_rows()
    )
    write_csv(
        args.output_dir / "paper4_mission_risk_design.csv", mission_risk_rows()
    )
    write_csv(
        args.output_dir / "paper4_service_model_sensitivity.csv",
        service_model_rows(),
    )
    print("PASS_PAPER4_MODEL_RISK_DESIGN")


if __name__ == "__main__":
    main()
