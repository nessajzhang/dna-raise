#!/usr/bin/env python3
"""Deterministic exact-binomial witness for the high-hazard screen edge case.

This is an arithmetic self-test, not a simulation.  It keeps the numerical
constants used in the manuscript and guards against turning a rare screen pass
into a safe-domain, mission-level, or independent-look claim.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import beta, binom


def exposure(age: float, acceleration: float = 0.0) -> float:
    return age * (1.0 + acceleration * max(age - 1.0, 0.0))


def dropout(hazard: float, age: float, acceleration: float = 0.0) -> float:
    return 1.0 - math.exp(-hazard * exposure(age, acceleration))


def service_recoverability(q: float, n: int = 10_000, tau: float = 0.10) -> float:
    return float(binom.cdf(math.floor(n * tau), n, q))


def main() -> None:
    qcrit = brentq(lambda q: service_recoverability(q) - 0.99, 1e-12, 0.10)
    q10 = dropout(0.012, 10.0)
    alpha_j = 0.05 / 11.0
    u5 = float(beta.ppf(1.0 - alpha_j, 6, 295))
    u6 = float(beta.ppf(1.0 - alpha_j, 7, 294))
    passing = []
    for y in range(301):
        upper = 1.0 if y == 300 else float(beta.ppf(1.0 - alpha_j, y + 1, 300 - y))
        propagated = 1.0 - (1.0 - upper) ** 2.0
        if service_recoverability(propagated) >= 0.99:
            passing.append(y)
    y_star = max(passing) if passing else -1
    screen_pass = float(binom.cdf(y_star, 300, dropout(0.012, 5.0)))

    expected = {
        "qcrit": 0.09322709318655772,
        "q10": 0.11307956328284252,
        "u5": 0.04690149884914084,
        "u6": 0.051832387067533625,
        "screen_pass_probability": 0.00036226371029259,
    }
    observed = {
        "qcrit": qcrit,
        "q10": q10,
        "u5": u5,
        "u6": u6,
        "screen_pass_probability": screen_pass,
    }
    for key, target in expected.items():
        assert math.isclose(observed[key], target, rel_tol=0.0, abs_tol=2e-14), (
            key,
            observed[key],
            target,
        )
    assert q10 > expected["qcrit"]
    assert y_star == 5
    print("PASS_PAPER4_HIGH_RISK_WITNESS")
    for key in expected:
        # Print the manuscript's declared decimal so the receipt remains
        # stable across scipy patch-level root-finding changes.
        print(f"{key}={expected[key]:.17g}")
    print(
        "interpretation=model-unsafe selective-risk witness; "
        "not a safe-domain diagnostic or mission approximation"
    )


if __name__ == "__main__":
    main()
