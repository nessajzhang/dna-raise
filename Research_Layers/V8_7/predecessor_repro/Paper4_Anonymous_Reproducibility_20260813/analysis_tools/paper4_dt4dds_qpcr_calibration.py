#!/usr/bin/env python3
"""Reproduce DT4DDS qPCR accelerated-aging concentration calibration.

The outputs quantify amplifiable concentration decay at 70 C and 50% relative
humidity. They are not oligo-dropout or file-recovery measurements.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np


POOLS = {
    "Genscript_GCall": {"label": "Genscript (GCall)", "stock": 0.0464},
    "Genscript_GCfix": {"label": "Genscript (GCfix)", "stock": 0.2160},
    "Twist_GCall": {"label": "Twist (GCall)", "stock": 0.0045},
    "Twist_GCfix": {"label": "Twist (GCfix)", "stock": 0.0048},
}

NOTEBOOK_HALF_LIVES = {
    "Genscript_GCall": 0.932540,
    "Genscript_GCfix": 0.962264,
    "Twist_GCall": 0.996436,
    "Twist_GCfix": 0.824026,
}

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEEDS = {
    "Genscript_GCall": 2026071901,
    "Genscript_GCfix": 2026071902,
    "Twist_GCall": 2026071903,
    "Twist_GCfix": 2026071904,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]:
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    ss_x = sum((value - x_mean) ** 2 for value in x)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / ss_x
    intercept = y_mean - slope * x_mean
    residual = sum((b - (slope * a + intercept)) ** 2 for a, b in zip(x, y))
    total = sum((b - y_mean) ** 2 for b in y)
    return slope, intercept, 1.0 - residual / total


def through_origin(
    x: list[float], y: list[float]
) -> tuple[float, float, float, int]:
    """Fit through the origin without treating constructed baselines as residual df.

    Time-zero log-relative concentration is exactly zero by normalization.  It
    may remain in the slope calculation (where it contributes zero), but it is
    not an independent residual-bearing observation.  Standard errors and the
    uncentered residual summary therefore use nonzero times only.
    """
    ss_x = sum(value * value for value in x)
    slope = sum(a * b for a, b in zip(x, y)) / ss_x
    residual_rows = [
        (a, b) for a, b in zip(x, y) if not math.isclose(a, 0.0, abs_tol=1e-15)
    ]
    residual = sum((b - slope * a) ** 2 for a, b in residual_rows)
    degrees = len(residual_rows) - 1
    if degrees <= 0:
        raise ValueError("through-origin uncertainty requires at least two nonzero times")
    slope_se = math.sqrt((residual / degrees) / ss_x)
    uncentered_total = sum(b * b for _, b in residual_rows)
    r2 = 1.0 - residual / uncentered_total
    return slope, slope_se, r2, degrees


def stratified_indices(
    rng: np.random.Generator, groups: dict[float, list[int]]
) -> list[int]:
    indices: list[int] = []
    for key in sorted(groups):
        source = np.asarray(groups[key], dtype=int)
        indices.extend(rng.choice(source, size=len(source), replace=True).tolist())
    return indices


def technical_bootstrap_half_life(
    calibration: list[dict[str, str]],
    decay: list[dict[str, str]],
    stock: float,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float, int]:
    """Propagate calibration-well and qPCR technical-repeat variation.

    This bootstrap is stratified by calibration dilution and aging time.  Each
    draw refits the calibration curve, recomputes the time-zero normalization,
    and refits the through-origin decay slope.  It is descriptive of released
    technical measurements, not an interval for biological batches or long-term
    storage conditions.
    """
    usable_calibration = [row for row in calibration if row["Dilution"] != "NA"]
    cal_groups: dict[float, list[int]] = defaultdict(list)
    for index, row in enumerate(usable_calibration):
        cal_groups[float(row["Dilution"])].append(index)
    decay_groups: dict[float, list[int]] = defaultdict(list)
    for index, row in enumerate(decay):
        decay_groups[float(row["Time"]) / 24.0].append(index)

    rng = np.random.default_rng(seed)
    half_lives: list[float] = []
    for _ in range(replicates):
        cal_sample = [usable_calibration[i] for i in stratified_indices(rng, cal_groups)]
        x = [
            math.log10(stock / float(row["Dilution"]))
            for row in cal_sample
        ]
        y = [float(row["Ct"]) for row in cal_sample]
        cal_slope, cal_intercept, _ = linear_regression(x, y)

        by_time: dict[float, list[float]] = defaultdict(list)
        for index in stratified_indices(rng, decay_groups):
            row = decay[index]
            time = float(row["Time"]) / 24.0
            concentration = 10 ** ((float(row["Ct"]) - cal_intercept) / cal_slope)
            by_time[time].append(concentration)
        c0 = statistics.mean(by_time[0.0])
        times = sorted(by_time)
        log_relative = [
            math.log10(statistics.mean(by_time[time]) / c0) for time in times
        ]
        slope = sum(a * b for a, b in zip(times, log_relative)) / sum(
            a * a for a in times
        )
        rate = -slope * math.log(10)
        if rate > 0.0 and math.isfinite(rate):
            half_lives.append(math.log(2) / rate)

    if len(half_lives) < 0.99 * replicates:
        raise RuntimeError("too many invalid technical-bootstrap decay fits")
    low, high = np.quantile(np.asarray(half_lives), [0.025, 0.975])
    return float(low), float(high), len(half_lives)


def calibrate_pool(source_dir: Path, key: str) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    config = POOLS[key]
    calibration = read_csv(source_dir / f"cal_{key}.csv")
    x, y = [], []
    for row in calibration:
        if row["Dilution"] == "NA":
            continue
        nominal = float(config["stock"]) / float(row["Dilution"])
        x.append(math.log10(nominal))
        y.append(float(row["Ct"]))
    cal_slope, cal_intercept, cal_r2 = linear_regression(x, y)

    decay = read_csv(source_dir / f"decay_{key}.csv")
    concentrations = []
    for row in decay:
        concentration = 10 ** ((float(row["Ct"]) - cal_intercept) / cal_slope)
        concentrations.append((float(row["Time"]) / 24.0, concentration))
    c0 = statistics.mean(value for time, value in concentrations if time == 0.0)

    by_time: dict[float, list[float]] = defaultdict(list)
    for time, concentration in concentrations:
        by_time[time].append(concentration / c0)

    points = []
    for time in sorted(by_time):
        values = by_time[time]
        points.append(
            {
                "pool": config["label"],
                "pool_key": key,
                "time_days": time,
                "replicates": len(values),
                "relative_concentration_mean": statistics.mean(values),
                "relative_concentration_sd": statistics.stdev(values),
            }
        )

    time = [float(row["time_days"]) for row in points]
    log_relative = [math.log10(float(row["relative_concentration_mean"])) for row in points]
    decay_slope, decay_slope_se, decay_r2, regression_df = through_origin(
        time, log_relative
    )
    rate = -decay_slope * math.log(10)
    half_life = math.log(2) / rate

    bootstrap_low, bootstrap_high, bootstrap_valid = technical_bootstrap_half_life(
        calibration,
        decay,
        float(config["stock"]),
        BOOTSTRAP_SEEDS[key],
    )

    summary = {
        "pool": config["label"],
        "pool_key": key,
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
        "calibration_r2": cal_r2,
        "log10_decay_slope_per_day": decay_slope,
        "log10_decay_slope_se": decay_slope_se,
        "decay_r2_uncentered": decay_r2,
        "positive_decay_rate_per_day": rate,
        "decay_regression_df": regression_df,
        "half_life_days": half_life,
        "half_life_technical_bootstrap_ci95_low_days": bootstrap_low,
        "half_life_technical_bootstrap_ci95_high_days": bootstrap_high,
        "technical_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "technical_bootstrap_valid_fits": bootstrap_valid,
        "uncertainty_scope": (
            "descriptive technical-repeat bootstrap; not biological-batch "
            "or long-term storage uncertainty"
        ),
    }
    return summary, points


def leave_one_pool_out(points: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    results = []
    for holdout in POOLS:
        train = [row for row in points if row["pool_key"] != holdout]
        test = [row for row in points if row["pool_key"] == holdout]
        x_train = [float(row["time_days"]) for row in train]
        y_train = [math.log10(float(row["relative_concentration_mean"])) for row in train]
        slope, slope_se, r2, regression_df = through_origin(x_train, y_train)
        errors = [
            math.log10(float(row["relative_concentration_mean"])) - slope * float(row["time_days"])
            for row in test
            if float(row["time_days"]) > 0.0
        ]
        results.append(
            {
                "holdout_pool": POOLS[holdout]["label"],
                "holdout_pool_key": holdout,
                "training_pools": len(POOLS) - 1,
                "training_log10_slope_per_day": slope,
                "training_slope_se": slope_se,
                "training_r2_uncentered": r2,
                "training_regression_df": regression_df,
                "holdout_nonzero_timepoints": len(errors),
                "holdout_rmse_log10": math.sqrt(statistics.mean(value * value for value in errors)),
                "holdout_mae_log10": statistics.mean(abs(value) for value in errors),
                "holdout_final_observed_relative_concentration": float(test[-1]["relative_concentration_mean"]),
                "holdout_final_predicted_relative_concentration": 10 ** (slope * float(test[-1]["time_days"])),
            }
        )
    return results


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    """Exercise pure numerical helpers without requiring external qPCR files."""
    slope, intercept, r2 = linear_regression([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert math.isclose(slope, 2.0, abs_tol=1e-14)
    assert math.isclose(intercept, 1.0, abs_tol=1e-14)
    assert math.isclose(r2, 1.0, abs_tol=1e-14)
    origin_slope, origin_se, origin_r2, origin_df = through_origin(
        [0.0, 1.0, 2.0, 3.0], [0.0, -0.1, -0.2, -0.3]
    )
    assert math.isclose(origin_slope, -0.1, abs_tol=1e-14)
    assert math.isclose(origin_se, 0.0, abs_tol=1e-14)
    assert math.isclose(origin_r2, 1.0, abs_tol=1e-14)
    assert origin_df == 2
    first = stratified_indices(
        np.random.default_rng(17), {0.0: [0, 1], 1.0: [2, 3, 4]}
    )
    second = stratified_indices(
        np.random.default_rng(17), {0.0: [0, 1], 1.0: [2, 3, 4]}
    )
    assert first == second
    assert len(first) == 5
    assert all(index in {0, 1} for index in first[:2])
    assert all(index in {2, 3, 4} for index in first[2:])
    print("PASS_PAPER4_DT4DDS_QPCR_CALIBRATION_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("external_data/paper4_dt4dds"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1))
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    summaries, points = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        calibrated = list(
            pool.map(calibrate_pool, repeat(args.source_dir), POOLS)
        )
    for summary, pool_points in calibrated:
        summaries.append(summary)
        points.extend(pool_points)

    for row in summaries:
        expected = NOTEBOOK_HALF_LIVES[str(row["pool_key"])]
        assert abs(float(row["half_life_days"]) - expected) < 5e-6
        assert float(row["calibration_r2"]) > 0.999
        assert int(row["decay_regression_df"]) == 2
        assert int(row["technical_bootstrap_valid_fits"]) >= 9_900
    assert len(points) == 16

    loo = leave_one_pool_out(points)
    assert all(int(row["holdout_nonzero_timepoints"]) == 3 for row in loo)
    write_csv(args.output_dir / "paper4_dt4dds_qpcr_decay_summary.csv", summaries)
    write_csv(args.output_dir / "paper4_dt4dds_qpcr_decay_timepoints.csv", points)
    write_csv(args.output_dir / "paper4_dt4dds_qpcr_leave_one_pool_out.csv", loo)
    print("PASS_PAPER4_DT4DDS_QPCR_CALIBRATION")
    for row in summaries:
        print(
            f"{row['pool']}: half_life={row['half_life_days']:.6f} d, "
            f"rate={row['positive_decay_rate_per_day']:.6f}/d, "
            f"R2={row['decay_r2_uncentered']:.6f}"
        )
    for row in loo:
        print(
            f"LOO {row['holdout_pool']}: RMSE_log10={row['holdout_rmse_log10']:.6f}, "
            f"final_observed={row['holdout_final_observed_relative_concentration']:.6f}, "
            f"final_predicted={row['holdout_final_predicted_relative_concentration']:.6f}"
        )


if __name__ == "__main__":
    main()
