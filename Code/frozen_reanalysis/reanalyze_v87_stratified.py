"""Read-only V8.7 held-out count audit and fixed-stratum bootstrap review.

This is a post hoc design-sensitivity analysis, not a replacement for the
locked V8.7 confirmatory bootstrap or a new simulation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STUDY_ID = "paper4_nc_v8_7_target_use_comparator_20260905"
POLICY = "posterior_mean+screen"
HAZARDS = ("0.002", "0.006", "0.012")
ALPHA_GATE = 0.05 / 6
REPLICATES = 20_000
COUNT_FIELDS = (
    "decisions", "candidate_no_refresh", "candidate_refresh",
    "eligible_candidates", "eligible_passes", "final_no_refresh",
    "candidate_under_target", "authorized_under_target",
    "any_authorized_under_target", "refreshes", "screen_overrides",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed(label: str) -> int:
    payload = f"{STUDY_ID}|bootstrap|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def draw_ratio(values: np.ndarray, label: str, fixed_strata: bool) -> float:
    """One-sided lower percentile of ratio of summed mission counts."""
    rng = np.random.default_rng(seed(label))
    groups = [values] if not fixed_strata else [values[values[:, 2] == h] for h in range(3)]
    patterns = []
    for group in groups:
        unique, counts = np.unique(group[:, :2], axis=0, return_counts=True)
        patterns.append((unique, counts / group.shape[0], group.shape[0]))
    output = np.empty(REPLICATES)
    for start in range(0, REPLICATES, 1024):
        stop = min(start + 1024, REPLICATES)
        numerator = np.zeros(stop - start)
        denominator = np.zeros(stop - start)
        for unique, probabilities, size in patterns:
            weights = rng.multinomial(size, probabilities, size=stop - start)
            numerator += weights @ unique[:, 0]
            denominator += weights @ unique[:, 1]
        output[start:stop] = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        )
    return float(np.quantile(output, ALPHA_GATE))


def draw_mean(values: np.ndarray, label: str, fixed_strata: bool) -> float:
    """One-sided upper percentile of paired screen-minus-periodic refreshes."""
    rng = np.random.default_rng(seed(label))
    groups = [values] if not fixed_strata else [values[values[:, 1] == h] for h in range(3)]
    patterns = []
    for group in groups:
        unique, counts = np.unique(group[:, 0], return_counts=True)
        patterns.append((unique, counts / group.shape[0], group.shape[0]))
    output = np.empty(REPLICATES)
    total = values.shape[0]
    for start in range(0, REPLICATES, 1024):
        stop = min(start + 1024, REPLICATES)
        numerator = np.zeros(stop - start)
        for unique, probabilities, size in patterns:
            weights = rng.multinomial(size, probabilities, size=stop - start)
            numerator += weights @ unique
        output[start:stop] = numerator / total
    return float(np.quantile(output, 1 - ALPHA_GATE))


def parse_selected(path: Path) -> dict[str, list[dict[str, int]]]:
    selected: dict[str, list[dict[str, int]]] = defaultdict(list)
    seen: dict[str, set[int]] = defaultdict(set)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = set(COUNT_FIELDS) | {
            "study_id", "cohort", "policy", "screened", "true_hazard", "episode"
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Required source fields are missing")
        for row in reader:
            if row["policy"] != POLICY:
                continue
            if row["study_id"] != STUDY_ID or row["cohort"] != "test" or row["screened"] != "1":
                raise ValueError("Selected policy row is not the locked held-out screen")
            hazard = row["true_hazard"]
            if hazard not in HAZARDS:
                raise ValueError(f"Unexpected hazard {hazard}")
            mission = int(row["episode"])
            if mission in seen[hazard]:
                raise ValueError(f"Duplicate mission {hazard}/{mission}")
            seen[hazard].add(mission)
            typed = {field: int(row[field]) for field in COUNT_FIELDS}
            typed["episode"] = mission
            if typed["decisions"] != 11:
                raise ValueError("Nonstandard decision count")
            if typed["candidate_no_refresh"] + typed["candidate_refresh"] != 11:
                raise ValueError("Candidate accounting failed")
            if typed["final_no_refresh"] + typed["refreshes"] != 11:
                raise ValueError("Final action accounting failed")
            if typed["candidate_no_refresh"] - typed["screen_overrides"] != typed["final_no_refresh"]:
                raise ValueError("Screen override accounting failed")
            if int(typed["authorized_under_target"] > 0) != typed["any_authorized_under_target"]:
                raise ValueError("Affected mission flag disagrees with action count")
            selected[hazard].append(typed)
    for hazard in HAZARDS:
        if len(selected[hazard]) != 20_000:
            raise ValueError(f"Expected 20,000 missions at {hazard}, found {len(selected[hazard])}")
    return selected


def aggregate(selected: dict[str, list[dict[str, int]]]) -> list[dict[str, int | str]]:
    result = []
    for hazard in HAZARDS:
        rows = selected[hazard]
        record: dict[str, int | str] = {"hazard": hazard, "missions": len(rows)}
        for field in COUNT_FIELDS:
            record[field] = sum(row[field] for row in rows)
        result.append(record)
    pooled: dict[str, int | str] = {"hazard": "pooled", "missions": 60_000}
    for field in COUNT_FIELDS:
        pooled[field] = sum(int(record[field]) for record in result)
    result.append(pooled)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--locked-gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = parse_selected(args.episodes)
    counts = aggregate(selected)
    pooled = counts[-1]
    assert (pooled["eligible_candidates"], pooled["eligible_passes"]) == (174_120, 147_451)
    assert (pooled["final_no_refresh"], pooled["refreshes"]) == (189_301, 470_699)
    assert [row["any_authorized_under_target"] for row in counts[:3]] == [0, 0, 33]

    # The same complete mission retains its candidate, decision and action counts.
    auth = np.asarray([
        (row["final_no_refresh"], row["decisions"], index)
        for index, hazard in enumerate(HAZARDS) for row in selected[hazard]
    ], dtype=np.int64)
    eligible = np.asarray([
        (row["eligible_passes"], row["eligible_candidates"], index)
        for index, hazard in enumerate(HAZARDS) for row in selected[hazard]
    ], dtype=np.int64)
    refresh = np.asarray([
        (row["refreshes"] - 11, index)
        for index, hazard in enumerate(HAZARDS) for row in selected[hazard]
    ], dtype=np.int64)
    with args.locked_gates.open(newline="", encoding="utf-8") as stream:
        gates = {row["gate"]: row for row in csv.DictReader(stream)}

    metrics = (
        ("eligible_passage", eligible, "ratio", 0.25,
         "test|posterior_mean+screen|eligible-pass"),
        ("overall_authorization", auth, "ratio", 0.05,
         "test|posterior_mean+screen|overall-auth"),
        ("refresh_gain", refresh, "mean", -0.5,
         "test|posterior_mean+screen|vs-periodic-5|refresh"),
    )
    contrasts = []
    for name, values, kind, threshold, locked_label in metrics:
        point = (float(values[:, 0].sum() / values[:, 1].sum()) if kind == "ratio"
                 else float(values[:, 0].mean()))
        original_recomputed = (
            draw_ratio(values, locked_label, False) if kind == "ratio"
            else draw_mean(values, locked_label, False)
        )
        original_locked = float(gates[name]["estimate_or_bound"])
        if abs(original_recomputed - original_locked) > 1e-12:
            raise ValueError(f"Original {name} bootstrap did not reproduce lock")
        new_label = f"review2|fixed-strata|{name}"
        stratified = (
            draw_ratio(values, new_label, True) if kind == "ratio"
            else draw_mean(values, new_label, True)
        )
        passed = stratified >= threshold if kind == "ratio" else stratified < threshold
        contrasts.append({
            "gate": name,
            "point_estimate": point,
            "original_pooled_bound": original_locked,
            "recomputed_original_pooled_bound": original_recomputed,
            "fixed_stratum_bound": stratified,
            "fixed_minus_original": stratified - original_locked,
            "criterion": ">=" if kind == "ratio" else "<",
            "threshold": threshold,
            "fixed_stratum_pass": int(passed),
            "replicates": REPLICATES,
            "alpha_gate": ALPHA_GATE,
            "new_seed_label": new_label,
        })

    # Aggregation is a commutative sum; check explicitly against reversed input.
    reversed_selected = {hazard: list(reversed(selected[hazard])) for hazard in HAZARDS}
    if aggregate(reversed_selected) != counts:
        raise AssertionError("Counts depend on input order")
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, records in (("stratum_counts.csv", counts),
                              ("original_vs_stratified.csv", contrasts)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "status": "POST_HOC_FIXED_STRATUM_REANALYSIS",
        "source_episodes_sha256": sha256(args.episodes),
        "source_locked_gates_sha256": sha256(args.locked_gates),
        "policy": POLICY,
        "cohort": "test",
        "stratum_counts": counts,
        "contrasts": contrasts,
        "original_pooled_reproduced": True,
        "order_invariance_pass": True,
        "note": "Frozen missions only; no simulation rerun. Approximate bootstrap bounds remain approximate.",
    }
    (args.output / "reanalysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stratum_counts": counts, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
