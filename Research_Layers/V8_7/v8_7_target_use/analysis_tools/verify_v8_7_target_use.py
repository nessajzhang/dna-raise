#!/usr/bin/env python3
"""Verify the append-only Paper 4 V8.7 target-use study."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_v8_7_target_use as study  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_run_manifest(directory: Path) -> None:
    manifest = read_json(directory / "RUN_MANIFEST.json")
    require(manifest.get("study_id") == study.STUDY_ID, "wrong run-manifest study")
    runtime = manifest.get("runtime")
    require(isinstance(runtime, dict), "run manifest lacks runtime receipt")
    require(str(runtime.get("python", "")).startswith("3.12."), "wrong Python")
    require(runtime.get("numpy") == "2.1.3", "wrong NumPy")
    require(runtime.get("scipy") == "1.14.1", "wrong SciPy")
    require(runtime.get("slurm_job_id") != "NOT_SLURM", "run was not under Slurm")
    require(
        manifest.get("script_sha256")
        == study.sha256_file(Path(study.__file__)),
        "run used a different study script",
    )
    for item in manifest["files"]:  # type: ignore[index]
        path = directory / str(item["path"])
        require(path.is_file(), f"missing manifested file: {path}")
        require(path.stat().st_size == int(item["bytes"]), f"byte mismatch: {path}")
        require(
            study.sha256_file(path) == str(item["sha256"]),
            f"hash mismatch: {path}",
        )


def verify_seed_rows(
    rows: Sequence[dict[str, str]],
    *,
    namespace: str,
) -> set[int]:
    seeds: set[int] = set()
    mission_keys: dict[tuple[float, int], int] = {}
    for row in rows:
        hazard = float(row["true_hazard"])
        episode = int(row["episode"])
        seed = int(row["root_seed"])
        expected = study.stable_seed(namespace, hazard, episode)
        require(seed == expected, f"seed mismatch for {namespace}/{hazard}/{episode}")
        key = (hazard, episode)
        if key in mission_keys:
            require(mission_keys[key] == seed, "policies do not share root seed")
        else:
            mission_keys[key] = seed
        seeds.add(seed)
    return seeds


def floats_equal(left: object, right: object, atol: float = 1e-12) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)


def compare_summary(
    recorded: Sequence[dict[str, str]],
    recomputed: Sequence[dict[str, object]],
) -> None:
    rec_by_policy = {row["policy"]: row for row in recorded}
    new_by_policy = {str(row["policy"]): row for row in recomputed}
    require(rec_by_policy.keys() == new_by_policy.keys(), "summary policy set mismatch")
    fields = (
        "missions",
        "mean_refreshes",
        "mean_final_no_refresh",
        "overall_authorization_rate",
        "overall_authorization_lower",
        "eligible_candidates",
        "eligible_passes",
        "eligible_passage_rate",
        "eligible_passage_lower",
        "risk_events_json",
        "risk_rates_json",
        "risk_upper_json",
        "risk_pass_all_strata",
        "eligible_passage_gate",
        "overall_authorization_gate",
        "preburden_qualified",
        "candidate_declared_envelope_invalid_missions",
        "authorized_declared_envelope_invalid_missions",
        "candidate_realized_chain_failure_missions",
        "authorized_realized_chain_failure_missions",
        "sampling_undercoverage_missions",
    )
    for policy in rec_by_policy:
        for field in fields:
            require(
                floats_equal(rec_by_policy[policy][field], new_by_policy[policy][field]),
                f"summary mismatch: {policy}/{field}",
            )


def verify_predecessor_sources(root: Path, predecessor_root: Path | None) -> None:
    baseline = root / "baseline_sources"
    study.verify_hash_list(baseline / "BASELINE_SOURCE_SHA256.txt")
    if predecessor_root is None:
        return
    for copied in baseline.glob("*.py"):
        original = predecessor_root / "analysis_tools" / copied.name
        require(original.is_file(), f"missing predecessor source: {original}")
        require(
            study.sha256_file(copied) == study.sha256_file(original),
            f"baseline source differs from predecessor: {copied.name}",
        )


def verify_basic(root: Path, predecessor_root: Path | None) -> dict[str, object]:
    design_lock = root / "design" / "DESIGN_LOCK.sha256"
    require(design_lock.is_file(), "missing DESIGN_LOCK.sha256")
    study.verify_hash_list(design_lock)
    verify_predecessor_sources(root, predecessor_root)

    tune_dir = root / "results" / "tune"
    test_dir = root / "results" / "test"
    negative_dir = root / "results" / "negative_control"
    for directory in (tune_dir, test_dir, negative_dir):
        verify_run_manifest(directory)

    study.verify_hash_list(tune_dir / "TUNING_LOCK.sha256")
    study.verify_hash_list(test_dir / "TEST_OUTPUT_LOCK.sha256")
    study.verify_hash_list(
        negative_dir / "NEGATIVE_CONTROL_OUTPUT_LOCK.sha256"
    )

    tune_rows = read_csv_gz(tune_dir / "adaptive_episodes.csv.gz")
    test_rows = read_csv_gz(test_dir / "adaptive_episodes.csv.gz")
    negative_rows = read_csv_gz(negative_dir / "adaptive_episodes.csv.gz")
    require(
        len(tune_rows) == len(study.HAZARDS) * study.TUNE_MISSIONS_PER_HAZARD * 4,
        "wrong tuning row count",
    )
    require(
        len(test_rows) == len(study.HAZARDS) * study.TEST_MISSIONS_PER_HAZARD * 4,
        "wrong test row count",
    )
    require(
        len(negative_rows) == study.NEGATIVE_CONTROL_MISSIONS * 4,
        "wrong negative-control row count",
    )
    study.key_audit(
        tune_rows,
        hazards=study.HAZARDS,
        missions_per_hazard=study.TUNE_MISSIONS_PER_HAZARD,
    )
    study.key_audit(
        test_rows,
        hazards=study.HAZARDS,
        missions_per_hazard=study.TEST_MISSIONS_PER_HAZARD,
    )
    study.key_audit(
        negative_rows,
        hazards=(0.006,),
        missions_per_hazard=study.NEGATIVE_CONTROL_MISSIONS,
    )

    tune_seeds = verify_seed_rows(
        tune_rows, namespace="paper4-v8.7|tune|20260905"
    )
    test_seeds = verify_seed_rows(
        test_rows, namespace="paper4-v8.7|test|20260905"
    )
    negative_seeds = verify_seed_rows(
        negative_rows, namespace="paper4-v8.7|negative-control|20260905"
    )
    require(tune_seeds.isdisjoint(test_seeds), "tune/test seed overlap")
    require(tune_seeds.isdisjoint(negative_seeds), "tune/negative seed overlap")
    require(test_seeds.isdisjoint(negative_seeds), "test/negative seed overlap")

    require(
        all(int(row["declared_envelope_valid"]) == 1 for row in tune_rows),
        "target-use tuning contains invalid declared envelope",
    )
    require(
        all(int(row["declared_envelope_valid"]) == 1 for row in test_rows),
        "target-use test contains invalid declared envelope",
    )
    require(
        all(int(row["declared_envelope_valid"]) == 0 for row in negative_rows),
        "negative control did not violate declarations",
    )
    require(
        sum(int(row["candidate_declared_envelope_invalid_looks"]) for row in test_rows)
        == 0,
        "target-use test has declared-envelope violations",
    )

    selection = study.load_locked_selection(
        tune_dir / "SELECTION.json", tune_dir / "TUNING_LOCK.sha256"
    )
    require(
        selection["adaptive_episodes_sha256"]
        == study.sha256_file(tune_dir / "adaptive_episodes.csv.gz"),
        "selection episode hash mismatch",
    )
    require(
        selection["adaptive_summary_sha256"]
        == study.sha256_file(tune_dir / "adaptive_summary.csv"),
        "selection summary hash mismatch",
    )
    require(
        selection["periodic_frontier_sha256"]
        == study.sha256_file(tune_dir / "periodic_frontier.csv"),
        "selection periodic hash mismatch",
    )
    require(
        read_json(test_dir / "TUNING_SELECTION_USED.json") == selection,
        "test did not use exact locked selection",
    )
    negative_result = read_json(negative_dir / "NEGATIVE_CONTROL_RESULT.json")
    require(negative_result.get("study_id") == study.STUDY_ID, "wrong negative study")
    require(
        negative_result.get("selected_policy")
        == selection.get("selected_adaptive_policy"),
        "negative control used a different selected policy",
    )
    require(
        not bool(negative_result.get("pooled_into_target_use")),
        "negative control was pooled into target-use evidence",
    )
    require(
        bool(negative_result["negative_control_detected"]),
        "negative control was not detected",
    )
    test_run_lock = read_json(test_dir / "TEST_RUN_LOCK.json")
    require(
        test_run_lock.get("negative_control_lock_sha256")
        == study.sha256_file(
            negative_dir / "NEGATIVE_CONTROL_OUTPUT_LOCK.sha256"
        ),
        "test did not bind the locked negative control",
    )
    require(
        test_run_lock.get("negative_control_result_sha256")
        == study.sha256_file(negative_dir / "NEGATIVE_CONTROL_RESULT.json"),
        "test did not bind the negative-control result",
    )
    primary = read_json(test_dir / "PRIMARY_RESULT.json")
    return {
        "study_id": study.STUDY_ID,
        "basic_status": "PASS",
        "tune_rows": len(tune_rows),
        "test_rows": len(test_rows),
        "negative_control_rows": len(negative_rows),
        "tune_unique_seeds": len(tune_seeds),
        "test_unique_seeds": len(test_seeds),
        "negative_unique_seeds": len(negative_seeds),
        "selected_adaptive_policy": selection["selected_adaptive_policy"],
        "primary_result_status": primary["status"],
        "all_primary_gates_pass": primary["all_primary_gates_pass"],
        "negative_control_detected": True,
        "physical_qualification": "NOT_RUN",
    }


def verify_deep(root: Path) -> dict[str, object]:
    tune_dir = root / "results" / "tune"
    test_dir = root / "results" / "test"
    negative_dir = root / "results" / "negative_control"
    tune_rows = read_csv_gz(tune_dir / "adaptive_episodes.csv.gz")
    test_rows = read_csv_gz(test_dir / "adaptive_episodes.csv.gz")
    negative_rows = read_csv_gz(negative_dir / "adaptive_episodes.csv.gz")

    tune_recomputed = study.adaptive_summary(tune_rows, label_prefix="tune")
    compare_summary(read_csv(tune_dir / "adaptive_summary.csv"), tune_recomputed)
    tune_reversed = study.adaptive_summary(
        list(reversed(tune_rows)), label_prefix="tune"
    )
    compare_summary(read_csv(tune_dir / "adaptive_summary.csv"), tune_reversed)
    periodic = study.periodic_frontier(study.StudyConfig())
    selection_recomputed = study.select_tuning(tune_recomputed, periodic)
    selection_recorded = read_json(tune_dir / "SELECTION.json")
    for field in (
        "selected_adaptive_policy",
        "selected_periodic_joint_interval",
        "selected_periodic_risk_only_interval",
        "adaptive_qualified_candidates",
        "periodic_joint_candidates",
        "periodic_risk_only_candidates",
        "thresholds",
    ):
        require(
            selection_recomputed[field] == selection_recorded[field],
            f"tuning selection mismatch: {field}",
        )

    test_recomputed = study.adaptive_summary(test_rows, label_prefix="test")
    compare_summary(read_csv(test_dir / "adaptive_summary.csv"), test_recomputed)
    test_reversed = study.adaptive_summary(
        list(reversed(test_rows)), label_prefix="test"
    )
    compare_summary(read_csv(test_dir / "adaptive_summary.csv"), test_reversed)
    gates, primary_recomputed = study.test_gate_table(
        test_rows, test_recomputed, selection_recorded, study.StudyConfig()
    )
    primary_recorded = read_json(test_dir / "PRIMARY_RESULT.json")
    for field in primary_recomputed:
        require(
            floats_equal(primary_recomputed[field], primary_recorded[field]),
            f"primary result mismatch: {field}",
        )
    if gates:
        recorded_gates = read_csv(test_dir / "PRIMARY_GATE_TABLE.csv")
        require(len(recorded_gates) == len(gates), "gate count mismatch")
        for old, new in zip(recorded_gates, gates, strict=True):
            for field in new:
                require(
                    floats_equal(old[field], new[field]),
                    f"gate mismatch: {new['gate']}/{field}",
                )

    selected = str(selection_recorded["selected_adaptive_policy"])
    selected_negative = [
        row for row in negative_rows if row["policy"] == selected
    ]
    negative_recorded = read_json(
        negative_dir / "NEGATIVE_CONTROL_RESULT.json"
    )
    require(
        sum(
            int(row["candidate_declared_envelope_invalid_looks"])
            for row in selected_negative
        )
        == int(negative_recorded["candidate_declared_envelope_invalid_looks"]),
        "negative declared-envelope count mismatch",
    )
    require(
        sum(
            int(row["candidate_realized_chain_failure_looks"])
            for row in selected_negative
        )
        == int(negative_recorded["candidate_realized_chain_failure_looks"]),
        "negative realized-failure count mismatch",
    )
    return {
        "deep_status": "PASS",
        "tuning_summary_recomputed": True,
        "selection_recomputed": True,
        "test_summary_recomputed": True,
        "primary_gates_recomputed": True,
        "row_order_invariance_recomputed": True,
        "negative_control_recomputed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor-root", type=Path)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = verify_basic(root, args.predecessor_root)
    if args.deep:
        report.update(verify_deep(root))
    report["verifier_status"] = "PASS"
    if args.report:
        study.write_json(args.report, report)
    print("PASS_V8_7_TARGET_USE_VERIFIER")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
