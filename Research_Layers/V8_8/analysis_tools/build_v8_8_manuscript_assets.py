#!/usr/bin/env python3
"""Build V8.8 manuscript tables from locked summaries without simulation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


STUDY_ID = "paper4_nc_v8_8_independent_structural_challenge_20260905"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def write_if_changed(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def copy_if_changed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha256(src) == sha256(dst):
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    test_dir = args.result_root / "test_locked_v1_finalized_hotfix1"
    control_dir = args.result_root / "control_locked_v1"
    paths = {
        "profile_summary": test_dir / "profile_summary.csv",
        "comparisons": test_dir / "profile_vs_equal_comparisons.csv",
        "gates": test_dir / "PRIMARY_GATE_TABLE.csv",
        "primary": test_dir / "PRIMARY_RESULT.json",
        "control": control_dir / "OFFLINE_CONTROL_RESULT.json",
        "conformance": args.result_root / "V8_8_RESULTS_CONFORMANCE.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing locked {label}: {path}")

    primary = json.loads(paths["primary"].read_text(encoding="utf-8"))
    control = json.loads(paths["control"].read_text(encoding="utf-8"))
    conformance = json.loads(paths["conformance"].read_text(encoding="utf-8"))
    if primary["study_id"] != STUDY_ID or primary["status"] != "PRIMARY_GATES_FAIL":
        raise RuntimeError("unexpected primary result")
    if primary["selected_gamma"] != 0.5 or primary["reference_gamma"] != 0.0:
        raise RuntimeError("unexpected locked profile selection")
    if abs(primary["refresh_difference_point_estimate"] - 0.0111) > 1e-12:
        raise RuntimeError("unexpected refresh point estimate")
    if abs(primary["refresh_difference_upper"] - 0.01335) > 1e-12:
        raise RuntimeError("unexpected refresh upper bound")
    if control["status"] != "OFFLINE_CONTROL_EXPOSED_MISSPECIFICATION":
        raise RuntimeError("unexpected offline-control status")
    if conformance["status"] != "PASS_V8_8_STRUCTURAL_CHALLENGE_CONFORMANCE":
        raise RuntimeError("V8.8 result conformance is not PASS")
    if conformance["frozen_v8_7_episode_records_changed"]:
        raise RuntimeError("conformance reports changed V8.7 episode records")

    with paths["profile_summary"].open(newline="", encoding="utf-8") as fh:
        profiles = list(csv.DictReader(fh))
    with paths["comparisons"].open(newline="", encoding="utf-8") as fh:
        comparisons = list(csv.DictReader(fh))
    with paths["gates"].open(newline="", encoding="utf-8") as fh:
        gates = list(csv.DictReader(fh))
    if len(profiles) != 7 or len(comparisons) != 6 or len(gates) != 6:
        raise RuntimeError("unexpected locked table dimensions")
    if sum(int(row["pass"]) for row in gates) != 5:
        raise RuntimeError("expected five passing gates and one failing gate")

    comp_by_profile = {row["candidate_profile"]: row for row in comparisons}
    profile_rows = []
    for row in sorted(profiles, key=lambda item: float(item["gamma"])):
        gamma = float(row["gamma"])
        mean_refresh = float(row["mean_refreshes_equal_hazard"])
        risk_upper = max(json.loads(row["risk_upper_json"]).values()) * 100
        passage_lower = float(row["eligible_passage_lower"]) * 100
        auth_lower = float(row["authorization_lower"]) * 100
        profile = row["profile"]
        if gamma == 0:
            contrast = "--"
        else:
            comp = comp_by_profile[profile]
            diff = float(comp["mean_refresh_difference_candidate_minus_equal"])
            lo = float(comp["two_sided_95_lower"])
            hi = float(comp["two_sided_95_upper"])
            contrast = f"${signed(diff)}$ [{lo:.4f}, {hi:.4f}]"
        gamma_text = f"\\mathbf{{{gamma:g}}}" if gamma == 0.5 else f"{gamma:g}"
        profile_rows.append(
            f"${gamma_text}$ & {mean_refresh:.4f} & {contrast} & "
            f"{risk_upper:.4f} & {passage_lower:.2f} & {auth_lower:.2f} \\\\"
        )

    gate_labels = {
        "risk_h0020": "Mission risk, $h=0.002$",
        "risk_h0060": "Mission risk, $h=0.006$",
        "risk_h0120": "Mission risk, $h=0.012$",
        "eligible_passage": "Eligible passage",
        "overall_authorization": "Overall authorization",
        "refresh_gain_vs_equal_spending": "Refresh difference",
    }
    gate_rows = []
    for row in gates:
        gate = row["gate"]
        bound = float(row["estimate_or_bound"])
        point = row["point_estimate"]
        if gate.startswith("risk_"):
            point_text = "0.0000\\%" if gate != "risk_h0060" else "0.0050\\%"
            bound_text = f"{bound * 100:.4f}\\%"
            criterion = "$\\leq 5\\%$"
        elif gate in {"eligible_passage", "overall_authorization"}:
            matching = next(p for p in profiles if float(p["gamma"]) == 0.5)
            key = "eligible_passage_rate" if gate == "eligible_passage" else "authorization_rate"
            point_text = f"{float(matching[key]) * 100:.4f}\\%"
            bound_text = f"{bound * 100:.4f}\\%"
            criterion = "$\\geq 25\\%$" if gate == "eligible_passage" else "$\\geq 5\\%$"
        else:
            point_text = signed(float(point), 5)
            bound_text = signed(bound, 5)
            criterion = "$<0$"
        verdict = "Pass" if int(row["pass"]) else "\\textbf{Fail}"
        gate_rows.append(
            f"{gate_labels[gate]} & {point_text} & {bound_text} & "
            f"{criterion} & {verdict} \\\\"
        )

    generated = args.source_dir / "generated_tables"
    source_data = args.source_dir / "source_data"
    write_if_changed(generated / "si_v88_profile_rows.tex", "\n".join(profile_rows) + "\n")
    write_if_changed(generated / "si_v88_gate_rows.tex", "\n".join(gate_rows) + "\n")

    copy_map = {
        paths["profile_summary"]: source_data / "Supplementary_Table_V8_8_profile_frontier.csv",
        paths["comparisons"]: source_data / "Supplementary_Table_V8_8_profile_contrasts.csv",
        paths["gates"]: source_data / "Supplementary_Table_V8_8_primary_gates.csv",
        paths["primary"]: source_data / "Supplementary_V8_8_primary_result.json",
        paths["control"]: source_data / "Supplementary_V8_8_offline_control.json",
    }
    for src, dst in copy_map.items():
        copy_if_changed(src, dst)

    outputs = [
        generated / "si_v88_profile_rows.tex",
        generated / "si_v88_gate_rows.tex",
        *copy_map.values(),
    ]
    receipt = {
        "study_id": STUDY_ID,
        "status": "PASS_V8_8_MANUSCRIPT_ASSETS",
        "simulation_called": False,
        "source": "locked summary, gate, primary-result, and offline-control files only",
        "primary_status": primary["status"],
        "selected_gamma": primary["selected_gamma"],
        "input_sha256": {str(path): sha256(path) for path in paths.values()},
        "output_sha256": {str(path): sha256(path) for path in outputs},
    }
    write_if_changed(args.receipt, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
