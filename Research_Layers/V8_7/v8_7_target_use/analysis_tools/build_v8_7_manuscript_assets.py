#!/usr/bin/env python3
"""Generate V8.7 manuscript table rows from locked tuning and test outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


DISPLAY = {
    "posterior_mean": "Posterior mean",
    "posterior_mean+screen": "Posterior mean + screen",
    "posterior_conservative": "Posterior 0.95 quantile",
    "posterior_conservative+screen": "Posterior 0.95 quantile + screen",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(value: str | float, digits: int = 3) -> str:
    return f"{100 * float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    table_dir = args.table_dir.resolve()

    tune_summary_path = root / "results" / "tune" / "adaptive_summary.csv"
    selection_path = root / "results" / "tune" / "SELECTION.json"
    test_summary_path = root / "results" / "test" / "adaptive_summary.csv"
    gate_path = root / "results" / "test" / "PRIMARY_GATE_TABLE.csv"
    periodic_path = root / "results" / "test" / "periodic_frontier.csv"
    primary_path = root / "results" / "test" / "PRIMARY_RESULT.json"
    negative_path = root / "results" / "negative_control" / "NEGATIVE_CONTROL_RESULT.json"

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    negative = json.loads(negative_path.read_text(encoding="utf-8"))
    if selection["selected_adaptive_policy"] != "posterior_mean+screen":
        raise RuntimeError("unexpected tuning selection")
    if primary["all_primary_gates_pass"] is not True:
        raise RuntimeError("primary gate conjunction did not pass")
    if negative["negative_control_detected"] is not True:
        raise RuntimeError("negative control was not detected")

    tune_by_policy = {row["policy"]: row for row in read_csv(tune_summary_path)}
    tuning_lines: list[str] = []
    for policy in ("posterior_mean+screen", "posterior_conservative+screen"):
        row = tune_by_policy[policy]
        risk_upper = json.loads(row["risk_upper_json"])
        status = "Selected" if policy == selection["selected_adaptive_policy"] else "Qualified"
        tuning_lines.append(
            "{} & {:.3f} & {:.2f} & {:.2f} & {:.4f} / {:.4f} / {:.4f} & {} \\\\".format(
                DISPLAY[policy],
                float(row["mean_refreshes"]),
                100 * float(row["eligible_passage_lower"]),
                100 * float(row["overall_authorization_lower"]),
                100 * float(risk_upper["h0020"]),
                100 * float(risk_upper["h0060"]),
                100 * float(risk_upper["h0120"]),
                status,
            )
        )
    tuning_output = table_dir / "si_v87_tuning_rows.tex"
    write_lines(tuning_output, tuning_lines)

    test_summary_rows = read_csv(test_summary_path)
    test_by_policy = {row["policy"]: row for row in test_summary_rows}
    selected_test = test_by_policy[primary["selected_adaptive_policy"]]
    test_lines: list[str] = []
    for row in test_summary_rows:
        policy = row["policy"]
        risk_events = json.loads(row["risk_events_json"])
        role = "Confirmatory" if policy == primary["selected_adaptive_policy"] else "Descriptive"
        eligible = "--" if int(row["screened"]) == 0 else f"{100 * float(row['eligible_passage_rate']):.2f}"
        test_lines.append(
            "{} & {} & {:.3f} & {:.2f} & {} & {:d} / {:d} / {:d} \\\\".format(
                DISPLAY[policy],
                role,
                float(row["mean_refreshes"]),
                100 * float(row["overall_authorization_rate"]),
                eligible,
                int(risk_events["h0020"]),
                int(risk_events["h0060"]),
                int(risk_events["h0120"]),
            )
        )
    test_output = table_dir / "si_v87_test_policy_rows.tex"
    write_lines(test_output, test_lines)

    gates = {row["gate"]: row for row in read_csv(gate_path)}
    selected_risk_rates = json.loads(selected_test["risk_rates_json"])
    gate_order = [
        ("risk_h0020", "Risk, hazard 0.002", r"\%"),
        ("risk_h0060", "Risk, hazard 0.006", r"\%"),
        ("risk_h0120", "Risk, hazard 0.012", r"\%"),
        ("eligible_passage", "Eligible passage", r"\%"),
        ("overall_authorization", "Overall authorization", r"\%"),
        ("refresh_gain", "Refresh difference", "refreshes"),
    ]
    gate_lines: list[str] = []
    for gate_id, label, unit in gate_order:
        row = gates[gate_id]
        value = float(row["estimate_or_bound"])
        threshold = float(row["threshold"])
        point = row["point_estimate"]
        if gate_id == "risk_h0020":
            point = selected_risk_rates["h0020"]
        elif gate_id == "risk_h0060":
            point = selected_risk_rates["h0060"]
        elif gate_id == "risk_h0120":
            point = selected_risk_rates["h0120"]
        elif gate_id == "eligible_passage":
            point = selected_test["eligible_passage_rate"]
        elif gate_id == "overall_authorization":
            point = selected_test["overall_authorization_rate"]
        if unit == r"\%":
            value_text = f"{100 * value:.4f}"
            point_text = "--" if point in (None, "") else f"{100 * float(point):.4f}"
            threshold_text = f"{row['operator']} {100 * threshold:.2f}"
        else:
            value_text = f"{value:.5f}"
            point_text = "--" if point in (None, "") else f"{float(point):.5f}"
            threshold_text = f"{row['operator']} {threshold:.1f}"
        gate_lines.append(
            f"{label} & {point_text} & {value_text} & {threshold_text} & {row['method']} & Yes \\\\"
        )
    gate_output = table_dir / "si_v87_primary_gate_rows.tex"
    write_lines(gate_output, gate_lines)

    periodic_groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(periodic_path):
        periodic_groups[int(row["periodic_interval_years"])].append(row)
    periodic_lines: list[str] = []
    for interval in sorted(periodic_groups):
        rows = periodic_groups[interval]
        if len(rows) != 3:
            raise RuntimeError(f"periodic interval {interval} does not have three hazard rows")
        first = rows[0]
        comparable = [
            "mean_refreshes_equal_weight_domain",
            "authorization_rate_equal_weight_domain",
            "risk_pass_all_strata",
            "joint_nonvacuity_pass",
        ]
        for field in comparable:
            if len({row[field] for row in rows}) != 1:
                raise RuntimeError(f"periodic aggregate mismatch: {interval}/{field}")
        periodic_lines.append(
            "{:d} & {:.1f} & {:.2f} & {} & {} \\\\".format(
                interval,
                float(first["mean_refreshes_equal_weight_domain"]),
                100 * float(first["authorization_rate_equal_weight_domain"]),
                "Yes" if int(first["risk_pass_all_strata"]) else "No",
                "Yes" if int(first["joint_nonvacuity_pass"]) else "No",
            )
        )
    periodic_output = table_dir / "si_v87_periodic_rows.tex"
    write_lines(periodic_output, periodic_lines)

    inputs = [
        tune_summary_path,
        selection_path,
        test_summary_path,
        gate_path,
        periodic_path,
        primary_path,
        negative_path,
    ]
    outputs = [tuning_output, test_output, gate_output, periodic_output]
    provenance = {
        "asset": "V8.7 manuscript table rows",
        "generation": "deterministic locked-output aggregation; no simulation",
        "input_sha256": {str(path.relative_to(root)): sha256_file(path) for path in inputs},
        "output_sha256": {str(path.relative_to(root)): sha256_file(path) for path in outputs},
        "selected_policy": selection["selected_adaptive_policy"],
        "primary_status": primary["status"],
        "negative_control_detected": negative["negative_control_detected"],
        "physical_qualification": "NOT_RUN",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS_V8_7_MANUSCRIPT_ASSET_BUILD")
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
