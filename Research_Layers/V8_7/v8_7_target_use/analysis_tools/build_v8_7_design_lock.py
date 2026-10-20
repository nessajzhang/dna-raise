#!/usr/bin/env python3
"""Create the immutable V8.7 evidence and design hash locks exactly once."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
DESIGN = ROOT / "design"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def clean_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".git" not in path.parts
        and path.name != ".DS_Store"
    )


def write_lock(path: Path, files: list[Path]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing lock: {path}")
    lines = []
    for target in sorted({item.resolve() for item in files}):
        if not target.is_file():
            raise FileNotFoundError(target)
        relative = os.path.relpath(target, path.parent.resolve())
        lines.append(f"{sha256_file(target)}  {relative}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def verify_lock(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split(None, 1)
        target = (path.parent / relative.strip().lstrip("*")).resolve()
        if sha256_file(target) != expected.upper():
            raise AssertionError(f"lock verification failed: {target}")
        count += 1
    return count


def main() -> None:
    evidence_files: list[Path] = [
        EVIDENCE / "TERMINOLOGY_LEDGER.md",
        EVIDENCE
        / "external_code"
        / "conformal-policy-control_9799c77f.tar.gz",
        EVIDENCE / "external_code" / "ipomdp_shielding_artifact_v1.zip",
    ]
    evidence_files.extend(clean_files(EVIDENCE / "literature"))
    evidence_files.extend(clean_files(EVIDENCE / "paper_cards"))
    evidence_lock = EVIDENCE / "EVIDENCE_SHA256.tsv"
    write_lock(evidence_lock, evidence_files)

    design_files: list[Path] = [
        ROOT / "README_V8_7.md",
        evidence_lock,
    ]
    design_files.extend(
        path
        for path in clean_files(DESIGN)
        if path.name != "DESIGN_LOCK.sha256"
    )
    design_files.extend(
        path
        for path in clean_files(ROOT / "analysis_tools")
        if path.suffix in {".py", ".sh"}
    )
    design_files.extend(clean_files(ROOT / "baseline_sources"))
    design_files.extend(clean_files(ROOT / "qa"))
    design_files.extend(clean_files(ROOT / "runtime_wheels_cp312_linux"))
    design_files.extend(evidence_files)
    design_lock = DESIGN / "DESIGN_LOCK.sha256"
    write_lock(design_lock, design_files)

    evidence_count = verify_lock(evidence_lock)
    design_count = verify_lock(design_lock)
    print(
        "PASS_V8_7_DESIGN_LOCK "
        f"evidence_files={evidence_count} design_files={design_count} "
        f"evidence_lock_sha256={sha256_file(evidence_lock)} "
        f"design_lock_sha256={sha256_file(design_lock)}"
    )


if __name__ == "__main__":
    main()
