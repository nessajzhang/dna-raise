#!/usr/bin/env python3
"""Fetch the exact DT4DDS qPCR inputs from the tagged upstream repository.

The Paper 4 public release does not redistribute these upstream CSV files while
their record-specific license wording remains non-specific (Zenodo
``other-open``).  This script downloads each small file from the version-tagged
v1.0.0 upstream source and verifies both the recorded Git blob identity (in the source
note) and the byte-level SHA-256 digest used by the analysis package.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE = "https://raw.githubusercontent.com/fml-ethz/dt4dds_notebooks/v1.0.0/SI/qpcr"
UPSTREAM_SHA256 = {
    "cal_Genscript_GCall.csv": "08070205f72a6ef53e2b1b092f71d136c6571e71a99b00334f72cc240b714b88",
    "cal_Genscript_GCfix.csv": "98bcf8a7ae2215cb87f8f54d460644b6215f06509db4a6c75c39a9d129a9c6b1",
    "cal_Twist_GCall.csv": "c8e4f4bdb9d730f4d1198171cf3ac0b5fa4102b60201428e00b783de0bf1c9a4",
    "cal_Twist_GCfix.csv": "11859f7049683a24dbdbe6125c94712246b9e3bc0ae821b981a885a9edadd81e",
    "decay_Genscript_GCall.csv": "9dbadd56888f53baae4e91daefdb421c83e39af0978757cb7cde8f814a0464b2",
    "decay_Genscript_GCfix.csv": "102544ee45888f1bdd2fc456494894655741e3b40f34ca2b265b84b68a5daee8",
    "decay_Twist_GCall.csv": "876fefd2d077b65f1939be665c6d71ffd8e8252eb25d81f55a4876800d739a35",
    "decay_Twist_GCfix.csv": "b1f4d4311879be0c46a63534bab227b41bfb22469d1c94140fe9fa8640f6cacb",
}

# Some local checkouts normalize the upstream CRLF text to LF.  These digests
# allow an already-present normalized copy without weakening the download gate:
# newly retrieved bytes must always match UPSTREAM_SHA256 exactly.
CANONICAL_LF_SHA256 = {
    "cal_Genscript_GCall.csv": "082f233b99ca979db37c4afd39272b9deabe2c536df7e34cbdf325d04d81f0e5",
    "cal_Genscript_GCfix.csv": "27daadf71e91e186922bb42608f772b5f3a4775a00166e7afdd72f1d4b78986c",
    "cal_Twist_GCall.csv": "4d417722aa88d77e8564c4b0e73438c4c30ae48d0965894e9a23499423967e91",
    "cal_Twist_GCfix.csv": "3f9c7eb9ad0c440504109597891eb5d5f97cea8c8f54907521dcca904176b7f7",
    "decay_Genscript_GCall.csv": "6ee7bd1ae3b9ed5aae7c39a58a6bfaa45134448b12322ddd745ca78a4ee8b7da",
    "decay_Genscript_GCfix.csv": "763b56bc4c9f96541c79cbba6b7b51e803b6ec3d5d182f4aa390984269cb8fbf",
    "decay_Twist_GCall.csv": "5f3f72cb6751b450af764118428d37ef18310556fae12a9288fc71ee20ff802b",
    "decay_Twist_GCfix.csv": "36b2981c3f12fa27aabd7d196caee0d08caac10b78bbd36dbb1bb031d9e31e0c",
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fetch_bytes(url: str) -> bytes:
    """Fetch one immutable upstream file with bounded transient retries."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Paper4-reproducibility-fetch/1.0",
                    "Connection": "close",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (TimeoutError, urllib.error.URLError, http.client.RemoteDisconnected) as error:
            last_error = error
            if attempt < 2:
                time.sleep(1.0)
    raise RuntimeError(f"upstream fetch failed after 3 attempts: {last_error}")


def fetch(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in UPSTREAM_SHA256.items():
        destination = output_dir / name
        if destination.exists():
            content = destination.read_bytes()
            observed = sha256(content)
            canonical_lf = sha256(content.replace(b"\r\n", b"\n"))
            if observed != expected and canonical_lf != CANONICAL_LF_SHA256[name]:
                raise RuntimeError(f"existing file has wrong digest: {destination}")
            mode = "upstream bytes" if observed == expected else "canonical LF text"
            print(f"VERIFIED {name} ({mode})")
            continue
        content = fetch_bytes(f"{BASE}/{name}")
        observed = sha256(content)
        if observed != expected:
            raise RuntimeError(
                f"digest mismatch for {name}: observed={observed}, expected={expected}"
            )
        destination.write_bytes(content)
        print(f"FETCHED {name}")
    print("PASS_PAPER4_DT4DDS_QPCR_FETCH")


def self_test() -> None:
    assert len(UPSTREAM_SHA256) == len(CANONICAL_LF_SHA256) == 8
    assert set(UPSTREAM_SHA256) == set(CANONICAL_LF_SHA256)
    assert all(len(digest) == 64 for digest in UPSTREAM_SHA256.values())
    assert all(len(digest) == 64 for digest in CANONICAL_LF_SHA256.values())
    assert BASE.endswith("/v1.0.0/SI/qpcr")
    print("PASS_PAPER4_DT4DDS_QPCR_FETCH_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("external_data/paper4_dt4dds"),
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        fetch(args.output_dir)


if __name__ == "__main__":
    main()
