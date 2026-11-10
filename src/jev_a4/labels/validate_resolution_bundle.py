#!/usr/bin/env python3
"""Independent consistency check for a response-resolution bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    bundle = args.bundle.resolve()
    report = json.loads((bundle / "report.json").read_text(encoding="utf-8"))
    sealed = json.loads((bundle / "artifact_sha256.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    checks = 0

    for relative, identity in sealed.items():
        path = bundle / relative
        if not path.resolve().is_relative_to(bundle):
            failures.append(f"unsafe sealed artifact path: {relative}")
            continue
        checks += 1
        if not path.is_file():
            failures.append(f"missing sealed artifact: {relative}")
            continue
        if path.stat().st_size != identity["size_bytes"]:
            failures.append(f"size drift: {relative}")
        if sha256(path) != identity["sha256"]:
            failures.append(f"hash drift: {relative}")

    summaries: dict[str, object] = {}
    for dataset in sorted(report["datasets"]):
        if Path(dataset).name != dataset or dataset in (".", ".."):
            failures.append("unsafe dataset name")
            continue
        src = source / dataset
        dst = bundle / dataset
        source_files = {"manifest": "manifest.json", "labels": "labels.npy",
                        "reason_codes": "reason_codes.npy", "effect_log2fc": "effect_log2fc.npy",
                        "q_best": "q_best.npy"}
        for key, filename in source_files.items():
            identity = report["datasets"][dataset]["source"][key]
            source_file = src / filename
            checks += 1
            if (not source_file.is_file() or source_file.stat().st_size != identity["size_bytes"]
                    or sha256(source_file) != identity["sha256"]):
                failures.append(f"source content drift: {dataset}/{filename}")
        labels = np.load(src / "labels.npy", mmap_mode="r", allow_pickle=False)
        reasons = np.load(src / "reason_codes.npy", mmap_mode="r", allow_pickle=False)
        effects = np.load(src / "effect_log2fc.npy", mmap_mode="r", allow_pickle=False)
        hard = np.load(dst / "hard_label.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(dst / "supervision_mask.npy", mmap_mode="r", allow_pickle=False)
        resolution = np.load(dst / "resolution_code.npy", mmap_mode="r", allow_pickle=False)
        point = np.load(dst / "point_effect_region.npy", mmap_mode="r", allow_pickle=False)
        checks += 10
        array_manifest = report["datasets"][dataset]["arrays"]
        for array_name, identity in array_manifest.items():
            if identity["path"] != array_name:
                failures.append(f"non-relative array path: {dataset}/{array_name}")
        if not (labels.shape == reasons.shape == effects.shape == hard.shape == mask.shape == resolution.shape == point.shape):
            failures.append(f"shape mismatch: {dataset}")
            continue
        expected_mask = labels >= 0
        if not np.array_equal(mask, expected_mask):
            failures.append(f"supervision mask mismatch: {dataset}")
        if not np.array_equal(hard[expected_mask], labels[expected_mask]):
            failures.append(f"resolved hard-label mismatch: {dataset}")
        if np.any(hard[~expected_mask] != -100):
            failures.append(f"unresolved row is trainable: {dataset}")
        if np.any((labels == 1) != (resolution == 1)):
            failures.append(f"Same/resolution mismatch: {dataset}")
        if np.any((labels == -1) != (resolution >= 3)):
            failures.append(f"unresolved/resolution mismatch: {dataset}")
        expected_resolution = np.empty(labels.shape, dtype=np.uint8)
        expected_resolution[expected_mask] = labels[expected_mask]
        for reason in range(1, 6):
            expected_resolution[reasons == reason] = reason + 2
        if not np.array_equal(resolution, expected_resolution):
            failures.append(f"reason mapping mismatch: {dataset}")
        ratio = float(report["datasets"][dataset]["statistical_contract"]["ratio_margin"])
        margin = float(np.log2(ratio))
        expected_point = np.full(labels.shape, -1, dtype=np.int8)
        finite = np.isfinite(effects)
        expected_point[finite] = 1
        expected_point[finite & (effects > margin)] = 0
        expected_point[finite & (effects < -margin)] = 2
        if not np.array_equal(point, expected_point):
            failures.append(f"point-region mismatch: {dataset}")
        counts = {str(code): int(np.count_nonzero(labels == code)) for code in (-1, 0, 1, 2)}
        summaries[dataset] = {
            "shape": [int(x) for x in labels.shape],
            "source_label_counts": counts,
            "unresolved_never_same": bool(not np.any((labels == -1) & (resolution == 1))),
            "unresolved_never_trainable": bool(np.all(hard[labels == -1] == -100)),
        }

    result = {
        "schema": "decider.response_resolution_validation.v3",
        "status": "PASS" if not failures else "FAIL",
        "validated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(source),
        "bundle": str(bundle),
        "validator_path": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__)),
        "checks": checks,
        "failures": failures,
        "datasets": summaries,
    }
    save(bundle / "final_validation.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "failures": failures}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
