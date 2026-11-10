#!/usr/bin/env python3
"""Build an explicit resolved-vs-unresolved label bundle.

This stage does not relabel statistical non-significance as biological
equivalence.  It preserves the existing audited strict calls and converts all
unresolved rows to the conventional ignore label -100.  A separate resolution
code records why each unresolved row could not be assigned Up/Same/Down.

The output is deliberately label-only.  It references, but never modifies, the
factorized expression/preprocessing store supplied with ``--source``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SOURCE_CLASS_NAMES = {-1: "unresolved", 0: "up", 1: "same", 2: "down"}
IGNORE_INDEX = -100
HARD_CLASS_NAMES = {0: "up", 1: "same", 2: "down"}
REASON_NAMES = {
    0: "resolved_strict_margin_call",
    1: "unresolved_insufficient_cells",
    2: "unresolved_low_effective_sample_size",
    3: "unresolved_zero_or_nonestimable_mean",
    4: "unresolved_degenerate_variance",
    5: "unresolved_insufficient_margin_evidence",
}
RESOLUTION_CODES = {
    0: "resolved_up",
    1: "resolved_same",
    2: "resolved_down",
    3: "unresolved_insufficient_cells",
    4: "unresolved_low_effective_sample_size",
    5: "unresolved_zero_or_nonestimable_mean",
    6: "unresolved_degenerate_variance",
    7: "unresolved_insufficient_margin_evidence",
}
POINT_REGION_CODES = {
    -1: "nonestimable",
    0: "point_up",
    1: "point_inside_equivalence_band",
    2: "point_down",
}


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
    }
    if include_sha256:
        result["sha256"] = sha256(path)
    return result


def artifact_identity(path: Path, relative_path: str) -> dict[str, Any]:
    """Identity for a movable bundle artifact; never record a staging path."""
    result = file_identity(path)
    result["path"] = relative_path
    return result


def bincount_dict(values: np.ndarray, codes: Iterable[int], names: dict[int, str]) -> dict[str, int]:
    return {names[code]: int(np.count_nonzero(values == code)) for code in codes}


def map_arrays(
    labels: np.ndarray,
    reasons: np.ndarray,
    effects: np.ndarray,
    *,
    log2_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map source arrays to safe explicit arrays; used by build and tests."""
    labels = np.asarray(labels)
    reasons = np.asarray(reasons)
    effects = np.asarray(effects)
    if labels.shape != reasons.shape or labels.shape != effects.shape:
        raise ValueError("labels, reasons, and effects must have identical shapes")
    if not labels.size or not np.isfinite(log2_margin) or log2_margin <= 0:
        raise ValueError("nonempty arrays and a positive finite margin are required")
    if not np.isin(labels, [-1, 0, 1, 2]).all():
        raise ValueError("unexpected source label code")
    if not np.isin(reasons, list(REASON_NAMES)).all():
        raise ValueError("unexpected source reason code")

    resolved = labels >= 0
    if np.any(resolved != (reasons == 0)):
        raise ValueError("source contract violation: resolved iff reason_code == 0")

    hard = np.full(labels.shape, IGNORE_INDEX, dtype=np.int8)
    hard[resolved] = labels[resolved].astype(np.int8, copy=False)
    supervision = resolved.astype(np.bool_, copy=False)

    resolution = np.empty(labels.shape, dtype=np.uint8)
    resolution[resolved] = labels[resolved].astype(np.uint8, copy=False)
    for reason in range(1, 6):
        resolution[reasons == reason] = np.uint8(reason + 2)

    point_region = np.full(labels.shape, -1, dtype=np.int8)
    finite = np.isfinite(effects)
    point_region[finite] = 1
    point_region[finite & (effects > log2_margin)] = 0
    point_region[finite & (effects < -log2_margin)] = 2

    if not np.array_equal(hard[supervision], labels[supervision]):
        raise AssertionError("resolved label drift")
    if np.any(hard[~supervision] != IGNORE_INDEX):
        raise AssertionError("unresolved row received a trainable hard label")
    if np.any(point_region[supervision] != labels[supervision]):
        raise ValueError("resolved labels disagree with their descriptive effect regions")
    return hard, supervision, resolution, point_region


def write_memmap(path: Path, values: np.ndarray, dtype: np.dtype[Any]) -> None:
    target = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=values.shape)
    target[:] = values
    target.flush()
    del target


def validate_source_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "decider.perturbation_classification.factorized.v1":
        raise ValueError(f"unsupported source schema in {dataset_dir}")
    for name in ("labels.npy", "reason_codes.npy", "effect_log2fc.npy", "q_best.npy"):
        if not (dataset_dir / name).is_file():
            raise FileNotFoundError(dataset_dir / name)


def process_dataset(source_dir: Path, destination_dir: Path) -> dict[str, Any]:
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    validate_source_manifest(source_dir, source_manifest)

    labels_path = source_dir / "labels.npy"
    reasons_path = source_dir / "reason_codes.npy"
    effects_path = source_dir / "effect_log2fc.npy"
    q_best_path = source_dir / "q_best.npy"
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if labels.ndim != 2:
        raise ValueError("labels must be a target-by-gene matrix")
    reasons = np.load(reasons_path, mmap_mode="r", allow_pickle=False)
    effects = np.load(effects_path, mmap_mode="r", allow_pickle=False)
    q_best = np.load(q_best_path, mmap_mode="r", allow_pickle=False)
    if labels.shape != q_best.shape:
        raise ValueError(f"q_best shape mismatch in {source_dir.name}")

    ratio = float(source_manifest["label_definition"]["ratio"])
    alpha = float(source_manifest["label_definition"]["alpha"])
    if not np.isfinite(ratio) or ratio <= 1 or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("invalid ratio margin or significance threshold")
    if not np.isfinite(q_best).all() or np.any((q_best < 0) | (q_best > 1)):
        raise ValueError("q_best must contain finite probabilities in [0, 1]")
    log2_margin = float(np.log2(ratio))
    hard, supervision, resolution, point_region = map_arrays(
        labels, reasons, effects, log2_margin=log2_margin
    )

    resolved = supervision
    unresolved = ~supervision
    if np.any(q_best[resolved] > alpha + 1e-6):
        raise ValueError(f"resolved row exceeds alpha in {source_dir.name}")
    if np.any(q_best[unresolved & (reasons == 5)] <= alpha):
        raise ValueError(f"margin-unresolved row has a qualifying q value in {source_dir.name}")

    expected_counts = source_manifest.get("label_counts", {})
    actual_counts = {str(code): int(np.count_nonzero(labels == code)) for code in (-1, 0, 1, 2)}
    if expected_counts and expected_counts != actual_counts:
        raise ValueError(f"source manifest label-count mismatch in {source_dir.name}")

    destination_dir.mkdir(parents=True)
    write_memmap(destination_dir / "hard_label.npy", hard, np.int8)
    write_memmap(destination_dir / "supervision_mask.npy", supervision, np.bool_)
    write_memmap(destination_dir / "resolution_code.npy", resolution, np.uint8)
    write_memmap(destination_dir / "point_effect_region.npy", point_region, np.int8)

    reason_counts = bincount_dict(reasons[unresolved], range(1, 6), REASON_NAMES)
    unresolved_point = bincount_dict(
        point_region[unresolved], (-1, 0, 1, 2), POINT_REGION_CODES
    )
    reason_by_point: dict[str, dict[str, int]] = {}
    for reason in range(1, 6):
        mask = unresolved & (reasons == reason)
        reason_by_point[REASON_NAMES[reason]] = bincount_dict(
            point_region[mask], (-1, 0, 1, 2), POINT_REGION_CODES
        )

    total = int(labels.size)
    resolved_n = int(np.count_nonzero(resolved))
    summary = {
        "dataset": source_dir.name,
        "shape": [int(x) for x in labels.shape],
        "total_pairs": total,
        "strict_class_counts": {
            HARD_CLASS_NAMES[code]: int(np.count_nonzero(labels == code)) for code in range(3)
        },
        "resolved_pairs": resolved_n,
        "resolved_fraction": resolved_n / total,
        "unresolved_pairs": total - resolved_n,
        "unresolved_fraction": (total - resolved_n) / total,
        "unresolved_reason_counts": reason_counts,
        "unresolved_point_effect_regions_descriptive_only": unresolved_point,
        "unresolved_reason_by_point_effect_region_descriptive_only": reason_by_point,
    }

    manifest = {
        "schema": "decider.response_resolution_labels.v3",
        "dataset": source_dir.name,
        "shape": summary["shape"],
        "class_contract": {
            "hard_label": {"up": 0, "same": 1, "down": 2, "unresolved_ignore_index": IGNORE_INDEX},
            "resolution_codes": {str(k): v for k, v in RESOLUTION_CODES.items()},
            "point_effect_region_codes": {str(k): v for k, v in POINT_REGION_CODES.items()},
            "supervision_rule": "hard_label contributes to classification loss iff supervision_mask is true",
            "unresolved_rule": "unresolved is label missingness, never a fourth deployment class and never Same",
            "point_effect_region_rule": "descriptive point estimate only; must not be used as a hard response label",
        },
        "statistical_contract": {
            "ratio_margin": ratio,
            "log2_margin": log2_margin,
            "alpha": alpha,
            "strict_calls_preserved_without_relabeling": True,
            "new_inference_performed": False,
            "cell_pool_approximation_inherited": True,
            "biological_replicate_inference": False,
        },
        "source": {
            "manifest": file_identity(source_manifest_path),
            "labels": file_identity(labels_path),
            "reason_codes": file_identity(reasons_path),
            "effect_log2fc": file_identity(effects_path),
            "q_best": file_identity(q_best_path),
        },
        "arrays": {
            name: artifact_identity(destination_dir / name, name)
            for name in (
                "hard_label.npy",
                "supervision_mask.npy",
                "resolution_code.npy",
                "point_effect_region.npy",
            )
        },
        "summary": summary,
    }
    json_dump(destination_dir / "manifest.json", manifest)
    return manifest


def markdown_report(bundle: dict[str, Any]) -> str:
    lines = [
        "# DECIDER resolved / unresolved label audit",
        "",
        f"Status: **{bundle['status']}**",
        "",
        "This bundle preserves the existing strict Up/Same/Down calls and makes unresolved rows impossible to confuse with Same. No model training was started.",
        "",
        "## Counts",
        "",
        "| Dataset | Total | Strict Up | Strict Same | Strict Down | Unresolved | Resolved % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bundle["datasets"].values():
        summary = item["summary"]
        classes = summary["strict_class_counts"]
        lines.append(
            f"| {summary['dataset']} | {summary['total_pairs']:,} | {classes['up']:,} | "
            f"{classes['same']:,} | {classes['down']:,} | {summary['unresolved_pairs']:,} | "
            f"{100.0 * summary['resolved_fraction']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Safety contract",
            "",
            "- `hard_label.npy`: Up=0, Same=1, Down=2, unresolved=`-100`.",
            "- `supervision_mask.npy`: only `true` rows may enter the hard classification loss.",
            "- `resolution_code.npy`: records the exact resolved class or unresolved reason.",
            "- `point_effect_region.npy`: descriptive point-estimate region only; it is not a training truth.",
            "- Statistical non-significance was never converted to Same.",
            "- No old predictor, source mean prediction, residual correction, or model output was used.",
            "",
            "## Important limitation",
            "",
            "The source labels use pooled-cell Welch/TOST approximations rather than independent biological-replicate inference. This bundle separates their meanings safely but does not upgrade the underlying evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def collect_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"):
        records[str(path.relative_to(root))] = {
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
    return records


def discover_datasets(source: Path) -> list[Path]:
    result = []
    for child in sorted(source.iterdir()):
        if not child.is_dir() or not (child / "manifest.json").is_file():
            continue
        manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") == "decider.perturbation_classification.factorized.v1":
            result.append(child)
    if not result:
        raise ValueError(f"no factorized datasets found under {source}")
    return result


def build(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    status_path = source / "status.json"
    if status_path.exists() and json.loads(status_path.read_text())["status"] != "PASS":
        raise ValueError("preprocessing source is incomplete or failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.building-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir()
    started = time.time()
    json_dump(stage / "status.json", {"status": "RUNNING", "training_started": False})
    try:
        code_dir = stage / "code"
        code_dir.mkdir()
        shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
        for companion_name in ("test_resolution_bundle.py", "validate_resolution_bundle.py"):
            companion = Path(__file__).with_name(companion_name)
            if companion.is_file():
                shutil.copy2(companion, code_dir / companion.name)

        datasets: dict[str, Any] = {}
        for dataset_dir in discover_datasets(source):
            datasets[dataset_dir.name] = process_dataset(dataset_dir, stage / dataset_dir.name)

        bundle = {
            "schema": "decider.response_resolution_bundle.v3",
            "status": "PASS",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_root": str(source),
            "output_root": str(output),
            "training_started": False,
            "model_modified": False,
            "datasets": datasets,
            "elapsed_seconds": time.time() - started,
        }
        json_dump(stage / "report.json", bundle)
        (stage / "report.md").write_text(markdown_report(bundle), encoding="utf-8")
        json_dump(stage / "status.json", {"status": "PASS", "training_started": False})
        json_dump(stage / "artifact_sha256.json", collect_artifacts(stage))
        stage.rename(output)
    except Exception:
        json_dump(stage / "status.json", {"status": "FAILED", "training_started": False})
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="optional JSON config with labels.source and labels.output")
    parser.add_argument(
        "--source",
        type=Path,
        help="factorized preprocessing root (overrides config)",
    )
    parser.add_argument("--output", type=Path, help="new label bundle directory (overrides config)")
    args = parser.parse_args(argv)
    if args.config:
        # Keep standalone copied bundle audit scripts usable without this import.
        if __package__ in (None, ""):
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from jev_a4.config import load_config
        cfg = load_config(args.config)
        labels = cfg.get("labels", {})
        source = args.source or labels.get("source") or cfg.get("data_preprocessed_root")
        output = args.output or labels.get("output") or cfg.get("label_resolution_root")
        args.source = Path(source) if source else None
        args.output = Path(output) if output else None
    if args.source is None or args.output is None:
        parser.error("provide --source and --output, or configure labels.source and labels.output")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build(args.source, args.output)
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
