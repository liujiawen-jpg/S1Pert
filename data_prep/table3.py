"""Derive Table III directly from strict labels and stable identity axes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_prep.cell_pool import csvwrite, identity, load_array, read_json, require, work_root, write_json

CELLS = ("hepg2", "jurkat", "rpe1", "k562_essential")


def summarize_arrays(labels, resolution, target_ids, query_ids):
    labels, resolution = np.asarray(labels), np.asarray(resolution)
    require(labels.shape == resolution.shape == (len(target_ids), len(query_ids)), "label/identity axes differ")
    require(len(set(target_ids)) == len(target_ids) and len(set(query_ids)) == len(query_ids), "duplicate stable identities")
    require(np.isin(labels, (-1, 0, 1, 2)).all() and np.isin(resolution, range(8)).all(), "invalid label/resolution code")
    require(np.array_equal(labels >= 0, resolution <= 2), "strict label/resolution disagreement")
    require(np.array_equal(labels[labels >= 0], resolution[labels >= 0]), "resolved classes disagree")
    qi = {name: index for index, name in enumerate(query_ids)}
    counts = np.zeros(3, dtype=np.int64)
    eligible, total, self_pairs = 0, 0, 0
    for target, name in enumerate(target_ids):
        nonself = np.ones(len(query_ids), dtype=bool)
        if name in qi:
            nonself[qi[name]] = False
            self_pairs += 1
        row, reason = labels[target][nonself], resolution[target][nonself]
        counts += np.bincount(row[row >= 0], minlength=3)
        eligible += int(np.sum((reason <= 2) | (reason == 7)))
        total += len(row)
    resolved = int(counts.sum())
    require(resolved <= eligible <= total, "invalid quality/resolution denominators")
    return {"targets": len(target_ids), "genes": len(query_ids), "total_candidate_pairs": total,
        "eligible_pairs": eligible, "up": int(counts[0]), "same": int(counts[1]), "down": int(counts[2]),
        "unresolved_pairs": total-resolved, "unresolved_percent": 100*(total-resolved)/total if total else None,
        "de_to_same_ratio": (int(counts[0])+int(counts[2]))/int(counts[1]) if counts[1] else None,
        "self_pairs_excluded": self_pairs}


def generate(root, output=None, cells=CELLS):
    root = Path(root)
    output = Path(output) if output else root / "paper/table3"
    rows, inputs = [], {}
    for cell in cells:
        data, strict = root / "data" / cell, root / "strict_labels" / cell
        axes = read_json(data / "dataset.json")
        manifest = read_json(strict / "manifest.json")
        preparation = read_json(data / "preparation_report.json")
        require(preparation["status"] == "PASS", "model-data preparation incomplete")
        require(identity(data / "dataset.json")["sha256"] == preparation["artifacts"]["dataset.json"]["sha256"],
                "prepared identity axes changed")
        require(identity(strict / "manifest.json")["sha256"] == preparation["source"]["strict_manifest"]["sha256"],
                "strict-label manifest differs from model-data preparation")
        for name in ("hard_label.npy", "resolution_code.npy"):
            require(identity(strict / name)["sha256"] == manifest["arrays"][name]["sha256"], "strict-label payload changed")
        labels, resolution = load_array(strict / "hard_label.npy"), load_array(strict / "resolution_code.npy")
        rows.append({"cell": cell, **summarize_arrays(labels, resolution, axes["target_ids"], axes["query_ids"]),
                     "label_manifest_sha256": identity(strict / "manifest.json")["sha256"]})
        inputs[cell] = {"axes": identity(data / "dataset.json"), "labels": identity(strict / "hard_label.npy"),
                        "resolution": identity(strict / "resolution_code.npy"), "manifest": identity(strict / "manifest.json")}
    csvwrite(output / "table_dataset_summary.csv", rows)
    write_json(output / "table3_audit.json", {"status": "PASS", "inputs": inputs, "rows": rows,
        "class_order": ["up", "same", "down"],
        "candidate_denominator": "all stable target/query pairs minus identity-matched self pairs",
        "eligible_denominator": "resolution code 0,1,2 or 7 after self exclusion",
        "unresolved_denominator": "all nonself candidates, including support/estimability failures",
        "stage1_protocol": {
            "normalization": "raw X / original per-cell UMI * 10000, linear scale",
            "margin": 1.2, "alpha": .05, "min_cells_per_group": 20, "min_nonzero_per_group": 10,
            "statistics": "scaled Welch one-sided margin tests and TOST intersection-union equivalence",
            "multiplicity": "per-target combined 3G BH, including self and low-support genes; invalid moments have p=1",
            "ordering": "BH precedes cell/nonzero eligibility; self is excluded from model records and reported pair counts",
            "unknown": "missing supervision, not Same or a fourth class",
            "point_region": "descriptive; never replaces strict labels"}})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=CELLS)
    args = parser.parse_args(argv)
    generate(work_root(args.work_dir), args.output, args.cells)


if __name__ == "__main__":
    main()
