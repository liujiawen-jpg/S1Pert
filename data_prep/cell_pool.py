"""Fixed, batch-stratified K562 cell pools and margin-reference adjudication.

Run from the repository root: python -m data_prep.cell_pool --help.
The full raw matrix is streamed only by the explicit ``build`` command.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SEEDS = (17, 29, 43, 61, 89)
AB_SEED = 20261016
CELL = "k562_essential"
BINS = ("all", "20_39", "40_79", "80_149", "ge150")
RULES = ("Naive q_up,q_down > 0.05", "S1Pert TOST")
RECORD_DTYPE = np.dtype([("target", "<i4"), ("query", "<i4"), ("label", "i1")])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def csvwrite(path, rows):
    require(bool(rows), "no result rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def work_root(value):
    value = value or os.environ.get("S1PERT_ROOT")
    require(bool(value), "supply --work-dir or set S1PERT_ROOT")
    return Path(value).expanduser().resolve()


def load_array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def assign_cell_split(groups, batches, eligible_groups, seed=AB_SEED):
    """Historical exact allocation: -1 excluded, 0 A, 1 B.

    Each group has nearest-integer n/3 A cells. Batch allocation uses largest
    remainders, seeded random ties and within-batch shuffling. Per-group RNGs
    make the assignment independent of which other groups are selected.
    """
    groups, batches = np.asarray(groups), np.asarray(batches)
    eligible_groups = np.asarray(eligible_groups, dtype=bool)
    require(groups.ndim == batches.ndim == 1 and groups.shape == batches.shape,
            "aligned groups and recorded batches required")
    require(groups.dtype.kind in "iu" and eligible_groups.ndim == 1 and len(eligible_groups),
            "integer groups and a nonempty eligibility axis required")
    require((groups >= -1).all() and (groups < len(eligible_groups)).all(), "invalid group index")
    require(isinstance(seed, (int, np.integer)) and seed >= 0, "invalid cell-split seed")
    assignment = np.full(len(groups), -1, dtype=np.int8)
    for group in np.flatnonzero(eligible_groups):
        rows = np.flatnonzero(groups == group)
        if not len(rows):
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(group)]))
        _, inverse, counts = np.unique(batches[rows], return_inverse=True, return_counts=True)
        n_a = (len(rows) + 1) // 3
        allocation = counts // 3
        extras = int(n_a - allocation.sum())
        order = np.lexsort((rng.random(len(counts)), -(counts % 3)))
        allocation[order[:extras]] += 1
        assignment[rows] = 1
        for batch, take in enumerate(allocation):
            selected = rows[inverse == batch].copy()
            rng.shuffle(selected)
            assignment[selected[:take]] = 0
        require(np.count_nonzero(assignment[rows] == 0) == n_a, "group allocation drift")
    return assignment


def accumulate(destination, groups, values, nonzero):
    keep = groups >= 0
    np.add.at(destination[0], groups[keep], values[keep])
    np.add.at(destination[1], groups[keep], values[keep] ** 2)
    np.add.at(destination[2], groups[keep], nonzero[keep])


def moments_from_sufficient(sums, squares, nonzero, counts):
    counts = np.asarray(counts)
    require(sums.shape == squares.shape == nonzero.shape and sums.shape[0] == len(counts),
            "sufficient-statistic axes differ")
    denominator = np.maximum(1, counts)[:, None]
    mean = (sums / denominator).astype("float32")
    variance = np.maximum(0, (squares - sums * sums / denominator)
                          / np.maximum(1, counts - 1)[:, None]).astype("float32")
    return mean, variance, nonzero.astype("int32")


def split_statistics(raw, umi, groups, assignment, n_groups):
    """Small-fixture implementation of the same streamed accumulation."""
    raw, umi, groups, assignment = map(np.asarray, (raw, umi, groups, assignment))
    require(raw.ndim == 2 and len(raw) == len(umi) == len(groups) == len(assignment), "axes differ")
    require(np.isfinite(raw).all() and (raw >= 0).all() and np.equal(raw, np.floor(raw)).all(),
            "raw expression must be finite nonnegative integer counts")
    require(np.isfinite(umi).all() and (umi > 0).all() and (raw.sum(1) <= umi + 1e-5).all(),
            "invalid original UMI denominator")
    require(np.isin(assignment, (-1, 0, 1)).all(), "invalid assignment")
    require(groups.dtype.kind in "iu" and (groups >= -1).all() and (groups < n_groups).all(),
            "invalid groups")
    require(not np.any((groups < 0) & (assignment >= 0)), "unmapped cell assigned to a pool")
    index = np.where(assignment >= 0, groups + assignment.astype(np.int64) * n_groups, -1)
    stats = [np.zeros((2 * n_groups, raw.shape[1]), dtype=d) for d in (np.float64, np.float64, np.int64)]
    accumulate(stats, index, raw.astype(np.float64) / umi[:, None] * 1e4, raw > 0)
    counts = np.bincount(index[index >= 0], minlength=2 * n_groups)
    return counts, moments_from_sufficient(*stats, counts)


def build(raw, store, data, output, *, seeds=SEEDS, split_seed=AB_SEED, gene_block=268, expected_cells=None):
    """Stream raw counts, verify pooled moments, then label independent A/B pools."""
    import h5py
    from jev_a4.data.preprocess import metadata_replogle
    from jev_a4.labels import label_core
    from jev_a4.labels.label_core import Moments, label_pair

    raw, store, data, output = map(Path, (raw, store, data, output))
    require(not output.exists(), "A/B output already exists; select a fresh output directory")
    require(gene_block > 0, "gene block must be positive")
    meta, axes = metadata_replogle(raw), read_json(data / "dataset.json")
    require(meta["batch"] is not None, "the historical protocol requires recorded batch metadata")
    require(meta["group_ids"][1:] == axes["target_ids"] and meta["gene_id"] == axes["query_ids"],
            "raw and prepared identity axes differ")
    np.testing.assert_array_equal(load_array(store / "group_id.npy"), meta["group_ids"])
    np.testing.assert_array_equal(load_array(store / "gene_id.npy"), meta["gene_id"])
    if expected_cells is not None:
        require(meta["n_cells"] == expected_cells, "optional source-cell count assertion failed")
    k, g = len(meta["group_ids"]), meta["n_genes"]
    groups, umi = meta["groups"], meta["umi"]
    original_n = np.bincount(groups[groups >= 0], minlength=k)
    np.testing.assert_array_equal(original_n, load_array(store / "group_n.npy"))
    eligible = np.zeros(k, dtype=bool)
    eligible[0] = True
    partitions = {}
    for seed in seeds:
        path = data / f"seed_{seed}/target_partition.npy"
        partition = load_array(path)
        require(partition.shape == (k - 1,) and np.isin(partition, (0, 1, 2)).all(), "invalid frozen target partition")
        eligible[1:] |= partition == 2
        partitions[str(seed)] = identity(path)
    assignment = assign_cell_split(groups, meta["batch"], eligible, split_seed)
    n_a = np.bincount(groups[assignment == 0], minlength=k)
    n_b = np.bincount(groups[assignment == 1], minlength=k)
    np.testing.assert_array_equal(n_a + n_b, original_n * eligible)
    np.testing.assert_array_equal(n_a, ((original_n + 1) // 3) * eligible)
    require(min(n_a[0], n_b[0]) >= 20, "each disjoint control pool needs at least 20 cells")
    source_manifest = read_json(store / "manifest.json")
    definition = source_manifest["label_definition"]
    require(float(definition["ratio"]) == 1.2 and float(definition["alpha"]) == .05
            and definition["min_cells"] == 20 and definition["min_nonzero"] == 10
            and definition["q_method"] == "BH over 3G hypotheses per condition",
            "source labels must use the fixed margin/alpha/support/3G-BH contract")
    output.mkdir(parents=True)
    bindings = {"raw": identity(raw), "source_manifest": identity(store / "manifest.json"),
                "source_labels": identity(store / "labels.npy"),
                "dataset": identity(data / "dataset.json"), "partitions": partitions,
                "label_engine": identity(label_core.__file__), "cell_pool_code": identity(__file__)}
    raw_stat = raw.stat()
    np.save(output / "cell_split.npy", assignment)
    for half, counts in (("a", n_a), ("b", n_b)):
        np.save(output / f"group_n_{half}.npy", counts)
    destination = {}
    for half in ("a", "b"):
        destination[half] = [np.lib.format.open_memmap(output / f"{name}_{half}.npy", mode="w+",
                            dtype=dtype, shape=(k, g)) for name, dtype in
                            (("mean", "float32"), ("variance", "float32"), ("nonzero", "int32"))]
    source = [load_array(store / f"{name}.npy") for name in ("mean", "variance", "nonzero")]
    require(all(a.shape == (k, g) for a in source), "source moment axes differ")
    row_total = np.zeros(len(groups), dtype=np.float64)
    ab_groups = np.where(assignment >= 0, groups + assignment.astype(np.int64) * k, -1)
    with h5py.File(raw, "r", rdcc_nbytes=8 << 20) as handle:
        x = handle["X"]
        row_step = x.chunks[0] if x.chunks else 1024
        for left in range(0, g, gene_block):
            right = min(g, left + gene_block)
            pooled = [np.zeros((k, right-left), dtype=d) for d in (np.float64, np.float64, np.int64)]
            halves = [np.zeros((2*k, right-left), dtype=d) for d in (np.float64, np.float64, np.int64)]
            for start in range(0, len(groups), row_step):
                stop = min(start + row_step, len(groups))
                block = np.asarray(x[start:stop, left:right], dtype=np.float64)
                require(np.isfinite(block).all() and (block >= 0).all() and np.equal(block, np.floor(block)).all(),
                        "X contains invalid raw counts")
                row_total[start:stop] += block.sum(axis=1)
                values = block / umi[start:stop, None] * 1e4
                accumulate(pooled, groups[start:stop], values, block > 0)
                accumulate(halves, ab_groups[start:stop], values, block > 0)
            check = moments_from_sufficient(*pooled, original_n)
            np.testing.assert_allclose(check[0], source[0][:, left:right], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(check[1], source[1][:, left:right], rtol=3e-5, atol=2e-5)
            np.testing.assert_array_equal(check[2], source[2][:, left:right])
            moments = moments_from_sufficient(*halves, np.r_[n_a, n_b])
            for index, half in enumerate(("a", "b")):
                for out, value in zip(destination[half], moments):
                    out[:, left:right] = value[index*k:(index+1)*k]
            print(f"cell-pool moments: {right}/{g} genes", flush=True)
    require((row_total <= umi + 1e-5).all(), "raw row sums exceed original UMI metadata")
    require((raw.stat().st_size, raw.stat().st_mtime_ns) == (raw_stat.st_size, raw_stat.st_mtime_ns),
            "raw matrix changed while streaming")
    for half, counts in (("a", n_a), ("b", n_b)):
        mean, variance, nonzero = destination[half]
        for array in destination[half]:
            array.flush()
        controls = Moments(int(counts[0]), mean[0].astype(float), variance[0].astype(float), nonzero[0].astype(float))
        labels = np.full((k-1, g), -1, dtype=np.int8)
        reasons = np.full((k-1, g), 1, dtype=np.uint8)
        for target in np.flatnonzero(eligible[1:]):
            if counts[target+1] < 20:
                continue
            result = label_pair(Moments(int(counts[target+1]), mean[target+1].astype(float),
                                variance[target+1].astype(float), nonzero[target+1].astype(float)), controls)
            labels[target], reasons[target] = result["labels"], result["reason_codes"]
        np.save(output / f"labels_{half}.npy", labels)
        np.save(output / f"reasons_{half}.npy", reasons)
    metadata = {"status": "PASS", "schema": "s1pert.cell_pool.v1", "model_seeds": list(seeds),
        "split_seed": split_seed, "target_ids": axes["target_ids"], "query_ids": axes["query_ids"],
        "group_ids": meta["group_ids"], "eligible_groups": eligible.tolist(), "bindings": bindings,
        "split": "target x recorded batch; nearest integer n/3 in A; remainder B",
        "scope": "union of fixed TEST target sets plus independent pooled controls",
        "moment_precision": "float64 accumulation; float32 storage before label tests",
        "label_definition": {"ratio": 1.2, "alpha": .05, "min_cells": 20, "min_nonzero": 10, "family": "3G BH"},
        "source_moments_reconstructed": True, "training_performed": False}
    metadata["artifacts"] = {p.name: identity(p) for p in sorted(output.glob("*.npy"))}
    write_json(output / "metadata.json", metadata)
    return metadata


def proxy_mask(a, b, original, own_query=-1):
    a, b, original = map(np.asarray, (a, b, original))
    require(a.ndim == 1 and a.shape == b.shape == original.shape, "label axes differ")
    require(all(np.isin(v, (-1, 0, 1, 2)).all() for v in (a, b, original)), "invalid label")
    mask = (a < 0) & (b >= 0)
    if own_query >= 0:
        require(own_query < len(a), "self query out of range")
        mask[own_query] = False
    return mask, mask & (original < 0)


def prepare(pool, store, data, output):
    pool, store, data, output = map(Path, (pool, store, data, output))
    require(not output.exists(), "prepared proxy directory already exists")
    meta = read_json(pool / "metadata.json")
    require(meta["status"] == "PASS", "cell pools must complete first")
    axes = read_json(data / "dataset.json")
    require(axes["target_ids"] == meta["target_ids"] and axes["query_ids"] == meta["query_ids"], "AB axes differ")
    require(sha256(data / "dataset.json") == meta["bindings"]["dataset"]["sha256"], "dataset identity changed")
    require(sha256(store / "manifest.json") == meta["bindings"]["source_manifest"]["sha256"], "source changed")
    require(sha256(store / "labels.npy") == meta["bindings"]["source_labels"]["sha256"], "original labels changed")
    a, b, original = [load_array(p) for p in (pool / "labels_a.npy", pool / "labels_b.npy", store / "labels.npy")]
    require(a.shape == b.shape == original.shape == (len(axes["target_ids"]), len(axes["query_ids"])), "label axes differ")
    for name in ("labels_a.npy", "labels_b.npy"):
        require(sha256(pool / name) == meta["artifacts"][name]["sha256"], "cell-pool labels changed")
    query_index = {name: i for i, name in enumerate(axes["query_ids"])}
    report = {"status": "PASS", "model_seeds": meta["model_seeds"], "per_seed": {},
              "definition": "TEST nonself A-unresolved/B-resolved; full-unknown is a nested original-unresolved subset",
              "source_labels": identity(store / "labels.npy"), "pool_metadata": identity(pool / "metadata.json")}
    output.mkdir(parents=True)
    for seed in meta["model_seeds"]:
        partition_path = data / f"seed_{seed}/target_partition.npy"
        require(sha256(partition_path) == meta["bindings"]["partitions"][str(seed)]["sha256"], "target split changed")
        partition = load_array(partition_path)
        pieces, unknown = [], []
        for target in np.flatnonzero(partition == 2):
            mask, nested = proxy_mask(a[target], b[target], original[target], query_index.get(axes["target_ids"][target], -1))
            queries = np.flatnonzero(mask)
            records = np.empty(len(queries), dtype=RECORD_DTYPE)
            records["target"], records["query"], records["label"] = target, queries, b[target, queries]
            pieces.append(records)
            unknown.append(nested[queries])
        records = np.concatenate(pieces) if pieces else np.empty(0, dtype=RECORD_DTYPE)
        nested = np.concatenate(unknown) if unknown else np.empty(0, dtype=bool)
        directory = output / f"seed_{seed}"
        directory.mkdir()
        np.save(directory / "records.npy", records)
        np.save(directory / "keys.npy", np.column_stack((records["target"], records["query"])).astype("int32"))
        np.save(directory / "original_unknown.npy", nested)
        report["per_seed"][str(seed)] = {"records": len(records), "original_unknown": int(nested.sum()),
            "artifacts": {p.name: identity(p) for p in sorted(directory.glob("*.npy"))}}
    write_json(output / "report.json", report)
    return report


def fidelity(pool, output, expected_reference_de=None):
    from jev_a4.labels.label_core import Moments, label_pair
    pool, output = Path(pool), Path(output)
    meta = read_json(pool / "metadata.json")
    require(meta["status"] == "PASS", "cell pools must complete first")
    for name, binding in meta["artifacts"].items():
        require(sha256(pool / name) == binding["sha256"], "cell-pool artifact changed: " + name)
    counts, mean, variance, nonzero, a, b = [load_array(pool / name) for name in
        ("group_n_a.npy", "mean_a.npy", "variance_a.npy", "nonzero_a.npy", "labels_a.npy", "labels_b.npy")]
    controls = Moments(int(counts[0]), mean[0].astype(float), variance[0].astype(float), nonzero[0].astype(float))
    qi = {name: i for i, name in enumerate(meta["query_ids"])}
    fields = ("candidate", "eligible", "b_de", "false_same", "b_resolved", "a_resolved", "both_resolved", "equal", "b_de_a_up", "b_de_a_same", "b_de_a_down", "b_de_a_unresolved")
    stats = {(rule, bucket): dict.fromkeys(fields, 0) for rule in RULES for bucket in BINS}
    for target in range(len(a)):
        n = int(counts[target+1])
        nonself = np.ones(a.shape[1], dtype=bool)
        own = qi.get(meta["target_ids"][target])
        if own is not None:
            nonself[own] = False
        if n < 20:
            # Historical 'all' candidate denominator counts selected small groups.
            if n > 0:
                for rule in RULES:
                    stats[(rule, "all")]["candidate"] += int(nonself.sum())
            continue
        result = label_pair(Moments(n, mean[target+1].astype(float), variance[target+1].astype(float), nonzero[target+1].astype(float)), controls)
        np.testing.assert_array_equal(result["labels"], a[target])
        naive = result["labels"].copy()
        naive[result["eligible"] & (result["q_up"] > .05) & (result["q_down"] > .05)] = 1
        eligible = result["eligible"] & nonself
        bde = eligible & np.isin(b[target], (0, 2))
        bucket = "20_39" if n < 40 else "40_79" if n < 80 else "80_149" if n < 150 else "ge150"
        for rule, labels in ((RULES[0], naive), (RULES[1], result["labels"])):
            ar = eligible & (labels >= 0)
            both = ar & (b[target] >= 0)
            values = dict(candidate=int(nonself.sum()), eligible=int(eligible.sum()), b_de=int(bde.sum()),
                false_same=int(np.sum(bde & (labels == 1))), b_resolved=int(np.sum(eligible & (b[target] >= 0))),
                a_resolved=int(ar.sum()), both_resolved=int(both.sum()), equal=int(np.sum(both & (labels == b[target]))))
            for code, name in ((0, "up"), (1, "same"), (2, "down"), (-1, "unresolved")):
                values["b_de_a_" + name] = int(np.sum(bde & (labels == code)))
            for key in ("all", bucket):
                for name, value in values.items():
                    stats[(rule, key)][name] += value
    if expected_reference_de is not None:
        require(stats[(RULES[1], "all")]["b_de"] == expected_reference_de, "optional Pool-B reference count assertion failed")
    rows = []
    for (rule, bucket), z in stats.items():
        rows.append(dict(rule=rule, cell_count_bin=bucket, **z,
            false_invariance_percent=100*z["false_same"]/z["b_de"] if z["b_de"] else None,
            resolved_concordance=z["equal"]/z["both_resolved"] if z["both_resolved"] else None,
            resolution_coverage_eligible_percent=100*z["a_resolved"]/z["eligible"] if z["eligible"] else None,
            resolution_coverage_candidate_percent=100*z["a_resolved"]/z["candidate"] if z["candidate"] else None))
    csvwrite(output / "table_tost_fidelity.csv", rows)
    write_json(output / "fidelity_provenance.json", {"status": "PASS", "pool_metadata": identity(pool / "metadata.json"),
        "all_candidate_denominator": "historical denominator includes selected groups with 1-19 A cells; all evaluated pairs have >=20",
        "false_invariance_denominator": "B DE among A-quality-eligible nonself pairs", "counts_derived_from_labels": True})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "prepare", "fidelity"))
    parser.add_argument("--work-dir")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-cells", type=int)
    parser.add_argument("--expected-reference-de", type=int)
    parser.add_argument("--gene-block", type=int, default=268)
    args = parser.parse_args(argv)
    root = work_root(args.work_dir)
    pool = args.pool or root / "cell_pool/pools"
    store, data = args.store or root / "preprocessed" / CELL, args.data or root / "data" / CELL
    if args.command == "build":
        build(args.raw or root / "raw/scPerturb_prepro/ReplogleWeissman2022_K562_essential.h5ad", store, data,
              args.output or pool, gene_block=args.gene_block, expected_cells=args.expected_cells)
    elif args.command == "prepare":
        prepare(pool, store, data, args.output or root / "cell_pool/prepared")
    else:
        fidelity(pool, args.output or root / "paper/fig2", args.expected_reference_de)


if __name__ == "__main__":
    main()
