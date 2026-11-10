"""Strict, response-blind-input preparation for independent A4 cell-line runs.

Only TRAIN hard labels enter response-tendency features.  Held-out labels are
used exclusively to store evaluation references.  Nothing here reads an old
prediction, a task checkpoint, or perturbed expression as a model covariate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jev_a4.config import load_config
from jev_a4.data.h5_metadata import text_array

SEEDS = (17, 29, 43, 61, 89)
PARTITIONS = ("train", "val", "test")
RECORD_DTYPE = np.dtype([("target", "<i4"), ("query", "<i4"), ("label", "i1")])
FEATURE_ORDER = [
    "p_up", "p_same", "p_down", "log_support", "normalized_entropy",
    "concentration", "top2_margin", "cell_query_source", "cell_fallback_source",
    "global_fallback_source", "up_minus_down", "evidence_available",
]
SCHEMA = "decider.a4.independent_cell.strict_712.v1"


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(values.dtype).encode())
    h.update(json.dumps(values.shape).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def _identity(path: Path) -> dict:
    p = Path(path).resolve()
    s = p.stat()
    return {"path": str(p), "size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns,
            "sha256": sha256_file(p)}


def _strings(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _decode_h5_column(handle, node) -> list[str]:
    """Decode both historical AnnData reference and modern categorical encodings."""
    return _strings(text_array(node.parent, node.name.rsplit("/", 1)[-1])[0])


def decode_raw_symbol_mapping(raw: Path) -> tuple[dict[str, str], dict]:
    """Read gene annotations only, never X or expression-derived var columns."""
    import h5py
    mapping: dict[str, str] = {}
    obs_map: dict[str, str] = {}
    with h5py.File(raw, "r") as h:
        var = h["var"]
        id_key = next((key for key in ("gene_id", "ensembl_id") if key in var),
                      str(var.attrs.get("_index", "_index")))
        if id_key not in var and isinstance(var.attrs.get("_index"), bytes):
            id_key = var.attrs["_index"].decode()
        symbol_key = next((k for k in ("gene_name", "gene_symbol", "gene_symbols") if k in var), None)
        if symbol_key is None:
            raise ValueError("raw var lacks an explicit gene-name annotation")
        ids = _decode_h5_column(h, var[id_key])
        symbols = _decode_h5_column(h, var[symbol_key])
        if len(ids) != len(symbols) or len(set(ids)) != len(ids):
            raise ValueError("raw var gene identities are duplicated or misaligned")
        for gid, symbol in zip(ids, symbols):
            if not symbol or symbol.isdecimal():
                raise ValueError(f"raw symbol is absent/numeric for {gid}")
            mapping[gid] = symbol
        obs = h["obs"]
        if "gene_id" in obs and "gene" in obs:
            obs_ids = _decode_h5_column(h, obs["gene_id"])
            obs_symbols = _decode_h5_column(h, obs["gene"])
            if len(obs_ids) != len(obs_symbols):
                raise ValueError("target metadata gene identity/name lengths differ")
            for gid, symbol in zip(obs_ids, obs_symbols):
                if not symbol or symbol.isdecimal():
                    raise ValueError("numeric/empty perturbation annotation")
                if not gid or gid.lower() in {"nan", "none", "na", "null"}:
                    # Canonical source preprocessing explicitly namespaces cases
                    # without an Ensembl identity; it is not inferred from effects.
                    gid = "SYMBOL:" + symbol
                existing = obs_map.get(gid)
                if existing is not None and existing != symbol:
                    raise ValueError(f"one target ID maps to multiple source symbols: {gid}")
                obs_map[gid] = symbol
    # var is the query-axis annotation.  Target-specific aliases from obs are
    # kept separate to avoid silently replacing an authoritative query symbol.
    return mapping, {"target_mapping": obs_map, "raw_path": str(raw),
                     "var_gene_id_column": id_key, "var_symbol_column": symbol_key,
                     "expression_matrix_read": False,
                     "mapping_method": "decode source categorical annotations by exact gene ID"}


def _duplicate_symbols(ids: list[str], symbols: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for gid, symbol in zip(ids, symbols):
        grouped[symbol].append(gid)
    return {s: sorted(set(v)) for s, v in grouped.items() if len(set(v)) > 1}


def resolve_target_identities(query_ids: list[str], query_symbols: list[str],
                              target_ids: list[str], target_symbols: list[str]) -> tuple[list[int], list[str], dict]:
    """Resolve only explicit source-symbol identities, never arbitrary ID aliases.

    A unique SYMBOL:source_name/query-name match is enough to identify a
    measured self response.  Different explicit Ensembl identities are never
    collapsed because their symbols happen to agree.  Ambiguous identities or
    two target rows resolving to one canonical identity block the preparation.
    """
    query_by_id = {g: i for i, g in enumerate(query_ids)}
    if len(query_by_id) != len(query_ids):
        raise ValueError("duplicate query identities prevent alias resolution")
    query_by_symbol: dict[str, list[int]] = defaultdict(list)
    for i, symbol in enumerate(query_symbols):
        query_by_symbol[symbol].append(i)
    indices, canonical, aliases = [], [], []
    for gid, symbol in zip(target_ids, target_symbols):
        q = query_by_id.get(gid, -1)
        resolved = gid
        if gid.startswith("SYMBOL:"):
            source_symbol = gid.split(":", 1)[1]
            if source_symbol != symbol:
                raise ValueError(f"namespaced target ID/source name mismatch: {gid}, {symbol}")
            matches = query_by_symbol.get(symbol, [])
            if len(matches) > 1:
                raise ValueError(f"ambiguous source-symbol target identity; do not guess a query ID: {gid}")
            if len(matches) == 1:
                q = matches[0]
                resolved = query_ids[q]
                aliases.append({"source_target_id": gid, "canonical_target_id": resolved,
                    "source_symbol": symbol, "query_index": q,
                    "rule": "explicit source-namespaced symbol matches exactly one query symbol"})
        indices.append(q)
        canonical.append(resolved)
    seen: dict[str, str] = {}
    for source, resolved in zip(target_ids, canonical):
        if resolved in seen:
            raise ValueError(f"multiple target rows represent one canonical identity; consolidate with explicit source audit before splitting: {seen[resolved]}, {source} -> {resolved}")
        seen[resolved] = source
    return indices, canonical, {"resolved_source_symbol_aliases": aliases,
        "resolved_alias_count": len(aliases), "ambiguous_aliases": 0,
        "canonical_target_collisions": 0,
        "alias_rule": "only source SYMBOL names with unique query-symbol match; never merge distinct explicit Ensembl identities",
        "collision_rule": "fatal, not silently split or merge source responses"}


def split_targets(target_ids: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the whole target axis; labels/counts never influence the assignment."""
    if len(set(target_ids)) != len(target_ids) or any(not t for t in target_ids):
        raise ValueError("target IDs must be nonempty and unique")
    n = len(target_ids)
    if n < 10:
        raise ValueError("at least ten targets required for a 70/10/20 split")
    ordered = np.array(sorted(range(n), key=lambda i: target_ids[i]), dtype=np.int64)
    shuffled = ordered[np.random.default_rng(int(seed)).permutation(n)]
    ntrain, nval = int(math.floor(.7 * n)), int(math.floor(.1 * n))
    partition = np.full(n, 2, dtype=np.int8)
    partition[shuffled[:ntrain]] = 0
    partition[shuffled[ntrain:ntrain + nval]] = 1
    train = ordered[partition[ordered] == 0]
    shuffled_train = train[np.random.default_rng(int(seed) + 1000003).permutation(len(train))]
    fold = np.full(n, -1, dtype=np.int8)
    fold[shuffled_train] = np.arange(len(train), dtype=np.int64) % 5
    return partition, fold


def make_records(labels: np.ndarray, mask: np.ndarray, target_indices: np.ndarray,
                 target_query_indices: np.ndarray, exclude_self_pairs: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rows, self_rows = [], []
    for t in np.asarray(target_indices, dtype=np.int64):
        y = np.asarray(labels[int(t)])
        m = np.asarray(mask[int(t)], dtype=bool)
        q = np.flatnonzero(m)
        if len(q) and np.any((y[q] < 0) | (y[q] > 2)):
            raise ValueError("supervision mask selects unresolved/invalid labels")
        own = int(target_query_indices[int(t)])
        self_q = q[q == own] if own >= 0 else np.empty(0, dtype=np.int64)
        if len(self_q):
            z = np.empty(len(self_q), dtype=RECORD_DTYPE)
            z["target"], z["query"], z["label"] = int(t), self_q, y[self_q]
            self_rows.append(z)
        if exclude_self_pairs and own >= 0:
            q = q[q != own]
        if len(q):
            z = np.empty(len(q), dtype=RECORD_DTYPE)
            z["target"], z["query"], z["label"] = int(t), q, y[q]
            rows.append(z)
    empty = np.empty(0, dtype=RECORD_DTYPE)
    return (np.concatenate(rows) if rows else empty,
            np.concatenate(self_rows) if self_rows else empty.copy())


def _probability_arrays(raw: np.ndarray, query_support: np.ndarray,
                        fallback: np.ndarray, eligible_targets: np.ndarray,
                        denominator: float) -> dict[str, np.ndarray]:
    raw = np.asarray(raw, dtype=np.int64)
    support = raw.sum(axis=1)
    probabilities = ((raw + 1.) / (support[:, None] + 3.)).astype(np.float32)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.))).sum(axis=1) / math.log(3.)
    ordered = np.sort(probabilities, axis=1)
    available = support > 0
    features = np.column_stack([
        probabilities, np.log1p(support) / denominator, entropy,
        probabilities.max(axis=1), ordered[:, -1] - ordered[:, -2],
        fallback == 0, fallback == 1, np.zeros(len(raw), dtype=bool),
        probabilities[:, 0] - probabilities[:, 2], available,
    ]).astype(np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("non-finite response-tendency features")
    coverage = np.divide(query_support, eligible_targets,
                         out=np.zeros(len(raw), dtype=np.float64), where=eligible_targets > 0)
    return {"features": features, "raw_counts": raw.astype(np.int32),
            "fallback": fallback.astype(np.int8), "query_coverage": coverage.astype(np.float32)}


def prior_arrays(train: np.ndarray, requested: dict[str, np.ndarray], target_partition: np.ndarray,
                 target_fold: np.ndarray, genes: int) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Fit only supplied TRAIN labels; requested held-out label fields are unread."""
    if train.dtype != RECORD_DTYPE:
        raise ValueError("unexpected record dtype")
    if len(train) == 0:
        raise ValueError("TRAIN has no strict downstream supervision")
    if np.any(target_partition[train["target"]] != 0):
        raise ValueError("prior fitting includes non-TRAIN targets")
    y = np.asarray(train["label"], dtype=np.int64)
    if np.any((y < 0) | (y > 2)):
        raise ValueError("prior fitting includes unresolved labels")
    folds = np.asarray(target_fold[train["target"]], dtype=np.int64)
    if np.any((folds < 0) | (folds >= 5)):
        raise ValueError("TRAIN target fold missing")
    total = np.zeros((genes, 3), dtype=np.int64)
    by_fold = np.zeros((5, genes, 3), dtype=np.int64)
    np.add.at(total, (train["query"], y), 1)
    np.add.at(by_fold, (folds, train["query"], y), 1)
    all_classes = total.sum(axis=0)
    fold_classes = by_fold.sum(axis=1)
    train_targets = int((target_partition == 0).sum())
    fold_targets = np.bincount(target_fold[target_partition == 0], minlength=5)
    chosen: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for partition, records in requested.items():
        q = np.asarray(records["query"], dtype=np.int64)
        if partition == "train":
            f = np.asarray(target_fold[records["target"]], dtype=np.int64)
            raw = total[q] - by_fold[f, q]
            fallback_counts = all_classes[None, :] - fold_classes[f]
            eligible = train_targets - fold_targets[f]
        else:
            raw = total[q].copy()
            fallback_counts = np.broadcast_to(all_classes, (len(records), 3))
            eligible = np.full(len(records), train_targets, dtype=np.int64)
        query_support = raw.sum(axis=1)
        fallback = (query_support == 0).astype(np.int8)
        raw = raw.copy()
        raw[fallback == 1] = fallback_counts[fallback == 1]
        chosen[partition] = raw, query_support, fallback, eligible
    denom = max(float(np.log1p(chosen["train"][0].sum(axis=1).max(initial=0))), 1.)
    outputs = {p: _probability_arrays(*v, denom) for p, v in chosen.items()}
    audit = {"labels_used_to_construct": ["TRAIN"], "nontrain_construction_label_reads": 0,
             "folds": 5, "alpha": 1., "support_log_denominator": denom,
             "support_scaling_fit_partition": "TRAIN", "feature_width": len(FEATURE_ORDER),
             "feature_order": FEATURE_ORDER, "train_class_counts": all_classes.tolist(),
             "target_fold_counts": fold_targets.tolist(),
             "fallback": "same cell TRAIN class prevalence only when query support is zero",
             "probability_meaning": "class tendency conditional on strict label resolution, not all-grid response prevalence",
             "query_coverage_meaning": "strict downstream support count / eligible TRAIN targets; original unresolved fraction is not a response class",
             "old_prediction_or_checkpoint_read": False, "source_mean_output_anchor": False,
             "residual_gate_blend_or_calibration": False}
    return outputs, audit


def _paths(cfg: dict, dataset: str) -> tuple[Path, Path]:
    paths = cfg.get("paths", {})
    data = cfg.get("data_preprocessed_root", cfg.get("data_root", paths.get("data_root")))
    label = cfg.get("label_resolution_root", cfg.get("labels_root", paths.get("labels_root")))
    if not data or not label:
        raise ValueError("configure data_preprocessed_root and label_resolution_root explicitly")
    if not dataset or Path(dataset).name != dataset or dataset in (".", ".."):
        raise ValueError("dataset must be a single path component")
    return Path(data) / dataset, Path(label) / dataset


def _load_axes(data: Path, source_manifest: dict, raw_override: Path | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    gene_ids = _strings(np.load(data / "gene_id.npy", allow_pickle=False))
    original_symbols = _strings(np.load(data / "gene_symbol.npy", allow_pickle=False))
    target_ids = _strings(np.load(data / "response_group_id.npy", allow_pickle=False))
    group_ids = _strings(np.load(data / "group_id.npy", allow_pickle=False))
    group_names = _strings(np.load(data / "group_name.npy", allow_pickle=False))
    if len(set(gene_ids)) != len(gene_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("duplicate gene or perturbation target identity")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("group ID axis is ambiguous")
    controls = [i for i, gid in enumerate(group_ids) if gid.upper() in {"CONTROL", "NON-TARGETING", "NON_TARGETING"}]
    if len(controls) != 1:
        raise ValueError("cannot establish exactly one control group")
    control_index = controls[0]
    group_index = {g: i for i, g in enumerate(group_ids)}
    if any(t not in group_index for t in target_ids):
        raise ValueError("reference target axis is missing from moments group axis")
    target_names = {g: group_names[i] for i, g in enumerate(group_ids)}
    mapping_audit = {"source": "preprocessed explicit symbol axis", "numeric_original_symbols": 0,
                     "expression_matrix_read": False}
    extra_targets = {}
    if any(not s or s.isdecimal() for s in original_symbols):
        source = source_manifest.get("source", {})
        recorded = source.get("raw", source).get("path")
        selected = raw_override or recorded
        if not selected or not Path(selected).is_file():
            raise FileNotFoundError("raw annotation source unavailable; set raw_h5ad_paths[dataset] to repair legacy numeric symbols")
        raw_path = Path(selected)
        decoded, mapping_audit = decode_raw_symbol_mapping(raw_path)
        extra_targets = mapping_audit.pop("target_mapping")
        if any(g not in decoded for g in gene_ids):
            raise ValueError("raw symbol mapping does not cover query gene IDs")
        symbols = [decoded[g] for g in gene_ids]
        mapping_audit["numeric_original_symbols"] = sum(s.isdecimal() for s in original_symbols)
        mapping_audit["source"] = "raw H5AD annotations, categorical decoding"
    else:
        symbols = original_symbols
    by_query = {g: i for i, g in enumerate(gene_ids)}
    target_symbols = []
    unresolved = []
    for gid in target_ids:
        q = by_query.get(gid, -1)
        if gid in extra_targets:
            symbol = extra_targets[gid]
        elif q >= 0:
            symbol = symbols[q]
        elif gid.startswith("SYMBOL:"):
            symbol = gid.split(":", 1)[1]
        else:
            symbol = target_names[gid]
        if not symbol or symbol.isdecimal() or symbol == gid:
            unresolved.append(gid)
        target_symbols.append(symbol)
    if unresolved:
        raise ValueError(f"unresolved/numeric target gene names: {unresolved[:10]}")
    if any(not s or s.isdecimal() for s in symbols):
        raise ValueError("unresolved/numeric query gene symbols")
    target_query_indices, canonical_target_ids, alias_audit = resolve_target_identities(
        gene_ids, symbols, target_ids, target_symbols)
    control_n = int(np.load(data / "group_n.npy", mmap_mode="r")[control_index])
    if control_n <= 0:
        raise ValueError("control group has no cells")
    # Index only the control row; perturbation moments are never materialized.
    means = np.asarray(np.load(data / "mean.npy", mmap_mode="r")[control_index], dtype=np.float64).copy()
    variances = np.asarray(np.load(data / "variance.npy", mmap_mode="r")[control_index], dtype=np.float64).copy()
    nonzero = np.asarray(np.load(data / "nonzero.npy", mmap_mode="r")[control_index], dtype=np.float64).copy()
    if any(v.shape != (len(gene_ids),) for v in (means, variances, nonzero)):
        raise ValueError("control moments query axis mismatch")
    if not all(np.isfinite(v).all() for v in (means, variances, nonzero)):
        raise ValueError("control moments contain non-finite values")
    if np.any(means < 0) or np.any(variances < 0) or np.any(nonzero < 0) or np.any(nonzero > control_n):
        raise ValueError("control moment range violation")
    detection = nonzero / control_n
    query_numeric = np.column_stack((np.log1p(means), np.log1p(variances), detection,
                                     np.ones(len(gene_ids)))).astype(np.float32)
    condition_numeric = np.zeros((len(target_ids), 4), dtype=np.float32)
    available_targets = np.asarray(target_query_indices) >= 0
    condition_numeric[available_targets] = query_numeric[np.asarray(target_query_indices)[available_targets]]
    duplicates = _duplicate_symbols(gene_ids, symbols)
    axes = {"schema": SCHEMA, "target_ids": target_ids, "target_symbols": target_symbols,
            "target_partition_group_ids": canonical_target_ids, "target_identity_alias_audit": alias_audit,
            "target_query_indices": target_query_indices, "query_ids": gene_ids, "query_symbols": symbols,
            "gene_ids": gene_ids, "gene_symbols": symbols, "control_index": control_index,
            "control_n": control_n, "control_mean": means.tolist(), "control_variance": variances.tolist(),
            "control_detection": detection.tolist(), "numeric_feature_order": ["log1p(control_mean)",
                "log1p(control_variance)", "control_detection", "measurement_available"],
            "duplicate_symbol_ids": duplicates, "mapping_audit": mapping_audit,
            "identity_contract": {"primary_key": "exact Ensembl/source-namespaced identity, never a symbol",
                "duplicate_ids": "fatal", "duplicate_symbols": "retain distinct IDs; never merge aliases; ambiguous symbol-only retrieval must be masked",
                "numeric_symbol": "fatal", "target_outside_query_axis": "control numeric unavailable; public target symbol retained"},
            "inputs": {"normalization": source_manifest["normalization"], "control_moments_only": True,
                       "perturbation_moments_used_as_input": False, "public_symbols_are_labels_blind_metadata": True,
                       "cross_cell_response_inputs": False}}
    return axes, condition_numeric, query_numeric


def prepare(cfg: dict, dataset: str, out: Path) -> dict:
    out = Path(out)
    if not np.allclose(cfg.get("split_ratio", [.7, .1, .2]), [.7, .1, .2]):
        raise ValueError("this paper protocol requires target TRAIN/VAL/TEST = 70/10/20")
    if int(cfg.get("prior", {}).get("folds", 5)) != 5 or float(cfg.get("prior", {}).get("alpha", 1.)) != 1.:
        raise ValueError("A4 evidence requires five folds and Laplace alpha=1")
    data, label_root = _paths(cfg, dataset)
    source_manifest = read_json(data / "manifest.json")
    label_manifest = read_json(label_root / "manifest.json")
    if label_manifest.get("schema") != "decider.response_resolution_labels.v3":
        raise ValueError("strict response-resolution v3 labels required")
    raw_override = cfg.get("raw_h5ad_paths", {}).get(dataset)
    axes, condition_numeric, query_numeric = _load_axes(
        data, source_manifest, Path(raw_override) if raw_override else None)
    axes["dataset"] = dataset
    labels = np.load(label_root / "hard_label.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(label_root / "supervision_mask.npy", mmap_mode="r", allow_pickle=False)
    expected_shape = (len(axes["target_ids"]), len(axes["query_ids"]))
    if labels.shape != expected_shape or mask.shape != expected_shape or mask.dtype != np.bool_:
        raise ValueError("strict labels/mask/axes mismatch")
    if np.any(np.asarray(mask) != np.isin(labels, [0, 1, 2])):
        raise ValueError("supervision mask must be exactly the set of strict 0/1/2 labels")
    if np.any(~np.isin(labels, [-100, 0, 1, 2])):
        raise ValueError("hard label contains invalid code")
    seeds = tuple(int(s) for s in cfg.get("seeds", SEEDS))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("paper protocol requires exactly five distinct split seeds")
    exclude_self = bool(cfg.get("exclude_self_pairs", True))
    out.mkdir(parents=True, exist_ok=True)
    source = {"preprocessed_manifest": _identity(data / "manifest.json"),
              "strict_manifest": _identity(label_root / "manifest.json"),
              "hard_label": _identity(label_root / "hard_label.npy"),
              "supervision_mask": _identity(label_root / "supervision_mask.npy")}
    logical = hashlib.sha256(json.dumps(axes, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode("utf-8"))
    logical.update(array_sha256(condition_numeric).encode())
    logical.update(array_sha256(query_numeric).encode())
    source["axes_and_control_inputs_sha256"] = logical.hexdigest()
    for name in ("hard_label", "supervision_mask"):
        declared = label_manifest.get("arrays", {}).get(name + ".npy", {}).get("sha256")
        if declared and declared != source[name]["sha256"]:
            raise ValueError(f"strict label source seal changed: {name}")
    declared_manifest = label_manifest.get("source", {}).get("manifest", {}).get("sha256")
    if declared_manifest and declared_manifest != source["preprocessed_manifest"]["sha256"]:
        raise ValueError("preprocessed metadata no longer matches strict label provenance")
    complete_path = out / "preparation_report.json"
    if complete_path.exists():
        existing = read_json(complete_path)
        if existing.get("source") != source or existing.get("seeds") != list(seeds) or existing.get("exclude_self_pairs") != exclude_self:
            raise ValueError("existing preparation belongs to different source/protocol")
        for relative, identity in existing["artifacts"].items():
            path = out / relative
            if not path.exists() or path.stat().st_size != identity["size_bytes"] or sha256_file(path) != identity["sha256"]:
                raise ValueError(f"existing preparation artifact changed: {relative}")
        return existing
    dump_json(out / "dataset.json", axes)
    np.save(out / "condition_numeric.npy", condition_numeric, allow_pickle=False)
    np.save(out / "query_numeric.npy", query_numeric, allow_pickle=False)
    reports = []
    partition_hashes = []
    for seed in seeds:
        seed_out = out / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        partition, fold = split_targets(axes["target_partition_group_ids"], seed)
        np.save(seed_out / "target_partition.npy", partition, allow_pickle=False)
        np.save(seed_out / "target_fold.npy", fold, allow_pickle=False)
        records, self_records = {}, {}
        for code, p in enumerate(PARTITIONS):
            targets = np.flatnonzero(partition == code)
            records[p], self_records[p] = make_records(labels, mask, targets,
                np.asarray(axes["target_query_indices"]), exclude_self)
        arrays, prior_audit = prior_arrays(records["train"], records, partition, fold, expected_shape[1])
        # Independently replay each excluded TRAIN fold from eligible records,
        # rather than merely asserting a property of the vectorized subtraction.
        train_fold = fold[records["train"]["target"]]
        for excluded in range(5):
            allowed = records["train"][train_fold != excluded]
            held = np.flatnonzero(train_fold == excluded)
            query_counts = np.zeros((expected_shape[1], 3), dtype=np.int64)
            np.add.at(query_counts, (allowed["query"], allowed["label"]), 1)
            expected = query_counts[records["train"]["query"][held]].copy()
            empty = expected.sum(axis=1) == 0
            expected[empty] = query_counts.sum(axis=0)
            if not np.array_equal(expected, arrays["train"]["raw_counts"][held]):
                raise AssertionError("independent same-fold exclusion replay failed")
        # Exact, independent replay of held-out input feature invariance.
        mutated = {p: r.copy() for p, r in records.items()}
        for p in ("val", "test"):
            mutated[p]["label"] = (mutated[p]["label"] + 1) % 3
        replay, _ = prior_arrays(records["train"], mutated, partition, fold, expected_shape[1])
        if any(not np.array_equal(arrays[p][k], replay[p][k]) for p in PARTITIONS for k in arrays[p]):
            raise AssertionError("held-out label mutation changed input evidence")
        for p in PARTITIONS:
            p_out = seed_out / p
            p_out.mkdir(parents=True, exist_ok=True)
            np.save(p_out / "records.npy", records[p], allow_pickle=False)
            for key, filename in (("features", "prior_features"), ("raw_counts", "prior_raw_counts"),
                                  ("fallback", "prior_fallback"), ("query_coverage", "prior_query_coverage")):
                np.save(p_out / (filename + ".npy"), arrays[p][key], allow_pickle=False)
            if len(self_records[p]):
                self_out = seed_out / "self_pairs"
                self_out.mkdir(exist_ok=True)
                np.save(self_out / (p + ".npy"), self_records[p], allow_pickle=False)
        counts = {}
        for code, p in enumerate(PARTITIONS):
            selected_targets = np.flatnonzero(partition == code)
            observed_targets = np.unique(records[p]["target"])
            counts[p] = {"targets": len(selected_targets), "targets_with_strict_downstream_pairs": len(observed_targets),
                "targets_with_zero_strict_downstream_pairs": len(selected_targets) - len(observed_targets),
                "strict_pairs": len(records[p]), "class_counts": np.bincount(records[p]["label"], minlength=3).tolist(),
                "strict_self_pairs": len(self_records[p]), "full_axis_pairs": len(selected_targets) * expected_shape[1]}
        seed_report = {"status": "PASS", "schema": SCHEMA, "dataset": dataset, "seed": seed,
            "split_unit": "whole perturbation target identity", "split_ratio": [7, 1, 2],
            "partition_codes": {"train": 0, "val": 1, "test": 2},
            "assignment": "sort canonical target identities, seeded NumPy permutation; floor(.7N),floor(.1N),remainder",
            "response_independent_split": True, "target_partition_sha256": array_sha256(partition),
            "target_fold_sha256": array_sha256(fold), "zero_strict_targets_retained_in_axis": True,
            "exclude_self_pairs": exclude_self, "strict_label_codes": {"up": 0, "same": 1, "down": 2},
            "unresolved_ignore_index": -100, "counts": counts,
            "prior": prior_audit, "prior_feature_width": len(FEATURE_ORDER), "prior_feature_order": FEATURE_ORDER,
            "audit": {"partition_exact_cover": True, "target_overlap": False,
                "heldout_label_mutation_invariance": True, "train_same_fold_excluded": True,
                "train_same_fold_independent_replay": True,
                "self_target_excluded_from_crossfit": True, "cross_cell_response_inputs": False},
            "statistical_contract": label_manifest["statistical_contract"], "source": source}
        dump_json(seed_out / "manifest.json", seed_report)
        partition_hashes.append(seed_report["target_partition_sha256"])
        reports.append(seed_report)
    if len(set(partition_hashes)) != len(seeds):
        raise AssertionError("five target split assignments are not distinct")
    artifacts = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "preparation_report.json" and not path.name.endswith(".tmp"):
            artifacts[str(path.relative_to(out))] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    report = {"status": "PASS", "schema": SCHEMA, "dataset": dataset, "seeds": list(seeds),
        "source": source, "exclude_self_pairs": exclude_self, "target_count": expected_shape[0],
        "query_count": expected_shape[1], "total_strict_pairs_before_self_exclusion": int(mask.sum()),
        "full_grid_pairs": int(labels.size), "strict_coverage_before_self_exclusion": float(mask.mean()),
        "split_assignments_distinct": True, "mapping_audit": axes["mapping_audit"],
        "target_identity_alias_audit": axes["target_identity_alias_audit"],
        "duplicate_symbol_ids": axes["duplicate_symbol_ids"],
        "artifact_bytes": sum(v["size_bytes"] for v in artifacts.values()), "artifacts": artifacts,
        "split_reports": [{"seed": r["seed"], "counts": r["counts"]} for r in reports],
        "old_prediction_or_task_weights_read": False, "cross_cell_response_inputs": False}
    dump_json(complete_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(load_config(args.config), args.dataset, args.out)
    print(json.dumps({"status": report["status"], "dataset": args.dataset,
                      "out": str(args.out), "artifact_bytes": report["artifact_bytes"]}))


if __name__ == "__main__":
    main()
