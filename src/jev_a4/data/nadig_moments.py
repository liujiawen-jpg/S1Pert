"""Portable Nadig metadata and raw-count moments, with no model dependencies.

The ``inventory`` and ``aggregate_moments`` routines preserve the source
metadata and descriptive aggregation rules used for the paper's Nadig moments.
See docs/DATA_PIPELINE.md for provenance and numerical reproduction limits.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from jev_a4.data.h5_metadata import text_array

CONTROL = "non-targeting"
MISSING_IDS = {"", "nan", "none", "na", "null"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def inventory(raw_paths):
    """Read annotations only; resolve missing target IDs across supplied sources.

    Cross-source information is restricted to gene identities and symbols, never
    expression or labels. Supply both Nadig sources for historical alias rules.
    """
    metadata, aliases_by_id, measured_symbols_by_id = {}, {}, {}
    for cell, path in raw_paths.items():
        before = source_identity(path)
        with h5py.File(path, "r") as handle:
            if not isinstance(handle["X"], h5py.Dataset) or len(handle["X"].shape) != 2:
                raise ValueError("Nadig requires original dense cells x genes count X")
            obs, var = handle["obs"], handle["var"]
            target_names = text_array(obs, "gene")[0]
            raw_target_ids = text_array(obs, "gene_id")[0]
            gene_names = text_array(var, "gene_name")[0]
            gene_ids = text_array(var, "gene_id")[0]
            symbol_axes = {}
            for name, identity in zip(gene_names, gene_ids):
                symbol_axes.setdefault(str(name), set()).add(str(identity))
            fallback, target_ids = {}, []
            for name, identity in zip(target_names, raw_target_ids):
                if identity.strip().lower() in MISSING_IDS:
                    choices = symbol_axes.get(str(name), set())
                    identity = next(iter(choices)) if len(choices) == 1 else "SYMBOL:" + str(name)
                    fallback[str(name)] = {"resolved_target_id": str(identity),
                        "rule": "unique measured var symbol" if len(choices) == 1 else "explicit symbol fallback"}
                target_ids.append(str(identity))
            target_ids = np.asarray(target_ids, dtype=str)
            barcodes = text_array(obs, "cell_barcode")[0]
            totals = np.asarray(obs["UMI_count"][:], dtype=np.float64)
            shape = handle["X"].shape
            if (shape != (len(target_names), len(gene_ids)) or len(target_ids) != shape[0]
                    or len(barcodes) != shape[0] or len(totals) != shape[0] or len(gene_names) != shape[1]):
                raise ValueError("raw metadata axes disagree")
            if (len(set(gene_ids)) != len(gene_ids) or len(set(barcodes)) != len(barcodes)
                    or any(g.strip().lower() in MISSING_IDS for g in gene_ids)
                    or not np.isfinite(totals).all() or (totals <= 0).any()):
                raise ValueError("duplicate/missing stable axes or invalid original UMI denominator")
            counts, aliases = Counter(map(str, target_ids)), {}
            for name, identity in zip(target_names, target_ids):
                if not name or not identity or (name == CONTROL) != (identity == CONTROL):
                    raise ValueError("invalid intervention/control identity")
                aliases.setdefault(str(identity), set()).add(str(name))
            if counts[CONTROL] < 20:
                raise ValueError("at least 20 original non-targeting controls are required")
            targets = sorted(identity for identity in counts if identity != CONTROL)
            for identity in targets:
                if any(delimiter in identity for delimiter in ("+", ",", ";", "|")):
                    raise ValueError("multi-target identity is outside the single-intervention contract")
                aliases_by_id.setdefault(identity, set()).update(aliases[identity])
            for identity, name in zip(gene_ids, gene_names):
                measured_symbols_by_id.setdefault(str(identity), set()).add(str(name))
            metadata[cell] = {"source": before, "shape": list(shape), "genes": gene_names,
                "gene_ids": gene_ids, "target_ids": target_ids,
                "target_aliases": {identity: sorted(values) for identity, values in aliases.items()},
                "missing_source_target_id_resolution": fallback, "targets": targets,
                "counts": counts, "totals": totals, "barcodes": barcodes,
                "metadata_sha256": digest({"target_ids": target_ids.tolist(), "barcodes": barcodes.tolist(),
                    "raw_target_ids": raw_target_ids.tolist(), "target_symbols": target_names.tolist(),
                    "genes": gene_ids.tolist(), "symbols": gene_names.tolist(), "umi": totals.tolist()})}
        if source_identity(path) != before:
            raise ValueError("raw file changed during metadata lock")
    by_symbol = {}
    for identity, aliases in aliases_by_id.items():
        if not identity.startswith("SYMBOL:"):
            for name in aliases:
                by_symbol.setdefault(name, set()).add(identity)
    replacements = {}
    for identity in aliases_by_id:
        if identity.startswith("SYMBOL:"):
            choices = by_symbol.get(identity[len("SYMBOL:"):], set())
            if len(choices) > 1:
                raise ValueError("ambiguous cross-cell missing target ID")
            if len(choices) == 1:
                replacements[identity] = next(iter(choices))
    for old, new in replacements.items():
        aliases_by_id.setdefault(new, set()).update(aliases_by_id.pop(old))
    if replacements:
        for item in metadata.values():
            item["target_ids"] = np.asarray([replacements.get(t, t) for t in item["target_ids"]], dtype=str)
            item["counts"] = Counter(map(str, item["target_ids"]))
            item["targets"] = sorted(t for t in item["counts"] if t != CONTROL)
            merged = {}
            for identity, aliases in item["target_aliases"].items():
                merged.setdefault(replacements.get(identity, identity), set()).update(aliases)
            item["target_aliases"] = {identity: sorted(aliases) for identity, aliases in merged.items()}
            for resolution in item["missing_source_target_id_resolution"].values():
                old = resolution["resolved_target_id"]
                if old in replacements:
                    resolution.update(resolved_target_id=replacements[old], rule="unique cross-cell obs symbol mapping")
    symbols_by_id = {identity: sorted(measured_symbols_by_id.get(identity, aliases))[0]
                     for identity, aliases in aliases_by_id.items()}
    for item in metadata.values():
        item["target_names"] = {identity: symbols_by_id[identity] for identity in item["targets"]}
        item["canonical_target_ids_sha256"] = digest(item["target_ids"].tolist())
    return metadata


def aggregate_moments(path, meta, block_cells=256):
    """One sequential dense-X scan; original float64 group reductions."""
    if type(block_cells) is not int or not 1 <= block_cells <= 1024:
        raise ValueError("block_cells must be an integer between 1 and 1024")
    identities = [CONTROL] + meta["targets"]
    index = {target: i for i, target in enumerate(identities)}
    row_groups = np.asarray([index[t] for t in meta["target_ids"]], dtype=np.int32)
    n_groups, n_genes = len(identities), len(meta["gene_ids"])
    sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    squares = np.zeros_like(sums)
    nonzero = np.zeros((n_groups, n_genes), dtype=np.int64)
    n = np.zeros(n_groups, dtype=np.int64)
    started, blocks, rows_seen = time.monotonic(), 0, 0
    count_digest = hashlib.sha256(meta["metadata_sha256"].encode())
    with h5py.File(path, "r") as handle:
        for start in range(0, len(row_groups), block_cells):
            stop = min(start + block_cells, len(row_groups))
            raw = np.asarray(handle["X"][start:stop, :], dtype=np.float64)
            totals = meta["totals"][start:stop]
            if (not np.isfinite(raw).all() or (raw < 0).any() or not np.equal(raw, np.rint(raw)).all()
                    or (raw.sum(axis=1) > totals + 1e-6).any()):
                raise ValueError("raw integer counts / original UMI normalization contract failed")
            count_digest.update(np.asarray([start, stop, n_genes], dtype="<i8").tobytes())
            count_digest.update(raw.astype("<f8", copy=False).tobytes(order="C"))
            values = raw * (10000. / totals[:, None])
            groups = row_groups[start:stop]
            order = np.argsort(groups, kind="stable")
            sorted_groups = groups[order]
            boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
            unique = sorted_groups[boundaries]
            sorted_values = values[order]
            sums[unique] += np.add.reduceat(sorted_values, boundaries, axis=0)
            squares[unique] += np.add.reduceat(np.square(sorted_values), boundaries, axis=0)
            nonzero[unique] += np.add.reduceat((raw[order] > 0).astype(np.int64), boundaries, axis=0)
            n[unique] += np.diff(np.r_[boundaries, len(groups)])
            rows_seen += stop - start
            blocks += 1
    if rows_seen != len(row_groups) or n.tolist() != [meta["counts"][i] for i in identities]:
        raise ValueError("scan did not cover every source cell exactly once")
    if source_identity(path) != meta["source"]:
        raise ValueError("raw file changed during response scan")
    mean = sums / n[:, None]
    variance = np.zeros_like(sums)
    valid = n >= 2
    numerator = squares[valid] - np.square(sums[valid]) / n[valid, None]
    if (numerator < -1e-8 * np.maximum(squares[valid], 1.)).any():
        raise FloatingPointError("negative response variance beyond rounding tolerance")
    variance[valid] = np.maximum(numerator, 0.) / (n[valid, None] - 1)
    return {"identities": identities, "n": n, "mean": mean, "variance": variance, "nonzero": nonzero,
        "audit": {"cells_read": rows_seen, "blocks": blocks, "seconds": time.monotonic() - started,
            "all_source_cells_used": True, "target_random_rescans": 0,
            "normalization": "raw integer X / original obs.UMI_count * 10000",
            "raw_count_logical_sha256": count_digest.hexdigest(),
            "logical_digest_encoding": "metadata_sha256 then each [start,stop,G] little-endian int64 plus row-major count float64 bytes",
            "group_counts_sha256": digest(dict(zip(identities, n.tolist())))}}
