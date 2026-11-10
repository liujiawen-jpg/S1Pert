#!/usr/bin/env python3
"""Build the disk-friendly DECIDER perturbation classification store.

The store is deliberately factorized: expression moments and labels are dense
NumPy arrays (memory-mappable), while identities and provenance are Parquet /
JSON. Raw H5AD files are never modified. Replogle is streamed in HDF5
chunk-sized blocks; Nadig uses the packaged original raw-count aggregator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jev_a4.data.h5_metadata import text_array
from jev_a4.data import nadig_moments
from jev_a4.labels import label_core as LC

SEED = 20261006
SCALE = 10000.0

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def ident(p):
    s = p.stat(); return {"path": str(p.resolve()), "size_bytes": s.st_size,
                          "mtime_ns": s.st_mtime_ns, "inode": s.st_ino}

def split_for(x):
    v = int(hashlib.sha256(json.dumps([SEED, "global-intervention-target", x],
             separators=(",", ":")).encode()).hexdigest(), 16)
    return "TRAIN" if v * 100 < (1 << 256) * 70 else "CAL" if v * 100 < (1 << 256) * 85 else "EVAL"

def metadata_replogle(path):
    with h5py.File(path, "r") as f:
        if not isinstance(f["X"], h5py.Dataset) or f["X"].ndim != 2:
            raise ValueError("Replogle requires original dense cells x genes count X")
        obs, var = f["obs"], f["var"]
        perturb, _ = text_array(obs, "perturbation")
        gene_symbol = text_array(var, "gene_name")[0]
        gene_id = text_array(var, "ensembl_id")[0]
        umi_name = "UMI_count" if "UMI_count" in obs else "UMI count"
        umi = np.asarray(obs[umi_name][:], dtype=np.float64)
        batch = np.asarray(obs["batch"][:]) if "batch" in obs else None
        if (f["X"].shape != (len(perturb), len(gene_id)) or len(gene_symbol) != len(gene_id)
                or umi.shape != (len(perturb),) or len(set(gene_id)) != len(gene_id)
                or (umi <= 0).any() or not np.isfinite(umi).all()):
            raise ValueError("invalid gene axis or library-size metadata")
        sym_to_id = {}
        for s, i in zip(gene_symbol, gene_id):
            sym_to_id.setdefault(s, []).append(i)
        # Only exact, unique measured symbols are assigned a stable query ID.
        target_symbols = sorted({x for x in perturb if x not in ("control", "non-targeting", "")})
        targets, excluded = [], {}
        for s in target_symbols:
            ids = sym_to_id.get(s, [])
            if len(ids) == 1: targets.append((s, ids[0]))
            else: excluded[s] = "not_a_unique_measured_gene_symbol"
        targets = sorted(targets, key=lambda z: z[1])
        target_id = {s: i for s, i in targets}
        groups = np.full(len(perturb), -1, dtype=np.int32)
        # group 0 is the metadata-defined pooled control; target rows follow.
        control = np.isin(perturb, ["control", "non-targeting", ""])
        if control.sum() < 20:
            raise ValueError("at least 20 metadata-defined controls are required")
        groups[control] = 0
        ids = ["CONTROL"] + [i for _, i in targets]
        for j, (s, i) in enumerate(targets, 1): groups[perturb == s] = j
        if (groups < 0).any():
            excluded["unmapped_rows"] = int((groups < 0).sum())
        return {"n_cells": len(perturb), "n_genes": len(gene_id), "gene_id": gene_id,
                "gene_symbol": gene_symbol, "perturbation": perturb, "groups": groups,
                "group_ids": ids, "group_names": ["CONTROL"] + [s for s, _ in targets],
                "umi": umi, "batch": batch, "source": ident(path), "excluded": excluded,
                "shape": list(f["X"].shape), "x_chunks": list(f["X"].chunks or ())}

def write_table(path, frame):
    frame.to_parquet(path, index=False, engine="pyarrow")

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def init_store(out, name, gene_id, gene_symbol, group_ids, group_names, n_cells, group_n, source_info, mode):
    d = Path(out) / name; d.mkdir(parents=True, exist_ok=False)
    G, K = len(gene_id), len(group_ids)
    np.save(d / "group_n.npy", np.asarray(group_n, dtype=np.int64))
    np.save(d / "gene_id.npy", np.asarray(gene_id, dtype="U"))
    np.save(d / "gene_symbol.npy", np.asarray(gene_symbol, dtype="U"))
    np.save(d / "group_id.npy", np.asarray(group_ids, dtype="U"))
    np.save(d / "group_name.npy", np.asarray(group_names, dtype="U"))
    write_table(d / "genes.parquet", pd.DataFrame({"gene_index": np.arange(G, dtype=np.int32),
        "gene_id": gene_id, "gene_symbol": gene_symbol}))
    splits = [split_for(x) if x not in ("CONTROL", "non-targeting") else "CONTROL" for x in group_ids]
    write_table(d / "groups.parquet", pd.DataFrame({"group_index": np.arange(K, dtype=np.int32),
        "group_id": group_ids, "group_name": group_names, "split": splits,
        "n_cells": group_n}))
    manifest = {"schema": "decider.perturbation_classification.factorized.v1",
                "dataset": name, "mode": mode, "source": source_info,
                "normalization": "X / obs original UMI_count * 10000",
                "label_definition": {"ratio": 1.2, "alpha": .05, "q_method": "BH over 3G hypotheses per condition",
                                     "min_cells": 20, "min_nonzero": 10,
                                     "unknown_code": -1, "class_codes": {"up": 0, "unchanged": 1, "down": 2}},
                "split_seed": SEED,
                "split_metadata_role": "legacy 70/15/15 annotation only; A4 recomputes whole-target 70/10/20 for five seeds",
                "all_measured_genes_retained": True,
                "self_pairs_retained_in_labels": True, "outcomes_separate_from_inputs": True,
                "shape": {"groups_including_control": K, "genes": G, "source_cells": n_cells}}
    save_json(d / "manifest.json", manifest)
    return d, manifest

def allocate_stats(d, K, G):
    mean = np.lib.format.open_memmap(d / "mean.npy", mode="w+", dtype="float32", shape=(K, G))
    var = np.lib.format.open_memmap(d / "variance.npy", mode="w+", dtype="float32", shape=(K, G))
    nz = np.lib.format.open_memmap(d / "nonzero.npy", mode="w+", dtype="int32", shape=(K, G))
    return mean, var, nz

def labels_from_stats(d, mean, var, nz, group_n):
    K, G = mean.shape; y = np.lib.format.open_memmap(d / "labels.npy", mode="w+", dtype="int8", shape=(K-1, G));
    reason = np.lib.format.open_memmap(d / "reason_codes.npy", mode="w+", dtype="uint8", shape=(K-1, G))
    effect = np.lib.format.open_memmap(d / "effect_log2fc.npy", mode="w+", dtype="float32", shape=(K-1, G))
    q = np.lib.format.open_memmap(d / "q_best.npy", mode="w+", dtype="float32", shape=(K-1, G))
    ctl = LC.Moments(int(group_n[0]), mean[0].astype(np.float64), var[0].astype(np.float64), nz[0].astype(np.float64))
    for i in range(1, K):
        # The declared label contract requires at least 20 cells per group.
        # Avoid calling the reference implementation with n<2; those rows are
        # unknown for the same reason and retain an explicit reason code.
        if int(group_n[i]) < 20:
            y[i-1] = -1
            reason[i-1] = 1
            effect[i-1] = np.nan
            q[i-1] = 1.0
            continue
        out = LC.label_pair(LC.Moments(int(group_n[i]), mean[i].astype(np.float64), var[i].astype(np.float64), nz[i].astype(np.float64)), ctl)
        y[i-1] = out["labels"]; reason[i-1] = out["reason_codes"]; effect[i-1] = out["effect_log2fc"].astype(np.float32)
        q[i-1] = np.minimum(np.minimum(out["q_up"], out["q_equivalent"]), out["q_down"]).astype(np.float32)
    for a in (y, reason, effect, q): a.flush()
    np.save(d / "response_group_id.npy", np.load(d / "group_id.npy", allow_pickle=False)[1:])
    return {"label_counts": {str(code): int((y == code).sum()) for code in (-1, 0, 1, 2)}}

def process_nadig(name, path, out, meta, block_cells=256):
    stats = nadig_moments.aggregate_moments(path, meta, block_cells)
    group_ids = stats["identities"]
    group_names = [nadig_moments.CONTROL] + [meta["target_names"][g] for g in group_ids[1:]]
    d, m = init_store(out, name, meta["gene_ids"], meta["genes"], group_ids, group_names,
        meta["shape"][0], stats["n"], {"raw": meta["source"], "raw_scan_audit": stats["audit"]},
        "streamed_h5ad_original_nadig_aggregation")
    mm, vv, zz = allocate_stats(d, len(group_ids), len(meta["gene_ids"]))
    mm[:] = stats["mean"]; vv[:] = stats["variance"]; zz[:] = stats["nonzero"]
    mm.flush(); vv.flush(); zz.flush()
    m["identity_audit"] = {"metadata_sha256": meta["metadata_sha256"],
        "canonical_target_ids_sha256": meta["canonical_target_ids_sha256"],
        "missing_source_target_id_resolution": meta["missing_source_target_id_resolution"],
        "cross_source_information": "gene identities and symbols only; no response values"}
    info = labels_from_stats(d, mm, vv, zz, stats["n"]); m.update(info); save_json(d / "manifest.json", m)
    return m

def process_replogle(name, path, out):
    meta = metadata_replogle(path); G = meta["n_genes"]; K = len(meta["group_ids"])
    group_counts = np.bincount(meta["groups"][meta["groups"] >= 0], minlength=K)
    d, m = init_store(out, name, meta["gene_id"], meta["gene_symbol"], meta["group_ids"], meta["group_names"], meta["n_cells"], group_counts, meta["source"], "streamed_h5ad")
    mean, var, nz = allocate_stats(d, K, G)
    with h5py.File(path, "r") as f:
        X, chunks = f["X"], meta["x_chunks"]; row_step = chunks[0] if chunks else 1024; col_step = max(chunks[1] * 4 if chunks else 256, 64)
        groups, umi = meta["groups"], meta["umi"]; row_total = np.zeros(meta["n_cells"], dtype=np.float64); raw_nonneg = True
        for g0 in range(0, G, col_step):
            g1 = min(G, g0 + col_step); s = np.zeros((K, g1-g0), dtype=np.float64); ss = np.zeros_like(s); z = np.zeros((K, g1-g0), dtype=np.int64)
            for r0 in range(0, meta["n_cells"], row_step):
                r1 = min(meta["n_cells"], r0 + row_step); raw = np.asarray(X[r0:r1, g0:g1], dtype=np.float64)
                if not np.isfinite(raw).all() or (raw < 0).any() or not np.allclose(raw, np.rint(raw), atol=0, rtol=0): raw_nonneg = False
                row_total[r0:r1] += raw.sum(axis=1); values = raw / umi[r0:r1, None] * SCALE; gi = groups[r0:r1]; valid = gi >= 0
                np.add.at(s, gi[valid], values[valid]); np.add.at(ss, gi[valid], values[valid] * values[valid]); np.add.at(z, gi[valid], (raw[valid] > 0).astype(np.int64))
            mean[:, g0:g1] = (s / np.maximum(1, group_counts[:,None])).astype(np.float32)
            den = np.maximum(1, group_counts - 1)[:,None]
            var[:, g0:g1] = np.maximum(0, (ss - s*s/np.maximum(1,group_counts[:,None])) / den).astype(np.float32)
            nz[:, g0:g1] = z.astype(np.int32)
        if not raw_nonneg or np.any(row_total > umi + 1e-5): raise ValueError("X is not a valid integer count matrix under UMI metadata")
    if ident(path) != meta["source"]:
        raise ValueError("raw file changed during response scan")
    mean.flush(); var.flush(); nz.flush(); info = labels_from_stats(d, mean, var, nz, group_counts); m.update(info); m["raw_count_validation"] = {"integer_nonnegative": True, "row_sum_le_original_UMI": True, "unmapped_rows_excluded": int((meta["groups"] < 0).sum())}; m["excluded"] = meta["excluded"]; save_json(d / "manifest.json", m); return m

def audit_adamson(path, out, name="adamson_10x010_audit"):
    d = Path(out) / name; d.mkdir(parents=True, exist_ok=False)
    with h5py.File(path, "r") as f:
        p, codes = text_array(f["obs"], "perturbation"); nperts = np.asarray(f["obs"]["nperts"][:]); umi = np.asarray(f["obs"]["UMI count"][:]);
        genes = text_array(f["var"], "gene_symbol")[0]; ens = text_array(f["var"], "ensembl_id")[0]
        if codes is None:
            raise ValueError("Adamson audit requires encoded missing perturbation categories")
        good = (codes < 0) & (nperts == 0); names = sorted(set(p[(~good) & (~np.isin(p, ["*", "Gal4-4(mod)"]))]))
        xshape = list(f["X/shape"][:]) if "shape" in f["X"] else [len(p), len(ens)]
        save_json(d / "manifest.json", {"schema":"decider.perturbation_classification.audit.v1","status":"AUDITED_NOT_INCLUDED","source":ident(path),"shape":xshape,"controls_nperts_zero_missing_perturbation":int(good.sum()),"control_identity_verified":bool(np.array_equal(codes < 0, nperts == 0)),"special_or_vector_controls_excluded":["*","Gal4-4(mod)","63(mod)"],"all_measured_genes_retained":True,"eligible_single_target_symbols":names,"normalization_denominator":"obs UMI count"})
        pd.DataFrame({"gene_index":np.arange(len(ens),dtype=np.int32),"gene_id":ens,"gene_symbol":genes}).to_parquet(d/"genes.parquet",index=False)
    return {"status":"AUDITED_NOT_INCLUDED", "controls":int(good.sum()), "eligible_symbols":len(names)}

def build(cfg, output=None):
    """Build the complete configured source bundle without historical caches."""
    settings = cfg.get("preprocess", {})
    destinations = output or settings.get("output") or cfg.get("data_preprocessed_root")
    if not destinations:
        raise ValueError("set preprocess.output or data_preprocessed_root, or pass --output")
    out = Path(destinations)
    datasets = settings.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("preprocess.datasets must map dataset names to {kind, raw}")
    for name, spec in datasets.items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("dataset names must be single path components")
        if spec.get("kind") not in {"nadig", "replogle", "adamson_audit"}:
            raise ValueError(f"unsupported preprocessing kind: {name}")
        if not spec.get("raw") or not Path(spec["raw"]).is_file():
            raise FileNotFoundError(f"missing raw H5AD for {name}: {spec.get('raw')}")
    if out.exists():
        raise FileExistsError(f"use a new output directory; refusing to overwrite {out}")
    block_cells = settings.get("block_cells", 256)
    if type(block_cells) is not int or not 1 <= block_cells <= 1024:
        raise ValueError("block_cells must be an integer between 1 and 1024")
    nadig_paths = {name: Path(spec["raw"]) for name, spec in datasets.items() if spec["kind"] == "nadig"}
    metadata = nadig_moments.inventory(nadig_paths)
    out.mkdir(parents=True, exist_ok=False)
    started, results = time.time(), {}
    save_json(out / "status.json", {"status": "RUNNING"})
    try:
        for name, spec in datasets.items():
            path = Path(spec["raw"])
            if spec["kind"] == "nadig":
                results[name] = process_nadig(name, path, out, metadata[name], block_cells)
            elif spec["kind"] == "replogle":
                results[name] = process_replogle(name, path, out)
            else:
                results[name] = audit_adamson(path, out, name)
        save_json(out / "manifest.json", {"schema": "decider.perturbation_classification.bundle.v1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol": "docs/DATA_PIPELINE.md", "datasets": results,
            "total_seconds": time.time() - started,
            "read_contract": "mmap mean/variance/nonzero/labels; Parquet identities; labels separate from model inputs"})
        save_json(out / "status.json", {"status": "PASS"})
    except Exception:
        save_json(out / "status.json", {"status": "FAILED"})
        raise
    return {"output": str(out), "datasets": list(results), "seconds": time.time() - started}


def main(argv=None):
    from jev_a4.config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="override configured output directory")
    args = parser.parse_args(argv)
    print(json.dumps(build(load_config(args.config), args.output), ensure_ascii=False))

if __name__ == "__main__": main()
