#!/usr/bin/env python3
"""Map pinned public protein embeddings to stable target IDs using metadata only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_aliases(path):
    """The pinned upstream file uses only plain one-line symbol pairs."""
    aliases = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+):\s+([A-Za-z0-9_.-]+)\s*", line)
        if not match or match[1] in aliases:
            raise ValueError("unsupported or duplicate public alias entry")
        aliases[match[1]] = match[2]
    return aliases


def build(source, metadata_paths, output, alias_tsv=None):
    source, output = Path(source), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("ESM output exists; choose a new directory")
    info = json.loads((source / "meta.json").read_text())
    matrix = np.load(source / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    keys = info["keys"]
    if (len(set(keys)) != len(keys) or matrix.ndim != 2
            or matrix.shape != (len(keys), info["dim"]) or not np.issubdtype(matrix.dtype, np.floating)):
        raise ValueError("invalid public ESM key/embedding axes")
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite public ESM embeddings")
    index = {key: i for i, key in enumerate(keys)}
    aliases = read_aliases(source / "aliases.yaml")
    wanted, symbol_ids = {}, {}
    for path in metadata_paths:
        metadata = json.loads(Path(path).read_text())
        ids, symbols = metadata["target_ids"], metadata["target_symbols"]
        if len(ids) != len(symbols) or len(set(ids)) != len(ids):
            raise ValueError("invalid target metadata axes")
        for identity, symbol in zip(ids, symbols):
            if not isinstance(identity, str) or not isinstance(symbol, str):
                raise ValueError("target identifiers and symbols must be strings")
            wanted.setdefault(identity, set()).add(symbol)
            symbol_ids.setdefault(symbol, set()).add(identity)
    # Optional public identifier mappings must be explicit, never inferred from responses.
    if alias_tsv:
        with Path(alias_tsv).open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not {"gene_id", "symbol"}.issubset(reader.fieldnames or []):
                raise ValueError("alias TSV requires gene_id and symbol columns")
            for row in reader:
                if row["gene_id"] in wanted and row["symbol"]:
                    wanted[row["gene_id"]].add(row["symbol"])
                    symbol_ids.setdefault(row["symbol"], set()).add(row["gene_id"])
    selected, mapping, missing = [], [], []
    for identity in sorted(wanted):
        if identity in index:
            hits = {identity}
        else:
            symbols = {symbol for symbol in wanted[identity]
                       if symbol and not symbol.isdigit() and len(symbol_ids[symbol]) == 1}
            hits = {symbol if symbol in index else aliases[symbol]
                    for symbol in symbols if symbol in index or aliases.get(symbol) in index}
        if len(hits) > 1:
            raise ValueError(f"ambiguous public ESM mapping for {identity}")
        if not hits:
            missing.append(identity)
            continue
        key = next(iter(hits))
        selected.append(index[key])
        mapping.append({"target_id": identity, "public_key": key, "public_row": index[key]})
    if not selected:
        raise ValueError("no target identities match public ESM embeddings")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".esm2-map-", dir=output.parent) as temporary:
        stage = Path(temporary) / "mapped"
        stage.mkdir()
        np.save(stage / "embeddings.npy", np.asarray(matrix[selected], dtype=np.float32), allow_pickle=False)
        mapped = {"format_version": 1, "name": "esm2", "layout": "dense", "index": "pert",
                  "keys": [row["target_id"] for row in mapping], "dim": int(info["dim"]), "dtype": "float32",
                  "missing_target_ids": missing, "key_mapping": mapping,
                  "response_data_read": False, "task_weights_loaded": False,
                  "source_provenance": info.get("provenance", {}),
                  "source_hashes": {name: sha256(source / name) for name in ("meta.json", "embeddings.npy", "aliases.yaml")},
                  "target_metadata_sha256": [sha256(path) for path in metadata_paths],
                  "additional_alias_tsv_sha256": sha256(alias_tsv) if alias_tsv else None}
        (stage / "meta.json").write_text(json.dumps(mapped, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        stage.rename(output)
    return {"matched": len(selected), "missing": len(missing), "dim": int(info["dim"]),
            "response_data_read": False, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("assets/esm2_public"))
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--alias-tsv", type=Path, help="optional reviewed public gene_id/symbol mapping")
    parser.add_argument("--out", type=Path, default=Path("assets/esm2"))
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source, args.metadata, args.out, args.alias_tsv), indent=2))


if __name__ == "__main__":
    main()
