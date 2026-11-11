"""Exact response-blind MAP/OmniPath edge and HGNC/NCBI bridge definitions."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd

ENSEMBL_RE = re.compile(r"^(ENS[A-Z]*G\d+)(?:\.\d+)?$")


def normalize_ensembl(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    match = ENSEMBL_RE.fullmatch(value.strip())
    return match.group(1) if match else None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_array_digest(items: list[Any] | tuple[Any, ...]) -> str:
    raw = json.dumps(list(items), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

def _iter_tsv_gzip(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames and reader.fieldnames[0].startswith("#"):
            first = reader.fieldnames[0]
            reader.fieldnames[0] = first[1:]
        yield from reader


def _human_gene_to_ensembl(path: Path) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str, str]] = set()
    stats = Counter()
    for row in _iter_tsv_gzip(path):
        tax = str(row.get("tax_id", row.get("#tax_id", ""))).strip()
        if tax != "9606":
            continue
        stats["human_rows"] += 1
        gene_id = str(row.get("GeneID", "")).strip()
        stable = normalize_ensembl(row.get("Ensembl_gene_identifier"))
        if not gene_id or not stable:
            stats["invalid_or_missing"] += 1
            continue
        triple = (tax, gene_id, stable)
        if triple in seen:
            stats["duplicate_gene_tuple"] += 1
            continue
        seen.add(triple)
        grouped[gene_id].add(stable)
    mapping = {key: next(iter(values)) for key, values in grouped.items() if len(values) == 1}
    ambiguous = {key: sorted(values) for key, values in grouped.items() if len(values) > 1}
    stats["one_to_one_gene_ids"] = len(mapping)
    stats["ambiguous_gene_ids"] = len(ambiguous)
    return mapping, ambiguous, dict(stats)


def _split_multi(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [token.strip() for token in re.split(r"[|,;]", str(value)) if token.strip() and token.strip() != "-"]


def _unique_map(pairs: Iterable[tuple[str, str]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for raw_key, raw_stable in pairs:
        stable = normalize_ensembl(raw_stable)
        key = str(raw_key).strip()
        if key and stable:
            grouped[key].add(stable)
    return (
        {key: next(iter(values)) for key, values in grouped.items() if len(values) == 1},
        {key: sorted(values) for key, values in grouped.items() if len(values) > 1},
    )


def _hgnc_maps(path: Path, geneid_map: dict[str, str]) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    columns = {column.lower(): column for column in frame.columns}
    status_col = columns.get("status")
    symbol_col = columns.get("symbol")
    ensembl_col = columns.get("ensembl_gene_id")
    entrez_col = columns.get("entrez_id")
    uniprot_col = columns.get("uniprot_ids")
    if not all([status_col, symbol_col, ensembl_col]):
        raise ValueError(f"HGNC required columns absent: {frame.columns.tolist()}")
    frame = frame.loc[frame[status_col].str.lower().eq("approved")].copy()
    symbol_pairs: list[tuple[str, str]] = []
    entrez_pairs: list[tuple[str, str]] = []
    uniprot_pairs: list[tuple[str, str]] = []
    conflict = 0
    for row in frame.to_dict("records"):
        stable = normalize_ensembl(row.get(ensembl_col))
        if not stable and entrez_col:
            stable = geneid_map.get(str(row.get(entrez_col, "")).strip())
        if not stable:
            continue
        if entrez_col:
            entrez = str(row.get(entrez_col, "")).strip()
            bridged = geneid_map.get(entrez)
            if bridged and bridged != stable:
                conflict += 1
                continue
            if entrez:
                entrez_pairs.append((entrez, stable))
        symbol_pairs.append((str(row.get(symbol_col, "")).strip(), stable))
        if uniprot_col:
            uniprot_pairs.extend((item, stable) for item in _split_multi(row.get(uniprot_col)))
    result: dict[str, dict[str, str]] = {}
    exclusions: dict[str, list[dict[str, Any]]] = {}
    for name, pairs in [("symbol", symbol_pairs), ("entrez", entrez_pairs), ("uniprot", uniprot_pairs)]:
        mapping, ambiguous = _unique_map(pairs)
        result[name] = mapping
        exclusions[name] = [{"raw_id": key, "candidate_stable_ids": values, "reason": "AMBIGUOUS_ONE_TO_MANY"} for key, values in sorted(ambiguous.items())]
    return result, exclusions, {
        "approved_rows": int(len(frame)),
        "symbol_one_to_one": len(result["symbol"]),
        "entrez_one_to_one": len(result["entrez"]),
        "uniprot_one_to_one": len(result["uniprot"]),
        "hgnc_ncbi_conflict_excluded": conflict,
        "ambiguous": {key: len(value) for key, value in exclusions.items()},
    }


def _norm_relation(value: Any) -> str:
    relation = str(value).strip().lower()
    # MAP embeds a continuously-valued confidence in many otherwise identical
    # relation labels.  Confidence is retained in raw_relations_json, but is
    # deliberately excluded from typed-edge identity.
    relation = re.sub(r"\s*\(with confidence\s*:\s*[^)]*\)\s*", " ", relation)
    relation = re.sub(r"\bconfidence\s*[:=]?\s*\(?\s*[0-9]+(?:\.[0-9]+)?\s*\)?", "confidence", relation)
    relation = re.sub(r"\([^)]*\b[0-9]+(?:\.[0-9]+)?\b[^)]*\)", "", relation)
    relation = re.sub(r"[^a-z0-9]+", "_", relation).strip("_")
    return relation or "unspecified"


def _typed_edge(u: str, v: str, directed: bool, relation: str) -> tuple[str, str, str]:
    if not directed and v < u:
        u, v = v, u
    edge_id = stable_hash(["typed_edge_v1", u, v, "directed" if directed else "undirected", relation])
    return u, v, edge_id


def _edge_accumulate(
    accumulator: dict[str, dict[str, Any]], u: str, v: str, directed: bool, relation: str,
    source: str, raw_relation: str, sign: str, references: Iterable[str]
) -> None:
    u, v, edge_id = _typed_edge(u, v, directed, relation)
    if u == v:
        raise ValueError("self loop must be filtered before accumulation")
    if edge_id not in accumulator:
        accumulator[edge_id] = {
            "edge_id": edge_id, "u": u, "v": v, "directed": bool(directed), "relation": relation,
            "sources": set(), "raw_relations": set(), "signs": set(), "references": set(), "evidence_rows": 0,
        }
    item = accumulator[edge_id]
    item["sources"].add(source)
    item["raw_relations"].add(raw_relation)
    item["signs"].add(sign)
    item["references"].update(value for value in references if value)
    item["evidence_rows"] += 1


def _resolve_omnipath(raw: Any, symbol: Any, maps: dict[str, dict[str, str]]) -> tuple[str | None, str]:
    stable = normalize_ensembl(raw)
    if stable:
        return stable, "ensembl_direct"
    raw_key = str(raw).strip()
    if raw_key in maps["uniprot"]:
        return maps["uniprot"][raw_key], "hgnc_uniprot"
    if raw_key in maps["entrez"]:
        return maps["entrez"][raw_key], "hgnc_entrez"
    symbol_key = str(symbol).strip()
    if symbol_key in maps["symbol"]:
        return maps["symbol"][symbol_key], "hgnc_approved_symbol"
    return None, "unmapped"


def _build_edges(paths: dict[str, Path], maps: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    accumulator: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    stats = Counter()
    relation_counts: dict[str, Counter] = {"MAP": Counter(), "OmniPath": Counter()}
    map_frame = pd.read_csv(paths["map"], dtype=str)
    required = ["Gene stable ID 1", "Gene stable ID 2", "relation"]
    if any(column not in map_frame.columns for column in required):
        raise ValueError(f"MAP schema mismatch: {map_frame.columns.tolist()}")
    for left, right, raw_relation in map_frame[required].itertuples(index=False, name=None):
        stats["map_rows"] += 1
        u, v = normalize_ensembl(left), normalize_ensembl(right)
        relation_counts["MAP"][str(raw_relation)] += 1
        if not u or not v:
            stats["map_invalid_id"] += 1
            continue
        if u == v:
            stats["map_self_loop"] += 1
            continue
        relation = _norm_relation(raw_relation)
        _edge_accumulate(accumulator, u, v, False, relation, "MAP", str(raw_relation), "unknown", [])
    del map_frame

    omni = pd.read_csv(paths["omnipath"], sep="\t", dtype=str, low_memory=False)
    lower = {column.lower(): column for column in omni.columns}
    if "source" not in lower or "target" not in lower:
        raise ValueError(f"OmniPath schema mismatch: {omni.columns.tolist()}")
    src_symbol_col = lower.get("source_genesymbol") or lower.get("sourcesymbol")
    dst_symbol_col = lower.get("target_genesymbol") or lower.get("targetsymbol")
    for row in omni.to_dict("records"):
        stats["omnipath_rows"] += 1
        raw_u, raw_v = row.get(lower["source"]), row.get(lower["target"])
        u, u_method = _resolve_omnipath(raw_u, row.get(src_symbol_col, "") if src_symbol_col else "", maps)
        v, v_method = _resolve_omnipath(raw_v, row.get(dst_symbol_col, "") if dst_symbol_col else "", maps)
        if not u or not v:
            stats["omnipath_unmapped_endpoint"] += 1
            if len(exclusions) < 10000:
                exclusions.append({"source": raw_u, "target": raw_v, "reason": "UNMAPPED_ENDPOINT", "source_method": u_method, "target_method": v_method})
            continue
        if u == v:
            stats["omnipath_self_loop"] += 1
            continue
        directed_col = lower.get("is_directed")
        directed = parse_bool(row.get(directed_col)) if directed_col else True
        raw_relation = str(row.get(lower.get("type"), "interaction"))
        relation = _norm_relation(raw_relation)
        stim = parse_bool(row.get(lower.get("is_stimulation"))) if lower.get("is_stimulation") else False
        inhib = parse_bool(row.get(lower.get("is_inhibition"))) if lower.get("is_inhibition") else False
        sign = "conflict" if stim and inhib else "positive" if stim else "negative" if inhib else "unknown"
        sources = _split_multi(row.get(lower.get("sources"), "OmniPath")) or ["OmniPath"]
        refs = _split_multi(row.get(lower.get("references"), ""))
        relation_counts["OmniPath"][raw_relation] += 1
        for source in sources:
            _edge_accumulate(accumulator, u, v, directed, relation, f"OmniPath:{source}", raw_relation, sign, refs)
    del omni

    rows = []
    for edge_id in sorted(accumulator):
        item = accumulator[edge_id]
        signs = sorted(item.pop("signs"))
        sources = sorted(item.pop("sources"))
        raw_relations = sorted(item.pop("raw_relations"))
        references = sorted(item.pop("references"))
        biological_signs = {value for value in signs if value in {"positive", "negative", "conflict"}}
        sign_conflict = "conflict" in biological_signs or {"positive", "negative"}.issubset(biological_signs)
        rows.append({
            **item,
            "source_category": "MAP+OmniPath" if "MAP" in sources and any(v.startswith("OmniPath:") for v in sources) else "MAP" if "MAP" in sources else "OmniPath",
            "sources_json": json.dumps(sources, ensure_ascii=False, separators=(",", ":")),
            "raw_relations_json": json.dumps(raw_relations, ensure_ascii=False, separators=(",", ":")),
            "signs_json": json.dumps(signs, separators=(",", ":")),
            "references_json": json.dumps(references, ensure_ascii=False, separators=(",", ":")),
            "sign_conflict": sign_conflict,
        })
    edge_frame = pd.DataFrame(rows).sort_values("edge_id", kind="mergesort").reset_index(drop=True)
    exclusion_frame = pd.DataFrame(exclusions, columns=["source", "target", "reason", "source_method", "target_method"])
    stats.update({
        "canonical_edges": len(edge_frame),
        "directed_edges": int(edge_frame["directed"].sum()),
        "undirected_edges": int((~edge_frame["directed"]).sum()),
        "sign_conflicts": int(edge_frame["sign_conflict"].sum()),
        "duplicate_evidence_rows_merged": int(edge_frame["evidence_rows"].sum() - len(edge_frame)),
    })
    return edge_frame, exclusion_frame, {"counts": dict(stats), "raw_relation_counts": {k: dict(v) for k, v in relation_counts.items()}}
