#!/usr/bin/env python3
"""Build ALIGNED's public-only SQLite inputs from raw public snapshots.

No downloads, response data, checkpoints or training are used by this command.
Outputs: WORK/assets/public_knowledge.sqlite (genes, public_edges),
WORK/assets/public_go.sqlite (terms, gene_annotations), and a provenance receipt.

Fetch explicitly, then build (run from the repository root):
  python scripts/download_assets.py --manifest data_prep/public_knowledge/sources.json --root WORK --group public_knowledge --dry-run
  python scripts/download_assets.py --manifest data_prep/public_knowledge/sources.json --root WORK --group public_knowledge
  python data_prep/build_public_knowledge.py --work-dir WORK

Historical hashes are checked by default. Live endpoints may no longer serve
those bytes. For an explicitly different, refreshed input protocol, first write
a separate unpinned download manifest with --refresh-download-manifest FILE,
use it with the downloader above, then build with --allow-new-snapshots. The
receipt retains actual hashes and never labels refreshed inputs historical.
Use a new workspace for refreshed snapshots; existing outputs are not replaced.

The 2026-10-02 inventory used GO 2026-08-05 and UniProt 2026_03. Reactome
membership was reviewed-human UniProt cross-references, not a full export.
Unsigned legacy measured-axis GO/STRING edges are omitted because ALIGNED
discards all unsigned edges. Signed rows, mappings and GO annotations retain
the original parsers. This is an ALIGNED input view, not the full historical KB.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_prep.public_knowledge import public_import
from data_prep.public_knowledge.signed_records import parse_omnipath

MANIFEST = Path(__file__).with_name("public_knowledge") / "sources.json"
SOURCES = ("GO", "STRING", "Reactome", "OmniPath-core", "OmniPath/CollecTRI", "OmniPath/SIGNOR")
SIGNED_SQL = ("SELECT subject,object,sign,directed,source FROM public_edges "
              "WHERE directed=1 AND sign IN ('positive','negative','conflict') "
              "ORDER BY subject,object,sign,directed,source")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked_sources(source_dir, *, allow_new_snapshots=False):
    manifest = json.loads(MANIFEST.read_text())
    paths, identities = {}, {}
    for asset in manifest["assets"]:
        key = asset["id"]
        path = Path(source_dir) / Path(asset["path"]).name
        if not path.is_file():
            raise FileNotFoundError(f"missing public snapshot: {path}")
        actual = sha256(path)
        match = (actual == asset["checksums"]["sha256"]
                 and path.stat().st_size == asset["size_bytes"])
        if not match and not allow_new_snapshots:
            raise ValueError(f"historical snapshot mismatch: {key}; use --allow-new-snapshots only for an explicitly refreshed protocol")
        paths[key] = path
        identities[key] = {"filename": path.name, "sha256": actual,
                           "size_bytes": path.stat().st_size,
                           "matches_historical_snapshot": match,
                           "historical_sha256": asset["checksums"]["sha256"]}
    return paths, identities, manifest


def derive_membership(pathways, xrefs, output):
    """Preserve the original reviewed-human fallback derivation, including order."""
    names = {}
    with Path(pathways).open() as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) >= 3 and row[2] == "Homo sapiens":
                names[row[0]] = row[1]
    rows = 0
    with Path(xrefs).open() as stream, Path(output).open("x") as destination:
        for record in csv.DictReader(stream, delimiter="\t"):
            for pathway in record.get("Reactome", "").split(";"):
                pathway = pathway.strip()
                if pathway in names:
                    destination.write("\t".join([record["Entry"], pathway,
                        "https://reactome.org/content/detail/" + pathway, names[pathway],
                        "UniProtKB_reviewed_cross_reference", "Homo sapiens"]) + "\n")
                    rows += 1
    return {"rows": rows, "sha256": sha256(output),
            "scope": "reviewed-human UniProt Reactome cross-references only"}


def canonical_signed_records(paths, output):
    """Original strict source parsers -> original legacy-record importer.

    Preserve the original spelling 'conflicting': _legacy maps that spelling to
    unknown, unlike the independent OmniPath-core parser's 'conflict'. Changing
    this would alter historical ALIGNED graph inputs and is not a portability fix.
    """
    records, audits = {}, {}
    for dataset in ("collectri", "signor"):
        path = paths[dataset + "_raw"]
        edges, audit = parse_omnipath(path.read_text(), dataset=dataset, source_sha256=sha256(path))
        records.update((edge.uid, edge) for edge in edges)
        audits[dataset] = audit
    with Path(output).open("x") as stream:
        for uid in sorted(records):
            stream.write(json.dumps(asdict(records[uid]), sort_keys=True) + "\n")
    return audits


def rows_digest(connection, query):
    digest = hashlib.sha256()
    count = 0
    for row in connection.execute(query):
        digest.update((json.dumps(list(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        count += 1
    return {"rows": count, "sha256": digest.hexdigest()}


def export_view(source, path, tables):
    with sqlite3.connect(path) as destination:
        destination.execute("PRAGMA journal_mode=DELETE")
        for table in tables:
            schema = source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
            destination.execute(schema)
            where = " WHERE source IN (" + ",".join("?" for _ in SOURCES) + ")" if table == "public_edges" else ""
            order = {"genes": "accession", "public_edges": "id", "terms": "term",
                     "gene_annotations": "gene,term,qualifiers,evidence,references_json,source,payload"}[table]
            cursor = source.execute(f"SELECT * FROM {table}{where} ORDER BY {order}", SOURCES if where else ())
            while rows := cursor.fetchmany(4096):
                destination.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in rows[0])})", rows)
        if "public_edges" in tables:
            destination.execute("CREATE INDEX edge_subject ON public_edges(subject)")
            destination.execute("CREATE INDEX edge_object ON public_edges(object)")
            destination.execute("CREATE INDEX gene_symbol ON genes(symbol)")
        else:
            destination.execute("CREATE INDEX annotation_gene ON gene_annotations(gene)")
            destination.execute("CREATE INDEX annotation_term ON gene_annotations(term)")
        actual = {x[0] for x in destination.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if actual != set(tables) or destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("public-only export table/integrity check failed")
        report = {"tables": {table: destination.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}}
        if "public_edges" in tables:
            report["edge_sources"] = dict(destination.execute("SELECT source,COUNT(*) FROM public_edges GROUP BY source"))
            report["signed_input_rows"] = rows_digest(destination, SIGNED_SQL)
            report["gene_mapping_rows"] = rows_digest(destination, "SELECT accession,symbol FROM genes ORDER BY accession,symbol")
        else:
            report["annotation_input_rows"] = rows_digest(destination, "SELECT gene,term,qualifiers FROM gene_annotations ORDER BY gene,term,qualifiers")
    report["sha256"] = sha256(path)
    return report


def build(work_dir, source_dir=None, *, allow_new_snapshots=False):
    work_dir = Path(work_dir).resolve()
    source_dir = Path(source_dir).resolve() if source_dir else work_dir / "assets/public_knowledge_sources"
    paths, identities, manifest = checked_sources(source_dir, allow_new_snapshots=allow_new_snapshots)
    assets = work_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    names = ("public_knowledge.sqlite", "public_go.sqlite", "public_knowledge.provenance.json")
    for name in names:
        if (assets / name).exists() or (assets / name).is_symlink():
            raise FileExistsError(f"refusing to overwrite public output: {assets / name}")
    lock = assets / ".public_knowledge_build.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        with tempfile.TemporaryDirectory(prefix=".public_knowledge_", dir=assets) as temporary:
            staging = Path(temporary)
            membership = staging / "reactome_membership_reviewed_human.tsv"
            membership_audit = derive_membership(paths["reactome_pathways"], paths["uniprot_reactome_xrefs"], membership)
            signed = staging / "canonical_signed_records.jsonl"
            signed_audit = canonical_signed_records(paths, signed)
            selected = {k: v for k, v in paths.items() if k not in ("collectri_raw", "signor_raw", "uniprot_reactome_xrefs")}
            selected.update(legacy_graph=signed, reactome_membership=membership)
            with sqlite3.connect(staging / "public_intermediate.sqlite") as connection:
                audit = public_import.import_public(connection, selected)
                connection.commit()
                graph_report = export_view(connection, staging / names[0], ("genes", "public_edges"))
                go_report = export_view(connection, staging / names[1], ("terms", "gene_annotations"))
            for item in audit["sources"].values():
                if "path" in item:
                    item["filename"] = Path(item.pop("path")).name
            historical = all(x["matches_historical_snapshot"] for x in identities.values())
            derived_match = membership_audit["sha256"] == manifest["historical_derived_membership"]["sha256"]
            if historical and not derived_match:
                raise ValueError("historical membership derivation differs from recorded bytes")
            if historical:
                for key in ("signed_input_rows", "gene_mapping_rows"):
                    if graph_report[key] != manifest["historical_consumed_rows"][key]:
                        raise ValueError(f"historical ALIGNED consumed-row digest differs: {key}")
            report = {"schema": "s1pert.aligned-public-inputs.v1",
                      "created_utc": datetime.now(timezone.utc).isoformat(),
                      "historical_raw_inputs_match": historical,
                      "historical_aligned_consumed_input_protocol": historical and derived_match,
                      "full_historical_database_reproduced": False,
                      "omitted": "unsigned measured-axis legacy GO/STRING rows, discarded by ALIGNED before graph construction",
                      "response_data_read": False, "empirical_tables_copied": False,
                      "source_allowlist": list(SOURCES), "raw_inputs": identities,
                      "source_code_sha256": manifest["source_code_sha256"],
                      "signed_parser_audit": signed_audit,
                      "derived_reactome_membership": membership_audit,
                      "import_audit": audit,
                      "outputs": {names[0]: graph_report, names[1]: go_report}}
            json_write(staging / names[2], report)
            for name in names:
                os.link(staging / name, assets / name)
    finally:
        lock.unlink()
    return report


def refresh_manifest(output):
    document = json.loads(MANIFEST.read_text())
    document["notes"].insert(0, "EXPLICIT REFRESH: unpinned current public downloads; not historical reproduction. Actual hashes must be recorded by downloader and builder.")
    for asset in document["assets"]:
        asset["historical_checksums"] = asset.pop("checksums")
        asset["historical_size_bytes"] = asset.pop("size_bytes")
        if asset["id"] == "go_obo":
            asset["url"] = "https://current.geneontology.org/ontology/go-basic.obo"
        elif asset["id"] == "go_gaf":
            asset["url"] = "https://current.geneontology.org/annotations/gaf/HUMAN-uniprot.gaf.gz"
    json_write(output, document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default=os.environ.get("S1PERT_ROOT"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--allow-new-snapshots", action="store_true")
    parser.add_argument("--refresh-download-manifest", type=Path,
                        help="write a separate unpinned current-download manifest; do not download or build")
    args = parser.parse_args(argv)
    if args.refresh_download_manifest:
        refresh_manifest(args.refresh_download_manifest)
        print(args.refresh_download_manifest)
        return
    if not args.work_dir:
        parser.error("--work-dir or S1PERT_ROOT is required")
    report = build(args.work_dir, args.source_dir, allow_new_snapshots=args.allow_new_snapshots)
    print(json.dumps({key: report[key] for key in ("schema", "historical_raw_inputs_match", "response_data_read", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
