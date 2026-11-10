#!/usr/bin/env python3
"""Build SUMMER's public-only edge database from the pinned PerturbQA archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile

KG_MD5 = "90977691ebac65e02765e34c31b66bd1"
MEMBERS = {"go.json": "GO", "reactome.json": "Reactome", "string.json": "STRING"}


def digest_file(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def edges(payload, source):
    graph = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(graph, dict):
        raise ValueError(f"invalid {source} adjacency mapping")
    for subject, values in graph.items():
        if not isinstance(subject, str) or not subject.strip() or not isinstance(values, list):
            raise ValueError(f"invalid {source} subject/adjacency")
        for value in values:
            target = value[0] if isinstance(value, list) and value else value
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"invalid {source} neighbor")
            # GO/Reactome remain gene -> term. STRING remains gene -> gene;
            # the SUMMER retrieval adapter handles its undirected interpretation.
            yield subject, target, source


def build(archive, output, *, expected_md5=KG_MD5):
    archive, output = Path(archive), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("public edge output exists; choose a fresh destination")
    if digest_file(archive, "md5") != expected_md5:
        raise ValueError("PerturbQA archive MD5 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".public-edges-", suffix=".sqlite", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    connection = sqlite3.connect(temporary)
    try:
        with connection, zipfile.ZipFile(archive) as handle:
            selected = {}
            for member in handle.infolist():
                basename = Path(member.filename).name
                if basename not in MEMBERS or "__MACOSX" in Path(member.filename).parts:
                    continue
                if basename in selected or member.file_size > 128 * 1024 * 1024:
                    raise ValueError("duplicate or oversized public graph member")
                selected[basename] = member
            if set(selected) != set(MEMBERS):
                raise ValueError("archive must contain GO, Reactome, and STRING JSON files")
            connection.execute("CREATE TABLE public_edges(subject TEXT NOT NULL, object TEXT NOT NULL, source TEXT NOT NULL, PRIMARY KEY(subject,object,source)) WITHOUT ROWID")
            report = {"schema": "s1pert.public-edges.v1", "archive_md5": expected_md5,
                      "archive_sha256": digest_file(archive), "members": {},
                      "response_data_read": False,
                      "historical_edge_database_equality_verified": False}
            for basename, source in MEMBERS.items():
                member = selected[basename]
                raw = handle.read(member)
                connection.executemany("INSERT OR IGNORE INTO public_edges VALUES(?,?,?)",
                                       edges(json.loads(raw), source))
                count = connection.execute("SELECT count(*) FROM public_edges WHERE source=?", (source,)).fetchone()[0]
                report["members"][source] = {"member": member.filename, "sha256": hashlib.sha256(raw).hexdigest(),
                                               "unique_edges": count}
                if not count:
                    raise ValueError(f"public graph source {source} is empty")
            connection.execute("CREATE INDEX edges_source ON public_edges(source)")
            connection.execute("CREATE TABLE provenance(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO provenance VALUES(?,?)", ("build", json.dumps(report, sort_keys=True)))
        connection.close()
        os.link(temporary, output)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("assets/baselines/kg.zip"))
    parser.add_argument("--out", type=Path, default=Path("assets/public_edges.sqlite"))
    args = parser.parse_args(argv)
    print(json.dumps(build(args.archive, args.out), indent=2))


if __name__ == "__main__":
    main()
