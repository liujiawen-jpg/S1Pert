"""Small, offline public-input fixtures; no response data, model or GPU."""
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from data_prep import build_public_knowledge as builder
from data_prep.public_knowledge import public_import, signed_records


def fixture_sources(directory):
    directory.mkdir()
    (directory / "go_obo.obo").write_text("format-version: 1.2\ndata-version: fixture\n\n[Term]\nid: GO:0000001\nname: first\nnamespace: biological_process\nalt_id: GO:9999999\nis_a: GO:0000002 ! second\n\n[Term]\nid: GO:0000002\nname: second\nnamespace: biological_process\n")
    with gzip.open(directory / "go_gaf.gaf.gz", "wt") as stream:
        stream.write("!gaf-version: 2.2\n")
        for symbol, qualifier, term in (("A", "", "GO:9999999"), ("A", "NOT", "GO:0000002"), ("B", "enables", "GO:0000002")):
            stream.write("\t".join(["UniProtKB", "P1", symbol, qualifier, term, "PMID:1", "EXP", "", "P", "", "", "protein", "taxon:9606", "20260101", "fixture"]) + "\n")
    fields = ["source", "target", "source_genesymbol", "target_genesymbol", "is_directed", "is_stimulation", "is_inhibition", "sources", "references", "type", "ncbi_tax_id_source", "ncbi_tax_id_target"]
    for filename, source, relation in (("omnipath_core.tsv", "core", "post_translational"), ("collectri_raw.tsv", "CollecTRI", "transcriptional"), ("signor_raw.tsv", "SIGNOR", "post_translational")):
        with (directory / filename).open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(fields)
            writer.writerow(["P1", "P2", "A", "B", "1", "1", "0", source, "PMID:1", relation, "9606", "9606"])
            writer.writerow(["P2", "P1", "B", "A", "1", "1", "1", source, "PMID:2", relation, "9606", "9606"])
            writer.writerow(["COMPLEX:P1_P2", "P2", "A", "B", "1", "1", "0", source, "PMID:3", relation, "9606", "9606"])
    (directory / "uniprot_tsv.tsv").write_text("Entry\tGene Names (primary)\nP1\tA\nP2\tB\nP3\tC D\n")
    (directory / "reactome_pathways.tsv").write_text("R-HSA-1\tFirst pathway\tHomo sapiens\nR-HSA-2\tSecond pathway\tHomo sapiens\nR-MMU-1\tMouse\tMus musculus\n")
    (directory / "reactome_hierarchy.tsv").write_text("R-HSA-1\tR-HSA-2\n")
    (directory / "uniprot_reactome_xrefs.tsv").write_text("Entry\tReactome\nP1\tR-HSA-1;R-MMU-1;\n")
    return directory


def test_public_views_preserve_historical_filters(tmp_path):
    sources = fixture_sources(tmp_path / "sources")
    report = builder.build(tmp_path / "work", sources, allow_new_snapshots=True)
    assert not report["historical_raw_inputs_match"]
    assert not report["full_historical_database_reproduced"]
    assert report["response_data_read"] is False
    assert report["empirical_tables_copied"] is False
    assert report["derived_reactome_membership"]["rows"] == 1
    assert report["signed_parser_audit"]["collectri"]["rejected"] == {"complex_entity": 1}
    with sqlite3.connect(tmp_path / "work/assets/public_knowledge.sqlite") as db:
        assert {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"genes", "public_edges"}
        assert dict(db.execute("SELECT accession,symbol FROM genes")) == {"P1": "A", "P2": "B", "P3": ""}
        rows = list(db.execute("SELECT subject,object,sign,source FROM public_edges"))
        assert ("A", "B", "positive", "OmniPath/CollecTRI") in rows
        assert ("B", "A", "unknown", "OmniPath/CollecTRI") in rows
        assert ("B", "A", "conflict", "OmniPath-core") in rows
        assert not any(source.endswith("-raw") for *_, source in rows)
        assert set(source for *_, source in rows).issubset(builder.SOURCES)
        assert not any(subject.startswith("COMPLEX") and source != "OmniPath-core" for subject, _, _, source in rows)
    with sqlite3.connect(tmp_path / "work/assets/public_go.sqlite") as db:
        assert {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"terms", "gene_annotations"}
        assert ("A", "GO:0000001", "[]") in list(db.execute("SELECT gene,term,qualifiers FROM gene_annotations"))
        assert json.loads(db.execute("SELECT qualifiers FROM gene_annotations WHERE gene='A' AND term='GO:0000002'").fetchone()[0]) == ["NOT"]
    assert str(tmp_path) not in json.dumps(report)
    with pytest.raises(FileExistsError):
        builder.build(tmp_path / "work", sources, allow_new_snapshots=True)


def test_default_rejects_drift_before_writing_outputs(tmp_path):
    sources = fixture_sources(tmp_path / "sources")
    with pytest.raises(ValueError, match="historical snapshot mismatch"):
        builder.build(tmp_path / "work", sources)
    assert not (tmp_path / "work").exists()


def test_refresh_is_explicit_and_compatible_with_shared_downloader(tmp_path):
    from scripts.download_assets import load_manifest
    path = tmp_path / "refresh.json"
    builder.refresh_manifest(path)
    manifest, root = load_manifest(path, tmp_path / "work")
    assert len(manifest["assets"]) == 9
    for asset in manifest["assets"]:
        assert "checksums" not in asset and "size_bytes" not in asset
        assert len(asset["historical_checksums"]["sha256"]) == 64
    assert "EXPLICIT REFRESH" in manifest["notes"][0]
    with pytest.raises(FileExistsError):
        builder.refresh_manifest(path)


def test_direct_help_without_installed_package(tmp_path):
    result = subprocess.run([sys.executable, str(builder.REPO / "data_prep/build_public_knowledge.py"), "--help"], cwd=tmp_path, text=True, capture_output=True, check=True)
    assert "--work-dir" in result.stdout and "--allow-new-snapshots" in result.stdout


def test_pinned_source_manifest_has_no_private_dependencies():
    from scripts.download_assets import load_manifest
    manifest, _ = load_manifest(builder.MANIFEST)
    assert len(manifest["assets"]) == 9
    assert all(len(asset["checksums"]["sha256"]) == 64 for asset in manifest["assets"])
    text = json.dumps(manifest)
    assert "/root/" not in text and "/mnt/" not in text
    assert "legacy_graph" not in {asset["id"] for asset in manifest["assets"]}


def test_scientific_importers_match_original_ast():
    """Portable fingerprints verified against originals during source packaging."""
    expected = json.loads(builder.MANIFEST.read_text())["source_ast_sha256"]
    tree = ast.parse(Path(public_import.__file__).read_text())
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == expected["public_import.py"]["module"]
    tree = ast.parse(Path(signed_records.__file__).read_text())
    nodes = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for name, digest in expected["mechanism_graph.py"].items():
        assert hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() == digest
