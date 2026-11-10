"""Public asset adapters tested with local fixtures and no network/model loads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
import pytest


def module(name):
    path = Path(__file__).parents[1] / "baselines" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fetch = module("fetch_external")
graph = module("prepare_public_edges")
esm = module("prepare_esm2")


def test_external_previews_are_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch, "git", lambda *a, **k: pytest.fail("Git used in preview"))
    monkeypatch.setattr(fetch, "downloader", lambda: pytest.fail("download used in preview"))
    for switch in ("--list", "--dry-run"):
        assert fetch.main([switch, "--root", str(tmp_path / "new")]) == 0
        listing = json.loads(capsys.readouterr().out)
        assert set(listing["repositories"]) == {"gears", "perturbqa", "state", "scgpt"}
        assert set(listing["assets"]) == {"kg", "genept", "gene2go", "esm2"}
        assert all(len(spec["commit"]) == 40 for spec in listing["repositories"].values())
    assert not (tmp_path / "new").exists()


def test_existing_external_directory_is_preserved(tmp_path):
    existing = tmp_path / fetch.REPOS["gears"]["path"]
    existing.mkdir(parents=True)
    (existing / "local.txt").write_text("preserve")
    with pytest.raises(ValueError, match="regular Git"):
        fetch.fetch_repo("gears", tmp_path)
    assert (existing / "local.txt").read_text() == "preserve"


def test_published_archive_checksums_match_classical_config():
    config = json.loads((fetch.ROOT / "baselines/classical/config.example.json").read_text())
    for name, key in (("kg", "perturbqa_kg"), ("genept", "genept")):
        spec = fetch.ASSETS[name][0]
        assert spec["checksums"]["md5"] == config["knowledge_resources"][key]["md5"]
        assert spec["url"] == config["knowledge_resources"][key]["url"]


def archive_fixture(path, omit=None):
    payloads = {"go.json": [{"GENE_A": [["GO:1", "enables"], ["GO:1", "enables"]]}, {}],
                "reactome.json": [{"GENE_A": [["Pathway 1", "nucleus"]]}, {}],
                "string.json": {"GENE_A": [["GENE_B", ["experiments"]]]}}
    with zipfile.ZipFile(path, "w") as handle:
        for name, payload in payloads.items():
            if name != omit:
                handle.writestr("perturbqa/datasets/kg/" + name, json.dumps(payload))
        handle.writestr("../../unrelated.txt", "never extracted")
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_graph_conversion_preserves_source_specific_relations(tmp_path):
    archive, output = tmp_path / "kg.zip", tmp_path / "edges.sqlite"
    md5 = archive_fixture(archive)
    report = graph.build(archive, output, expected_md5=md5)
    with sqlite3.connect(output) as db:
        assert db.execute("SELECT subject,object,source FROM public_edges ORDER BY source").fetchall() == [
            ("GENE_A", "GO:1", "GO"), ("GENE_A", "Pathway 1", "Reactome"),
            ("GENE_A", "GENE_B", "STRING")]
        saved = json.loads(db.execute("SELECT value FROM provenance WHERE key='build'").fetchone()[0])
    assert report == saved
    assert not report["response_data_read"]
    assert not report["historical_edge_database_equality_verified"]
    with pytest.raises(FileExistsError):
        graph.build(archive, output, expected_md5=md5)


def test_graph_checksums_and_missing_sources_fail_without_output(tmp_path):
    archive, output = tmp_path / "kg.zip", tmp_path / "edges.sqlite"
    md5 = archive_fixture(archive, omit="go.json")
    with pytest.raises(ValueError, match="MD5"):
        graph.build(archive, output)
    with pytest.raises(ValueError, match="must contain"):
        graph.build(archive, output, expected_md5=md5)
    assert not output.exists()
    assert not list(tmp_path.glob(".public-edges-*"))


def esm_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "meta.json").write_text(json.dumps({"keys": ["GENE_A", "NEW_NAME"], "dim": 3}))
    (source / "aliases.yaml").write_text("OLD_NAME: NEW_NAME\n")
    np.save(source / "embeddings.npy", np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    metadata = tmp_path / "dataset.json"
    metadata.write_text(json.dumps({"target_ids": ["ENSG1", "ENSG2", "ENSG3"],
                                    "target_symbols": ["GENE_A", "OLD_NAME", "MISSING"]}))
    return source, metadata


def test_esm_maps_public_symbols_and_explicit_aliases_only(tmp_path):
    source, metadata = esm_fixture(tmp_path)
    output = tmp_path / "mapped"
    report = esm.build(source, [metadata], output)
    assert report["matched"] == 2 and report["missing"] == 1
    info = json.loads((output / "meta.json").read_text())
    assert info["keys"] == ["ENSG1", "ENSG2"]
    assert info["missing_target_ids"] == ["ENSG3"]
    assert info["key_mapping"][1]["public_key"] == "NEW_NAME"
    np.testing.assert_array_equal(np.load(output / "embeddings.npy"), [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(FileExistsError):
        esm.build(source, [metadata], output)


def test_esm_rejects_ambiguous_aliases_and_invalid_axes(tmp_path):
    source, metadata = esm_fixture(tmp_path)
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text("gene_id\tsymbol\nENSG1\tNEW_NAME\n")
    with pytest.raises(ValueError, match="ambiguous"):
        esm.build(source, [metadata], tmp_path / "mapped", aliases)
    np.save(source / "embeddings.npy", np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="axes"):
        esm.build(source, [metadata], tmp_path / "mapped")
    assert not (tmp_path / "mapped").exists()
