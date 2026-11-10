"""Offline tests: synthetic HTTP responses, never a real asset download."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3

import pytest

from jev_a4.knowledge import import_public_tsv


spec = importlib.util.spec_from_file_location("download_assets", Path(__file__).parents[1] / "scripts/download_assets.py")
download_assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_assets)


def asset(data=b"synthetic asset"):
    return {"id": "fixture", "group": "data", "path": "raw/fixture.bin",
            "url": "https://example.org/fixture.bin", "size_bytes": len(data),
            "checksums": {"sha256": hashlib.sha256(data).hexdigest()}}


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data)), "ETag": '"v1"', **(headers or {})}


def seed_partial(tmp_path, item, data, validator='"v1"'):
    path = tmp_path / item["path"]
    path.parent.mkdir(parents=True)
    path.with_name(path.name + ".part").write_bytes(data)
    path.with_name(path.name + ".part.json").write_text(json.dumps({
        "identity": {key: item.get(key) for key in ("url", "size_bytes", "checksums")},
        "validator": validator}))
    return path


def test_complete_download_verified_receipt_and_idempotence(tmp_path, monkeypatch):
    payload = b"synthetic asset"
    item = asset(payload)
    monkeypatch.setattr(download_assets, "urlopen", lambda *a, **k: Response(payload))
    result = download_assets.download(item, tmp_path)
    path = tmp_path / item["path"]
    assert result["status"] == "downloaded"
    assert path.read_bytes() == payload
    assert not path.with_name(path.name + ".part").exists()
    receipt = json.loads(path.with_name(path.name + ".download.json").read_text())
    assert receipt["published_checksum_verified"]
    monkeypatch.setattr(download_assets, "urlopen", lambda *a, **k: pytest.fail("already present"))
    assert download_assets.download(item, tmp_path)["status"] == "verified_existing"


def test_resume_requires_correct_range_and_validator(tmp_path, monkeypatch):
    payload = b"synthetic asset"
    item = asset(payload)
    path = seed_partial(tmp_path, item, payload[:4])
    def respond(request, **kwargs):
        assert request.get_header("Range") == "bytes=4-"
        assert request.get_header("If-range") == '"v1"'
        return Response(payload[4:], 206, {"Content-Range": f"bytes 4-{len(payload)-1}/{len(payload)}"})
    monkeypatch.setattr(download_assets, "urlopen", respond)
    download_assets.download(item, tmp_path)
    assert path.read_bytes() == payload


@pytest.mark.parametrize("status,headers", [(200, {}), (206, {"Content-Range": "bytes 0-14/15"}),
                                           (206, {"Content-Range": "bytes 4-14/15", "ETag": '"v2"'})])
def test_bad_resume_preserves_partial(tmp_path, monkeypatch, status, headers):
    item = asset(b"synthetic asset")
    path = seed_partial(tmp_path, item, b"synt")
    monkeypatch.setattr(download_assets, "urlopen", lambda *a, **k: Response(b"hetic asset", status, headers))
    with pytest.raises(ValueError):
        download_assets.download(item, tmp_path)
    assert not path.exists()
    assert path.with_name(path.name + ".part").read_bytes() == b"synt"


def test_wrong_checksum_does_not_publish_or_overwrite(tmp_path, monkeypatch):
    item = asset(b"good")
    monkeypatch.setattr(download_assets, "urlopen", lambda *a, **k: Response(b"evil"))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        download_assets.download(item, tmp_path)
    path = tmp_path / item["path"]
    assert not path.exists()
    assert path.with_name(path.name + ".part").read_bytes() == b"evil"
    path.write_bytes(b"old!")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        download_assets.download(item, tmp_path)
    assert path.read_bytes() == b"old!"


def test_truncated_response_is_retained_for_resume(tmp_path, monkeypatch):
    item = asset(b"synthetic asset")
    monkeypatch.setattr(download_assets, "urlopen", lambda *a, **k: Response(b"synt", headers={"Content-Length": "15"}))
    with pytest.raises(ValueError, match="incomplete response"):
        download_assets.download(item, tmp_path)
    assert not (tmp_path / item["path"]).exists()
    assert (tmp_path / (item["path"] + ".part")).read_bytes() == b"synt"


def test_unpinned_export_receipt_detects_later_modification(tmp_path, monkeypatch):
    item = asset(b"good")
    item.update(size_bytes=None, checksums={})
    monkeypatch.setattr(download_assets, "urlopen", lambda *a, **k: Response(b"good", headers={"X-UniProt-Release": "fixture"}))
    download_assets.download(item, tmp_path)
    (tmp_path / item["path"]).write_bytes(b"evil")
    with pytest.raises(ValueError, match="receipt"):
        download_assets.download(item, tmp_path)


def test_unknown_existing_file_without_digest_is_never_adopted(tmp_path):
    item = asset(b"data")
    item["checksums"] = {}
    path = tmp_path / item["path"]
    path.parent.mkdir()
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="no published checksum"):
        download_assets.download(item, tmp_path)
    assert path.read_bytes() == b"data"


def test_git_blob_checksum_handles_empty_source_file(tmp_path, monkeypatch):
    item = asset(b"")
    item["checksums"] = {"git_sha1": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"}
    monkeypatch.setattr(download_assets, "urlopen", lambda *a, **k: Response(b""))
    assert download_assets.download(item, tmp_path)["size_bytes"] == 0


def test_dry_run_and_list_are_offline_and_do_not_create_destinations(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(download_assets, "urlopen", lambda *a, **k: pytest.fail("network during preview"))
    for switch in ("--list", "--dry-run"):
        assert download_assets.main([switch, "--root", str(tmp_path / "unused")]) == 0
        result = json.loads(capsys.readouterr().out)
        assert {"hepg2", "jurkat", "rpe1", "k562_essential"} <= {x["id"] for x in result}
    assert not (tmp_path / "unused").exists()
    with pytest.raises(SystemExit):
        download_assets.main(["--group", "backbone", "--root", str(tmp_path / "unused")])
    assert not (tmp_path / "unused").exists()


def test_manifest_destinations_agree_with_preprocessing_configuration():
    from jev_a4.config import load_config
    manifest, root = download_assets.load_manifest(download_assets.DEFAULT_MANIFEST)
    configured = load_config(download_assets.DEFAULT_MANIFEST.with_name("preprocess.example.json"))["preprocess"]["datasets"]
    for item in manifest["assets"]:
        if item["group"] == "data":
            assert str(root / item["path"]) == configured[item["id"]]["raw"]


def test_manifest_rejects_traversal_and_symlink_escape(tmp_path):
    item = asset()
    item["path"] = "../outside"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "s1pert.public-assets.v1", "root": ".", "assets": [item]}))
    with pytest.raises(ValueError, match="unsafe"):
        download_assets.load_manifest(manifest)
    item["path"] = "escape/file"
    (tmp_path / "escape").symlink_to(tmp_path.parent)
    manifest.write_text(json.dumps({"schema": "s1pert.public-assets.v1", "root": ".", "assets": [item]}))
    with pytest.raises(ValueError, match="escapes"):
        download_assets.load_manifest(manifest)


def test_uniprot_export_imports_primary_symbols_and_omits_unmapped(tmp_path):
    source = tmp_path / "public.tsv"
    source.write_text("Entry\tGene Names (primary)\tProtein names\tFunction [CC]\n"
                      "P1\tGENE_A GENE_B\tProtein\tPublic function.\n"
                      "P2\t\tUnmapped\tNo symbol.\n")
    destination = tmp_path / "public.sqlite"
    assert import_public_tsv(source, destination) == 2
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT symbol FROM genes ORDER BY symbol").fetchall() == [("GENE_A",), ("GENE_B",)]


def test_failed_annotation_import_leaves_no_database(tmp_path):
    source = tmp_path / "public.tsv"
    source.write_text("accession\tsymbol\tprotein_name\tfunction_text\nP1\tGENE_A\tA\tvalid\nP2\t123\tB\tinvalid\n")
    destination = tmp_path / "public.sqlite"
    with pytest.raises(ValueError, match="numeric"):
        import_public_tsv(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".public-functions-*"))
