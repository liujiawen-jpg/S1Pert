import json
from pathlib import Path
import sqlite3
import sys

import pytest

from jev_a4.config import load_config
from jev_a4.knowledge import import_public_tsv


def test_paths_resolve_against_config_not_cwd(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"project_root": "work", "runtime_python": None,
        "decider_snapshot": "weights", "raw_h5ad_paths": {"hepg2": "raw.h5ad"},
        "preprocess": {"datasets": {"hepg2": {"raw": "raw.h5ad", "cache": "moments"}}}}))
    monkeypatch.chdir(tmp_path.parent)
    cfg = load_config(path)
    assert cfg["project_root"] == str(tmp_path / "work")
    assert cfg["runtime_python"] == sys.executable
    assert cfg["decider_source"] == str(tmp_path / "weights")
    assert cfg["preprocess"]["datasets"]["hepg2"]["cache"] == str(tmp_path / "moments")
    assert not (tmp_path / "work").exists()


def test_unset_variable_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("A4_NO_SUCH_ASSET", raising=False)
    path = tmp_path / "config.json"
    path.write_text('{"decider_snapshot":"${A4_NO_SUCH_ASSET}/weights"}')
    with pytest.raises(ValueError, match="unset"):
        load_config(path)


def test_virtualenv_python_symlink_and_roundtrip_are_preserved(tmp_path):
    executable = tmp_path / "base-python"
    executable.write_text("fixture, not executed")
    virtual = tmp_path / "venv-python"
    virtual.symlink_to(executable)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"runtime_python": "./venv-python"}))
    cfg = load_config(path)
    assert cfg["runtime_python"] == str(virtual)
    path.write_text(json.dumps(cfg))
    assert load_config(path) == cfg


def test_public_knowledge_import_and_overwrite_guard(tmp_path):
    source = tmp_path / "public.tsv"
    source.write_text("accession\tsymbol\tprotein_name\tfunction_text\nP1\tGENE_A\tA\tSynthetic annotation.\n")
    out = tmp_path / "public.sqlite"
    assert import_public_tsv(source, out) == 1
    db = sqlite3.connect(out)
    assert db.execute("SELECT symbol FROM genes").fetchall() == [("GENE_A",)]
    assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("genes",)]
    db.close()
    with pytest.raises(FileExistsError):
        import_public_tsv(source, out)
