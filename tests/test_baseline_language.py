"""Small CPU checks for the historical Qwen LoRA adapter; no model assets."""
import importlib.util
import json
from pathlib import Path
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[1] / "baselines/language/qwen3_lora.py"
spec = importlib.util.spec_from_file_location("qwen3_lora_tested", SCRIPT)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def records():
    return np.array([(target, query, (target + query) % 3)
                     for target in range(5) for query in range(2)],
                    dtype=[("target", "i4"), ("query", "i4"), ("label", "i1")])


def test_train_evidence_excludes_held_target_fold():
    rows = records()
    folds = np.arange(5)
    whole, cross = adapter.priors(rows, folds, 2)
    changed = rows.copy()
    changed["label"][changed["target"] == 2] = (changed["label"][changed["target"] == 2] + 1) % 3
    altered_whole, altered_cross = adapter.priors(changed, folds, 2)
    np.testing.assert_array_equal(cross[2], altered_cross[2])
    assert not np.array_equal(whole, altered_whole)


def test_heldout_labels_do_not_enter_numeric_inputs(tmp_path, monkeypatch):
    root = tmp_path / "toy"
    root.mkdir()
    np.save(root / "condition_numeric.npy", np.ones((5, 4), dtype="f4"))
    np.save(root / "query_numeric.npy", np.ones((2, 4), dtype="f4"))
    monkeypatch.setattr(adapter, "DATA", tmp_path)
    train = records()
    heldout = train[:2].copy()
    meta = {"dataset": "toy", "query_ids": ["A", "B"]}
    adjacency = {i: np.array([], dtype="i4") for i in range(5)}
    expected, relation = adapter.static_inputs(heldout, "test", train, np.arange(5), meta, adjacency)
    heldout["label"] = (heldout["label"] + 1) % 3
    actual, other_relation = adapter.static_inputs(heldout, "test", train, np.arange(5), meta, adjacency)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(other_relation, relation)
    np.testing.assert_allclose(actual[:, 8:].sum(1), 1)


def test_fresh_direct_head_and_probability_order():
    class TinyBackbone(torch.nn.Module):
        def forward(self, input_ids, attention_mask, **kwargs):
            return types.SimpleNamespace(last_hidden_state=torch.ones((*input_ids.shape, 8)))
    model = adapter.LoRAQA(TinyBackbone(), 8).eval()
    logits, path, relation = model(torch.zeros((2, 3), dtype=torch.long),
                                   torch.ones((2, 3)), torch.zeros((2, 11)))
    assert logits.shape == path.shape == relation.shape == (2, 3)
    torch.testing.assert_close(logits.softmax(-1).sum(1), torch.ones(2))
    assert model.answer.in_features == 267


def test_asset_paths_are_relative_to_configuration(tmp_path, monkeypatch):
    for name in ("data", "snapshot"):
        (tmp_path / name).mkdir()
    (tmp_path / "edges.parquet").write_bytes(b"fixture only; not read")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"data_root": "data", "backbone_snapshot": "snapshot",
                               "knowledge_graph": "edges.parquet", "output_root": "outputs"}))
    monkeypatch.chdir(tmp_path.parent)
    adapter.configure_assets(cfg)
    assert adapter.SNAPSHOT == tmp_path / "snapshot"
    assert adapter.KG == tmp_path / "edges.parquet"
    assert adapter.OUT == tmp_path / "outputs"
    assert not adapter.OUT.exists()
