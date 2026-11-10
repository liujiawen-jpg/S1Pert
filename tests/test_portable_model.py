"""CPU contract tests for the real A4 implementation with small synthetic tensors."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from jev_a4.core import train
from jev_a4.core.metrics import average_precision, classification_metrics
from jev_a4.core.vendor import native_engine
from jev_a4.core.vendor.encode_joint_last8_cache import _run_tuple_batch


def subprocess_environment():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    source = str(Path(train.__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join([source, env.get("PYTHONPATH", "")])
    return env


def test_metrics_reject_noninteger_labels_and_nonfinite_scores():
    for labels in ([0.5], [float("nan")], [3], ["0"]):
        with pytest.raises(ValueError):
            classification_metrics(labels, [[1., 0., 0.]])
    with pytest.raises(ValueError):
        average_precision([False], [float("nan")])


def test_joint_evidence_is_removed_before_frozen_last8():
    torch = pytest.importorskip("torch")
    condition = np.full((1, 16, 2), 1., dtype="f2")
    query = np.full((1, 16, 2), 2., dtype="f2")
    cards = np.full((1, 4, 2), 9., dtype="f2")
    tuples = np.array([[0, 0, 0, -1]], dtype="i4")
    masks = []

    def last8(states, mask):
        masks.append(mask.clone())
        # A deterministic contextual operation makes pre-layer masking
        # observable; this fake operation tests cache plumbing, not accuracy.
        total = (states * mask[:, :, None]).sum(1, keepdim=True)
        return states + total

    kwargs = dict(condition_states=condition, query_states=query, card_states=cards,
                  forward_last8=last8, device="cpu", compute_dtype=torch.float32)
    present = _run_tuple_batch(tuples, evidence_column=2, **kwargs)
    absent = _run_tuple_batch(tuples, evidence_column=3, **kwargs)
    assert masks[0].all()
    assert masks[1][:, :32].all() and not masks[1][:, 32:].any()
    assert present.shape == absent.shape == (1, 4, 2)
    np.testing.assert_array_equal(absent[:, 2], 0.)
    assert np.all(present[:, :2] != absent[:, :2])
    altered = np.full_like(cards, 300.)
    other = _run_tuple_batch(tuples, evidence_column=3, **{**kwargs, "card_states": altered})
    np.testing.assert_array_equal(absent, other)


def test_only_actual_a4_components_are_vendored():
    vendor = Path(native_engine.__file__).parent
    manifest = json.loads((vendor / "source_manifest.json").read_text())
    for name, entry in manifest["files"].items():
        contents = (vendor / name).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == entry["current_sha256"]
        definitions = {n.name for n in ast.parse(contents).body if isinstance(n, ast.FunctionDef)}
        assert definitions == set(entry["retained_functions"]) | set(entry.get("portable_helpers", []))
    assert not (vendor / "protocol_utils.py").exists()
    assert not hasattr(native_engine, "inject_lora")
    assert not hasattr(train.vendor_module(), "train_stage")


def test_external_source_cannot_silently_use_an_installed_decider(tmp_path):
    package = tmp_path / "decider"
    package.mkdir()
    for name in ("__init__.py", "model.py", "prompt.py"):
        (package / name).write_text("")
    other = types.ModuleType("decider")
    other.__file__ = "/unrelated/installed/decider/__init__.py"
    with patch.dict(sys.modules, {"decider": other}):
        with pytest.raises(ImportError, match="different DECIDER source"):
            native_engine._external_decider_components(tmp_path)
    with pytest.raises(FileNotFoundError, match="external DECIDER"):
        native_engine._external_decider_components(tmp_path / "missing")


def test_checkpoint_mapping_rejects_unknown_tensors():
    names = {"model.layers.0.weight"}
    assert native_engine.checkpoint_target("model.language_model.layers.0.weight", names, []) in names
    assert native_engine.checkpoint_target("auxiliary.weight", names, [r"auxiliary\."]) is None
    with pytest.raises(ValueError, match="unexpected model tensor"):
        native_engine.checkpoint_target("unknown.weight", names, [])


def test_test_labels_remain_locked_when_restore_is_missing(tmp_path):
    (tmp_path / "restore_report.json").write_text(json.dumps({"status": "FAIL"}))
    with patch.object(train, "load_selected", return_value=(None, None, {"best_checkpoint_sha256": "a"})), \
            patch.object(train, "Partition") as partition:
        with pytest.raises(PermissionError, match="independent restore"):
            train.evaluate_test({}, "synthetic", 17, tmp_path, tmp_path, tmp_path)
        partition.assert_not_called()


def test_memory_safe_import_is_lazy_and_install_is_idempotent():
    code = """
import sys
from jev_a4.core import train, train_memory_safe
train_memory_safe.install()
first = train.Partition.batch
train_memory_safe.install()
assert first is train.Partition.batch
assert 'torch' not in sys.modules
assert train.RESTORE_ENTRYPOINT.name == 'train_memory_safe.py'
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env=subprocess_environment(), check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["preflight", "launch", "encode", "train_memory_safe", "pipeline"])
def test_module_help_works_outside_the_source_directory(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", f"jev_a4.core.{module}", "--help"],
        cwd=tmp_path, env=subprocess_environment(), capture_output=True, text=True,
        timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not list(tmp_path.iterdir())


def test_memory_safe_synthetic_train_restore_and_sealed_test(tmp_path):
    pytest.importorskip("torch")
    # No backbone, snapshot, real labels, or GPU is used. This invokes the
    # production execution layer and its independent restore subprocess.
    output = tmp_path / "synthetic"
    env = subprocess_environment()
    result = subprocess.run(
        [sys.executable, "-m", "jev_a4.core.train_memory_safe", "smoke", "--out", str(output)],
        capture_output=True, text=True, env=env, timeout=90, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    smoke = json.loads((output / "smoke_report.json").read_text())
    assert smoke["scientific_status"] == "NOT_EVALUATED_SYNTHETIC_ONLY"
    assert smoke["independent_restore"] == smoke["sealed_test_evaluation"] == "PASS"
    report = json.loads((output / "run" / "training_report.json").read_text())
    assert report["test_labels_opened"] is False
    assert report["main_exposure_min"] == report["main_exposure_max"] == report["epochs_completed"]
    assert report["bindings"]["execution_only_overlay_sha256"]
    assert [entry["partition"] for entry in report["partition_cache_reads"]] == ["train", "val"]
    assert all(entry["mode"] == "MEMMAP_BOUNDED_GATHER" and not entry["whole_partition_CPU_copy"]
               for entry in report["partition_cache_reads"])
    assert report["independent_restore"]["test_labels_opened"] is False
