import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_pipeline", ROOT / "src/jev_a4/core/pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def config():
    return pipeline.load_config(ROOT / "configs/a4.example.json")


def test_plan_twenty_independent_runs():
    result = pipeline.plan(config())
    assert len(result) == 20
    assert len({(r["dataset"], r["split_seed"]) for r in result}) == 20
    assert all(r["task_head_initialization"] == "fresh" for r in result)


def test_duplicate_seed_blocked():
    cfg = config()
    cfg["seeds"][-1] = cfg["seeds"][0]
    with pytest.raises(ValueError):
        pipeline.plan(cfg)


def test_old_task_initialization_blocked():
    cfg = config()
    cfg["initial_task_checkpoint"] = "/old/best.pt"
    with pytest.raises(ValueError):
        pipeline.plan(cfg)


def test_means_sample_variance_and_missing_metric():
    reports = [{"seed": s, "metrics": {"macro_f1": float(i), "per_class": {
        "up": {"average_precision": None if i == 0 else .5, "support": i + 1}}}}
        for i, s in enumerate(config()["seeds"])]
    value = pipeline.aggregate(reports, config()["seeds"])
    assert value["status"] == "COMPLETE"
    assert value["metrics"]["macro_f1"]["mean"] == 2
    assert value["metrics"]["macro_f1"]["sample_variance"] == 2.5
    assert value["metrics"]["per_class.up.average_precision"]["n_defined"] == 4
    assert "per_class.up.support" not in value["metrics"]


def test_incomplete_summary_not_five_seed_result():
    value = pipeline.aggregate([{ "seed": 17, "metrics": {"macro_f1": .3}}], config()["seeds"])
    assert value["status"] == "INCOMPLETE"
    assert value["metrics"]["macro_f1"]["sample_variance"] is None


def test_duplicate_results_blocked():
    result = {"seed": 17, "metrics": {"macro_f1": .5}}
    with pytest.raises(ValueError):
        pipeline.aggregate([result, result], config()["seeds"])


def test_manifest_digest_rechecked(tmp_path):
    artifact = tmp_path / "array.npy"
    artifact.write_bytes(b"one")
    pipeline.dump(tmp_path / "manifest.json", {"status": "PASS", "artifacts": {
        "array.npy": {"bytes": 3, "sha256": pipeline.sha(artifact)}}})
    assert pipeline.verify_artifact_manifest(tmp_path, "manifest.json", "bytes")["files"] == 1
    artifact.write_bytes(b"two")
    with pytest.raises(ValueError):
        pipeline.verify_artifact_manifest(tmp_path, "manifest.json", "bytes")
