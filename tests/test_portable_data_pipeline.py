"""Small synthetic H5AD tests; no production matrices or model downloads."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from jev_a4.config import load_config
from jev_a4.core import prepare
from jev_a4.data import nadig_moments, preprocess
from jev_a4.labels import build_resolution_bundle
from jev_a4.labels.label_core import Moments, label_pair


def _text(group, name, values):
    group.create_dataset(name, data=np.asarray(values, dtype=h5py.string_dtype()))


def synthetic_h5ad(path: Path, kind: str, targets=20, per_group=24):
    genes = [f"ENSG{i:04d}" for i in range(4)]
    symbols = [f"GENE_{i}" for i in range(4)]
    blocks = []
    for group in range(targets + 1):
        baseline = np.asarray([20, 20, 20, 0] if group == 0 else [40, 20, 10, 0])
        jitter = (np.arange(per_group) % 3 - 1)[:, None] * np.asarray([1, 1, 1, 0])
        blocks.append(baseline + jitter)
    raw = np.concatenate(blocks).astype(np.int32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=raw, chunks=(12, 4))
        obs, var = handle.create_group("obs"), handle.create_group("var")
        obs.create_dataset("UMI_count", data=np.full(len(raw), 100))
        _text(var, "gene_name", symbols)
        if kind == "nadig":
            _text(var, "gene_id", genes)
            _text(obs, "gene_id", np.repeat(["non-targeting"] + [f"ENSG{i:04d}" for i in range(targets)], per_group))
            _text(obs, "gene", np.repeat(["non-targeting"] + [f"GENE_{i}" for i in range(targets)], per_group))
            _text(obs, "cell_barcode", [f"CELL_{i}" for i in range(len(raw))])
        else:
            # Replogle only includes targets mapping to one measured symbol.
            _text(var, "ensembl_id", genes)
            _text(obs, "perturbation", np.repeat(["control"] + [f"GENE_{i}" for i in range(targets)], per_group))
    return raw


def test_raw_to_strict_labels_to_five_a4_splits(tmp_path):
    raw_path = tmp_path / "fixture.h5ad"
    raw = synthetic_h5ad(raw_path, "nadig")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "preprocess": {"output": "moments", "block_cells": 12,
                       "datasets": {"fixture": {"kind": "nadig", "raw": "fixture.h5ad"}}},
        "data_preprocessed_root": "moments", "label_resolution_root": "labels",
        "seeds": list(prepare.SEEDS), "exclude_self_pairs": True,
    }))
    cfg = load_config(config_path)
    result = preprocess.build(cfg)
    data = Path(result["output"]) / "fixture"
    expected = (raw.reshape(21, 24, 4) * 100.).mean(1)
    np.testing.assert_allclose(np.load(data / "mean.npy"), expected)
    np.testing.assert_allclose(np.load(data / "variance.npy"),
        (raw.reshape(21, 24, 4) * 100.).var(1, ddof=1), rtol=1e-6)
    source_labels = np.load(data / "labels.npy")
    np.testing.assert_array_equal(source_labels, np.tile([0, 1, 2, -1], (20, 1)))
    with pytest.raises(FileExistsError):
        preprocess.build(cfg)
    build_resolution_bundle.build(Path(result["output"]), tmp_path / "labels")
    hard = np.load(tmp_path / "labels/fixture/hard_label.npy")
    np.testing.assert_array_equal(hard[:, 3], -100)
    report = prepare.prepare(cfg, "fixture", tmp_path / "prepared")
    assert report["status"] == "PASS"
    assert report["seeds"] == [17, 29, 43, 61, 89]
    for seed in prepare.SEEDS:
        manifest = prepare.read_json(tmp_path / f"prepared/seed_{seed}/manifest.json")
        assert [manifest["counts"][p]["targets"] for p in prepare.PARTITIONS] == [14, 2, 4]
        assert manifest["audit"]["train_same_fold_independent_replay"]
        assert manifest["audit"]["heldout_label_mutation_invariance"]
        assert manifest["prior"]["nontrain_construction_label_reads"] == 0
    validator = Path(build_resolution_bundle.__file__).with_name("validate_resolution_bundle.py")
    completed = subprocess.run([sys.executable, str(validator),
        "--source", result["output"], "--bundle", str(tmp_path / "labels")],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_replogle_excludes_unmapped_targets_and_matches_linear_moments(tmp_path):
    path = tmp_path / "replogle.h5ad"
    raw = synthetic_h5ad(path, "replogle", targets=5)
    output = tmp_path / "out"
    manifest = preprocess.process_replogle("fixture", path, output)
    np.testing.assert_allclose(np.load(output / "fixture/mean.npy"),
                              (raw.reshape(6, 24, 4) * 100.).mean(1)[:5])
    assert manifest["raw_count_validation"]["unmapped_rows_excluded"] == 24
    assert np.load(output / "fixture/response_group_id.npy").tolist() == [f"ENSG{i:04d}" for i in range(4)]


def test_nadig_rejects_normalized_input_instead_of_silent_count_reuse(tmp_path):
    path = tmp_path / "nadig.h5ad"
    synthetic_h5ad(path, "nadig")
    with h5py.File(path, "r+") as handle:
        handle["X"][0, 0] = 101  # Exceeds the original library denominator.
    meta = nadig_moments.inventory({"fixture": path})["fixture"]
    with pytest.raises(ValueError, match="raw integer counts"):
        nadig_moments.aggregate_moments(path, meta, block_cells=12)


def test_nadig_missing_target_identity_uses_only_cross_source_annotations(tmp_path):
    first, second = tmp_path / "first.h5ad", tmp_path / "second.h5ad"
    synthetic_h5ad(first, "nadig")
    synthetic_h5ad(second, "nadig")
    with h5py.File(first, "r+") as handle:
        handle["obs/gene_id"][-24:] = "nan"
    single = nadig_moments.inventory({"first": first})
    assert single["first"]["target_ids"][-1] == "SYMBOL:GENE_19"
    together = nadig_moments.inventory({"first": first, "second": second})
    assert together["first"]["target_ids"][-1] == "ENSG0019"
    assert together["first"]["counts"]["ENSG0019"] == 24
    assert together["first"]["missing_source_target_id_resolution"]["GENE_19"]["rule"] == "unique cross-cell obs symbol mapping"


def test_tost_same_requires_positive_evidence_and_low_support_stays_unresolved():
    controls = Moments(100, np.array([10., 10., 10., 0.]),
                       np.array([1., 1., 1., 0.]), np.array([100, 100, 100, 0]))
    treated = Moments(100, np.array([20., 10., 5., 0.]),
                      np.array([1., 1., 1., 0.]), np.array([100, 100, 100, 0]))
    assert label_pair(treated, controls)["labels"].tolist() == [0, 1, 2, -1]
    imprecise = Moments(100, np.array([10., 10., 10., 0.]),
                        np.array([10000., 10000., 10000., 0.]), np.array([100, 100, 100, 0]))
    assert label_pair(imprecise, controls)["labels"].tolist() == [-1, -1, -1, -1]
    low_support = Moments(100, treated.mean, treated.variance, np.array([9, 9, 9, 0]))
    assert label_pair(low_support, controls)["reason_codes"].tolist() == [2, 2, 2, 3]


def test_prepare_requires_explicit_portable_sources():
    with pytest.raises(ValueError, match="configure data_preprocessed_root"):
        prepare._paths({}, "fixture")


def test_strict_label_builder_rejects_nonfinite_q_values(tmp_path):
    path = tmp_path / "fixture.h5ad"
    synthetic_h5ad(path, "replogle", targets=3)
    preprocess.process_replogle("fixture", path, tmp_path / "moments")
    q_path = tmp_path / "moments/fixture/q_best.npy"
    q = np.load(q_path)
    q[0, 0] = np.nan
    np.save(q_path, q)
    with pytest.raises(ValueError, match="finite probabilities"):
        build_resolution_bundle.build(tmp_path / "moments", tmp_path / "labels")
    assert not (tmp_path / "labels").exists()


def test_label_validator_checks_original_source_hashes(tmp_path):
    path = tmp_path / "fixture.h5ad"
    synthetic_h5ad(path, "replogle", targets=3)
    source, output = tmp_path / "moments", tmp_path / "labels"
    preprocess.process_replogle("fixture", path, source)
    build_resolution_bundle.build(source, output)
    # Drift that leaves labels and point regions unchanged must still be caught.
    q_path = source / "fixture/q_best.npy"
    q = np.load(q_path)
    q[0, 0] = 0.04
    np.save(q_path, q)
    validator = Path(build_resolution_bundle.__file__).with_name("validate_resolution_bundle.py")
    completed = subprocess.run([sys.executable, str(validator), "--source", str(source),
                                "--bundle", str(output)], capture_output=True, text=True)
    assert completed.returncode == 1
    assert "source content drift: fixture/q_best.npy" in completed.stdout
