"""Synthetic CPU tests: no public datasets, weights, or task training."""
import csv
from pathlib import Path

import numpy as np
import pytest

from data_prep import cell_pool as cp
from data_prep.table3 import summarize_arrays
from experiments import cell_pool as boundary


def test_batch_split_is_disjoint_exact_and_group_independent():
    groups = np.repeat(np.arange(3), [61, 80, 121])
    batches = np.arange(len(groups)) % 5
    a = cp.assign_cell_split(groups, batches, [True, True, True])
    repeated = cp.assign_cell_split(groups, batches, [True, False, True])
    np.testing.assert_array_equal(a[groups != 1], repeated[groups != 1])
    assert np.all(repeated[groups == 1] == -1)
    for group, count in enumerate([61, 80, 121]):
        selected = groups == group
        assert np.sum(a[selected] == 0) == (count + 1) // 3
        assert np.sum(a[selected] == 1) == count - (count + 1) // 3
        for batch in np.unique(batches):
            mask = selected & (batches == batch)
            n, actual = int(mask.sum()), int(np.sum(a[mask] == 0))
            assert actual in {n // 3, (n + 2) // 3}


def test_linear_original_umi_statistics():
    raw = np.asarray([[1, 2], [2, 4], [5, 2], [4, 6], [8, 3], [9, 4]])
    umi = np.asarray([100, 200, 100, 200, 400, 200])
    groups = np.zeros(6, dtype=np.int32)
    assignment = np.asarray([0, 0, 0, 1, 1, 1])
    counts, (mean, variance, nonzero) = cp.split_statistics(raw, umi, groups, assignment, 1)
    values = raw / umi[:, None] * 1e4
    np.testing.assert_array_equal(counts, [3, 3])
    for half in (0, 1):
        expected = values[assignment == half]
        np.testing.assert_array_equal(mean[half], expected.mean(0).astype("float32"))
        np.testing.assert_allclose(variance[half], expected.var(0, ddof=1).astype("float32"))
        np.testing.assert_array_equal(nonzero[half], [3, 3])
    with pytest.raises(ValueError, match="integer"):
        cp.split_statistics(raw + .1, umi, groups, assignment, 1)
    with pytest.raises(ValueError, match="UMI"):
        cp.split_statistics(raw, np.ones(6), groups, assignment, 1)


def test_proxy_nested_scope_and_self_exclusion():
    a = np.array([-1, -1, -1, 1, -1])
    b = np.array([0, 2, -1, 0, 1])
    original = np.array([-1, 0, -1, -1, -1])
    proxy, nested = cp.proxy_mask(a, b, original, own_query=4)
    np.testing.assert_array_equal(proxy, [True, True, False, False, False])
    np.testing.assert_array_equal(nested, [True, False, False, False, False])


def test_table3_all_candidates_including_low_support():
    labels = np.array([[0, -1, 1], [2, 1, -1]], dtype=np.int8)
    reasons = np.array([[0, 7, 1], [2, 1, 4]], dtype=np.uint8)
    row = summarize_arrays(labels, reasons, ["g0", "g1"], ["g0", "g1", "g2"])
    assert row["total_candidate_pairs"] == 4
    assert row["eligible_pairs"] == 3
    assert (row["up"], row["same"], row["down"]) == (0, 1, 1)
    assert row["unresolved_percent"] == 50
    assert row["de_to_same_ratio"] == 1
    with pytest.raises(ValueError, match="disagreement"):
        summarize_arrays(labels, np.zeros_like(reasons), ["g0", "g1"], ["g0", "g1", "g2"])


def test_bh_family_precedes_nonzero_filter():
    from jev_a4.labels.label_core import Moments, adjust, label_pair
    controls = Moments(40, np.array([10., 10.]), np.array([4., 4.]), np.array([40, 40]))
    treated = Moments(40, np.array([13., 40.]), np.array([4., 4.]), np.array([40, 5]))
    result = label_pair(treated, controls)
    assert result["p_up"][1] < 1  # retained in 3G despite insufficient nonzero support
    assert not result["eligible"][1] and result["labels"][1] == -1
    expected = adjust(np.r_[result["p_up"], result["p_equivalent"], result["p_down"]]).reshape(3, 2)
    np.testing.assert_array_equal(expected[0], result["q_up"])


def make_pool_fixture(tmp_path, monkeypatch):
    import h5py
    from jev_a4.data import preprocess
    from jev_a4.labels.label_core import Moments, label_pair
    rng = np.random.default_rng(7)
    group_n = np.array([90, 90, 90])
    groups = np.repeat(np.arange(3), group_n)
    raw = np.concatenate([rng.poisson(mean, size=(90, 5)) for mean in
                          ([20]*5, [20, 40, 20, 5, 20], [20, 20, 45, 5, 21])]).astype("int32")
    umi = np.full(len(raw), 1000.)
    source, data, pool = tmp_path / "store", tmp_path / "data", tmp_path / "pool"
    source.mkdir(); data.mkdir()
    raw_path = tmp_path / "raw.h5ad"
    with h5py.File(raw_path, "w") as handle:
        handle.create_dataset("X", data=raw, chunks=(17, 2))
    axes = {"target_ids": ["g0", "g1"], "query_ids": [f"g{i}" for i in range(5)]}
    cp.write_json(data / "dataset.json", axes)
    for seed, partition in ((17, [0, 2]), (29, [2, 0])):
        directory = data / f"seed_{seed}"
        directory.mkdir()
        np.save(directory / "target_partition.npy", np.array(partition, dtype="int8"))
    metadata = {"group_ids": ["CONTROL"]+axes["target_ids"], "gene_id": axes["query_ids"],
                "n_genes": 5, "n_cells": len(raw), "groups": groups, "umi": umi,
                "batch": np.arange(len(raw)) % 4}
    monkeypatch.setattr(preprocess, "metadata_replogle", lambda path: metadata)
    statistics = [np.zeros((3, 5), dtype=dtype) for dtype in (float, float, np.int64)]
    cp.accumulate(statistics, groups, raw / umi[:, None] * 1e4, raw > 0)
    moments = cp.moments_from_sufficient(*statistics, group_n)
    for name, array in zip(("mean", "variance", "nonzero"), moments):
        np.save(source / f"{name}.npy", array)
    np.save(source / "group_n.npy", group_n)
    np.save(source / "group_id.npy", np.asarray(["CONTROL"] + axes["target_ids"]))
    np.save(source / "gene_id.npy", np.asarray(axes["query_ids"]))
    ctl = Moments(90, *(a[0].astype(float) for a in moments))
    labels = np.asarray([label_pair(Moments(90, *(a[i].astype(float) for a in moments)), ctl)["labels"] for i in (1, 2)])
    np.save(source / "labels.npy", labels)
    cp.write_json(source / "manifest.json", {"label_definition": {"ratio": 1.2, "alpha": .05,
        "min_cells": 20, "min_nonzero": 10, "q_method": "BH over 3G hypotheses per condition"}})
    return raw_path, source, data, pool


def test_stream_build_prepare_and_fidelity(tmp_path, monkeypatch):
    raw, store, data, pool = make_pool_fixture(tmp_path, monkeypatch)
    report = cp.build(raw, store, data, pool, seeds=(17, 29), gene_block=2, expected_cells=270)
    assert report["source_moments_reconstructed"] and not report["training_performed"]
    np.testing.assert_array_equal(np.load(pool / "group_n_a.npy"), [30]*3)
    np.testing.assert_array_equal(np.load(pool / "group_n_b.npy"), [60]*3)
    results = cp.fidelity(pool, tmp_path / "fig2")
    assert len(results) == 10
    by_key = {(r["rule"], r["cell_count_bin"]): r for r in results}
    assert by_key[cp.RULES[1], "all"]["b_de"] > 0
    assert by_key[cp.RULES[0], "all"]["candidate"] == 8
    with pytest.raises(ValueError, match="assertion"):
        cp.fidelity(pool, tmp_path / "fig2", expected_reference_de=999)
    prepared = cp.prepare(pool, store, data, tmp_path / "prepared")
    assert prepared["status"] == "PASS"
    for seed in (17, 29):
        rows = np.load(tmp_path / f"prepared/seed_{seed}/records.npy")
        keys = np.load(tmp_path / f"prepared/seed_{seed}/keys.npy")
        assert keys.dtype.names is None
        np.testing.assert_array_equal(keys[:, 0], rows["target"])
    with pytest.raises(ValueError, match="already exists"):
        cp.build(raw, store, data, pool, seeds=(17, 29))


def test_prediction_keys_and_matching_are_label_blind():
    axes = {"target_ids": ["a", "b"], "query_ids": ["a", "b", "c"]}
    keys = np.array([[1, 2], [1, 0]])
    t, q = boundary.validate_keys(keys, axes, np.array([0, 2]))
    np.testing.assert_array_equal(t, [1, 1])
    np.testing.assert_array_equal(q, [2, 0])
    matched, rows = boundary.match_keys(keys, np.array([[1, 0]]), 3)
    np.testing.assert_array_equal(matched, [False, True])
    np.testing.assert_array_equal(rows, [-1, 0])
    with pytest.raises(ValueError, match="self"):
        boundary.validate_keys(np.array([[1, 1]]), axes, np.array([0, 2]))
    with pytest.raises(ValueError, match="without labels"):
        boundary.validate_keys(np.array([(1, 2, 0)], dtype=cp.RECORD_DTYPE), axes, np.array([0, 2]))
    with pytest.raises(ValueError, match="TEST"):
        boundary.validate_keys(np.array([[0, 2]]), axes, np.array([0, 2]))


def make_boundary_fixture(root):
    prepared, predictions = root / "cell_pool/prepared", root / "cell_pool/predictions"
    report = {"status": "PASS", "model_seeds": [17, 29], "per_seed": {}}
    for seed in report["model_seeds"]:
        data, pred = prepared / f"seed_{seed}", predictions / f"seed_{seed}"
        data.mkdir(parents=True); pred.mkdir(parents=True)
        records = np.asarray([(0, 1, 0), (0, 2, 1), (1, 2, 2), (1, 3, 0)], dtype=cp.RECORD_DTYPE)
        keys = np.column_stack((records["target"], records["query"]))
        np.save(data / "records.npy", records)
        np.save(data / "keys.npy", keys)
        np.save(data / "original_unknown.npy", np.array([True, False, True, False]))
        report["per_seed"][str(seed)] = {"artifacts": {p.name: cp.identity(p) for p in data.glob("*.npy")}}
        np.save(pred / "keys.npy", keys)
        probs = np.eye(3)[records["label"]] * .8 + .2/3
        if seed == 29:
            probs[0] = [.2, .6, .2]
        np.save(pred / "probabilities.npy", probs.astype("float32"))
        np.save(pred / "p1_probabilities.npy", np.full((4, 3), 1/3, dtype="float32"))
        cp.write_json(pred / "report.json", {"status": "PASS", "seed": seed,
            "all_requested_records_covered": True, "task_weights_frozen": True, "training_started": False,
            "class_order": ["up", "same", "down"], "source": {"keys": cp.identity(data / "keys.npy")},
            "outputs": {p.name: cp.identity(p) for p in pred.glob("*.npy")}})
    cp.write_json(prepared / "report.json", report)


def test_boundary_evaluation_uses_sample_sd_and_seals(tmp_path):
    make_boundary_fixture(tmp_path)
    rows = boundary.evaluate(tmp_path)
    assert len(rows) == 20
    full = next(r for r in rows if r["subset"] == boundary.SUBSETS[0] and r["model"] == "S1Pert" and r["metric"] == "macro_f1")
    assert full["n"] == 2 and full["std"] > 0 and full["mean_records"] == 4
    report = cp.read_json(tmp_path / "paper/fig2/boundary_report.json")
    values = [r["macro_f1"] for r in report["per_seed"] if r["subset"] == boundary.SUBSETS[0] and r["model"] == "S1Pert"]
    assert full["std"] == pytest.approx(np.std(values, ddof=1))
    path = tmp_path / "cell_pool/predictions/seed_17/keys.npy"
    np.save(path, np.zeros((4, 2), dtype=int))
    with pytest.raises(ValueError, match="changed"):
        boundary.evaluate(tmp_path)


def test_fig2_uses_derived_counts_and_checks_summary(tmp_path, monkeypatch):
    from plotting.plot_fig2_label_boundary import plot, load_inputs
    raw, store, data, pool = make_pool_fixture(tmp_path, monkeypatch)
    cp.build(raw, store, data, pool, seeds=(17, 29), gene_block=2)
    rows = cp.fidelity(pool, tmp_path / "paper/fig2")
    total = next(r["b_de"] for r in rows if r["rule"] == cp.RULES[1] and r["cell_count_bin"] == "all")
    make_boundary_fixture(tmp_path)
    boundary.evaluate(tmp_path)
    plot(tmp_path / "paper/fig2", expected_reference_de=total)
    audit = cp.read_json(tmp_path / "paper/fig2/fig2_audit.json")
    assert audit["reference_de_pairs"] == total
    assert audit["font"]["requested"] == "Times New Roman"
    assert audit["font"]["selected"] in ("Times New Roman", "DejaVu Serif")
    assert audit["paper_filename"] == "fig_label_boundary.pdf"
    assert (tmp_path / "paper/fig2/fig_label_boundary.pdf").stat().st_size > 1000
    assert (tmp_path / "paper/fig2/fig2_label_boundary.pdf").stat().st_size > 1000
    path = tmp_path / "paper/fig2/table_boundary_generalization.csv"
    with path.open(newline="") as stream:
        values = list(csv.DictReader(stream))
    values[0]["mean"] = "0.1234567"
    cp.csvwrite(path, values)
    with pytest.raises(ValueError, match="per-seed"):
        load_inputs(path.parent)


def test_table3_latex_escapes_cell_names():
    from plotting.make_table3_stats import render
    row = summarize_arrays(np.array([[1]]), np.array([[1]]), ["t"], ["g"])
    text = render([{**row, "cell": "k562_essential"}])
    assert "k562\\_essential" in text
    assert "\\begin{tabular}" in text


def test_frozen_extension_head_with_synthetic_generic_tail(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from jev_a4.core.vendor.prior_model import make_prior_model
    torch.set_num_threads(1)
    rng = np.random.default_rng(123)
    hidden = 16
    seed_cache = tmp_path / "seed_17"
    seed_cache.mkdir()
    arrays = {"condition_block24.npy": rng.normal(size=(2, 16, hidden)).astype("float16"),
              "query_block24.npy": rng.normal(size=(3, 16, hidden)).astype("float16")}
    for name, value in arrays.items():
        np.save(tmp_path / name, value)
    for name in ("cards_final.npy", "cards_block24.npy"):
        np.save(seed_cache / name, rng.normal(size=(2, 4, hidden)).astype("float16"))

    class Shared:
        cfg = {}
        condition_num = np.zeros((2, 4), dtype="float32")
        query_num = np.zeros((3, 4), dtype="float32")
        condition = arrays["condition_block24.npy"]
        query = arrays["query_block24.npy"]

        def tensor(self, value, floating=False):
            result = torch.from_numpy(np.array(value, copy=True))
            return result.float() if floating else result

        def gather(self, name, indices):
            return self.tensor(getattr(self, name)[indices], floating=True)

    model = make_prior_model({"decoder_width": 32, "decoder_blocks": 2, "attention_heads": 4,
                              "ffn_width": 64, "backbone_hidden_width": hidden, "dropout": .1}, 12)
    monkeypatch.setattr(boundary, "frozen_tail", lambda cfg: lambda states, mask: states)
    features = np.zeros((3, 12), dtype="float32")
    features[:, :3] = 1/3
    features[:, -1] = 1
    runner = boundary.FrozenBoundaryPredictor(model, SimpleNamespace(shared=Shared()), tmp_path,
                                             tmp_path, 17, features, np.array([0, 1, 0]))
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    probabilities = runner.predict(np.array([0, 1]), np.array([2, 0]))
    boundary.validate_probabilities(probabilities, 2)
    assert not any(p.requires_grad for p in model.parameters())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_table3_file_generation_and_tamper_rejection(tmp_path):
    from data_prep.table3 import generate
    data, strict = tmp_path / "data/rpe1", tmp_path / "strict_labels/rpe1"
    data.mkdir(parents=True); strict.mkdir(parents=True)
    cp.write_json(data / "dataset.json", {"target_ids": ["t"], "query_ids": ["g"]})
    np.save(strict / "hard_label.npy", np.array([[1]], dtype="int8"))
    np.save(strict / "resolution_code.npy", np.array([[1]], dtype="uint8"))
    cp.write_json(strict / "manifest.json", {"arrays": {p.name: cp.identity(p) for p in strict.glob("*.npy")}})
    cp.write_json(data / "preparation_report.json", {"status": "PASS",
        "source": {"strict_manifest": cp.identity(strict / "manifest.json")},
        "artifacts": {"dataset.json": cp.identity(data / "dataset.json")}})
    rows = generate(tmp_path, cells=("rpe1",))
    assert rows[0]["same"] == 1 and rows[0]["unresolved_pairs"] == 0
    np.save(strict / "hard_label.npy", np.array([[2]], dtype="int8"))
    with pytest.raises(ValueError, match="payload changed"):
        generate(tmp_path, cells=("rpe1",))
