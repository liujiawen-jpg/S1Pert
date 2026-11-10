"""Leakage and identity tests for independently split strict supervision."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import json

import numpy as np

from jev_a4.core import prepare


class SplitTests(unittest.TestCase):
    def test_exact_cover_712_whole_target_and_distinct_seeds(self):
        ids = [f"ENSG{i:08d}" for i in range(101)]
        hashes = []
        for seed in prepare.SEEDS:
            p, f = prepare.split_targets(ids, seed)
            self.assertEqual(np.bincount(p).tolist(), [70, 10, 21])
            self.assertTrue(np.all((f[p == 0] >= 0) & (f[p == 0] < 5)))
            self.assertTrue(np.all(f[p != 0] == -1))
            hashes.append(prepare.array_sha256(p))
        self.assertEqual(len(set(hashes)), 5)

    def test_axis_order_does_not_change_id_assignment(self):
        ids = [f"ENSG{i:08d}" for i in range(40)]
        p, f = prepare.split_targets(ids, 17)
        shuffled = ids[::-1]
        p2, f2 = prepare.split_targets(shuffled, 17)
        self.assertEqual(dict(zip(ids, p)), dict(zip(shuffled, p2)))
        self.assertEqual(dict(zip(ids, f)), dict(zip(shuffled, f2)))

    def test_duplicate_ids_fatal(self):
        with self.assertRaises(ValueError):
            prepare.split_targets(["same"] * 20, 17)


class PriorTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"ENSG{i:08d}" for i in range(40)]
        self.partition, self.fold = prepare.split_targets(self.ids, 17)
        self.labels = np.array([[(t + q) % 3 for q in range(8)] for t in range(40)], dtype=np.int8)
        self.mask = np.ones_like(self.labels, dtype=bool)
        self.labels[0, 3] = -100
        self.mask[0, 3] = False
        self.records = {p: prepare.make_records(self.labels, self.mask,
            np.flatnonzero(self.partition == code), np.full(40, -1))[0]
            for code, p in enumerate(prepare.PARTITIONS)}

    def test_heldout_label_mutation_invariance(self):
        original, audit = prepare.prior_arrays(self.records["train"], self.records,
            self.partition, self.fold, 8)
        changed = {p: r.copy() for p, r in self.records.items()}
        changed["val"]["label"] = -100
        changed["test"]["label"] = 99
        replay, _ = prepare.prior_arrays(self.records["train"], changed,
            self.partition, self.fold, 8)
        for p in prepare.PARTITIONS:
            for key in original[p]:
                np.testing.assert_array_equal(original[p][key], replay[p][key])
        self.assertEqual(audit["nontrain_construction_label_reads"], 0)

    def test_same_fold_labels_do_not_affect_own_fold_evidence(self):
        original, _ = prepare.prior_arrays(self.records["train"], self.records,
            self.partition, self.fold, 8)
        changed = {p: r.copy() for p, r in self.records.items()}
        rowfold = self.fold[changed["train"]["target"]]
        changed["train"]["label"][rowfold == 0] = 2
        replay, _ = prepare.prior_arrays(changed["train"], changed,
            self.partition, self.fold, 8)
        for key in original["train"]:
            np.testing.assert_array_equal(original["train"][key][rowfold == 0],
                                          replay["train"][key][rowfold == 0])

    def test_feature_width_and_normalization(self):
        arrays, _ = prepare.prior_arrays(self.records["train"], self.records,
            self.partition, self.fold, 8)
        for p, r in self.records.items():
            a = arrays[p]["features"]
            self.assertEqual(a.shape, (len(r), 12))
            self.assertEqual(a.dtype, np.float32)
            np.testing.assert_allclose(a[:, :3].sum(axis=1), 1., atol=1e-6)
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue(np.all(a[:, 9] == 0))

    def test_own_target_and_unresolved_excluded(self):
        own = np.arange(40) % 8
        r, s = prepare.make_records(self.labels, self.mask, np.arange(40), own)
        self.assertTrue(np.all(r["query"] != own[r["target"]]))
        self.assertTrue(np.isin(r["label"], [0, 1, 2]).all())
        self.assertTrue(np.all(s["query"] == own[s["target"]]))

    def test_invalid_mask_is_rejected(self):
        mask = self.mask.copy()
        mask[0, 3] = True
        with self.assertRaises(ValueError):
            prepare.make_records(self.labels, mask, np.arange(40), np.full(40, -1))

    def test_nontrain_fit_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare.prior_arrays(self.records["test"], self.records, self.partition, self.fold, 8)


class MappingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py unavailable")
    def test_historical_h5_categories_decode_and_target_metadata(self):
        import h5py
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metadata.h5ad"
            with h5py.File(path, "w") as h:
                var = h.create_group("var")
                var.create_dataset("gene_id", data=np.array(["ENSG1", "ENSG2"], dtype=h5py.string_dtype()))
                cats = var.create_dataset("names", data=np.array(["GENE_A", "GENE_B"], dtype=h5py.string_dtype()))
                symbols = var.create_dataset("gene_name", data=np.array([1, 0], dtype=np.int32))
                symbols.attrs["categories"] = cats.ref
                obs = h.create_group("obs")
                obs.create_dataset("gene_id", data=np.array(["ENSG3"], dtype=h5py.string_dtype()))
                obs.create_dataset("gene", data=np.array(["GENE_C"], dtype=h5py.string_dtype()))
            mapping, audit = prepare.decode_raw_symbol_mapping(path)
            self.assertEqual(mapping, {"ENSG1": "GENE_B", "ENSG2": "GENE_A"})
            self.assertEqual(audit["target_mapping"], {"ENSG3": "GENE_C"})
            self.assertFalse(audit["expression_matrix_read"])

    def test_duplicate_symbols_retained_as_distinct_id_warning(self):
        self.assertEqual(prepare._duplicate_symbols(["ENSG1", "ENSG2", "ENSG3"], ["A", "A", "B"]),
                         {"A": ["ENSG1", "ENSG2"]})

    def test_unique_source_symbol_identity_resolves_self_and_canonical_target(self):
        indices, canonical, audit = prepare.resolve_target_identities(
            ["ENSG1", "ENSG2"], ["A", "B"], ["SYMBOL:A", "ENSG3"], ["A", "C"])
        self.assertEqual(indices, [0, -1])
        self.assertEqual(canonical, ["ENSG1", "ENSG3"])
        self.assertEqual(audit["resolved_alias_count"], 1)

    def test_symbol_alias_collision_blocks_target_split(self):
        with self.assertRaisesRegex(ValueError, "one canonical identity"):
            prepare.resolve_target_identities(["ENSG1"], ["A"], ["SYMBOL:A", "ENSG1"], ["A", "A"])

    def test_ambiguous_source_symbol_does_not_guess_query(self):
        with self.assertRaisesRegex(ValueError, "do not guess"):
            prepare.resolve_target_identities(["ENSG1", "ENSG2"], ["A", "A"], ["SYMBOL:A"], ["A"])

    def test_distinct_explicit_ensembl_ids_are_not_symbol_merged(self):
        indices, canonical, audit = prepare.resolve_target_identities(
            ["ENSG1"], ["A"], ["ENSG2"], ["A"])
        self.assertEqual(indices, [-1])
        self.assertEqual(canonical, ["ENSG2"])
        self.assertEqual(audit["resolved_alias_count"], 0)


class IntegrationTests(unittest.TestCase):
    def test_full_prepare_only_control_numeric_resume_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "source" / "fixture"
            label = root / "labels" / "fixture"
            data.mkdir(parents=True)
            label.mkdir(parents=True)
            genes = np.array([f"ENSG{i:04d}" for i in range(6)])
            targets = np.array([f"ENSG{i:04d}" for i in range(20)])
            symbols = np.array([f"GENE_{i}" for i in range(6)])
            groups = np.concatenate((["CONTROL"], targets))
            names = np.array(["CONTROL"] + [f"GENE_{i}" for i in range(20)])
            arrays = {"gene_id": genes, "gene_symbol": symbols, "response_group_id": targets,
                      "group_id": groups, "group_name": names, "group_n": np.full(21, 50)}
            for key in ("mean", "variance", "nonzero"):
                values = np.full((21, 6), 999999., dtype=np.float64)
                values[0] = 2. if key != "nonzero" else 25.
                arrays[key] = values
            for key, values in arrays.items():
                np.save(data / (key + ".npy"), values)
            prepare.dump_json(data / "manifest.json", {"normalization": "count / UMI * 10000"})
            labels = np.array([[(t + g) % 3 for g in range(6)] for t in range(20)], dtype=np.int8)
            labels[3, :] = -100
            mask = labels >= 0
            np.save(label / "hard_label.npy", labels)
            np.save(label / "supervision_mask.npy", mask)
            prepare.dump_json(label / "manifest.json", {"schema": "decider.response_resolution_labels.v3",
                               "statistical_contract": {"ratio_margin": 1.2}})
            cfg = {"data_preprocessed_root": str(root / "source"), "label_resolution_root": str(root / "labels"),
                   "seeds": list(prepare.SEEDS), "exclude_self_pairs": True}
            out = root / "prepared"
            report = prepare.prepare(cfg, "fixture", out)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["split_assignments_distinct"])
            numeric = np.load(out / "query_numeric.npy")
            np.testing.assert_allclose(numeric[:, :2], np.log1p(2.), atol=1e-6)
            np.testing.assert_allclose(numeric[:, 2], .5)
            cond = np.load(out / "condition_numeric.npy")
            self.assertTrue(np.all(cond[6:] == 0))
            for seed in prepare.SEEDS:
                manifest = prepare.read_json(out / f"seed_{seed}" / "manifest.json")
                self.assertTrue(manifest["audit"]["train_same_fold_independent_replay"])
                self.assertEqual(sum(c["targets"] for c in manifest["counts"].values()), 20)
                self.assertEqual(sum(c["targets_with_zero_strict_downstream_pairs"] for c in manifest["counts"].values()), 1)
            self.assertEqual(prepare.prepare(cfg, "fixture", out), report)
            with (out / "query_numeric.npy").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaises(ValueError):
                prepare.prepare(cfg, "fixture", out)


if __name__ == "__main__":
    unittest.main()
