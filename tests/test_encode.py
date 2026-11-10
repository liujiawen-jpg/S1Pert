"""CPU audits of A4 semantic compilation and encoding identity contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "src/jev_a4/core/encode.py"
spec = importlib.util.spec_from_file_location("a4_encoder_tested", MODULE)
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)


def metadata():
    return {"dataset": "toy", "target_ids": ["ENSG_A", "SYMBOL:B"],
        "target_symbols": ["A", "B"], "target_query_indices": [0, -1],
        "query_ids": ["ENSG_A", "ENSG_D"], "query_symbols": ["A", "D"],
        "control_mean": [2., 4.], "control_variance": [1., 2.],
        "control_detection": [.2, .5], "duplicate_symbol_ids": {}}


class FakeSession:
    model_load_sha256 = "fake-smoke-contextual-only"

    def encode_texts(self, texts, block, final, count, batch_size, progress=None):
        for i, text in enumerate(texts):
            block[i] = float(len(text) % 5) / 10
            final[i] = float(len(text) % 7) / 10
        return {"maximum_tokens": 10, "truncated": 0}

    def encode_joint(self, tuples, condition, query, cards, present, absent, batch_size, progress=None):
        for i, (c, q, k) in enumerate(tuples):
            present[i, 0] = condition[c].mean(0)
            present[i, 1] = query[q].mean(0)
            present[i, 2] = cards[k].mean(0)
            present[i, 3] = .3
            absent[i] = present[i]
            absent[i, 2] = 0.


class CompilationTests(unittest.TestCase):
    def test_semantic_facts_identities_units_and_questions(self):
        texts, coverage = enc.compile_texts(metadata(), {"A": "A regulates a program.", "D": "D protein."})
        self.assertIn("A (ENSG_A)", texts["condition"][0])
        self.assertIn("D (ENSG_D)", texts["query"][1])
        self.assertIn("variance=2", texts["query"][1])
        self.assertIn("Question:", texts["query"][1])
        self.assertIn("Options:", texts["query"][1])
        self.assertIn(enc.TASK_PREFIX, texts["condition"][0])
        self.assertIn("mean=unknown", texts["condition"][1])
        self.assertEqual(coverage["query_function_present"], 2)

    def test_numeric_symbols_fail_not_silent_function_missing(self):
        m = metadata(); m["query_symbols"][1] = "4230"
        with self.assertRaisesRegex(ValueError, "numeric gene symbol"):
            enc.compile_texts(m, {})

    def test_duplicate_symbols_do_not_assign_ambiguous_function(self):
        m = metadata(); m["duplicate_symbol_ids"] = {"D": ["ENSG_D", "ENSG_D2"]}
        texts, coverage = enc.compile_texts(m, {"D": "SECRET WRONG FUNCTION"})
        self.assertNotIn("SECRET", texts["query"][1])
        self.assertIn("multiple measured gene identifiers", texts["query"][1])
        self.assertEqual(coverage["query_function_present"], 0)

    def test_illegal_control_values_fail(self):
        for column, bad in (("control_mean", -1.), ("control_variance", np.nan), ("control_detection", 1.2)):
            m = metadata(); m[column][0] = bad
            with self.assertRaises(ValueError):
                enc.compile_texts(m, {})

    def test_only_public_genes_table_is_read(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "public.sqlite"
            con = sqlite3.connect(database)
            con.executescript("CREATE TABLE genes(symbol TEXT,accession TEXT,protein_name TEXT,function_text TEXT);"
                "CREATE TABLE empirical_events(secret TEXT);"
                "INSERT INTO genes VALUES('A','P1','Protein A','Function A');"
                "INSERT INTO empirical_events VALUES('FORBIDDEN RESPONSE');")
            con.commit(); con.close()
            result = enc.load_functions(database)
            self.assertEqual(result["A"], "Protein A Function A")
            self.assertNotIn("FORBIDDEN", str(result))

    def test_token_budget_is_exact_and_never_truncated(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens): return list(range(len(text)))
        ids, lengths = enc.tokenize_checked(Tokenizer(), ["abcd"], 4)
        self.assertEqual(lengths.tolist(), [4])
        with self.assertRaisesRegex(ValueError, "no truncation"):
            enc.tokenize_checked(Tokenizer(), ["abcde"], 4)
        with self.assertRaises(ValueError):
            enc.tokenize_checked(Tokenizer(), [""], 4)

    def test_compress_landmarks_match_selected_A4(self):
        x = np.arange(40 * 8, dtype=np.float32).reshape(40, 8)
        c = enc.compress16(x)
        self.assertEqual(c.shape, (16, 8))
        np.testing.assert_array_equal(c[-4:], x[-4:])
        d = enc.compress4(x)
        self.assertEqual(d.shape, (4, 8))
        np.testing.assert_array_equal(d[-1], x[-1])
        self.assertTrue(np.isfinite(enc.compress16(x[:1])).all())
        with self.assertRaises(ValueError): enc.compress4(x[:0])

    def test_card_categories_match_original_thresholds(self):
        self.assertEqual(enc.categories([.8, .1, .1], 70, 0),
            ("upregulated", "strong", "broad", "cell-and-query specific"))
        self.assertEqual(enc.categories([.34, .33, .33], 2, 1),
            ("mixed", "weak", "limited", "cell-wide fallback"))
        self.assertEqual(len(enc.finite_cards()), 109)
        self.assertIn("not the answer", enc.finite_cards()[1]["text"])

    def test_card_ids_use_prior_only_and_unavailable_is_zero(self):
        features = np.zeros((3, 12), dtype=np.float32)
        features[:, :3] = [.8, .1, .1]; features[:, 11] = [1, 0, 1]
        ids = enc.compile_card_ids(features, np.array([[10, 5, 1], [0, 0, 0], [10, 5, 1]]), np.zeros(3))
        self.assertEqual(int(ids[1]), 0)
        self.assertEqual(int(ids[0]), int(ids[2]))
        with self.assertRaises(ValueError):
            enc.compile_card_ids(features[:, :11], np.zeros((3, 3)), np.zeros(3))

    def test_structural_tuples_are_label_invariant_and_record_aligned(self):
        records = np.array([(2, 3, 0), (1, 5, 1), (2, 3, 2)],
            dtype=[("target", "i4"), ("query", "i4"), ("label", "i1")])
        cards = np.array([5, 8, 5])
        tuples, mapping = enc.structural_tuples(records, cards)
        self.assertEqual(len(tuples), 2)
        np.testing.assert_array_equal(tuples[mapping], np.column_stack([records["target"], records["query"], cards]))
        changed = records.copy(); changed["label"] = [2, 0, 1]
        t2, m2 = enc.structural_tuples(changed, cards)
        np.testing.assert_array_equal(tuples, t2); np.testing.assert_array_equal(mapping, m2)


class CacheTests(unittest.TestCase):
    def make_data(self, directory):
        m = metadata()
        enc.atomic_json(directory / "dataset.json", m)
        enc.atomic_npy(directory / "condition_numeric.npy", np.zeros((2, 4), dtype=np.float32))
        enc.atomic_npy(directory / "query_numeric.npy", np.zeros((2, 4), dtype=np.float32))
        database = directory / "public.sqlite"
        con = sqlite3.connect(database)
        con.execute("CREATE TABLE genes(symbol TEXT,accession TEXT,protein_name TEXT,function_text TEXT)")
        con.execute("INSERT INTO genes VALUES('A','P1','A','Function A')"); con.commit(); con.close()
        for partition in enc.PARTITIONS:
            dst = directory / "seed_17" / partition
            enc.atomic_npy(dst / "records.npy", np.array([(0, 1, 0), (1, 0, 2)],
                dtype=[("target", "i4"), ("query", "i4"), ("label", "i1")]))
            features = np.zeros((2, 12), dtype=np.float32)
            features[:, :3] = [.8, .1, .1]; features[:, 11] = 1.
            enc.atomic_npy(dst / "prior_features.npy", features)
            enc.atomic_npy(dst / "prior_raw_counts.npy", np.array([[8, 1, 1], [8, 1, 1]], dtype=np.int32))
            enc.atomic_npy(dst / "prior_fallback.npy", np.zeros(2, dtype=np.int8))
        return {"knowledge_db": str(database), "max_tokens": 1024}

    def test_shared_and_seed_cache_schema_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "data"; directory.mkdir()
            cache = Path(temp) / "cache"
            cfg = self.make_data(directory)
            with patch.object(enc, "_model_identity", return_value={"test": "frozen-contextual"}):
                shared = enc.encode_dataset(cfg, "toy", directory, cache, session=FakeSession())
                seed = enc.encode_split(cfg, "toy", 17, directory, cache, session=FakeSession())
                self.assertEqual(shared["status"], "PASS")
                self.assertFalse(shared["word_embedding_average_used"])
                self.assertEqual(seed["status"], "PASS")
                self.assertEqual(np.load(cache / "condition_final32.npy").shape, (2, 16, 2560))
                self.assertEqual(np.load(cache / "seed_17" / "train" / "joint_present.npy").shape, (2, 4, 2560))
                self.assertTrue(np.all(np.load(cache / "seed_17" / "train" / "joint_absent.npy")[:, 2] == 0))
                enc.encode_split(cfg, "toy", 17, directory, cache, session=FakeSession())
                with (cache / "seed_17" / "train" / "joint_ids.npy").open("ab") as stream:
                    stream.write(b"tamper")
                with self.assertRaisesRegex(ValueError, "payload changed"):
                    enc.encode_split(cfg, "toy", 17, directory, cache, session=FakeSession())

    def test_vendor_source_hashes_match_declared_sources(self):
        directory = MODULE.parent / "vendor"
        manifest = enc.read_json(directory / "source_manifest.json")
        for name, row in manifest["files"].items():
            self.assertEqual(enc.sha256_file(directory / name), row["current_sha256"])


if __name__ == "__main__":
    unittest.main()
