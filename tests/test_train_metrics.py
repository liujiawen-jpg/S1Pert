import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytest

from jev_a4.core.metrics import average_precision, classification_metrics
from jev_a4.core.train import (PairSampler, Partition, SharedCache, evidence_keep,
                   exact_joint_loss, load_selected, model_config, settings,
                   vendor_module)


class MetricTests(unittest.TestCase):
    def test_perfect_three_classes(self):
        result = classification_metrics([0, 1, 2], np.eye(3), [2, 3, 4])
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(result["balanced_accuracy"], 1)
        self.assertEqual(result["nll"], 0)
        self.assertEqual(result["brier_multiclass_sum"], 0)
        self.assertEqual(result["changed"]["average_precision"], 1)

    def test_all_same_does_not_inflate_macro(self):
        result = classification_metrics([0, 1, 2], np.tile([0., 1., 0.], (3, 1)))
        self.assertAlmostEqual(result["macro_f1"], 1 / 6)
        self.assertAlmostEqual(result["balanced_accuracy"], 1 / 3)
        self.assertEqual(result["per_class"]["up"]["recall"], 0)

    def test_average_precision_tied_scores(self):
        self.assertAlmostEqual(average_precision([True, False, True, False], [.2] * 4), .5)
        self.assertIsNone(average_precision([False, False], [.1, .9]))
        self.assertEqual(average_precision([True, False], [.9, .1]), 1)

    def test_target_equal_weighting_not_pooling(self):
        labels = np.array([0] * 9 + [1])
        p = np.tile([1., 0., 0.], (10, 1))
        result = classification_metrics(labels, p, [10] * 9 + [11])
        self.assertAlmostEqual(result["accuracy"], .9)
        self.assertAlmostEqual(result["equal_target_weighted_confusion_metrics"]["accuracy"], .5)

    def test_json_finite_missing_class(self):
        result = classification_metrics([1, 1], [[.1, .8, .1], [.1, .8, .1]])
        json.dumps(result, allow_nan=False)
        self.assertIsNone(result["per_class"]["down"]["average_precision"])

    def test_invalid_probabilities_rejected(self):
        for p in ([[.2, .2, .2]], [[float("nan"), 0, 1]], [[-.1, 1.1, 0]]):
            with self.assertRaises(ValueError):
                classification_metrics([0], p)


class PairTests(unittest.TestCase):
    def setUp(self):
        self.records = np.empty(30, dtype=[("target", "i4"), ("query", "i4"), ("label", "i1")])
        self.records["target"] = np.repeat(np.arange(6), 5)
        self.records["query"] = np.tile(np.arange(5), 6)
        self.records["label"] = (self.records["target"] + self.records["query"]) % 3

    def test_sampler_label_query_target_contract(self):
        sampler = PairSampler(self.records)
        for kind in ("changed_same", "up_down"):
            a, b = sampler.sample(kind, 2048, np.random.default_rng(5))
            np.testing.assert_array_equal(self.records[a]["query"], self.records[b]["query"])
            self.assertTrue(np.all(self.records[a]["target"] != self.records[b]["target"]))
            if kind == "changed_same":
                self.assertTrue(np.all(self.records[a]["label"] != 1))
                self.assertTrue(np.all(self.records[b]["label"] == 1))
            else:
                self.assertTrue(np.all(self.records[a]["label"] == 0))
                self.assertTrue(np.all(self.records[b]["label"] == 2))

    def test_sampler_is_reproducible(self):
        sampler = PairSampler(self.records)
        a, b = sampler.sample("up_down", 32, np.random.default_rng(43))
        c, d = sampler.sample("up_down", 32, np.random.default_rng(43))
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)

    def test_sampler_empty_opposite_classes(self):
        self.records["label"] = 1
        sampler = PairSampler(self.records)
        a, b = sampler.sample("up_down", 32, np.random.default_rng(43))
        self.assertEqual(len(a), 0)
        self.assertEqual(len(b), 0)

    def test_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError):
            PairSampler(np.r_[self.records, self.records[:1]])

    def test_no_replacement_epoch_exposure(self):
        seen = np.zeros(117, dtype=int)
        for epoch in range(1, 12):
            schedule = np.random.default_rng(epoch).permutation(len(seen))
            for start in range(0, len(schedule), 16):
                seen[schedule[start:start + 16]] += 1
            np.testing.assert_array_equal(seen, np.full(len(seen), epoch))


class TrainingContractTests(unittest.TestCase):
    def test_locked_A4_architecture(self):
        self.assertEqual(settings({})["effective_batch"], 4096)
        with self.assertRaises(ValueError):
            settings({"decoder_width": 256})

    def test_evidence_keep_determinism(self):
        a = evidence_keep(1024, .1, np.random.default_rng(17))
        b = evidence_keep(1024, .1, np.random.default_rng(17))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(a.any() and (~a).any())

    def test_nonconverged_test_blocked_before_torch_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "convergence.json").write_text(json.dumps({"status": "NOT_CONVERGED"}))
            with self.assertRaises(PermissionError):
                load_selected({}, "fake", 17, path, path, path, require_converged=True)

    def test_exact_hierarchical_loss_equals_probability_nll(self):
        torch = pytest.importorskip("torch")
        changed = torch.tensor([.3, -.7, .9], requires_grad=True)
        direction = torch.tensor([-.4, .2, .8], requires_grad=True)
        labels = torch.tensor([0, 1, 2])
        loss = exact_joint_loss(changed, direction, labels)
        c, d = changed.sigmoid(), direction.sigmoid()
        probs = torch.stack([c * d, 1 - c, c * (1 - d)], dim=1)
        expected = -torch.log(probs[torch.arange(3), labels]).mean()
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        loss.backward()
        self.assertEqual(float(direction.grad[1]), 0)
        self.assertTrue(torch.isfinite(changed.grad).all())

    def test_fresh_head_direct_output_and_gradients(self):
        torch = pytest.importorskip("torch")
        torch.set_num_threads(1)
        cfg = {"decoder_width": 32, "decoder_blocks": 2, "attention_heads": 4,
               "ffn_width": 64, "backbone_hidden_width": 16, "dropout": .1}
        model = vendor_module().make_prior_model(model_config(cfg), 12)
        n = 3
        a = torch.randn(n, 16, 16)
        m = torch.ones(n, 16, dtype=torch.bool)
        card = torch.randn(n, 4, 16)
        card_mask = torch.ones(n, 4, dtype=torch.bool)
        changed, direction = model(a, m, torch.randn(n, 4), a, m, torch.randn(n, 4),
                                   card, card_mask, torch.randn(n, 12),
                                   joint=card, joint_mask=card_mask)
        probabilities = model.probabilities(changed, direction)
        torch.testing.assert_close(probabilities.sum(1), torch.ones(n))
        exact_joint_loss(changed, direction, torch.tensor([0, 1, 2])).backward()
        self.assertGreater(float(model.changed_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.direction_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.query_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.joint_attention.o.weight.grad.abs().sum()), 0)

    def test_one_keep_mask_controls_semantic_numeric_and_joint(self):
        torch = pytest.importorskip("torch")
        class TinyShared:
            device = "cpu"
            condition_num = np.zeros((2, 4), dtype="f4")
            query_num = np.zeros((2, 4), dtype="f4")
            def gather(self, field, ids):
                return torch.zeros(len(ids), 16, 16)
            def tensor(self, array, floating=False):
                value = torch.from_numpy(np.array(array, copy=True))
                return value.float() if floating else value
        partition = object.__new__(Partition)
        partition.shared = TinyShared()
        partition.records = np.array([(0, 0, 0), (1, 1, 2)],
                                     dtype=[("target", "i4"), ("query", "i4"), ("label", "i1")])
        partition.features = np.ones((2, 12), dtype="f4")
        partition.cards = np.ones((2, 4, 16), dtype="f2")
        partition.card_ids = np.array([0, 1])
        partition.joint_ids = np.array([0, 1])
        partition.joint_present = np.full((2, 4, 16), 2, dtype="f2")
        partition.joint_absent = np.full((2, 4, 16), 9, dtype="f2")
        batch = partition.batch(np.array([0, 1]), np.array([True, False]))
        self.assertTrue(batch[7][0].all())
        self.assertFalse(batch[7][1].any())
        torch.testing.assert_close(batch[8][0], torch.ones(12))
        torch.testing.assert_close(batch[8][1, :3], torch.full((3,), 1 / 3))
        self.assertEqual(float(batch[8][1, 3:].abs().sum()), 0)
        self.assertEqual(float(batch[10][0].mean()), 2)
        self.assertEqual(float(batch[10][1].mean()), 9)


if __name__ == "__main__":
    unittest.main()
