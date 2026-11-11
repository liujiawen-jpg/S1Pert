"""Frozen S1Pert/P1 boundary inference and cell-pool evaluation (no training).

Prediction accepts only structural keys. Pool-B labels are opened exclusively
by the separate evaluation command. New keys run the same frozen decoder tail
and selected direct task head; known keys reuse that checkpoint's sealed TEST
probabilities after a predeclared execution-parity check.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_prep.cell_pool import (CELL, SEEDS, csvwrite, identity, load_array,
                                 read_json, require, sha256, work_root, write_json)

PARITY_MAX_ABS = .01
PARITY_MEAN_ABS = .002
SUBSETS = ("A_UNRESOLVED_B_RESOLVED", "FULL_UNKNOWN_B_RESOLVED")
METRICS = ("macro_f1", "balanced_accuracy", "up_f1", "down_f1", "same_f1")


def validate_probabilities(values, n):
    require(values.shape == (n, 3) and np.isfinite(values).all() and (values >= 0).all()
            and (values <= 1).all() and np.allclose(values.sum(1), 1, atol=2e-5),
            "invalid [up,same,down] probabilities")


def train_priors(data, cache, seed):
    """Replay the original TRAIN-only query features and validate every TEST row."""
    from jev_a4.core.encode import compile_card_ids
    from jev_a4.core.prepare import _probability_arrays
    data, cache = Path(data), Path(cache)
    axes = read_json(data / "dataset.json")
    partition = load_array(data / f"seed_{seed}/target_partition.npy")
    manifest = read_json(data / f"seed_{seed}/manifest.json")
    prior = manifest["prior"]
    require(prior["labels_used_to_construct"] == ["TRAIN"] and
            prior["old_prediction_or_checkpoint_read"] is False and prior["source_mean_output_anchor"] is False
            and prior["residual_gate_blend_or_calibration"] is False, "invalid TRAIN-evidence contract")
    train = load_array(data / f"seed_{seed}/train/records.npy")
    require(np.isin(train["label"], (0, 1, 2)).all() and (partition[train["target"]] == 0).all(),
            "prior reconstruction requires strict TRAIN records")
    counts = np.zeros((len(axes["query_ids"]), 3), dtype=np.int64)
    np.add.at(counts, (train["query"], train["label"]), 1)
    support = counts.sum(1)
    fallback = (support == 0).astype(np.int8)
    raw = counts.copy()
    raw[fallback == 1] = counts.sum(0)
    features = _probability_arrays(raw, support, fallback,
        np.full(len(raw), int(np.sum(partition == 0)), dtype=np.int64),
        float(prior["support_log_denominator"]))["features"]
    cards = compile_card_ids(features, raw, fallback)
    strict = load_array(data / f"seed_{seed}/test/records.npy")
    np.testing.assert_array_equal(features[strict["query"]], load_array(data / f"seed_{seed}/test/prior_features.npy"))
    np.testing.assert_array_equal(cards[strict["query"]], load_array(cache / f"seed_{seed}/test/card_ids.npy"))
    return features, cards


def validate_keys(keys, axes, partition):
    keys = np.asarray(keys)
    require(keys.ndim == 2 and keys.shape[1] == 2 and keys.dtype.kind in "iu", "keys must be integer [N,2], without labels")
    t, q = keys[:, 0], keys[:, 1]
    require((t >= 0).all() and (t < len(axes["target_ids"])).all()
            and (q >= 0).all() and (q < len(axes["query_ids"])).all(), "structural key outside axes")
    require((partition[t] == 2).all(), "boundary prediction requires fixed TEST targets")
    qi = {name: i for i, name in enumerate(axes["query_ids"])}
    own = np.asarray([qi.get(name, -1) for name in axes["target_ids"]])
    require((own[t] != q).all(), "self pairs are excluded")
    require(len(np.unique(t.astype(np.int64)*len(axes["query_ids"])+q)) == len(t), "duplicate requested keys")
    return t, q


def match_keys(requested, reference, gene_count):
    requested, reference = np.asarray(requested), np.asarray(reference)
    key = requested[:, 0].astype(np.int64)*gene_count + requested[:, 1]
    old = reference[:, 0].astype(np.int64)*gene_count + reference[:, 1]
    require(len(np.unique(old)) == len(old), "duplicate reference keys")
    order = np.argsort(old, kind="stable")
    positions = np.searchsorted(old[order], key)
    matched = positions < len(old)
    matched[matched] &= old[order[positions[matched]]] == key[matched]
    rows = np.full(len(key), -1, dtype=np.int64)
    rows[matched] = order[positions[matched]]
    return matched, rows


def frozen_tail(cfg):
    """Load only the original 8 decoder layers, norm and weightless rotary map.

    The historical boundary run used the official CPU BF16 implementation.
    The complete generic weight file is checksum-verified before loading its
    selected tensor slices; no task checkpoint is read by this function.
    """
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding)
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    snapshot = Path(cfg["decider_snapshot"])
    weight_path = snapshot / "model.safetensors"
    require(sha256(weight_path) == cfg["expected_backbone_weight_sha256"], "generic backbone checksum differs")
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    require(config.num_hidden_layers == 32 and config.hidden_size == 2560, "wrong generic backbone architecture")
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        layers = torch.nn.ModuleList(Qwen3_5DecoderLayer(config, i) for i in range(24, 32))
        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for index, layer in enumerate(layers, 24):
            prefix = f"model.language_model.layers.{index}."
            state = {key[len(prefix):]: handle.get_tensor(key) for key in keys if key.startswith(prefix)}
            require(bool(state), "missing frozen decoder layer")
            layer.load_state_dict(state, strict=True, assign=True)
        norm.load_state_dict({"weight": handle.get_tensor("model.language_model.norm.weight")}, strict=True, assign=True)
    layers.eval().requires_grad_(False)
    norm.eval().requires_grad_(False)
    require(all(p.dtype == torch.bfloat16 and p.device.type == "cpu" for p in list(layers.parameters()) + list(norm.parameters())),
            "tail must remain frozen CPU BF16")
    rotary = Qwen3_5TextRotaryEmbedding(config, device=torch.device("cpu"))

    def forward(states, mask):
        position = torch.arange(states.shape[1], device="cpu").view(1, 1, -1).expand(4, len(states), -1)
        kwargs = dict(config=config, inputs_embeds=states, attention_mask=mask,
                      past_key_values=None, position_ids=position[0])
        masks = {"full_attention": create_causal_mask(**kwargs),
                 "linear_attention": create_recurrent_attention_mask(**kwargs)}
        rotations = rotary(states, position[1:])
        hidden = states
        for index, layer in enumerate(layers, 24):
            hidden = layer(hidden, position_embeddings=rotations,
                           attention_mask=masks[config.layer_types[index]], position_ids=position[0],
                           past_key_values=None, use_cache=False)
        return norm(hidden)
    return forward


class FrozenBoundaryPredictor:
    def __init__(self, model, partition, data, cache, seed, features, cards):
        self.model = model.eval().requires_grad_(False)
        self.shared = partition.shared
        self.features, self.cards = features, cards
        self.card_final = load_array(Path(cache) / f"seed_{seed}/cards_final.npy")
        self.condition_block = load_array(Path(cache) / "condition_block24.npy")
        self.query_block = load_array(Path(cache) / "query_block24.npy")
        self.card_block = load_array(Path(cache) / f"seed_{seed}/cards_block24.npy")
        self.tail = frozen_tail(partition.shared.cfg)

    def head(self, t, q, joint):
        import torch
        shared = self.shared
        unique_t, inverse = np.unique(t, return_inverse=True)
        # Historical CPU extension casts semantic landmarks to BF16 before
        # the frozen head; numeric inputs stay float32.
        condition = shared.gather("condition", unique_t).to(torch.bfloat16)
        query = shared.gather("query", q).to(torch.bfloat16)
        prior = shared.tensor(self.card_final[self.cards[q]], floating=True).to(torch.bfloat16)
        jt = shared.tensor(joint, floating=True).to(torch.bfloat16)
        values = (condition, torch.ones(condition.shape[:2], dtype=torch.bool),
            shared.tensor(shared.condition_num[unique_t]).float(),
            query, torch.ones(query.shape[:2], dtype=torch.bool), shared.tensor(shared.query_num[q]).float(),
            prior, torch.ones(prior.shape[:2], dtype=torch.bool), shared.tensor(self.features[q]).float(),
            torch.from_numpy(inverse.astype(np.int64)), jt, torch.ones(jt.shape[:2], dtype=torch.bool), None)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            changed, direction = self.model(*values, prior_dropout=0.0)
            p = self.model.probabilities(changed, direction).float().numpy()
        validate_probabilities(p, len(t))
        return p

    def predict(self, t, q, return_joint=False):
        import torch
        from jev_a4.core.vendor.encode_joint_last8_cache import _run_tuple_batch
        tuples = np.column_stack((t, q, self.cards[q])).astype(np.int32)
        with torch.inference_mode():
            joint = _run_tuple_batch(tuples, evidence_column=2, condition_states=self.condition_block,
                query_states=self.query_block, card_states=self.card_block, forward_last8=self.tail,
                device="cpu", compute_dtype=torch.bfloat16)
        p = self.head(t, q, joint)
        return (p, joint) if return_joint else p


def difference(actual, expected):
    error = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return {"max_abs": float(error.max(initial=0)), "mean_abs": float(error.mean()) if error.size else 0.0}


def predict(root, seed, *, keys=None, model_dir=None, main_run_root=None, output=None, batch_size=16, parity_only=False):
    import torch
    from experiments.ablation import load_checkpoint
    root = Path(root)
    data, cache = root / "data" / CELL, root / "cache" / CELL
    keys = Path(keys) if keys else root / f"cell_pool/prepared/seed_{seed}/keys.npy"
    run_root = Path(main_run_root) if main_run_root else root / "runs/main"
    model_dir = Path(model_dir) if model_dir else run_root / CELL / f"seed_{seed}"
    output = Path(output) if output else root / f"cell_pool/predictions/seed_{seed}"
    require(not output.exists(), "prediction output already exists; use a fresh directory")
    require(batch_size > 0, "batch size must be positive")
    axes = read_json(data / "dataset.json")
    key_array = load_array(keys)
    partition = load_array(data / f"seed_{seed}/target_partition.npy")
    t, q = validate_keys(key_array, axes, partition)
    model, strict, backend, seal = load_checkpoint(model_dir, data, cache, device="cpu")
    require(int(seal["seed"]) == seed and seal["dataset"] == CELL, "wrong selected model identity")
    cfg = strict.shared.cfg
    require(cfg.get("paper_ablation", {}).get("arm", "S1Pert") == "S1Pert"
            and not cfg.get("numeric_only", False) and not cfg.get("flat_output", False)
            and cfg.get("joint_enabled", True), "boundary requires the full S1Pert model")
    torch.set_num_threads(int(cfg.get("cpu_threads", 2)))
    features, cards = train_priors(data, cache, seed)
    saved_path = model_dir / "test_predictions.npz"
    test_report = read_json(model_dir / "test_report.json")
    require(test_report["status"] == "PASS" and test_report["seed"] == seed and test_report["dataset"] == CELL
            and test_report["checkpoint_sha256"] == seal["best_checkpoint_sha256"]
            and test_report["test_predictions_sha256"] == sha256(saved_path), "sealed TEST predictions changed")
    with np.load(saved_path, allow_pickle=False) as saved:
        old_keys = np.column_stack((saved["target"], saved["query"]))
        old_p = np.asarray(saved["probabilities"], dtype=np.float32)
    np.testing.assert_array_equal(old_keys[:, 0], strict.records["target"])
    np.testing.assert_array_equal(old_keys[:, 1], strict.records["query"])
    validate_probabilities(old_p, len(old_keys))
    runner = FrozenBoundaryPredictor(model, strict, data, cache, seed, features, cards)
    selected = np.unique(np.linspace(0, len(old_keys)-1, min(8, len(old_keys)), dtype=np.int64))
    require(len(selected) > 0, "no original TEST rows for execution parity")
    actual, joint = runner.predict(old_keys[selected, 0], old_keys[selected, 1], return_joint=True)
    parity = difference(actual, old_p[selected])
    parity.update(status="PASS" if parity["max_abs"] <= PARITY_MAX_ABS and parity["mean_abs"] <= PARITY_MEAN_ABS else "FAIL",
                  tolerance_max_abs=PARITY_MAX_ABS, tolerance_mean_abs=PARITY_MEAN_ABS,
                  test_indices=selected.tolist(), scope="eight deterministic rows; CPU/GPU BF16 execution tolerance")
    require(parity["status"] == "PASS", "frozen extension execution parity failed; no fallback or retuning allowed")
    matched, rows = match_keys(key_array, old_keys, len(axes["query_ids"]))
    output.mkdir(parents=True)
    report = {"status": "PARITY_PASS" if parity_only else "RUNNING", "seed": seed, "dataset": CELL,
        "task_weights_frozen": True, "training_started": False, "class_order": ["up", "same", "down"],
        "parity": parity, "source": {"checkpoint": identity(model_dir / "best.pt"),
            "convergence": identity(model_dir / "convergence.json"), "keys": identity(keys),
            "sealed_test_predictions": identity(saved_path)}, "requested_label_field_read": False,
        "control_role": "original full-pool untreated context; A/B controls adjudicate labels only",
        "prior_role": "TRAIN-only input; standalone P1 reported separately, never added to S1Pert output",
        "new_records": int((~matched).sum()), "reused_records": int(matched.sum()),
        "all_requested_records_covered": False}
    write_json(output / "report.json", report)
    if parity_only:
        return report
    np.save(output / "keys.npy", key_array)
    probabilities = np.lib.format.open_memmap(output / "probabilities.npy", mode="w+", dtype="float32", shape=(len(t), 3))
    probabilities[:] = np.nan
    probabilities[matched] = old_p[rows[matched]]
    new = np.flatnonzero(~matched)
    for start in range(0, len(new), batch_size):
        selected = new[start:start+batch_size]
        probabilities[selected] = runner.predict(t[selected], q[selected])
        if start % (batch_size * 100) == 0:
            print(f"seed {seed}: {min(start+batch_size, len(new))}/{len(new)} new boundary keys", flush=True)
    validate_probabilities(probabilities, len(t))
    probabilities.flush()
    np.save(output / "p1_probabilities.npy", features[q, :3].astype(np.float32))
    report.update(status="PASS", all_requested_records_covered=True,
        outputs={name: identity(output / name) for name in ("keys.npy", "probabilities.npy", "p1_probabilities.npy")})
    write_json(output / "report.json", report)
    return report


def metric_values(labels, probabilities, targets):
    from jev_a4.core.metrics import classification_metrics
    if not len(labels):
        return {key: None for key in METRICS}
    result = classification_metrics(labels, probabilities, targets=targets)
    return {"macro_f1": result["macro_f1"], "balanced_accuracy": result["balanced_accuracy"],
            **{name+"_f1": result["per_class"][name]["f1"] for name in ("up", "down", "same")}}


def summarize_rows(rows):
    summary = []
    for subset in SUBSETS:
        for method in ("S1Pert", "P1"):
            selected = [row for row in rows if row["subset"] == subset and row["model"] == method]
            for metric in METRICS:
                values = np.asarray([row[metric] for row in selected if row[metric] is not None], dtype=float)
                summary.append({"subset": subset, "model": method, "metric": metric,
                    "mean": float(values.mean()) if len(values) else None,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "n": len(values), "mean_records": float(np.mean([r["records"] for r in selected])) if selected else None})
    return summary


def evaluate(root, *, prepared=None, predictions=None, output=None):
    root = Path(root)
    prepared = Path(prepared) if prepared else root / "cell_pool/prepared"
    predictions = Path(predictions) if predictions else root / "cell_pool/predictions"
    output = Path(output) if output else root / "paper/fig2"
    preparation = read_json(prepared / "report.json")
    require(preparation["status"] == "PASS", "proxy preparation incomplete")
    rows, inputs = [], {}
    for seed in preparation["model_seeds"]:
        data_dir, pred_dir = prepared / f"seed_{seed}", predictions / f"seed_{seed}"
        for name, binding in preparation["per_seed"][str(seed)]["artifacts"].items():
            require(sha256(data_dir / name) == binding["sha256"], "evaluation payload changed: " + name)
        records, keys, nested = [load_array(data_dir / name) for name in ("records.npy", "keys.npy", "original_unknown.npy")]
        report = read_json(pred_dir / "report.json")
        require(report["status"] == "PASS" and report["all_requested_records_covered"]
                and report["task_weights_frozen"] and not report["training_started"], "frozen predictions incomplete")
        require(report["seed"] == seed and report["class_order"] == ["up", "same", "down"], "wrong prediction identity")
        require(report["source"]["keys"]["sha256"] == sha256(data_dir / "keys.npy"), "predictor did not consume bound label-free keys")
        for name, binding in report["outputs"].items():
            require(sha256(pred_dir / name) == binding["sha256"], "prediction payload changed: " + name)
        np.testing.assert_array_equal(load_array(pred_dir / "keys.npy"), keys)
        np.testing.assert_array_equal(keys, np.column_stack((records["target"], records["query"])))
        require(nested.shape == (len(records),) and nested.dtype == bool, "invalid nested subset")
        for method, filename in (("S1Pert", "probabilities.npy"), ("P1", "p1_probabilities.npy")):
            p = load_array(pred_dir / filename)
            validate_probabilities(p, len(records))
            for subset, mask in ((SUBSETS[0], np.ones(len(records), dtype=bool)), (SUBSETS[1], nested)):
                rows.append({"seed": seed, "subset": subset, "model": method, "records": int(mask.sum()),
                    **metric_values(records["label"][mask], p[mask], records["target"][mask])})
        inputs[str(seed)] = {"prepared_records": identity(data_dir / "records.npy"),
                             "prediction_report": identity(pred_dir / "report.json")}
    csvwrite(output / "boundary_per_seed.csv", rows)
    summary = summarize_rows(rows)
    csvwrite(output / "table_boundary_generalization.csv", summary)
    write_json(output / "boundary_report.json", {"status": "PASS", "per_seed": rows, "summary": summary,
        "inputs": inputs, "sample_standard_deviation_ddof": 1,
        "subset_definition": "FULL_UNKNOWN_B_RESOLVED is the original-full-unresolved subset of A_UNRESOLVED_B_RESOLVED",
        "scope": "selected technical cell-pool proxy; five overlapping whole-target holdouts, not biological replicates",
        "task_weights_frozen": True, "training_performed": False})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "evaluate"))
    parser.add_argument("--work-dir")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--keys", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--main-run-root", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args(argv)
    root = work_root(args.work_dir)
    if args.command == "predict":
        require(args.seed is not None, "predict requires --seed")
        predict(root, args.seed, keys=args.keys, model_dir=args.model_dir, main_run_root=args.main_run_root, output=args.output,
                batch_size=args.batch_size, parity_only=args.parity_only)
    else:
        evaluate(root, prepared=args.prepared, predictions=args.predictions, output=args.output)


if __name__ == "__main__":
    main()
