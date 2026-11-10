#!/usr/bin/env python3
"""Response-blind, frozen DECIDER-4b-v1 encodings for the selected A4 method.

Only public functions, identities, untreated controls, and independently audited
TRAIN-only cross-fitted tendencies are translated into semantic inputs. Current
response labels are never selected. No historical task weights or predictions
are loaded. The A4 architecture is retained exactly: condition16/query16/prior4
block-24 landmarks, contextualized by frozen blocks 24:32, then four role means.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

import numpy as np

from jev_a4.config import load_config

HERE = Path(__file__).resolve().parent
SCHEMA = "decider.a4.paper.encoding.v1"
PARTITIONS = ("train", "val", "test")
_WEIGHT_DIGESTS = {}
TASK_PREFIX = (
    "Task definition: predict the direction of the query gene's population-average RNA response "
    "after CRISPR interference of the named target in the named cell line. The allowed outcomes are "
    "upregulated, unchanged, and downregulated relative to untreated controls. Use the intervention, "
    "cell context, untreated measurements, and public functional descriptions as uncertain evidence. "
    "A known relationship does not guarantee a detectable RNA response, missing knowledge does not "
    "imply unchanged, and the response label itself is never included in the evidence.\n\n"
)
UNITS = "Untreated-control RNA uses linear counts normalized to 10,000 counts per cell. "
DISCLAIMER = ("These annotations are background evidence and do not establish a signed causal response. "
              "Missing evidence does not imply an unchanged response.")
QUESTION = "How does the query gene population mean RNA change relative to untreated controls?"
OPTIONS = (
    "Upregulated: the population mean RNA ratio is greater than 1.2.",
    "Unchanged: the population mean RNA ratio is between 1/1.2 and 1.2.",
    "Downregulated: the population mean RNA ratio is less than 1/1.2.",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_functions(database):
    """Public KB only; deterministic longest function, matching the selected A4."""
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    # This historical database also contains empirical response tables. Even
    # accidental future SQL must not open them during semantic compilation.
    def public_only(action, table, column, database_name, trigger):
        if action == sqlite3.SQLITE_READ and table != "genes":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    connection.set_authorizer(public_only)
    result = {}
    for symbol, protein, function in connection.execute(
            "SELECT symbol,protein_name,function_text FROM genes ORDER BY symbol,accession"):
        text = " ".join(x.strip() for x in (protein or "", function or "") if x and x.strip())
        if len(text) > len(result.get(str(symbol), "")):
            result[str(symbol)] = text[:320]
    connection.close()
    return result


def compile_texts(metadata, functions):
    """Factorized legacy A4 facts; semantic input never sees perturbed measurements."""
    targets = metadata["target_ids"]
    target_symbols = metadata["target_symbols"]
    genes = metadata.get("query_ids", metadata.get("gene_ids"))
    symbols = metadata.get("query_symbols", metadata.get("gene_symbols"))
    target_queries = metadata["target_query_indices"]
    if not (len(targets) == len(target_symbols) == len(target_queries)) or len(genes) != len(symbols):
        raise ValueError("dataset identity axes are misaligned")
    duplicate_symbols = set(metadata.get("duplicate_symbol_ids", {}))
    means = np.asarray(metadata["control_mean"], dtype=np.float64)
    variances = np.asarray(metadata["control_variance"], dtype=np.float64)
    detection = np.asarray(metadata["control_detection"], dtype=np.float64)
    if any(len(x) != len(genes) for x in (means, variances, detection)):
        raise ValueError("control statistics/query axis mismatch")
    if not np.isfinite(means).all() or not np.isfinite(variances).all() or not np.isfinite(detection).all():
        raise ValueError("non-finite control statistics")
    if (means < 0).any() or (variances < 0).any() or ((detection < 0) | (detection > 1)).any():
        raise ValueError("invalid untreated-control statistics")

    def stats(index):
        if int(index) < 0:
            return "mean=unknown, variance=unknown, detection fraction=unknown."
        index = int(index)
        if index >= len(genes):
            raise ValueError("target query index outside axis")
        return (f"mean={means[index]:.5g}, variance={variances[index]:.5g}, "
                f"detection fraction={detection[index]:.5g}.")

    def public(symbol):
        if str(symbol).isdigit():
            raise ValueError(f"numeric gene symbol cannot be used for knowledge lookup: {symbol}")
        if symbol in duplicate_symbols:
            return "The symbol maps to multiple measured gene identifiers; no unambiguous concise function was assigned."
        return functions.get(symbol) or "No concise public function was available."

    cell = metadata.get("dataset", metadata.get("name"))
    if not cell:
        raise ValueError("cell context missing")
    conditions = []
    queries = []
    condition_coverage = query_coverage = 0
    for target, symbol, query_index in zip(targets, target_symbols, target_queries):
        function = public(symbol)
        condition_coverage += int(symbol in functions and symbol not in duplicate_symbols)
        conditions.append(TASK_PREFIX + "Context:\n" +
            f"Cell context: {cell}. Intervention: CRISPR interference targets {symbol} ({target}); "
            "the intended action is reduced target transcription, while dose, efficiency and time are unknown.\n" +
            UNITS + "Target control " + stats(query_index) +
            "\nPublic target function: " + function + "\n" + DISCLAIMER)
    for gene, symbol, index in zip(genes, symbols, range(len(genes))):
        function = public(symbol)
        query_coverage += int(symbol in functions and symbol not in duplicate_symbols)
        queries.append(TASK_PREFIX + "Query state:\n" + f"Query gene {symbol} ({gene}) control " +
                       stats(index) + "\nPublic query function: " + function +
                       "\n\nQuestion: " + QUESTION + "\nOptions:\n" + "\n".join(OPTIONS))
    return {"condition": conditions, "query": queries}, {
        "target_function_present": condition_coverage,
        "query_function_present": query_coverage,
        "targets": len(targets), "queries": len(genes),
        "ambiguous_symbols": len(duplicate_symbols),
        "numeric_symbol_lookup": False,
    }


def compress16(sequence):
    sequence = np.asarray(sequence)
    if sequence.ndim != 2 or len(sequence) == 0:
        raise ValueError("empty/non-matrix contextual sequence")
    edges = np.linspace(0, len(sequence), 13, dtype=np.int64)
    result = np.empty((16, sequence.shape[1]), dtype=np.float16)
    for index in range(12):
        left = min(int(edges[index]), len(sequence) - 1)
        right = max(left + 1, min(int(edges[index + 1]), len(sequence)))
        result[index] = sequence[left:right].mean(axis=0, dtype=np.float32).astype(np.float16)
    tail_index = np.clip(np.arange(len(sequence) - 4, len(sequence)), 0, len(sequence) - 1)
    result[12:] = sequence[tail_index]
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite compressed contextual landmarks")
    return result


def compress4(sequence):
    sequence = np.asarray(sequence)
    if sequence.ndim != 2 or len(sequence) == 0:
        raise ValueError("empty/non-matrix evidence sequence")
    edges = np.linspace(0, len(sequence), 4, dtype=np.int64)
    result = np.empty((4, sequence.shape[1]), dtype=np.float16)
    for index in range(3):
        left = min(int(edges[index]), len(sequence) - 1)
        right = max(left + 1, min(int(edges[index + 1]), len(sequence)))
        result[index] = sequence[left:right].mean(axis=0, dtype=np.float32).astype(np.float16)
    result[3] = sequence[-1]
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite compressed evidence landmarks")
    return result


def tokenize_checked(tokenizer, texts, max_tokens):
    tokenized = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    lengths = np.asarray([len(row) for row in tokenized], dtype=np.int32)
    if len(lengths) == 0 or lengths.min() <= 0 or lengths.max() > int(max_tokens):
        raise ValueError(f"input exceeds token budget ({max_tokens}); no truncation permitted")
    return tokenized, lengths


def categories(probabilities, support, fallback):
    probabilities = np.asarray(probabilities)
    ordered = np.sort(probabilities)
    maximum = float(ordered[-1])
    margin = float(ordered[-1] - ordered[-2])
    tendency = "mixed" if margin < 0.08 else ("upregulated", "unchanged", "downregulated")[int(np.argmax(probabilities))]
    concentration = "weak" if maximum < .50 or margin < .10 else ("moderate" if maximum < .75 else "strong")
    support_level = "limited" if support < 10 else ("moderate" if support < 50 else "broad")
    if int(fallback) not in (0, 1, 2):
        raise ValueError("invalid prior fallback code")
    source = ("cell-and-query specific", "cell-wide fallback", "global fallback")[int(fallback)]
    return tendency, concentration, support_level, source


def card_text(key):
    tendency, concentration, support, source = key
    meanings = {
        "upregulated": "Across other training perturbation targets, this query gene more often increased.",
        "unchanged": "Across other training perturbation targets, this query gene more often stayed within the unchanged class.",
        "downregulated": "Across other training perturbation targets, this query gene more often decreased.",
        "mixed": "Across other training perturbation targets, no single response direction clearly dominated.",
    }
    return ("Evidence field: cross-fitted response tendency for the query gene in the current cell context. " +
        meanings[tendency] + " The tendency has " + concentration + " concentration, " + support +
        " support, and uses a " + source + " estimate. This evidence was computed only from other "
        "TRAIN perturbation targets. It is not the answer for the current intervention target. A target-specific "
        "mechanism and the current cell state can make the response deviate from this tendency. Missing or weak "
        "evidence does not imply unchanged. Predict the current target-query response directly from all fields.")


def finite_cards():
    cards = [{"card_id": 0, "key": ["unavailable"] * 4,
              "text": "Cross-fitted response-tendency evidence is unavailable for this record. Do not infer unchanged from its absence."}]
    for key in itertools.product(("upregulated", "unchanged", "downregulated", "mixed"),
            ("weak", "moderate", "strong"), ("limited", "moderate", "broad"),
            ("cell-and-query specific", "cell-wide fallback", "global fallback")):
        cards.append({"card_id": len(cards), "key": list(key), "text": card_text(key)})
    return cards


def compile_card_ids(features, raw_counts, fallback):
    features = np.asarray(features)
    raw_counts = np.asarray(raw_counts)
    fallback = np.asarray(fallback)
    if features.ndim != 2 or features.shape[1] != 12:
        raise ValueError("A4 prior features require exactly 12 columns")
    if raw_counts.shape != (len(features), 3) or fallback.shape != (len(features),):
        raise ValueError("prior feature/raw-count/fallback alignment mismatch")
    if not np.isfinite(features).all() or (raw_counts < 0).any():
        raise ValueError("invalid prior features/counts")
    p = features[:, :3]
    if ((p < 0) | (p > 1)).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("prior probabilities are invalid")
    lookup = {tuple(card["key"]): card["card_id"] for card in finite_cards()}
    result = np.zeros(len(features), dtype=np.int16)
    for i in range(len(features)):
        if features[i, 11] > .5:
            result[i] = lookup[categories(p[i], int(raw_counts[i].sum()), int(fallback[i]))]
    return result


def structural_tuples(records, card_ids):
    if records.dtype.names is None or not {"target", "query"}.issubset(records.dtype.names):
        raise ValueError("records need structural target/query fields")
    if len(records) != len(card_ids):
        raise ValueError("records/card alignment mismatch")
    # Deliberately never index records['label']; hashing the physical file is integrity only.
    rows = np.column_stack([records["target"], records["query"], card_ids]).astype(np.int32)
    if (rows < 0).any():
        raise ValueError("negative structural/card identity")
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    return unique, inverse.astype(np.int32)


def _artifact(path):
    path = Path(path)
    result = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        result.update(shape=list(array.shape), dtype=str(array.dtype))
    return result


def validate_manifest(directory, expected_identity=None):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "PASS" or manifest.get("schema") != SCHEMA:
        raise ValueError(f"incompatible/unfinished encoding manifest: {directory}")
    if expected_identity is not None and manifest.get("identity") != expected_identity:
        raise ValueError(f"encoding identity differs; refusing stale cache: {directory}")
    for name, expected in manifest.get("artifacts", {}).items():
        actual = _artifact(directory / name)
        if actual != expected:
            raise ValueError(f"encoding payload changed: {directory / name}")
    return manifest


def _model_identity(cfg):
    snapshot = Path(cfg["decider_snapshot"])
    tokenizer = Path(cfg.get("tokenizer_snapshot", snapshot))
    config = read_json(snapshot / "decider_config.json")
    if config.get("version") != "4b-v1":
        raise ValueError("A4 paper model is the selected 4b-v1, not another release")
    files = {}
    for path in sorted(snapshot.glob("model*.safetensors")):
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in _WEIGHT_DIGESTS:
            _WEIGHT_DIGESTS[key] = sha256_file(path)
        files[str(path)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                            "sha256": _WEIGHT_DIGESTS[key]}
    if not files:
        raise FileNotFoundError("DECIDER weights not found")
    expected = cfg.get("expected_backbone_weight_sha256")
    if expected and (len(files) != 1 or next(iter(files.values()))["sha256"] != expected):
        raise ValueError("configured frozen backbone SHA256 mismatch")
    return {"version": "4b-v1", "weight_files": files,
        "decider_config": sha256_file(snapshot / "decider_config.json"),
        "model_config": sha256_file(snapshot / "config.json"),
        "tokenizer": sha256_file(tokenizer / "tokenizer.json"),
        "tokenizer_config": sha256_file(tokenizer / "tokenizer_config.json"),
        "native_loader": sha256_file(HERE / "vendor" / "native_engine.py"),
        "split_executor": sha256_file(HERE / "vendor" / "shared_core.py"),
        "encoder": sha256_file(__file__)}


class Session:
    """One frozen model instance can encode a dataset and all five split seeds."""
    def __init__(self, cfg, audit_path):
        import torch
        self.cfg = dict(cfg)
        self.cfg.setdefault("tokenizer_snapshot", self.cfg["decider_snapshot"])
        self.cfg.setdefault("decider_source", self.cfg["decider_snapshot"])
        self.native = _module(HERE / "vendor" / "native_engine.py", "_a4_native_encoder")
        self.core = _module(HERE / "vendor" / "shared_core.py", "_a4_split_executor")
        self.model = self.native._load_model(self.cfg, "B6", Path(audit_path))
        self.text_model = self.model.lm.model
        self.text_model.eval().requires_grad_(False)
        if int(self.text_model.config.hidden_size) != 2560 or int(self.text_model.config.num_hidden_layers) != 32:
            raise ValueError("selected A4 DECIDER architecture mismatch")
        if any(p.requires_grad for p in self.text_model.parameters()):
            raise ValueError("A4 encoder must remain frozen")
        self.tokenizer = self.model.tok
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.torch = torch
        self.hidden = 2560
        self.model_load_sha256 = sha256_file(audit_path)

    def encode_texts(self, texts, block_out, final_out, count, batch_size, progress=None):
        """Stream 0:24 and 24:32 once; never average raw word embeddings."""
        torch = self.torch
        compress = compress16 if count == 16 else compress4
        maximum = int(self.cfg.get("max_tokens", 1024))
        tokenized, lengths = tokenize_checked(self.tokenizer, texts, maximum)
        order = np.argsort(lengths, kind="stable")
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            selected_lengths = lengths[chosen]
            width = int(selected_lengths.max())
            ids = np.full((len(chosen), width), self.pad_id, dtype=np.int64)
            for i, index in enumerate(chosen):
                ids[i, :lengths[index]] = tokenized[index]
            mask = np.arange(width)[None, :] < selected_lengths[:, None]
            ids_t = torch.from_numpy(ids).to("cuda")
            mask_t = torch.from_numpy(mask).to("cuda")
            with torch.inference_mode():
                block = self.core.split_forward(self.text_model, ids_t, mask_t, start=0, end=24)
                final = self.core.split_forward(self.text_model, ids_t, mask_t, start=24, end=32,
                                               hidden_states=block, final_norm=True)
            if not torch.isfinite(block).all() or not torch.isfinite(final).all():
                raise FloatingPointError("non-finite frozen DECIDER contextual states")
            block_np = block.float().cpu().numpy()
            final_np = final.float().cpu().numpy()
            for i, index in enumerate(chosen):
                length = int(lengths[index])
                block_out[index] = compress(block_np[i, :length])
                final_out[index] = compress(final_np[i, :length])
            if progress is not None:
                atomic_json(progress, {"status": "RUNNING", "completed": min(start + batch_size, len(order)),
                                       "total": len(order), "phase": "text_encoding"})
            del block, final, block_np, final_np
        return {"minimum_tokens": int(lengths.min()), "maximum_tokens": int(lengths.max()),
                "total_tokens": int(lengths.sum()), "truncated": 0, "length_grouped": True}

    def check_split_parity(self, text):
        torch = self.torch
        values, _ = tokenize_checked(self.tokenizer, [text], int(self.cfg.get("max_tokens", 1024)))
        ids = torch.tensor(values, device="cuda", dtype=torch.long)
        mask = torch.ones(ids.shape, device="cuda", dtype=torch.bool)
        with torch.inference_mode():
            full = self.text_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            block = self.core.split_forward(self.text_model, ids, mask, start=0, end=24)
            split = self.core.split_forward(self.text_model, ids, mask, start=24, end=32,
                                           hidden_states=block, final_norm=True)
        diff = (full.float() - split.float()).abs()
        report = {"status": "PASS" if float(diff.max()) <= 2e-3 else "FAIL",
                  "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                  "same_batch_shape": True, "boundary": 24}
        if report["status"] != "PASS":
            raise ValueError("full/split DECIDER executor parity failed")
        return report

    def encode_joint(self, tuples, condition, query, cards, present, absent, batch_size, progress=None):
        torch = self.torch
        joint_core = _module(HERE / "vendor" / "encode_joint_last8_cache.py", "_a4_joint_executor")

        def last8(states, mask):
            dummy = torch.zeros(mask.shape, device="cuda", dtype=torch.long)
            return self.core.split_forward(self.text_model, dummy, mask, start=24, end=32,
                                           hidden_states=states, final_norm=True)

        tuple4 = np.column_stack([tuples, np.full(len(tuples), -1, dtype=np.int32)])
        parity = []
        for start in range(0, len(tuples), batch_size):
            batch = tuple4[start:start + batch_size]
            for column, output in ((2, present), (3, absent)):
                output[start:start + len(batch)] = joint_core._run_tuple_batch(
                    batch, evidence_column=column, condition_states=condition,
                    query_states=query, card_states=cards, forward_last8=last8,
                    device="cuda", compute_dtype=torch.bfloat16)
            if start == 0 or start + batch_size >= len(tuples):
                for column, output in ((2, present), (3, absent)):
                    replay = joint_core._run_tuple_batch(batch, evidence_column=column,
                        condition_states=condition, query_states=query, card_states=cards,
                        forward_last8=last8, device="cuda", compute_dtype=torch.bfloat16)
                    written = np.asarray(output[start:start + len(batch)])
                    error = float(np.max(np.abs(written.astype(np.float32) - replay.astype(np.float32))))
                    if error != 0:
                        raise ValueError("frozen joint cache same-generation-batch replay differs")
                    parity.append({"batch_start": start, "batch_rows": len(batch),
                                   "prior": "present" if column == 2 else "absent", "max_abs": error})
            if progress is not None:
                atomic_json(progress, {"status": "RUNNING", "completed": min(start + batch_size, len(tuples)),
                                       "total": len(tuples), "phase": "frozen_joint_last8"})
        return {"status": "PASS", "same_generation_batch_replay": parity}

    def close(self):
        del self.text_model, self.model
        self.torch.cuda.empty_cache()


def _new_memmap(path, shape):
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


def _shared_identity(cfg, dataset, data_dir):
    return {"dataset": dataset, "dataset_json_sha256": sha256_file(Path(data_dir) / "dataset.json"),
        "condition_numeric_sha256": sha256_file(Path(data_dir) / "condition_numeric.npy"),
        "query_numeric_sha256": sha256_file(Path(data_dir) / "query_numeric.npy"),
        "public_function_database_sha256": sha256_file(cfg["knowledge_db"]),
        "model": _model_identity(cfg), "max_tokens": int(cfg.get("max_tokens", 1024)),
        "prompt_sha256": object_sha([TASK_PREFIX, UNITS, DISCLAIMER, QUESTION, OPTIONS]),
        "layout": {"condition": 16, "query": 16, "hidden": 2560, "contextual_layer_boundary": 24}}


def encode_dataset(cfg, dataset, data_dir, cache_dir, session=None):
    """Cache each response-blind condition/query once, independent of split seed."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    identity = _shared_identity(cfg, dataset, data_dir)
    if (cache_dir / "manifest.json").exists():
        return validate_manifest(cache_dir, identity)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Incomplete outputs are not automatically overwritten: restart means an explicit new cache directory.
    if any((cache_dir / name).exists() for name in ("condition_final32.npy", "query_final32.npy")):
        raise FileExistsError("incomplete shared cache exists; use a new cache directory")
    metadata = read_json(data_dir / "dataset.json")
    texts, coverage = compile_texts(metadata, load_functions(cfg["knowledge_db"]))
    atomic_json(cache_dir / "texts.json", texts)
    owns = session is None
    if owns:
        session = Session(cfg, cache_dir / "model_load.json")
    started = time.monotonic()
    token_audit = {}
    try:
        for role in ("condition", "query"):
            shape = (len(texts[role]), 16, 2560)
            block = _new_memmap(cache_dir / f"{role}_block24.npy", shape)
            final = _new_memmap(cache_dir / f"{role}_final32.npy", shape)
            token_audit[role] = session.encode_texts(texts[role], block, final, 16,
                int(cfg.get("cache_encode_batch", 16)), cache_dir / "progress.json")
            block.flush(); final.flush()
            del block, final
        artifacts = {name: _artifact(cache_dir / name) for name in (
            "condition_block24.npy", "condition_final32.npy", "query_block24.npy", "query_final32.npy", "texts.json")}
        manifest = {"status": "PASS", "schema": SCHEMA, "kind": "shared_condition_query",
            "identity": identity, "artifacts": artifacts, "coverage": coverage,
            "token_audit": token_audit, "model_load_sha256": session.model_load_sha256,
            "response_blind": True, "response_labels_read": False,
            "old_task_weights_or_predictions_read": False, "frozen_backbone": True,
            "version": "4b-v1", "landmarks": 16, "hidden_width": 2560,
            "word_embedding_average_used": False, "elapsed_seconds": time.monotonic() - started}
        atomic_json(cache_dir / "manifest.json", manifest)
        return manifest
    finally:
        if owns:
            session.close()


def _seed_identity(cfg, dataset, seed, data_dir, cache_dir):
    seed_dir = Path(data_dir) / f"seed_{seed}"
    inputs = {}
    for partition in PARTITIONS:
        inputs[partition] = {name: sha256_file(seed_dir / partition / name)
            for name in ("records.npy", "prior_features.npy", "prior_raw_counts.npy", "prior_fallback.npy")}
    return {"dataset": dataset, "seed": int(seed), "inputs": inputs,
        "shared_manifest_sha256": sha256_file(Path(cache_dir) / "manifest.json"),
        "model": _model_identity(cfg), "cards_sha256": object_sha(finite_cards()),
        "architecture": "condition16_query16_prior4_frozen24to32_rolemeans4",
        "absent_prior_removed_before_last8": True}


def encode_split(cfg, dataset, seed, data_dir, cache_dir, session=None):
    """Prior cards and both frozen joint states; unique tuple maps remain record-aligned."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    validate_manifest(cache_dir, _shared_identity(cfg, dataset, data_dir))
    seed_out = cache_dir / f"seed_{seed}"
    identity = _seed_identity(cfg, dataset, seed, data_dir, cache_dir)
    if (seed_out / "manifest.json").exists():
        manifest = validate_manifest(seed_out, identity)
        for partition in PARTITIONS:
            if sha256_file(seed_out / partition / "manifest.json") != manifest["partitions"][partition]:
                raise ValueError("seed child-manifest binding changed")
            validate_manifest(seed_out / partition)
        return manifest
    seed_out.mkdir(parents=True, exist_ok=True)
    if (seed_out / "cards_final.npy").exists():
        raise FileExistsError("incomplete seed cache exists; use a new cache directory")
    cards = finite_cards()
    atomic_json(seed_out / "cards.json", cards)
    owns = session is None
    if owns:
        session = Session(cfg, seed_out / "model_load.json")
    started = time.monotonic()
    try:
        card_block = _new_memmap(seed_out / "cards_block24.npy", (len(cards), 4, 2560))
        card_final = _new_memmap(seed_out / "cards_final.npy", (len(cards), 4, 2560))
        token_audit = session.encode_texts([TASK_PREFIX + "Evidence state field follows. " + x["text"] for x in cards],
            card_block, card_final, 4, int(cfg.get("card_encode_batch", 32)), seed_out / "progress.json")
        card_block.flush(); card_final.flush()
        condition = np.load(cache_dir / "condition_block24.npy", mmap_mode="r", allow_pickle=False)
        query = np.load(cache_dir / "query_block24.npy", mmap_mode="r", allow_pickle=False)
        partition_manifests = {}
        for partition in PARTITIONS:
            src = data_dir / f"seed_{seed}" / partition
            out = seed_out / partition
            out.mkdir(parents=True, exist_ok=True)
            records = np.load(src / "records.npy", mmap_mode="r", allow_pickle=False)
            features = np.load(src / "prior_features.npy", mmap_mode="r", allow_pickle=False)
            raw = np.load(src / "prior_raw_counts.npy", mmap_mode="r", allow_pickle=False)
            fallback = np.load(src / "prior_fallback.npy", mmap_mode="r", allow_pickle=False)
            ids = compile_card_ids(features, raw, fallback)
            tuples, joint_ids = structural_tuples(records, ids)
            if len(tuples) and (tuples[:, 0].max() >= len(condition) or tuples[:, 1].max() >= len(query)):
                raise ValueError("record structural identity exceeds shared axes")
            atomic_npy(out / "card_ids.npy", ids)
            atomic_npy(out / "joint_tuples.npy", tuples)
            atomic_npy(out / "joint_ids.npy", joint_ids)
            shape = (len(tuples), 4, 2560)
            present = _new_memmap(out / "joint_present.npy", shape)
            absent = _new_memmap(out / "joint_absent.npy", shape)
            joint_parity = session.encode_joint(tuples, condition, query, card_block, present, absent,
                int(cfg.get("joint_cache_batch", 256)), out / "progress.json")
            present.flush(); absent.flush()
            del present, absent
            artifact_names = ("card_ids.npy", "joint_tuples.npy", "joint_ids.npy", "joint_present.npy", "joint_absent.npy")
            partition_manifest = {"status": "PASS", "schema": SCHEMA, "kind": "split_joint",
                "identity": {"seed_identity": object_sha(identity), "partition": partition,
                             "inputs": identity["inputs"][partition]},
                "artifacts": {name: _artifact(out / name) for name in artifact_names},
                "records": len(records), "unique_tuples": len(tuples), "landmarks": 4, "hidden_width": 2560,
                "joint_rows": "unique_tuple", "record_joint_ids": "joint_ids.npy",
                "tuple_columns": ["target", "query", "present_card_id"],
                "response_labels_read": False, "frozen_last8": True, "layers": [24, 32],
                "prior_absent_masked_before_last8": True, "posthoc_masking": False,
                "dropout_contract": "trainer jointly zeros prior numeric/card and selects joint_absent for the same records",
                "public_direct_relations": False, "lora": False, "online_parity": joint_parity}
            atomic_json(out / "manifest.json", partition_manifest)
            partition_manifests[partition] = sha256_file(out / "manifest.json")
        card_block.flush(); card_final.flush()
        del card_block, card_final, condition, query
        manifest = {"status": "PASS", "schema": SCHEMA, "kind": "seed_prior_and_joint",
            "identity": identity, "artifacts": {name: _artifact(seed_out / name)
                for name in ("cards.json", "cards_block24.npy", "cards_final.npy")},
            "partitions": partition_manifests, "token_audit": token_audit,
            "model_load_sha256": session.model_load_sha256,
            "response_blind_except_crossfitted_TRAIN_tendency": True,
            "current_record_response_read": False, "task_checkpoint_read": False,
            "word_embedding_average_used": False, "elapsed_seconds": time.monotonic() - started}
        atomic_json(seed_out / "manifest.json", manifest)
        return manifest
    finally:
        if owns:
            session.close()


def smoke(cfg, dataset, seed, data_dir, cache_dir, max_rows=4):
    """Tiny real-input GPU check, never a substitute for a production cache."""
    import torch
    if max_rows < 1 or max_rows > 4:
        raise ValueError("GPU smoke is limited to at most four real rows")
    data_dir, out = Path(data_dir), Path(cache_dir)
    out.mkdir(parents=True, exist_ok=False)
    metadata = read_json(data_dir / "dataset.json")
    texts, coverage = compile_texts(metadata, load_functions(cfg["knowledge_db"]))
    src = data_dir / f"seed_{seed}" / "train"
    records = np.load(src / "records.npy", mmap_mode="r", allow_pickle=False)[:max_rows]
    features = np.load(src / "prior_features.npy", mmap_mode="r", allow_pickle=False)[:max_rows]
    raw = np.load(src / "prior_raw_counts.npy", mmap_mode="r", allow_pickle=False)[:max_rows]
    fallback = np.load(src / "prior_fallback.npy", mmap_mode="r", allow_pickle=False)[:max_rows]
    card_ids = compile_card_ids(features, raw, fallback)
    conditions, ci = np.unique(records["target"], return_inverse=True)
    queries, qi = np.unique(records["query"], return_inverse=True)
    card_indices, ki = np.unique(card_ids, return_inverse=True)
    session = Session(cfg, out / "model_load.json")
    try:
        split_parity = session.check_split_parity(texts["query"][int(queries[0])])
        states = {}
        audits = {}
        selected_texts = {"condition": [texts["condition"][i] for i in conditions],
                          "query": [texts["query"][i] for i in queries],
                          "cards": [TASK_PREFIX + "Evidence state field follows. " + finite_cards()[i]["text"] for i in card_indices]}
        for role, selected in selected_texts.items():
            width = 4 if role == "cards" else 16
            block = np.empty((len(selected), width, 2560), dtype=np.float16)
            final = np.empty_like(block)
            audits[role] = session.encode_texts(selected, block, final, width, min(4, len(selected)))
            atomic_npy(out / f"{role}_block24.npy", block)
            atomic_npy(out / f"{role}_final32.npy", final)
            states[role] = block
        tuples = np.column_stack([ci, qi, ki]).astype(np.int32)
        present = np.empty((len(records), 4, 2560), dtype=np.float16)
        absent = np.empty_like(present)
        joint_parity = session.encode_joint(tuples, states["condition"], states["query"], states["cards"], present, absent, 4)
        atomic_npy(out / "joint_present.npy", present)
        atomic_npy(out / "joint_absent.npy", absent)
        if not np.isfinite(present).all() or not np.isfinite(absent).all():
            raise FloatingPointError("smoke non-finite cached joint")
        absent_prior_zero = bool(np.count_nonzero(absent[:, 2]) == 0)
        if not absent_prior_zero:
            raise ValueError("absent prior role mean is not zero")
        changed = bool(np.any(present != absent))
        if not changed:
            raise ValueError("present/absent prior context unexpectedly identical")
        report = {"status": "PASS", "scope": "tiny_real_input_semantic_and_joint_smoke_only",
            "dataset": dataset, "seed": seed, "real_rows": len(records), "version": "4b-v1",
            "coverage": coverage, "token_audit": audits, "all_states_finite": True,
            "absent_prior_role_zero": absent_prior_zero, "prior_context_changes_joint": changed,
            "condition_query_landmarks": 16, "prior_and_joint_landmarks": 4,
            "frozen_last8": [24, 32], "current_record_label_read": False,
            "historical_task_checkpoint_read": False, "training_started": False,
            "full_split_parity": split_parity, "frozen_joint_replay": joint_parity,
            "maximum_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}
        atomic_json(out / "report.json", report)
        return report
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-rows", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if "gpu_uuid" in cfg:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_uuid"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.smoke:
        if args.seed is None:
            parser.error("--smoke requires --seed")
        result = smoke(cfg, args.dataset, args.seed, args.data_dir, args.cache_dir, args.max_rows)
    elif args.all_seeds:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        session = Session(cfg, args.cache_dir / "model_load.json")
        try:
            shared = encode_dataset(cfg, args.dataset, args.data_dir, args.cache_dir, session=session)
            seeds = cfg["seeds"]
            result = {"shared": shared, "seeds": {str(seed): encode_split(
                cfg, args.dataset, seed, args.data_dir, args.cache_dir, session=session) for seed in seeds}}
        finally:
            session.close()
    elif args.seed is not None:
        result = encode_split(cfg, args.dataset, args.seed, args.data_dir, args.cache_dir)
    else:
        result = encode_dataset(cfg, args.dataset, args.data_dir, args.cache_dir)
    print(json.dumps({"status": "PASS", "dataset": args.dataset, "cache_dir": str(args.cache_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
