#!/usr/bin/env python3
"""Fresh A4 task heads: TRAIN updates, VAL selection, sealed TEST evaluation.

Frozen generic DECIDER representations are inputs, never old task predictions.
No task checkpoint is loaded for initialization. Test labels are not opened by
the training process; the separate evaluator requires an immutable convergence
seal and the exact selected checkpoint digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from jev_a4.config import load_config
from jev_a4.core.metrics import classification_metrics

ROOT = Path(__file__).resolve().parent
CHECKPOINT_SCHEMA = "decider.a4.fresh_head.checkpoint.v1"
RESTORE_ENTRYPOINT = Path(__file__).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def settings(cfg):
    values = dict(physical_batch=4096, effective_batch=4096, evaluation_batch=4096,
                  lr=1e-4, weight_decay=.01, min_epochs=10, max_epochs=100,
                  patience=10, min_delta=1e-4, grad_clip=1.0,
                  prior_dropout=.10, pair_weight=.02, pair_margin=.05,
                  pairs_per_update=512, scheduler_patience=3, scheduler_factor=.5,
                  min_lr=1e-6, preload_shared_gpu=True, shared_gpu_budget_gib=4,
                  rss_limit_gib=64, host_available_min_gib=2,
                  gpu_memory_limit_gib=48, restore_timeout_seconds=600,
                  partition_ram_cache_budget_gib=40,
                  physical_batch_candidates=[4096, 2048, 1024],
                  restore_atol=2e-5, restore_preview_rows=128)
    values.update(cfg.get("training", {}))
    for key in ("physical_batch", "effective_batch", "evaluation_batch", "min_epochs",
                "max_epochs", "patience", "pairs_per_update"):
        values[key] = int(values[key])
    if values["min_epochs"] > values["max_epochs"] or min(values["physical_batch"], values["effective_batch"]) <= 0:
        raise ValueError("invalid training schedule")
    if not 0 <= float(values["prior_dropout"]) < 1 or values["pair_weight"] < 0:
        raise ValueError("invalid evidence dropout/pair weight")
    if not cfg.get("smoke", False):
        expected = dict(decoder_width=512, decoder_blocks=2, attention_heads=8,
                        ffn_width=2048, backbone_hidden_width=2560)
        for key, value in expected.items():
            if int(cfg.get(key, value)) != value:
                raise ValueError(f"A4 architecture must retain {key}={value}")
    return values


def model_config(cfg):
    return {key: cfg.get(key, default) for key, default in dict(
        decoder_width=512, decoder_blocks=2, attention_heads=8, ffn_width=2048,
        backbone_hidden_width=2560, dropout=.10).items()}


def vendor_module():
    from jev_a4.core.vendor import prior_model
    return prior_model


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_joint_loss(changed, direction, labels, reduction="mean"):
    import torch
    import torch.nn.functional as F
    changed_truth = (labels != 1).float()
    losses = F.binary_cross_entropy_with_logits(changed.float(), changed_truth, reduction="none")
    losses = losses + changed_truth * F.binary_cross_entropy_with_logits(
        direction.float(), (labels == 0).float(), reduction="none")
    if not torch.isfinite(losses).all():
        raise FloatingPointError("non-finite direct hierarchical classification loss")
    return losses.sum() if reduction == "sum" else losses.mean()


def evidence_keep(rows, dropout, rng):
    return np.ones(rows, dtype=bool) if not dropout else rng.random(rows) >= dropout


class ResourceGuard:
    """Bounded checks; no conservative double counting of shared/mapped pages."""
    def __init__(self, config):
        self.rss_limit = float(config["rss_limit_gib"]) * 2**30
        self.min_available = float(config["host_available_min_gib"]) * 2**30
        self.gpu_limit = float(config["gpu_memory_limit_gib"]) * 2**30
        self.peak_rss = 0

    def check(self):
        status = Path("/proc/self/status")
        if status.exists():
            values = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                      for line in status.read_text().splitlines() if ":" in line}
            rss = int(values.get("VmRSS", "0 kB").split()[0]) * 1024
            self.peak_rss = max(self.peak_rss, rss)
            if self.rss_limit > 0 and rss > self.rss_limit:
                raise MemoryError(f"TASK_RSS_LIMIT measured RSS {rss / 2**30:.3f} GiB")
        info = Path("/proc/meminfo")
        if info.exists():
            available = next(int(line.split()[1]) * 1024 for line in info.read_text().splitlines()
                             if line.startswith("MemAvailable:"))
            if available < self.min_available:
                raise MemoryError("HOST_AVAILABLE_MEMORY_LIMIT")
        import torch
        if torch.cuda.is_available() and self.gpu_limit > 0:
            allocated = torch.cuda.memory_allocated()
            if allocated > self.gpu_limit:
                raise MemoryError(f"TASK_CUDA_ALLOCATION_LIMIT {allocated / 2**30:.3f} GiB")


def cache_metadata(directory, kind):
    """Verify sealed cache metadata and artifact sizes without another huge scan.

    Encoder creation/reuse validates full payload SHA; its immutable manifest
    records those SHA identities. Final root verification also checks payloads.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("status") != "PASS" or manifest.get("schema") != "decider.a4.paper.encoding.v1"
            or manifest.get("kind") != kind):
        raise ValueError(f"cache is not an audited PASS {kind}: {directory}")
    for name, artifact in manifest.get("artifacts", {}).items():
        if Path(name).name != name:
            raise ValueError("cache artifact names may not escape their directory")
        if (directory / name).stat().st_size != int(artifact["bytes"]):
            raise ValueError(f"sealed cache artifact size changed: {name}")
    return manifest


class SharedCache:
    def __init__(self, data_dir, cache_dir, cfg, device, guard=None):
        import torch
        self.data_dir, self.cache_dir = Path(data_dir), Path(cache_dir)
        self.device = device
        self.cfg = cfg
        self.guard = guard or ResourceGuard(settings(cfg))
        self.ram_cache_used = 0
        self.ram_cache_budget = float(settings(cfg)["partition_ram_cache_budget_gib"]) * 2**30
        self.ram_cache_reports = []
        if not cfg.get("smoke", False):
            manifest = cache_metadata(self.cache_dir, "shared_condition_query")
            if (manifest.get("response_labels_read") is not False
                    or manifest.get("old_task_weights_or_predictions_read") is not False
                    or manifest.get("version") != "4b-v1"):
                raise ValueError("shared representation cache violates response-blind generic pretraining contract")
        self.condition_num = np.load(self.data_dir / "condition_numeric.npy", mmap_mode="r", allow_pickle=False)
        self.query_num = np.load(self.data_dir / "query_numeric.npy", mmap_mode="r", allow_pickle=False)
        self.condition = np.load(self.cache_dir / "condition_final32.npy", mmap_mode="r", allow_pickle=False)
        self.query = np.load(self.cache_dir / "query_final32.npy", mmap_mode="r", allow_pickle=False)
        hidden = int(model_config(cfg)["backbone_hidden_width"])
        if self.condition.shape[1:] != (16, hidden) or self.query.shape[1:] != (16, hidden):
            raise ValueError("shared landmarks do not have the A4 fixed16 boundary")
        if self.condition_num.shape != (len(self.condition), 4) or self.query_num.shape != (len(self.query), 4):
            raise ValueError("numeric and semantic axes mismatch")
        self.cuda_shared = False
        st = settings(cfg)
        if str(device).startswith("cuda") and st["preload_shared_gpu"]:
            required = self.condition.nbytes + self.query.nbytes
            free, _ = torch.cuda.mem_get_info(device)
            if required < min(float(st["shared_gpu_budget_gib"]) * 2**30, free * .15):
                self.condition = torch.as_tensor(np.array(self.condition, copy=True), device=device)
                self.query = torch.as_tensor(np.array(self.query, copy=True), device=device)
                self.cuda_shared = True

    def preload_partition(self, partition):
        required = int(partition.joint_present.nbytes + partition.joint_absent.nbytes)
        remaining = self.ram_cache_budget - self.ram_cache_used
        if required <= remaining:
            self.guard.check()
            # Sequential reads occur once per process. Assignment releases the
            # old mapping immediately; no /tmp copy and no TEST prefetch.
            partition.joint_present = np.array(partition.joint_present, copy=True, order="C")
            partition.joint_absent = np.array(partition.joint_absent, copy=True, order="C")
            self.ram_cache_used += required
            self.guard.check()
            report = {"partition": partition.name, "mode": "RAM_SEQUENTIAL_PRELOAD",
                      "bytes": required, "ram_cache_used_gib": self.ram_cache_used / 2**30}
        else:
            report = {"partition": partition.name, "mode": "MEMMAP_BUDGET_FALLBACK",
                      "bytes": required, "reason": "partition_joint_bytes_exceed_remaining_RAM_budget",
                      "remaining_budget_gib": max(0.0, remaining / 2**30)}
        self.ram_cache_reports.append(report)

    def gather(self, field, indices):
        import torch
        values = getattr(self, field)
        if torch.is_tensor(values):
            index = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            return values[index].to(dtype=torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32)
        return self.tensor(np.asarray(values[indices]), floating=True)

    def tensor(self, array, floating=False):
        import torch
        value = torch.from_numpy(np.array(array, copy=True, order="C"))
        if str(self.device).startswith("cuda"):
            value = value.pin_memory().to(self.device, non_blocking=True)
            if floating:
                value = value.to(torch.bfloat16)
        elif floating:
            value = value.float()
        return value


class Partition:
    """Only the requested partition opens; the trainer never constructs TEST."""
    def __init__(self, shared, seed, name):
        if name not in {"train", "val", "test"}:
            raise ValueError("unsupported partition")
        self.shared, self.name = shared, name
        data = shared.data_dir / f"seed_{seed}" / name
        cache = shared.cache_dir / f"seed_{seed}" / name
        if not shared.cfg.get("smoke", False):
            seed_root = shared.cache_dir / f"seed_{seed}"
            seed_manifest = cache_metadata(seed_root, "seed_prior_and_joint")
            if seed_manifest.get("task_checkpoint_read") is not False:
                raise ValueError("seed cache used a task checkpoint")
            if sha256(cache / "manifest.json") != seed_manifest["partitions"][name]:
                raise ValueError("seed/partition manifest identity drift")
            leaf = cache_metadata(cache, "split_joint")
            if (leaf.get("response_labels_read") is not False or leaf.get("frozen_last8") is not True
                    or leaf.get("prior_absent_masked_before_last8") is not True
                    or leaf.get("public_direct_relations") is not False or leaf.get("lora") is not False):
                raise ValueError("joint cache does not implement the selected frozen A4")
            for file in ("records.npy", "prior_features.npy"):
                if sha256(data / file) != leaf["identity"]["inputs"][file]:
                    raise ValueError("joint cache/record evidence provenance mismatch")
        self.records = np.load(data / "records.npy", mmap_mode="r", allow_pickle=False)
        if not self.records.dtype.names or not {"target", "query", "label"} <= set(self.records.dtype.names):
            raise ValueError("records require target/query/label")
        self.features = np.load(data / "prior_features.npy", mmap_mode="r", allow_pickle=False)
        self.cards = np.load(shared.cache_dir / f"seed_{seed}" / "cards_final.npy", mmap_mode="r", allow_pickle=False)
        self.card_ids = np.load(cache / "card_ids.npy", mmap_mode="r", allow_pickle=False)
        self.joint_present = np.load(cache / "joint_present.npy", mmap_mode="r", allow_pickle=False)
        self.joint_absent = np.load(cache / "joint_absent.npy", mmap_mode="r", allow_pickle=False)
        mapped = cache / "joint_ids.npy"
        self.joint_ids = np.load(mapped, mmap_mode="r", allow_pickle=False) if mapped.exists() else np.arange(len(self.records))
        n, hidden = len(self.records), shared.condition.shape[-1]
        if not n or self.features.ndim != 2 or len(self.features) != n or self.features.shape[1] < 4:
            raise ValueError("empty partition or invalid numeric evidence")
        if self.card_ids.shape != (n,) or self.joint_ids.shape != (n,):
            raise ValueError("evidence ids are not record aligned")
        if self.cards.shape[1:] != (4, hidden) or self.joint_present.shape[1:] != (4, hidden):
            raise ValueError("card/joint cache does not use A4 four landmarks")
        if self.joint_absent.shape != self.joint_present.shape:
            raise ValueError("present/masked joint caches differ in shape")
        if np.min(self.card_ids) < 0 or np.max(self.card_ids) >= len(self.cards):
            raise ValueError("invalid prior card id")
        if np.min(self.joint_ids) < 0 or np.max(self.joint_ids) >= len(self.joint_present):
            raise ValueError("invalid joint id")
        if np.any((self.records["label"] < 0) | (self.records["label"] > 2)):
            raise ValueError("partition contains unresolved/non-strict label")
        if np.any((self.records["target"] < 0) | (self.records["target"] >= len(shared.condition))):
            raise ValueError("invalid target id")
        if np.any((self.records["query"] < 0) | (self.records["query"] >= len(shared.query))):
            raise ValueError("invalid query id")
        shared.preload_partition(self)

    def batch(self, indices, keep=None):
        import torch
        indices = np.asarray(indices, dtype=np.int64)
        if keep is None:
            keep = np.ones(len(indices), dtype=bool)
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != indices.shape:
            raise ValueError("one authoritative evidence mask per row is required")
        rows = self.records[indices]
        unique_c, c_inverse = np.unique(rows["target"], return_inverse=True)
        condition = self.shared.gather("condition", unique_c)
        query = self.shared.gather("query", np.asarray(rows["query"], dtype=np.int64))
        prior = self.shared.tensor(self.cards[self.card_ids[indices]], floating=True)
        prior_num = np.array(self.features[indices], dtype=np.float32, copy=True)
        prior_num[~keep] = 0
        prior_num[~keep, :3] = 1 / 3
        # The last feature is the explicit available flag. All other support
        # features are also neutralized for dropped rows.
        prior_num[~keep, -1] = 0
        joint_ids = self.joint_ids[indices]
        joint = np.array(self.joint_present[joint_ids], copy=True)
        if not keep.all():
            joint[~keep] = self.joint_absent[joint_ids[~keep]]
        if not np.isfinite(prior_num).all() or not np.isfinite(joint).all():
            raise FloatingPointError("non-finite cached evidence")
        device = self.shared.device
        condition_mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=device)
        query_mask = torch.ones(query.shape[:2], dtype=torch.bool, device=device)
        prior_mask = torch.as_tensor(keep[:, None].repeat(4, axis=1), device=device)
        joint_tensor = self.shared.tensor(joint, floating=True)
        joint_mask = torch.ones(joint_tensor.shape[:2], dtype=torch.bool, device=device)
        return (condition, condition_mask,
                self.shared.tensor(self.shared.condition_num[unique_c]).float(),
                query, query_mask,
                self.shared.tensor(self.shared.query_num[rows["query"]]).float(),
                prior, prior_mask, self.shared.tensor(prior_num).float(),
                torch.as_tensor(c_inverse, device=device, dtype=torch.long),
                joint_tensor, joint_mask, None)


class PairSampler:
    """Vectorized record-weighted same-query/different-target A4 pairs."""
    def __init__(self, records):
        self.records = records
        # A strict data grid contains at most one row for any target/query.
        keys = np.rec.fromarrays([records["target"], records["query"]])
        if len(np.unique(keys)) != len(records):
            raise ValueError("duplicated target/query rows would invalidate pair partner sampling")
        self.query = np.asarray(records["query"], dtype=np.int64)
        self.labels = np.asarray(records["label"], dtype=np.int64)
        maximum = int(self.query.max()) + 1
        self.pools = {}
        for name, mask in (("changed", self.labels != 1), ("same", self.labels == 1),
                           ("up", self.labels == 0), ("down", self.labels == 2)):
            rows = np.flatnonzero(mask)
            rows = rows[np.argsort(self.query[rows], kind="stable")]
            counts = np.bincount(self.query[rows], minlength=maximum)
            self.pools[name] = (rows, np.r_[0, np.cumsum(counts)], counts)
        self.eligible = {}
        for kind, first, second in (("changed_same", "changed", "same"), ("up_down", "up", "down")):
            a, _, ac = self.pools[first]
            b, _, bc = self.pools[second]
            self.eligible[kind] = np.r_[a[bc[self.query[a]] > 0], b[ac[self.query[b]] > 0]]

    def sample(self, kind, count, rng):
        anchors = self.eligible[kind]
        if not len(anchors) or count <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        anchor = anchors[rng.integers(len(anchors), size=int(count))]
        first, second = ("changed", "same") if kind == "changed_same" else ("up", "down")
        is_first = self.labels[anchor] != 1 if kind == "changed_same" else self.labels[anchor] == 0
        partner = np.empty(len(anchor), dtype=np.int64)
        for mask, name in ((is_first, second), (~is_first, first)):
            rows, offsets, counts = self.pools[name]
            query = self.query[anchor[mask]]
            position = offsets[query] + (rng.random(len(query)) * counts[query]).astype(np.int64)
            partner[mask] = rows[position]
        left, right = np.where(is_first, anchor, partner), np.where(is_first, partner, anchor)
        if np.any(self.records[left]["target"] == self.records[right]["target"]):
            raise AssertionError("pair crossed different-target contract")
        return left, right


def autocast(device):
    import torch
    from contextlib import nullcontext
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()


def predict(model, partition, batch_size, guard=None, indices=None):
    import torch
    model.eval()
    indices = np.arange(len(partition.records)) if indices is None else np.asarray(indices, dtype=np.int64)
    probabilities = np.empty((len(indices), 3), dtype=np.float32)
    with torch.inference_mode(), autocast(partition.shared.device):
        for start in range(0, len(indices), batch_size):
            if guard:
                guard.check()
            chunk = indices[start:start + batch_size]
            changed, direction = model(*partition.batch(chunk), prior_dropout=0.0)
            result = model.probabilities(changed, direction).cpu().numpy()
            if not np.isfinite(result).all():
                raise FloatingPointError("non-finite prediction")
            probabilities[start:start + len(chunk)] = result
    return probabilities


def bindings(cfg, data_dir, cache_dir, seed):
    # Cache payload SHA identities are established by the encoder's PASS
    # manifests. Bind those manifests, rather than re-reading multi-GB caches
    # every epoch. Root independent final verification rechecks payload hashes.
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    manifests = {}
    for root in (data_dir, data_dir / f"seed_{seed}", cache_dir, cache_dir / f"seed_{seed}"):
        for name in ("manifest.json", "split_manifest.json", "audit.json"):
            path = root / name
            if path.exists():
                manifests[str(path)] = sha256(path)
    for partition in ("train", "val", "test"):
        path = cache_dir / f"seed_{seed}" / partition / "manifest.json"
        if path.exists():
            manifests[str(path)] = sha256(path)
    return {"config_sha256": object_sha(cfg), "manifest_sha256": manifests,
            "generic_factory_sha256": sha256(ROOT / "vendor" / "prior_model.py"),
            "trainer_sha256": sha256(Path(__file__)), "metrics_sha256": sha256(ROOT / "metrics.py")}


def choose_physical_batch(model, train_data, cfg, seed, guard):
    import torch
    st = settings(cfg)
    candidates = [int(st["physical_batch"])] + [int(v) for v in st["physical_batch_candidates"]]
    candidates = sorted(set(min(v, int(st["effective_batch"])) for v in candidates if v > 0), reverse=True)
    trials = []
    for physical in candidates:
        size = min(physical, len(train_data.records))
        model.zero_grad(set_to_none=True)
        try:
            guard.check()
            if train_data.shared.device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            ids = np.arange(size)
            labels = torch.as_tensor(np.asarray(train_data.records["label"][ids], dtype=np.int64),
                                     device=train_data.shared.device)
            keep = evidence_keep(size, st["prior_dropout"], np.random.default_rng(seed))
            started = time.monotonic()
            with autocast(train_data.shared.device):
                changed, direction = model(*train_data.batch(ids, keep), prior_dropout=0.0)
                loss = exact_joint_loss(changed, direction, labels)
            loss.backward()
            if train_data.shared.device.startswith("cuda"):
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                if peak > min(.85 * total, float(st["gpu_memory_limit_gib"]) * 2**30):
                    trials.append({"physical_batch": physical, "status": "REJECTED_SAFETY_MARGIN",
                                   "peak_gpu_gib": peak / 2**30})
                    continue
            else:
                peak = 0
            trials.append({"physical_batch": physical, "probed_rows": size, "status": "PASS",
                           "seconds": time.monotonic() - started, "peak_gpu_gib": peak / 2**30})
            return physical, trials
        except torch.cuda.OutOfMemoryError as exc:
            trials.append({"physical_batch": physical, "status": "OOM_PROBE_ONLY", "error": str(exc)})
        finally:
            model.zero_grad(set_to_none=True)
            if train_data.shared.device.startswith("cuda"):
                torch.cuda.empty_cache()
    raise RuntimeError(f"bounded physical batch probes all failed: {trials}")


def save_checkpoint(path, model, meta):
    import torch
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save({"schema": CHECKPOINT_SCHEMA, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "meta": meta}, temporary)
    os.replace(temporary, path)


def selected_checkpoint(model, probabilities, metrics, epoch, cfg, seed, dataset, out, bound):
    meta = {"dataset": dataset, "seed": int(seed), "selected_epoch": int(epoch),
            "model_config": model_config(cfg), "prior_feature_width": int(model.prior_numeric[0].in_features),
            "bindings": bound, "validation_metrics": metrics,
            "fresh_task_head": True, "old_task_checkpoint_initialization": False,
            "direct_three_class_prediction": True, "lora": False, "direct_relations": False,
            "old_prediction_output_anchor": False, "output_residual_correction": False,
            "parameter_update_labels": ["TRAIN"], "selection_labels": ["VAL"]}
    save_checkpoint(out / "best.pt", model, meta)
    atomic_npz(out / "best_val_predictions.npz", probabilities=probabilities,
               checkpoint_sha256=np.asarray(sha256(out / "best.pt")))


def train(cfg, dataset, seed, data_dir, cache_dir, out):
    import torch
    if cfg.get("cpu_threads"):
        torch.set_num_threads(int(cfg["cpu_threads"]))
    st = settings(cfg)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("refuse to overwrite an existing training run")
    out.mkdir(parents=True, exist_ok=True)
    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("production A4 training requires a usable CUDA device")
    if device == "cpu" and not cfg.get("smoke", False):
        raise RuntimeError("CPU is restricted to synthetic smoke tests")
    guard = ResourceGuard(st)
    seed_everything(int(seed))
    shared = SharedCache(data_dir, cache_dir, cfg, device, guard=guard)
    train_data = Partition(shared, seed, "train")
    val_data = Partition(shared, seed, "val")
    if set(np.unique(train_data.records["target"])) & set(np.unique(val_data.records["target"])):
        raise ValueError("TRAIN/VAL perturbation target overlap")
    bound = bindings(cfg, data_dir, cache_dir, seed)
    factory = vendor_module()
    probe = factory.make_prior_model(model_config(cfg), train_data.features.shape[1]).to(device)
    physical, trials = choose_physical_batch(probe, train_data, cfg, int(seed), guard)
    del probe
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    # Batch probes do not influence initialization or the optimization stream.
    seed_everything(int(seed))
    model = factory.make_prior_model(model_config(cfg), train_data.features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(st["lr"]), weight_decay=float(st["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(st["scheduler_factor"]), patience=int(st["scheduler_patience"]),
        threshold=float(st["min_delta"]), threshold_mode="abs", min_lr=float(st["min_lr"]))
    sampler = PairSampler(train_data.records)
    atomic_json(out / "batch_probe.json", {"status": "PASS", "selected_physical_batch": physical,
                "effective_batch": st["effective_batch"], "trials": trials,
                "partition_cache_reads": shared.ram_cache_reports,
                "probe_discarded_and_fresh_initialization_reset": True})
    exposure = np.zeros(len(train_data.records), dtype=np.uint16)
    pair_exposure = np.zeros(len(train_data.records), dtype=np.uint32)
    history, best_score, best_epoch, meaningful_best, stale = [], None, None, -float("inf"), 0
    started = time.monotonic()
    converged = False
    for epoch in range(1, st["max_epochs"] + 1):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), epoch, 20261012]))
        schedule = rng.permutation(len(train_data.records))
        model.train()
        loss_sum = cu_sum = ud_sum = 0.0
        updates = 0
        for start in range(0, len(schedule), st["effective_batch"]):
            guard.check()
            chosen = schedule[start:start + st["effective_batch"]]
            optimizer.zero_grad(set_to_none=True)
            # Every selected logical update includes all its real rows exactly
            # once; a small final update is normalized by its own row count.
            for left in range(0, len(chosen), physical):
                micro = chosen[left:left + physical]
                keep = evidence_keep(len(micro), float(st["prior_dropout"]), rng)
                labels = torch.as_tensor(np.asarray(train_data.records["label"][micro], dtype=np.int64), device=device)
                with autocast(device):
                    changed, direction = model(*train_data.batch(micro, keep), prior_dropout=0.0)
                    summed = exact_joint_loss(changed, direction, labels, reduction="sum")
                    (summed / len(chosen)).backward()
                loss_sum += float(summed.detach())
            # Full evidence is used for pair objectives. Their additional
            # exposures are separately reported, never counted as epochs.
            if float(st["pair_weight"]) > 0:
                for kind in ("changed_same", "up_down"):
                    a, b = sampler.sample(kind, st["pairs_per_update"], rng)
                    if not len(a):
                        continue
                    all_pair = np.r_[a, b]
                    with autocast(device):
                        pc, pd = model(*train_data.batch(all_pair), prior_dropout=0.0)
                        raw = pc if kind == "changed_same" else pd
                        pair_loss = torch.relu(float(st["pair_margin"]) - raw[:len(a)] + raw[len(a):]).mean()
                        (float(st["pair_weight"]) * pair_loss).backward()
                    np.add.at(pair_exposure, all_pair, 1)
                    if kind == "changed_same":
                        cu_sum += float(pair_loss.detach())
                    else:
                        ud_sum += float(pair_loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(st["grad_clip"]), error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            exposure[chosen] += 1
            updates += 1
            if updates == 1 or updates % 25 == 0:
                atomic_json(out / "progress.json", {"status": "RUNNING", "dataset": dataset, "seed": int(seed),
                    "epoch": epoch, "main_rows_this_epoch": min(start + len(chosen), len(schedule)),
                    "train_rows": len(schedule), "optimizer_updates_this_epoch": updates,
                    "elapsed_seconds": time.monotonic() - started})
        if not np.all(exposure == epoch):
            raise AssertionError("main TRAIN exposure is not exactly once per epoch")
        probabilities = predict(model, val_data, min(physical, st["evaluation_batch"]), guard)
        metrics = classification_metrics(val_data.records["label"], probabilities, val_data.records["target"])
        score = (metrics["macro_f1"], -metrics["nll"])
        if best_score is None or score > best_score:
            best_score, best_epoch = score, epoch
            selected_checkpoint(model, probabilities, metrics, epoch, cfg, seed, dataset, out, bound)
        if metrics["macro_f1"] > meaningful_best + float(st["min_delta"]):
            meaningful_best, stale = metrics["macro_f1"], 0
        else:
            stale += 1
        scheduler.step(metrics["macro_f1"])
        row = {"epoch": epoch, "train_dropout_nll": loss_sum / len(schedule),
               "changed_same_pair_loss_mean_per_update": cu_sum / updates,
               "up_down_pair_loss_mean_per_update": ud_sum / updates,
               "validation": metrics, "main_exposure_min": int(exposure.min()),
               "main_exposure_max": int(exposure.max()), "pair_exposure_min": int(pair_exposure.min()),
               "pair_exposure_max": int(pair_exposure.max()), "stale_epochs": stale,
               "learning_rate_next_epoch": float(optimizer.param_groups[0]["lr"]),
               "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
               "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        atomic_json(out / "history.json", history)
        print(json.dumps({"stage": "train", "dataset": dataset, "seed": int(seed), "epoch": epoch,
                          "val_macro_f1": metrics["macro_f1"], "stale_epochs": stale}), flush=True)
        if epoch >= st["min_epochs"] and stale >= st["patience"]:
            converged = True
            break
    checkpoint_sha = sha256(out / "best.pt")
    report = {"status": "CONVERGED" if converged else "NOT_CONVERGED", "dataset": dataset, "seed": int(seed),
        "epochs_completed": epoch, "selected_epoch": best_epoch, "best_checkpoint_sha256": checkpoint_sha,
        "validation_selection": "highest pooled three-class Macro-F1; exact tie lower NLL; then earlier epoch",
        "convergence_definition": "at least min_epochs and patience consecutive epochs without VAL Macro-F1 gain > min_delta relative to running meaningful best",
        "min_epochs": st["min_epochs"], "max_epochs": st["max_epochs"], "patience": st["patience"],
        "min_delta": st["min_delta"], "terminal_stale_epochs": stale,
        "test_labels_opened": False, "fresh_task_head": True, "lora": False,
        "old_task_checkpoint_initialization": False, "old_prediction_output_anchor": False,
        "output_residual_correction": False, "train_records": len(train_data.records), "val_records": len(val_data.records),
        "main_exposure_min": int(exposure.min()), "main_exposure_max": int(exposure.max()),
        "pair_exposure_min": int(pair_exposure.min()), "pair_exposure_max": int(pair_exposure.max()),
        "physical_batch": physical, "effective_batch": st["effective_batch"], "peak_rss_gib": guard.peak_rss / 2**30,
        "partition_cache_reads": shared.ram_cache_reports,
        "elapsed_seconds": time.monotonic() - started, "bindings": bound}
    atomic_json(out / "convergence.json", report)
    # Restore the selected model only to inspect immutable VAL predictions;
    # this is verification, never reuse of task weights for a new training run.
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    preview_ids = np.arange(min(st["restore_preview_rows"], len(val_data.records)))
    expected = predict(model, val_data, min(physical, st["evaluation_batch"]), guard, preview_ids)
    atomic_npz(out / "restore_preview.npz", indices=preview_ids, probabilities=expected,
               checkpoint_sha256=np.asarray(checkpoint_sha))
    del model, optimizer, checkpoint, sampler, train_data, val_data, shared
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    # Independent process, no inherited model objects or optimizer state.
    config_path = out / "effective_config.json"
    atomic_json(config_path, cfg)
    command = [sys.executable, str(RESTORE_ENTRYPOINT), "restore-check", "--config", str(config_path),
               "--dataset", dataset, "--seed", str(seed), "--data-dir", str(data_dir),
               "--cache-dir", str(cache_dir), "--out", str(out)]
    with (out / "restore_check.log").open("w") as log:
        restored = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False,
                                  timeout=float(st["restore_timeout_seconds"]))
    if restored.returncode:
        raise RuntimeError("independent selected-checkpoint restore validation failed; see restore_check.log")
    report["independent_restore"] = json.loads((out / "restore_report.json").read_text())
    atomic_json(out / "training_report.json", report)
    return report


def load_selected(cfg, dataset, seed, data_dir, cache_dir, out, *, require_converged):
    out = Path(out)
    seal = json.loads((out / "convergence.json").read_text())
    if require_converged and seal.get("status") != "CONVERGED":
        raise PermissionError("TEST is locked: validation convergence has not been confirmed")
    if seal.get("dataset") != dataset or int(seal.get("seed", -1)) != int(seed):
        raise ValueError("dataset/split seed does not match convergence seal")
    if sha256(out / "best.pt") != seal["best_checkpoint_sha256"]:
        raise ValueError("selected checkpoint digest changed after seal")
    current = bindings(cfg, data_dir, cache_dir, seed)
    if current != seal["bindings"]:
        raise ValueError("code/config/evidence provenance differs from the sealed training run")
    import torch
    if cfg.get("cpu_threads"):
        torch.set_num_threads(int(cfg["cpu_threads"]))
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint["meta"]["bindings"] != current:
        raise ValueError("checkpoint schema or bindings mismatch")
    if not checkpoint["meta"].get("fresh_task_head") or checkpoint["meta"].get("old_task_checkpoint_initialization"):
        raise ValueError("checkpoint violates fresh direct task-head contract")
    device = str(cfg.get("device", "cuda"))
    shared = SharedCache(data_dir, cache_dir, cfg, device)
    model = vendor_module().make_prior_model(checkpoint["meta"]["model_config"],
                                             checkpoint["meta"]["prior_feature_width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, shared, seal


def restore_check(cfg, dataset, seed, data_dir, cache_dir, out):
    out = Path(out)
    model, shared, seal = load_selected(cfg, dataset, seed, data_dir, cache_dir, out, require_converged=False)
    expected = np.load(out / "restore_preview.npz", allow_pickle=False)
    if str(expected["checkpoint_sha256"]) != seal["best_checkpoint_sha256"]:
        raise ValueError("restore preview checkpoint binding mismatch")
    validation = Partition(shared, seed, "val")
    actual = predict(model, validation, settings(cfg)["evaluation_batch"], indices=expected["indices"])
    maximum = float(np.abs(actual - expected["probabilities"]).max())
    tolerance = float(settings(cfg)["restore_atol"])
    if not np.allclose(actual, expected["probabilities"], atol=tolerance, rtol=tolerance):
        raise ValueError(f"independent inference drift {maximum}")
    report = {"status": "PASS", "independent_process_pid": os.getpid(), "partition": "VAL",
              "rows": len(actual), "max_absolute_probability_difference": maximum,
              "tolerance": tolerance, "checkpoint_sha256": seal["best_checkpoint_sha256"],
              "test_labels_opened": False}
    atomic_json(out / "restore_report.json", report)
    return report


def evaluate_test(cfg, dataset, seed, data_dir, cache_dir, out):
    out = Path(out)
    model, shared, seal = load_selected(cfg, dataset, seed, data_dir, cache_dir, out, require_converged=True)
    restored = json.loads((out / "restore_report.json").read_text())
    if restored.get("status") != "PASS" or restored.get("checkpoint_sha256") != seal["best_checkpoint_sha256"]:
        raise PermissionError("TEST is locked until independent restore has passed")
    # This is the first access to held-out TEST labels/cache for this evaluator.
    test = Partition(shared, seed, "test")
    guard = ResourceGuard(settings(cfg))
    probabilities = predict(model, test, settings(cfg)["evaluation_batch"], guard)
    result = classification_metrics(test.records["label"], probabilities, test.records["target"])
    # P1 is a standalone comparator; its probabilities are never fed into or
    # added to the model's output. Numeric prior evidence already was an input.
    p1 = np.asarray(test.features[:, :3], dtype=np.float64)
    if np.allclose(p1.sum(1), 1, atol=2e-5) and np.all(p1 >= 0):
        result["standalone_P1_reference"] = classification_metrics(test.records["label"], p1, test.records["target"])
    atomic_npz(out / "test_predictions.npz", target=np.asarray(test.records["target"]),
               query=np.asarray(test.records["query"]), labels=np.asarray(test.records["label"]),
               probabilities=probabilities, checkpoint_sha256=np.asarray(seal["best_checkpoint_sha256"]),
               dataset=np.asarray(dataset), seed=np.asarray(seed))
    report = {"status": "PASS", "dataset": dataset, "seed": int(seed),
              "checkpoint_sha256": seal["best_checkpoint_sha256"], "selected_epoch": seal["selected_epoch"],
              "metrics": result, "test_predictions_sha256": sha256(out / "test_predictions.npz"),
              "test_used_for_training_or_selection": False, "peak_rss_gib": guard.peak_rss / 2**30}
    report["partition_cache_reads"] = shared.ram_cache_reports
    atomic_json(out / "test_report.json", report)
    return report


def synthetic_smoke(out, device="cpu"):
    """Tiny artificial tensors: validates mechanics, never scientific evidence."""
    import torch
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("smoke output must be a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    data, cache = out / "data", out / "cache"
    data.mkdir()
    cache.mkdir()
    rng = np.random.default_rng(20261012)
    hidden, ntarget, nquery, ncard = 16, 10, 8, 8
    np.save(data / "condition_numeric.npy", rng.normal(size=(ntarget, 4)).astype("f4"))
    np.save(data / "query_numeric.npy", rng.normal(size=(nquery, 4)).astype("f4"))
    np.save(cache / "condition_final32.npy", rng.normal(size=(ntarget, 16, hidden)).astype("f2"))
    np.save(cache / "query_final32.npy", rng.normal(size=(nquery, 16, hidden)).astype("f2"))
    (cache / "seed_17").mkdir()
    np.save(cache / "seed_17" / "cards_final.npy", rng.normal(size=(ncard, 4, hidden)).astype("f2"))
    for partition, target_ids in (("train", np.arange(6)), ("val", np.arange(6, 8)), ("test", np.arange(8, 10))):
        d, c = data / "seed_17" / partition, cache / "seed_17" / partition
        d.mkdir(parents=True)
        c.mkdir(parents=True)
        target = np.repeat(target_ids, nquery)
        query = np.tile(np.arange(nquery), len(target_ids))
        records = np.empty(len(target), dtype=[("target", "i4"), ("query", "i4"), ("label", "i1")])
        records["target"], records["query"], records["label"] = target, query, (target + query) % 3
        features = np.zeros((len(target), 12), dtype="f4")
        features[:, :3], features[:, -1] = 1 / 3, 1
        np.save(d / "records.npy", records)
        np.save(d / "prior_features.npy", features)
        np.save(c / "card_ids.npy", query)
        np.save(c / "joint_ids.npy", np.arange(len(target), dtype="i8"))
        np.save(c / "joint_present.npy", rng.normal(size=(len(target), 4, hidden)).astype("f2"))
        np.save(c / "joint_absent.npy", rng.normal(size=(len(target), 4, hidden)).astype("f2"))
    cfg = {"smoke": True, "device": device, "cpu_threads": 1,
           "runtime_python": sys.executable,
           "decoder_width": 32, "decoder_blocks": 2, "attention_heads": 4,
           "ffn_width": 64, "backbone_hidden_width": hidden, "dropout": .1,
           "training": {"physical_batch": 16, "effective_batch": 16, "evaluation_batch": 16,
                        "physical_batch_candidates": [16, 8], "min_epochs": 2, "max_epochs": 4,
                        "patience": 2, "min_delta": 2.0, "pairs_per_update": 4,
                        "restore_preview_rows": 8, "preload_shared_gpu": False,
                        "host_available_min_gib": 0}}
    atomic_json(out / "smoke_config.json", cfg)
    train_report = train(cfg, "synthetic", 17, data, cache, out / "run")
    if train_report["status"] != "CONVERGED":
        raise AssertionError("synthetic stopping-mechanics test did not reach its synthetic plateau")
    test_report = evaluate_test(cfg, "synthetic", 17, data, cache, out / "run")
    report = {"status": "PASS", "scientific_status": "NOT_EVALUATED_SYNTHETIC_ONLY",
              "training_status": train_report["status"], "epochs": train_report["epochs_completed"],
              "independent_restore": train_report["independent_restore"]["status"],
              "sealed_test_evaluation": test_report["status"], "device": device}
    atomic_json(out / "smoke_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate", "restore-check", "smoke"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="synthetic smoke only")
    arguments = parser.parse_args(argv)
    if arguments.action == "smoke":
        print(json.dumps(synthetic_smoke(arguments.out, arguments.device), allow_nan=False), flush=True)
        return
    if any(getattr(arguments, key) is None for key in ("config", "dataset", "seed", "data_dir", "cache_dir")):
        parser.error("production and restore actions require config/dataset/seed/data-dir/cache-dir")
    cfg = load_config(arguments.config)
    if str(cfg.get("device", "cuda")).startswith("cuda") and cfg.get("gpu_uuid") is not None:
        # Select the configured device before the first lazy Torch import.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_uuid"])
    function = {"train": train, "evaluate": evaluate_test, "restore-check": restore_check}[arguments.action]
    try:
        result = function(cfg, arguments.dataset, arguments.seed, arguments.data_dir,
                          arguments.cache_dir, arguments.out)
    except Exception as exc:
        arguments.out.mkdir(parents=True, exist_ok=True)
        atomic_json(arguments.out / f"{arguments.action}_failure.json", {
            "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
            "dataset": arguments.dataset, "seed": arguments.seed,
            "automatic_restart": False})
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
