"""Shared audit, sealed data and evaluation helpers. No previous predictor is used."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import random
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)

def digest_file(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(CFG["threads"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

def data(cell, seed, include_test=False):
    base = Path(CFG["a4_data"]) / cell
    meta = json.loads((base / "dataset.json").read_text())
    part_path = base / f"seed_{seed}" / "target_partition.npy"
    part = np.load(part_path, allow_pickle=False)
    assert part.shape == (len(meta["target_ids"]),)
    assert set(np.unique(part)) <= {0, 1, 2}
    records = {}
    identities = {"dataset_json": digest_file(base / "dataset.json"), "partition": digest_file(part_path)}
    for i, name in enumerate(["train", "val", "test"]):
        if name == "test" and not include_test:
            continue
        path = base / f"seed_{seed}" / name / "records.npy"
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        assert rows.dtype.names == ("target", "query", "label")
        assert np.isin(rows["label"], [0, 1, 2]).all()
        assert (part[rows["target"]] == i).all()
        assert (rows["target"] >= 0).all() and (rows["query"] < len(meta["query_ids"])).all()
        tid = np.asarray(meta["target_ids"])
        qid = np.asarray(meta["query_ids"])
        assert (tid[rows["target"]] != qid[rows["query"]]).all()
        keys = rows["target"].astype(np.int64) * len(qid) + rows["query"]
        assert len(np.unique(keys)) == len(keys)
        records[name] = rows
        identities[name] = digest_file(path)
    return meta, part, records, identities

def hard_predictions(effect, down, up):
    effect = np.asarray(effect)
    if not np.isfinite(effect).all() or not down <= 0 <= up:
        raise ValueError("invalid effects or conversion thresholds")
    return np.where(effect > up, 0, np.where(effect < down, 2, 1)).astype(np.int8)

def hard_metrics(labels, predictions, targets=None, scores=None):
    # Do not pretend a hard class label is a calibrated probability.
    m = load_module(CFG["metrics"], "a4_metrics")
    y, p = np.asarray(labels), np.asarray(predictions)
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    out = m.metrics_from_confusion(cm)
    changed, called = y != 1, p != 1
    tp = int((changed & called).sum())
    prec = tp / max(1, int(called.sum()))
    rec = tp / max(1, int(changed.sum()))
    out.update(records=len(y), class_order=["up", "same", "down"], nll=None,
               brier_multiclass_sum=None,
               probability_status="NOT_AVAILABLE: native continuous prediction plus validation threshold adapter",
               changed={"precision": prec, "recall": rec, "f1": 2*prec*rec/max(1e-30,prec+rec)})
    if targets is not None:
        unique, inv, counts = np.unique(targets, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inv]
        weighted = np.bincount(y*3+p, weights=weights, minlength=9).reshape(3,3)
        out["targets"] = len(unique)
        out["equal_target_weighted_confusion_metrics"] = m.metrics_from_confusion(weighted)
    if scores is not None:
        from sklearn.metrics import average_precision_score, roc_auc_score
        for name, s, truth in [("up", scores, y==0), ("same", -np.abs(scores), y==1), ("down", -scores, y==2)]:
            out["per_class"][name]["average_precision_from_continuous_score"] = float(average_precision_score(truth,s)) if truth.any() else None
            out["per_class"][name]["auroc_from_continuous_score"] = float(roc_auc_score(truth,s)) if len(np.unique(truth))==2 else None
        out["continuous_score_note"] = "up=effect; down=-effect; same=-abs(effect); rankings, NOT class probabilities"
    return out

def probability_metrics(rows, probabilities):
    from sklearn.metrics import roc_auc_score
    m = load_module(CFG["metrics"], "a4_metrics")
    out = m.classification_metrics(rows["label"], probabilities, targets=rows["target"])
    for i, name in enumerate(["up", "same", "down"]):
        truth = rows["label"] == i
        out["per_class"][name]["auroc"] = float(roc_auc_score(truth, probabilities[:, i])) if len(np.unique(truth)) == 2 else None
    return out

def select_thresholds(effect, labels):
    """Exact confusion evaluation on a predeclared 33x33 validation-only grid."""
    effect = np.asarray(effect, dtype=np.float64)
    labels = np.asarray(labels)
    if not np.isfinite(effect).all():
        raise FloatingPointError("non-finite validation effects")
    # Quantiles depend ONLY on VAL model scores, never on TEST or a baseline.
    quantiles = np.linspace(0, 1, 33)
    neg, pos = np.maximum(-effect, 0), np.maximum(effect, 0)
    downs = -np.unique(np.r_[0., np.quantile(neg, quantiles), float(neg.max())+1e-6])
    ups = np.unique(np.r_[0., np.quantile(pos, quantiles), float(pos.max())+1e-6])
    support = np.bincount(labels, minlength=3)
    sorted_by_class = [np.sort(effect[labels == i]) for i in range(3)]
    best = None
    for down in downs:
        low = np.asarray([np.searchsorted(s, down, side="left") for s in sorted_by_class])
        for up in ups:
            high = support - np.asarray([np.searchsorted(s, up, side="right") for s in sorted_by_class])
            cm = np.stack([high, support-low-high, low], axis=1).astype(float)
            f1 = np.divide(2*np.diag(cm), cm.sum(0)+cm.sum(1), out=np.zeros(3), where=(cm.sum(0)+cm.sum(1))>0).mean()
            candidate = (float(f1), -float(abs(down)+up), float(down), float(up))
            if best is None or candidate > best:
                best = candidate
    return {"down": best[2], "up": best[3], "val_macro_f1": best[0], "source": "VAL only", "grid": "33 quantiles per side plus zero and all-same boundary"}

def check_resources():
    import psutil
    rss = psutil.Process().memory_info().rss
    if rss > CFG["max_rss_gib"] * (1 << 30):
        raise MemoryError("TASK_RSS_LIMIT: actual process RSS exceeded declared limit")
    # Deliberately do NOT treat reclaimable NFS page cache as model RSS.
    return {"rss_gib": rss/(1<<30), "unix_time": time.time()}
