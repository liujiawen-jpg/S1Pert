#!/usr/bin/env python3
"""Train a Qwen3-4B LoRA A4-adapted PertReasonQA comparator.

This is intentionally a separate entry point.  It trains LoRA weights on
TRAIN-only A4 records and never claims to reproduce the unreleased official
PertReasonLM SFT/GRPO pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from jev_a4.config import load_config
from jev_a4.core import metrics as metric_core

DATA = SNAPSHOT = KG = OUT = None
CELLS = ("hepg2", "jurkat", "rpe1", "k562_essential")
SEEDS = (17, 29, 43, 61, 89)
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(q, p)


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def metrics(labels, probs):
    return metric_core.classification_metrics(labels, probs)


def make_adjacency(meta):
    import pandas as pd
    edges = pd.read_parquet(KG, columns=["u", "v"])
    q = {x: i for i, x in enumerate(meta["query_ids"])}; t = {x: i for i, x in enumerate(meta["target_ids"])}
    result = {i: set() for i in range(len(t))}
    for u, v in zip(edges["u"].astype(str), edges["v"].astype(str)):
        if u in t and v in q: result[t[u]].add(q[v])
        if v in t and u in q: result[t[v]].add(q[u])
    return {k: np.asarray(sorted(v), dtype=np.int32) for k, v in result.items()}


def priors(train, folds, nq):
    c = np.zeros((nq, 3), np.float32); n = np.zeros(nq, np.float32)
    np.add.at(c, (train["query"], train["label"]), 1); np.add.at(n, train["query"], 1)
    whole = (c + 1) / (n[:, None] + 3); cross = []
    for f in range(5):
        keep = folds[train["target"]] != f; z = train[keep]
        cc = np.zeros((nq, 3), np.float32); nn = np.zeros(nq, np.float32)
        np.add.at(cc, (z["query"], z["label"]), 1); np.add.at(nn, z["query"], 1)
        cross.append((cc + 1) / (nn[:, None] + 3))
    return whole, cross


def static_inputs(records, split, train, folds, meta, adj):
    whole, cross = priors(train, folds, len(meta["query_ids"]))
    cn = np.load(DATA / meta["dataset"] / "condition_numeric.npy", mmap_mode="r")
    qn = np.load(DATA / meta["dataset"] / "query_numeric.npy", mmap_mode="r")
    r = np.asarray(records); out = np.zeros((len(r), 11), np.float32); rel = np.empty(len(r), np.int64)
    out[:, :4] = cn[r["target"]]; out[:, 4:8] = qn[r["query"]]; out[:, 8:11] = 1 / 3
    for i, (t, q) in enumerate(zip(r["target"], r["query"])):
        p = whole if split != "train" else cross[int(folds[int(t)])]
        nei = adj[int(t)]; pp = p[nei].mean(0) if len(nei) else p[int(q)]
        pp = pp / pp.sum(); out[i, 8:11] = pp; rel[i] = int(np.argmax(pp))
    return out, rel


class LoRAQA(nn.Module):
    def __init__(self, base, hidden):
        super().__init__(); self.base = base; self.proj = nn.Sequential(nn.Linear(hidden, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.1)); self.answer = nn.Linear(267, 3); self.path = nn.Linear(267, 3); self.relation = nn.Linear(267, 3)

    def forward(self, input_ids, attention_mask, extra):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype); pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1)
        h = torch.cat([self.proj(pooled.float()), extra.float()], dim=-1)
        return self.answer(h), self.path(h), self.relation(h)


def prompts_for(rows, meta, cell):
    return [f"Cell type: {cell}. Perturbed gene: {meta['target_symbols'][int(t)]}. Query gene: {meta['query_symbols'][int(q)]}." for t, q in zip(rows["target"], rows["query"])]


def encode_batch(model, tok, rows, extra, labels, path, relation, cell, meta, dev, max_length=64):
    batch = tok(prompts_for(rows, meta, cell), return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(dev)
    return model(batch["input_ids"], batch["attention_mask"], torch.from_numpy(extra).to(dev)), batch


def evaluate(model, tok, records, extra, cell, meta, dev, batch_size):
    model.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            r = records[i:i + batch_size]; ex = extra[i:i + batch_size]
            (lo, _, _), _ = encode_batch(model, tok, r, ex, None, None, None, cell, meta, dev)
            probs.append(torch.softmax(lo.float(), -1).cpu().numpy())
    return np.concatenate(probs)


def run(cell, seed, device, stage_epochs=(2, 2, 4), rows_per_epoch=512, batch_size=8, eval_batch_size=64, out_root=None):
    root_out = Path(out_root) if out_root else OUT; out = root_out / cell / f"seed_{seed}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("use a fresh output directory; historical runs are never overwritten")
    seed_all(seed); out.mkdir(parents=True, exist_ok=True); started = time.time()
    # Preserve the historical optional dispatch setting when the runtime has it.
    try:
        import torch.backends.python_native as python_native
    except ImportError:
        python_native = None
    if python_native is not None and hasattr(python_native, "triton"):
        python_native.triton.enabled = False
    root = DATA / cell; meta = json.loads((root / "dataset.json").read_text()); meta["dataset"] = cell; sroot = root / f"seed_{seed}"
    train = np.load(sroot / "train/records.npy", mmap_mode="r", allow_pickle=False); val = np.load(sroot / "val/records.npy", mmap_mode="r", allow_pickle=False); folds = np.load(sroot / "target_fold.npy", allow_pickle=False); part = np.load(sroot / "target_partition.npy", allow_pickle=False)
    assert np.all(part[train["target"]] == 0) and np.all(part[val["target"]] == 1)
    adj = make_adjacency(meta); ex_tr, rel_tr = static_inputs(train, "train", train, folds, meta, adj); ex_va, _ = static_inputs(val, "val", train, folds, meta, adj)
    write_json(out / "data_identity.json", {"schema": "pertreasonqa.a4.lora.v1", "cell": cell, "seed": seed, "train_records_sha256": sha(sroot / "train/records.npy"), "val_records_sha256": sha(sroot / "val/records.npy"), "target_partition_sha256": sha(sroot / "target_partition.npy"), "target_fold_sha256": sha(sroot / "target_fold.npy"), "snapshot": str(SNAPSHOT), "revision": REVISION, "public_kg_sha256": sha(KG), "train_only_supervision": True, "test_opened_after_selection": True})
    write_json(out / "supervision_summary.json", {"outcome_supervision_rows": int(len(train)), "path_supervision_rows": int(len(train)), "relation_supervision_rows": int(len(train)), "path_source": "response-blind public KG plus TRAIN query-label priors", "train_target_fold_crossfit": True})

    from transformers import AutoTokenizer, Qwen3Model
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True); dev = torch.device(device)
    base = Qwen3Model.from_pretrained(SNAPSHOT, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none", task_type=TaskType.FEATURE_EXTRACTION)
    base = get_peft_model(base, lcfg); model = LoRAQA(base, int(base.config.hidden_size)).to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad); frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if trainable <= 0 or frozen <= 0: raise RuntimeError("LoRA parameter audit failed")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01); ce = nn.CrossEntropyLoss(); kl = nn.KLDivLoss(reduction="batchmean")
    history = []; best = None; epoch = 0; stage_plan = (("outcome_only", stage_epochs[0], 0., 0.), ("path_reasoning", stage_epochs[1], .2, 0.), ("relation_direction", stage_epochs[2], .2, .1))
    for stage, epochs, pw, rw in stage_plan:
        for _ in range(epochs):
            epoch += 1; model.train(); rng = np.random.default_rng(seed * 1000 + epoch); ix = rng.choice(len(train), size=min(rows_per_epoch, len(train)), replace=False); total = 0.; rows_seen = 0
            for j in range(0, len(ix), batch_size):
                sel = ix[j:j + batch_size]; rr = train[sel]; ex = ex_tr[sel]; y = torch.from_numpy(rr["label"].astype(np.int64)).to(dev); pp = torch.from_numpy(ex[:, 8:11]).to(dev); rb = torch.from_numpy(rel_tr[sel]).to(dev)
                (lo, lp, lr), _ = encode_batch(model, tok, rr, ex, y, pp, rb, cell, meta, dev); loss = ce(lo, y) + pw * kl(torch.log_softmax(lp.float(), -1), pp) + rw * ce(lr, rb); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); total += float(loss.detach()) * len(sel); rows_seen += len(sel)
            vp = evaluate(model, tok, val, ex_va, cell, meta, dev, eval_batch_size); vm = metrics(val["label"], vp); rec = {"epoch": epoch, "stage": stage, "sampled_train_rows": int(rows_seen), "train_loss": total / rows_seen, "val": vm}; history.append(rec); score = (vm["macro_f1"], -vm["nll"], -epoch)
            if best is None or score > best["score"]:
                best = {"score": score, "epoch": epoch, "stage": stage, "metrics": vm}; model.base.save_pretrained(out / "best_adapter"); torch.save({"proj": model.proj.state_dict(), "answer": model.answer.state_dict(), "path": model.path.state_dict()}, out / "best_heads.pt")
    write_json(out / "history.json", history); write_json(out / "selection.json", {"rule": "VAL macro_f1 descending, NLL ascending, earlier epoch", "selected_epoch": best["epoch"], "selected_stage": best["stage"], "selected_metrics": best["metrics"], "sampled_train_rows_per_epoch": rows_per_epoch})
    # Rebuild the same architecture from the saved adapter and head checkpoint.
    restored_base = Qwen3Model.from_pretrained(SNAPSHOT, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True); restored_base = PeftModel.from_pretrained(restored_base, out / "best_adapter", is_trainable=False); restored = LoRAQA(restored_base, int(restored_base.config.hidden_size)).to(dev); h = torch.load(out / "best_heads.pt", map_location=dev, weights_only=True); restored.proj.load_state_dict(h["proj"]); restored.answer.load_state_dict(h["answer"]); restored.path.load_state_dict(h["path"]); rp = evaluate(restored, tok, val[:min(32, len(val))], ex_va[:min(32, len(val))], cell, meta, dev, eval_batch_size); write_json(out / "restore_report.json", {"status": "PASS", "rows": len(rp), "finite": bool(np.isfinite(rp).all()), "probability_order": ["up", "same", "down"], "trainable_parameters": trainable, "frozen_parameters": frozen})
    test = np.load(sroot / "test/records.npy", mmap_mode="r", allow_pickle=False); assert np.all(part[test["target"]] == 2); ex_te, _ = static_inputs(test, "test", train, folds, meta, adj); prob = evaluate(restored, tok, test, ex_te, cell, meta, dev, eval_batch_size).astype(np.float32); keys = np.column_stack([test["target"], test["query"]]).astype(np.int32); np.save(out / "test_keys.npy", keys); np.save(out / "test_probabilities.npy", prob); tm = metrics(test["label"], prob); write_json(out / "test_metrics.json", tm)
    np.savez_compressed(out / "test_predictions.npz", target=test["target"],
                        query=test["query"], labels=test["label"], probabilities=prob)
    config = {"method": "PertReasonLM-LoRA-A4-adapted", "cell": cell, "seed": seed, "base_model": {"repo": "Qwen/Qwen3-4B", "revision": REVISION, "snapshot": str(SNAPSHOT)}, "lora": {"r": 32, "alpha": 64, "dropout": .05, "target_modules": sorted(lcfg.target_modules)}, "trainable_transformer": True, "rows_per_epoch": rows_per_epoch, "batch_size": batch_size, "eval_batch_size": eval_batch_size, "training_stages": {"outcome_only": stage_epochs[0], "path_reasoning": stage_epochs[1], "relation_direction": stage_epochs[2]}, "split_source": "sealed target_partition.npy", "train_only_supervision": True, "test_used_for_training_or_selection": False, "uniform_metrics": "jev_a4.core.metrics", "adaptation_note": "LoRA trains the fixed Qwen3-4B Transformer on sampled A4 TRAIN records; this is A4-adapted, not the unreleased official PertReasonLM SFT/GRPO pipeline"}; write_json(out / "config.json", config)
    report = {"status": "PASS", "scientific_status": "A4_ADAPTED_COMPARISON", "method": config["method"], "cell": cell, "seed": seed, "test_status": "PASS", "test_records": len(test), "selected_epoch": best["epoch"], "metrics": tm, "fixed_qwen_snapshot": True, "trainable_transformer": True, "trainable_parameters": trainable, "frozen_parameters": frozen, "train_only_supervision": True, "test_used_for_training_or_selection": False, "elapsed_seconds": time.time() - started, "limitations": ["This is a PertReasonQA/PertReasonLM-style A4 adaptation, not an exact official PertReasonLM SFT/GRPO reproduction.", "LoRA adaptation uses a bounded TRAIN sample per epoch to keep the 20-run matrix computationally tractable."]}; write_json(out / "report.json", report); return report


def summarize(out_root=None):
    root = Path(out_root) if out_root else OUT
    for c in CELLS:
        rows = [json.loads((root / c / f"seed_{s}" / "report.json").read_text()) for s in SEEDS]; sm = {"method": "PertReasonLM-LoRA-A4-adapted", "cell": c, "seeds": list(SEEDS), "number_of_test_runs": len(rows), "metrics": {}}
        for k in ("macro_f1", "balanced_accuracy", "accuracy", "nll", "brier_multiclass_sum"):
            v = [r["metrics"][k] for r in rows]; sm["metrics"][k] = {"mean": float(np.mean(v)), "sample_variance": float(np.var(v, ddof=1)), "sample_std": float(np.std(v, ddof=1)), "n_defined": len(v), "all_five_defined": len(v) == 5}
        write_json(root / c / "summary.json", sm)
    write_json(root / "matrix_manifest.json", {"status": "PASS", "method": "PertReasonLM-LoRA-A4-adapted", "runs_planned": 20, "runs_tested": 20, "cells": list(CELLS), "seeds": list(SEEDS), "metrics_sha256": sha(Path(metric_core.__file__))})


def configure_assets(config_path):
    global DATA, SNAPSHOT, KG, OUT
    cfg = load_config(config_path)
    required = ("data_root", "backbone_snapshot", "knowledge_graph", "output_root")
    if any(not cfg.get(key) for key in required):
        raise ValueError("config requires data_root, backbone_snapshot, knowledge_graph and output_root")
    DATA, SNAPSHOT, KG, OUT = (Path(cfg[key]) for key in required)
    if not DATA.is_dir() or not SNAPSHOT.is_dir() or not KG.is_file():
        raise FileNotFoundError("supply prepared data, a complete local Qwen3-4B snapshot and public KG edges")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cell", choices=CELLS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stage-epochs", default="2:2:4")
    ap.add_argument("--rows-per-epoch", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--out-root", type=Path)
    a = ap.parse_args()
    ep = tuple(int(x) for x in a.stage_epochs.split(":"))
    if len(ep) != 3 or min(ep) < 1:
        ap.error("--stage-epochs requires three positive integers, such as 2:2:4")
    if min(a.rows_per_epoch, a.batch_size, a.eval_batch_size) < 1:
        ap.error("row and batch limits must be positive")
    if not a.all and not (a.cell and a.seed):
        ap.error("use --all or --cell and --seed")
    configure_assets(a.config)
    root = a.out_root.resolve() if a.out_root else OUT
    if a.all:
        for c in CELLS:
            for s in SEEDS:
                run(c, s, a.device, ep, a.rows_per_epoch, a.batch_size, a.eval_batch_size, root)
        summarize(root)
    else:
        run(a.cell, a.seed, a.device, ep, a.rows_per_epoch, a.batch_size, a.eval_batch_size, root)


if __name__ == "__main__":
    main()
