#!/usr/bin/env python3
"""One full-width A4 update on four real encoded inputs: startup only."""
import argparse
import os
from pathlib import Path

import numpy as np

from jev_a4.config import load_config
from jev_a4.core import encode, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--semantic-smoke-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("gpu_uuid") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_uuid"])
    import torch
    train.settings(cfg)
    torch.set_num_threads(int(cfg["cpu_threads"]))
    seed = int(cfg.get("seeds", [17])[0])
    train.seed_everything(seed)
    src = args.data_dir / f"seed_{seed}" / "train"
    r = np.load(src / "records.npy", allow_pickle=False)[:4]
    features = np.load(src / "prior_features.npy", allow_pickle=False)[:4]
    raw = np.load(src / "prior_raw_counts.npy", allow_pickle=False)[:4]
    fallback = np.load(src / "prior_fallback.npy", allow_pickle=False)[:4]
    _, ci = np.unique(r["target"], return_inverse=True)
    _, qi = np.unique(r["query"], return_inverse=True)
    _, pi = np.unique(encode.compile_card_ids(features, raw, fallback), return_inverse=True)
    def tensor(array, boolean=False):
        return torch.as_tensor(np.array(array, copy=True), device="cuda",
                               dtype=torch.bool if boolean else torch.float32)
    smoke = args.semantic_smoke_dir
    condition = tensor(np.load(smoke / "condition_final32.npy")[ci])
    query = tensor(np.load(smoke / "query_final32.npy")[qi])
    prior = tensor(np.load(smoke / "cards_final32.npy")[pi])
    joint = tensor(np.load(smoke / "joint_present.npy"))
    cnum = tensor(np.load(args.data_dir / "condition_numeric.npy")[r["target"]])
    qnum = tensor(np.load(args.data_dir / "query_numeric.npy")[r["query"]])
    cmask = torch.ones((len(r), 16), dtype=torch.bool, device="cuda")
    pmask = torch.ones((len(r), 4), dtype=torch.bool, device="cuda")
    labels = torch.as_tensor(r["label"].astype("int64"), device="cuda")
    model = train.vendor_module().make_prior_model(train.model_config(cfg), 12).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    with train.autocast("cuda"):
        c, d = model(condition, cmask, cnum, query, cmask, qnum, prior, pmask,
                     tensor(features), joint=joint, joint_mask=pmask, prior_dropout=0.)
        loss = train.exact_joint_loss(c, d, labels)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    probability = model.probabilities(c.float(), d.float())
    if not torch.isfinite(probability).all() or not torch.allclose(probability.sum(1), torch.ones(len(r), device="cuda"), atol=1e-5):
        raise ValueError("full A4 probabilities invalid")
    train.atomic_json(args.out, {"status": "PASS", "scope": "startup_one_update_not_scientific_result",
        "real_rows": len(r), "decoder_width": 512, "backbone_hidden_width": 2560,
        "loss": float(loss.detach()), "gradient_norm": float(grad),
        "fresh_task_head": True, "old_task_weights_or_predictions_read": False,
        "maximum_gpu_allocated_bytes": torch.cuda.max_memory_allocated()})


if __name__ == "__main__":
    main()
