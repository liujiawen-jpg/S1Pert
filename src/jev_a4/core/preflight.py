#!/usr/bin/env python3
"""Bounded startup checks. Formal training is not started by this program."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from jev_a4.core import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = pipeline.load_config(args.config)
    pipeline.validate_config(cfg)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = pipeline.source_identity(args.config)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu_uuid"]), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS=str(cfg["cpu_threads"]), MKL_NUM_THREADS=str(cfg["cpu_threads"]))
    checks = []
    def check(name, command, timeout):
        started = time.monotonic()
        with (out / (name + ".log")).open("w") as log:
            proc = subprocess.run(list(map(str, command)), env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout, check=False)
        checks.append({"check": name, "status": "PASS" if proc.returncode == 0 else "FAIL",
                       "returncode": proc.returncode, "elapsed_seconds": time.monotonic() - started})
        pipeline.dump(out / "progress.json", {"checks": checks, "formal_training_started": False})
        if proc.returncode:
            raise RuntimeError(f"{name} failed; see its preflight log")
    try:
        project = Path(cfg["project_root"])
        project.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(project).free < cfg["execution"]["minimum_disk_free_gib"] * 2**30:
            raise OSError("insufficient disk space")
        weight = Path(cfg["decider_snapshot"]) / "model.safetensors"
        if pipeline.sha(weight) != cfg["expected_backbone_weight_sha256"]:
            raise ValueError("frozen generic DECIDER snapshot SHA mismatch")
        checks.append({"check": "generic_backbone_weight_sha256", "status": "PASS",
                       "task_checkpoint_initialization": False})
        # Preparation is reusable, fully sealed, and performs five real split
        # and TRAIN-only evidence audits for each cell, without any training.
        for dataset in cfg["datasets"]:
            check(dataset + "_real_prepare", [cfg["runtime_python"], pipeline.HERE / "prepare.py",
                  "--config", args.config, "--dataset", dataset, "--out", project / "data" / dataset], 1800)
        first = cfg["datasets"][0]
        check("real_decider_gpu_smoke", [cfg["runtime_python"], pipeline.HERE / "encode.py",
              "--config", args.config.resolve(), "--dataset", first, "--data-dir", project / "data" / first,
              "--cache-dir", out / "real_decider_gpu_smoke", "--seed", str(cfg["seeds"][0]), "--smoke", "--max-rows", "4"], 900)
        check("full_A4_head_real_input_gpu_smoke", [cfg["runtime_python"], pipeline.HERE / "head_smoke.py",
              "--config", args.config.resolve(), "--data-dir", project / "data" / first,
              "--semantic-smoke-dir", out / "real_decider_gpu_smoke", "--out", out / "full_A4_head_report.json"], 300)
        check("synthetic_head_gpu_chain", [cfg["runtime_python"], pipeline.HERE / "train_memory_safe.py", "smoke",
              "--out", out / "synthetic_head_gpu_chain", "--device", "cuda"], 900)
        if pipeline.source_identity(args.config) != source:
            raise ValueError("source/config drift during preflight")
        report = {"status": "PASS", "scope": "startup_engineering_checks_not_scientific_results",
                  "source_identity": source, "checks": checks, "formal_training_started": False,
                  "gpu_uuid": cfg["gpu_uuid"], "finished_at": pipeline.now()}
        pipeline.dump(out / "report.json", report)
        print(json.dumps({"status": "PASS", "report": str(out / "report.json")}, ensure_ascii=False))
    except BaseException as error:
        pipeline.dump(out / "report.json", {"status": "PARTIAL", "checks": checks,
                      "error": f"{type(error).__name__}: {error}", "formal_training_started": False,
                      "source_identity": source})
        raise


if __name__ == "__main__":
    main()
