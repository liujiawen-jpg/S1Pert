#!/usr/bin/env python3
"""Detach exactly one checked workflow; no recurring monitoring or retry loop."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

from jev_a4.core import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()
    cfg = pipeline.load_config(args.config)
    pipeline.validate_config(cfg)
    checks = pipeline.read_json(args.preflight_report)
    if checks.get("status") != "PASS" or checks.get("scope") != "startup_engineering_checks_not_scientific_results":
        raise ValueError("startup requires a completed engineering preflight, not a historical report")
    if checks["source_identity"] != pipeline.source_identity(args.config):
        raise ValueError("preflight used different source or configuration")
    official = Path(cfg["project_root"])
    current = pipeline.source_identity(args.config)["code"]
    # This repository itself is the versioned source, not a temporary copy of
    # a machine-specific second code directory. Preflight binds every module.
    previous = official / "launch.json"
    if previous.exists():
        identity = pipeline.read_json(previous)
        state_path = Path(identity["run_dir"]) / "state.json"
        if not state_path.exists() or pipeline.read_json(state_path).get("status") == "RUNNING":
            raise RuntimeError("previous launch is active or unverified; refusing a duplicate")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = official / "runs" / ("a4_five_splits_" + stamp)
    logs = official / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / (run_dir.name + ".log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu_uuid"]), PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    with log_path.open("x") as log:
        proc = subprocess.Popen([cfg["runtime_python"], str(Path(__file__).with_name("pipeline.py")),
                                 "--config", str(args.config.resolve()), "--run-dir", str(run_dir)],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True, cwd=Path(__file__).parent)
    report = {"status": "LAUNCHED", "pid": proc.pid, "run_dir": str(run_dir),
              "log_path": str(log_path), "gpu_uuid": cfg["gpu_uuid"],
              "preflight_report": str(args.preflight_report.resolve()),
              "preflight_report_sha256": pipeline.sha(args.preflight_report),
              "source_identity": current, "started_at": pipeline.now(),
              "continuous_monitor_installed": False}
    pipeline.dump(previous, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
