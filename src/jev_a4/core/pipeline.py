#!/usr/bin/env python3
"""One-shot A4 paper workflow, not a recurring monitoring service.

All children run sequentially on one explicitly chosen GPU. Completed validation
plateau + independent restore precede TEST evaluation. No historical task weights
or old predicted answers enter this workflow. Stage errors stop and seal PARTIAL.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from jev_a4.config import load_config

HERE = Path(__file__).resolve().parent


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_identity(config_path):
    return {"config_sha256": sha(config_path), "code": {
        str(path.relative_to(HERE)): sha(path)
        for path in sorted(HERE.rglob("*.py"))},
        "package_code": {str(path.relative_to(HERE.parent)): sha(path)
                         for path in sorted(HERE.parent.rglob("*.py"))},
        "resolved_config_sha256": hashlib.sha256(json.dumps(load_config(config_path),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def validate_config(cfg):
    if cfg.get("method") != "A4" or cfg.get("smoke"):
        raise ValueError("production workflow must use A4, not a smoke/other method")
    seeds = cfg.get("seeds", [])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("five distinct split seeds required")
    datasets = cfg.get("datasets", [])
    if not datasets or len(set(datasets)) != len(datasets) or any(
            name not in {"hepg2", "jurkat", "rpe1", "k562_essential"} for name in datasets):
        raise ValueError("choose distinct supported cell lines; each is trained independently")
    if cfg.get("split_ratio") != [.7, .1, .2] or cfg.get("exclude_self_pairs") is not True:
        raise ValueError("target-grouped 70/10/20 downstream protocol required")
    if cfg.get("prior") != {"folds": 5, "alpha": 1.0}:
        raise ValueError("A4 requires TRAIN-only fivefold evidence with alpha=1")
    gpu = str(cfg.get("gpu_uuid", ""))
    if not (gpu.isdigit() or gpu.startswith(("GPU-", "MIG-"))) or "," in gpu:
        raise ValueError("select one GPU explicitly by index or UUID")
    forbidden = ("old_checkpoint", "initial_task_checkpoint", "baseline_predictions", "output_anchor")
    if any(cfg.get(key) for key in forbidden):
        raise ValueError("old predictor answers or task weights must not initialize A4")
    schedule = cfg["training"]
    if schedule["min_epochs"] < 10 or schedule["max_epochs"] < schedule["min_epochs"] or schedule["patience"] < 10:
        raise ValueError("full epoch validation-convergence protocol required")


def plan(cfg):
    validate_config(cfg)
    return [{"dataset": dataset, "split_seed": seed, "optimization_seed": seed,
             "task_head_initialization": "fresh", "train_val_test_target_ratio": [7, 1, 2],
             "test_gate": "VAL convergence seal + independent selected-checkpoint restore PASS"}
            for dataset in cfg["datasets"] for seed in cfg["seeds"]]


def metric_values(value, prefix=""):
    """Numerical performance metrics only; supports/counts are kept per seed."""
    excluded = {"support", "predicted_support", "records", "targets", "class_count", "classes_present"}
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if key in excluded or key.endswith("support"):
            continue
        if isinstance(item, dict):
            result.update(metric_values(item, name))
        elif "confusion" not in key and isinstance(item, (float, int)) and not isinstance(item, bool) and math.isfinite(item):
            result[name] = float(item)
    return result


def aggregate(reports, expected_seeds):
    seeds = [int(report["seed"]) for report in reports]
    if len(set(seeds)) != len(seeds) or any(seed not in expected_seeds for seed in seeds):
        raise ValueError("duplicate or unexpected seeds in aggregate")
    values = [metric_values(report["metrics"]) for report in reports]
    result = {}
    for key in sorted(set().union(*(set(value) for value in values))):
        collected = [value[key] for value in values if key in value]
        count = len(collected)
        mean = sum(collected) / count
        variance = sum((value - mean)**2 for value in collected) / (count - 1) if count > 1 else None
        result[key] = {"mean": mean, "sample_variance": variance,
                       "sample_std": math.sqrt(variance) if variance is not None else None,
                       "n_defined": count, "all_five_defined": count == len(expected_seeds)}
    complete = set(seeds) == set(expected_seeds)
    return {"status": "COMPLETE" if complete else "INCOMPLETE", "seeds_completed": seeds,
            "seeds_expected": expected_seeds, "number_of_test_runs": len(reports),
            "variance_definition": "sample variance ddof=1 across five distinct target splits and optimization seeds; overlapping repeated holdouts are not independent folds",
            "metrics": result, "per_seed": reports,
            "missing_class_AP_convention": "null excluded; n_defined disclosed; never treated as zero"}


def verify_artifact_manifest(directory, manifest_name, byte_key):
    directory = Path(directory)
    manifest = read_json(directory / manifest_name)
    if manifest.get("status") != "PASS":
        raise ValueError(f"unsealed artifact directory: {directory}")
    count = total = 0
    for relative, identity in manifest.get("artifacts", {}).items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("artifact may not escape its sealed directory")
        if path.stat().st_size != identity[byte_key] or sha(path) != identity["sha256"]:
            raise ValueError(f"sealed artifact changed: {path}")
        count += 1
        total += path.stat().st_size
    return {"files": count, "bytes": total}


def verify_dataset(data_dir, cache_dir, seeds):
    checks = [verify_artifact_manifest(data_dir, "preparation_report.json", "size_bytes"),
              verify_artifact_manifest(cache_dir, "manifest.json", "bytes")]
    root_digest = sha(Path(cache_dir) / "manifest.json")
    for seed in seeds:
        directory = Path(cache_dir) / f"seed_{seed}"
        manifest = read_json(directory / "manifest.json")
        if manifest["identity"]["shared_manifest_sha256"] != root_digest:
            raise ValueError("seed cache is bound to a different shared cache")
        checks.append(verify_artifact_manifest(directory, "manifest.json", "bytes"))
        for partition in ("train", "val", "test"):
            if sha(directory / partition / "manifest.json") != manifest["partitions"][partition]:
                raise ValueError("partition manifest digest changed")
            checks.append(verify_artifact_manifest(directory / partition, "manifest.json", "bytes"))
    return {"status": "PASS", "files": sum(c["files"] for c in checks),
            "bytes": sum(c["bytes"] for c in checks), "full_payload_sha256_recomputed": True}


class Workflow:
    def __init__(self, cfg, config_path, run_dir):
        self.cfg, self.config_path, self.run_dir = cfg, Path(config_path).resolve(), Path(run_dir).resolve()
        self.project = Path(cfg["project_root"]).resolve()
        if not self.run_dir.is_relative_to(self.project / "runs"):
            raise ValueError("run directory must be under the new project runs directory")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "logs").mkdir()
        self.bound = source_identity(self.config_path)
        self.state = {"status": "RUNNING", "method": "A4", "pid": os.getpid(), "started_at": now(),
                      "current_stage": "initializing", "planned_runs": plan(cfg), "stages": [],
                      "runs": [], "source_identity": self.bound, "gpu_uuid": str(cfg["gpu_uuid"]),
                      "independent_cell_line_training": True, "continuous_monitor_installed": False}
        dump(self.run_dir / "state.json", self.state)
        dump(self.run_dir / "run_plan.json", self.state["planned_runs"])
        self.child = None
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu_uuid"]),
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                        OMP_NUM_THREADS=str(cfg.get("cpu_threads", 8)),
                        MKL_NUM_THREADS=str(cfg.get("cpu_threads", 8)), PYTHONUNBUFFERED="1")

    def checkpoint(self):
        dump(self.run_dir / "state.json", self.state)

    def stop_child(self):
        if self.child is None or self.child.poll() is not None:
            return
        os.killpg(self.child.pid, signal.SIGTERM)
        try:
            self.child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait(timeout=10)

    def stage(self, name, arguments, report_path, expected):
        if source_identity(self.config_path) != self.bound:
            raise ValueError("source/config changed after launch; refusing mixed-code experiment")
        self.state["current_stage"] = name
        row = {"stage": name, "started_at": now(), "status": "RUNNING", "report_path": str(report_path)}
        self.state["stages"].append(row)
        self.checkpoint()
        log_path = self.run_dir / "logs" / (name + ".log")
        command = [self.cfg["runtime_python"], *map(str, arguments)]
        row["log_path"] = str(log_path)
        started = time.monotonic()
        with log_path.open("w") as log:
            self.child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                          env=self.env, start_new_session=True, cwd=HERE)
            row["pid"] = self.child.pid
            self.checkpoint()
            try:
                code = self.child.wait(timeout=float(self.cfg["execution"]["stage_timeout_seconds"]))
            except BaseException:
                self.stop_child()
                row.update(status="STOPPED", finished_at=now(), elapsed_seconds=time.monotonic() - started)
                self.checkpoint()
                raise
            finally:
                self.child = None
        row.update(returncode=code, finished_at=now(), elapsed_seconds=time.monotonic() - started)
        if code:
            row["status"] = "FAIL"
            self.checkpoint()
            raise RuntimeError(f"stage {name} failed with return code {code}; see {log_path}")
        result = read_json(report_path)
        if result.get("status") not in expected:
            row["status"] = "FAIL"
            self.checkpoint()
            raise ValueError(f"unexpected report status in {name}: {result.get('status')}")
        row.update(status=result["status"], report_sha256=sha(report_path))
        self.checkpoint()
        return result

    def run(self):
        try:
            import shutil
            if shutil.disk_usage(self.project).free < self.cfg["execution"]["minimum_disk_free_gib"] * 2**30:
                raise OSError("insufficient persistent disk space for all frozen caches")
            reports_by_cell = {}
            for dataset in self.cfg["datasets"]:
                data = self.project / "data" / dataset
                cache = self.project / "cache" / dataset
                self.stage(dataset + "_prepare", [HERE / "prepare.py", "--config", self.config_path,
                           "--dataset", dataset, "--out", data], data / "preparation_report.json", {"PASS"})
                self.stage(dataset + "_encode_five_splits", [HERE / "encode.py", "--config", self.config_path,
                           "--dataset", dataset, "--data-dir", data, "--cache-dir", cache, "--all-seeds"],
                           cache / f"seed_{self.cfg['seeds'][-1]}" / "manifest.json", {"PASS"})
                reports = []
                for seed in self.cfg["seeds"]:
                    out = self.run_dir / dataset / f"seed_{seed}"
                    common = ["--config", self.config_path, "--dataset", dataset, "--seed", seed,
                              "--data-dir", data, "--cache-dir", cache, "--out", out]
                    trained = self.stage(f"{dataset}_{seed}_train", [HERE / "train_memory_safe.py", "train", *common],
                                         out / "training_report.json", {"CONVERGED", "NOT_CONVERGED"})
                    record = {"dataset": dataset, "seed": seed, "epochs_completed": trained["epochs_completed"],
                              "selected_epoch": trained["selected_epoch"], "training_status": trained["status"],
                              "checkpoint_path": str(out / "best.pt"), "checkpoint_sha256": trained["best_checkpoint_sha256"],
                              "main_exposure_min": trained["main_exposure_min"], "main_exposure_max": trained["main_exposure_max"]}
                    self.state["runs"].append(record)
                    if trained["status"] != "CONVERGED":
                        record["test_status"] = "LOCKED_NOT_CONVERGED"
                        self.checkpoint()
                        if not self.cfg["execution"]["continue_other_seeds_if_not_converged"]:
                            raise RuntimeError("validation convergence not reached; TEST remains locked")
                        continue
                    tested = self.stage(f"{dataset}_{seed}_test", [HERE / "train_memory_safe.py", "evaluate", *common],
                                        out / "test_report.json", {"PASS"})
                    if tested["checkpoint_sha256"] != trained["best_checkpoint_sha256"]:
                        raise ValueError("TEST checkpoint is not the selected training checkpoint")
                    if tested["test_used_for_training_or_selection"] is not False:
                        raise ValueError("TEST selection leakage")
                    record["test_status"] = "PASS"
                    reports.append(tested)
                    reports_by_cell[dataset] = aggregate(reports, self.cfg["seeds"])
                    dump(self.run_dir / dataset / "summary.json", reports_by_cell[dataset])
                    self.checkpoint()
                if self.cfg["execution"]["verify_full_cache_payload_at_finish"]:
                    self.state["current_stage"] = dataset + "_final_integrity"
                    self.checkpoint()
                    verification = verify_dataset(data, cache, self.cfg["seeds"])
                    dump(self.run_dir / dataset / "cache_integrity.json", verification)
                reports_by_cell[dataset] = aggregate(reports, self.cfg["seeds"])
                dump(self.run_dir / dataset / "summary.json", reports_by_cell[dataset])
            complete = len(self.state["runs"]) == len(self.state["planned_runs"]) and all(
                item.get("test_status") == "PASS" for item in self.state["runs"])
            if source_identity(self.config_path) != self.bound:
                raise ValueError("source/config drift at final sealing")
            report = {"status": "PASS" if complete else "PARTIAL", "scientific_status": "COMPARISONS_PENDING",
                      "method": "A4", "started_at": self.state["started_at"], "finished_at": now(),
                      "runs_planned": len(self.state["planned_runs"]), "runs_tested": sum(
                          item.get("test_status") == "PASS" for item in self.state["runs"]),
                      "datasets": reports_by_cell, "runs": self.state["runs"], "source_identity": self.bound,
                      "old_prediction_output_anchor": False, "old_task_checkpoint_initialization": False,
                      "direct_current_model_prediction": True, "continuous_monitor_installed": False,
                      "scope": "strict statistically supported non-self pairs; conditional performance, not full-grid performance"}
            dump(self.run_dir / "report.json", report)
            write_summary(self.run_dir, report)
            self.state.update(status=report["status"], current_stage="FINISHED", finished_at=report["finished_at"])
            self.checkpoint()
            return 0 if complete else 2
        except BaseException as error:
            self.stop_child()
            self.state.update(status="PARTIAL", finished_at=now(), error=f"{type(error).__name__}: {error}")
            self.checkpoint()
            dump(self.run_dir / "report.json", {**self.state, "scientific_status": "NOT_COMPLETE",
                 "traceback": traceback.format_exc(), "automatic_retry": False})
            (self.run_dir / "report.md").write_text(f"# A4 实验未完整完成\n\n状态：PARTIAL。\n\n停止阶段：{self.state['current_stage']}\n\n原因：{type(error).__name__}: {error}\n\n未执行失败后的训练或测试；请查看该阶段日志。\n")
            return 2


def write_summary(run_dir, report):
    lines = ["# A4 五随机划分论文实验", "", f"工程状态：{report['status']}；测试完成：{report['runs_tested']}/{report['runs_planned']}。", "",
             "各细胞系独立报告；均值与样本方差来自五种不同靶点划分及相应训练种子。尚未完成所有对比方法，不能据此宣称优于基线。", "",
             "| 细胞系 | 测试次数 | Macro-F1 均值 | 样本方差 | Balanced accuracy 均值 |", "| --- | ---: | ---: | ---: | ---: |"]
    rows = []
    for dataset, result in report["datasets"].items():
        metrics = result["metrics"]
        def fmt(key, field):
            value = metrics.get(key, {}).get(field)
            return "NA" if value is None else f"{value:.6f}"
        lines.append(f"| {dataset} | {result['number_of_test_runs']} | {fmt('macro_f1','mean')} | {fmt('macro_f1','sample_variance')} | {fmt('balanced_accuracy','mean')} |")
        for key, value in metrics.items():
            rows.append({"dataset": dataset, "metric": key, **value})
    lines += ["", "各类 precision、recall、F1、AP，变化检测指标，NLL、Brier 和靶点等权结果见 report.json 与 metrics_summary.csv。", "",
              "未裁决组合不计作 Same；当前标签统计为细胞级 Welch/TOST 近似，不等于生物重复推断。", ""]
    (Path(run_dir) / "report.md").write_text("\n".join(lines))
    with (Path(run_dir) / "metrics_summary.csv").open("w", newline="") as stream:
        fields = ["dataset", "metric", "mean", "sample_variance", "sample_std", "n_defined", "all_five_defined"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.plan:
        print(json.dumps(plan(cfg), ensure_ascii=False, indent=2))
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required unless --plan")
    validate_config(cfg)
    workflow = Workflow(cfg, args.config, args.run_dir)
    def cancelled(signum, frame):
        raise KeyboardInterrupt(f"workflow received signal {signum}")
    signal.signal(signal.SIGTERM, cancelled)
    return workflow.run()


if __name__ == "__main__":
    raise SystemExit(main())
