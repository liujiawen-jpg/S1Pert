"""Portable paper-workspace initializer and explicit execution planner.

Use --work-dir once, or export S1PERT_ROOT. Expensive actions print their exact
commands unless --execute is supplied. Existing outputs are never erased.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
CELLS = ("hepg2", "jurkat", "rpe1", "k562_essential")
SEEDS = (17, 29, 43, 61, 89)
METHODS = ("S1Pert", "GenePT-Gene", "GenePT-Prot", "Qwen3-4B-LoRA",
           "GEARS", "scGPT", "STATE", "ALIGNED", "SUMMER", "P1")
CLASSICAL = {"GenePT-Gene": "genept_gene", "GenePT-Prot": "genept_prot",
             "GEARS": "gears", "scGPT": "scgpt"}


def workspace(value=None):
    value = value or os.environ.get("S1PERT_ROOT")
    if not value:
        raise ValueError("Specify --work-dir or set S1PERT_ROOT")
    return Path(value).expanduser().resolve()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")


def initialize(work, data_root=None, gpu="0"):
    """Create relocatable configs without downloading or reading biological data."""
    work = workspace(work); config_dir = work / "configs"
    if config_dir.exists():
        raise FileExistsError("Workspace configs already exist; edit them explicitly or choose a new workspace")
    raw = Path(data_root).expanduser().resolve() if data_root else work / "raw"
    config_dir.mkdir(parents=True)
    def rel(path): return os.path.relpath(path, config_dir)
    a4 = json.loads((REPO / "configs/a4.example.json").read_text())
    a4.update(project_root="..", data_preprocessed_root="../preprocessed",
              label_resolution_root="../strict_labels", knowledge_db="../assets/public_functions.sqlite",
              decider_snapshot="../assets/DECIDER-4b-v1", tokenizer_snapshot="../assets/DECIDER-4b-v1",
              decider_source="../assets/DECIDER-4b-v1", gpu_uuid=str(gpu), runtime_python=None)
    write_json(config_dir / "a4.json", a4)
    pp = json.loads((REPO / "configs/preprocess.example.json").read_text())
    pp["preprocess"]["output"] = "../preprocessed"
    for row in pp["preprocess"]["datasets"].values():
        row["raw"] = rel(raw / row["raw"].split("data/raw/", 1)[1])
    write_json(config_dir / "preprocess.json", pp)
    write_json(config_dir / "labels.json", {"labels": {"source": "../preprocessed", "output": "../strict_labels"}})
    downloads = json.loads((REPO / "configs/downloads.json").read_text())
    downloads["root"] = ".."
    for row in downloads["assets"]:
        if row["group"] == "data":
            # The asset downloader forbids destinations outside its configured
            # root. An external --data-root is an existing input location, not
            # an authorization to download outside this workspace.
            row["path"] = row["path"].replace("data/raw/", "raw/", 1)
    write_json(config_dir / "downloads.json", downloads)
    metrics = REPO / "src/jev_a4/core/metrics.py"
    for family in ("classical", "state", "summer"):
        cfg = json.loads((REPO / "baselines" / family / "config.example.json").read_text())
        cfg["gpu_uuid"] = str(gpu)
        mapping = dict(root=work / "baselines/classical", a4_data=work / "data", a4_root=work,
                       moments=work / "preprocessed", raw_root=raw, metrics=metrics, a4_metrics=metrics,
                       scgpt_source=work / "external/scGPT", gears_source=work / "external/GEARS",
                       scgpt_pretrained=work / "assets/scgpt", gene2go=work / "assets/baselines/gene2go_all.pkl",
                       perturbqa_source=work / "external/PerturbQA", assets_root=work / "assets/baselines",
                       expression_root=work / "baselines/classical/prepared", esm_source=work / "assets/esm2",
                       official_source=work / "external/state", model=work / "assets/qwen3_8b",
                       knowledge_db=work / "assets/public_functions.sqlite", public_edges_db=work / "assets/public_edges.sqlite")
        for key in cfg.keys() & mapping.keys(): cfg[key] = rel(mapping[key])
        write_json(config_dir / (family + ".json"), cfg)
    write_json(config_dir / "qwen3_lora.json", dict(data_root="../data", backbone_snapshot="../assets/Qwen3-4B",
               knowledge_graph="../assets/public_edges.parquet", output_root="../baselines/qwen3_lora"))
    aligned = json.loads((REPO / "baselines/aligned/configs/official_default_a4.json").read_text())
    write_json(config_dir / "aligned.json", aligned)
    write_json(work / "workspace.json", dict(schema="s1pert.paper-workspace.v1", cells=CELLS, seeds=SEEDS,
               raw_data_directory=rel(raw), initialized_without_training=True,
               external_data_root_is_read_only=bool(data_root)))
    print(f"Created configs in {config_dir}; no data downloaded and no model started.")


def command(script, *args, python=None):
    return [python or sys.executable, str(REPO / script), *map(str, args)]


def plan(work, action, methods, cells=CELLS, seeds=SEEDS, python=None):
    work = workspace(work); cfg = work / "configs"; cmds = []
    def core(script, *args): return command("src/jev_a4/core/" + script, *args, python=python)
    def cli(*args): return [python or sys.executable, "-m", "jev_a4", *map(str, args)]
    if action == "download":
        return [command("scripts/download_assets.py", "--manifest", cfg / "downloads.json", "--root", work,
                        "--group", group, *( ["--include-weights"] if group == "backbone" else []), python=python)
                for group in ("data", "annotations", "backbone")]
    if action == "data":
        return [cli("preprocess", "--config", cfg / "preprocess.json"), cli("labels", "--config", cfg / "labels.json"),
                cli("validate-labels", "--source", work / "preprocessed", "--bundle", work / "strict_labels"),
                cli("import-knowledge", "--tsv", work / "assets/uniprot_reviewed_human.tsv", "--out", work / "assets/public_functions.sqlite")]
    if action == "main":
        # Core pipeline itself enforces configured independent cells and five
        # seeds, full epochs, validation convergence and independent restoration.
        return [core("pipeline.py", "--config", cfg / "a4.json", "--run-dir", work / "runs/main")]
    if action == "prepare":
        return [core("prepare.py", "--config", cfg / "a4.json", "--dataset", cell,
                     "--out", work / "data" / cell) for cell in cells]
    if action == "encode":
        return [core("encode.py", "--config", cfg / "a4.json", "--dataset", cell,
                     "--data-dir", work / "data" / cell, "--cache-dir", work / "cache" / cell,
                     "--all-seeds") for cell in cells]
    if action != "baselines": raise ValueError(action)
    if any(m in methods for m in ("STATE", "SUMMER")) and (tuple(cells) != CELLS or tuple(seeds) != SEEDS):
        raise ValueError("STATE/SUMMER queues and summaries use their configuration's full matrix. "
                         "Edit the corresponding workspace config explicitly and use the native "
                         "baseline entry point for a subset; CLI cell/seed overrides are not ignored.")
    def init_family(family):
        b = work / "baselines" / family
        if not (b / "workspace.json").exists():
            cmds.append(command("baselines/run.py", "init", "--family", family, "--config", cfg / (family + ".json"), "--work", b, python=python))
        return b
    def native(b, *args):
        return command("baselines/run.py", "execute", "--work", b, "--", *args, python=python)
    cl_methods = [m for m in methods if m in CLASSICAL]
    if cl_methods or "STATE" in methods:
        b = init_family("classical")
        if any(m.startswith("GenePT") for m in cl_methods):
            cmds += [native(b, "prepare.py", "genept", "--cell", cell) for cell in cells]
        if any(m in methods for m in ("GEARS", "scGPT", "STATE")):
            cmds += [native(b, "prepare.py", "expression", "--cell", cell) for cell in cells]
        for method in cl_methods:
            for cell in cells:
                for seed in seeds:
                    if method == "GEARS": cmds.append(native(b, "prepare.py", "native_graphs", "--cell", cell, "--seed", seed))
                    cmds.append(native(b, "train.py", "train", "--method", CLASSICAL[method], "--cell", cell, "--seed", seed,
                                       "--out", b / "runs" / CLASSICAL[method] / cell / f"seed_{seed}"))
    if "STATE" in methods:
        b = init_family("state")
        cmds.append(native(b, "run.py", "preflight"))
        for cell in cells:
            for seed in seeds:
                cmds.extend([native(b, "run.py", stage, "--cell", cell, "--seed", seed) for stage in ("train", "evaluate")])
        cmds.append(native(b, "run.py", "summary"))
    if "SUMMER" in methods:
        b = init_family("summer")
        cmds += [native(b, "fast_summer.py", stage) for stage in ("preflight", "pipeline")]
    if "Qwen3-4B-LoRA" in methods:
        for cell in cells:
            for seed in seeds:
                cmds.append(command("baselines/language/qwen3_lora.py", "--config", cfg / "qwen3_lora.json",
                                    "--cell", cell, "--seed", seed, python=python))
    if "ALIGNED" in methods:
        for cell in cells:
            for seed in seeds:
                for stage in ("prepare", "train", "evaluate"):
                    cmds.append([python or sys.executable, "-m", "baselines.aligned.run", stage,
                                 "--work-dir", str(work), "--cell", cell, "--seed", str(seed),
                                 "--config", str(cfg / "aligned.json"), "--device", "cuda"])
    if "P1" in methods:
        for cell in cells:
            for seed in seeds:
                cmds.append(command("scripts/export_prior.py", "--partition", work / "data" / cell / f"seed_{seed}/test",
                                    "--out", work / "baselines/p1" / cell / f"seed_{seed}/test_predictions.npz", python=python))
    return cmds


def execute(commands, work, action):
    logs = work / "paper/execution_logs"; logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{time.time_ns()}"
    env = os.environ.copy(); env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
    env.setdefault("PYTHONHASHSEED", "0")
    a4_config = work / "configs/a4.json"
    if a4_config.exists():
        gpu = json.loads(a4_config.read_text()).get("gpu_uuid")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for i, cmd in enumerate(commands):
        start = time.perf_counter(); log = logs / f"{action}_{stamp}_{i:03d}.log"
        with log.open("x") as f:
            p = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        write_json(log.with_suffix(".json"), dict(command=cmd, seconds=time.perf_counter()-start, returncode=p.returncode,
                                                 action=action, log=log.name))
        if p.returncode: raise RuntimeError(f"Stage failed; remaining commands not started. See {log}")


def prediction_path(work, method, cell, seed):
    suffix = Path(cell) / f"seed_{seed}"
    if method == "S1Pert": return work / "runs/main" / suffix / "test_predictions.npz"
    if method in CLASSICAL: return work / "baselines/classical/runs" / CLASSICAL[method] / suffix / "test_predictions.npz"
    if method in ("STATE", "SUMMER"): return work / "baselines" / method.lower() / "runs" / suffix / "test_predictions.npz"
    if method == "Qwen3-4B-LoRA": return work / "baselines/qwen3_lora" / suffix / "test_predictions.npz"
    if method == "P1": return work / "baselines/p1" / suffix / "test_predictions.npz"
    return work / "baselines/aligned/runs" / suffix / "test_predictions.npz"


def register(work, methods, allow_partial=False, output=None):
    import numpy as np
    from jev_a4.evaluation import load_prediction
    from scripts.import_predictions import convert
    destination = Path(output) if output else work / "paper/predictions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise FileExistsError(destination)
    rows = []; missing = []
    for method in methods:
        for cell in CELLS:
            for seed in SEEDS:
                records = work / "data" / cell / f"seed_{seed}/test/records.npy"
                pred = prediction_path(work, method, cell, seed)
                if not pred.exists() and method in CLASSICAL and (pred.parent / "report.json").exists():
                    report = json.loads((pred.parent / "report.json").read_text())
                    if report.get("status") == "PASS":
                        kw = {"hard": pred.parent / "test_predictions.npy"}
                        for field, name in (("probabilities", "test_probabilities.npy"), ("effect", "test_effect.npy")):
                            if (pred.parent / name).exists(): kw[field] = pred.parent / name
                        convert(records, pred.parent / "test_keys.npy", pred, **kw)
                if not pred.exists() or not records.exists():
                    missing.append((method, cell, seed)); continue
                load_prediction(np.load(records, mmap_mode="r", allow_pickle=False), pred)
                rows.append(dict(method=method, cell=cell, seed=seed,
                                 records=os.path.relpath(records, destination.parent),
                                 predictions=os.path.relpath(pred, destination.parent)))
    if missing and not allow_partial:
        raise FileNotFoundError(f"Missing {len(missing)} method/splits; no complete table claimed. Example: {missing[:5]}")
    if not rows: raise ValueError("No valid predictions found")
    with destination.open("x", newline="") as f:
        writer = csv.DictWriter(f, ["method", "cell", "seed", "records", "predictions"]); writer.writeheader(); writer.writerows(rows)
    write_json(destination.with_suffix(".coverage.json"), dict(available=len(rows), expected=len(methods)*20,
               missing=missing, complete=not missing, test_pair_order_verified=True))
    return destination


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("init", "download", "data", "prepare", "encode", "main", "baselines", "register"))
    p.add_argument("--work-dir", type=Path); p.add_argument("--data-root", type=Path)
    p.add_argument("--gpu", default="0"); p.add_argument("--python", help="Interpreter for this method/environment")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=["S1Pert"])
    p.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    p.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    p.add_argument("--execute", action="store_true"); p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--out", type=Path); a = p.parse_args(argv); work = workspace(a.work_dir)
    if a.action == "init": return initialize(work, a.data_root, a.gpu)
    if a.action == "register": return print(register(work, a.methods, a.allow_partial, a.out))
    if a.action in ("main", "encode") and (tuple(a.cells) != CELLS or tuple(a.seeds) != SEEDS):
        p.error("Main/encode run the cells and seeds in configs/a4.json; edit that config explicitly")
    if a.action == "baselines" and any(m in a.methods for m in ("S1Pert",)):
        p.error("Use action main for S1Pert; select baseline --methods explicitly")
    cmds = plan(work, a.action, a.methods, a.cells, a.seeds, a.python)
    print(json.dumps({"commands": cmds, "execute": a.execute}, indent=2))
    if a.execute: execute(cmds, work, a.action)


if __name__ == "__main__": main()
