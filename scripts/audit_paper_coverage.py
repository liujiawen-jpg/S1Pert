#!/usr/bin/env python3
"""Audit the seven paper stages' source producers and consumers, without running them.

Only Python's standard library is used. No modules under audit are imported;
no network, models, training, cache generation or result plotting is started.
SOURCE_COVERAGE_PASS means declared source files, executable definitions and
input-reading call sites exist and parse. It is NOT scientific reproduction,
dependency installation, GPU validation, numerical equivalence or metric PASS.
Optional --work-dir inventories generated filenames only, without loading data.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def unit(path, symbols=(), calls=()):
    return {"path": path, "symbols": list(symbols), "calls": list(calls)}


# This is a source/dataflow map, not a stored experimental result. Numerical
# performance values are never part of the audit or its pass condition.
STAGES = [
    {
        "stage": 1, "name": "Raw preprocessing, adjudication, splits and Table III",
        "paper": ["Algorithm 1", "Table I", "Table III"], "depends_on": [],
        "consumes": ["external:raw_h5ad", "external:public_gene_functions"],
        "produces": ["normalized_moments", "strict_labels", "target_splits", "crossfit_evidence", "table3"],
        "upstream": [
            unit("src/jev_a4/data/preprocess.py", ["build", "process_nadig", "process_replogle"]),
            unit("src/jev_a4/data/nadig_moments.py", ["aggregate_moments"]),
            unit("src/jev_a4/labels/label_core.py", ["adjust", "scaled_welch", "label_pair"]),
            unit("src/jev_a4/labels/build_resolution_bundle.py", ["process_dataset", "build"]),
            unit("src/jev_a4/core/prepare.py", ["prepare", "split_targets", "prior_arrays"]),
            unit("src/jev_a4/knowledge.py"), unit("experiments/paper.py", ["initialize", "plan"]),
        ],
        "downstream": [unit("data_prep/table3.py", ["generate", "summarize_arrays"], ["load_array"]),
                       unit("plotting/make_table3_stats.py", ["main", "render"], ["generate"])],
        "commands": ["python -m experiments.paper data --work-dir WORK --execute",
                     "python -m experiments.paper prepare --work-dir WORK --execute",
                     "python -m experiments.run_section table3 --work-dir WORK --execute"],
        "artifact_patterns": ["preprocessed/*/manifest.json", "strict_labels/*/hard_label.npy",
                              "data/*/preparation_report.json", "paper/table3/table_dataset_summary.csv"],
    },
    {
        "stage": 2, "name": "Frozen multilevel and joint landmark caching",
        "paper": ["Section III-D", "Equations 4-8"], "depends_on": [1],
        "consumes": ["target_splits", "crossfit_evidence", "external:decider_4b_snapshot"],
        "produces": ["shared_landmarks", "joint_present_absent", "evidence_cards"],
        "upstream": [
            unit("src/jev_a4/core/encode.py", ["Session", "compress16", "compress4", "finite_cards", "encode_dataset", "encode_split"]),
            unit("src/jev_a4/core/vendor/shared_core.py", ["split_forward"]),
            unit("src/jev_a4/core/vendor/encode_joint_last8_cache.py", ["pool_contextual_landmarks", "_run_tuple_batch"]),
            unit("src/jev_a4/core/vendor/native_engine.py", ["_load_model"]),
        ],
        "downstream": [unit("src/jev_a4/core/train.py", ["SharedCache", "Partition"], ["load"])],
        "commands": ["python -m experiments.paper encode --work-dir WORK --execute"],
        "artifact_patterns": ["cache/*/manifest.json", "cache/*/seed_*/manifest.json",
                              "cache/*/seed_*/test/joint_present.npy", "cache/*/seed_*/test/joint_absent.npy"],
    },
    {
        "stage": 3, "name": "S1Pert and eight compared baselines, Table II",
        "paper": ["Section IV-B", "Table II"], "depends_on": [1, 2],
        "consumes": ["strict_labels", "target_splits", "crossfit_evidence", "shared_landmarks", "joint_present_absent",
                     "external:baseline_backbones_and_sources", "external:public_graph_snapshots"],
        "produces": ["selected_models", "heldout_predictions", "prediction_registry", "table2"],
        "methods": ["S1Pert", "GenePT-Gene", "GenePT-Prot", "Qwen3-4B-LoRA", "GEARS", "scGPT", "STATE", "ALIGNED", "SUMMER"],
        "upstream": [
            unit("src/jev_a4/core/pipeline.py", ["Workflow", "validate_config"]),
            unit("src/jev_a4/core/train.py", ["train", "restore_check", "evaluate_test"]),
            unit("src/jev_a4/core/train_memory_safe.py", ["main"]),
            unit("src/jev_a4/core/vendor/prior_model.py", ["make_prior_model"]),
            unit("baselines/run.py"), unit("baselines/fetch_external.py", ["fetch_repo"]),
            unit("baselines/classical/prepare.py"),
            unit("baselines/classical/train.py", ["fit_genept", "fit_epochs", "restore"]),
            unit("baselines/classical/models.py", ["GenePTClassifier", "gears_model", "scgpt_model"]),
            unit("baselines/language/qwen3_lora.py", ["LoRAQA", "run", "evaluate"]),
            unit("baselines/language/build_graph.py", ["build"]),
            unit("baselines/state/run.py", ["train", "evaluate"]), unit("baselines/state/native.py"),
            unit("baselines/aligned/run.py", ["train", "evaluate"]),
            unit("baselines/aligned/aligned_data.py", ["build_public_graph"]),
            unit("baselines/aligned/aligned_engine.py", ["AlignedEngine"]),
            unit("baselines/aligned/aligned_runner.py", ["train", "evaluate"]),
            unit("baselines/aligned/official_source/aligned/abl.py"),
            unit("baselines/aligned/official_source/aligned/reasoner.py"),
            unit("baselines/aligned/official_source/aligned/learner_adap.py"),
            unit("data_prep/build_public_knowledge.py", ["build"]),
            unit("baselines/summer/code/fast_summer.py", ["FastEngine", "pipeline", "evaluate_full"]),
            unit("baselines/prepare_public_edges.py", ["build"]),
            unit("baselines/prepare_esm2.py", ["build"]),
            unit("experiments/main_table2.py", ["main"]),
        ],
        "downstream": [unit("experiments/paper.py", ["register"], ["load_prediction"]),
                       unit("src/jev_a4/evaluation.py", ["load_prediction", "evaluate", "summarize"], ["load"]),
                       unit("plotting/make_table2_main.py", ["build"], ["DictReader", "evaluate"])],
        "commands": ["python -m experiments.main_table2 --work-dir WORK --execute"],
        "artifact_patterns": ["runs/main/*/seed_*/test_predictions.npz", "baselines/*/runs/**/test_predictions.npz",
                              "baselines/qwen3_lora/*/seed_*/test_predictions.npz", "paper/predictions.csv",
                              "paper/table2/table_II_per_split.csv", "paper/table2/table_II.tex"],
    },
    {
        "stage": 4, "name": "Retrained component ablations and target permutation",
        "paper": ["Section IV-C", "Table IV"], "depends_on": [1, 2, 3],
        "consumes": ["target_splits", "crossfit_evidence", "shared_landmarks", "joint_present_absent", "selected_models"],
        "produces": ["ablation_predictions", "permutation_predictions", "table4"],
        "variants": ["NO_JEV", "NO_JOINT", "FLAT", "NO_RANK", "NO_MASK", "TARGET_PERMUTATION"],
        "upstream": [unit("experiments/ablation.py", ["variant_config", "permutation", "permute_checkpoint", "main"]),
                     unit("experiments/ablation_runtime/model.py"), unit("experiments/ablation_runtime/trainer.py"),
                     unit("experiments/ablation_runtime/contract.py")],
        "downstream": [unit("plotting/make_table4_ablation.py", ["collect", "summarize", "render_table"], ["load", "evaluate"])],
        "commands": ["python -m experiments.run_section table4 --work-dir WORK --execute"],
        "artifact_patterns": ["paper/ablation/*/*/seed_*/test_predictions.npz", "paper/table4/table4.csv"],
    },
    {
        "stage": 5, "name": "Cross-target ranking and prior-stress evaluation",
        "paper": ["Section IV-D", "Fig. 1"], "depends_on": [1, 3],
        "consumes": ["target_splits", "crossfit_evidence", "heldout_predictions", "prediction_registry"],
        "produces": ["standalone_p1", "prior_stress_summary", "fig1"],
        "upstream": [unit("scripts/export_prior.py", ["export"]),
                     unit("experiments/prior_stress.py", ["train_prior", "masks", "run"]),
                     unit("src/jev_a4/evaluation.py", ["within_gene", "binary_auc"])],
        "downstream": [unit("plotting/plot_fig1_prior_stress.py", ["render"], ["read_csv", "save_figure"])],
        "commands": ["python -m experiments.run_section fig1 --work-dir WORK --execute"],
        "artifact_patterns": ["paper/prior_stress/per_split.csv", "paper/prior_stress/summary.csv",
                              "paper/figures/fig_prior_stress.pdf"],
    },
    {
        "stage": 6, "name": "Disjoint cell pools and zero-retraining boundary evaluation",
        "paper": ["Section IV-E", "Fig. 2"], "depends_on": [1, 2, 3],
        "consumes": ["external:raw_h5ad", "normalized_moments", "strict_labels", "target_splits",
                     "crossfit_evidence", "shared_landmarks", "selected_models"],
        "produces": ["pool_a_b_labels", "cell_pool_fidelity", "boundary_predictions", "fig2"],
        "upstream": [unit("data_prep/cell_pool.py", ["assign_cell_split", "split_statistics", "build", "prepare", "fidelity"]),
                     unit("experiments/cell_pool.py", ["FrozenBoundaryPredictor", "predict", "evaluate"])],
        "downstream": [unit("plotting/plot_fig2_label_boundary.py", ["load_inputs", "plot"], ["read_csv", "savefig"])],
        "commands": ["python -m experiments.run_section fig2 --work-dir WORK --execute"],
        "artifact_patterns": ["cell_pool/predictions/seed_*/probabilities.npy", "paper/fig2/boundary_per_seed.csv",
                              "paper/fig2/fig2_label_boundary.pdf"],
    },
    {
        "stage": 7, "name": "Measured online and fresh end-to-end runtime",
        "paper": ["Section IV-F", "Fig. 3"], "depends_on": [1, 2, 3],
        "consumes": ["target_splits", "crossfit_evidence", "shared_landmarks", "selected_models",
                     "external:baseline_backbones_and_sources"],
        "produces": ["measured_timings", "fig3"],
        "upstream": [unit("experiments/runtime_recipes.py", ["build"]),
                     unit("experiments/runtime.py", ["measure_phase", "run"], ["perf_counter"]),
                     unit("experiments/runtime_worker.py", ["execute"], ["measure_phase"]),
                     unit("experiments/runtime_factories.py", ["s1pert_online", "summer_online", "summer_offline"]),
                     unit("baselines/aligned/run.py", ["load_for_timing"])],
        "downstream": [unit("plotting/plot_fig3_runtime.py", ["render"], ["read_text", "save_figure"])],
        "commands": ["python -m experiments.run_section fig3 --runtime-mode fresh --work-dir WORK --execute"],
        "artifact_patterns": ["paper/runtime_fresh/timings.json", "paper/figures/fig_runtime_efficiency.pdf"],
    },
]

SHARED_FILES = ["experiments/paper.py", "experiments/run_section.py", "src/jev_a4/config.py",
                "scripts/download_assets.py", "configs/downloads.json", "baselines/adapter_inventory.json",
                "baselines/language/graph_sources.json", "data_prep/public_knowledge/sources.json"]


def _stub(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    return not body or all(isinstance(item, ast.Pass) or
        (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and item.value.value is Ellipsis)
        for item in body)


def inspect_unit(root, specification):
    relative = Path(specification["path"])
    result = {**specification, "status": "SOURCE_PRESENT", "issues": []}
    if relative.is_absolute() or ".." in relative.parts:
        result["issues"].append("unsafe repository-relative source path")
    else:
        path = root / relative
        if not path.is_file():
            result["issues"].append("source file missing")
        elif not path.resolve().is_relative_to(root):
            result["issues"].append("source resolves outside repository")
        else:
            payload = path.read_bytes()
            result.update(sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
            try:
                if path.suffix == ".py":
                    tree = ast.parse(payload.decode("utf-8"), filename=str(relative))
                    definitions = {node.name: node for node in ast.walk(tree)
                                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                    for name in specification.get("symbols", []):
                        if name not in definitions:
                            result["issues"].append("required definition missing: " + name)
                        elif _stub(definitions[name]):
                            result["issues"].append("required definition is a stub: " + name)
                    calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                             for node in ast.walk(tree) if isinstance(node, ast.Call)
                             and isinstance(node.func, (ast.Name, ast.Attribute))}
                    for name in specification.get("calls", []):
                        if name not in calls:
                            result["issues"].append("required input/measurement call missing: " + name)
                elif path.suffix == ".json":
                    json.loads(payload)
            except (SyntaxError, UnicodeDecodeError, ValueError) as error:
                result["issues"].append("unparseable source: " + str(error))
    if result["issues"]:
        result["status"] = "SOURCE_MISSING_OR_INVALID"
    return result


def audit(repo=REPO, work_dir=None, stages=None, shared_files=None):
    root = Path(repo).resolve()
    definitions = copy.deepcopy(STAGES if stages is None else stages)
    expected_ids = list(range(1, 8))
    issues = []
    ids = [stage["stage"] for stage in definitions]
    if ids != expected_ids:
        issues.append("exactly seven ordered paper stages are required")
    producers = {}
    for stage in definitions:
        for artifact in stage["produces"]:
            if artifact in producers:
                issues.append("duplicate logical artifact producer: " + artifact)
            producers[artifact] = stage["stage"]
    checked = []
    for stage in definitions:
        row = copy.deepcopy(stage)
        stage_issues = []
        for dependency in stage["depends_on"]:
            if dependency not in ids or dependency >= stage["stage"]:
                stage_issues.append(f"invalid upstream stage: {dependency}")
        for artifact in stage["consumes"]:
            if artifact.startswith("external:"):
                continue
            producer = producers.get(artifact)
            if producer not in stage["depends_on"] and producer != stage["stage"]:
                stage_issues.append("missing declared upstream producer: " + artifact)
        for direction in ("upstream", "downstream"):
            row[direction] = [inspect_unit(root, spec) for spec in stage[direction]]
            if not row[direction]:
                stage_issues.append(direction + " source mapping is empty")
            stage_issues.extend(item["path"] + ": " + error for item in row[direction] for error in item["issues"])
        row["issues"] = stage_issues
        row["status"] = "SOURCE_COVERED" if not stage_issues else "SOURCE_GAP"
        row["scientific_execution_verified"] = False
        if work_dir is not None:
            work = Path(work_dir).expanduser().resolve()
            row["artifact_presence_only"] = [
                {"pattern": pattern, "files_present": sum(path.is_file() for path in work.glob(pattern)),
                 "contents_validated": False} for pattern in stage["artifact_patterns"]]
        checked.append(row)
        issues.extend(f"stage {stage['stage']}: {error}" for error in stage_issues)
    shared = [inspect_unit(root, unit(path)) for path in (SHARED_FILES if shared_files is None else shared_files)]
    issues.extend(item["path"] + ": " + error for item in shared for error in item["issues"])
    return {
        "schema": "s1pert.paper-source-coverage.v1",
        "status": "SOURCE_COVERAGE_PASS" if not issues else "SOURCE_COVERAGE_FAIL",
        "scope": "Static source definitions, declared producer/consumer links, input-reading/measurement call sites and file hashes",
        "scientific_reproduction_verified": False,
        "training_or_inference_started": False, "network_accessed": False,
        "external_dependencies_checked": False, "numerical_results_checked": False,
        "no_hardcoded_results_proven": False,
        "limits": ["Static presence is not numerical or scientific correctness.",
                   "Required read-call checks catch simple plotting stubs; they do not prove complete data lineage.",
                   "Source snapshot availability, software installation, hardware and full runs are separate checks.",
                   "Optional artifact counts are presence-only; no count is interpreted as an experiment PASS."],
        "stages_expected": len(expected_ids), "stages_source_covered": sum(row["status"] == "SOURCE_COVERED" for row in checked),
        "stages": checked, "shared_sources": shared, "issues": issues,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--work-dir", type=Path, default=os.environ.get("S1PERT_ROOT"))
    parser.add_argument("--output", type=Path, help="write JSON once; existing output is never replaced")
    args = parser.parse_args(argv)
    report = audit(args.repo_root, args.work_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "SOURCE_COVERAGE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
