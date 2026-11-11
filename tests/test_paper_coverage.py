"""Source-coverage audit tests: no model imports, data downloads or training."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

from scripts.audit_paper_coverage import REPO, STAGES, audit, inspect_unit, unit


def test_all_seven_source_stages_are_mapped_not_experiment_passes():
    report = audit()
    assert report["status"] == "SOURCE_COVERAGE_PASS", report["issues"]
    assert report["stages_source_covered"] == 7
    assert [row["stage"] for row in report["stages"]] == list(range(1, 8))
    for key in ("scientific_reproduction_verified", "training_or_inference_started",
                "network_accessed", "external_dependencies_checked", "numerical_results_checked",
                "no_hardcoded_results_proven"):
        assert report[key] is False
    assert all(row["upstream"] and row["downstream"] for row in report["stages"])
    assert all(not row["scientific_execution_verified"] for row in report["stages"])
    assert all(len(item["sha256"]) == 64 for row in report["stages"]
               for direction in ("upstream", "downstream") for item in row[direction])


def test_missing_upstream_fails_even_when_plotting_source_exists():
    stages = copy.deepcopy(STAGES)
    stages[3]["upstream"][0]["path"] = "missing/ablation.py"
    report = audit(stages=stages)
    assert report["status"] == "SOURCE_COVERAGE_FAIL"
    assert report["stages"][3]["status"] == "SOURCE_GAP"
    assert all(item["status"] == "SOURCE_PRESENT" for item in report["stages"][3]["downstream"])
    assert any("missing/ablation.py" in error for error in report["issues"])


def test_missing_stage_and_broken_producer_link_fail():
    missing = audit(stages=STAGES[:-1])
    assert missing["status"] == "SOURCE_COVERAGE_FAIL"
    assert "exactly seven ordered paper stages are required" in missing["issues"]
    stages = copy.deepcopy(STAGES)
    stages[4]["consumes"].append("unimplemented_prediction_producer")
    broken = audit(stages=stages)
    assert broken["status"] == "SOURCE_COVERAGE_FAIL"
    assert any("missing declared upstream producer" in error for error in broken["issues"])


def test_stub_syntax_missing_reader_and_path_escape_fail(tmp_path):
    path = tmp_path / "module.py"
    path.write_text('def run():\n    """Placeholder."""\n    pass\n')
    assert "is a stub" in inspect_unit(tmp_path, unit("module.py", ["run"]))["issues"][0]
    path.write_text("def run():\n    return [0.1, 0.2]\n")
    assert "input/measurement call missing" in inspect_unit(tmp_path, unit("module.py", ["run"], ["read_csv"]))["issues"][0]
    path.write_text("def run(:\n")
    assert "unparseable source" in inspect_unit(tmp_path, unit("module.py"))["issues"][0]
    assert inspect_unit(tmp_path, unit("../outside.py"))["status"] == "SOURCE_MISSING_OR_INVALID"


def test_artifact_presence_never_implies_result_validation(tmp_path):
    output = tmp_path / "paper/table2/table_II.tex"
    output.parent.mkdir(parents=True)
    output.write_text("not an evaluated result")
    report = audit(work_dir=tmp_path)
    counts = report["stages"][2]["artifact_presence_only"]
    assert next(row for row in counts if row["pattern"].endswith("table_II.tex"))["files_present"] == 1
    assert all(row["contents_validated"] is False for row in counts)
    assert report["scientific_reproduction_verified"] is False


def test_methods_map_matches_declared_execution_registry():
    tree = ast.parse((REPO / "experiments/paper.py").read_text())
    declaration = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "METHODS" for target in node.targets))
    assert set(STAGES[2]["methods"]) == set(ast.literal_eval(declaration)) - {"P1"}
    assert {name for stage in STAGES for name in stage["paper"]} >= {
        "Table II", "Table III", "Table IV", "Fig. 1", "Fig. 2", "Fig. 3"}


def test_cli_standalone_and_exclusive_report_write(tmp_path):
    script = REPO / "scripts/audit_paper_coverage.py"
    output = tmp_path / "audit.json"
    command = [sys.executable, str(script), "--output", str(output)]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report["status"] == "SOURCE_COVERAGE_PASS"
    assert json.loads(output.read_text()) == report
    original = output.read_bytes()
    repeated = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert repeated.returncode != 0 and output.read_bytes() == original
