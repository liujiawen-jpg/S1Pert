"""Portable baseline staging and known metrics, without GPUs or external assets."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from jev_a4.evaluation import binary_auc, evaluate

ROOT = Path(__file__).resolve().parents[1]


def run_command(arguments, cwd):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
    return subprocess.run([sys.executable, *map(str, arguments)], cwd=cwd,
                          env=env, capture_output=True, text=True, timeout=30)


def initialize(family, directory):
    work = directory / family
    result = run_command([
        ROOT / "baselines/run.py", "init", "--family", family,
        "--config", ROOT / f"baselines/{family}/config.example.json",
        "--work", work,
    ], directory)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["models_started"] is False
    return work, work / "code" if family == "summer" else work


@pytest.mark.parametrize("family", ["classical", "state", "summer"])
def test_staged_helpers_import_and_paths_resolve(family, tmp_path):
    work, staged = initialize(family, tmp_path)
    result = run_command([
        "-c", "import common; import sys; assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; assert 'vllm' not in sys.modules",
    ], staged)
    assert result.returncode == 0, result.stderr
    cfg = json.loads((staged / "config.json").read_text())
    data_key = "a4_root" if family == "summer" else "a4_data"
    expected = ROOT / "workspace/a4" if family == "summer" else ROOT / "workspace/a4/data"
    assert Path(cfg[data_key]) == expected.resolve()
    assert cfg["python"] == sys.executable
    assert not list((work / "runs").iterdir())
    assert not list((work / "prepared").iterdir())


@pytest.mark.parametrize("family,script", [
    ("classical", "prepare.py"), ("state", "run.py"), ("summer", "fast_summer.py"),
])
def test_execution_plan_does_not_open_assets_or_launch(family, script, tmp_path):
    work, staged = initialize(family, tmp_path)
    result = run_command([
        ROOT / "baselines/run.py", "execute", "--work", work, "--plan", "--", script, "--help",
    ], tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [sys.executable, str(staged / script), "--help"]
    assert not (work / "reports/launch_seal.json").exists()
    assert not list((work / "runs").iterdir())


@pytest.mark.parametrize("family,script", [
    ("classical", "prepare.py"), ("summer", "fast_summer.py"),
])
def test_lightweight_stage_help_outside_workspace(family, script, tmp_path):
    work, staged = initialize(family, tmp_path)
    result = run_command([staged / script, "--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not list((work / "runs").iterdir())


def test_mixed_invalid_predictions_have_known_metric_denominators(tmp_path):
    rows = np.array([(i, 0, i % 3) for i in range(6)],
                    dtype=[("target", "i4"), ("query", "i4"), ("label", "i1")])
    path = tmp_path / "predictions.npz"
    np.savez(path, target=rows["target"], query=rows["query"], labels=rows["label"],
             y_pred=np.array([0, 1, 2, -1, 0, 1], dtype="i1"))
    result = evaluate(rows, path)
    assert result["macro_f1"] == pytest.approx(5 / 9)
    assert result["balanced_accuracy"] == result["accuracy"] == 0.5
    assert result["invalid_rate"] == pytest.approx(1 / 6)
    assert result["within_gene_change_auc"] is None
    assert result["within_gene_polarity_auc"] is None
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5
    assert binary_auc([0, 1], [0.0, 1.0]) == 1.0
