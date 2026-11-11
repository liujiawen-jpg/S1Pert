"""Portable paper orchestration and raw-prediction statistics; CPU/offline only."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments import paper
from jev_a4.config import load_config
from plotting import make_table2_main as table2


def run_script(path, args, cwd):
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return subprocess.run([sys.executable, str(ROOT / path), *map(str, args)],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=30)


def test_initialized_workspace_moves_without_stale_internal_paths(tmp_path):
    old = tmp_path / 'original'
    paper.initialize(old, gpu='3')
    new = tmp_path / 'deeper/moved'
    new.parent.mkdir()
    shutil.move(old, new)
    cfg = load_config(new / 'configs/a4.json')
    assert Path(cfg['project_root']) == new
    assert Path(cfg['data_preprocessed_root']) == new / 'preprocessed'
    assert Path(cfg['knowledge_db']) == new / 'assets/public_functions.sqlite'
    assert cfg['gpu_uuid'] == '3'
    pp = load_config(new / 'configs/preprocess.json')
    for spec in pp['preprocess']['datasets'].values():
        assert Path(spec['raw']).is_relative_to(new / 'raw')
    assert not (new / 'runs').exists()
    with pytest.raises(FileExistsError):
        paper.initialize(new)


@pytest.mark.parametrize('family', ['classical', 'state', 'summer'])
def test_staging_after_workspace_move_rebinds_package_metrics(tmp_path, family):
    old = tmp_path / 'initial'
    paper.initialize(old)
    moved = tmp_path / 'nested/relocated'
    moved.parent.mkdir()
    shutil.move(old, moved)
    script = ROOT / 'baselines/run.py'
    spec = importlib.util.spec_from_file_location('workspace_baseline_stager', script)
    stager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stager)
    work = moved / 'baselines' / family
    stager.init(family, moved / 'configs' / (family + '.json'), work)
    staged = work / 'code' if family == 'summer' else work
    cfg = json.loads((staged / 'config.json').read_text())
    metric_key = 'metrics' if family == 'classical' else 'a4_metrics'
    assert Path(cfg[metric_key]) == ROOT / 'src/jev_a4/core/metrics.py'
    assert Path(cfg['a4_root' if family == 'summer' else 'a4_data']) == (moved if family == 'summer' else moved / 'data')
    assert cfg['python'] == sys.executable


@pytest.mark.parametrize('script', [
    'experiments/paper.py', 'plotting/make_table2_main.py', 'experiments/ablation.py',
    'experiments/ablation_runtime/trainer.py', 'plotting/make_table4_ablation.py',
])
def test_new_direct_script_help_needs_no_install_or_working_directory(script, tmp_path):
    result = run_script(script, ['--help'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout
    assert not list(tmp_path.iterdir())


def test_planned_core_commands_match_actual_cli_arguments(tmp_path):
    paper.initialize(tmp_path / 'workspace')
    work = tmp_path / 'workspace'
    for action in ('prepare', 'encode', 'main'):
        commands = paper.plan(work, action, ['S1Pert'])
        assert commands
        for command in commands:
            relative = str(Path(command[1]).relative_to(ROOT))
            # Executing only parser help cannot open the configured datasets.
            env = os.environ.copy()
            env['PYTHONPATH'] = str(ROOT / 'src')
            env['CUDA_VISIBLE_DEVICES'] = ''
            result = subprocess.run([*command, '--help'], cwd=tmp_path, env=env,
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, relative + ': ' + result.stderr
            for token in command[2:]:
                if token.startswith('--'):
                    assert token in result.stdout, (relative, token)
    assert not (work / 'data').exists()
    assert not (work / 'runs').exists()


def test_baseline_plan_arguments(tmp_path):
    paper.initialize(tmp_path / 'workspace')
    work = tmp_path / 'workspace'
    methods = ['GenePT-Gene', 'GenePT-Prot', 'GEARS', 'scGPT', 'STATE', 'SUMMER', 'Qwen3-4B-LoRA', 'P1']
    commands = paper.plan(work, 'baselines', methods)
    assert any('native_graphs' in command for command in commands)
    assert any('qwen3_lora.py' in ' '.join(command) for command in commands)
    assert any('export_prior.py' in ' '.join(command) for command in commands)
    for command in commands:
        assert command[0] == sys.executable
        assert Path(command[1]).is_file()
        if 'train.py' in command:
            assert '--out' in command and '--method' in command
    assert not (work / 'baselines').exists()


def test_configured_queue_methods_reject_ignored_subset_overrides(tmp_path):
    work = tmp_path / 'workspace'
    paper.initialize(work)
    for method in ('STATE', 'SUMMER'):
        with pytest.raises(ValueError, match='full matrix'):
            paper.plan(work, 'baselines', [method], ['hepg2'], [17])


def fake_predictions(work, cells=paper.CELLS, seeds=paper.SEEDS):
    patterns = [[0, 1, 2, 0, 1, 2], [1] * 6, [0] * 6,
                [1, 2, 0, 1, 2, 0], [1, 1, 2, 0, 1, 2]]
    for cell in cells:
        for seed in seeds:
            directory = work / 'data' / cell / f'seed_{seed}/test'
            directory.mkdir(parents=True)
            rows = np.array([(i, 0, i % 3) for i in range(6)],
                            dtype=[('target', 'i4'), ('query', 'i4'), ('label', 'i1')])
            np.save(directory / 'records.npy', rows)
            path = paper.prediction_path(work, 'S1Pert', cell, seed)
            path.parent.mkdir(parents=True)
            pred = np.asarray(patterns[paper.SEEDS.index(seed)], dtype=np.int8)
            np.savez(path, target=rows['target'], query=rows['query'], labels=rows['label'], y_pred=pred)


def test_registration_paths_relocate_and_preserve_exact_pair_alignment(tmp_path):
    work = tmp_path / 'work'
    fake_predictions(work)
    registry = paper.register(work, ['S1Pert'])
    with registry.open() as stream:
        entries = list(csv.DictReader(stream))
    assert len(entries) == 20
    assert all(not Path(row['predictions']).is_absolute() for row in entries)
    assert all(not Path(row['records']).is_absolute() for row in entries)
    with pytest.raises(FileExistsError):
        paper.register(work, ['S1Pert'])
    moved = tmp_path / 'moved'
    shutil.move(work, moved)
    for row in entries:
        assert (moved / 'paper' / row['records']).is_file()
        assert (moved / 'paper' / row['predictions']).is_file()


def test_registration_missing_guard_and_direct_entry_point(tmp_path):
    work = tmp_path / 'work'
    fake_predictions(work, ['hepg2'], [17])
    with pytest.raises(FileNotFoundError, match='Missing'):
        paper.register(work, ['S1Pert'])
    assert not (work / 'paper/predictions.csv').exists()
    result = run_script('experiments/paper.py', ['register', '--work-dir', work,
                        '--methods', 'S1Pert', '--allow-partial'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    coverage = json.loads((work / 'paper/predictions.coverage.json').read_text())
    assert coverage['available'] == 1 and coverage['expected'] == 20
    assert not coverage['complete']


def test_table_two_recomputes_raw_metrics_and_five_seed_sample_sd(tmp_path):
    work = tmp_path / 'work'
    fake_predictions(work)
    registry = paper.register(work, ['S1Pert'])
    rows, summary = table2.build(registry, work / 'table2', methods=['S1Pert'])
    expected = np.asarray([1., 1/6, 1/6, 0., 37/45])
    actual = [row['macro_f1'] for row in rows if row['cell'] == 'hepg2']
    np.testing.assert_allclose(actual, expected)
    aggregate = next(row for row in summary if row['cell'] == 'hepg2' and row['metric'] == 'macro_f1')
    assert aggregate['valid_splits'] == 5
    assert aggregate['mean'] == pytest.approx(expected.mean())
    assert aggregate['sample_sd'] == pytest.approx(expected.std(ddof=1))
    assert (work / 'table2/table_II.tex').is_file()
    with pytest.raises(FileExistsError):
        table2.build(registry, work / 'table2', methods=['S1Pert'])


def test_table_two_rejects_missing_split_and_missing_entire_method(tmp_path):
    work = tmp_path / 'work'
    fake_predictions(work, ['hepg2'], [17])
    partial = paper.register(work, ['S1Pert'], allow_partial=True)
    with pytest.raises(ValueError, match='incomplete'):
        table2.build(partial, work / 'missing_splits', methods=['S1Pert'])
    full = tmp_path / 'full'
    fake_predictions(full)
    registry = paper.register(full, ['S1Pert'])
    # Default Table II requires every paper method, not only those that happen
    # to have entries. A deliberately selected subset is a separate workflow.
    with pytest.raises(ValueError, match='incomplete'):
        table2.build(registry, full / 'missing_methods')


def test_registration_rejects_reordered_prediction_pairs(tmp_path):
    work = tmp_path / 'work'
    fake_predictions(work, ['hepg2'], [17])
    path = paper.prediction_path(work, 'S1Pert', 'hepg2', 17)
    with np.load(path) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    values['target'] = values['target'][::-1]
    np.savez(path, **values)
    with pytest.raises(ValueError, match='pair order'):
        paper.register(work, ['S1Pert'], allow_partial=True)
