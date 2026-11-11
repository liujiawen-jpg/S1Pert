"""Artificial CPU ALIGNED preparation, both rounds, independent restore/export."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from baselines.aligned import aligned_data as adapter

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def fixture_workspace(root):
    data = root / 'data/hepg2'
    split = data / 'seed_17'
    split.mkdir(parents=True)
    genes, symbols = [f'g{i}' for i in range(6)], list('ABCDEF')
    meta = {'target_ids': genes, 'query_ids': genes, 'target_symbols': symbols, 'query_symbols': symbols}
    write_json(data / 'dataset.json', meta)
    partition = np.array([0, 0, 0, 1, 2, 2], dtype=np.int8)
    np.save(split / 'target_partition.npy', partition)
    dtype = [('target', '<i4'), ('query', '<i4'), ('label', 'i1')]
    for name, code in (('train', 0), ('val', 1), ('test', 2)):
        directory = split / name
        directory.mkdir()
        records = np.array([(t, q, (t+q) % 3) for t in np.flatnonzero(partition == code)
                            for q in range(6) if q != t], dtype=dtype)
        np.save(directory / 'records.npy', records)
    array_hash = hashlib.sha256(str(partition.dtype).encode()+json.dumps(partition.shape).encode()+partition.tobytes()).hexdigest()
    write_json(split / 'manifest.json', {'status': 'PASS', 'dataset': 'hepg2', 'seed': 17,
        'split_ratio': [7, 1, 2], 'exclude_self_pairs': True, 'target_partition_sha256': array_hash})
    write_json(data / 'preparation_report.json', {'status': 'PASS', 'exclude_self_pairs': True,
        'artifacts': {str(p.relative_to(data)): {'sha256': adapter.sha256(p)} for p in data.rglob('*') if p.is_file()}})
    assets = root / 'assets'
    assets.mkdir()
    with sqlite3.connect(assets / 'public_knowledge.sqlite') as con:
        con.execute('CREATE TABLE genes(accession TEXT,symbol TEXT)')
        con.executemany('INSERT INTO genes VALUES (?,?)', [(f'P{i}', s) for i, s in enumerate(symbols)])
        con.execute('CREATE TABLE public_edges(subject TEXT,object TEXT,sign TEXT,directed INTEGER,source TEXT)')
        con.executemany('INSERT INTO public_edges VALUES (?,?,?,?,?)',
            [('A', 'B', 'positive', 1, 'OmniPath-core'), ('B', 'C', 'positive', 1, 'OmniPath-core'),
             ('C', 'A', 'positive', 1, 'OmniPath-core'), ('D', 'E', 'positive', 1, 'OmniPath-core'),
             ('A', 'C', 'negative', 1, 'OmniPath-core'), ('B', 'A', 'negative', 1, 'OmniPath-core'),
             ('C', 'B', 'negative', 1, 'OmniPath-core'), ('E', 'D', 'negative', 1, 'OmniPath-core'),
             ('A', 'F', 'positive', 1, 'OmniPath-core'), ('A', 'F', 'negative', 1, 'OmniPath-core'),
             ('B', 'E', 'unknown', 0, 'STRING')])
        con.execute('CREATE TABLE empirical_events(secret TEXT)')
    with sqlite3.connect(assets / 'public_go.sqlite') as con:
        con.execute('CREATE TABLE terms(id TEXT)')
        con.execute('CREATE TABLE gene_annotations(gene TEXT,term TEXT,qualifiers TEXT)')
        con.executemany('INSERT INTO gene_annotations VALUES (?,?,?)',
            [('A', 'GO:0000001', '[]'), ('A', 'GO:0000001', '[]'), ('B', 'GO:0000002', '["NOT"]')])
    config = root / 'configs/aligned.json'
    write_json(config, {'profile': 'budgeted', 'weight_init_epochs': 1, 'pretrain_epochs': 1,
        'pretrain_reinforce_epochs': 1, 'retrain_epochs': 1, 'retrain_reinforce_epochs': 1,
        'refine_epochs': 1, 'cpu_threads': 1})
    return data, config


def test_official_hashes_and_original_engine_are_preserved():
    from baselines.aligned.aligned_runtime import fingerprint
    declaration = adapter.read_json(REPO / 'baselines/aligned/source_availability.json')
    actual = fingerprint()
    expected = declaration['expected_source_hashes_from_export_metadata']
    for name in ('aligned_engine.py', 'official_source/aligned/abl.py',
                 'official_source/aligned/reasoner.py', 'official_source/aligned/learner_adap.py'):
        assert actual[name] == expected[name]
    assert declaration['original_core_files_verified'] == 7
    assert (REPO / 'baselines/aligned/official_source/LICENSE').is_file()


def test_public_graph_preserves_conflict_and_not_rules(tmp_path):
    fixture_workspace(tmp_path)
    adapter.configure(tmp_path)
    dataset = adapter.load_dataset('hepg2', 17)
    assert 'records_test' not in dataset
    graph = adapter.build_public_graph(dataset['meta'], dataset['gene_ids'])
    assert graph['P'].nnz == 4 and graph['N'].nnz == 4
    assert graph['report']['conflicting_pairs_dropped'] == 1
    assert graph['GO_count'].tolist() == [1, 0, 0, 0, 0, 0]
    assert graph['report']['response_tables_read'] == []
    con = adapter.public_connection(tmp_path / 'assets/public_knowledge.sqlite', ['genes', 'public_edges'])
    with pytest.raises(sqlite3.DatabaseError):
        con.execute('SELECT * FROM empirical_events').fetchall()
    con.close()


def test_continuous_hybrid_score_retains_neural_softmax():
    import torch
    from baselines.aligned.aligned_runner import score_records

    class Toy(torch.nn.Module):
        def forward(self, x):
            native = torch.tensor([[[.1, .2, .7], [.7, .2, .1], [.2, .6, .2]]], device=x.device).repeat(len(x), 1, 1)
            r = torch.tensor([[.2, .8, .5]], device=x.device).repeat(len(x), 1)
            return native, r
    x = -np.eye(3, dtype=np.float32)[:1]
    kb = torch.zeros((3, 3))
    kb[0, 1] = 1
    result = score_records(Toy(), kb, x, np.array([0, 0, 0]), np.array([0, 1, 2]))
    np.testing.assert_allclose(result['probabilities'], [[.7, .2, .1], [0, 0, 1], [.2, .6, .2]])
    np.testing.assert_array_equal(result['prediction'], [0, 2, 1])
    assert not np.all((result['probabilities'] == 0) | (result['probabilities'] == 1))


def test_full_two_round_cpu_recipe_restore_export_and_timing(tmp_path):
    import torch
    from baselines.aligned import run
    from baselines.aligned import aligned_runner as runner
    fixture_workspace(tmp_path)
    result = run.train(tmp_path, 'hepg2', 17, device='cpu')
    output = tmp_path / 'baselines/aligned/runs/hepg2/seed_17'
    assert result['completed_stages'] == list(runner.STAGES)
    assert result['test_opened'] is False and result['selected_stage'] == 'refine1'
    assert not (output / 'test_access.json').exists()
    checkpoint = torch.load(output / 'final.pt', map_location='cpu', weights_only=False)
    assert checkpoint['engine']['refiner']['finalized']
    with pytest.raises(ValueError, match='new process'):
        run.evaluate(tmp_path, 'hepg2', 17, device='cpu')
    command = [sys.executable, '-m', 'baselines.aligned.run', 'evaluate', '--work-dir', str(tmp_path),
               '--cell', 'hepg2', '--seed', '17', '--device', 'cpu']
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src')+os.pathsep+str(REPO))
    completed = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = adapter.read_json(output / 'report.json')
    assert report['status'] == 'PASS' and report['restore']['pid'] != report['restore']['training_pid']
    with np.load(output / 'test_predictions.npz', allow_pickle=False) as result:
        assert len(result['target']) == 10
        np.testing.assert_array_equal(result['prediction'], result['probabilities'].argmax(1))
        np.testing.assert_allclose(result['probabilities'].sum(1), 1, atol=2e-6)
        expected = result['probabilities'].copy()
        assert {'probabilities', 'labels', 'neural_probs', 'R'} <= set(result.files)
    timing = run.load_for_timing(tmp_path, 'hepg2', 17, device='cpu')
    assert timing['n_pairs'] == 10 and not timing['metadata']['saved_predictions_replayed']
    np.testing.assert_allclose(timing['run'](), expected, rtol=0, atol=0)
    timing['cleanup']()
    assert run.evaluate(tmp_path, 'hepg2', 17, device='cpu')['status'] == 'PASS'


def test_prepared_seals_and_smoke_test_gate(tmp_path):
    from baselines.aligned import run
    from baselines.aligned.aligned_runtime import load_prepared
    fixture_workspace(tmp_path)
    adapter.configure(tmp_path)
    report = adapter.prepare('hepg2', 17)
    assert not report['test_opened']
    path = tmp_path / 'baselines/aligned/prepared/hepg2/seed_17/Y_train.npy'
    values = np.load(path)
    values[0, 1] = 1 if values[0, 1] == 0 else 0
    np.save(path, values)
    with pytest.raises(ValueError, match='artifact changed'):
        load_prepared('hepg2', 17)
