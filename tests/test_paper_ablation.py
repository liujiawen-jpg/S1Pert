"""Artificial CPU-only tests; no real profiles, pretrained models or downloads."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.ablation import (ARMS, SEEDS, permutation, permute_checkpoint,
                                  run_directory, variant_config, work_directory)
from plotting.make_table4_ablation import collect, render_table, summarize


@pytest.fixture
def tiny_inputs(tmp_path):
    rng = np.random.default_rng(47)
    cell = 'synthetic'
    data, cache = tmp_path / 'data' / cell, tmp_path / 'cache' / cell
    data.mkdir(parents=True)
    cache.mkdir(parents=True)
    hidden, nquery = 8, 4
    np.save(data / 'condition_numeric.npy', rng.normal(size=(10, 4)).astype('f4'))
    np.save(data / 'query_numeric.npy', rng.normal(size=(nquery, 4)).astype('f4'))
    np.save(cache / 'condition_final32.npy', rng.normal(size=(10, 16, hidden)).astype('f2'))
    np.save(cache / 'query_final32.npy', rng.normal(size=(nquery, 16, hidden)).astype('f2'))
    (cache / 'seed_17').mkdir()
    np.save(cache / 'seed_17/cards_final.npy', rng.normal(size=(nquery, 4, hidden)).astype('f2'))
    for part, targets in [('train', range(6)), ('val', range(6, 8)), ('test', range(8, 10))]:
        d, c = data / 'seed_17' / part, cache / 'seed_17' / part
        d.mkdir(parents=True)
        c.mkdir(parents=True)
        rows = np.empty(len(targets) * nquery, dtype=[('target', 'i4'), ('query', 'i4'), ('label', 'i1')])
        rows['target'] = np.repeat(list(targets), nquery)
        rows['query'] = np.tile(np.arange(nquery), len(targets))
        rows['label'] = (rows['target'] + rows['query']) % 3
        features = np.zeros((len(rows), 12), dtype='f4')
        features[:, :3], features[:, -1] = 1 / 3, 1
        np.save(d / 'records.npy', rows)
        np.save(d / 'prior_features.npy', features)
        np.save(c / 'card_ids.npy', rows['query'])
        np.save(c / 'joint_ids.npy', np.arange(len(rows)))
        np.save(c / 'joint_present.npy', rng.normal(size=(len(rows), 4, hidden)).astype('f2'))
        np.save(c / 'joint_absent.npy', rng.normal(size=(len(rows), 4, hidden)).astype('f2'))
    cfg = dict(smoke=True, device='cpu', cpu_threads=1, runtime_python=sys.executable,
               decoder_width=8, decoder_blocks=1, attention_heads=2, ffn_width=16,
               backbone_hidden_width=hidden, dropout=0.,
               training=dict(physical_batch=8, physical_batch_candidates=[8], effective_batch=8,
                             evaluation_batch=8, min_epochs=2, max_epochs=3, patience=1,
                             min_delta=2., pairs_per_update=2, restore_preview_rows=8,
                             host_available_min_gib=0))
    return tmp_path, data, cache, cfg


@pytest.mark.parametrize('arm', ARMS)
def test_exact_variant_flags_and_no_source_config_mutation(arm):
    source = {'training': {'lr': 1e-4}, 'device': 'cpu'}
    original = copy.deepcopy(source)
    cfg = variant_config(source, arm, Path('workspace'), 'synthetic')
    assert source == original
    assert cfg['joint_enabled'] == (arm not in ('NO_JEV', 'NO_JOINT'))
    assert cfg['numeric_only'] == (arm == 'NO_JEV')
    assert cfg['flat_output'] == (arm == 'FLAT')
    assert cfg['training']['pair_weight'] == (0 if arm == 'NO_RANK' else .02)
    assert cfg['training']['prior_dropout'] == (0 if arm == 'NO_MASK' else .10)
    assert cfg['training']['lr'] == source['training']['lr']
    assert not cfg['paper_ablation']['old_prediction_output_anchor']


def test_permutation_matches_historical_salt_and_preserves_evidence():
    rows = np.array([(i, j, (i+j) % 3) for i in range(8) for j in range(3)],
                    dtype=[('target', 'i4'), ('query', 'i4'), ('label', 'i1')])
    features = np.repeat(rows['query'][:, None], 12, axis=1)
    cards = rows['query'].copy()
    rng = np.random.default_rng(np.random.SeedSequence([17, 20261108]))
    expected = np.arange(len(rows))
    for query in range(3):
        ids = np.flatnonzero(rows['query'] == query)
        expected[ids] = rng.permutation(ids)
    actual = permutation(rows, features, cards, 17)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(rows['query'][actual], rows['query'])
    assert np.any(rows['target'][actual] != rows['target'])
    contaminated = features.copy()
    contaminated[:, 0] = rows['target']
    with pytest.raises(ValueError, match='numeric evidence'):
        permutation(rows, contaminated, cards, 17)
    with pytest.raises(ValueError, match='historical text'):
        permutation(rows, features, rows['target'], 17)


@pytest.mark.parametrize('arm', ARMS)
def test_every_arm_has_direct_finite_gradients(tiny_inputs, arm):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from experiments.ablation_runtime import trainer
    work, data, cache, original = tiny_inputs
    cfg = variant_config(original, arm, work, 'synthetic')
    shared = trainer.SharedCache(data, cache, cfg, 'cpu')
    part = trainer.Partition(shared, 17, 'train')
    model = trainer.head_factory.make_prior_model(trainer.model_config(cfg), 12)
    inputs = part.batch(np.arange(6), np.array([True, False] * 3))
    c, d = model(*inputs)
    loss = trainer.base.exact_joint_loss(c, d, torch.as_tensor(part.records['label'][:6].astype('int64')))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if arm == 'NO_JEV':
        for state, mask in [(0, 1), (3, 4), (6, 7)]:
            assert torch.count_nonzero(inputs[state]) == 0
            assert torch.count_nonzero(inputs[mask]) == 0
        assert inputs[10] is None
        assert torch.count_nonzero(inputs[2]) > 0
        assert torch.count_nonzero(inputs[5]) > 0
        assert torch.count_nonzero(inputs[8]) > 0
    if arm in ('NO_JEV', 'NO_JOINT'):
        assert not hasattr(model, 'joint_attention')
        assert inputs[10] is None
    if arm == 'FLAT':
        assert not hasattr(model, 'changed_head')
        assert not hasattr(model, 'direction_head')
        captured = []
        handle = model.flat_head.register_forward_hook(lambda module, args, output: captured.append(output))
        c, d = model(*inputs)
        handle.remove()
        probabilities = torch.stack((c.sigmoid() * d.sigmoid(), 1 - c.sigmoid(),
                                     c.sigmoid() * (1-d.sigmoid())), dim=1)
        torch.testing.assert_close(probabilities, captured[0].softmax(1), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('arm', ['S1Pert', 'FLAT'])
def test_cpu_retrain_restore_evaluate_and_permute(tiny_inputs, arm, monkeypatch):
    pytest.importorskip('torch')
    from experiments.ablation_runtime import trainer
    work, data, cache, original = tiny_inputs
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(ROOT / 'src'), str(ROOT), os.environ.get('PYTHONPATH', '')]))
    cfg = variant_config(original, arm, work, 'synthetic')
    out = run_directory(work, arm, 'synthetic', 17)
    report = trainer.base.train(cfg, 'synthetic', 17, data, cache, out)
    assert report['status'] == 'CONVERGED'
    assert report['independent_restore']['status'] == 'PASS'
    result = trainer.base.evaluate_test(cfg, 'synthetic', 17, data, cache, out)
    assert result['status'] == 'PASS'
    assert not result['test_used_for_training_or_selection']
    if arm == 'S1Pert':
        destination = run_directory(work, 'TARGET_PERMUTATION', 'synthetic', 17)
        diagnostic = permute_checkpoint(out, data, cache, destination, device='cpu')
        assert diagnostic['status'] == 'PASS'
        assert diagnostic['query_unchanged'] and diagnostic['numeric_evidence_unchanged']
        rows, missing = collect(work, 'synthetic', [17], allow_incomplete=True)
        assert len(rows) == 2 and len(missing) == 5
        assert len(render_table(summarize(rows, [17]))) == 7
        with pytest.raises(FileNotFoundError, match='Missing experiment outputs'):
            collect(work, 'synthetic', [17])


def test_table_sample_sd_and_duplicate_protection():
    rows = [dict(arm='S1Pert', cell='synthetic', seed=seed, macro_f1=value)
            for seed, value in zip(SEEDS, [.1, .2, .3, .4, .5])]
    summary = summarize(rows, SEEDS)
    metric = next(x for x in summary if x['arm'] == 'S1Pert' and x['metric'] == 'macro_f1')
    assert metric['mean'] == pytest.approx(.3)
    assert metric['sample_sd'] == pytest.approx(np.std([.1, .2, .3, .4, .5], ddof=1))
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(rows + rows[:1], SEEDS)


def test_plan_is_portable_and_does_not_open_model_assets(tmp_path, monkeypatch):
    monkeypatch.setenv('S1PERT_ROOT', str(tmp_path))
    assert work_directory() == tmp_path
    config = tmp_path / 'config.json'
    config.write_text('{}')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT)])
    result = subprocess.run([sys.executable, str(ROOT / 'experiments/ablation.py'), 'plan',
                             '--config', str(config)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert len(plan['jobs']) == 30 and plan['models_started'] is False
    assert not (tmp_path / 'paper').exists()
