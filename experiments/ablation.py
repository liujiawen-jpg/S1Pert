"""Table IV: fresh component retraining and frozen-checkpoint target permutation.

Run after core prepare/encode. No historical task weights initialize any arm.
The only reused representations are the response-blind generic landmark caches.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / 'src')]
from jev_a4.config import load_config
from jev_a4.core import train as core

SEEDS = (17, 29, 43, 61, 89)
ARMS = ('S1Pert', 'NO_JEV', 'NO_JOINT', 'FLAT', 'NO_RANK', 'NO_MASK')
PERMUTATION_SALT = 20261108


def work_directory(value=None):
    value = value or os.environ.get('S1PERT_ROOT')
    if not value:
        raise ValueError('Set --work-dir or S1PERT_ROOT to the prepared workspace')
    return Path(value).expanduser().resolve()


def run_directory(work, arm, cell, seed, main_run_root=None):
    if arm == 'S1Pert' and main_run_root is not None:
        return Path(main_run_root).resolve() / cell / f'seed_{seed}'
    return Path(work) / 'paper/ablation' / arm / cell / f'seed_{seed}'


def variant_config(original, arm, work, cell):
    """Historical Stage-4 interventions; optimizer/schedule remain in base config."""
    if arm not in ARMS:
        raise ValueError('Unknown Table IV arm')
    cfg = copy.deepcopy(original)
    cfg.update(project_root=str(work), base_cache_dir=str(Path(work) / 'cache' / cell), gpu_cache_gib=24,
               branch_variant='FULL', joint_enabled=arm not in ('NO_JEV', 'NO_JOINT'),
               flat_output=arm == 'FLAT', numeric_enabled=True, numeric_only=arm == 'NO_JEV',
               unsynchronized_masking=False)
    st = cfg.setdefault('training', {})
    st.update(preload_shared_gpu=False, partition_ram_cache_budget_gib=0)
    st['pair_weight'] = 0.0 if arm == 'NO_RANK' else .02
    st['prior_dropout'] = 0.0 if arm == 'NO_MASK' else .10
    cfg['paper_ablation'] = dict(arm=arm, fresh_task_head=True,
        parameter_update_labels=['TRAIN'], selection_labels=['VAL'],
        old_prediction_output_anchor=False, historical_permutation_salt=PERMUTATION_SALT)
    return cfg


def permutation(records, features, cards, seed):
    """Shuffle targets within each query with historical evidence exactly fixed.

    Permuting whole batch row IDs also selects the joint cache for the shuffled
    target/query tuple. Labels remain in original row order for evaluation.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, PERMUTATION_SALT]))
    perm = np.arange(len(records))
    order = np.argsort(records['query'], kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(records['query'][order])) + 1, len(records)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        ids = order[a:b]
        perm[ids] = rng.permutation(ids)
    if not np.array_equal(records['query'][perm], records['query']):
        raise ValueError('Permutation changed query')
    if not np.array_equal(features[perm], features):
        raise ValueError('Permutation changed historical numeric evidence')
    if cards is not None and not np.array_equal(cards[perm], cards):
        raise ValueError('Permutation changed historical text')
    return perm


def load_checkpoint(model_dir, data_dir, cache_dir, device=None):
    """Verify a fresh direct checkpoint and return model, TEST partition, backend, seal.

    Supports main core and ablation checkpoints. An optional device override is
    execution-only: original code/config/input identities are checked first.
    """
    import torch
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / 'effective_config.json').read_text())
    seal = json.loads((model_dir / 'convergence.json').read_text())
    if seal.get('status') != 'CONVERGED':
        raise PermissionError('TEST requires a converged checkpoint')
    digest = core.sha256(model_dir / 'best.pt')
    if digest != seal['best_checkpoint_sha256']:
        raise ValueError('Checkpoint digest changed')
    restored = json.loads((model_dir / 'restore_report.json').read_text())
    if restored.get('status') != 'PASS' or restored.get('checkpoint_sha256') != digest:
        raise PermissionError('Independent checkpoint restoration has not passed')
    checkpoint = torch.load(model_dir / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint['schema'] == 'decider.a4.complete_branch_ablation.fresh_head.v1':
        from experiments.ablation_runtime import trainer
        backend = trainer.base
    elif checkpoint['schema'] == core.CHECKPOINT_SCHEMA:
        from jev_a4.core import train_memory_safe
        train_memory_safe.install()
        backend = train_memory_safe.train
    else:
        raise ValueError('Unsupported checkpoint schema')
    meta = checkpoint['meta']
    if (not meta.get('fresh_task_head') or meta.get('old_task_checkpoint_initialization')
            or meta.get('old_prediction_output_anchor') or meta.get('output_residual_correction')):
        raise ValueError('Checkpoint is not an independent direct task head')
    seed, cell = int(seal['seed']), seal['dataset']
    if meta['dataset'] != cell or int(meta['seed']) != seed:
        raise ValueError('Checkpoint split identity mismatch')
    current = backend.bindings(cfg, data_dir, cache_dir, seed)
    if current != seal['bindings'] or current != meta['bindings']:
        raise ValueError('Checkpoint code/config/input identity drift')
    runtime_cfg = copy.deepcopy(cfg)
    if device is not None:
        runtime_cfg['device'] = device
    if runtime_cfg.get('cpu_threads'):
        torch.set_num_threads(int(runtime_cfg['cpu_threads']))
    device = str(runtime_cfg.get('device', 'cuda'))
    shared = backend.SharedCache(data_dir, cache_dir, runtime_cfg, device)
    model = backend.vendor_module().make_prior_model(meta['model_config'], meta['prior_feature_width'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    part = backend.Partition(shared, seed, 'test')
    return model, part, backend, seal


def permute_checkpoint(model_dir, data_dir, cache_dir, output, device=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new permutation output directory')
    model, part, backend, seal = load_checkpoint(model_dir, data_dir, cache_dir, device)
    batch = backend.settings(part.shared.cfg)['evaluation_batch']
    # Match the selected checkpoint's unmodified evaluation before intervention.
    ids = np.arange(min(len(part.records), batch))
    reference = backend.predict(model, part, batch, indices=ids)
    with np.load(Path(model_dir) / 'test_predictions.npz', allow_pickle=False) as source:
        for key, field in [('target', 'target'), ('query', 'query'), ('labels', 'label')]:
            if not np.array_equal(source[key], part.records[field]):
                raise ValueError('Reference prediction record order mismatch')
        expected = source['probabilities'][ids]
    np.testing.assert_allclose(reference, expected, rtol=0, atol=2e-5)
    perm = permutation(part.records, part.features, part.card_ids, int(seal['seed']))
    base_batch = part.batch
    part.batch = lambda indices, keep=None: base_batch(perm[indices], keep)
    started = time.monotonic()
    try:
        predictions = backend.predict(model, part, batch, part.shared.guard)
    finally:
        part.batch = base_batch
    output.mkdir(parents=True, exist_ok=True)
    backend.atomic_npz(output / 'test_predictions.npz', target=part.records['target'],
        query=part.records['query'], labels=part.records['label'], probabilities=predictions,
        checkpoint_sha256=np.asarray(seal['best_checkpoint_sha256']))
    np.save(output / 'permutation.npy', perm, allow_pickle=False)
    from jev_a4.evaluation import evaluate
    report = dict(status='PASS', arm='Target permutation', dataset=seal['dataset'], seed=seal['seed'],
        checkpoint_sha256=seal['best_checkpoint_sha256'],
        source_predictions_sha256=core.sha256(Path(model_dir) / 'test_predictions.npz'),
        test_predictions_sha256=core.sha256(output / 'test_predictions.npz'),
        permutation_sha256=core.sha256(output / 'permutation.npy'), salt=PERMUTATION_SALT,
        query_unchanged=True, numeric_evidence_unchanged=True, evidence_cards_unchanged=True,
        labels_unchanged=True, target_moved_fraction=float(np.mean(part.records['target'][perm] != part.records['target'])),
        reference_max_abs=float(np.abs(reference - expected).max()),
        elapsed_seconds=time.monotonic() - started,
        metrics=evaluate(part.records, output / 'test_predictions.npz'))
    backend.atomic_json(output / 'test_report.json', report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan', 'train', 'permutation', 'all'))
    p.add_argument('--work-dir', type=Path)
    p.add_argument('--config', type=Path, help='Default: WORK/configs/a4.json')
    p.add_argument('--cell', default='k562_essential')
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--main-run-root', type=Path, help='Use unchanged main reference ROOT/CELL/seed_N')
    p.add_argument('--device', help='Optional inference-only device override')
    a = p.parse_args(argv)
    work = work_directory(a.work_dir)
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.arms)) != len(a.arms):
        p.error('Duplicate seeds/arms are not allowed')
    cfg = load_config(a.config or work / 'configs/a4.json')
    jobs = [dict(arm=arm, cell=a.cell, seed=seed,
                 output=str(run_directory(work, arm, a.cell, seed, a.main_run_root)),
                 action='reference_only' if arm == 'S1Pert' and a.main_run_root else 'fresh_train')
            for arm in a.arms for seed in a.seeds]
    if a.action == 'plan':
        print(json.dumps(dict(jobs=jobs, models_started=False,
                             permutation_salt=PERMUTATION_SALT), indent=2))
        return
    data, cache = work / 'data' / a.cell, work / 'cache' / a.cell
    if a.action in ('train', 'all'):
        for job in jobs:
            out = Path(job['output'])
            if job['action'] == 'reference_only':
                if not (out / 'test_predictions.npz').is_file():
                    raise FileNotFoundError('Run the main reference before Table IV: ' + str(out))
                continue
            if out.exists() and any(out.iterdir()):
                raise FileExistsError('Training outputs are immutable; use a fresh workspace: ' + str(out))
            config_path = work / 'paper/ablation/configs' / job['arm'] / a.cell / f"seed_{job['seed']}.json"
            arm_cfg = variant_config(cfg, job['arm'], work, a.cell)
            core.atomic_json(config_path, arm_cfg)
            args = ['--config', str(config_path), '--dataset', a.cell, '--seed', str(job['seed']),
                    '--data-dir', str(data), '--cache-dir', str(cache), '--out', str(out)]
            logdir = work / 'paper/ablation/logs' / job['arm'] / a.cell
            logdir.mkdir(parents=True, exist_ok=True)
            for stage in ('train', 'evaluate'):
                with (logdir / f"seed_{job['seed']}_{stage}.log").open('x') as log:
                    subprocess.run([sys.executable, str(REPOSITORY / 'experiments/ablation_runtime/trainer.py'),
                                    stage, *args], stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=float(cfg.get('execution', {}).get('stage_timeout_seconds', 108000)))
                if stage == 'train':
                    report = json.loads((out / 'training_report.json').read_text())
                    if report.get('status') != 'CONVERGED':
                        raise RuntimeError('Arm did not converge; TEST remains locked: ' + str(out))
            print(json.dumps(job | {'status': 'PASS'}), flush=True)
    if a.action in ('permutation', 'all'):
        for seed in a.seeds:
            source = run_directory(work, 'S1Pert', a.cell, seed, a.main_run_root)
            destination = run_directory(work, 'TARGET_PERMUTATION', a.cell, seed)
            print(json.dumps(permute_checkpoint(source, data, cache, destination, a.device)), flush=True)


if __name__ == '__main__':
    main()
