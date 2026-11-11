"""Portable ALIGNED-MLP A4 public-signed-KG adaptation.

The six-stage schedule is init/pretrain/retrain0/refine0/retrain1/refine1.
Task weights start fresh. This standalone baseline is never an S1Pert output
anchor. Its internal final neural/knowledge switch is the original ALIGNED
prediction rule, retained for a faithful independent comparison.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from . import aligned_data as data

CELLS = ('hepg2', 'jurkat', 'rpe1', 'k562_essential')
SEEDS = (17, 29, 43, 61, 89)
ROOT = Path(__file__).resolve().parent


def paths(work_dir, cell, seed, output=None, prepared_root=None, public_db=None, go_db=None):
    work = data.configure(work_dir, prepared_root=prepared_root, public_db=public_db, go_db=go_db)
    if cell not in CELLS or int(seed) not in SEEDS:
        raise ValueError('unsupported paper cell or split seed')
    return work, Path(output) if output else work / 'baselines/aligned/runs' / cell / f'seed_{seed}'


def ensure_prepared(cell, seed):
    from .aligned_runtime import load_prepared
    directory = data.PREPARED / cell / f'seed_{seed}'
    if not directory.exists():
        for path in (data.PUBLIC_DB, data.GO_DB):
            if not path.is_file():
                raise FileNotFoundError(f'Public knowledge input missing: {path}. Run python -m data_prep.build_public_knowledge first.')
        data.prepare(cell, seed)
    else:
        load_prepared(cell, seed)
    return directory


def train(work_dir, cell, seed, *, output=None, prepared_root=None, config=None, device='cuda',
          public_db=None, go_db=None, resume=False, deadline=None):
    from . import aligned_runner as runner
    work, output = paths(work_dir, cell, seed, output, prepared_root, public_db, go_db)
    ensure_prepared(cell, seed)
    candidate = work / 'configs/aligned.json'
    config = Path(config) if config else candidate if candidate.exists() else ROOT / 'configs/official_default_a4.json'
    return runner.train(SimpleNamespace(cell=cell, seed=int(seed), output=output, config=config,
                                       device=device, resume=resume, deadline=deadline))


def evaluate(work_dir, cell, seed, *, output=None, prepared_root=None, device='cuda', restore_only=False):
    from . import aligned_runner as runner
    _, output = paths(work_dir, cell, seed, output, prepared_root)
    frozen = data.read_json(output / 'frozen.json')
    if frozen['cell'] != cell or int(frozen['seed']) != int(seed):
        raise ValueError('requested cell/seed differs from frozen run')
    return runner.evaluate(SimpleNamespace(output=output, device=device, val_only=restore_only))


def _timing_structure(work, cell, seed, gene_ids):
    """Read only structural columns of the fixed TEST records."""
    from scipy import sparse
    base = Path(work) / 'data' / cell
    meta = data.read_json(base / 'dataset.json')
    seal = data.read_json(base / 'preparation_report.json')
    if seal['status'] != 'PASS':
        raise ValueError('strict data preparation incomplete')
    for relative in ('dataset.json', f'seed_{seed}/target_partition.npy', f'seed_{seed}/test/records.npy'):
        if data.sha256(base / relative) != seal['artifacts'][relative]['sha256']:
            raise ValueError('structural source changed: ' + relative)
    partition = np.load(base / f'seed_{seed}/target_partition.npy', allow_pickle=False)
    records = np.load(base / f'seed_{seed}/test/records.npy', mmap_mode='r', allow_pickle=False)
    t, q = records['target'], records['query']  # TEST labels deliberately unread.
    if not len(t) or not np.all(partition[t] == 2):
        raise ValueError('requires nonempty fixed TEST structural keys')
    axis = {gene: i for i, gene in enumerate(gene_ids)}
    targets = np.flatnonzero(partition == 2)
    local = np.full(len(partition), -1, dtype=np.int64)
    local[targets] = np.arange(len(targets))
    x = sparse.csr_matrix((-np.ones(len(targets), dtype=np.float32),
        (np.arange(len(targets)), [axis[meta['target_ids'][i]] for i in targets])), shape=(len(targets), len(gene_ids)))
    rows = local[t]
    genes = np.asarray([axis[meta['query_ids'][i]] for i in q], dtype=np.int64)
    return x, rows, genes


def load_for_timing(work_dir, cell, seed, run_dir=None, device=None, prepared_root=None):
    """Restore outside timed region; each run executes live final inference.

    If needed, a separate process performs full-VAL restoration before TEST
    structural keys are opened. Timing includes MLP softmax, R, final-KB matrix
    multiplication/clamp, original switch and record selection, not saved scores.
    """
    import torch
    from . import aligned_runner as runner
    from . import aligned_runtime as runtime
    from .aligned_engine import AdaptorMLP
    device = str(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    work, output = paths(work_dir, cell, seed, run_dir, prepared_root)
    frozen = data.read_json(output / 'frozen.json')
    if frozen['completed_stages'] != list(runner.STAGES) or frozen['selected_stage'] != 'refine1':
        raise ValueError('both complete ALIGNED rounds are required')
    if frozen['cell'] != cell or int(frozen['seed']) != int(seed):
        raise ValueError('frozen model identity differs')
    prepared = runtime.load_prepared(cell, seed)
    if runner.run_identity(cell, int(seed), frozen['config'], prepared) != frozen['identity']:
        raise ValueError('frozen model source/config/data bindings changed')
    if data.sha256(output / 'final.pt') != frozen['final_checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    restore_path = output / 'restore_report.json'
    if not restore_path.exists():
        command = [sys.executable, '-m', 'baselines.aligned.run', 'restore', '--work-dir', str(work),
                   '--cell', cell, '--seed', str(seed), '--output', str(output), '--device', device,
                   '--prepared-root', str(data.PREPARED)]
        subprocess.run(command, check=True, cwd=ROOT.parents[1])
    restore = data.read_json(restore_path)
    if (restore.get('status') not in ('PASS', 'RESTORE_ONLY_PASS') or not restore.get('fresh_process')
            or not restore.get('full_val_exact_match') or restore.get('pid') == frozen['training_pid']
            or restore.get('final_checkpoint_sha256') != frozen['final_checkpoint_sha256']
            or restore.get('source') != frozen['identity']['source'] or restore.get('data') != frozen['identity']['data']):
        raise ValueError('independent restoration gate failed')
    checkpoint = torch.load(output / 'final.pt', map_location='cpu', weights_only=False, mmap=True)
    if checkpoint['identity'] != frozen['identity'] or checkpoint['driver']['completed_stages'] != list(runner.STAGES):
        raise ValueError('frozen checkpoint internal identity differs')
    state = checkpoint['engine']
    n = len(prepared['gene_ids'])
    with torch.random.fork_rng(devices=[]):
        model = AdaptorMLP(n, frozen['config']['hidden_dim'], n)
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    kb = state['kb']['KB'].to(device)
    x, rows, genes = _timing_structure(work, cell, seed, prepared['gene_ids'])
    del state, checkpoint, prepared
    owned = {'model': model, 'kb': kb}

    def run():
        return runner.score_records(owned['model'], owned['kb'], x, rows, genes,
                                    batch_size=int(frozen['config']['evaluation_batch_size']), device=device)['probabilities']

    def synchronize():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)

    def cleanup():
        owned.clear()
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    return {'run': run, 'synchronize': synchronize, 'n_pairs': len(rows), 'device': device, 'cleanup': cleanup,
        'metadata': {'method': 'ALIGNED', 'checkpoint_sha256': frozen['final_checkpoint_sha256'],
            'selected_stage': 'refine1', 'test_response_labels_read': False,
            'train_val_labels_opened_for_integrity_validation': True, 'saved_predictions_replayed': False,
            'includes': ['MLP softmax', 'learned R switch', 'final KB deduction', 'hybrid probability', 'record selection'],
            'model_and_final_KB_restoration_timed_as_setup': True}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train', 'restore', 'evaluate'))
    parser.add_argument('--work-dir')
    parser.add_argument('--cell', choices=CELLS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--output', type=Path, help='exact run directory; prepare uses --prepared-root')
    parser.add_argument('--prepared-root', type=Path, help='parent containing cell/seed_N prepared arrays')
    parser.add_argument('--public-db', type=Path)
    parser.add_argument('--go-db', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deadline', help='optional explicit ISO8601 deadline; no historical deadline is inherited')
    args = parser.parse_args(argv)
    work, output = paths(args.work_dir, args.cell, args.seed, args.output, args.prepared_root, args.public_db, args.go_db)
    if args.action == 'prepare':
        report = data.prepare(args.cell, args.seed)
    elif args.action == 'train':
        report = train(work, args.cell, args.seed, output=output, prepared_root=args.prepared_root,
            public_db=args.public_db, go_db=args.go_db, config=args.config, device=args.device,
            resume=args.resume, deadline=args.deadline)
    else:
        report = evaluate(work, args.cell, args.seed, output=output, prepared_root=args.prepared_root,
                          device=args.device, restore_only=args.action == 'restore')
    print(json.dumps({'status': report['status'], 'cell': args.cell, 'seed': args.seed, 'output': str(output)}))
    return 2 if report['status'] == 'PARTIAL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
