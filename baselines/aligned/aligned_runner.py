"""Six-stage ALIGNED controller and independent-process, restore-gated evaluation.

Only evaluate opens TEST, after exact full-VAL restoration. The selected output
is always hybrid after refine1; no validation-based checkpoint or output choice.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .aligned_data import load_dataset, sha256
from .aligned_engine import DEFAULTS, ADAPTATIONS
from .aligned_runtime import load_prepared, build_engine, fingerprint, atomic_json

STAGES = ('init', 'pretrain', 'retrain0', 'refine0', 'retrain1', 'refine1')
from jev_a4.core import metrics as _metrics
METRICS_SOURCE = Path(_metrics.__file__)
metrics_from_confusion = _metrics.metrics_from_confusion
RESERVE_SECONDS = 15 * 60


def utcnow():
    return datetime.now(timezone.utc)


def parse_deadline(value):
    deadline = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        raise ValueError('deadline requires an explicit timezone')
    return deadline.astimezone(timezone.utc)


def read_json(path):
    return json.loads(Path(path).read_text())


def configure_runtime(config, seed):
    threads = int(config.get('cpu_threads', 4))
    if threads < 1:
        raise ValueError('cpu_threads must be positive')
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    supplied = read_json(path)
    config = {**DEFAULTS, **supplied}
    profile = config.get('profile')
    if profile not in ('official_default_a4', 'budgeted', 'smoke_e2e'):
        raise ValueError('config profile must explicitly declare official_default_a4, budgeted, or smoke_e2e')
    counts = ('weight_init_epochs', 'pretrain_epochs', 'pretrain_reinforce_epochs',
              'retrain_epochs', 'retrain_reinforce_epochs', 'refine_epochs')
    for key in counts:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['rounds'] != 2:
        raise ValueError('ALIGNED always requires both complete abduction/refinement rounds')
    fixed = ('hidden_dim', 'closure', 'closure_type', 'pretrain_C', 'retrain_C',
             'weight_init_lr', 'training_lr', 'refine_lr', 'refine_C', 'refine_t', 'refine_t0', 'refine_tol', 'gamma')
    for key in fixed:
        if config[key] != DEFAULTS[key]:
            raise ValueError(f'this runner preserves the official objective parameter {key}')
    if profile == 'official_default_a4' and any(config[key] != DEFAULTS[key] for key in counts):
        raise ValueError('reduced schedules must be explicitly declared budgeted or smoke_e2e')
    if profile == 'official_default_a4' and config['batch_size'] != DEFAULTS['batch_size']:
        raise ValueError('official batch_size must stay 64: it determines CE summation and baseline EMA')
    if not isinstance(config['batch_size'], int) or config['batch_size'] < 1:
        raise ValueError('batch_size must be positive')
    config.setdefault('evaluation_batch_size', 64)
    if not isinstance(config['evaluation_batch_size'], int) or config['evaluation_batch_size'] < 1:
        raise ValueError('evaluation_batch_size must be positive')
    config.setdefault('cpu_threads', 4)
    return config


def source_fingerprint():
    return {**fingerprint(), 'a4_metrics.py': sha256(METRICS_SOURCE)}


def data_fingerprint(data):
    return {'report_sha256': sha256(data['prepared'] / 'report.json'),
            'artifacts_sha256': data['report']['artifacts_sha256'],
            'sealed_sha256': data['report']['identity']['sha256'],
            'gene_ids_sha256': data['report']['identity']['gene_ids_sha256']}


def run_identity(cell, seed, config, data):
    return {'cell': cell, 'seed': int(seed), 'config': config,
            'source': source_fingerprint(), 'data': data_fingerprint(data),
            'prediction_rule': 'fixed where(R>0.5,KB.deduce,neural_argmax) after refine1',
            'selection': 'none; complete six stages; no VAL selection',
            'adaptations': ADAPTATIONS}


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def atomic_torch(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def checkpoint(engine, driver, identity):
    return {'format_version': 1, 'engine': engine.state_dict(), 'driver': driver,
            'identity': identity, 'when_utc': utcnow().isoformat()}


def begin_stage(engine, driver, data, config):
    stage = STAGES[driver['next_stage']]
    if stage == 'init':
        engine.begin_weight_init(epochs=config['weight_init_epochs'], lr=config['weight_init_lr'])
    elif stage == 'pretrain':
        engine.set_data(data['X_train'], data['Y_train'], data['mask_train'])
        engine.begin_training(stage, epochs=config['pretrain_epochs'],
            reinforce_epochs=config['pretrain_reinforce_epochs'], C=config['pretrain_C'],
            lr=config['training_lr'], gamma=config['gamma'])
    elif stage.startswith('retrain'):
        # Freeze this round's modified labels once. The following refine stage
        # consumes this exact tensor, including after an interrupted process.
        driver['modified'] = engine.pseudo_labels(data['X_train'])['modified']
        engine.set_data(data['X_train'], driver['modified'], update_weight=True)
        engine.begin_training(stage, epochs=config['retrain_epochs'],
            reinforce_epochs=config['retrain_reinforce_epochs'], C=config['retrain_C'],
            lr=config['training_lr'], gamma=config['gamma'])
    else:
        if driver['modified'] is None:
            raise ValueError('refine requires the same round-frozen pseudo labels')
        engine.begin_refine(data['X_train'], driver['modified'], stage=stage,
            k=config['closure'], epochs=config['refine_epochs'], lr=config['refine_lr'],
            C=config['refine_C'], t=config['refine_t'], t0=config['refine_t0'], tol=config['refine_tol'])
    driver['active_stage'] = stage


def train(args):
    output = Path(args.output)
    config = load_config(args.config)
    deadline = parse_deadline(args.deadline) if getattr(args, 'deadline', None) else datetime.max.replace(tzinfo=timezone.utc)
    configure_runtime(config, int(args.seed))
    with output_lock(output):
        data = load_prepared(args.cell, int(args.seed))
        identity = run_identity(args.cell, int(args.seed), config, data)
        existing_identity = output / 'identity.json'
        if existing_identity.exists():
            if read_json(existing_identity) != identity:
                raise ValueError('run identity changed; refusing to overwrite or resume')
            if not args.resume:
                raise ValueError('existing run requires --resume')
        elif args.resume:
            raise ValueError('resume requires an existing identity and checkpoint')
        if (output / 'frozen.json').exists():
            return read_json(output / 'frozen.json')
        atomic_json(existing_identity, identity)
        atomic_json(output / 'config.json', config)
        engine, weights_report = build_engine(data, device=getattr(args, 'device', 'cuda'), batch_size=config['batch_size'])
        driver = {'next_stage': 0, 'active_stage': None, 'completed_stages': [], 'modified': None}
        if args.resume:
            state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
            if state.get('format_version') != 1 or state['identity'] != identity:
                raise ValueError('checkpoint identity changed')
            driver = state['driver']
            if driver['completed_stages'] != list(STAGES[:driver['next_stage']]):
                raise ValueError('checkpoint stage order is inconsistent')
            engine.load_state_dict(state['engine'])
        atomic_json(output / 'label_weights.json', weights_report)
        atomic_torch(output / 'last.pt', checkpoint(engine, driver, identity))
        last_save, last_status = time.monotonic(), 0.
        started = time.monotonic()
        with (output / 'events.jsonl').open('a') as events:
            while driver['next_stage'] < len(STAGES):
                if utcnow() >= deadline - timedelta(seconds=RESERVE_SECONDS):
                    atomic_torch(output / 'last.pt', checkpoint(engine, driver, identity))
                    status = {'status': 'PARTIAL', 'reason': 'deadline_reserve',
                        'deadline': deadline.isoformat(), 'reserve_seconds': RESERVE_SECONDS,
                        'stage': driver['active_stage'], 'next_stage': driver['next_stage'],
                        'completed_stages': driver['completed_stages'], 'test_opened': False,
                        'when_utc': utcnow().isoformat()}
                    atomic_json(output / 'status.json', status)
                    events.write(json.dumps(status) + '\n'); events.flush()
                    return status
                if driver['active_stage'] is None:
                    begin_stage(engine, driver, data, config)
                    atomic_torch(output / 'last.pt', checkpoint(engine, driver, identity))
                    last_save = time.monotonic()
                    atomic_json(output / 'status.json', {'status': 'RUNNING',
                        'stage': driver['active_stage'], 'step': 0, 'test_opened': False,
                        'when_utc': utcnow().isoformat()})
                before = time.monotonic()
                if engine.mode == 'weight_init':
                    progress = engine.weight_init_iteration()
                elif engine.mode == 'training':
                    progress = engine.train_iteration()
                else:
                    progress = engine.refine_iteration()
                progress = {**progress, 'stage': driver['active_stage'], 'status': 'RUNNING',
                    'step_seconds': time.monotonic() - before, 'process_elapsed_seconds': time.monotonic() - started,
                    'when_utc': utcnow().isoformat(), 'test_opened': False}
                events.write(json.dumps(progress, allow_nan=False) + '\n')
                if engine.done:
                    ended = driver['active_stage']
                    driver['completed_stages'].append(ended)
                    driver['next_stage'] += 1
                    driver['active_stage'] = None
                    if ended.startswith('refine'):
                        driver['modified'] = None
                now = time.monotonic()
                if progress['done'] or now - last_status >= 5:
                    atomic_json(output / 'status.json', {**progress, 'completed_stages': driver['completed_stages']})
                    events.flush()
                    last_status = now
                if progress['done'] or now - last_save >= 600:
                    atomic_torch(output / 'last.pt', checkpoint(engine, driver, identity))
                    last_save = time.monotonic()
        # Only this fixed endpoint is frozen. There is no VAL score comparison.
        atomic_torch(output / 'final.pt', checkpoint(engine, driver, identity))
        val = engine.predict(data['X_val'], batch_size=config['evaluation_batch_size'], include_r=False)
        atomic_npz(output / 'val_reference.npz', **{k: v.numpy() for k, v in val.items()})
        frozen = {'status': 'TRAIN_COMPLETE_PENDING_RESTORE_EVAL', 'identity': identity,
            'cell': args.cell, 'seed': int(args.seed), 'config': config,
            'training_pid': os.getpid(), 'completed_stages': driver['completed_stages'],
            'final_checkpoint_sha256': sha256(output / 'final.pt'),
            'val_reference_sha256': sha256(output / 'val_reference.npz'),
            'selected_stage': 'refine1', 'test_opened': False,
            'when_utc': utcnow().isoformat()}
        atomic_json(output / 'frozen.json', frozen)
        atomic_json(output / 'status.json', frozen)
        return frozen


def hard_label_metrics(records, prediction):
    labels = np.asarray(records['label'], dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if len(labels) == 0 or labels.shape != prediction.shape or not np.isin(prediction, [0, 1, 2]).all():
        raise ValueError('nonempty aligned valid A4 hard labels required')
    cm = np.bincount(labels * 3 + prediction, minlength=9).reshape(3, 3)
    result = metrics_from_confusion(cm)
    unique, inverse, counts = np.unique(records['target'], return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    weighted_cm = np.bincount(labels * 3 + prediction, weights=weights, minlength=9).reshape(3, 3)
    result['equal_target_weighted_confusion_metrics'] = metrics_from_confusion(weighted_cm)
    changed_true, changed_pred = labels != 1, prediction != 1
    tp = int(np.count_nonzero(changed_true & changed_pred))
    precision = tp / max(int(changed_pred.sum()), 1)
    recall = tp / max(int(changed_true.sum()), 1)
    for row in result['per_class'].values():
        row['average_precision'] = None
    result.update(records=len(labels), targets=len(unique), class_order=['up', 'same', 'down'],
        nll=None, brier_multiclass_sum=None,
        changed={'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'support': int(changed_true.sum()), 'average_precision': None},
        equal_target_weighting='each target total weight 1; metrics derived from weighted confusion',
        probability_metrics_status='unavailable: fixed hybrid supplies hard labels, not calibrated probabilities')
    return result


def score_records(model, kb_matrix, x_all, row_target, row_gene, batch_size=64, device='cpu'):
    """Original final hybrid score: neural softmax, or final-KB point mass.

    This is the inspected continuous-score export, not a one-hot encoding of
    every hard prediction. Native classes [-1,0,+1] map to [Up,Same,Down].
    """
    count = len(row_target)
    probabilities = np.empty((count, 3), dtype=np.float32)
    neural = np.empty_like(probabilities)
    switches = np.empty(count, dtype=np.float32)
    deduction = np.empty(count, dtype=np.int64)
    neural_hard = np.empty(count, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for left in range(0, x_all.shape[0], batch_size):
            right = min(left + batch_size, x_all.shape[0])
            indices = np.flatnonzero((row_target >= left) & (row_target < right))
            if not len(indices):
                continue
            block = x_all[left:right]
            block = block.toarray() if hasattr(block, 'toarray') else np.asarray(block)
            x = torch.as_tensor(block, dtype=torch.float32, device=device)
            native, r = model(x)
            deduced = torch.clamp(x @ kb_matrix, -1., 1.).int()
            rows = torch.as_tensor(row_target[indices] - left, dtype=torch.long, device=device)
            genes = torch.as_tensor(row_gene[indices], dtype=torch.long, device=device)
            selected, gate, signed = native[rows, genes], r[rows, genes], deduced[rows, genes].long()
            canonical, deduced_class = selected.flip(-1), 1-signed
            scores = torch.where((gate > .5)[:, None],
                torch.nn.functional.one_hot(deduced_class, num_classes=3).to(canonical.dtype), canonical)
            probabilities[indices], neural[indices] = scores.cpu().numpy(), canonical.cpu().numpy()
            switches[indices], deduction[indices] = gate.cpu().numpy(), deduced_class.cpu().numpy()
            neural_hard[indices] = (1-(selected.argmax(-1)-1)).cpu().numpy()
    if (not np.isfinite(probabilities).all() or not np.isfinite(switches).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=2e-6)):
        raise ValueError('invalid final hybrid probabilities')
    hard = np.where(switches > .5, deduction, neural_hard)
    if not np.array_equal(probabilities.argmax(1), hard):
        raise ValueError('class-order tie changes the original hybrid argmax')
    return {'probabilities': probabilities, 'neural_probs': neural, 'R': switches,
            'prediction': hard, 'neural_prediction': neural_hard, 'deduction_prediction': deduction}


def record_predictions(data, partition, predictions, output, engine):
    records = data['records_' + partition]
    row, gene = data['row_target_' + partition], data['row_gene_' + partition]
    mapped = {}
    for name in ('hybrid', 'neural', 'deduction'):
        value = predictions[name]
        matrix = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if not np.isin(matrix, [-1, 0, 1]).all():
            raise ValueError('invalid native ALIGNED prediction')
        mapped[name] = 1 - matrix[row, gene]
    scores = score_records(engine.model, engine.kb.KB, data['X_' + partition], row, gene,
                          batch_size=engine.batch_size, device=str(engine.device))
    for name, score_name in (('hybrid', 'prediction'), ('neural', 'neural_prediction'), ('deduction', 'deduction_prediction')):
        if not np.array_equal(mapped[name], scores[score_name]):
            raise ValueError('continuous-score export differs from restored ' + name + ' decision')
    destination = output / f'{partition}_predictions.npz'
    atomic_npz(destination, target=records['target'], query=records['query'], labels=records['label'],
               label=records['label'], **scores)
    atomic_json(destination.with_suffix('.json'), {'status': 'PASS', 'schema': 'aligned.frozen-hybrid-scores.v1',
        'method': 'ALIGNED', 'partition': partition, 'class_order': ['Up', 'Same', 'Down'],
        'formula': 'where(R>0.5,one_hot(1-final_KB.deduce(X)),final_neural_probs[..., [2,1,0]])',
        'continuous_score_source': 'final neural softmax on R<=0.5; final KB point mass on R>0.5',
        'calibrated': False, 'rows': len(records), 'output_sha256': sha256(destination),
        'sealed_hybrid_argmax_exact_match': True})
    metrics = {name: hard_label_metrics(records, value) for name, value in mapped.items()}
    metrics['hybrid'] = _metrics.classification_metrics(records['label'], scores['probabilities'], records['target'])
    return metrics


def evaluate(args):
    output = Path(args.output)
    with output_lock(output):
        completed_report = None
        if (output / 'report.json').exists():
            report = read_json(output / 'report.json')
            if report.get('status') == 'PASS':
                completed_report = report
        frozen = read_json(output / 'frozen.json')
        config, identity = frozen['config'], frozen['identity']
        val_only = bool(getattr(args, 'val_only', False))
        if config['profile'] == 'smoke_e2e' and not val_only:
            raise ValueError('smoke_e2e cannot open TEST; use --val-only')
        if completed_report is None and frozen['training_pid'] == os.getpid():
            raise ValueError('evaluation must run in a new process after training exits')
        if frozen['completed_stages'] != list(STAGES) or frozen['selected_stage'] != 'refine1':
            raise ValueError('both complete ALIGNED rounds are required')
        if identity['source'] != source_fingerprint():
            raise ValueError('source identity changed since training')
        if (config != identity['config'] or load_config(output / 'config.json') != config
                or frozen['cell'] != identity['cell'] or int(frozen['seed']) != identity['seed']):
            raise ValueError('frozen config identity changed')
        configure_runtime(config, int(frozen['seed']))
        data = load_prepared(frozen['cell'], int(frozen['seed']))
        if run_identity(frozen['cell'], frozen['seed'], config, data) != identity:
            raise ValueError('data/config/source identity changed before restoration')
        if read_json(output / 'identity.json') != identity:
            raise ValueError('run identity file changed')
        if sha256(output / 'final.pt') != frozen['final_checkpoint_sha256']:
            raise ValueError('frozen checkpoint changed')
        if sha256(output / 'val_reference.npz') != frozen['val_reference_sha256']:
            raise ValueError('frozen VAL reference changed')
        if completed_report is not None:
            access = read_json(output / 'test_access.json')
            if (completed_report['identity'] != identity
                    or completed_report['checkpoint_sha256'] != frozen['final_checkpoint_sha256']
                    or access.get('status') != 'COMPLETE'
                    or access.get('report_sha256') != sha256(output / 'report.json')):
                raise ValueError('completed report identity/hash or TEST completion guard changed')
            return completed_report  # Identity verified; never reopen TEST.
        state = torch.load(output / 'final.pt', map_location='cpu', weights_only=False)
        if state['identity'] != identity or state['driver']['completed_stages'] != list(STAGES):
            raise ValueError('final checkpoint identity/stages mismatch')
        engine, _ = build_engine(data, device=getattr(args, 'device', 'cuda'), batch_size=config['batch_size'])
        engine.load_state_dict(state['engine'])
        restored = engine.predict(data['X_val'], batch_size=config['evaluation_batch_size'], include_r=False)
        with np.load(output / 'val_reference.npz', allow_pickle=False) as reference:
            if set(reference.files) != {'hybrid', 'neural', 'deduction'}:
                raise ValueError('incomplete frozen VAL reference')
            for name in reference.files:
                if not np.array_equal(reference[name], restored[name].numpy()):
                    raise ValueError(f'exact independent VAL restoration failed: {name}')
        val_metrics = record_predictions(data, 'val', restored, output, engine)
        restore_report = {'status': 'RESTORE_ONLY_PASS' if val_only else 'PASS',
            'fresh_process': True, 'pid': os.getpid(), 'training_pid': frozen['training_pid'],
            'full_val_exact_match': True, 'val': val_metrics, 'test_opened': False,
            'final_checkpoint_sha256': frozen['final_checkpoint_sha256'],
            'source': identity['source'], 'data': identity['data'], 'when_utc': utcnow().isoformat()}
        atomic_json(output / 'restore_report.json', restore_report)
        if val_only:
            return restore_report
        if (output / 'test_access.json').exists():
            raise ValueError('TEST access was already attempted; do not silently rerun it')
        atomic_json(output / 'test_access.json', {'status': 'STARTED', 'pid': os.getpid(),
            'restore_passed': True, 'when_utc': utcnow().isoformat()})
        # This is the only include_test=True call in the controller. Everything
        # above used TRAIN/VAL features and the fixed post-refine1 state only.
        full = load_dataset(frozen['cell'], int(frozen['seed']), include_test=True)
        if (full['gene_ids'] != data['gene_ids'] or any(
                full['identity']['sha256'].get(k) != v for k, v in identity['data']['sealed_sha256'].items())):
            raise ValueError('sealed source/axes changed at final TEST gate')
        x_test = full['X_test'].toarray() if hasattr(full['X_test'], 'toarray') else full['X_test']
        test_prediction = engine.predict(np.asarray(x_test, dtype=np.float32),
            batch_size=config['evaluation_batch_size'], include_r=False)
        test_metrics = record_predictions(full, 'test', test_prediction, output, engine)
        report = {'status': 'PASS', 'method': 'ALIGNED', 'cell': frozen['cell'], 'seed': frozen['seed'],
            'profile': config['profile'], 'scientific_status': 'A4_OFFICIAL_ALIGNED_OBJECTIVE_ADAPTATION'
                if config['profile'] == 'official_default_a4' else 'BUDGETED_A4_ALIGNED_ADAPTATION',
            'selected_stage': 'refine1', 'main_prediction': 'hybrid', 'validation_selection': False,
            'val': val_metrics['hybrid'], 'test': test_metrics['hybrid'],
            'diagnostic_predictions': {'val': {k: v for k, v in val_metrics.items() if k != 'hybrid'},
                'test': {k: v for k, v in test_metrics.items() if k != 'hybrid'}},
            'restore': restore_report, 'identity': identity, 'test_identity': full['identity'],
            'checkpoint_sha256': frozen['final_checkpoint_sha256'], 'test_opened': True,
            'when_utc': utcnow().isoformat()}
        atomic_json(output / 'report.json', report)
        atomic_json(output / 'test_access.json', {'status': 'COMPLETE', 'pid': os.getpid(),
            'report_sha256': sha256(output / 'report.json'), 'when_utc': utcnow().isoformat()})
        atomic_json(output / 'status.json', {'status': 'PASS', 'phase': 'evaluate',
            'report': str(output / 'report.json'), 'when_utc': utcnow().isoformat()})
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--cell', required=True)
    train_parser.add_argument('--seed', required=True, type=int)
    train_parser.add_argument('--config', required=True, type=Path)
    train_parser.add_argument('--deadline', required=True)
    train_parser.add_argument('--resume', action='store_true')
    eval_parser = sub.add_parser('evaluate')
    eval_parser.add_argument('--val-only', action='store_true')
    for command in (train_parser, eval_parser):
        command.add_argument('--output', required=True, type=Path)
        command.add_argument('--device', default='cuda')
    args = parser.parse_args()
    try:
        result = (train if args.action == 'train' else evaluate)(args)
        print(json.dumps({'status': result['status'], 'output': str(args.output)}), flush=True)
        return 2 if result['status'] == 'PARTIAL' else 0
    except BaseException as exc:
        atomic_json(args.output / 'failure.json', {'status': 'PARTIAL', 'action': args.action,
            'error': repr(exc), 'pid': os.getpid(), 'when_utc': utcnow().isoformat()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
