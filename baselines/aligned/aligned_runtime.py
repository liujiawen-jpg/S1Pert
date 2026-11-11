"""Shared audited data/initialization helpers; no TEST access here."""
import json
import hashlib
import os
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from . import aligned_data
from .aligned_data import sha256, load_dataset
from .aligned_engine import AlignedEngine, RegulatoryKB

ROOT = Path(__file__).resolve().parent

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)

def fingerprint():
    provenance = json.loads((ROOT / 'official_source' / 'core_provenance.json').read_text())
    if provenance['commit'] != '264e26e3fb826799c7f5e7637fa776d3feca4553':
        raise ValueError('unexpected official ALIGNED source commit')
    for relative, identity in provenance['files'].items():
        path = ROOT / 'official_source' / relative
        content = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if blob != identity['git_blob_sha'] or sha256(path) != identity['sha256']:
            raise ValueError(f'official source changed: {relative}')
    files = [ROOT / name for name in ('aligned_engine.py', 'aligned_data.py', 'aligned_runtime.py', 'aligned_runner.py', 'run.py')]
    files += sorted((ROOT / 'official_source' / 'aligned').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files if p.exists()}

def load_prepared(cell, seed):
    if aligned_data.PREPARED is None:
        aligned_data.configure()
    path = aligned_data.PREPARED / cell / f'seed_{seed}'
    report = json.loads((path / 'report.json').read_text())
    if report['status'] != 'PASS' or report['test_opened']:
        raise ValueError('preparation must pass without TEST access')
    for name, expected in report['artifacts_sha256'].items():
        if sha256(path / name) != expected:
            raise ValueError(f'prepared artifact changed: {name}')
    if sha256(ROOT / 'aligned_data.py') != report['adapter_sha256']:
        raise ValueError('adapter changed since preparation')
    # Revalidate current sealed TRAIN/VAL source against the prepared identities.
    live = load_dataset(cell, seed, include_test=False)
    if live['identity']['sha256'] != report['identity']['sha256']:
        raise ValueError('sealed TRAIN/VAL source changed')
    data = {'prepared': path, 'report': report,
            'gene_ids': json.loads((path / 'gene_ids.json').read_text())}
    for name in ('train', 'val'):
        data['X_' + name] = sparse.load_npz(path / f'X_{name}.npz').toarray().astype(np.float32)
        for key in ('Y', 'mask', 'records', 'row_target', 'row_gene', 'target_indices'):
            data[f'{key}_{name}'] = np.load(path / f'{key}_{name}.npy', allow_pickle=False)
    return data

def label_weights(deduction, labels, mask, P, N, go_count):
    """Paper E.3, restricted to observed TRAIN entries for agreement."""
    deduction, labels, mask = np.asarray(deduction), np.asarray(labels), np.asarray(mask, dtype=bool)
    observed = mask.sum(0)
    accuracy = np.divide(((deduction == labels) & mask).sum(0), observed,
                         out=np.zeros(labels.shape[1], dtype=np.float64), where=observed > 0)
    incoming = np.asarray((P + N).sum(axis=0)).ravel() / (2 * labels.shape[1])
    go_count = np.asarray(go_count, dtype=np.float64)
    go = go_count / go_count.max() if go_count.max() else np.zeros_like(go_count)
    weights = np.clip(.5 * accuracy + .2 * incoming + .3 * go, 0, 1).astype(np.float32)
    return weights, {'formula': 'clip(.5*TRAIN_agreement+.2*signed_in_degree/(2*n)+.3*GO/maxGO,0,1)',
        'unobserved_gene_agreement': 0, 'genes_without_observed_train_labels': int((observed == 0).sum()),
        'min': float(weights.min()), 'max': float(weights.max()), 'mean': float(weights.mean()),
        'sha256': hashlib.sha256(weights.tobytes()).hexdigest(), 'test_used': False, 'val_used': False}

def build_engine(data, device='cuda', batch_size=64):
    p = data['prepared']
    kb = RegulatoryKB(str(p / 'P.npz'), str(p / 'N.npz'), device=device)
    kb.closure_(T=5, closure_type='weighted')
    with torch.no_grad():
        deductions = torch.cat([kb.deduce(torch.as_tensor(x, device=device)).cpu()
            for x in np.array_split(data['X_train'], max(1, (len(data['X_train']) + batch_size - 1) // batch_size))]).numpy()
    weights, report = label_weights(deductions, data['Y_train'], data['mask_train'],
        sparse.load_npz(p / 'P.npz'), sparse.load_npz(p / 'N.npz'), np.load(p / 'GO_count.npy'))
    n = len(data['gene_ids'])
    engine = AlignedEngine(n, n, kb, weights, device=device, batch_size=batch_size)
    engine.set_data(data['X_train'], data['Y_train'], data['mask_train'])
    return engine, report
