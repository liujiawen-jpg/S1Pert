"""Sealed A4 adapter for ALIGNED. CPU only; no task checkpoints or predictions.

Graph convention: P[source, destination]=1 for activation and N[...] for
inhibition. Unknown/conflicting signs are never converted to a signed edge.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
from scipy import sparse


A4_ROOT = PUBLIC_DB = GO_DB = PREPARED = None


def configure(work_dir=None, *, prepared_root=None, public_db=None, go_db=None):
    """Set paths explicitly; importing the module opens no dataset or database."""
    global A4_ROOT, PUBLIC_DB, GO_DB, PREPARED
    value = work_dir or os.environ.get('S1PERT_ROOT')
    if not value:
        raise ValueError('supply --work-dir or S1PERT_ROOT')
    A4_ROOT = Path(value).expanduser().resolve()
    PREPARED = Path(prepared_root) if prepared_root else A4_ROOT / 'baselines/aligned/prepared'
    PUBLIC_DB = Path(public_db) if public_db else A4_ROOT / 'assets/public_knowledge.sqlite'
    GO_DB = Path(go_db) if go_db else A4_ROOT / 'assets/public_go.sqlite'
    return A4_ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_dataset(cell, seed, include_test=False, *, a4_root=None):
    """Return full-axis signed interventions and masked signed response labels.

    X is CSR float32 (one target per row, value -1 for CRISPRi); Y is int64
    with up=+1, same=0, down=-1 and unresolved/unmeasured=-100. The partition
    rows include targets with zero strict response records. records_* retain
    the original A4 class codes. Only include_test=True opens TEST records.
    """
    started = time.monotonic()
    if a4_root is None and A4_ROOT is None:
        configure()
    base = Path(a4_root or A4_ROOT) / 'data' / str(cell)
    seed_dir = base / f'seed_{int(seed)}'
    meta = read_json(base / 'dataset.json')
    manifest = read_json(seed_dir / 'manifest.json')
    if (manifest.get('status') != 'PASS' or manifest.get('dataset') != cell
            or manifest.get('seed') != int(seed) or manifest.get('split_ratio') != [7, 1, 2]
            or manifest.get('exclude_self_pairs') is not True):
        raise ValueError('requires matching sealed target split and non-self protocol')
    identity_path = base / 'preparation_report.json'
    prepared_identity = read_json(identity_path)
    if prepared_identity.get('status') != 'PASS' or prepared_identity.get('exclude_self_pairs') is not True:
        raise ValueError('requires completed strict model-data preparation')
    artifacts = prepared_identity['artifacts']
    completed_identity = {name: artifacts[relative]['sha256'] for name, relative in {
        'dataset_json': 'dataset.json', 'partition': f'seed_{seed}/target_partition.npy',
        'train': f'seed_{seed}/train/records.npy', 'val': f'seed_{seed}/val/records.npy'}.items()}
    if include_test:
        completed_identity['test'] = artifacts[f'seed_{seed}/test/records.npy']['sha256']
    hashes = {'dataset_json': sha256(base / 'dataset.json'),
              'partition': sha256(seed_dir / 'target_partition.npy')}
    if any(completed_identity.get(k) != v for k, v in hashes.items()):
        raise ValueError('dataset/partition differs from strict preparation identity')
    partition = np.load(seed_dir / 'target_partition.npy', allow_pickle=False)
    # A4 code/prepare.py hashes array dtype+JSON shape+bytes, whereas GEARS
    # identity hashes the entire .npy file. Verify both distinct conventions.
    array_hash = hashlib.sha256(str(partition.dtype).encode() +
        json.dumps(partition.shape).encode() + np.ascontiguousarray(partition).tobytes()).hexdigest()
    if manifest.get('target_partition_sha256') != array_hash:
        raise ValueError('target partition differs from sealed manifest')
    targets, queries = list(meta['target_ids']), list(meta['query_ids'])
    if (partition.shape != (len(targets),) or not np.isin(partition, [0, 1, 2]).all()
            or len(set(targets)) != len(targets) or len(set(queries)) != len(queries)):
        raise ValueError('invalid sealed identity axes or partition')
    gene_ids = sorted(set(targets) | set(queries))
    gene_index = {gene: i for i, gene in enumerate(gene_ids)}
    target_gene = np.array([gene_index[x] for x in targets], dtype=np.int64)
    query_gene = np.array([gene_index[x] for x in queries], dtype=np.int64)
    result = {'gene_ids': gene_ids, 'gene_index': gene_index, 'meta': meta,
              'target_partition': partition, 'manifest': manifest}
    names = ['train', 'val'] + (['test'] if include_test else [])
    partitions = {'train': 0, 'val': 1, 'test': 2}
    for name in names:
        path = seed_dir / name / 'records.npy'
        hashes[name] = sha256(path)
        # Existing fit identity intentionally contains TRAIN/VAL only. TEST
        # hashes are recorded here only after the explicit final-evaluation gate.
        if (name != 'test' or name in completed_identity) and completed_identity.get(name) != hashes[name]:
            raise ValueError(f'{name} records differ from strict preparation identity')
        records = np.load(path, mmap_mode='r', allow_pickle=False)
        if records.dtype.names != ('target', 'query', 'label'):
            raise ValueError('unexpected sealed records dtype')
        t, q, labels = records['target'], records['query'], records['label']
        if (np.any(t < 0) or np.any(t >= len(targets)) or np.any(q < 0)
                or np.any(q >= len(queries)) or not np.isin(labels, [0, 1, 2]).all()):
            raise ValueError('record index/class outside sealed axes')
        if np.any(partition[t] != partitions[name]):
            raise ValueError('target leaked across sealed partitions')
        gt, gq = target_gene[t], query_gene[q]
        if np.any(gt == gq):
            raise ValueError('sealed comparison must exclude self responses')
        keys = t.astype(np.int64) * len(queries) + q
        if len(np.unique(keys)) != len(keys):
            raise ValueError('duplicate sealed target/query record')
        target_indices = np.flatnonzero(partition == partitions[name])
        local = np.full(len(targets), -1, dtype=np.int64)
        local[target_indices] = np.arange(len(target_indices))
        row_target = local[t]
        shape = (len(target_indices), len(gene_ids))
        x = sparse.csr_matrix((-np.ones(len(target_indices), dtype=np.float32),
            (np.arange(len(target_indices)), target_gene[target_indices])), shape=shape)
        y = np.full(shape, -100, dtype=np.int64)
        y[row_target, gq] = np.array([1, 0, -1], dtype=np.int64)[labels]
        result.update({f'X_{name}': x, f'Y_{name}': y, f'mask_{name}': y != -100,
            f'records_{name}': records, f'row_target_{name}': row_target,
            f'row_gene_{name}': gq, f'target_indices_{name}': target_indices})
    result['identity'] = {'cell': str(cell), 'seed': int(seed), 'sha256': hashes,
        'gene_ids_sha256': hashlib.sha256(json.dumps(gene_ids, separators=(',', ':')).encode()).hexdigest(),
        'manifest_sha256': sha256(seed_dir / 'manifest.json'),
        'reference_identity': identity_path.name, 'reference_identity_sha256': sha256(identity_path),
        'opened_partitions': names,
        'include_test': bool(include_test), 'label_mapping': {'0': 1, '1': 0, '2': -1},
        'unobserved_label': -100, 'intervention': 'CRISPRi=-1',
        'axis': 'lexicographically sorted canonical target/query identity union',
        'elapsed_seconds': time.monotonic() - started}
    return result


def public_connection(path, allowed_tables):
    """Read-only SQLite plus an authorizer denying empirical/response tables."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA query_only=ON')
    allowed = set(allowed_tables)
    def authorize(action, table, column, db, trigger):
        if action == sqlite3.SQLITE_READ and table not in allowed:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    connection.set_authorizer(authorize)
    return connection


def build_public_graph(meta, gene_ids, *, public_db=None, go_db=None):
    """Induce conservative public signed graph without reading response labels.

    Meta canonical ID->symbol must be one-to-one on the chosen gene axis and
    present as a canonical genes.symbol. Ambiguous symbols are left isolated.
    Exact accession->canonical symbol mappings are accepted only if unique.
    Explicit conflict or both signs for a canonical directed pair drops BOTH.
    Unknown/undirected edges and self-loops are excluded. Degree counts unique
    retained signed directed incidences (in+out). GO counts distinct non-NOT
    public gene_annotations terms across all namespaces, without propagation.
    """
    started = time.monotonic()
    public_db, go_db = public_db or PUBLIC_DB, go_db or GO_DB
    if public_db is None or go_db is None:
        raise ValueError('configure public signed knowledge and GO databases first')
    axis = {g: i for i, g in enumerate(gene_ids)}
    candidates = defaultdict(set)
    for id_key, symbol_key in [('target_ids', 'target_symbols'), ('query_ids', 'query_symbols')]:
        if len(meta[id_key]) != len(meta[symbol_key]):
            raise ValueError('canonical gene identity/symbol axes differ')
        for gene, symbol in zip(meta[id_key], meta[symbol_key]):
            if gene in axis and symbol:
                candidates[gene].add(str(symbol))
    reverse = defaultdict(set)
    for gene, symbols in candidates.items():
        for symbol in symbols:
            reverse[symbol].add(gene)
    ambiguous = sorted(s for s, ids in reverse.items() if len(ids) > 1)
    connection = public_connection(public_db, ['genes', 'public_edges'])
    accessions = defaultdict(set)
    canonical = set()
    for accession, symbol in connection.execute('SELECT accession,symbol FROM genes'):
        if symbol:
            canonical.add(symbol)
            accessions[accession].add(symbol)
    mapped = {}
    unmapped = {}
    for gene in gene_ids:
        symbols = candidates.get(gene, set())
        if len(symbols) != 1:
            unmapped[gene] = 'missing_or_conflicting_metadata_symbol'
        else:
            symbol = next(iter(symbols))
            if len(reverse[symbol]) != 1:
                unmapped[gene] = 'symbol_maps_to_multiple_canonical_ids'
            elif symbol not in canonical:
                unmapped[gene] = 'not_a_public_canonical_symbol'
            else:
                mapped[symbol] = axis[gene]
    def resolve(endpoint):
        # A literal canonical symbol wins only as an exact canonical identity.
        if endpoint in canonical:
            return mapped.get(endpoint)
        syms = accessions.get(endpoint, set())
        return mapped.get(next(iter(syms))) if len(syms) == 1 else None
    raw = Counter()
    pairs = defaultdict(set)
    pair_sources = defaultdict(set)
    for subject, obj, sign, directed, source in connection.execute(
            'SELECT subject,object,sign,directed,source FROM public_edges'):
        raw[f'sign={sign};directed={directed}'] += 1
        if directed != 1 or sign not in ('positive', 'negative', 'conflict'):
            continue
        i, j = resolve(subject), resolve(obj)
        if i is None or j is None or i == j:
            continue
        pairs[(i, j)].add(sign)
        pair_sources[(i, j)].add(source or 'unspecified')
    connection.close()
    edges = {'positive': [], 'negative': []}
    retained_sources = Counter()
    conflicts = 0
    for pair, signs in pairs.items():
        if 'conflict' in signs or len(signs) > 1:
            conflicts += 1
            continue
        edges[next(iter(signs))].append(pair)
        retained_sources.update(pair_sources[pair])
    def matrix(pairs):
        if not pairs:
            return sparse.csr_matrix((len(gene_ids), len(gene_ids)), dtype=np.float32)
        row, col = np.asarray(pairs, dtype=np.int64).T
        return sparse.csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)),
                                 shape=(len(gene_ids), len(gene_ids)))
    p, n = matrix(edges['positive']), matrix(edges['negative'])
    degree = np.asarray((p+n).sum(axis=0)).ravel() + np.asarray((p+n).sum(axis=1)).ravel()
    go = public_connection(go_db, ['gene_annotations', 'terms'])
    unique_terms = defaultdict(set)
    annotation_rows, not_rows = 0, 0
    for gene, term, qualifiers in go.execute('SELECT gene,term,qualifiers FROM gene_annotations'):
        if gene not in mapped:
            continue
        annotation_rows += 1
        try:
            qualifier_items = json.loads(qualifiers or '[]')
        except (ValueError, TypeError):
            qualifier_items = str(qualifiers or '').split('|')
        if isinstance(qualifier_items, str):
            qualifier_items = qualifier_items.split('|')
        if any(str(x).upper() == 'NOT' for x in qualifier_items):
            not_rows += 1
            continue
        if str(term).startswith('GO:'):
            unique_terms[gene].add(term)
    go.close()
    go_count = np.zeros(len(gene_ids), dtype=np.int64)
    for symbol, index in mapped.items():
        go_count[index] = len(unique_terms[symbol])
    report = {'status': 'PASS', 'nodes': len(gene_ids), 'mapped_nodes': len(mapped),
        'mapping_coverage': len(mapped)/max(len(gene_ids), 1), 'unmapped_gene_reasons': unmapped,
        'ambiguous_symbols': ambiguous, 'positive_edges': p.nnz, 'negative_edges': n.nnz,
        'conflicting_pairs_dropped': conflicts, 'isolated_nodes': int(np.count_nonzero(degree == 0)),
        'source_edge_sign_counts': dict(raw), 'retained_edge_sources': dict(retained_sources),
        'orientation': 'row=source,column=destination',
        'sign_rule': 'directed positive/negative only; explicit conflict or both signs drops both; unknown excluded',
        'mapping_rule': 'exact unambiguous metadata canonical ID/symbol and public canonical symbol/accession; no fuzzy aliases',
        'node_rule': 'full target/query identity union, no response-dependent selection',
        'GO_count_rule': 'distinct directly annotated GO terms across all namespaces; exclude NOT; no ontology propagation',
        'GO_candidate_rows': annotation_rows, 'GO_NOT_rows_excluded': not_rows,
        'degree_rule': 'retained signed directed unique in-degree plus out-degree',
        'public_db': str(public_db), 'go_db': str(go_db),
        'public_db_sha256': sha256(public_db), 'go_db_sha256': sha256(go_db),
        'response_tables_read': [], 'task_weights_read': False,
        'elapsed_seconds': time.monotonic() - started}
    return {'P': p, 'N': n, 'GO_count': go_count,
            'degree_count': degree.astype(np.int64), 'report': report}


def prepare(cell, seed, output=None):
    started = time.monotonic()
    data = load_dataset(cell, seed)
    graph = build_public_graph(data['meta'], data['gene_ids'])
    destination = Path(output) if output else PREPARED / cell / f'seed_{seed}'
    if destination.exists():
        raise FileExistsError('prepared output exists; use a new directory or validate it through train')
    destination.mkdir(parents=True)
    (destination / 'gene_ids.json').write_text(json.dumps(data['gene_ids']))
    for name in ('train', 'val'):
        sparse.save_npz(destination / f'X_{name}.npz', data[f'X_{name}'])
        for key in ('Y', 'mask', 'row_target', 'row_gene', 'target_indices', 'records'):
            np.save(destination / f'{key}_{name}.npy', data[f'{key}_{name}'], allow_pickle=False)
    for name in ('P', 'N'):
        sparse.save_npz(destination / f'{name}.npz', graph[name])
    for name in ('GO_count', 'degree_count'):
        np.save(destination / f'{name}.npy', graph[name], allow_pickle=False)
    report = {'status': 'PASS', 'identity': data['identity'], 'graph': graph['report'],
        'X_train_shape': list(data['X_train'].shape), 'X_val_shape': list(data['X_val'].shape),
        'train_records': len(data['records_train']), 'val_records': len(data['records_val']),
        'train_zero_label_targets': int(np.count_nonzero(~data['mask_train'].any(axis=1))),
        'test_opened': False, 'adapter_sha256': sha256(__file__),
        'elapsed_seconds': time.monotonic()-started}
    report['artifacts_sha256'] = {p.name: sha256(p) for p in destination.iterdir()
                                  if p.is_file() and p.name != 'report.json'}
    (destination / 'report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare'])
    parser.add_argument('--cell', required=True, choices=['hepg2','jurkat','rpe1','k562_essential'])
    parser.add_argument('--seed', type=int, default=17, choices=[17,29,43,61,89])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = prepare(args.cell, args.seed, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in ['identity','graph','artifacts_sha256']}, indent=2))
