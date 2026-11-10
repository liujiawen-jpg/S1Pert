"""Sealed data, public knowledge and shared metrics. Never read old predictions."""
from pathlib import Path
from collections import defaultdict, Counter
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / 'code/config.json').read_text())
LABELS = ('up', 'same', 'down')
CELL_TEXT = {
    'hepg2': 'HepG2 human hepatocellular carcinoma cells',
    'jurkat': 'Jurkat human T lymphocyte leukemia cells',
    'rpe1': 'RPE1 hTERT-immortalized retinal pigment epithelial cells',
    'k562_essential': 'K562 human myeloid leukemia cells (essential-gene screen)',
}

def relocated(path):
    """Use explicitly configured portable paths."""
    return Path(path)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Data:
    def __init__(self, cell, seed):
        self.cell, self.seed = cell, int(seed)
        self.base = Path(CFG['a4_root']) / 'data' / cell
        self.meta = read(self.base / 'dataset.json')
        self.seed_dir = self.base / f'seed_{seed}'
        self.manifest = read(self.seed_dir / 'manifest.json')
        self.preparation = read(self.base / 'preparation_report.json')
        if self.preparation['status'] != 'PASS':
            raise ValueError('Reference preparation did not pass')
        if (self.manifest['status'] != 'PASS' or self.manifest['split_ratio'] != [7, 1, 2]
                or not self.manifest['exclude_self_pairs']):
            raise ValueError('Unsealed or incompatible reference data')
        self.partition = np.load(self.seed_dir / 'target_partition.npy', mmap_mode='r')
        self.fold = np.load(self.seed_dir / 'target_fold.npy', mmap_mode='r')
        self.t = self.meta['target_symbols']
        self.q = self.meta['query_symbols']
        if any(str(x).isdigit() for x in self.t + self.q):
            raise ValueError('Numeric gene symbols cannot be used for public lookup')
        self.rows = {}
        for path in (self.base / 'dataset.json', self.seed_dir / 'manifest.json',
                     self.seed_dir / 'target_partition.npy', self.seed_dir / 'target_fold.npy'):
            self.verify_artifact(path)

    def verify_artifact(self, path):
        identity = self.preparation['artifacts'][str(path.relative_to(self.base))]
        if path.stat().st_size != identity['size_bytes'] or sha(path) != identity['sha256']:
            raise ValueError('Sealed A4 artifact changed: ' + str(path))

    def records(self, part, *, final=False):
        if part == 'test' and not final:
            raise PermissionError('TEST locked until final evaluation')
        if part not in self.rows:
            path = self.seed_dir / part / 'records.npy'
            self.verify_artifact(path)
            rows = np.load(path, mmap_mode='r')
            if rows.dtype.names != ('target', 'query', 'label'):
                raise ValueError('Unexpected record schema')
            t, q = rows['target'], rows['query']
            if (np.any(t < 0) or np.any(t >= len(self.t)) or np.any(q < 0)
                    or np.any(q >= len(self.q)) or np.any(self.partition[t] != ('train', 'val', 'test').index(part))):
                raise ValueError('Target split or identity mismatch')
            if np.any(np.asarray(self.meta['target_ids'])[t] == np.asarray(self.meta['query_ids'])[q]):
                raise ValueError('Self pair is not allowed')
            if not np.isin(rows['label'], [0, 1, 2]).all():
                raise ValueError('Unresolved record entered supervised data')
            self.rows[part] = rows
        return self.rows[part]

    def facts(self, target, query, functions):
        def stats(i):
            if i < 0:
                return 'unavailable'
            return (f"mean={self.meta['control_mean'][i]:.5g}, "
                    f"variance={self.meta['control_variance'][i]:.5g}, "
                    f"detection fraction={self.meta['control_detection'][i]:.5g}")
        a, d = self.t[int(target)], self.q[int(query)]
        ambiguous = set(self.meta.get('duplicate_symbol_ids', {}))
        def public(gene):
            if gene in ambiguous:
                return 'Symbol maps to multiple measured gene IDs; no unambiguous function assigned.'
            return functions.get(gene, 'No concise public description available.')
        return (f"Cell: {CELL_TEXT[self.cell]}. Intervention: CRISPRi knockdown of {a}; "
                'intended target reduction, realized efficiency and time unknown.\n'
                f"Target {a} ({self.meta['target_ids'][int(target)]}) public function: {public(a)}\n"
                f"Query {d} ({self.meta['query_ids'][int(query)]}) public function: {public(d)}\n"
                'Untreated RNA: linear counts normalized to 10,000 per cell. '
                f"Target control: {stats(int(self.meta['target_query_indices'][int(target)]))}. "
                f"Query control: {stats(int(query))}.\n"
                f"Question: How does population-average RNA of {d} change after suppressing {a}?\n"
                'up: ratio >1.2; same: ratio in [1/1.2,1.2]; down: ratio <1/1.2. '
                'Missing knowledge does not imply same, and connectivity does not prove a detectable change.')

    def identity(self):
        return {str(p): sha(p) for p in (self.base / 'dataset.json', self.seed_dir / 'manifest.json',
                self.seed_dir / 'target_partition.npy', self.seed_dir / 'target_fold.npy',
                self.seed_dir / 'train/records.npy', self.seed_dir / 'val/records.npy')}


def functions():
    c = sqlite3.connect(f"file:{CFG['knowledge_db']}?mode=ro", uri=True)
    c.execute('PRAGMA query_only=ON')
    def public_only(action, table, *args):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table != 'genes' else sqlite3.SQLITE_OK
    c.set_authorizer(public_only)
    result = {}
    for symbol, protein, function in c.execute('SELECT symbol,protein_name,function_text FROM genes ORDER BY symbol,accession'):
        text = ' '.join(x.strip() for x in (protein or '', function or '') if x.strip())
        if len(text) > len(result.get(symbol, '')):
            result[symbol] = text[:320]
    c.close()
    return result


class Retrieval:
    """Official SUMMER retrieval priorities, with public KG-defined neighbors.

    Current response is not an input. For TRAIN probes exclude the ENTIRE
    target fold; held-out inputs retrieve from strict TRAIN records only.
    """
    def __init__(self, data):
        self.data = data
        self.train = data.records('train')
        self.terms, self.direct, self.postings = defaultdict(set), defaultdict(set), defaultdict(set)
        conn = sqlite3.connect(f"file:{CFG['public_edges_db']}?mode=ro", uri=True)
        for src, dst, source in conn.execute('SELECT subject,object,source FROM public_edges WHERE source IN (?,?,?)',
                                             ('GO', 'Reactome', 'STRING')):
            if source == 'STRING':
                self.direct[src].add(dst); self.direct[dst].add(src)
            else:
                self.terms[src].add(dst); self.postings[dst].add(src)
        conn.close()
        path = Path(CFG['perturbqa_source']) / 'examples/summer/prompts/retrieve.py'
        self.native = module(path, 'native_summer_retrieve')
        # Upstream snippet omits its numpy import and assumes defaultdict inputs.
        self.native.np = np
        self.indices = {}

    @lru_cache(maxsize=50000)
    def close(self, symbol, axis):
        allowed = set(self.data.t if axis == 'target' else self.data.q)
        scores = Counter()
        for term in self.terms.get(symbol, ()):
            for other in self.postings[term]:
                if other in allowed and other != symbol:
                    scores[other] += 1
        values = {g: n / max(1, len(self.terms[symbol]) + len(self.terms[g]) - n) for g, n in scores.items()}
        for g in self.direct.get(symbol, ()):
            if g in allowed and g != symbol:
                values[g] = values.get(g, 0.) + 1.
        return tuple(sorted(values, key=lambda g: (-values[g], g))[:CFG['summer']['close_neighbors']])

    def index(self, excluded_fold=None):
        if excluded_fold not in self.indices:
            seen, reverse, lookup = defaultdict(set), defaultdict(set), {}
            for row in self.train:
                t, q, y = map(int, row)
                if excluded_fold is not None and self.data.fold[t] == excluded_fold:
                    continue
                a, d = self.data.t[t], self.data.q[q]
                ambiguous = self.data.meta.get('duplicate_symbol_ids', {})
                if a in ambiguous or d in ambiguous:
                    continue  # Never collapse conflicting Ensembl identities by symbol.
                seen[a].add(d); reverse[d].add(a); lookup[a, d] = y
            self.indices[excluded_fold] = seen, reverse, lookup
        return self.indices[excluded_fold]

    def examples(self, target, query, part):
        fold = int(self.data.fold[target]) if part == 'train' else None
        seen, reverse, lookup = self.index(fold)
        a, d = self.data.t[target], self.data.q[query]
        # Upstream random seed is fixed per query, but isolate numpy RNG state.
        state = np.random.get_state()
        try:
            pairs = self.native.get_pairs(pert=a, gene=d,
                close_perts=self.close(a, 'target'), close_genes=self.close(d, 'query'),
                seen=seen, seen_gene=reverse, budget=CFG['summer']['retrieval_budget'], seed=self.data.seed)
        finally:
            np.random.set_state(state)
        pairs = list(dict.fromkeys(pairs))
        if (a, d) in pairs:
            raise ValueError('Current answer leaked into retrieved examples')
        return [{'target': p, 'query': g, 'label': LABELS[lookup[p, g]]} for p, g in pairs]


def stratified_candidates(rows, seed):
    spec = CFG['synthpert']
    n = min(len(rows), spec['candidate_max'], max(spec['candidate_min'], int(np.ceil(len(rows) * spec['candidate_fraction']))))
    rng = np.random.default_rng(seed)
    # Round-robin target/class groups to avoid selecting only common responses.
    groups = defaultdict(list)
    for i in rng.permutation(len(rows)):
        groups[int(rows['target'][i]), int(rows['label'][i])].append(int(i))
    keys = sorted(groups)
    rng.shuffle(keys)
    result = []
    while len(result) < n:
        active = []
        for key in keys:
            if groups[key]:
                result.append(groups[key].pop())
                active.append(key)
                if len(result) == n:
                    break
        keys = active
        if not keys:
            break
    return np.asarray(result, dtype=np.int64)


def parse_answer(text):
    matches = re.findall(r'<answer>\s*(up|same|down)\s*</answer>', text, flags=re.I)
    return LABELS.index(matches[0].lower()) if len(matches) == 1 else -1


def valid_trace(text, label):
    reason = re.search(r'<reason>(.*?)</reason>', text, flags=re.I | re.S)
    return bool(reason and len(reason.group(1).split()) >= 12 and parse_answer(text) == int(label))


def parse_grade(text):
    # Exactly one grade; a quoted word 'excellent' alone is not acceptance.
    matches = re.findall(r'<grade>\s*([1-5])\s*</grade>', text, flags=re.I)
    return int(matches[0]) if len(matches) == 1 else None


def metrics(rows, predictions):
    y, p = rows['label'], np.asarray(predictions)
    if len(y) != len(p) or not len(y):
        raise ValueError('All eligible rows, including invalid outputs, must be scored')
    cm = np.zeros((3, 4), dtype=np.int64)
    np.add.at(cm, (y, np.where(np.isin(p, [0, 1, 2]), p, 3)), 1)
    per_class = {}
    for i, name in enumerate(LABELS):
        tp, support, called = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        precision, recall = tp / max(called, 1), tp / max(support, 1)
        per_class[name] = dict(precision=precision, recall=recall,
            f1=2 * precision * recall / max(precision + recall, 1e-30), support=support, predicted_support=called)
    ct, cp = y != 1, np.isin(p, [0, 2])
    tp = int((ct & cp).sum()); precision = tp / max(1, int(cp.sum())); recall = tp / max(1, int(ct.sum()))
    out = dict(per_class=per_class, macro_f1=float(np.mean([v['f1'] for v in per_class.values()])),
        up_down_f1=(per_class['up']['f1'] + per_class['down']['f1']) / 2,
        balanced_accuracy=float(np.mean([v['recall'] for v in per_class.values() if v['support']])),
        accuracy=float(np.mean(y == p)), invalid_rate=float(np.mean(p < 0)), records=len(y),
        changed=dict(precision=precision, recall=recall, f1=2 * precision * recall / max(precision + recall, 1e-30)),
        confusion_matrix=cm.tolist(), confusion_columns=['up', 'same', 'down', 'invalid'],
        probability_metrics=None, probability_note='Native generated decisions; no invented probabilities or AUC')
    _, inverse, counts = np.unique(rows['target'], return_inverse=True, return_counts=True)
    wcm = np.zeros((3, 4), dtype=np.float64)
    np.add.at(wcm, (y, np.where(np.isin(p, [0, 1, 2]), p, 3)), 1. / counts[inverse])
    out['equal_target_weighted_confusion'] = wcm.tolist()
    return out


def resource_check():
    import psutil
    rss = psutil.Process().memory_info().rss
    if rss > CFG['max_rss_gib'] * (1 << 30):
        raise MemoryError('Own RSS exceeded bounded budget; no global cache purge')
    # Reclaimable file cache is NOT counted as irrecoverable cgroup memory.
    stats = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if limit != 'max':
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        reclaimable = int(stats.get('inactive_file', 0))
        if int(limit) - current + reclaimable < 2 * (1 << 30):
            raise MemoryError('Less than 2 GiB effective cgroup headroom after reclaimable cache')
    return {'rss_gib': rss / (1 << 30), 'unix': time.time()}
