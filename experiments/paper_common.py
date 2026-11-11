"""Portable statistics and file helpers shared by the paper experiments."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

METRICS = ('macro_f1', 'balanced_accuracy', 'up_precision', 'up_recall', 'up_f1',
           'same_precision', 'same_recall', 'same_f1', 'down_precision', 'down_recall',
           'down_f1', 'within_gene_change_auc', 'within_gene_polarity_auc')

def work_root(value=None):
    value = value or os.environ.get('S1PERT_ROOT')
    if not value:
        raise ValueError('Specify --work-dir or S1PERT_ROOT.')
    return Path(value).expanduser().resolve()

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def write_csv(path, rows):
    rows = list(rows)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            result.update(chunk)
    return result.hexdigest()

def relative_path(path, base):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()

def auc_binary(y, score):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); score = np.asarray(score, float)
    if len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError('Invalid ranking score axis or non-finite scores.')
    positive = int(y.sum()); negative = len(y) - positive
    if not positive or not negative:
        return None
    return float((rankdata(score)[y].sum() - positive*(positive+1)/2) / (positive*negative))

def classification(y, prediction, query=None, change=None, polarity=None):
    """Native three-class macro metrics; -1 predictions count as failures."""
    y = np.asarray(y, int); prediction = np.asarray(prediction, int)
    if y.shape != prediction.shape or y.ndim != 1:
        raise ValueError('Truth/prediction shape mismatch.')
    if not np.isin(y, [0, 1, 2]).all() or not np.isin(prediction, [-1, 0, 1, 2]).all():
        raise ValueError('Expected resolved truth 0/1/2 and prediction -1/0/1/2.')
    result = {key: None for key in METRICS}
    result.update(n=len(y), invalid_predictions=int((prediction == -1).sum()),
                  valid_change_queries=0, valid_polarity_queries=0)
    if not len(y):
        return result
    cm = np.bincount(y*4 + np.where(prediction < 0, 3, prediction), minlength=12).reshape(3, 4)
    actual = cm.sum(1); called = cm[:, :3].sum(0); tp = np.diag(cm[:, :3])
    precision = np.divide(tp, called, out=np.zeros(3), where=called > 0)
    recall = np.divide(tp, actual, out=np.zeros(3), where=actual > 0)
    f1 = np.divide(2*tp, actual+called, out=np.zeros(3), where=actual+called > 0)
    result.update(macro_f1=float(f1.mean()), balanced_accuracy=float(recall[actual > 0].mean()))
    for i, name in enumerate(('up', 'same', 'down')):
        result.update({name+'_precision':float(precision[i]), name+'_recall':float(recall[i]),
                       name+'_f1':float(f1[i]), name+'_support':int(actual[i])})
    if change is None or polarity is None:
        return result
    order = np.argsort(query, kind='stable')
    q = np.asarray(query)[order]; labels = y[order]
    c = np.asarray(change)[order]; p = np.asarray(polarity)[order]
    borders = np.r_[0, np.flatnonzero(q[1:] != q[:-1])+1, len(q)]
    changes, polarities = [], []
    for lo, hi in zip(borders[:-1], borders[1:]):
        value = auc_binary(labels[lo:hi] != 1, c[lo:hi])
        if value is not None:
            changes.append(value)
        mask = labels[lo:hi] != 1
        value = auc_binary(labels[lo:hi][mask] == 0, p[lo:hi][mask])
        if value is not None:
            polarities.append(value)
    result.update(within_gene_change_auc=float(np.mean(changes)) if changes else None,
                  within_gene_polarity_auc=float(np.mean(polarities)) if polarities else None,
                  valid_change_queries=len(changes), valid_polarity_queries=len(polarities))
    return result

def summarize(rows, reference='S1Pert'):
    """Sample SD, paired split-wise Wilcoxon, Holm over every registered comparator."""
    from scipy.stats import wilcoxon
    groups = {}
    seen = set()
    for row in rows:
        identity = (row['model'], row['cell'], int(row['seed']), row['regime'])
        if identity in seen:
            raise ValueError(f'Duplicate result: {identity}')
        seen.add(identity)
        groups.setdefault((row['model'], row['cell'], row['regime']), []).append(row)
        groups.setdefault((row['model'], 'ALL_CELLS', row['regime']), []).append(row)
    output = []
    for (model, cell, regime), items in groups.items():
        ref = {(r['cell'], int(r['seed'])): r for r in groups.get((reference, cell, regime), [])}
        for metric in METRICS:
            valid = [r for r in items if r.get(metric) is not None and np.isfinite(float(r[metric]))]
            values = [float(r[metric]) for r in valid]
            differences = [float(ref[(r['cell'], int(r['seed']))][metric])-float(r[metric])
                for r in valid if (r['cell'], int(r['seed'])) in ref
                and ref[(r['cell'], int(r['seed']))].get(metric) is not None]
            p = None
            if model != reference and differences:
                p = 1. if not np.any(differences) else float(wilcoxon(differences, method='auto').pvalue)
            output.append(dict(model=model, cell=cell, regime=regime, metric=metric,
                n_splits=len(values), mean=float(np.mean(values)) if values else None,
                std=float(np.std(values, ddof=1)) if len(values)>1 else None,
                paired_n=len(differences), wilcoxon_two_sided_p=p, wilcoxon_holm_p=None))
    families = {}
    for row in output:
        if row['wilcoxon_two_sided_p'] is not None:
            families.setdefault((row['cell'], row['regime'], row['metric']), []).append(row)
    for family in families.values():
        previous = 0.
        for i, row in enumerate(sorted(family, key=lambda r:r['wilcoxon_two_sided_p'])):
            previous = max(previous, min(1., (len(family)-i)*row['wilcoxon_two_sided_p']))
            row['wilcoxon_holm_p'] = previous
    return output
