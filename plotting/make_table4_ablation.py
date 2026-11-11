"""Recompute Table IV from generated, pair-aligned held-out predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / 'src')]
from experiments.ablation import ARMS, SEEDS, run_directory, work_directory
from jev_a4.core.train import sha256
from jev_a4.evaluation import evaluate

METRICS = ('macro_f1', 'balanced_accuracy', 'up_precision', 'up_recall', 'up_f1',
           'same_precision', 'same_recall', 'same_f1', 'down_precision', 'down_recall',
           'down_f1', 'within_gene_change_auc', 'within_gene_polarity_auc')
TABLE_ARMS = (*ARMS, 'TARGET_PERMUTATION')
DISPLAY = {'NO_JEV': 'NO_Jev', 'TARGET_PERMUTATION': 'Target permutation'}


def collect(work, cell, seeds, main_run_root=None, allow_incomplete=False):
    rows, missing = [], []
    for arm in TABLE_ARMS:
        for seed in seeds:
            directory = run_directory(work, arm, cell, seed, main_run_root)
            predictions = directory / 'test_predictions.npz'
            report_path = directory / 'test_report.json'
            if not predictions.is_file() or not report_path.is_file():
                missing.append(dict(arm=arm, cell=cell, seed=seed))
                continue
            report = json.loads(report_path.read_text())
            if report.get('status') != 'PASS' or report.get('test_predictions_sha256') != sha256(predictions):
                raise ValueError('Prediction file is not the report-sealed output: ' + str(directory))
            source = run_directory(work, 'S1Pert' if arm == 'TARGET_PERMUTATION' else arm,
                                   cell, seed, main_run_root)
            seal = json.loads((source / 'convergence.json').read_text())
            checkpoint = seal['best_checkpoint_sha256']
            if (seal.get('status') != 'CONVERGED' or seal.get('dataset') != cell
                    or int(seal.get('seed', -1)) != seed or sha256(source / 'best.pt') != checkpoint
                    or report.get('checkpoint_sha256') != checkpoint):
                raise ValueError('Evaluation checkpoint seal mismatch')
            records = np.load(Path(work) / 'data' / cell / f'seed_{seed}/test/records.npy', allow_pickle=False)
            rows.append(dict(arm=arm, cell=cell, seed=seed, prediction_sha256=sha256(predictions),
                             **evaluate(records, predictions)))
    if missing and not allow_incomplete:
        raise FileNotFoundError('Missing experiment outputs; run ablation train/permutation first: '
                                + json.dumps(missing))
    return rows, missing


def summarize(rows, expected_seeds):
    keys = [(row['arm'], row['cell'], int(row['seed'])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate arm/cell/split result')
    summary = []
    for arm in TABLE_ARMS:
        group = [row for row in rows if row['arm'] == arm]
        for metric in METRICS:
            values = np.asarray([row[metric] for row in group if row.get(metric) is not None], float)
            if not np.isfinite(values).all():
                raise ValueError('Non-finite experiment metric')
            summary.append(dict(arm=arm, metric=metric, n_splits=len(values), expected_splits=len(expected_seeds),
                mean=float(values.mean()) if len(values) else None,
                sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None))
    return summary


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_table(summary):
    columns = [('Macro-F1', 'macro_f1'), ('BA', 'balanced_accuracy'),
               ('Up-F1', 'up_f1'), ('Same-F1', 'same_f1'), ('Down-F1', 'down_f1'),
               ('Chg AUC', 'within_gene_change_auc'), ('Pol AUC', 'within_gene_polarity_auc')]
    index = {(row['arm'], row['metric']): row for row in summary}
    result = []
    for arm in TABLE_ARMS:
        row = {'Variant': DISPLAY.get(arm, arm)}
        for title, metric in columns:
            value = index[(arm, metric)]
            if value['mean'] is None:
                text = '--'
            elif value['sample_sd'] is None:
                text = f"{value['mean']:.4f} (n=1)"
            else:
                text = f"{value['mean']:.4f} ± {value['sample_sd']:.4f}"
            row[title] = text
        result.append(row)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--cell', default='k562_essential')
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--main-run-root', type=Path)
    p.add_argument('--allow-incomplete', action='store_true', help='Explicit diagnostic subset, not the full paper table')
    a = p.parse_args(argv)
    work = work_directory(a.work_dir)
    if len(set(a.seeds)) != len(a.seeds):
        p.error('Duplicate seeds are not allowed')
    rows, missing = collect(work, a.cell, a.seeds, a.main_run_root, a.allow_incomplete)
    summary = summarize(rows, a.seeds)
    table = render_table(summary)
    out = a.out or work / 'paper/table4'
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'per_split.csv', rows)
    write_csv(out / 'summary.csv', summary)
    write_csv(out / 'table4.csv', table)
    escape = lambda value: str(value).replace('_', r'\_').replace('±', r'$\pm$')
    fields = list(table[0])
    lines = [r'\begin{tabular}{l' + 'c' * (len(fields)-1) + '}', r'\toprule',
             ' & '.join(map(escape, fields)) + r' \\', r'\midrule']
    lines += [' & '.join(escape(row[key]) for key in fields) + r' \\' for row in table]
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'table4.tex').write_text('\n'.join(lines) + '\n')
    complete = not missing and tuple(a.seeds) == SEEDS and a.cell == 'k562_essential'
    (out / 'manifest.json').write_text(json.dumps(dict(status='PASS', complete_paper_matrix=complete,
        computed_from='generated held-out prediction NPZs', standard_deviation='sample; ddof=1',
        seeds=a.seeds, cell=a.cell, missing=missing), indent=2) + '\n')
    print(json.dumps(dict(status='PASS', output=str(out), complete_paper_matrix=complete)))


if __name__ == '__main__':
    main()
