"""Recompute Table II from pair-aligned raw predictions, then emit CSV and LaTeX."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments.paper import CELLS, SEEDS, METHODS, workspace
from jev_a4.evaluation import evaluate, summarize, write_csv

METRICS = ("macro_f1", "balanced_accuracy", "up_precision", "up_recall", "up_f1",
           "same_precision", "same_recall", "same_f1", "down_precision", "down_recall", "down_f1")
DISPLAY = dict(hepg2="HepG2", jurkat="Jurkat", rpe1="RPE1", k562_essential="K562")


def build(registry, output, *, methods=None, allow_partial=False):
    registry, output = Path(registry), Path(output)
    with registry.open() as f: entries = list(csv.DictReader(f))
    paper_methods = [m for m in METHODS if m != "P1"]
    methods = methods or paper_methods
    entries = [r for r in entries if r["method"] in methods]
    keys = [(r["method"], r["cell"], int(r["seed"])) for r in entries]
    if len(set(keys)) != len(keys): raise ValueError("Duplicate method/cell/seed")
    if any(c not in CELLS or s not in SEEDS for _, c, s in keys): raise ValueError("Unexpected cell/split seed")
    expected = {(m, c, s) for m in methods for c in CELLS for s in SEEDS}
    missing = sorted(expected - set(keys))
    if missing and not allow_partial: raise ValueError(f"Table II incomplete: {len(missing)} splits missing")
    rows = []
    for entry in entries:
        records = (registry.parent / entry["records"]).resolve(); pred = (registry.parent / entry["predictions"]).resolve()
        values = evaluate(np.load(records, mmap_mode="r", allow_pickle=False), pred)
        rows.append(dict(method=entry["method"], cell=entry["cell"], seed=int(entry["seed"]), **values))
    if not rows: raise ValueError("No raw predictions supplied")
    output.mkdir(parents=True, exist_ok=False)
    summary = summarize(rows)
    write_csv(output / "table_II_per_split.csv", rows); write_csv(output / "table_II_summary.csv", summary)
    lookup = {(r["method"], r["cell"], r["metric"]): r for r in summary}
    formatted = []
    for cell in CELLS:
        for method in methods:
            result = dict(dataset=DISPLAY[cell], method=method)
            for metric in METRICS:
                r = lookup.get((method, cell, metric))
                result[metric] = (f"{r['mean']:.3f} ± {r['sample_sd']:.3f}" if r and r["valid_splits"] == 5 else "-")
            formatted.append(result)
    write_csv(output / "table_II_formatted.csv", formatted)
    # No rounding-based ranks: highlight exact means, and share tied ranks.
    def latex_value(method, cell, metric):
        r = lookup.get((method, cell, metric))
        if not r or r["valid_splits"] != 5: return "--"
        values = sorted({lookup[(m, cell, metric)]["mean"] for m in methods
                         if (m, cell, metric) in lookup and lookup[(m, cell, metric)]["valid_splits"] == 5}, reverse=True)
        value = f"{r['mean']:.3f} \\pm {r['sample_sd']:.3f}"
        if r["mean"] == values[0]: value = r"\mathbf{" + value + "}"
        elif len(values) > 1 and r["mean"] == values[1]: value = r"\underline{" + value + "}"
        return "$" + value + "$"
    lines = [r"\begin{table*}[t]", r"\centering\scriptsize",
             r"\caption{Target-held-out classification (mean $\pm$ sample standard deviation over five splits).}",
             r"\begin{tabular}{llrrrrrrrrrrr}\toprule",
             r"Dataset & Method & Macro-F1 & BA & Up-P & Up-R & Up-F1 & Same-P & Same-R & Same-F1 & Down-P & Down-R & Down-F1 \\", r"\midrule"]
    for cell in CELLS:
        for method in methods:
            lines.append(" & ".join([DISPLAY[cell], method.replace("_", r"\_")] + [latex_value(method, cell, metric) for metric in METRICS]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines += [r"\end{tabular}", r"\end{table*}"]
    (output / "table_II.tex").write_text("\n".join(lines) + "\n")
    (output / "coverage.json").write_text(json.dumps(dict(complete=not missing,
        complete_paper_matrix=not missing and set(methods)==set(paper_methods), missing=missing,
        expected_splits=len(expected), present_splits=len(keys), ddof=1,
        repeated_target_holdouts_not_independent_folds=True, raw_predictions_recomputed=True), indent=2) + "\n")
    return rows, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--work-dir", type=Path)
    p.add_argument("--registry", type=Path); p.add_argument("--out", type=Path)
    p.add_argument("--methods", nargs="+"); p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args(argv)
    work = workspace(a.work_dir)
    build(a.registry or work / "paper/predictions.csv", a.out or work / "paper/table2",
          methods=a.methods, allow_partial=a.allow_partial)


if __name__ == "__main__": main()
