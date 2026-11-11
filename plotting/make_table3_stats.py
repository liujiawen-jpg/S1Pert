"""Compute Table III and emit CSV, audited JSON and a LaTeX table."""
from __future__ import annotations

import argparse
from pathlib import Path

from data_prep.cell_pool import work_root
from data_prep.table3 import CELLS, generate


def render(rows):
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r"Cell & Targets & Genes & Candidates & Eligible & Up & Same & Down & Unresolved (\%) & DE/Same \\", r"\midrule"]
    for row in rows:
        values = [str(row["cell"]).replace("_", r"\_")] + [f"{row[key]:,}" for key in
                  ("targets", "genes", "total_candidate_pairs", "eligible_pairs", "up", "same", "down")]
        values += ["--" if row[key] is None else f"{row[key]:.2f}" for key in ("unresolved_percent", "de_to_same_ratio")]
        lines.append(" & ".join(values) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=CELLS)
    args = parser.parse_args(argv)
    root = work_root(args.work_dir)
    output = args.output or root / "paper/table3"
    rows = generate(root, output, args.cells)
    (output / "table_dataset_summary.tex").write_text(render(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
