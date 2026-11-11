"""Render Fig. 2 exclusively from newly generated fidelity/boundary outputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from data_prep.cell_pool import BINS, RULES, identity, require, work_root, write_json
from experiments.cell_pool import METRICS, SUBSETS, summarize_rows


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_inputs(directory):
    directory = Path(directory)
    fidelity = read_csv(directory / "table_tost_fidelity.csv")
    lookup = {}
    for row in fidelity:
        key = row["rule"], row["cell_count_bin"]
        require(key not in lookup, "duplicate fidelity row")
        lookup[key] = row
        total = int(row["b_de"])
        require(sum(int(row["b_de_a_"+name]) for name in ("up", "same", "down", "unresolved")) == total,
                "Pool-B DE fates do not sum to their denominator")
        require(int(row["false_same"]) == int(row["b_de_a_same"]), "false-Same count differs")
    require(set(lookup) == {(rule, bucket) for rule in RULES for bucket in BINS}, "missing fidelity bins/rules")
    for bucket in BINS:
        require(int(lookup[RULES[0], bucket]["b_de"]) == int(lookup[RULES[1], bucket]["b_de"]),
                "rule denominators differ")
    per_seed = read_csv(directory / "boundary_per_seed.csv")
    parsed = []
    for row in per_seed:
        parsed.append({**row, "seed": int(row["seed"]), "records": int(row["records"]),
                       **{m: float(row[m]) if row[m] else None for m in METRICS}})
    keyset = {(r["subset"], r["model"], r["seed"]) for r in parsed}
    require(len(keyset) == len(parsed), "duplicate boundary seed row")
    seeds = sorted({r["seed"] for r in parsed})
    require(bool(seeds) and keyset == {(s, m, seed) for s in SUBSETS for m in ("S1Pert", "P1") for seed in seeds},
            "boundary methods/subsets do not share split identities")
    summary = summarize_rows(parsed)
    saved = {(r["subset"], r["model"], r["metric"]): r for r in read_csv(directory / "table_boundary_generalization.csv")}
    require(len(saved) == len(summary), "boundary summary axes differ")
    for row in summary:
        reference = saved[row["subset"], row["model"], row["metric"]]
        for field in ("mean", "std", "mean_records"):
            value = float(reference[field]) if reference[field] else None
            require((value is None and row[field] is None) or
                    (value is not None and row[field] is not None and np.isclose(value, row[field], rtol=1e-10, atol=1e-12)),
                    "boundary summary does not match per-seed metrics")
        require(int(reference["n"]) == row["n"], "boundary split count differs")
    return lookup, summary, seeds


def plot(directory, output=None, expected_reference_de=None, font_family="Times New Roman",
         font_fallback="DejaVu Serif"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotting.paper_plot_common import setup_font
    directory = Path(directory)
    output = Path(output) if output else directory
    fidelity, summary, seeds = load_inputs(directory)
    total = int(fidelity[RULES[1], "all"]["b_de"])
    require(total > 0, "no Pool-B reference DE pairs")
    if expected_reference_de is not None:
        require(total == expected_reference_de, "optional reference-DE count assertion failed")
    summary = {(r["subset"], r["model"], r["metric"]): r for r in summary}
    font = setup_font(font_family, font_fallback)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13.6, 5.0), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.35, 2.0))
    axa, axb = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    cgrid = grid[2].subgridspec(2, 1)
    axc = [fig.add_subplot(cgrid[i]) for i in range(2)]
    colors = {"DE": "#477ca8", "Same": "#bd5f4f", "Unresolved": "#c8ced5"}
    for index, rule in enumerate((RULES[1], RULES[0])):
        row = fidelity[rule, "all"]
        values = [("DE", int(row["b_de_a_up"])+int(row["b_de_a_down"])),
                  ("Same", int(row["b_de_a_same"])), ("Unresolved", int(row["b_de_a_unresolved"]))]
        bottom = 0.0
        for category, count in values:
            height = 100*count/total
            axa.bar(index, height, bottom=bottom, width=.62, color=colors[category],
                    label=category if index == 0 else None, edgecolor="white", linewidth=.5)
            if height >= 7:
                axa.text(index, bottom+height/2, f"{count:,}\n({height:.1f}%)", ha="center", va="center", fontsize=8)
            bottom += height
    axa.set(xticks=(0, 1), xticklabels=("S1Pert\nTOST", "Naive\nnon-significance"), ylim=(0, 105),
            ylabel="Pool-A decision (%)", title=f"(a) Fate of Pool-B DE\n{total:,} reference pairs")
    axa.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=3, frameon=False, fontsize=8)
    positions = np.arange(len(BINS))
    for rule, name, color, offset in ((RULES[1], "S1Pert TOST", "#477ca8", -.12),
                                     (RULES[0], "Naive", "#bd5f4f", .12)):
        values = [100*float(fidelity[rule, bucket]["resolved_concordance"])
                  if fidelity[rule, bucket]["resolved_concordance"] else np.nan for bucket in BINS]
        axb.plot(positions+offset, values, marker="o", markersize=4, linewidth=1.2, color=color, label=name)
    axb.set(xticks=positions, xticklabels=("All ≥20", "20–39", "40–79", "80–149", "≥150"),
            ylabel="Concordance among jointly resolved (%)", title="(b) Cell-pool label concordance", ylim=(0, 103))
    axb.tick_params(axis="x", rotation=25)
    axb.grid(axis="y", color=".9", linewidth=.5)
    axb.legend(frameon=False, loc="lower right")
    names = ("Macro-F1", "Bal. acc.", "Up-F1", "Down-F1", "Same-F1")
    for panel, subset, title in zip(axc, SUBSETS, ("A-unresolved / B-resolved", "+ original full-pool unresolved")):
        mean_records = summary[subset, "S1Pert", METRICS[0]]["mean_records"]
        for model, offset, color in (("P1", -.18, "#a7adb3"), ("S1Pert", .18, "#477ca8")):
            entries = [summary[subset, model, metric] for metric in METRICS]
            values = [100*r["mean"] if r["mean"] is not None else np.nan for r in entries]
            errors = [100*r["std"] if r["std"] is not None else 0 for r in entries]
            panel.bar(positions+offset, values, yerr=errors, capsize=2, width=.34, color=color, label=model,
                      error_kw={"linewidth": .8})
        panel.set(xticks=positions, xticklabels=names, ylim=(0, 105), ylabel="Score (%)",
                  title=f"{title} · mean n={mean_records:,.0f}")
        panel.grid(axis="y", color=".92", linewidth=.5)
        panel.set_axisbelow(True)
    axc[0].set_title("(c) Frozen-model boundary performance\n" + axc[0].get_title())
    axc[0].legend(frameon=False, ncol=2, loc="upper right")
    output.mkdir(parents=True, exist_ok=True)
    names = []
    for stem in ("fig_label_boundary", "fig2_label_boundary"):
        for extension in ("pdf", "png"):
            name = f"{stem}.{extension}"
            fig.savefig(output / name, dpi=300, bbox_inches="tight")
            names.append(name)
    plt.close(fig)
    write_json(output / "fig2_audit.json", {"status": "PASS", "reference_de_pairs": total, "seeds": seeds,
        "inputs": {name: identity(directory / name) for name in
            ("table_tost_fidelity.csv", "boundary_per_seed.csv", "table_boundary_generalization.csv")},
        "errors": "sample standard deviation across repeated target holdouts (ddof=1)",
        "counts": "all annotations derived from supplied result files", "font_family": font["selected"],
        "font": font, "paper_filename": "fig_label_boundary.pdf",
        "compatibility_alias": "fig2_label_boundary",
        "outputs": {name: identity(output / name) for name in names}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-reference-de", type=int)
    parser.add_argument("--font", "--font-family", dest="font_family", default="Times New Roman")
    parser.add_argument("--font-fallback", default="DejaVu Serif")
    args = parser.parse_args(argv)
    root = work_root(args.work_dir)
    plot(args.input or root / "paper/fig2", args.output, args.expected_reference_de,
         args.font_family, args.font_fallback)


if __name__ == "__main__":
    main()
