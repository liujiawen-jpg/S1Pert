# Table III and Fig. 2: label support and the cell-pool boundary

These stages compute the paper's statistics from prepared inputs. Table III
reads the strict label arrays. Fig. 2 starts from the original K562-essential
count matrix, creates the fixed disjoint A/B cell pools, adjudicates both pools,
and evaluates the already-trained S1Pert and its standalone P1 comparator.

Run commands from the repository root after installing the package. All stages
accept `--work-dir`; the equivalent environment setting is `S1PERT_ROOT`.

## Inputs and outputs

The work directory contains:

```text
raw/scPerturb_prepro/ReplogleWeissman2022_K562_essential.h5ad
preprocessed/{cell}/                 # linear moments and original strict labels
strict_labels/{cell}/                # hard_label.npy, resolution_code.npy
data/{cell}/                        # immutable identities and five target splits
cache/k562_essential/                # frozen shared/card/joint representations
runs/main/k562_essential/seed_{s}/   # selected, restored, tested S1Pert checkpoint
cell_pool/pools/                    # generated cell assignment, moments, A/B labels
cell_pool/prepared/seed_{s}/         # label-free keys and separate B references
cell_pool/predictions/seed_{s}/      # frozen S1Pert/P1 probabilities and seals
paper/table3/                       # derived CSV, JSON audit, LaTeX
paper/fig2/                         # fidelity, boundary metrics, PDF, PNG, audits
```

The main checkpoint directory requires `effective_config.json`, `best.pt`,
`convergence.json`, `restore_report.json`, `test_report.json`, and
`test_predictions.npz`. `--main-run-root` selects another parent containing
`k562_essential/seed_{s}`; `--model-dir` selects one exact model directory.
The original config/cache/code/checkpoint bindings must still validate. A moved
or modified historical checkpoint is not automatically declared equivalent to
a newly reproduced checkpoint.

## Stage 1 and Table III

The source preprocessing uses `X / original_UMI * 10,000` on the linear scale.
Group moments accumulate in float64 and are stored in float32 before testing.
Labels use a 1.2-fold margin, alpha 0.05, scaled Welch one-sided tests and the
TOST intersection-union p-value. Both groups need at least 20 cells and 10
nonzero observations per queried gene, with positive estimable means and
nondegenerate test variance. Class codes are Up=0, Same=1, Down=2; unresolved
is missing supervision.

**Ordering in the original implementation:** each target's BH family contains
all 3G hypotheses, including the self gene and genes with insufficient nonzero
support. Invalid mean/variance tests receive p=1. Cell/nonzero eligibility is
applied after BH; self pairs are removed when constructing model records and
reporting counts. Filtering support or self genes out before BH changes the
historical label protocol. The portable code preserves the original ordering.

```bash
python -m plotting.make_table3_stats --work-dir /path/to/work
```

The script derives all four cells' target/query counts, nonself candidate and
eligible pairs, Up/Same/Down counts, unresolved percentage and DE/Same ratio.
Unresolved percentage uses **all nonself candidates**, including low-support
and nonestimable pairs. Eligible pairs have resolution code 0, 1, 2 or 7.
Identity-matched self pairs are removed from every denominator and class count.
The audit records hashes of all input label arrays and identity metadata.

## Stage 6: fixed disjoint A/B adjudication

Preparation and encoding of all five K562 target splits must precede this
stage. The pool stage reads no task predictions or learned weights.

```bash
python -m data_prep.cell_pool build --work-dir /path/to/work
python -m data_prep.cell_pool prepare --work-dir /path/to/work
python -m data_prep.cell_pool fidelity --work-dir /path/to/work
```

The cell split uses seed 20261016. It is stratified by target/control group and
the recorded batch. Pool A receives nearest-integer n/3 cells in each group;
Pool B receives the remainder. Largest-remainder allocation distributes A
cells across batches, with seeded random ties and within-batch shuffling.
Each group has its own RNG, preserving assignments when another group's
eligibility changes. Target and control cells are mutually exclusive between
A and B, and the same cell assignment is reused across model seeds
17, 29, 43, 61 and 89.

The historical pool scope is the **union of the five fixed TEST target sets**,
plus control cells. Other target rows retain unresolved labels and zero pool
counts. Full pooled moments are reconstructed for all source groups and
checked against Stage-1 moments. Original UMI denominators, integer counts,
gene identities and frozen target-split hashes are validated. The release
retains A/B moment arrays, allowing fidelity to rerun without rereading raw X;
the historical implementation discarded target moments and later rebuilt A.

Both pools use the original 1.2/0.05/20/10/3G-BH label protocol. The comparison
rule copies the margin labels and calls Same whenever a pair is quality
eligible and both `q_up > 0.05` and `q_down > 0.05`. This naive rule does not
test equivalence.

Fidelity bins use Pool-A perturbed-cell counts: all evaluated targets with
at least 20 cells, 20–39, 40–79, 80–149 and at least 150. False invariance is
the fraction called Same among Pool-B DE pairs that satisfy Pool-A quality
eligibility and self exclusion. Concordance uses pairs resolved in both pools.
The output also records candidate and eligible coverage separately.

One historical denominator detail is retained: the `all` **candidate coverage**
denominator additionally includes selected groups with 1–19 A cells. The
reference-DE and concordance denominators still contain only evaluated groups
with at least 20 A cells. This does not affect the displayed Fig. 2a/b metrics.

All counts are derived from the arrays. Optional `--expected-cells` on `build`
and `--expected-reference-de` on `fidelity`/plotting assert a supplied audit
expectation; they do not supply a count to the calculation.

## Frozen S1Pert and P1 boundary inference

For each original target split, preparation selects nonself TEST pairs with
unresolved A and resolved B labels. `keys.npy` contains only target/query
indices; `records.npy` holds the separate B evaluation labels.
`original_unknown.npy` selects the nested subset also unresolved under the
original full-pool labels. Consequently `FULL_UNKNOWN_B_RESOLVED` is nested in
`A_UNRESOLVED_B_RESOLVED`; it is not all full-pool-unknown/B-resolved pairs.

```bash
for seed in 17 29 43 61 89; do
  python -m experiments.cell_pool predict --work-dir /path/to/work --seed "$seed"
done
python -m experiments.cell_pool evaluate --work-dir /path/to/work
python -m plotting.plot_fig2_label_boundary --work-dir /path/to/work
```

The inference command loads the converged, independently restored S1Pert head,
checks the original bindings, and freezes its weights. It reconstructs the
TRAIN-only query prior and verifies its numeric features and card identities
against every original TEST row. P1 uses those priors as an independent
comparator. Its predictions are never added to the S1Pert output.

For previously unseen structural pairs, the command loads only DECIDER's
frozen upper eight layers and final norm, combines the existing condition,
query and prior block-24 landmarks, and applies the same selected direct head.
It uses the historical CPU BF16 tail implementation. The complete generic
weight file is checksum-verified before selected tensors are loaded.

Eight deterministic original TEST rows provide an execution-parity check.
The historical predeclared tolerances are maximum absolute probability error
0.01 and mean absolute error 0.002 for CPU/GPU BF16 execution. Failure stops
inference. Matching existing TEST keys reuse the same checkpoint's sealed
probabilities; remaining keys are computed by the frozen model. `--parity-only`
produces a parity report but no evaluable prediction seal; use a separate
`--output` directory for that diagnostic.

Pool-B perturbed responses and labels are evaluation references only. The
frozen model retains its original full-pool untreated-control context inputs;
A/B-disjoint controls are used for label adjudication. This is a selected
technical cell-pool reproducibility proxy. It does not estimate performance
on every unresolved pair or provide independent biological replication.

## Plot and verification

Fig. 2a plots all derived Pool-A DE/Same/unresolved fates of Pool-B DE pairs.
Fig. 2b plots concordance over the five cell-count bins. Fig. 2c displays
Macro-F1, balanced accuracy and Up/Down/Same F1 for both nested subsets, with
sample standard deviation (`ddof=1`) across the repeated target holdouts.
The five target splits overlap, so their standard deviation describes split
variation. Plotting reads generated CSVs, checks summaries against per-seed
results and emits a source-hash audit. The paper output is
`fig_label_boundary.pdf` (plus PNG); `fig2_label_boundary` aliases are retained.
Times New Roman is preferred when installed, with DejaVu Serif as the default
fallback. `--font` (alias `--font-family`) and `--font-fallback` select installed
fonts; the audit records the requested and selected font. No font is bundled.

Dependencies are the package's data stack (NumPy, SciPy, pandas, PyArrow and
h5py), Matplotlib for plotting, and the pinned core PyTorch/Transformers/
safetensors environment for real frozen inference. No additional training
framework is needed for this stage.

```bash
python -m pytest -q tests/test_paper_cell_pool.py
```

The tests use tiny artificial count matrices and probabilities. They cover
exact/disjoint batch allocation, original-UMI linear moments, streamed pooled
reconstruction, strict/naive labels, nested/self-excluded subsets, Table III
denominators, label-free key validation, prediction seals, sample standard
deviations and plots with derived counts. They do not validate scientific
scores or run formal training.
