# Data preparation and strict response labels

The portable path is public download → raw H5AD → factorized moments and
statistical calls → explicit strict-label bundle → independent S1Pert/A4
preparation for each cell line. Public function text and a frozen DECIDER v1
snapshot enter separately at representation encoding.
It never loads historical predictions or task-trained weights. The statistical
label code is packaged locally; no external source directory is imported.

## Verified public sources

The release manifest, `configs/downloads.json`, records the public URLs,
published checksums where available, expected sizes, and destination paths.
The sources were checked on 2026-11-10 using directory listings, repository
metadata, and HTTP headers. No full source matrix or pretrained weights were
downloaded during release validation.

| Input | Public source | Verification |
|---|---|---|
| HepG2 | [GEO GSE264667 original supplement](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/) | `GSE264667_hepg2_raw_singlecell_01.h5ad`, 5,614,460,941 bytes |
| Jurkat | [GEO GSE264667 original supplement](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/) | `GSE264667_jurkat_raw_singlecell_01.h5ad`, 9,366,490,264 bytes |
| RPE1 | [scPerturb Zenodo record 13350497](https://zenodo.org/records/13350497) | `ReplogleWeissman2022_rpe1.h5ad`, 1,236,886,900 bytes, published MD5 |
| K562 essential | [scPerturb Zenodo record 13350497](https://zenodo.org/records/13350497) | `ReplogleWeissman2022_K562_essential.h5ad`, 1,546,729,675 bytes, published MD5 |
| Public gene functions | [UniProt API](https://www.uniprot.org/help/api_queries) | Reviewed human entries, accession/primary gene/protein name/function fields |
| Frozen encoder | [Mapika/decider-4b v1](https://huggingface.co/Mapika/decider-4b/tree/c23ab4d2483e7c6a93484463e9afe1688b29bedd) | Immutable commit `c23ab4d2483e7c6a93484463e9afe1688b29bedd`, weight SHA256 matches configuration |

The two GEO originals match the source sizes used by the paper. The similarly
named Nadig files in Arc Institute's Hugging Face mirror have different bytes
and sizes; the manifest therefore uses GEO. K562 genome-wide and normalized
Replogle files are different inputs and must not replace the listed files.
All four matrices total about 17.8 GB before derived stores and caches.

The downloader has no third-party dependencies. It requires explicit selection,
downloads into `.part` files, validates sizes and every supplied checksum, and
publishes complete files atomically without replacing an existing destination.
Interrupted downloads can resume only when their source identity and HTTP range
response agree. A changed server validator or an ignored range causes a stop.
Each completed file has a `.download.json` receipt with local SHA256, retrieval
time, and server metadata. Existing files are checked again before reuse.
Without a published digest, the receipt checks subsequent local consistency;
it does not establish independent publisher authentication. GEO does not expose
a checksum in the verified listing. Zenodo supplies MD5; Hugging Face supplies
Git blob SHA1 for small files and SHA256 for large files.

The UniProt export is live: a new download can contain newer annotations than
the paper. Save its receipt and use the same export across comparative runs.
Exact historical text reproduction requires the original public annotation
snapshot; the release does not claim that the live export recreates its bytes.
The importer accepts UniProt's actual TSV headers, uses primary gene symbols,
omits entries with no primary symbol, and keeps public function text only.
The configured SQLite file is built locally from this TSV.

If a host is unavailable, obtain the exact named file from its linked record and
verify its size/checksum before using it at the configured path. No substitute
dataset is selected automatically. A manually supplied GEO file lacks a downloader
receipt and will not be silently adopted by the downloader; after independent
verification it can be passed directly to preprocessing. For a stale `.lock`
left by a terminated process, first establish that no downloader is running,
then remove only that file. Preserve mismatched files for inspection and choose
a fresh destination or resolve the source mismatch explicitly.

## Commands

Install the package and its data dependencies as described in the README.
Run these commands from the repository root. The example download destinations
already agree with the preprocessing, label, and A4 configurations. Paths inside a
configuration are relative to that JSON file, not the current working directory.
Environment variables in configured paths are expanded; unset variables fail.
Explicit command-line paths are relative to the current working directory.

```bash
python scripts/download_assets.py --list
python scripts/download_assets.py --group data --dry-run

# These commands actually download the selected public inputs.
python scripts/download_assets.py --group data
python scripts/download_assets.py --group annotations
python -m jev_a4.knowledge --tsv assets/uniprot_reviewed_human.tsv --out assets/public_functions.sqlite

s1pert preprocess --config configs/preprocess.example.json
s1pert labels --config configs/labels.example.json
s1pert validate-labels --source workspace/preprocessed --bundle workspace/strict_labels
s1pert prepare --config configs/a4.example.json --dataset hepg2 --out workspace/a4/data/hepg2

# Separate explicit opt-in: includes 8.4 GB frozen encoder weights.
python scripts/download_assets.py --group backbone --dry-run
python scripts/download_assets.py --group backbone --include-weights
```

`--list` and `--dry-run` are offline and create no directories. `--group` and
`--asset` are repeatable. `--root` changes the asset destination root only;
update downstream configuration paths if using it. Weights are never fetched
without `--include-weights`. No downloader command starts training. The downloaded
snapshot includes the exact `decider/model.py`, `decider/prompt.py`, package
initializer, tokenizer, and configurations required by the frozen loader;
`decider_source`, `tokenizer_snapshot`, and `decider_snapshot` point to that
same `assets/DECIDER-4b-v1` directory.

Repeat `prepare` with each remaining dataset name, or use the four-dataset
pipeline documented in the README. Its encoding/training stages consume
`workspace/a4/data/<dataset>`, `assets/public_functions.sqlite`, and the frozen
encoder snapshot. GPU dependencies and runtime checks are described in the
README; CPU preprocessing and strict-label validation can run before acquiring
weights. S1Pert task-specific parameters are initialized independently.

The equivalent module entrypoints are:

```bash
python -m jev_a4.data.preprocess --config configs/preprocess.example.json
python -m jev_a4.labels.build_resolution_bundle --source workspace/preprocessed --output workspace/strict_labels
python -m jev_a4.labels.validate_resolution_bundle --source workspace/preprocessed --bundle workspace/strict_labels
python -m jev_a4.core.prepare --config configs/a4.example.json --dataset hepg2 --out workspace/a4/data/hepg2
```

The source files also retain direct script entrypoints. For example,
`python src/jev_a4/core/prepare.py --config ... --dataset ... --out ...` works
from a source checkout with the declared dependencies installed.

Preprocessing accepts `--output` to override its configured destination. Label
building accepts `--source` and `--output` overrides, or a configuration with
`labels.source` and `labels.output`. Neither stage overwrites an existing output
directory. A failed preprocessing directory retains `status.json: FAILED` for
inspection and is rejected by the label builder. Use a new destination to retry.

## Input configuration

`configs/preprocess.example.json` declares the four independent A4 sources:

| Dataset | `kind` | Expected source |
|---|---|---|
| `hepg2` | `nadig` | Nadig HepG2 raw single-cell H5AD |
| `jurkat` | `nadig` | Nadig Jurkat raw single-cell H5AD |
| `rpe1` | `replogle` | Replogle RPE1 H5AD containing original counts |
| `k562_essential` | `replogle` | Replogle K562 essential H5AD containing original counts |

Each dataset entry requires `kind` and `raw`. A dataset name must be a single
directory name. `preprocess.output` is the new factorized root;
`preprocess.block_cells` defaults to 256 and must be between 1 and 1024.
This controls the Nadig row block only; it is not a cap on cells or genes.

Supply both Nadig inputs together to preserve the historical metadata-only
cross-source missing-ID resolution. Their expression values and response labels
are never shared as A4 inputs. A single supplied source is supported, but may
have fewer metadata aliases available and should not be claimed to reproduce
the historical two-source identity mapping automatically.

The A4 preparation configuration requires `data_preprocessed_root` and
`label_resolution_root`. Optional `raw_h5ad_paths: {dataset: path}` provides
relocated source annotation files for repairing numeric/missing symbols in
legacy preprocessed stores. A freshly built portable store already contains
decoded symbols and does not need these files for preparation.

## Raw H5AD contract

For Nadig and Replogle, `X` must be a dense HDF5 dataset with shape
`cells × measured genes`, containing finite nonnegative integer counts. Sparse
H5AD encodings and log-normalized `X` are not accepted by these adapters.
There is no implicit conversion during preprocessing. Every cell's measured count sum
must not exceed its original positive, finite UMI denominator.

| Source | Required `obs` columns | Required `var` columns |
|---|---|---|
| Nadig | `gene`, `gene_id`, unique `cell_barcode`, numeric `UMI_count` | unique `gene_id`, `gene_name` |
| Replogle | `perturbation`, numeric `UMI_count` or `UMI count` | unique `ensembl_id`, `gene_name` |

String annotations support plain strings, current `categories/codes` groups,
legacy HDF5 category references, and legacy `__categories` tables. Category
codes are validated; missing categorical values decode to the empty string.

Nadig controls must have both name and identity `non-targeting`. At least 20
controls are required. Explicit target IDs are retained; missing IDs use a unique
measured gene-symbol match or the namespace `SYMBOL:<name>`. Unique matching
target metadata across the supplied Nadig sources may resolve that namespace.
Ambiguous cross-source identity matches fail. Multi-target identifiers containing
`+`, `,`, `;`, or `|` are rejected. All declared cells and measured gene columns
are retained. Targets with fewer than 20 cells remain on the target axis and
receive unresolved labels.

Replogle controls are the original adapter's metadata values `control`,
`non-targeting`, or an empty perturbation annotation. This rule is source-specific
and should be audited before applying the adapter to another dataset. Only
perturbations whose symbols match exactly one measured gene ID are retained as
targets. Excluded symbols and the number of excluded cells are recorded. No
measured query gene is filtered out.

An optional `kind: adamson_audit` entry performs the original Adamson metadata
audit only. It expects `obs.perturbation` with encoded missing categories,
`obs.nperts`, `obs.UMI count`, `var.gene_symbol`, and `var.ensembl_id`. It produces
`AUDITED_NOT_INCLUDED`, never A4 training data or strict labels. It is not part of
the default four-source example.

## Moments and statistical labels

For each cell, linear normalized expression is `y = X / original_UMI × 10000`.
This is not log-normalization. Group means, sample variances (`ddof=1`), and
nonzero-cell counts are calculated for pooled controls and every retained
perturbation. Nadig uses the original float64 row-block grouped reductions;
Replogle retains the original HDF5 chunk-based column/row reduction. Output means
and variances are float32 and nonzero counts are int32; statistical tests cast
the stored moments back to float64, matching the historical preprocessing path.

Nadig reads `X` sequentially once. Its group accumulators occupy `O(targets ×
genes)` memory and can require several GB. Row blocking does not make these
accumulators constant-memory. Replogle accumulates a query-column block at a
time and writes memory-mapped arrays. Neither adapter reads the entire raw
cell-by-gene matrix into memory.

The packaged `labels/label_core.py` retains the original tests:

1. Ratio margin `r=1.2`, significance threshold `alpha=0.05`.
2. Up tests `E[y_t] > r E[y_c]`; Down tests
   `E[y_t] < E[y_c] / r` using scaled one-sided Welch tests.
3. Same requires both one-sided equivalence tests (TOST), with
   `p_same = max(p_lower, p_upper)` and a point estimate strictly inside the band.
4. Benjamini–Hochberg correction is applied jointly to all three claims for all
   measured genes in a condition (`3G` hypotheses). Untestable hypotheses enter
   this family as `p=1`.
5. Both groups need at least 20 cells and at least 10 nonzero cells per gene;
   means must be positive and both scaled Welch variances estimable.

Non-significance is never converted to Same. The source label array uses
Up=0, Same=1, Down=2, unresolved=-1. Reasons are: 1 insufficient cells,
2 low effective sample support, 3 zero/nonestimable mean, 4 degenerate variance,
5 insufficient margin evidence; reason 0 means a resolved strict call.

The strict-label stage preserves these calls and changes only the unresolved
storage convention to `hard_label=-100`, with `supervision_mask=False`.
`resolution_code` stores classes 0–2 and unresolved reasons 3–7.
`point_effect_region` describes the observed log2 fold-change region; it must
never be used as substitute supervision. All unresolved cases remain missing
labels, not a fourth prediction class. The builder records SHA256 provenance
and seals its output; the independent validator checks content and seals.

The pooled-cell Welch/TOST labels are approximations. They do not perform
independent biological-replicate inference, fit a negative-binomial model, or
establish a universal false-discovery guarantee under correlated cells. Strict
label coverage can differ substantially by target and query gene. Performance
on resolved pairs is conditional on that resolution process.

## Factorized output schema

Each dataset directory contains:

| Artifact | Shape / role |
|---|---|
| `gene_id.npy`, `gene_symbol.npy`, `genes.parquet` | query gene identity and display symbol |
| `group_id.npy`, `group_name.npy`, `group_n.npy`, `groups.parquet` | pooled control first, then target groups |
| `mean.npy`, `variance.npy`, `nonzero.npy` | `(groups including control, genes)` |
| `response_group_id.npy` | target identities aligned to response-label rows |
| `labels.npy`, `reason_codes.npy`, `effect_log2fc.npy`, `q_best.npy` | `(targets, genes)` |
| `manifest.json` | source identity, normalization, rules, counts, exclusions |

The `split` column of `groups.parquet` is retained historical 70/15/15 hash
metadata. It is not an A4 partition and is never consulted by A4 preparation.

## A4 target splits and legal model inputs

A4 splits the complete canonical target axis independently within each cell
line, before label filtering. It sorts stable identities, uses a seeded NumPy
permutation, and assigns `floor(0.7N)` TRAIN, `floor(0.1N)` VAL, and the remainder
TEST. Default seeds are **17, 29, 43, 61, 89**; the protocol requires five distinct
seeds. Targets with zero strict downstream labels remain in this split axis.
Ambiguous identities and two source targets mapping to one canonical identity
fail rather than being silently divided across partitions.

Only resolved downstream target–query pairs become records. Measured self-pairs
are excluded by the default paper configuration and saved separately. Query
numeric features are `log1p(control mean)`, `log1p(control variance)`, control
detection fraction, and measurement availability. Target numeric features use
the target gene's same control row if measurable. Perturbation means and
variances never become A4 covariates; measured controls provide biological
background only.

Response-tendency features use strict TRAIN labels from the same cell line.
TRAIN targets are assigned five folds; a TRAIN record excludes its entire
target fold from label counts. VAL/TEST records use all TRAIN counts. Query
counts receive Laplace smoothing with alpha=1. If query support is zero, only
the eligible same-cell TRAIN class counts provide fallback features. These are
inputs to the newly trained model, not a prediction added to its output.
Held-out labels are used solely for evaluation records. Preparation independently
replays fold exclusion and verifies that changing held-out label values does not
change input features. It also seals source identities and generated artifacts.

## Provenance and validation boundary

The small `data/nadig_moments.py` adapter preserves the paper's `inventory` and
`aggregate_moments` metadata and moment-aggregation rules. Only metadata
identity resolution and raw-count moments are reused;
old fitted priors, predictions, null-calibration outputs, and model code are not
part of this adapter. H5AD string decoding uses the packaged corrected decoder.
The release computes fresh raw moments and no longer requires an opaque old
moment cache. Existing audited factorized stores can instead enter directly at
the strict-label stage.

Tests cover small synthetic Nadig and Replogle H5AD inputs, agreement with direct
normalized means and sample variances, count-contract rejection, strict TOST
semantics, unresolved mapping, label-bundle validation, five whole-target splits,
and TRAIN-only cross-fit leakage guards. No production matrix, pretrained weight,
training run, or real-data score is exercised by these tests. Reproduction of
historical large-data results still requires a full run using the original
source files and the recorded protocol. Floating-point reduction order, row
block size, and corrected legacy metadata decoding can affect exact bytes and
borderline calls; identical scientific formulas alone do not certify identical
historical artifacts.

中文提醒：旧预测、旧任务权重、固定来源均值预测均不接入本流程；实测 control 仅作
合法背景输入。未解析标签不等于 Same，原始 `groups.parquet` 的旧划分也不是 A4 划分。
