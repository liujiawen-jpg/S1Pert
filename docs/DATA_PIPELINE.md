# Data preparation and strict response labels

The portable pipeline runs in this order: public download → raw H5AD → factorized moments and statistical calls → explicit strict-label bundle → independent S1Pert/A4 preparation for each cell line. Representation encoding separately loads public function text and a frozen DECIDER v1 snapshot. The pipeline never loads historical predictions or task-trained weights. Statistical label code is packaged locally and does not import an external source directory.

## Verified public sources

The release manifest at `configs/downloads.json` records public URLs, available publisher checksums, expected sizes and destination paths. Source verification on 2026-11-10 used directory listings, repository metadata and HTTP headers. Release validation did not download a complete source matrix or any pretrained weights.

| Input | Public source | Verification |
|---|---|---|
| HepG2 | [GEO GSE264667 original supplement](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/) | `GSE264667_hepg2_raw_singlecell_01.h5ad`, 5,614,460,941 bytes |
| Jurkat | [GEO GSE264667 original supplement](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/) | `GSE264667_jurkat_raw_singlecell_01.h5ad`, 9,366,490,264 bytes |
| RPE1 | [scPerturb Zenodo record 13350497](https://zenodo.org/records/13350497) | `ReplogleWeissman2022_rpe1.h5ad`, 1,236,886,900 bytes, published MD5 |
| K562 essential | [scPerturb Zenodo record 13350497](https://zenodo.org/records/13350497) | `ReplogleWeissman2022_K562_essential.h5ad`, 1,546,729,675 bytes, published MD5 |
| Public gene functions | [UniProt API](https://www.uniprot.org/help/api_queries) | Reviewed human entries, accession/primary gene/protein name/function fields |
| Frozen encoder | [Mapika/decider-4b v1](https://huggingface.co/Mapika/decider-4b/tree/c23ab4d2483e7c6a93484463e9afe1688b29bedd) | Immutable commit `c23ab4d2483e7c6a93484463e9afe1688b29bedd`, weight SHA256 matches configuration |

The two GEO originals match the source sizes used by the paper. Similarly named Nadig files in the Arc Institute Hugging Face mirror differ in bytes and size, so the manifest uses GEO. K562 genome-wide and normalized Replogle files are separate inputs and cannot replace the listed files. The four matrices total about 17.8 GB before creation of derived stores and caches.

The downloader has no third-party dependencies. Every download requires explicit selection. Data is first written to a `.part` file. The downloader validates the expected size and every supplied checksum, then publishes the completed file atomically. It never replaces an existing destination.

An interrupted download resumes only if the source identity and HTTP range response still match. A changed server validator or ignored range stops the download. Each completed file receives a `.download.json` receipt with its local SHA256, retrieval time and server metadata. Existing files are validated again before reuse.

When no publisher digest is available, the receipt verifies later local consistency but does not provide independent publisher authentication. The verified GEO listing does not expose a checksum. Zenodo provides MD5. Hugging Face provides Git blob SHA1 for small files and SHA256 for large files.

The UniProt export is live, so a new download can contain annotations newer than the paper. Save the receipt and reuse the same export across comparative runs. Exact historical text reproduction requires the original public annotation snapshot. The release does not claim byte-level reproduction from a live export. The importer accepts the current UniProt TSV headers, uses primary gene symbols, omits entries without a primary symbol and retains only public function text. It builds the configured SQLite file locally from the TSV.

If a host is unavailable, obtain the exact named file from its linked record. Verify its size and checksum before placing it at the configured path. The pipeline does not select a substitute dataset automatically. A manually supplied GEO file has no downloader receipt, so the downloader will not adopt it silently. After independent verification, pass the file directly to preprocessing.

Before removing a stale `.lock` from a terminated process, confirm that no downloader is running. Remove only the stale lock file. Preserve mismatched files for inspection. Use a fresh destination or resolve the source mismatch explicitly.

## Commands

Install the package and data dependencies as described in the README, then run the following commands from the repository root. The example download destinations match the preprocessing, label and A4 configurations. Paths in a configuration file are resolved relative to that JSON file, not the current working directory. Configured paths expand environment variables and fail on unset variables. Explicit command-line paths are resolved relative to the current working directory.

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

`--list` and `--dry-run` work offline and create no directories. Both `--group` and `--asset` can be repeated. `--root` changes only the asset destination root. Update downstream configuration paths after changing it. The downloader never fetches weights without `--include-weights`, and no download command starts training.

The downloaded snapshot contains the exact `decider/model.py`, `decider/prompt.py`, package initializer, tokenizer and configurations required by the frozen loader. `decider_source`, `tokenizer_snapshot`, and `decider_snapshot` all point to the same `assets/DECIDER-4b-v1` directory.

Repeat `prepare` for each remaining dataset, or use the four-dataset pipeline from the README. Encoding and training consume `workspace/a4/data/<dataset>`, `assets/public_functions.sqlite`, and the frozen encoder snapshot. The README describes GPU dependencies and runtime checks. CPU preprocessing and strict-label validation can run before acquisition of the weights. S1Pert task-specific parameters are initialized independently.

The equivalent module entrypoints are:

```bash
python -m jev_a4.data.preprocess --config configs/preprocess.example.json
python -m jev_a4.labels.build_resolution_bundle --source workspace/preprocessed --output workspace/strict_labels
python -m jev_a4.labels.validate_resolution_bundle --source workspace/preprocessed --bundle workspace/strict_labels
python -m jev_a4.core.prepare --config configs/a4.example.json --dataset hepg2 --out workspace/a4/data/hepg2
```

The source files also provide direct script entry points. For example, run `python src/jev_a4/core/prepare.py --config ... --dataset ... --out ...` from a source checkout with the declared dependencies installed.

Use `--output` to override the configured preprocessing destination. Label building accepts `--source` and `--output` overrides, or a configuration containing `labels.source` and `labels.output`. Neither stage overwrites an existing output directory. A failed preprocessing directory retains `status.json: FAILED` for inspection. The label builder rejects that directory. Use a new destination for the retry.

## Input configuration

`configs/preprocess.example.json` declares the four independent A4 sources:

| Dataset | `kind` | Expected source |
|---|---|---|
| `hepg2` | `nadig` | Nadig HepG2 raw single-cell H5AD |
| `jurkat` | `nadig` | Nadig Jurkat raw single-cell H5AD |
| `rpe1` | `replogle` | Replogle RPE1 H5AD containing original counts |
| `k562_essential` | `replogle` | Replogle K562 essential H5AD containing original counts |

Each dataset entry requires `kind` and `raw`. The dataset name must be a single directory name. `preprocess.output` defines the new factorized root. `preprocess.block_cells` defaults to 256 and accepts values from 1 through 1024. This setting controls only the Nadig row block. It does not limit the number of cells or genes.

Supply both Nadig inputs to preserve the historical metadata-only, cross-source resolution of missing IDs. Their expression values and response labels are never shared as A4 inputs. A single source is supported but can provide fewer metadata aliases. It does not automatically reproduce the historical two-source identity mapping.

The A4 preparation configuration requires `data_preprocessed_root` and `label_resolution_root`. The optional `raw_h5ad_paths: {dataset: path}` field provides relocated source annotation files for repair of numeric or missing symbols in legacy preprocessed stores. A newly built portable store already contains decoded symbols and does not need these files during preparation.

## Raw H5AD contract

For Nadig and Replogle, `X` must be a dense HDF5 dataset with shape `cells × measured genes`. It must contain finite, nonnegative integer counts. These adapters reject sparse H5AD encodings and log-normalized `X`. Preprocessing does not perform an implicit conversion. The measured count sum for each cell must not exceed its original positive, finite UMI denominator.

| Source | Required `obs` columns | Required `var` columns |
|---|---|---|
| Nadig | `gene`, `gene_id`, unique `cell_barcode`, numeric `UMI_count` | unique `gene_id`, `gene_name` |
| Replogle | `perturbation`, numeric `UMI_count` or `UMI count` | unique `ensembl_id`, `gene_name` |

String annotations can use plain strings, current `categories/codes` groups, legacy HDF5 category references, or legacy `__categories` tables. Category codes are validated. Missing categorical values decode to the empty string.

Nadig controls must use `non-targeting` as both name and identity. At least 20 controls are required. Explicit target IDs are retained. A missing ID uses either a unique measured gene-symbol match or the `SYMBOL:<name>` namespace. Matching target metadata across the supplied Nadig sources can resolve this namespace when the match is unique. Ambiguous cross-source identity matches fail.

Multi-target identifiers containing `+`, `,`, `;`, or `|` are rejected. All declared cells and measured gene columns are retained. Targets with fewer than 20 cells remain on the target axis and receive unresolved labels.

Replogle controls use the original adapter metadata values `control`, `non-targeting`, or an empty perturbation annotation. This rule is source-specific. Audit it before applying the adapter to another dataset. A perturbation is retained as a target only if its symbol matches exactly one measured gene ID. The pipeline records excluded symbols and the number of excluded cells. It does not filter any measured query gene.

An optional `kind: adamson_audit` entry runs only the original Adamson metadata audit. It expects `obs.perturbation` with encoded missing categories, `obs.nperts`, `obs.UMI count`, `var.gene_symbol`, and `var.ensembl_id`. Its output is `AUDITED_NOT_INCLUDED`, never A4 training data or strict labels. The default four-source example does not include this entry.

## Moments and statistical labels

Each cell uses the linear normalization `y = X / original_UMI × 10000`. This is not log-normalization. The pipeline calculates group means, sample variances (`ddof=1`) and nonzero-cell counts for pooled controls and every retained perturbation. Nadig uses the original float64 grouped reductions over row blocks. Replogle retains the original HDF5 chunk-based column and row reduction. Output means and variances use float32. Nonzero counts use int32. Statistical tests cast the stored moments back to float64 to match the historical preprocessing path.

Nadig reads `X` sequentially once. Its group accumulators use `O(targets × genes)` memory and can require several GB. Row blocking does not make the accumulator memory constant. Replogle processes one query-column block at a time and writes memory-mapped arrays. Neither adapter loads the entire raw cell-by-gene matrix into memory.

The packaged `labels/label_core.py` retains the original tests:

1. Ratio margin `r=1.2`, significance threshold `alpha=0.05`.
2. Up tests use `E[y_t] > r E[y_c]`. Down tests use `E[y_t] < E[y_c] / r`. Both use scaled one-sided Welch tests.
3. Same requires both one-sided equivalence tests (TOST), with `p_same = max(p_lower, p_upper)` and a point estimate strictly inside the band.
4. Benjamini–Hochberg correction is applied jointly to all three claims for all measured genes in a condition (`3G` hypotheses). Untestable hypotheses enter this family as `p=1`.
5. Both groups need at least 20 cells and at least 10 nonzero cells per gene. Means must be positive, and both scaled Welch variances must be estimable.

Non-significance is never converted to Same. The source label array uses Up=0, Same=1, Down=2, and unresolved=-1. Reason 1 means insufficient cells. Reason 2 means low effective sample support. Reason 3 means a zero or nonestimable mean. Reason 4 means degenerate variance. Reason 5 means insufficient margin evidence. Reason 0 marks a resolved strict call.

The strict-label stage preserves these calls and changes only the unresolved storage convention to `hard_label=-100`, with `supervision_mask=False`. `resolution_code` stores classes 0–2 and unresolved reasons 3–7. `point_effect_region` describes the observed log2 fold-change region. It must never serve as substitute supervision. Every unresolved case remains a missing label, not a fourth prediction class. The builder records SHA256 provenance and seals its output. The independent validator checks both the content and seals.

The pooled-cell Welch/TOST labels are approximations. They do not perform inference over independent biological replicates, fit a negative-binomial model, or establish a universal false-discovery guarantee under correlated cells. Strict-label coverage can differ substantially across target and query genes. Performance on resolved pairs is conditional on the resolution process.

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

The `split` column in `groups.parquet` retains the historical 70/15/15 hash metadata. It is not an A4 partition, and A4 preparation never reads it.

## A4 target splits and legal model inputs

A4 splits the complete canonical target axis independently for each cell line before label filtering. It sorts stable identities, applies a seeded NumPy permutation, and assigns `floor(0.7N)` targets to TRAIN and `floor(0.1N)` to VAL. The remaining targets go to TEST. Default seeds are **17, 29, 43, 61, 89**. The protocol requires five distinct seeds. Targets with no strict downstream labels remain on the split axis. Ambiguous identities fail. Two source targets mapped to one canonical identity also fail instead of being split silently across partitions.

Only resolved downstream target–query pairs become records. The default paper configuration excludes measured self-pairs and saves them separately. Query numeric features are `log1p(control mean)`, `log1p(control variance)`, control detection fraction and measurement availability. Target numeric features use the same control row for the target gene when the gene is measurable. Perturbation means and variances never become A4 covariates. Measured controls provide only the biological background.

Response-tendency features use strict TRAIN labels from the same cell line. TRAIN targets are assigned to five folds. A TRAIN record excludes its entire target fold from the label counts. VAL and TEST records use all TRAIN counts. Query counts use Laplace smoothing with alpha=1. When query support is zero, eligible same-cell TRAIN class counts provide the only fallback features. These values are inputs to the newly trained model, not predictions added to its output.

Held-out labels are used only in evaluation records. Preparation independently replays fold exclusion and verifies that changes to held-out label values do not alter input features. It also seals source identities and generated artifacts.

## Provenance and validation boundary

The small `data/nadig_moments.py` adapter preserves the paper's `inventory` and `aggregate_moments` metadata and moment-aggregation rules. It reuses only metadata identity resolution and raw-count moments. Old fitted priors, predictions, null-calibration outputs and model code are not part of this adapter. H5AD string decoding uses the packaged corrected decoder. The release computes fresh raw moments and no longer requires the opaque legacy moment cache. Existing audited factorized stores can instead enter directly at the strict-label stage.

Tests cover small synthetic Nadig and Replogle H5AD inputs, agreement with direct normalized means and sample variances, count-contract rejection, strict TOST semantics, unresolved mapping, label-bundle validation, five whole-target splits, and TRAIN-only cross-fit leakage guards. These tests do not exercise a production matrix, pretrained weight, training run, or real-data score. Reproducing historical large-data results still requires a complete run with the original source files and recorded protocol. Floating-point reduction order, row-block size and corrected legacy metadata decoding can affect exact bytes and borderline calls. Matching scientific formulas alone does not certify byte-identical historical artifacts.

中文提醒：旧预测、旧任务权重、固定来源均值预测均不接入本流程。实测 control 仅作合法背景输入。未解析标签不等于 Same，原始 `groups.parquet` 的旧划分也不是 A4 划分。
