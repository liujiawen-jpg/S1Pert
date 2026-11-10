# Reproducibility and execution boundaries

S1Pert packages the A4 scientific workflow for review and new runs. The external assets listed below are required to reproduce the biological results. Follow the [README](../README.md) for the installed command interface and the [source inventory](../provenance/source_inventory.json) for provenance. The internal `jev_a4` module name, A4 configuration fields, and artifact schemas are retained for code compatibility; the public command is `s1pert`.

## What can be checked on a CPU

Small synthetic tests cover preparation, target splitting, TRAIN-only propensity construction, labeling, metrics, configuration, and the classification head when PyTorch is available. These checks do not load the 4B backbone or establish biological accuracy. The `train` module's synthetic smoke path also performs tiny optimization steps and writes a synthetic checkpoint; it is an execution test, not a trained biological model.

The release preparation did not run a real-data training experiment, download weights, or regenerate the historical semantic caches. Historical results are not evidence that a new machine or a newly resolved dependency environment has reproduced them.

## Dependencies: three distinct scopes

The package targets Python 3.11 or newer. The historical handoff records CPython 3.12.14. Recorded versions below come from the handoff's package metadata; they are not a new clean-install validation or a complete lockfile.

| Scope | Required packages | Historical versions recorded |
| --- | --- | --- |
| Array preparation and metrics | NumPy | 1.26.4 |
| HDF5 preprocessing, metadata, and label construction | h5py, pandas, pyarrow, SciPy, NumPy | 3.16.0, 2.3.3, 25.0.1, 1.17.1, 1.26.4 |
| Test runner | pytest | 9.1.1 |
| Synthetic CPU head checks and task-head training | PyTorch | 2.7.1+cu128 was installed historically; this is a CUDA build, not a CPU installation instruction |
| Frozen 4B encoding | PyTorch, Transformers, tokenizers, safetensors, flash-linear-attention/fla-core, Triton | 2.7.1+cu128, 5.17.0, 0.23.2, 0.8.0, 0.5.2/0.5.2, 3.3.1 |

For a minimal CPU development environment, install this package with its `test` extra and the base scientific dependencies. Add a platform-compatible PyTorch installation for the synthetic head checks. The `train` extra declares the PyTorch dependency but does not choose a CPU/CUDA distribution for the user. See [pyproject.toml](../pyproject.toml) and the [historical backbone versions](../requirements-backbone-tested.txt).

From the repository root, after creating and activating a Python environment:

```sh
python -m pip install -e ".[test]"
python -m pytest
```

After separately installing a compatible CPU PyTorch build, the optional `s1pert head-smoke --out runs/synthetic-check` command exercises tiny synthetic optimization and restore/evaluation mechanics. Use a fresh output directory. It does not encode text through the backbone. Tests requiring PyTorch may be skipped when PyTorch is absent; report those skips rather than treating them as executed checks.

The historical environment also contained `decider-ai==1.2.1`, Scanpy, AnnData, torch-geometric, and other packages. Their presence does not establish that A4 used them. The A4 source reads HDF5 through h5py, computes its classification metrics directly, and imports the model snapshot's `decider` code. Installing `decider-ai` is not a replacement for that snapshot.

Full encoding is a CUDA workflow. The loader constructs Qwen3.5 through Transformers, manually loads safetensors, and uses a specific `fla.modules.conv.causal_conv1d` interface. The layer executor accesses Qwen3.5 internals. These interfaces require a compatible PyTorch/CUDA/GPU/library combination. A successful package installation or CPU test is insufficient to establish 4B compatibility.

## External assets that must be supplied

1. **Raw expression data or compatible preprocessed stores.** The historical four datasets are HepG2 and Jurkat from Nadig et al. (GSE264667), and RPE1 and K562 essential from Replogle et al., using the recorded source transformations. Dataset access terms and redistribution rights remain separate from this repository.
2. **Strict label-resolution stores.** Supply or generate labels together with the manifests, masks, reason codes, and aligned gene/target identities expected by the preparation code. Unknown labels must remain unknown.
3. **Public functional annotations.** Supply a lawful annotation source with the fields `accession`, `symbol`, `protein_name`, and `function_text`. The original encoder reads only the SQLite `genes` table, orders rows by symbol/accession, chooses the longest combined protein-name/function text per symbol, and truncates the selected text to 320 characters. A public TSV export is not directly interchangeable with the original SQLite database; use the documented import command. Empirical response tables are not part of this input.
4. **A complete frozen backbone snapshot.** Supply the checkpoint, complete tokenizer, model configuration, and matching snapshot `decider` source. These are external assets, not files in this release.
5. **Writable output storage.** Encoding produces large local semantic caches; training produces checkpoints and predictions. Keep these outside the source archive.

The recorded backbone is `Mapika/decider-4b` v1, revision `c23ab4d2483e7c6a93484463e9afe1688b29bedd`, with source commit `b44b4c9880a67291206499b86aac89004850134a` in the snapshot manifest. The historical unsharded weight file was 8,411,558,400 bytes with SHA256 `50b88d1ede147c96d2a2e0edabe5807aede7a8ea913577d006c438a5253b17fd`. That digest was read from existing metadata, not recomputed against the large weight during release preparation. It identifies DECIDER; it is not an official hosted TypeSafe Jev checkpoint.

Use a fresh process for encoding so that an already-imported, unrelated `decider` module cannot override `decider_source`. A4 uses the snapshot's model wrapper and hidden representations; it does not use the native DECIDER option probabilities as its biological predictions.

## Configuration and workflow

Use the supplied example configuration as a template. Paths are resolved relative to the configuration file, with explicit environment-variable expansion. `check-config` parses the configuration and resolves paths; it does not establish that external data or weights exist or are compatible. `run --plan` displays the stage plan. Neither substitutes for a GPU execution check.

| Configuration field | Meaning |
| --- | --- |
| `project_root` | New writable run/output location |
| `runtime_python` | Interpreter used for isolated subprocesses |
| `data_preprocessed_root` | Factorized expression-moment and identity stores |
| `label_resolution_root` | Strict label and mask stores aligned to the data |
| `knowledge_db` | Public-function SQLite input, containing the required `genes` fields |
| `decider_snapshot` | Complete local backbone checkpoint and configuration |
| `tokenizer_snapshot` | Complete matching local tokenizer |
| `decider_source` | Directory containing the matching snapshot `decider` package |
| `expected_backbone_weight_sha256` | Identity check for the intended checkpoint |
| `device`, `gpu_uuid` | Explicit accelerator selection for production execution |

The workflow is preprocessing → strict label resolution → target splits and evidence preparation → frozen encoding → fresh task-head training → independent restore check → sealed TEST evaluation. The high-level CLI exposes these stages. Consult each command's `--help`; preprocessing additionally uses the `preprocess` section of the configuration.

The historical Nadig preprocessing script reused previously audited expression moments. The portable release supplies a separate streaming moment generator adapted from the earlier preprocessing helper so that those hidden cache paths are no longer a prerequisite. This is a portability change checked with synthetic data; equivalence on the complete historical raw datasets has not been re-established. Its origin is recorded separately in the source inventory. Replogle preprocessing retains streaming HDF5 processing. File layout, counts, normalization denominators, and gene identities must match the relevant source contract; an arbitrary h5ad is not interchangeable.

The original configuration requested 600 GiB free disk before production execution, a 40 GiB partition RAM cache budget, and a 48 GiB GPU memory ceiling. The handoff described roughly 435 GiB of semantic caches. These are historical settings and estimates, not guaranteed minimum hardware specifications. The memory-safe training overlay changes cache placement/loading; it does not license changes to the scientific model or data split.

## Fidelity of the portable core

The core was compared with the selected A4 source and its memory-safe execution overlay. The fresh decoder, frozen contiguous-layer executor, joint landmark pooling, loss, optimization schedule, split seeds, and random-number salt retain their original calculations. Portability changes resolve resource paths from the configuration location, use package-qualified imports, reject incompatible inputs, and route training, evaluation, and independent checkpoint replay through the same memory-safe implementation. These changes do not replace predictions or alter metric definitions.

CPU checks use small artificial arrays and a miniature head. They can verify the direct output formula, evidence masking, leakage guards, cache identity checks, target splits, and independent checkpoint replay. They do not validate the external 4B runtime, GPU cache preload path, complete real-data preprocessing, or agreement with previously reported biological scores. Full encoding and reproduction remain new runs using the supplied external assets.

## Scientific contract to preserve

- Train each cell line independently. Use five independent target-level 70/10/20 TRAIN/VAL/TEST splits with seeds `17, 29, 43, 61, 89`. These are repeated holdouts, not disjoint five-fold cross-validation; seed-to-seed variation includes both split and optimization variation.
- Use the strict non-self target/query set for the main analysis. `UP=0`, `SAME=1`, `DOWN=2`; unresolved supervision is ignored (`-100` after resolution), never filled with SAME. SAME requires equivalence evidence. The statistical labels use cell-level Welch/TOST approximations, not inference across independent biological replicates.
- Normalize using the recorded original UMI denominator and preserve gene alignment. Do not replace the gene universe with a convenient common subset or resample splits to improve a result.
- Construct propensity inputs from TRAIN labels only. For a TRAIN row, exclude its entire target fold; VAL/TEST rows use only TRAIN-derived statistics. The prior is an input feature, never an additive output or fixed prediction anchor.
- Freeze the general pretrained backbone. Initialize all task-head weights from scratch. The measured target-context control means, variances, and detection rates are legal background inputs; they do not supply an old prediction to repair.
- Keep the A4 context lengths (condition 16, query 16, prior 4), frozen last-eight-layer interaction, and task head (width 512, two blocks, eight attention heads, FFN width 2048). A4 does not use LoRA or a graph-neural-network branch.
- Select by VAL pooled Macro-F1, then lower NLL, then earlier epoch. Preserve the declared convergence gate and independent restore check before TEST. Hitting the maximum epoch limit is not itself convergence. TEST is not a tuning set.

The reviewed forward path ends in two learned logits: `changed` and `direction`. With `g = sigmoid(changed)` and `d = sigmoid(direction)`, the output is `[g*d, 1-g, g*(1-d)]` in UP/SAME/DOWN order. No previous model probability, fixed source-response mean, or old task checkpoint is added or mixed into this output. P1 propensity metrics are an independent comparator. Ordinary internal attention residual connections do not change this contract.

## Limits of the release

This code release does not contain historical task checkpoints, raw data, public gene databases, TRAIN response caches, full tokenizers, 4B weights, or prediction arrays. It therefore cannot reproduce the reported test scores from the archive alone. Fresh runs have new source/configuration identities; adapted source hashes must not be represented as the unchanged historical sealed source. A synthetic PASS means the tested mechanics worked, not that full reproduction, biological performance, upstream licensing, or arbitrary GPU compatibility has been established.
