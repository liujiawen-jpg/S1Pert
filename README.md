# S1Pert

**Population-average perturbation response classification with DECIDER-4B.**
Given a cell context, a single-gene CRISPRi intervention and a query gene,
S1Pert predicts **Up, Same or Down**. This anonymous source release contains
the selected **A4** implementation, public-data acquisition, preprocessing,
strict statistical labels, frozen semantic encoding, training, evaluation and
portable comparison adapters.

[中文说明](README_zh.md) · [Data](docs/DATA_PIPELINE.md) ·
[Model](docs/MODEL_AND_TRAINING.md) · [Evaluation](docs/EVALUATION.md) ·
[Baselines](docs/BASELINES.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

## Method and protocol

Three evidence branches describe the cell/intervention, query gene, and
TRAIN-only historical response tendencies. Frozen **Mapika DECIDER-4b v1**
encodes the branches and jointly contextualizes their landmarks through its
last eight layers. A freshly initialized two-block attention decoder combines
these representations and numeric evidence to predict change and conditional
direction. Its probabilities are `[P(change)P(up|change), 1-P(change),
P(change)(1-P(up|change))]`.

DECIDER is frozen; the selected A4 uses no LoRA. Task weights are initialized
from scratch for every cell line/split. Historical tendencies are model inputs,
not a fixed prediction added to a learned correction. No old predictor or task
checkpoint supplies the model's answer. The actual backbone and revision are
specified in `configs/downloads.json`; no hosted Jev service is required.

| Setting | Value |
|---|---|
| Independent datasets | HepG2, Jurkat, RPE1, K562 essential |
| Split unit and ratio | Whole perturbation targets; 70/10/20 TRAIN/VAL/TEST |
| Split seeds | 17, 29, 43, 61, 89 (five repeated holdouts) |
| Supervision | Statistically resolved non-self pairs; Up=0, Same=1, Down=2 |
| Same definition | TOST equivalence inside the prespecified 1.2-fold band |
| Unresolved labels | Excluded from three-class loss and primary evaluation |
| TRAIN propensity | Five-fold target cross-fitting; VAL/TEST use TRAIN only |
| Selection | Validation Macro-F1; NLL then earlier epoch for ties |
| Training | AdamW, LR 1e-4, weight decay 0.01; effective batch 4096 |
| Convergence | Minimum 10, maximum 100 epochs; patience 10 |
| Reporting | Per-split metrics; mean, sample SD and variance across five seeds |

All measured query genes are retained in the data axes; the three-class
supervised subset contains only statistically resolved pairs. See the data
documentation for coverage, expression denominators and source-specific rules.

## 1. Install

Use Linux and Python 3.11+. Start a new environment and install a suitable
PyTorch build for your CPU/GPU before the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip install 'torch>=2.7'
s1pert --help
```

Frozen encoding additionally requires the stack in
`requirements-backbone-tested.txt`, including the pinned DECIDER snapshot source
and tokenizer. Baselines have separate dependency files and environments.
Installation, `--help`, and plan/list commands never download data or train.
The compatibility CLI `jev-a4` and Python package name `jev_a4` remain available.

## 2. Download public inputs

```bash
python scripts/download_assets.py --list
python scripts/download_assets.py --group data --dry-run
python scripts/download_assets.py --group data
python scripts/download_assets.py --group annotations
s1pert import-knowledge --tsv assets/uniprot_reviewed_human.tsv \
  --out assets/public_functions.sqlite
```

The four raw matrices total about 17.8 GB. The manifest uses original GEO
HepG2/Jurkat files and the exact scPerturb RPE1/K562 essential files. Published
checksums or verified sizes and download receipts protect file integrity.
Public gene-function exports from UniProt are live; save the receipt and share
the same snapshot across methods. They do not guarantee historical annotation
byte identity.

Backbone download is a separate, explicit opt-in:

```bash
python scripts/download_assets.py --group backbone --include-weights
```

Large weights, tokenizer payloads, raw data, derived arrays, caches and task
checkpoints are **not contained in this ZIP**. The downloader fetches them to
ignored local directories. External source licenses remain with their assets.

## 3. Preprocess and construct strict labels

```bash
s1pert preprocess --config configs/preprocess.example.json
s1pert labels --config configs/labels.example.json
s1pert validate-labels --source workspace/preprocessed --bundle workspace/strict_labels
```

Preprocessing streams counts, repairs and checks gene identities, and builds
factorized population statistics. Labeling retains Up/Down directional evidence
and Same equivalence evidence separately from unresolved pairs. A failed or
existing output is preserved for inspection. Paths in JSON resolve relative to
that JSON file. Copy examples to `*.local.json` to customize paths and budgets.

## 4. Run S1Pert

```bash
cp configs/a4.example.json configs/a4.local.json
s1pert check-config --config configs/a4.local.json
s1pert run --config configs/a4.local.json --plan

# Requires real assets and a configured GPU; performs small engineering checks.
s1pert preflight --config configs/a4.local.json --out workspace/preflight

# Explicitly starts the automatic study after preflight succeeds.
s1pert launch --config configs/a4.local.json \
  --preflight-report workspace/preflight/report.json
```

The finite workflow runs preparation → frozen caches → fresh decoder training
→ validation selection → independent-process recovery → locked TEST evaluation.
It stops on failed gates and writes a report; it installs no recurring monitor.
The default covers four cell lines × five seeds. Individual `prepare`, `encode`,
`train`, `restore-check`, and `evaluate` commands support `--help`.

The example retains the original 600-GiB free-space guard for large frozen
caches. Set resource limits for your machine after reviewing the memory-mapped
data flow. The selected `best.pt` contains the task head; it is not a complete
optimizer-state resume checkpoint. See reproducibility documentation for safe
restart behavior.

## 5. Run comparisons and evaluate

[BASELINES.md](docs/BASELINES.md) gives commands, required assets, training
budgets and adaptations for P1, GenePT-Gene, GenePT-Prot, GEARS, scGPT, STATE,
SUMMER-Qwen3-8B and Qwen3-4B-LoRA. The available ALIGNED prediction exports can
be evaluated, but its exact historical training source was unavailable when
this package was assembled; no replacement is presented as that experiment.

Register pair-aligned outputs in a copy of `configs/predictions.example.csv`:

```bash
s1pert compare --registry configs/predictions.local.csv --out workspace/comparison
```

Outputs include class-wise precision/recall/F1, Macro-F1, BA, within-gene
Chg-AUC/Pol-AUC, per-split CSVs and mean/SD/variance summaries. Hard-only methods
keep unavailable AUCs blank. Full details are in [EVALUATION.md](docs/EVALUATION.md).

## 6. Validate and package

```bash
python -m pytest -q
python scripts/check_release.py
python scripts/package_release.py --out ../S1Pert_anonymous_source.zip
```

The release audit rejects machine paths, contact details, credentials, binary
payloads and symlinks. The archive excludes Git history, environment files,
logs and runtime outputs, normalizes ZIP timestamps/permissions, and includes
per-file SHA256 hashes. `VALIDATION_REPORT.json` records executed packaging and
CPU checks; packaging does not rerun the biological GPU study.

```text
configs/              protocol, download manifest and editable path examples
src/jev_a4/data/       streamed raw-data preprocessing
src/jev_a4/labels/     statistical labels and integrity validation
src/jev_a4/core/       A4 preparation, encoding, training and selected-head recovery
src/jev_a4/evaluation.py   aligned cross-method metrics and CSV summaries
baselines/            portable comparison adapters and external-asset setup
scripts/              downloads, prediction conversion, anonymity audit and ZIP
tests/                CPU synthetic fixtures and leakage/identity guards
docs/                 data, method, execution and evaluation details
```

No remote GitHub repository is created by this release. Publish through an
anonymous account during blind review; an ordinary personal GitHub account
reveals its owner regardless of archive contents. A project-wide source license
has not been assigned; [LICENSE_STATUS.md](LICENSE_STATUS.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) distinguish local code from
separately obtained dependencies.
