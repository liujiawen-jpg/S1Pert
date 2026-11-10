# Comparison adapters

All methods evaluate the same statistically resolved TEST pairs and target-held-out
partitions as S1Pert. Reuse prepared records: do not independently resample rows.
Train each cell separately with seeds 17, 29, 43, 61, 89. The adapters expose the
inputs, losses and budgets that actually distinguish the methods.

| Method | Packaged implementation | Required external assets |
|---|---|---|
| P1 | TRAIN-only class-propensity exporter | S1Pert prepared evidence |
| GenePT-Gene / GenePT-Prot | Frozen public vectors + multinomial logistic classifier | Pinned GenePT archive |
| GEARS | Official GEARS_Model and official loss, native expression adapter | Pinned GEARS, public gene2go, legal TRAIN/control expression |
| scGPT | Official TransformerGenerator and pretrained encoder, native expression adapter | scGPT source, whole-human checkpoint |
| STATE | Official StateTransition model and energy loss, strict-query adapter | Pinned STATE source, public ESM2 vectors, legal expression |
| SUMMER | Official retrieval-priority helper + local Qwen3-8B prompting, vLLM | PerturbQA source, public functions/edges, generic Qwen3-8B |
| Qwen3-4B-LoRA | Original local discriminative LoRA comparator | Generic Qwen3-4B, public graph |
| ALIGNED | Existing exported predictions can use the common evaluator | Exact historical training adapter unavailable; see `baselines/aligned/` |

Only local task adapters are distributed. External source is acquired separately
with its original attribution/license. [Asset preparation](../baselines/ASSETS.md)
lists source pins, downloads, schema conversions and manual prerequisites.

## Shared setup

Run S1Pert preprocessing, labeling and `prepare` for each cell before any baseline.
The baseline inputs must point to `workspace/a4/data`, not a different split.
From the repository root:

```bash
python baselines/fetch_external.py --list
python baselines/fetch_external.py --repo gears --repo perturbqa --repo state --repo scgpt --dry-run
# Explicit network acquisition (no model execution):
python baselines/fetch_external.py --repo gears --repo perturbqa --repo state --repo scgpt
python baselines/fetch_external.py --asset genept --asset kg --asset gene2go --asset esm2
```

Use separate environments for the classical, STATE and vLLM stacks. Install
S1Pert in each. JSON resource paths resolve relative to the JSON file. Copy the
family's example in the same directory to `config.local.json`, edit it, then use
`baselines/run.py init` to stage a **new** workspace. Initialization copies small
adapter code, resolves asset paths, and does not download, train or load weights.
`execute --plan -- ...` prints a command without executing it.

## P1

```bash
python scripts/export_prior.py \
  --partition workspace/a4/data/hepg2/seed_17/test \
  --out workspace/predictions/P1/hepg2/seed_17/test_predictions.npz
```

P1 reads the prepared TRAIN-derived probability features. For unseen queries it
uses the prepared TRAIN-only global fallback. It is an independent comparator;
these exported answers are not fed to S1Pert or any new predictor.

## GenePT, GEARS and scGPT

```bash
python -m pip install -r baselines/requirements-classical.txt
cp baselines/classical/config.example.json baselines/classical/config.local.json
python baselines/run.py init --family classical \
  --config baselines/classical/config.local.json --work workspace/classical

python baselines/run.py execute --work workspace/classical -- prepare.py genept
python baselines/run.py execute --work workspace/classical -- prepare.py expression --cell hepg2
# GEARS only: constructs public GO and TRAIN/control coexpression graphs.
python baselines/run.py execute --work workspace/classical -- prepare.py native_graphs --cell hepg2 --seed 17

python baselines/run.py execute --work workspace/classical -- train.py train \
  --method genept_gene --cell hepg2 --seed 17 --out runs/genept_gene/hepg2/seed_17
```

`--out` in the native command is relative to the adapter workspace; an absolute
output path is also supported.
Change `--method` to `genept_prot`, `gears`, or `scgpt` and give each a distinct
output directory. Prepare expression once per cell, and GEARS graphs once per
cell/seed. Formal training automatically invokes independent restoration and
TEST after successful selection. GAT is not part of this release's run plan.

GenePT fits a concatenated target/query vector classifier through an equivalent
factorized linear calculation. It selects C from 0.01/0.1/1/10 on validation
Macro-F1, with L-BFGS up to 1000 iterations. This is multinomial logistic
adaptation, not the original two-binary-head task.

GEARS/scGPT preserve native expression prediction. They receive legal TRAIN
single-cell expression and controls, which is additional continuous supervision
relative to S1Pert's categorical labels. GEARS uses TRAIN/control coexpression
and public GO. scGPT extends missing public vocabulary symbols with fresh
embeddings, retains intervention tokens during chunking, and covers the complete
query axis. Validation selects asymmetric down/up effect thresholds and the
checkpoint; TEST is never used for threshold selection. Training is 10–100
epochs with patience 10. The exact default configuration is shipped beside code.

Generic missing GenePT vectors use a TRAIN-identity **embedding** mean, not a
response mean. This does not insert a prior predictor into the output.

## STATE

Install the pinned official source and its dependencies in a separate environment.
Prepare the exact ESM2 asset mapping and classical per-cell expression stores as
described in ASSETS.md. Configure `expression_root` to that classical workspace.

```bash
cp baselines/state/config.example.json baselines/state/config.local.json
python baselines/run.py init --family state --config baselines/state/config.local.json --work workspace/state
python baselines/run.py execute --work workspace/state -- run.py preflight
python baselines/run.py execute --work workspace/state -- run.py train --cell hepg2 --seed 17
python baselines/run.py execute --work workspace/state -- run.py evaluate --cell hepg2 --seed 17
python baselines/run.py execute --work workspace/state -- run.py summary
```

STATE uses fresh task weights, the official eight-layer/768-width transition
model and energy distance, 128 cells per set, batch 16 conditions, LR 1e-4.
Its loss is restricted to strict TRAIN query dimensions. Frozen ESM2 vectors
identify interventions. No old response-trained STATE checkpoint is loaded.
Validation selects native-effect thresholds and checkpoints; independent
restoration precedes TEST. The unused auxiliary count decoder is omitted.
This is `STATE-ST-native-strict-A4-adapted`, not the entire pretrained SE+ST suite.

## SUMMER

This comparison is training-free: it generates public summaries, retrieves legal
TRAIN examples using the official SUMMER priorities and emits a rationale plus
one class tag. The local **Qwen3-8B** replaces the original Llama3 backbone.
It uses no teacher-generated labels or task fine-tuning.

```bash
python -m pip install -r baselines/requirements-summer.txt
python baselines/prepare_public_edges.py --archive assets/baselines/kg.zip --out assets/public_edges.sqlite
cp baselines/summer/config.example.json baselines/summer/config.local.json
python baselines/run.py init --family summer --config baselines/summer/config.local.json --work workspace/summer
python baselines/run.py execute --work workspace/summer -- fast_summer.py preflight
python baselines/run.py execute --work workspace/summer -- fast_summer.py pipeline
```

The queue covers all configured cells/seeds, caches native generations and stops
on failed gates. All groups are newly generated in this release; no historical
HepG2 result is injected. Full identities protect completed-result reuse. Invalid
outputs count as errors, never Same. Hard-label exports support classification
metrics; no probabilities or ranking AUCs are fabricated. Public-edge rebuilds
are documented new snapshots and do not assert equality to an unavailable old DB.

## Qwen3-4B-LoRA and ALIGNED

[Qwen adapter instructions](../baselines/language/README.md) describe the retained
local implementation: LoRA r=32/alpha=64, 512 sampled TRAIN rows per epoch and
2:2:4 objective stages. Its actual historical name was
`PertReasonLM-LoRA-A4-adapted`; the code is a local Qwen classifier, **not** the
official PertReasonLM SFT/GRPO system. Do not label this small-budget comparator
as a converged full-data LoRA experiment.

[ALIGNED source availability](../baselines/aligned/README.md) records the missing
training implementation. Provided ALIGNED outputs can be registered with the
same pair-aligned evaluation command; the package does not claim a new ALIGNED
training run or silently substitute a plain neural network.

## Export and summarize

Follow [EVALUATION.md](EVALUATION.md) for classical array conversion and the
registry CSV. STATE, SUMMER and Qwen portable outputs carry direct pair identity.
Repeat the documented run commands for all four cell lines and five seeds.
Report missing runs explicitly; do not replace missing cells/seeds with a
different checkpoint, budget or subset.

CPU tests and configuration checks qualify the release's mechanics. No full
baseline GPU study was repeated while packaging, and exact historical numerical
identity additionally depends on the documented external asset snapshots.
