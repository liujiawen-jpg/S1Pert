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
| ALIGNED | Actual six-stage MLP/REINFORCE/two-round knowledge-refinement adapter | Public signed graph, direct GO annotations; required official core is bundled at its verified commit |

Local task adapters and the required attributed ALIGNED core are distributed.
Other external source is acquired separately with its original license.
[Asset preparation](../baselines/ASSETS.md)
lists source pins, downloads, schema conversions and manual prerequisites.

## Shared setup

Run S1Pert preprocessing, labeling and `prepare` for each cell before any baseline.
The baseline inputs must point to the shared work directory's `data/`, not a different split.
Below, `W` denotes the shared work directory (the same value as `S1PERT_ROOT`).
From the repository root, initialize it once if it does not already exist:

```bash
python -m experiments.paper init --work-dir W
python baselines/fetch_external.py --root W --list
python baselines/fetch_external.py --root W --repo gears --repo perturbqa --repo state --repo scgpt --dry-run
# Explicit network acquisition (no model execution):
python baselines/fetch_external.py --root W --repo gears --repo perturbqa --repo state --repo scgpt
python baselines/fetch_external.py --root W --asset genept --asset kg --asset gene2go --asset esm2
```

Use separate environments for the classical, STATE and vLLM stacks. Install
S1Pert in each. JSON resource paths resolve relative to the JSON file. The paper
initializer writes each family's correctly rooted JSON under `W/configs/`.
Review these configs, then use `baselines/run.py init` to stage a **new** family
workspace. Initialization copies small
adapter code, resolves asset paths, and does not download, train or load weights.
`execute --plan -- ...` prints a command without executing it.

## P1

```bash
python scripts/export_prior.py \
  --partition W/data/hepg2/seed_17/test \
  --out W/baselines/p1/hepg2/seed_17/test_predictions.npz
```

P1 reads the prepared TRAIN-derived probability features. For unseen queries it
uses the prepared TRAIN-only global fallback. It is an independent comparator;
these exported answers are not fed to S1Pert or any new predictor.

## GenePT, GEARS and scGPT

```bash
python -m pip install -r baselines/requirements-classical.txt
python baselines/run.py init --family classical \
  --config W/configs/classical.json --work W/baselines/classical

python baselines/run.py execute --work W/baselines/classical -- prepare.py genept
python baselines/run.py execute --work W/baselines/classical -- prepare.py expression --cell hepg2
# GEARS only: constructs public GO and TRAIN/control coexpression graphs.
python baselines/run.py execute --work W/baselines/classical -- prepare.py native_graphs --cell hepg2 --seed 17

python baselines/run.py execute --work W/baselines/classical -- train.py train \
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
python baselines/run.py init --family state --config W/configs/state.json --work W/baselines/state
python baselines/run.py execute --work W/baselines/state -- run.py preflight
python baselines/run.py execute --work W/baselines/state -- run.py train --cell hepg2 --seed 17
python baselines/run.py execute --work W/baselines/state -- run.py evaluate --cell hepg2 --seed 17
python baselines/run.py execute --work W/baselines/state -- run.py summary
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
python baselines/prepare_public_edges.py --archive W/assets/baselines/kg.zip --out W/assets/public_edges.sqlite
python baselines/run.py init --family summer --config W/configs/summer.json --work W/baselines/summer
python baselines/run.py execute --work W/baselines/summer -- fast_summer.py preflight
python baselines/run.py execute --work W/baselines/summer -- fast_summer.py pipeline
```

The queue covers all configured cells/seeds, caches native generations and stops
on failed gates. All groups are newly generated in this release; no historical
HepG2 result is injected. Full identities protect completed-result reuse. Invalid
outputs count as errors, never Same. Hard-label exports support classification
metrics; no probabilities or ranking AUCs are fabricated. Public-edge rebuilds
are documented new snapshots and do not assert equality to an unavailable old DB.

## Qwen3-4B-LoRA

[Qwen adapter instructions](../baselines/language/README.md) describe the retained
local implementation: LoRA r=32/alpha=64, 512 sampled TRAIN rows per epoch and
2:2:4 objective stages. Its actual historical name was
`PertReasonLM-LoRA-A4-adapted`; the code is a local Qwen classifier, **not** the
official PertReasonLM SFT/GRPO system. Do not label this small-budget comparator
as a converged full-data LoRA experiment.

## ALIGNED

The [ALIGNED adapter](../baselines/aligned/README.md) ports the actual inspected
training implementation. All seven previously recorded core hashes matched the
source supplied from its original execution environment. The scientific engine
and pinned official core retain their exact bytes; workspace paths, input seals
and probability export are portable adaptations.

```bash
python -m pip install -r baselines/requirements-aligned.txt
python scripts/download_assets.py --manifest data_prep/public_knowledge/sources.json --root W --group public_knowledge --dry-run
python scripts/download_assets.py --manifest data_prep/public_knowledge/sources.json --root W --group public_knowledge
python -m data_prep.build_public_knowledge --work-dir W
python -m baselines.aligned.run prepare --work-dir W --cell hepg2 --seed 17
python -m baselines.aligned.run train --work-dir W --cell hepg2 --seed 17 --device cuda
python -m baselines.aligned.run restore --work-dir W --cell hepg2 --seed 17 --device cuda
python -m baselines.aligned.run evaluate --work-dir W --cell hepg2 --seed 17 --device cuda
```

The public builder reconstructs signed edges, canonical mappings and direct GO
annotations from explicit public snapshots. It verifies historical input hashes
by default; a changed snapshot requires an explicitly refreshed protocol.
Only the public tables consumed by ALIGNED are exported. Dataset responses,
task predictions and empirical-response tables never enter this knowledge input.

ALIGNED uses the official hidden-64 MLP and learned R adaptor. Its fixed endpoint
is the second knowledge-refinement round, following initialization, pretraining
and two complete retrain/refine rounds. Each run starts with fresh task weights;
only TRAIN targets contribute to pseudo-label training and graph refinement.
VAL verifies independent restoration and does not select a stage. The original
hybrid output is retained: final neural softmax when R≤0.5 and final KB deduction
point mass when R>0.5. Canonical `test_predictions.npz` includes continuous
probabilities and exact target/query/labels, with argmax checked against the
original hybrid decision. No ALIGNED output feeds S1Pert.

Defaults are `W/baselines/aligned/{prepared,runs}/CELL/seed_SEED`. The formal
config is `W/configs/aligned.json` when supplied, otherwise the bundled official
default. `--prepared-root` and `--output` support isolated runtime experiments.
The training and restoration/evaluation commands run in separate processes.
See the adapter README for objective details, original-code adaptations,
snapshot requirements and CPU fixture tests.

## Export and summarize

Follow [EVALUATION.md](EVALUATION.md) for classical array conversion and the
registry CSV. STATE, SUMMER and Qwen portable outputs carry direct pair identity.
Repeat the documented run commands for all four cell lines and five seeds.
Report missing runs explicitly; do not replace missing cells/seeds with a
different checkpoint, budget or subset.

CPU tests and configuration checks qualify the release's mechanics. No full
baseline GPU study was repeated while packaging, and exact historical numerical
identity additionally depends on the documented external asset snapshots.
