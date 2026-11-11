# Comparison adapters

All methods use the same statistically resolved TEST pairs and target-held-out partitions as S1Pert. Reuse the prepared records without resampling rows. Train each cell separately with seeds 17, 29, 43, 61, and 89. Each adapter documents the method-specific inputs, losses, and compute budget.

| Method | Packaged implementation | Required external assets |
|---|---|---|
| P1 | TRAIN-only class-propensity exporter | S1Pert prepared evidence |
| GenePT-Gene / GenePT-Prot | Frozen public vectors + multinomial logistic classifier | Pinned GenePT archive |
| GEARS | Official GEARS_Model and official loss, native expression adapter | Pinned GEARS, public gene2go, legal TRAIN/control expression |
| scGPT | Official TransformerGenerator and pretrained encoder, native expression adapter | scGPT source, whole-human checkpoint |
| STATE | Official StateTransition model and energy loss, strict-query adapter | Pinned STATE source, public ESM2 vectors, legal expression |
| SUMMER | Official retrieval-priority helper + local Qwen3-8B prompting, vLLM | PerturbQA source, public functions/edges, generic Qwen3-8B |
| Qwen3-4B-LoRA | Original local discriminative LoRA comparator | Generic Qwen3-4B, public graph |
| ALIGNED | Actual six-stage MLP/REINFORCE/two-round knowledge-refinement adapter | Public signed graph and direct GO annotations. Required official core is bundled at its verified commit |

This repository includes the local task adapters and the required attributed ALIGNED core. Acquire all other external source separately under its original license. [Asset preparation](../baselines/ASSETS.md) lists source pins, downloads, schema conversions, and manual prerequisites.

## Shared setup

Run S1Pert preprocessing, labeling, and `prepare` for each cell before running any baseline. Baseline inputs must use `data/` in the shared work directory. Do not use a different split. In the commands below, `W` is the shared work directory and has the same value as `S1PERT_ROOT`. Initialize it once from the repository root if it does not exist:

```bash
python -m experiments.paper init --work-dir W
python baselines/fetch_external.py --root W --list
python baselines/fetch_external.py --root W --repo gears --repo perturbqa --repo state --repo scgpt --dry-run
# Explicit network acquisition (no model execution):
python baselines/fetch_external.py --root W --repo gears --repo perturbqa --repo state --repo scgpt
python baselines/fetch_external.py --root W --asset genept --asset kg --asset gene2go --asset esm2
```

Use separate environments for the classical, STATE, and vLLM stacks. Install S1Pert in each environment. JSON resource paths resolve relative to the JSON file. The paper initializer writes a correctly rooted JSON file for each family under `W/configs/`. Review the configs, then run `baselines/run.py init` to stage a **new** family workspace. Initialization copies the small adapter modules and resolves asset paths. It does not download assets, train models, or load weights. `execute --plan -- ...` prints a command without running it.

## P1

```bash
python scripts/export_prior.py \
  --partition W/data/hepg2/seed_17/test \
  --out W/baselines/p1/hepg2/seed_17/test_predictions.npz
```

P1 reads the prepared probability features derived from TRAIN. Unseen queries use the prepared TRAIN-only global fallback. P1 is an independent comparator. Its exported predictions do not enter S1Pert or any new predictor.

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

In the native command, a relative `--out` path resolves from the adapter workspace. Absolute output paths are also supported. Set `--method` to `genept_prot`, `gears`, or `scgpt`, and assign a distinct output directory to each method. Prepare expression once per cell and GEARS graphs once per cell and seed. After successful selection, the formal training command runs independent restoration and TEST automatically. GAT is not part of this release's run plan.

GenePT fits a classifier over concatenated target and query vectors through an equivalent factorized linear calculation. Validation Macro-F1 selects C from 0.01, 0.1, 1, and 10. L-BFGS runs for up to 1,000 iterations. This adapter uses multinomial logistic adaptation, not the original two-binary-head task.

GEARS and scGPT preserve native expression prediction. Both receive legal TRAIN single-cell expression and controls. This input provides additional continuous supervision beyond the categorical labels used by S1Pert. GEARS uses TRAIN/control coexpression and public GO data. scGPT assigns fresh embeddings to missing public-vocabulary symbols, retains intervention tokens during chunking, and covers the full query axis. Validation selects the asymmetric down/up effect thresholds and the checkpoint. TEST is never used for threshold selection. Training runs for 10–100 epochs with a patience of 10. The exact default configuration is included with the code.

Missing generic GenePT vectors use a TRAIN-identity **embedding** mean, not a response mean. This fallback does not add a prior predictor to the output.

## STATE

Install the pinned official source and its dependencies in a separate environment. Prepare the exact ESM2 asset mapping and the classical per-cell expression stores described in ASSETS.md. Set `expression_root` to the classical workspace.

```bash
python baselines/run.py init --family state --config W/configs/state.json --work W/baselines/state
python baselines/run.py execute --work W/baselines/state -- run.py preflight
python baselines/run.py execute --work W/baselines/state -- run.py train --cell hepg2 --seed 17
python baselines/run.py execute --work W/baselines/state -- run.py evaluate --cell hepg2 --seed 17
python baselines/run.py execute --work W/baselines/state -- run.py summary
```

STATE uses fresh task weights, the official eight-layer transition model with width 768, and the official energy-distance loss. Each set contains 128 cells. The condition batch size is 16, and the learning rate is 1e-4. The loss is restricted to strict TRAIN query dimensions. Frozen ESM2 vectors identify interventions. No previous response-trained STATE checkpoint is loaded. Validation selects the native-effect thresholds and checkpoints. Independent restoration runs before TEST. The unused auxiliary count decoder is omitted. This adapter is `STATE-ST-native-strict-A4-adapted`, not the full pretrained SE+ST suite.

## SUMMER

This comparison requires no training. It generates public summaries, retrieves legal TRAIN examples with the official SUMMER priorities, and emits a rationale plus one class tag. The local **Qwen3-8B** replaces the original Llama3 backbone. The adapter uses no teacher-generated labels or task fine-tuning.

```bash
python -m pip install -r baselines/requirements-summer.txt
python baselines/prepare_public_edges.py --archive W/assets/baselines/kg.zip --out W/assets/public_edges.sqlite
python baselines/run.py init --family summer --config W/configs/summer.json --work W/baselines/summer
python baselines/run.py execute --work W/baselines/summer -- fast_summer.py preflight
python baselines/run.py execute --work W/baselines/summer -- fast_summer.py pipeline
```

The queue covers all configured cells and seeds, caches native generations, and stops when a gate fails. This release generates every group from scratch and does not inject historical HepG2 results. Full identities protect the reuse of completed results. Invalid outputs count as errors, never as Same. Hard-label exports support classification metrics. The adapter does not fabricate probabilities or ranking AUCs. Rebuilt public-edge databases are documented as new snapshots and are not presented as identical to an unavailable previous database.

## Qwen3-4B-LoRA

[Qwen adapter instructions](../baselines/language/README.md) describe the retained local implementation. It uses LoRA r=32/alpha=64, 512 sampled TRAIN rows per epoch, and 2:2:4 objective stages. Its historical name was `PertReasonLM-LoRA-A4-adapted`. The code implements a local Qwen classifier, **not** the official PertReasonLM SFT/GRPO system. Treat it as a small-budget comparator, not a converged full-data LoRA experiment.

## ALIGNED

The [ALIGNED adapter](../baselines/aligned/README.md) ports the inspected training implementation. All seven previously recorded core hashes matched the source supplied from the original execution environment. The scientific engine and pinned official core retain their exact bytes. Workspace paths, input seals, and probability export are portable adaptations.

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

The public builder reconstructs signed edges, canonical mappings, and direct GO annotations from explicit public snapshots. By default, it verifies historical input hashes. A changed snapshot requires an explicitly refreshed protocol. The builder exports only the public tables consumed by ALIGNED. Dataset responses, task predictions, and empirical-response tables never enter this knowledge input.

ALIGNED uses the official hidden-64 MLP and learned R adaptor. Its fixed endpoint is the second knowledge-refinement round after initialization, pretraining, and two complete retrain/refine rounds. Each run starts with fresh task weights. Only TRAIN targets contribute to pseudo-label training and graph refinement. VAL verifies independent restoration and does not select a stage. The original hybrid output is retained. The final output uses neural softmax when R≤0.5 and a KB-deduction point mass when R>0.5. The canonical `test_predictions.npz` contains continuous probabilities and exact target, query, and label arrays. Its argmax is checked against the original hybrid decision. No ALIGNED output enters S1Pert.

The default paths are `W/baselines/aligned/{prepared,runs}/CELL/seed_SEED`. The formal config is `W/configs/aligned.json` when supplied. Otherwise, the adapter uses the bundled official default. `--prepared-root` and `--output` support isolated runtime experiments. Training and restoration/evaluation run in separate processes. See the adapter README for objective details, original-code adaptations, snapshot requirements, and CPU fixture tests.

## Export and summarize

Follow [EVALUATION.md](EVALUATION.md) for classical array conversion and the registry CSV. Portable outputs from STATE, SUMMER, and Qwen include direct pair identities. Repeat the documented commands for all four cell lines and all five seeds. Report missing runs explicitly. Do not replace a missing cell or seed with a different checkpoint, budget, or subset.

CPU tests and configuration checks validate the release mechanics. Packaging did not include a repeat of the full baseline GPU study. Exact reproduction of historical numerical results also depends on the documented external asset snapshots.
