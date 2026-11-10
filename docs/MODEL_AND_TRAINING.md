# S1Pert model and training contract

S1Pert implements the selected A4 method: a frozen DECIDER `4b-v1`
encoder and a freshly initialized task head for perturbation-response
classification. The production architecture is preserved. The small synthetic
CPU smoke is a test of execution and checkpoint mechanics, not a scientific
model or a reported biological result.

## Legal inputs and labels

Each example identifies an intervention target, cell context and query gene.
Inputs include measured untreated-control statistics, public gene-function
descriptions and TRAIN-derived response-tendency evidence. Control measurements
provide biological background and numeric input features; they are not an old
prediction. The model outputs class probabilities, not reconstructed expression.

Classes are ordered `[up, same, down]` (`same` means `unchanged`). Only strict
resolved labels enter training and scoring; unresolved labels and self-target
pairs are excluded by preparation. See the label implementation and data
documentation for the statistical resolution procedure. Each cell dataset and
split seed is trained independently. Target groups, not individual rows, define
the TRAIN/VAL/TEST separation.

The evidence has 12 numeric features: three smoothed class proportions, log
support, normalized entropy, concentration, top-two margin, three source flags,
up-minus-down proportion and an availability flag. Five-fold cross-fitting
excludes the entire held-out TRAIN target fold when constructing its evidence.
VAL and TEST evidence use TRAIN observations only. Laplace smoothing uses
`alpha=1`; zero query support falls back to same-cell TRAIN class counts. This
evidence is an input to trainable projections. It is never added to output
logits or probabilities as a fixed prediction.

## Three semantic streams and the last eight layers

The external frozen encoder has 32 layers and hidden width 2560. Text is encoded
into three streams: intervention/context, query, and TRAIN-derived evidence
card. The condition and query streams each have 16 landmarks: 12 segment means
and four final token states. A card has four landmarks: three segment means and
the final token state. These are contextual transformer states, not averages of
raw word embeddings. Encoding rejects overlong inputs instead of truncating.

The direct head consumes final-layer condition, query and evidence landmarks.
A second feature path concatenates their **layer-24** landmarks in the order
`condition16 + query16 + evidence4`. This 36-position sequence runs through the
same frozen layers 24–31 and final normalization. Four masked contextual means
(condition, query, evidence, all positions) become the joint landmarks.

Two joint caches are required: evidence present and evidence absent. For the
absent cache, evidence is removed **before** the last eight layers execute.
This prevents a dropped card from remaining encoded in the other joint states.
The encoder checks full/split execution parity and cache replay using the same
generation batch boundaries. There is no LoRA and no task update to any of the
32 backbone layers. Direct public relation cards are not part of this A4.

## Fresh task head

Production settings are locked to decoder width 512, two decoder blocks, eight
attention heads, feed-forward width 2048 and backbone width 2560. Condition,
query, evidence and joint landmarks receive separate learned projections.
Condition/query numeric vectors have width four; evidence has width 12. Role
embeddings, numeric projections and a readout token are trained from scratch.

Each decoder block applies self-attention to query/evidence/readout tokens,
cross-attention to the condition stream, and a feed-forward network. The
normalized readout attends to projected joint landmarks. The joint attention's
output projection starts at zero, matching the selected A4 initialization;
it is a fresh internal network component. No old task head is loaded.

Two linear heads produce logits `c` (changed) and `d` (up given changed):

```text
P(up)   = sigmoid(c) * sigmoid(d)
P(same) = 1 - sigmoid(c)
P(down) = sigmoid(c) * (1 - sigmoid(d))
```

Ordinary internal skip connections operate on learned hidden states. There is
no old predictor, task checkpoint, source mean, output anchor, learned output
correction, blending, calibration or distillation in the predictive dataflow.
The separately reported `standalone_P1_reference` evaluates the TRAIN-derived
class proportions as an independent comparator; it never changes A4 output.

## Objective, exposure and selection

The main loss is exactly the negative log probability of the true class:
binary cross-entropy for changed, plus changed-only binary cross-entropy for
direction. Direction receives no label gradient for unchanged examples.
Evidence dropout defaults to 0.10. One row mask controls card attention, numeric
neutralization and selection of the precomputed present/absent joint cache.
Production calls the head with `prior_dropout=0` because this authoritative
mask has already been applied before the head.

Every TRAIN row is visited exactly once per shuffled epoch. Effective batch
size defaults to 4096; microbatches accumulate the same main objective. Two
additional pair objectives compare the same query under different targets:
changed versus unchanged, and up versus down. Each uses a hinge margin 0.05,
weight 0.02 and 512 sampled pairs per optimizer update. Pair exposure is tracked
separately from the once-per-epoch main exposure.

AdamW defaults to learning rate `1e-4`, weight decay `0.01` and gradient clipping
at 1.0. A validation Macro-F1 plateau reduces learning rate by 0.5 after three
stale epochs, with floor `1e-6`. Defaults require at least 10 epochs and stop
after 10 consecutive epochs without a meaningful Macro-F1 gain above `1e-4`.
The maximum is 100 epochs. Reaching that limit alone is **not convergence**.
All settings and realized exposure are recorded in run artifacts.

VAL selects highest pooled three-class Macro-F1, then lower NLL for an exact
tie, then the earlier epoch. TEST has no role in updates or selection. Macro-F1
always averages three classes, assigning F1=0 to an absent class; balanced
accuracy averages recall over classes present in the reference. Average
precision groups score ties and returns null if a class has no positives.
Equal-target metrics derive from a confusion matrix in which each target has
total weight one; they are not means of individual target F1 values.

## Memory-bounded execution

The public training, evaluation and restore commands use
`core/train_memory_safe.py`. This preserves the scientific model and training
schedule while changing cache placement. Whole-partition CPU copies are
disabled. Joint caches use bounded memory-mapped gathering, or chunked GPU
preloading when capacity permits. GPU staging preserves the cache dtype;
conversion happens at gather. Original limits are retained: 64 MiB host staging,
a 32 GiB joint-cache budget and 14 GiB reserved for forward/backward execution.
The GPU path verifies exact input-tensor parity against ordinary gathering for
present, absent and mixed evidence masks. These GPU checks require a compatible
GPU runtime; the repository's CPU tests exercise the bounded-gather fallback.

Physical-batch probes discard their model and reset initialization before
training. CPU execution is restricted to synthetic smoke tests. Production
training requires CUDA; the configured `gpu_uuid` is applied before Torch is
imported. There is no implicit automatic restart or silent switch to a smaller
scientific architecture.

## Commands and restoration boundaries

After installing the package and preparing data and semantic caches:

```bash
s1pert train --config configs/a4.local.json --dataset hepg2 --seed 17 \
  --data-dir workspace/a4/data/hepg2 --cache-dir workspace/a4/cache/hepg2 --out workspace/a4/runs/hepg2_seed17
s1pert restore-check --config configs/a4.local.json --dataset hepg2 --seed 17 \
  --data-dir workspace/a4/data/hepg2 --cache-dir workspace/a4/cache/hepg2 --out workspace/a4/runs/hepg2_seed17
s1pert evaluate --config configs/a4.local.json --dataset hepg2 --seed 17 \
  --data-dir workspace/a4/data/hepg2 --cache-dir workspace/a4/cache/hepg2 --out workspace/a4/runs/hepg2_seed17
s1pert head-smoke --out workspace/a4-synthetic-smoke
```

These are illustrative paths. Resource paths in the JSON config resolve
relative to that config file. `train` requires a fresh, empty output directory.
It saves a selected `best.pt`, convergence record, VAL preview and effective
config, then invokes a fresh process through the same memory-safe entry point
to restore the selected head and replay VAL probabilities. Independent restore
is an inference verification, not continuation of optimization. Optimizer and
scheduler states are not saved for training resume.

TEST evaluation requires confirmed convergence, the exact sealed checkpoint
digest, matching code/config/cache-manifest bindings, and a successful
independent restore report bound to that checkpoint. The training process never
constructs the TEST partition. It does bind the TEST cache manifest identity,
which contains provenance rather than held-out response labels. Failed or
nonconverged runs do not unlock TEST.

Run seals intentionally bind resolved paths and implementation hashes. Moving
an already sealed run, changing configuration/code or substituting a cache will
not transparently restore under the old seal. This portable source release is
for reproducible new runs; it does not claim hash identity with historical
sealed artifacts or provide a bypass for their checks.

## External model and extracted components

DECIDER source code, model weights and tokenizer assets are not bundled. Supply
`decider_snapshot`, `tokenizer_snapshot` and `decider_source` explicitly (the
latter two default to the snapshot directory). The source root must contain
`decider/model.py`, `decider/prompt.py` and `decider/__init__.py`. The loader
rejects a previously imported DECIDER package from another location, verifies
every backbone tensor's name/shape/dtype, and retains the audited Qwen3.5 Triton
convolution path. See the environment documentation for its CUDA dependencies.

Only locally sourced helper functions actually called by A4 are retained under `vendor/`:
the frozen loader, contiguous layer executor, joint pooling/batching and fresh
task-head factory. Historical training, LoRA, calibration and standalone CLIs
were removed. `vendor/source_manifest.json` records original file hashes,
retained functions and hashes of this portable implementation. No original
third-party model snapshot source or weights were copied into this package.
Local provenance alone does not establish authorship or redistribution rights;
see `THIRD_PARTY_NOTICES.md` and the release audit before publication.
