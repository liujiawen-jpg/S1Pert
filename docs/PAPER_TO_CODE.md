# Paper-to-code execution map

This map links every requested paper artifact to its experiment producer and
renderer. Source coverage, synthetic engineering validation and new biological
reproduction are separate records. `scripts/audit_paper_coverage.py` checks the
seven-stage source map without importing a model or starting an experiment.

## Summary

| Paper artifact | Upstream entry point | Aggregation / renderer | Workspace output |
|---|---|---|---|
| Algorithm 1; Table III | `jev_a4.data.preprocess`, `jev_a4.labels`; `experiments.paper data` and `prepare` | `plotting.make_table3_stats` | `strict_labels`, `paper/table3` |
| Multilevel caching | `jev_a4.core.prepare`, `jev_a4.core.encode` | Source/cache seals | `data`, `cache` |
| Table II | `experiments.main_table2`; actual nine method adapters | `plotting.make_table2_main` | `paper/table2` |
| Table IV | `experiments.ablation` | `plotting.make_table4_ablation` | `paper/table4` |
| Fig. 1 | `experiments.prior_stress` | `plotting.plot_fig1_prior_stress` | `paper/prior_stress` |
| Fig. 2 | `data_prep.cell_pool`, `experiments.cell_pool` | `plotting.plot_fig2_label_boundary` | `cell_pool`, `paper/fig2` |
| Fig. 3 | `experiments.runtime_recipes`, `experiments.runtime` | `plotting.plot_fig3_runtime` | `paper/runtime_fresh` |

Paths denote modules or directories in this repository. Individual renderers
write manifests naming their input files, statistics and generated PDF/PNG paths.
The source archive includes no real-result table from which to fabricate a run.

## Stage 1 — statistical truth and shared experiment inputs

`src/jev_a4/data/preprocess.py` streams original counts from the four public matrices;
`src/jev_a4/labels/` implements statistical adjudication. Commands and exact file
schemas are in `DATA_PIPELINE.md`.

The scientific contract is linear CP10k, scaled Welch Up/Down tests, TOST with
rho 1.2, alpha 0.05, joint 3G BH, and final support/nonself filtering. Invalid
test moments contribute p=1. The joint testing family includes genes before
the final support filter: changing this order changes q-values. Unresolved
is stored separately from Same and excluded from supervised three-class labels.

`src/jev_a4/core/prepare.py` creates per-cell target-held-out 7:1:2 records for
17/29/43/61/89. Its TRAIN target folds build out-of-fold historical evidence;
held-out target truth cannot be retrieved as an input. The 12-number evidence,
three-tier provenance, four basal covariates, public functions and 108+1 evidence
cards are bound to the preparation seal. Main-table baselines reuse these target
partitions and TEST row identities.

Table III reads resolved label arrays and derives Up/Same/Down counts across
the four full prepared cell datasets. It does not sum five overlapping splits.

## Stage 2 — immutable generic backbone, reusable landmarks

`src/jev_a4/core/encode.py` and its vendored primitives use the pinned DECIDER
native tokenizer/source. Backbone tensors are frozen. The production dimensions
are 32 layers × width 2560. Single-stream encoding is split at layer 24; Gamma16
retains 12 segment means and four tail states, Gamma4 three means and one tail.
Condition/query/evidence provide 16+16+4 tokens for the joint upper-eight-layer
forward. Pooling produces four joint summaries, with separate present/absent
evidence caches. Absence is applied before joint interaction.

The cache identity includes source, model, input and configuration hashes. New
workspace paths can be configured freely before construction. Moving sealed
historical caches or changing source/model mtimes is not automatically accepted
as an unchanged run; the historical absolute-path identity checks remain active.

## Stage 3 — actual nine-method execution, raw predictions, Table II

Core A4 code lives in `src/jev_a4/core/train.py` (including evaluation),
`src/jev_a4/core/pipeline.py` and `src/jev_a4/core/vendor/prior_model.py`.
The decoder is freshly initialized, width 512, two blocks, with a zero-initialized
internal joint cross-attention projection. Change/polarity probabilities produce
the final three classes by argmax. Ranking weight is 0.02, margin 0.05; evidence
dropout is 0.10 and synchronized across corresponding inputs. Converged validation
selection and independent-process restoration precede accepted TEST reporting.

Baseline source locations:

| Method | Implementation |
|---|---|
| GenePT-Gene / GenePT-Prot / GEARS / scGPT | `baselines/classical/` |
| Qwen3-4B-LoRA | `baselines/language/qwen3_lora.py`, graph builder alongside |
| STATE | `baselines/state/` |
| SUMMER | `baselines/summer/`, common public-edge converter |
| ALIGNED | `baselines/aligned/`, byte-preserved engine and pinned official core |
| P1 (stress reference) | `scripts/export_prior.py`, `experiments/prior_stress.py` |

ALIGNED uses six stages: initialization, pretraining, retrain0/refine0,
retrain1/refine1. It exports final native hybrid probabilities, preserving the
continuous neural branch and the learned knowledge switch. Public signed-edge
and accession mapping row digests were checked against the historical consumed
inputs. Its exact external source, license, engine hashes, portable adapter
changes and dependency list are included. Its native hybrid rule belongs to an
independent comparison model, never to S1Pert's output.

The main table checks ordered target/query/label keys, three-class column order,
finite valid probabilities where available, and a complete 20 splits/method.
Means and sample SD derive from individual split metrics; class metrics are not
replaced with accuracy. Separate environments are selected by `--python-map`.

## Stage 4 — fresh component removal and fixed-model diagnostic

`experiments/ablation_runtime/` preserves the actual model factory, partition
logic and training code; `source_provenance.json` records inspected source hashes.

- NO_JEV removes semantic and joint states, retaining numeric condition/query
  covariates and 12-dimensional historical evidence.
- NO_JOINT removes only joint summaries and their decoder interaction.
- FLAT changes the head to softmax CE and retains the historical ranking losses
  through its logit bridge; it is not a simultaneous rank-loss ablation.
- NO_RANK sets the ranking weight to zero.
- NO_MASK sets synchronized evidence dropout to zero.

Every variant is fitted from fresh task weights on K562's five original splits.
Target permutation restores the full model, shuffles target identities within
query, leaves query/evidence fixed and reuses the appropriate recombined caches.
Permutation can retain fixed points by chance; it is not a derangement or a new
training run. The full-model reference must pass the same checkpoint gates.

## Stage 5 — per-query ranking and prior stress

P1 is recomputed from TRAIN query-level class counts with add-one smoothing and
a TRAIN-global fallback at zero query support. It must match the prepared TEST
probability evidence. Its exported predictions are an independent reference.

Chg AUC is the mean defined per-query AUC for changed versus Same across TEST
targets. Pol AUC is the corresponding Up-versus-Down AUC among truly changed
pairs. For probability models, scores are p_up+p_down and
p_up/max(p_up+p_down,1e-30). Continuous expression models use absolute and signed
effects. Queries with only one relevant truth class have undefined AUC and are
excluded only from that AUC, with counts retained.

Historical normalized entropy bins are [0,1/3), [1/3,2/3), [2/3,1]. The paper
request called them tertiles; the packaged default preserves the actual fixed
bins, and the optional TRAIN-quantile protocol is separately identified.
Counter-propensity uses truth different from the TRAIN-majority class; Down-F1
is displayed in its own panel region. Support counts use resolved TRAIN pairs,
not total cells. Paired Wilcoxon tests and Holm correction include the registered
comparator family before six-method display selection.

## Stage 6 — disjoint cell-pool labels and frozen boundary predictions

`data_prep/cell_pool.py` partitions original cells with the historical seeded,
batch-stratified nearest-one-third allocation and derives both pools' moments.
The candidate target universe is the union of five held-out TEST target sets,
with common controls, so the same A/B cells are shared across split-specific
evaluation. TRAIN/VAL/TEST targets themselves are not resplit here.

The statistical and naive algorithms evaluate Pool A using Pool B as independent
response evidence. Reference-DE fates and concordance require the specified
support; the coverage accounting also retains smaller Pool-A groups. Cell-count
strata refer to Pool A. All counts, including historical reference totals, are
derived. Boundary prediction restores frozen A4 heads and computes required
generic landmark combinations without task fitting. The original full-control
input context remains; Pool-B perturbed labels are used only for scoring.

The complete detailed denominator, population and nesting definitions are in
`CELL_POOL.md`. Fig. 2 is not presented as training a new model on Pool A.

## Stage 7 — measured time, not historical constants

Runtime recipes specify actual setup, preparation, training and inference phases
for S1Pert, ALIGNED and SUMMER on HepG2/K562. Fresh mode creates separate cold
cache/model directories. Native live factories compute predictions on every
call; saved predictions may be used for a parity check, never as timed output.

Offline S1Pert encoding and SUMMER public-summary construction are measured once
and explicitly amortized over five splits. Setup is included in end-to-end cost
and excluded from loaded-model online cost. ALIGNED includes both rounds and
its final knowledge inference. Per-10,000 latency scales a complete test pass.
Fig. 3 rejects inconsistent online timing scopes and incomplete end-to-end logs.
One measured split has no invented standard deviation.

## Reproduction and release boundaries

The downloadable source contains the seven experiment producers and renderers,
not downloaded model weights, task checkpoints, biological matrices or private
databases. Public asset manifests and builders are included. Full biological
results require execution using recorded inputs; source coverage and synthetic
tests do not assert all published numbers have been regenerated during packaging.

Reconstruction differences already documented in the asset guide include live
UniProt snapshots, scGPT's public rebuild pin versus its historical unversioned
copy, STATE alias coverage and SUMMER's rebuilt public relation database. Each is
fingerprinted; none is silently claimed byte-identical to an unavailable asset.
Shared target partitions and labels remain the comparison contract.

The top-level main pipeline is a new-run launcher. Completed checkpoints support
independent restoration, not arbitrary optimizer continuation. Inspect incomplete
preparation/cache directories before rerunning them; the historical core can
replace incomplete arrays and explicitly repeated evaluation outputs. The source
packager itself refuses an existing destination and validates every archive hash.
