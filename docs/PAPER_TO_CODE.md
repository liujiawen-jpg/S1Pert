# Paper-to-code execution map

Use this map to locate the experiment producer and renderer for each paper
artifact. Source coverage, synthetic engineering validation and new biological
reproduction are recorded separately. `scripts/audit_paper_coverage.py` checks
the seven-stage source map without importing a model or starting an experiment.

## Summary

| Paper artifact | Upstream entry point | Aggregation / renderer | Workspace output |
|---|---|---|---|
| Algorithm 1 and Table III | `jev_a4.data.preprocess`, `jev_a4.labels`, `experiments.paper data` and `prepare` | `plotting.make_table3_stats` | `strict_labels`, `paper/table3` |
| Multilevel caching | `jev_a4.core.prepare`, `jev_a4.core.encode` | Source/cache seals | `data`, `cache` |
| Table II | `experiments.main_table2` and the nine method adapters | `plotting.make_table2_main` | `paper/table2` |
| Table IV | `experiments.ablation` | `plotting.make_table4_ablation` | `paper/table4` |
| Fig. 1 | `experiments.prior_stress` | `plotting.plot_fig1_prior_stress` | `paper/prior_stress` |
| Fig. 2 | `data_prep.cell_pool`, `experiments.cell_pool` | `plotting.plot_fig2_label_boundary` | `cell_pool`, `paper/fig2` |
| Fig. 3 | `experiments.runtime_recipes`, `experiments.runtime` | `plotting.plot_fig3_runtime` | `paper/runtime_fresh` |

Each path identifies a module or directory in this repository. Every renderer
writes a manifest with its input files, statistics and generated PDF/PNG paths.
The source archive does not contain precomputed real-result tables. Each run must
generate its own results.

## Stage 1 — statistical truth and shared experiment inputs

`src/jev_a4/data/preprocess.py` streams original counts from the four public
matrices. `src/jev_a4/labels/` implements statistical adjudication. See
`DATA_PIPELINE.md` for commands and exact file schemas.

The implementation uses linear CP10k, scaled Welch Up/Down tests, TOST with rho
1.2, alpha 0.05, joint 3G BH, and final support/nonself filtering. Invalid test
moments contribute p=1. The joint testing family includes all genes before the
final support filter. Changing this order changes the q-values. Unresolved labels
are stored separately from Same and excluded from supervised three-class labels.

`src/jev_a4/core/prepare.py` creates target-held-out 7:1:2 records for each cell
and seed in 17/29/43/61/89. TRAIN target folds produce out-of-fold historical
evidence. Held-out target truth is never available as an input. The preparation
seal binds the 12-number evidence, three-tier provenance, four basal covariates,
public functions and 108+1 evidence cards. Main-table baselines reuse the same
target partitions and TEST row identities.

Table III reads the resolved label arrays and computes Up/Same/Down counts across
the four complete prepared cell datasets. It does not sum the five overlapping
splits.

## Stage 2 — immutable generic backbone, reusable landmarks

`src/jev_a4/core/encode.py` and its vendored primitives use the pinned native
DECIDER tokenizer and source. All backbone tensors remain frozen. Production uses
32 layers with width 2560. Single-stream encoding splits at layer 24. Gamma16
retains 12 segment means and four tail states. Gamma4 retains three means and one
tail state.

Condition, query and evidence inputs provide 16+16+4 tokens for the joint forward
pass through the upper eight layers. Pooling produces four joint summaries.
Separate caches store the evidence-present and evidence-absent versions. Evidence
removal occurs before joint interaction.

Cache identity includes hashes for the source, model, input and configuration.
Configure new workspace paths before cache construction. Moving a sealed historical
cache does not automatically preserve its run identity. Changes to source or model
mtimes also invalidate the identity. Historical absolute-path checks remain active.

## Stage 3 — actual nine-method execution, raw predictions, Table II

Core A4 code is in `src/jev_a4/core/train.py` (including evaluation),
`src/jev_a4/core/pipeline.py` and `src/jev_a4/core/vendor/prior_model.py`. The
freshly initialized decoder has width 512 and two blocks. Its internal joint
cross-attention projection starts at zero. Change and polarity probabilities
produce the final three classes by argmax. The ranking weight is 0.02 and the
margin is 0.05. Evidence dropout is 0.10 and remains synchronized across matching
inputs. Accepted TEST reporting requires converged validation selection followed
by restoration in an independent process.

Baseline source locations:

| Method | Implementation |
|---|---|
| GenePT-Gene / GenePT-Prot / GEARS / scGPT | `baselines/classical/` |
| Qwen3-4B-LoRA | `baselines/language/qwen3_lora.py`, graph builder alongside |
| STATE | `baselines/state/` |
| SUMMER | `baselines/summer/`, common public-edge converter |
| ALIGNED | `baselines/aligned/`, byte-preserved engine and pinned official core |
| P1 (stress reference) | `scripts/export_prior.py`, `experiments/prior_stress.py` |

ALIGNED runs six stages: initialization, pretraining, retrain0, refine0, retrain1
and refine1. It exports the final native hybrid probabilities and preserves both
the continuous neural branch and learned knowledge switch. Public signed-edge and
accession-mapping row digests were checked against the historical consumed inputs.
The release records the exact external source, license, engine hashes, portable
adapter changes and dependency list. The native hybrid rule belongs only to the
independent comparison model. S1Pert never uses it as an output rule.

The main table checks ordered target/query/label keys, three-class column order,
finite valid probabilities when available, and all 20 splits for every method.
Means and sample SD come from individual split metrics. Accuracy does not replace
class-level metrics. Use `--python-map` to select separate environments.

## Stage 4 — fresh component removal and fixed-model diagnostic

`experiments/ablation_runtime/` preserves the model factory, partition logic and
training code. `source_provenance.json` records the inspected source hashes.

- NO_JEV removes semantic and joint states. It retains numeric condition/query
  covariates and 12-dimensional historical evidence.
- NO_JOINT removes only joint summaries and their decoder interaction.
- FLAT changes the head to softmax CE and retains the historical ranking losses
  through its logit bridge. It does not ablate the ranking loss.
- NO_RANK sets the ranking weight to zero.
- NO_MASK sets synchronized evidence dropout to zero.

Every variant starts from fresh task weights and fits the five original K562
splits. Target permutation restores the full model, shuffles target identities
within each query, keeps the query and evidence fixed, and reuses the appropriate
recombined caches. A permutation can retain fixed points by chance. It is neither
a derangement nor a new training run. The full-model reference must pass the same
checkpoint gates.

## Stage 5 — per-query ranking and prior stress

P1 is recomputed from TRAIN query-level class counts. It uses add-one smoothing
and a TRAIN-global fallback for queries with zero support. P1 must match the
prepared TEST probability evidence. Its exported predictions form an independent
reference.

Chg AUC is the mean of defined per-query AUC values for changed versus Same across
TEST targets. Pol AUC is the corresponding Up-versus-Down AUC among truly changed
pairs. Probability models use p_up+p_down and p_up/max(p_up+p_down,1e-30) as
scores. Continuous expression models use absolute and signed effects. A query
with only one relevant truth class has an undefined AUC. The calculation excludes
the query only from the affected AUC and retains its count.

Historical normalized entropy bins are [0,1/3), [1/3,2/3), [2/3,1]. The paper
request called them tertiles. The packaged default preserves the actual fixed
bins. The optional TRAIN-quantile protocol has a separate identifier.

Counter-propensity covers truth labels different from the TRAIN-majority class.
Down-F1 appears in its own panel region. Support counts use resolved TRAIN pairs,
not total cells. Paired Wilcoxon tests and Holm correction include the full
registered comparator family before selection of the six displayed methods.

## Stage 6 — disjoint cell-pool labels and frozen boundary predictions

`data_prep/cell_pool.py` partitions the original cells with the historical seeded,
batch-stratified nearest-one-third allocation. It then derives moments for both
pools. The candidate target universe is the union of the five held-out TEST target
sets. Common controls keep the same A/B cells across split-specific evaluations.
This stage does not resplit the TRAIN/VAL/TEST targets.

The statistical and naive algorithms evaluate Pool A with Pool B as independent
response evidence. Reference-DE fates and concordance require the specified
support. Coverage accounting also retains smaller Pool-A groups. Cell-count strata
refer to Pool A. Every count is derived, including historical reference totals.

Boundary prediction restores frozen A4 heads and computes the required generic
landmark combinations without task fitting. It retains the original full-control
input context. Pool-B perturbed labels are used only for scoring.

See `CELL_POOL.md` for the complete denominator, population and nesting definitions.
Fig. 2 does not train a new model on Pool A.

## Stage 7 — measured time, not historical constants

Runtime recipes define the measured setup, preparation, training and inference
phases for S1Pert, ALIGNED and SUMMER on HepG2/K562. Fresh mode creates separate
cold cache and model directories. Native live factories compute predictions on
every call. Saved predictions can support a parity check but never serve as timed
output.

Offline S1Pert encoding and SUMMER public-summary construction are measured once
and explicitly amortized over five splits. End-to-end cost includes setup.
Loaded-model online cost excludes it. ALIGNED includes both rounds and its final
knowledge inference. Per-10,000 latency is scaled from a complete test pass.
Fig. 3 rejects inconsistent online timing scopes and incomplete end-to-end logs.
A single measured split does not receive an invented standard deviation.

## Reproduction and release boundaries

The downloadable source contains all seven experiment producers and renderers.
It does not contain downloaded model weights, task checkpoints, biological
matrices or private databases. Public asset manifests and builders are included.
Full biological results require execution with the recorded inputs. Source
coverage and synthetic tests do not claim regeneration of every published number
during packaging.

The asset guide documents reconstruction differences for live UniProt snapshots,
the pinned public scGPT rebuild versus its historical unversioned copy, STATE alias
coverage and SUMMER's rebuilt public relation database. Each item has a fingerprint.
No rebuilt item is presented as byte-identical to an unavailable asset. Shared
target partitions and labels remain the comparison contract.

The top-level main pipeline launches new runs. Completed checkpoints support
independent restoration, not arbitrary optimizer continuation. Inspect incomplete
preparation and cache directories before rerunning them. The historical core can
replace incomplete arrays and outputs from an explicitly repeated evaluation.
The source packager refuses an existing destination and validates every archive
hash.
