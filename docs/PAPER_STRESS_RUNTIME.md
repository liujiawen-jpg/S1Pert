# Prior-stress evaluation and runtime benchmarks

These stages consume the prepared workspace and the predictions produced by the
packaged training/inference programs. Every plotted value is computed from raw
per-pair predictions or an executed timing phase.

## Workspace and prediction contract

Set one workspace root, or pass `--work-dir` to each command:

```bash
export S1PERT_ROOT=/path/to/workspace
```

Stage 3 produces `data/CELL/seed_SEED/{train,test}/records.npy`, sealed landmark
caches, selected checkpoints, and baseline predictions. Register completed test
predictions in `paper/predictions.csv`:

```csv
method,cell,seed,records,predictions
S1Pert,hepg2,17,../data/hepg2/seed_17/test/records.npy,../runs/main/hepg2/seed_17/test_predictions.npz
```

The complete registry contains each available method on four cell lines and the
five seeds 17, 29, 43, 61, 89. Paths in the CSV are relative to that CSV. The
records array is structured with `target`, `query`, and `label` fields. Each
prediction NPZ contains aligned `target`, `query`, and `labels`, plus either:

- `probabilities` or `probs`, with columns Up, Same, Down;
- `y_pred` and native continuous `effect` or `delta_expr`;
- native `y_pred` only, for classification-only methods.

Native invalid answers (`-1`) remain errors in classification metrics. Hard
labels alone provide no continuous ranking score, so their AUC is unavailable.
The evaluator verifies exact sample order and labels, unique target–query pairs,
and disjoint TRAIN/test target sets. Optional `train_records` and
`prior_features` registry columns override the corresponding workspace paths.

## Stage 5: raw predictions → prior-stress metrics → Fig. 1

```bash
python experiments/prior_stress.py --work-dir "$S1PERT_ROOT"
python plotting/plot_fig1_prior_stress.py --work-dir "$S1PERT_ROOT"
```

The evaluator constructs P1 directly from TRAIN labels. For each query gene it
adds one observation to each class count, then normalizes. Zero-support genes
use the TRAIN-global smoothed class frequencies. If the prepared 12-dimensional
test evidence exists, its first three entries must match these probabilities.
P1 is saved as an independent comparator; it is not added to S1Pert predictions.

The output directory is `paper/prior_stress/`:

| File | Contents |
|---|---|
| `per_split.csv` | Metrics for every method, cell line, seed, and stratum |
| `summary.csv` | Per-cell and ALL_CELLS means, sample SD, paired tests |
| `input_provenance.csv` | Input hashes, ranking availability, entropy cutoffs |
| `predictions/P1/…/test_predictions.npz` | Newly computed P1 predictions |
| `manifest.json` | Protocol and aggregation identity |

### Ranking and strata

Change AUC uses `p_up + p_down` and compares changed against unchanged pairs for
each fixed query gene across held-out targets. Polarity AUC uses
`p_up / max(p_up + p_down, 1e-30)` on truly changed pairs and compares Up against
Down. This zero-denominator convention matches the historical evaluator,
including ALIGNED's valid `[0, 1, 0]` outputs. Continuous predictors instead use
`abs(effect)` and signed `effect`. The reported AUC is the unweighted mean of
defined per-query AUCs; queries lacking both relevant classes are excluded from
that AUC only. Valid query counts are saved.

The historical entropy protocol uses **fixed normalized entropy thirds**:
`H < 1/3`, `1/3 <= H < 2/3`, and `H >= 2/3`, where
`H = -sum(p log p) / log(3)`. These are equal-width bins, not empirical tertiles.
Use the following separate protocol to study actual quantile strata:

```bash
python experiments/prior_stress.py --work-dir "$S1PERT_ROOT" \
  --entropy-mode train-quantiles --output "$S1PERT_ROOT/paper/prior_stress_quantiles"
```

Quantile cutoffs are fitted on supported TRAIN query genes, without using test
labels. The manifest distinguishes the two protocols. A paper describing the
historical result should use “normalized entropy thirds” or “entropy bins.”

Counter-propensity pairs have truth different from the TRAIN-majority class.
Support strata use counts of resolved TRAIN target–query pairs: at least 50,
10–49, 1–9, and zero. Test labels determine evaluation truth and the reversal
mask; they never determine the historical evidence or entropy cutoffs.

Fig. 1 displays S1Pert, P1, ALIGNED, GenePT-Prot, STATE, and GEARS. Panel **a**
shows overall Macro-F1 against Pol AUC, with Chg AUC labels. Panel **b** shows
entropy-stratified Macro-F1 and counter-propensity Down-F1 on separate axes.
Panel **c** shows support-stratified Macro-F1. Means and sample SD come from the
20 cell/seed results. The plotting default requires all 20 splits for every
displayed metric. Empty or undefined strata remain missing rather than zero.

Paired two-sided Wilcoxon tests use matching cell/seed results. Holm correction
includes every comparator registered with the evaluator for a given
cell/stratum/metric. Selecting six plotting methods does not shrink that family.
The five target-held-out splits overlap and are not five disjoint CV folds or
independent biological replicates.

## Stage 7: executed timing phases → Fig. 3

### Separate method environments

Use separate environments for A4/ALIGNED and SUMMER. The tested A4 backbone
stack uses Transformers 5.17.0; SUMMER's vLLM 0.10.2 stack uses Transformers
4.56.2. Loading both stacks into one Python process is not supported. Create
`configs/runtime_python_map.json` with the actual installed interpreter paths:

```json
{
  "S1Pert": "/path/to/a4-environment/bin/python",
  "ALIGNED": "/path/to/aligned-environment/bin/python",
  "SUMMER": "/path/to/summer-environment/bin/python"
}
```

Each environment needs the package's shared scientific dependencies, in addition
to its method-specific requirements. Paths can be absolute or relative to the
mapping file. Interpreter symlinks are preserved so virtual environments remain
active. The map is passed to the timing runner, not the recipe builder.

Mapped factories execute in isolated method processes. Their online timer starts
after importing packages and restoring the model, with GPU synchronization
inside the worker. Process launch, imports, cleanup, and result transfer are
recorded separately as `worker_overhead_seconds`: excluded from loaded-model
online latency, included in end-to-end cost. The actual worker interpreter,
package versions, source hashes, warmups, and individual timings are recorded.
Fresh command phases replace `{python}` with the same method interpreter and
retain their load-inclusive wall-time scope. Omitting `--python-map` preserves
the in-process behavior for an intentionally compatible single environment.

### Concrete online benchmark

After S1Pert, ALIGNED, and SUMMER have completed their native workflows, write an execution
plan using the actual HepG2/K562 test sizes:

```bash
python experiments/runtime_recipes.py --work-dir "$S1PERT_ROOT" --mode online
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 python experiments/runtime.py \
  --work-dir "$S1PERT_ROOT" --plan "$S1PERT_ROOT/paper/runtime_online_plan.json" \
  --python-map "$S1PERT_ROOT/configs/runtime_python_map.json" \
  --output "$S1PERT_ROOT/paper/runtime_online"
```

This **runs real full-test inference**. It may be long for SUMMER. The plan
builder itself only writes JSON and starts no models.
Select the same physical GPU in the launch environment and the `gpu_uuid`
fields of the A4/SUMMER configurations.

The A4 factory restores the converged, independently verified checkpoint through
`experiments.ablation.load_checkpoint`. It replays saved native predictions on
a small prefix when available, then measures the actual decoder forward pass,
including feature gathering and device synchronization. Model restoration is
recorded separately from loaded-model online latency.

The SUMMER factory loads the initialized baseline workspace, Qwen3-8B, and the
identity-matched public-only summary cache. It recreates native retrieval and
prompts, generates new rationales and decisions through vLLM, and includes
retrieval/tokenization in measured online time. It never reads saved test
answers. Prefix KV state is reset before each timed repetition; native prefix
caching remains enabled within a test pass. `PYTHONHASHSEED=0` stabilizes the
native retrieval ordering. The synchronous vLLM generation call completes its
worker GPU work before returning.

Online-only plans do not provide complete S1Pert end-to-end cost: their training
and offline encoding phases are absent. The timing log records this explicitly.

### Fresh end-to-end benchmark

```bash
python experiments/runtime_recipes.py --work-dir "$S1PERT_ROOT" --mode fresh
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 python experiments/runtime.py \
  --work-dir "$S1PERT_ROOT" --plan "$S1PERT_ROOT/paper/runtime_fresh_plan.json" \
  --python-map "$S1PERT_ROOT/configs/runtime_python_map.json" \
  --output "$S1PERT_ROOT/paper/runtime_fresh"
```

The section launcher forwards the same map:

```bash
python experiments/run_section.py fig3 --work-dir "$S1PERT_ROOT" \
  --runtime-mode fresh --python-map "$S1PERT_ROOT/configs/runtime_python_map.json" \
  --execute
```

Fresh mode writes into new `paper/runtime_cache`, `paper/runtime_models`,
`paper/runtime_aligned_prepared`, `paper/runtime_aligned`, and
`paper/runtime_summer` directories. It executes the real all-five-seed A4
landmark encoder, trains a new task head to the configured validation-convergence
criterion, and measures complete test inference. Offline encoding cost is
divided by five to state the five-split reuse scenario explicitly. A completed
cache cannot be substituted for the measured cold build.

For SUMMER, fresh mode initializes a new native baseline workspace and actually
generates its public-only gene summaries before test inference. The measured
summary-build cost is also amortized across five splits. SUMMER has no
task-training phase. Factory/model setup contributes to end-to-end time even
though it is excluded from the loaded-model online region.

ALIGNED's fresh recipe explicitly prepares public-graph inputs in a new
`runtime_aligned_prepared` directory, then trains against that exact directory.
This preparation is measured once for its split. Its native training includes
the two-round abductive alignment/refinement protocol. Loaded-model inference
uses the final trained neural probabilities and the learned symbolic switch,
including final knowledge-base deductions, rather than pre-alignment scores.

The generated plans cover all three methods at seed 17 on HepG2 and K562. Run the builder with
`--seed 29` and a fresh output/workspace when additional measured splits are
required. Model quality comparison still uses the complete five-seed protocol.

### ALIGNED assets and final-prediction timing

The restored historical adapter is packaged in `baselines/aligned`. Its
preparation consumes the workspace's public relation and GO databases,
`assets/public_knowledge.sqlite` and `assets/public_go.sqlite`. Use the packaged
asset-building workflow before executing the benchmark. The timing factory is
`baselines.aligned.run:load_for_timing`; it restores the final-refinement model
and applies the native final alignment rule. Fig. 3 requires successful actual
measurements for S1Pert, ALIGNED, and SUMMER on both cell lines.

### Timing log contract

`runtime.py` accepts jobs containing `method`, `cell`, `seed`, `test_pairs`, and
`phases`. A phase has one execution interface:

- `command`: an argument array executed without a shell; wall time includes
  process startup and model loading;
- `factory`: `module:function`, returning `run`, `synchronize`, `n_pairs`, and
  optional `cleanup`, `device`, `metadata`; model initialization occurs outside
  the online timed region. CUDA factories must synchronize.

The runner supports `repeat`, `warmup`, `timeout_seconds` for command or isolated
factory subprocesses, and
`amortize_over` for offline caches. A shared `cache_id` reuses an actual measurement
within one invocation, with identical execution parameters required. Commands
and factory arguments can use `{work_dir}`, `{repo_dir}`, and `{python}`.

`timings.json` preserves every repetition, warmup, model setup time, command,
scope, environment, and failure. Isolated factories also retain their request,
result, and worker log. `per_split.csv` and `phases.csv` summarize these
measurements. Per-10,000-pair latency is the measured complete-pass time scaled
by `10000 / n_pairs`. End-to-end values require all relevant phases and the
complete test axis; a 10,000-row slice is not accepted as full-test cost.

```bash
python plotting/plot_fig3_runtime.py --work-dir "$S1PERT_ROOT" \
  --timing "$S1PERT_ROOT/paper/runtime_fresh/timings.json"
```

Fig. 3 labels loaded-model online timings and load-inclusive command timings
differently and refuses a mixture of these scopes. Its values come exclusively
from successful timing logs. Sample SD is shown when multiple splits were
measured; a single split receives no fabricated error bar.

## Fonts and lightweight verification

Both figure scripts use an installed Times New Roman when available, with
DejaVu Serif as the configurable fallback. Font files are not bundled. Select
another installed family with `--font` and `--font-fallback`.

```bash
python -m pytest -q tests/test_paper_stress_runtime.py
```

These CPU tests create artificial per-pair predictions, check alignment/leakage
guards and metrics, derive the prior-stress summaries, render both figures, and
execute short subprocess timing phases, including isolated factories and
per-method interpreter selection. Synthetic test outputs are mechanics checks,
separate from paper results.
