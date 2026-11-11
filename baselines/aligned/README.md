# ALIGNED-MLP: public-signed-KG A4 adaptation

This directory contains the actual six-stage implementation used for the
independent ALIGNED comparison, ported to the release workspace. All seven
core source hashes matched the supplied source-identity record before porting.
The source was available in another execution environment; the earlier release
inspection did not have that environment's path visibility.

The upstream implementation is [ALIGNED](https://github.com/yfxiang0112/Aligned)
by Yuanfang Xiang and Lun Ai, commit
`264e26e3fb826799c7f5e7637fa776d3feca4553`, under CC BY 4.0. The required official
Python files retain their exact bytes and Git-blob/SHA256 checks in
`official_source/core_provenance.json`; the upstream license notice is included
in `official_source/LICENSE`. Cite *Adaptive Data-Knowledge Alignment in Genetic
Perturbation Prediction*, ICLR 2026. Wrapper adaptations described below are
release changes, not upstream endorsements.

## Reproduce

Run from the repository root with the package installed. First prepare the
four cells' shared strict data and five whole-target splits. Build the public
knowledge inputs with `python -m data_prep.build_public_knowledge --help` and
follow its explicit fetch/build commands. ALIGNED consumes:

- `W/assets/public_knowledge.sqlite`: genes and public signed/directed edges;
- `W/assets/public_go.sqlite`: direct GO annotations with qualifiers;
- `W/data/C/`: canonical target/query identities and sealed strict records.

```bash
python -m baselines.aligned.run prepare --work-dir W --cell hepg2 --seed 17
python -m baselines.aligned.run train --work-dir W --cell hepg2 --seed 17 --device cuda
python -m baselines.aligned.run restore --work-dir W --cell hepg2 --seed 17 --device cuda
python -m baselines.aligned.run evaluate --work-dir W --cell hepg2 --seed 17 --device cuda
```

Restore runs in a process independent of training and checks full-VAL outputs
exactly. Evaluate repeats that gate and opens TEST only after it passes.
Train/evaluate must be separate processes. A failed or partial run is never
an accepted result. Resume requires explicit `--resume` and unchanged source,
config and data identities. Optional `--deadline` accepts a new
timezone-qualified ISO8601 date; no old queue or GPU schedule is restarted.

Defaults are `W/baselines/aligned/prepared/C/seed_S` and
`W/baselines/aligned/runs/C/seed_S`. `--prepared-root` changes the parent
containing C/seed_S; `--output` changes the exact run directory. Training
validates existing preparation or creates it when absent. `--public-db` and
`--go-db` select explicit public-only databases. `S1PERT_ROOT` is equivalent
to `--work-dir`.

The default config is `W/configs/aligned.json` when supplied, otherwise
`configs/official_default_a4.json` beside this README. Repeat for HepG2, Jurkat,
RPE1 and K562 essential, seeds 17, 29, 43, 61, 89. The `smoke_e2e` profile permits
only VAL restoration; its TEST gate is closed. Reduced `budgeted` schedules
are labelled separately from the full paper schedule.

## Preserved method and adaptation

The original scientific engine is byte-identical. It retains the official
hidden-64 MLP, learned R adaptor, REINFORCE, two-round pseudo-label regeneration
and knowledge refinement, and fixed post-refine1 endpoint. The full schedule
uses up to 1000 R-initialization iterations, 300 pretrain iterations, then two
rounds of 150×100 retrain iterations and up to 5000 refinement iterations with
original tolerances. Batch size is 64. The probabilities-to-CE behavior,
per-batch EMA, fixed REINFORCE permutation and nonnegative refinement clamp
are preserved, including implementation-specific behavior.

The A4 adaptation uses the complete sorted canonical target/query union,
CRISPRi=-1 one-hot inputs and Up/Same/Down=+1/0/-1 labels. Missing labels are
-100 and masked in supervised CE. Only TRAIN targets enter pseudo-label
training and refinement. Initial label weights combine observed TRAIN
agreement, signed incoming degree and direct non-NOT GO counts with
coefficients 0.5/0.2/0.3. VAL and TEST neither determine these weights nor
select a stage.

Graph mapping is exact and unambiguous. Only directed positive or negative
edges survive; conflicting signs, unknown signs and self edges are excluded.
Unsigned association is never assigned an invented sign. Newly downloaded
public snapshots may differ from the historical snapshot; builder and prepared
reports retain source hashes and graph-coverage counts.

Portable changes replace private paths, replace the historical completed-GEARS
identity dependency with shared strict preparation seals, reuse common metrics,
and integrate the inspected final probability exporter. Original hashes remain
in `source_availability.json`; new checkpoints bind ported source hashes rather
than claiming unchanged historical identity. Historical weights, logs and
numeric results are not bundled.

## Probabilities and runtime

`test_predictions.npz` retains every strict TEST record's target, query, labels,
final prediction, probabilities, neural_probs, R, and diagnostic predictions.
Canonical probabilities are final neural softmax in [Up, Same, Down] order
when R≤0.5; when R>0.5 they are the point mass of the final KB deduction.
The exported argmax must exactly match the original hybrid decision.
Unswitched neural probabilities remain continuous; all rows are not converted
to one-hot outputs. These hybrid scores are not claimed to be calibrated.

This is the independent baseline's native neural/knowledge rule. No ALIGNED
prediction, KB output or R switch is connected to S1Pert.

`baselines.aligned.run:load_for_timing` restores outside the timed region and
returns live inference, synchronization, pair count and cleanup callbacks.
Each timed call computes the final MLP, R, KB matrix deduction and hybrid
scores; saved predictions are never replayed. Fresh preparation can use a new
`--prepared-root` to measure its cost explicitly.

Dependencies: NumPy, SciPy, pandas, scikit-learn, NetworkX, PyTorch and PyTorch
Geometric (imported by the unchanged official learner). The small CPU tests in
`tests/test_aligned_portable.py` cover both rounds, separate-process restoration
and continuous-score export using artificial data only.
