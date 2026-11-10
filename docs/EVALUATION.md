# Evaluation and comparison contract

Each cell line is trained independently. Hold out complete perturbation targets
with train/validation/test ratio 70/10/20 and seeds **17, 29, 43, 61, 89**.
These are five repeated random holdouts, not disjoint five-fold cross-validation.
Internal TRAIN evidence uses a separate five-fold exclusion procedure. Labels
are Up=0, Same=1, Down=2; unresolved pairs are excluded, not converted to Same.

Select S1Pert checkpoints on validation Macro-F1 (NLL then earlier epoch breaks
ties), require the declared convergence gate, restore independently, then open
TEST. Baselines reuse exactly these target partitions and record order. Native
expression predictors fit class thresholds only on validation; language baseline
training budgets and adaptations are listed in BASELINES.md.

## Saved predictions

The shared evaluator requires an NPZ containing `target`, `query`, and `labels`
arrays that match `test/records.npy` exactly. Include `y_pred` plus real
`probabilities` of shape [N,3] in [Up,Same,Down] order, or a continuous
`delta_expr` score for native expression predictors. It accepts `y_pred=-1`
for invalid generated answers, counting them as errors. Hard-only predictions
receive classification metrics; their AUCs remain unavailable.

The S1Pert and language adapters already export this schema. Classical adapters
export separate arrays; convert them with exact pair-key verification:

```bash
python scripts/import_predictions.py \
  --records workspace/a4/data/hepg2/seed_17/test/records.npy \
  --keys workspace/classical/runs/gears/hepg2/seed_17/test_keys.npy \
  --hard workspace/classical/runs/gears/hepg2/seed_17/test_predictions.npy \
  --effect workspace/classical/runs/gears/hepg2/seed_17/test_effect.npy \
  --out workspace/predictions/GEARS/hepg2/seed_17/test_predictions.npz
```

For GenePT use `--probabilities .../test_probabilities.npy` in place of
`--effect`. Do not convert hard labels to one-hot probabilities to manufacture
ranking or calibration metrics. STATE keeps native continuous effects; SUMMER
keeps parsed decisions and saved text. AUC requires real scores, not text labels.

## Tables

Copy `configs/predictions.example.csv`, register one row per method/cell/seed,
and point each row at its actual output. CSV paths resolve relative to the
registry. Evaluation checks all identities before writing a new output folder.

```bash
s1pert compare --registry configs/predictions.local.csv --out workspace/comparison
# Equivalent: python -m jev_a4.evaluation --registry ... --out ...
```

`per_split.csv` contains Macro-F1, balanced accuracy, accuracy, invalid-output
rate, class-wise precision/recall/F1/support, within-gene Chg-AUC and Pol-AUC.
`summary.csv` contains means, sample SD and sample variance (`ddof=1`), valid
split counts and registered split counts, by cell and ALL_CELLS. Register all
five seeds per cell before reporting a completed 20-split method.

Macro-F1 averages three class F1 values; absent true-and-predicted classes
contribute zero. BA averages recall over classes present in ground truth.
Precision uses zero when no positives are predicted. These conventions match
the original shared evaluator.

Chg-AUC ranks changed vs Same targets **within each query gene** using
`p_up+p_down` (or `abs(delta_expr)`). Pol-AUC uses only true changed pairs and
ranks Up vs Down using `p_up/(p_up+p_down)` (or signed `delta_expr`). Only genes
with both binary classes contribute; eligible gene AUCs are equally averaged
within each split. Report the eligible-query counts. Undefined AUC stays blank.

Do not interpret 20 repeated splits as 20 independent biological datasets.
Any paired hypothesis test must align method, cell and split keys, state its
comparison family, and correct that prespecified family. This source release
does not regenerate the historical manuscript's significance stars.
