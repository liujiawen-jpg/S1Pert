# ALIGNED source availability

The historical ALIGNED comparison is documented by a supplied twenty-split prediction export and its source digests. The referenced local adapter files (`aligned_engine.py`, `aligned_data.py`, `aligned_runtime.py`, and `aligned_runner.py`) and its exact upstream source copy were not found in the inspected accessible source trees. They are not included here. This directory is an availability record, not a runnable ALIGNED training implementation. Known source hashes are listed in `source_availability.json`.

The recorded prediction rule uses the final refined knowledge base on rows whose gate exceeds 0.5 and the final neural classifier elsewhere. In the benchmark's `[Up, Same, Down]` order, the export recorded:

```text
where(R > 0.5,
      one_hot(1 - final_KB.deduce(X)),
      final_neural_probs[..., [2, 1, 0]])
```

Consequently, some final probabilities are point masses. Replacing them with the ungated neural scores would change the measured comparator. This is the independent ALIGNED baseline's recorded behavior; no ALIGNED prediction, knowledge-base output, or gate feeds the S1Pert model.

Recovering the exact training source, upstream version and redistribution terms is required before a runnable historical reproduction can be supplied. Predictions, fitted weights and private run metadata are not bundled. No implementation was invented from the exported metrics.

If the original export is supplied separately, its `resolved.npz` files can be evaluated by the shared `python -m jev_a4.evaluation` command. Register each file together with its matching strict TEST `records.npy` in the evaluator's CSV (`method,cell,seed,records,predictions`). The evaluator verifies target/query/label order and uses the exported final probabilities, including the original knowledge-base gate. Add `--help` for its arguments.
