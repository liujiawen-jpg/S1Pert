# Qwen3-4B-LoRA adapted comparator

`qwen3_lora.py` is a portable version of the local comparator used for the completed four-cell, five-seed matrix. The historical registry called it `PertReasonLM-LoRA-A4-adapted`. It is a Qwen3-4B LoRA classifier adapted to the S1Pert data protocol; it does not implement the official PertReasonLM SFT/GRPO pipeline.

## Inputs and prediction

Each prompt contains the cell type, perturbation target symbol, and query symbol. The model pools the Qwen hidden states and combines them with eight untreated-control numeric features and three TRAIN-derived class proportions. Public graph edges select neighboring query genes for the proportions. Five-fold target cross-fitting excludes the entire TRAIN target fold; VAL and TEST use TRAIN observations only.

Fresh linear heads predict the answer and auxiliary objectives. Final class probabilities are the answer head's softmax in `[up, same, down]` order. The class proportions are input features and auxiliary supervision; they are not added to the output. No old task predictor is loaded.

The retained configuration is Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, LoRA rank 32, alpha 64, dropout 0.05, and attention plus MLP projection targets. Three stages run for `2:2:4` epochs. Each epoch samples at most 512 TRAIN rows without replacement; training batch size is eight. The public command defaults to evaluation batch 512, matching the completed matrix rather than the historical script's fallback of 64. Selection uses VAL Macro-F1, then NLL, then the earlier epoch.

## Run

Install S1Pert and a compatible PyTorch, Transformers build with `Qwen3Model`, and PEFT. The package's pandas/pyarrow dependencies read public graph edges. Full execution requires the complete local Qwen3-4B snapshot and sufficient CUDA memory; no download is initiated by the script.

Copy `qwen3_lora.example.json` to a local configuration and supply:

- `data_root`: S1Pert's prepared `data` directory, including strict records, target partitions/folds, identities, and control numeric arrays.
- `backbone_snapshot`: the complete matching general pretrained Qwen3-4B snapshot, without any historical task adapter.
- `knowledge_graph`: a public graph Parquet file with string columns `u` and `v`, matching the prepared gene identifiers.
- `output_root`: a new writable output directory.

Resource paths resolve relative to the configuration file. From the repository root:

```sh
python baselines/language/qwen3_lora.py --help
python baselines/language/qwen3_lora.py --config baselines/language/qwen3_lora.local.json --cell hepg2 --seed 17
python baselines/language/qwen3_lora.py --config baselines/language/qwen3_lora.local.json --all
```

The last two commands start training. Existing nonempty per-run output directories are rejected. This release preparation did not execute them.

Each run additionally exports `test_predictions.npz` with the exact TEST target/query/label order and the original probabilities. Register this file with the shared `python -m jev_a4.evaluation` command to obtain the common comparison table; the extra export does not change predictions.

## Fidelity and limits

The model, TRAIN prior calculation, graph use, prompts, losses, optimizer, sampling, selection, and answer inference retain the original adapter's calculations. Changes replace machine-specific paths with configuration, import the common metric implementation, guard an optional Torch dispatch setting, reject output overwrites, and add a pair-aligned prediction export. The original source digest is in `source_provenance.json`. This LoRA comparator is distinct from the frozen-Qwen same-architecture comparison.

Historical restore reconstructs the selected adapter and answer/projection/path heads and checks finite output on up to 32 VAL rows. It does not compare restored probabilities with a saved reference or run the S1Pert convergence gate. The auxiliary relation head was not saved; it is unused by answer inference. These limitations are retained and disclosed. No historical adapters, checkpoints, public graph, predictions, or model weights are included. CPU tests cover the input construction, direct head interface, and path configuration; full CUDA LoRA training and biological results were not reproduced during packaging.
