# Source and third-party notices

## Portable baseline adapters

The `baselines/` directory contains local benchmark adapters. Official GEARS,
scGPT, STATE and PerturbQA/SUMMER implementations are acquired separately by
explicit commands; the archive does not relicense or bundle their source trees.
`baselines/fetch_external.py` records repository URLs and revisions. Preserve each
downloaded project's original license and attribution. Public GenePT/ESM2 vectors,
knowledge archives, Qwen models and scGPT checkpoints likewise remain external
assets governed by their original terms. See `baselines/ASSETS.md` for provenance.

Qwen3-4B-LoRA and SUMMER-Qwen3-8B are disclosed local task adaptations, not claims
of reproducing the unmodified source papers.

## ALIGNED official core

`baselines/aligned/official_source/aligned/` contains the required original
ALIGNED core by Yuanfang Xiang and Lun Ai, from
[yfxiang0112/Aligned](https://github.com/yfxiang0112/Aligned), commit
`264e26e3fb826799c7f5e7637fa776d3feca4553`. The upstream declares Creative Commons
Attribution 4.0 International (CC BY 4.0). Its notice is retained in
`baselines/aligned/official_source/LICENSE`; the
[upstream license file](https://github.com/yfxiang0112/Aligned/blob/264e26e3fb826799c7f5e7637fa776d3feca4553/LICENSE)
was verified during packaging. Official source bytes and Git-blob/SHA256
identities are checked by `core_provenance.json`.

Cite Yuanfang Xiang and Lun Ai, *Adaptive Data-Knowledge Alignment in Genetic
Perturbation Prediction*, ICLR 2026. The A4 adaptation uses the official MLP,
R adaptor and symbolic reasoner with local strict-label/public-signed-KG inputs.
The inspected local scientific engine is retained; release changes provide
workspace paths, strict-data identity gates and continuous hybrid score export.
Details and original adapter hashes are in `baselines/aligned/README.md` and
`source_availability.json`. These adaptations do not imply upstream endorsement,
and the upstream license does not assign a repository-wide license to the
surrounding local research code.

This document records what could be established from the supplied execution handoff and source files. It does not assign a license to this repository. See [LICENSE_STATUS.md](LICENSE_STATUS.md) and [provenance/source_inventory.json](provenance/source_inventory.json).

## A4 implementation and the `vendor` directory

The A4 preparation, encoding, training, metrics, preprocessing, and label-resolution files were supplied as local research implementation code. Their headers and the handoff do not identify a repository-wide copyright holder or license grant.

`src/jev_a4/core/vendor/` is a historical directory name. It is not evidence that every file there comes from an external open-source package. The supplied source manifest traces `native_engine.py` to an earlier local DECIDER loading experiment and traces `shared_core.py` and `encode_joint_last8_cache.py` to an earlier local P4-v2 experiment. The handoff also includes `prior_model.py` and `protocol_utils.py` as sealed A4 source, but does not provide an equivalent original-source license declaration for them. The portable version retains only the helpers needed by A4 where applicable; historical source hashes identify the inputs to the adaptation, not the new files' hashes.

The release reuses generic model-loading, representation, and preprocessing code. It does not redistribute old task weights or inject old predictions into A4. Local provenance is not proof of copyright ownership; authorship and any incorporated third-party snippets still require maintainer review before choosing a project license.

## External DECIDER backbone

The external backbone is Mapika's DECIDER-4b v1, based on Qwen/Qwen3.5-4B-Base. The retained handoff model card declares `license: apache-2.0`. The recorded snapshot revision is `c23ab4d2483e7c6a93484463e9afe1688b29bedd`.

The upstream identifiers are `Mapika/decider-4b`, `Qwen/Qwen3.5-4B-Base`, and the `Mapika/decider` source repository. They were read from the historical local model card. The current upstream repositories and full license/NOTICE texts were not fetched during this audit. The model-card label is recorded as a declaration, not as a complete redistribution audit.

Backbone weights, tokenizer payload, and upstream snapshot `decider` implementation are not redistributed in this code archive. Users obtaining them separately must retain and follow the terms accompanying that exact revision. A4's use of this backbone does not confer its license on the surrounding research implementation. The project name does not imply affiliation with TypeSafe or that official hosted Jev weights are included.

## Python libraries

The implementation relies on separately installed scientific libraries, including NumPy, SciPy, pandas, h5py, pyarrow, PyTorch, Transformers, tokenizers, safetensors, flash-linear-attention/fla-core, Triton, and pytest for tests. These package names are dependency acknowledgments, not bundled copies. Their distributions carry their own licenses and notices. The historical handoff contains version metadata, not a complete license collection, so this document does not claim exact license coverage for each recorded binary package.

`decider-ai==1.2.1` was installed in the historical environment, but A4 normally imported the model snapshot's `decider` code through `decider_source`. Installing or licensing the former does not establish the provenance of the latter.

## Expression datasets and public annotations

The scientific data sources described in the supplied methods are:

- Nadig et al., *Transcriptome-wide Analysis of Differential Expression in Perturbation Atlases*, Nature Genetics 57, 1228–1237 (2025), DOI `10.1038/s41588-025-02169-3`; GSE264667 for the HepG2 and Jurkat inputs.
- Replogle et al., *Mapping Information-rich Genotype-phenotype Landscapes with Genome-scale Perturb-seq*, Cell 185, 2559–2575.e28 (2022), DOI `10.1016/j.cell.2022.05.013`; the historical RPE1 and K562 essential inputs used the recorded scPerturb preprocessing variant.

Raw and processed expression payloads, public annotation payloads, and empirical response caches are not included. Dataset citations do not establish redistribution permission. The handoff public-function export identifies selected fields from a `genes` table, but does not establish the complete upstream annotation source/version/license chain; do not infer a permissive database license from the words “public knowledge.” Supply annotations separately under their applicable terms and record their provenance.
