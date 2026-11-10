# Public baseline assets

Run these commands from the repository root. They acquire public source code or
generic knowledge inputs; they do not install dependencies, launch training,
download task-trained baseline checkpoints, or run evaluation.

```bash
python baselines/fetch_external.py --list
python baselines/fetch_external.py --repo gears --repo perturbqa --repo state --dry-run

# Explicit acquisition; omit sources/assets for methods you are not running.
python baselines/fetch_external.py --repo gears --repo perturbqa --repo state
python baselines/fetch_external.py --asset kg --asset genept --asset gene2go
```

Listing and dry runs are offline and create no directories. Git and Python's
standard library are sufficient for acquisition. File downloads reuse the main
downloader's resume, size/checksum, receipt, and no-overwrite checks. Repositories
are fetched at exact commits into new temporary checkouts and published only
after verifying their commit. Existing checkouts must have the same origin,
commit, and clean working tree; changed sources are never reset or replaced.
No Git hooks, package installers, or submodules are run by the helper.

## Sources and pins

| Input | Public source | Destination / pin |
|---|---|---|
| GEARS code | [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS) | `external/GEARS`, `f374e43e197b295016d80395d7a54ddb81cc6769` |
| PerturbQA / SUMMER retrieval | [Genentech/PerturbQA](https://github.com/Genentech/PerturbQA) | `external/PerturbQA`, `df2fe3a7802f2a46a43876b4f8674734aad3a30a` |
| STATE code | [ArcInstitute/state](https://github.com/ArcInstitute/state) | `external/state`, `9bbfe78a434a55205e4de834e1ea99f85f7a3add` |
| Public knowledge graph | [PerturbQA Zenodo record 14915313](https://zenodo.org/records/14915313) | `assets/baselines/kg.zip`; 46,080,292 bytes; MD5 `90977691ebac65e02765e34c31b66bd1` |
| GenePT vectors | [GenePT Zenodo record 10833191](https://zenodo.org/records/10833191) | `assets/baselines/GenePT_emebdding_v2.zip`; 574,395,233 bytes; MD5 `3f6ce4317e3a0091978ae5cb8fbf05a3` |
| GEARS gene-to-GO map | [Harvard Dataverse file 6153417](https://dataverse.harvard.edu/file.xhtml?fileId=6153417) | `assets/baselines/gene2go_all.pkl`; 9,462,558 bytes; MD5 `77c9af0c61c30ea4d7a85680f4d122dc` |

These pins and archive metadata were checked against the publishers on
2026-11-10. The historical GenePT/knowledge archives were inspected locally to
verify formats; no new archive was downloaded. GenePT contains pickle objects;
the classical adapter accepts the checksum-pinned author archive, not arbitrary
unverified pickle inputs. The graph converter reads JSON only and never extracts
archive members to filesystem paths.

The gene-to-GO URL is the one used by the pinned GEARS source. Dataverse's public
metadata supplies the size and MD5 above, but a redirected HTTP HEAD request
returned 403 in the release environment. `--asset gene2go` performs the explicit
download and rejects unsuccessful or mismatched responses. If transfer is
blocked, use the linked official file page and save its `gene2go.pkl` as
`assets/baselines/gene2go_all.pkl`, checking the listed size/MD5 before use.
No mirror or newly generated GO mapping is substituted automatically.

The four raw matrices come from the main data downloader and remain under
`data/raw/Nadig2024_GSE264667_raw/` and `data/raw/scPerturb_prepro/`.
Classical `raw_root` must point to `data/raw`; it does not require a second copy.
Prepared labels and target splits remain those of `workspace/a4/data`.

## SUMMER public edge database

```bash
python baselines/prepare_public_edges.py \
  --archive assets/baselines/kg.zip --out assets/public_edges.sqlite
```

The converter verifies the pinned archive MD5 before reading `go.json`,
`reactome.json`, and `string.json`. It creates
`public_edges(subject TEXT, object TEXT, source TEXT)` with source values
`GO`, `Reactome`, and `STRING`, plus a provenance table with archive/member hashes
and edge counts. GO and Reactome rows are gene-to-term edges; STRING rows are
gene-to-gene edges. Duplicate triples are removed. STRING's undirected treatment
occurs in the baseline's retrieval adapter. The other archive resources and all
empirical response tables are excluded.

This uses the same public upstream knowledge collection as the baseline input
recipe. Equality to the historical combined database's hidden edge snapshot has
not been established. The converter records
`historical_edge_database_equality_verified: false`; scores obtained with this
rebuild must be identified as a public-asset rebuild.

SUMMER also needs `assets/public_functions.sqlite`, built by the main UniProt
pipeline, and a generic Qwen3-8B snapshot in `assets/qwen3_8b`. Acquire Qwen3-8B
from its [official model repository](https://huggingface.co/Qwen/Qwen3-8B), pin a
revision, and retain file hashes. This helper does not choose a changing model
revision or download those weights. The same live-UniProt snapshot limitation
described in `docs/DATA_PIPELINE.md` applies here.

## STATE protein representations

STATE consumes generic ESM2 protein features under stable target IDs. The source
used by the reference preparation is available publicly in
[Arc Institute PIE_sources](https://huggingface.co/datasets/arcinstitute/PIE_sources)
at commit `fb624a5158a7f37dfa13a015cdacadb985549d26`. Its `esm2` source contains
19,203 rows × 1,280 dimensions, generated from reviewed human UniProt proteins
with ESM-2 650M. These are generic protein representations, not PIE or STATE
task-trained weights. The helper fetches only three files from that source:
`meta.json`, `embeddings.npy`, and `aliases.yaml`, approximately 98.5 MB total.

```bash
python baselines/fetch_external.py --asset esm2 --dry-run
python baselines/fetch_external.py --asset esm2
python baselines/prepare_esm2.py --source assets/esm2_public --out assets/esm2 \
  --metadata workspace/a4/data/hepg2/dataset.json \
  --metadata workspace/a4/data/jurkat/dataset.json \
  --metadata workspace/a4/data/rpe1/dataset.json \
  --metadata workspace/a4/data/k562_essential/dataset.json
```

The mapper requires NumPy. It reads target identities and symbols only, uses
exact public keys or the pinned source's explicit alias pairs, and refuses
ambiguous matches. Optional `--alias-tsv public_mapping.tsv` accepts a reviewed
public mapping with `gene_id` and `symbol` columns; its SHA256 is recorded.
No response labels or expression values are used. Output `meta.json` contains
the mapped stable `keys`, `dim`, missing IDs, complete key mapping, and source
hashes; `embeddings.npy` follows that row order. This directly matches the
baseline's `esm_source` contract.

The historical preparation also consulted an auxiliary public-identifier alias
table not included in this release. Without the same table, a rebuild can have
different coverage. Missing IDs remain absent from the feature table: the STATE
adapter supplies zero features and reports them, while retaining the prescribed
evaluation records. The mapper neither invents sequences nor trains a new
protein encoder. Verify coverage before comparing reconstructed results.

## scGPT source and pretrained asset boundary

```bash
python baselines/fetch_external.py --repo scgpt --dry-run
python baselines/fetch_external.py --repo scgpt
```

The historical source copy had no Git metadata. This helper offers a verified
public rebuild pin, `cebd6fae655b9c585a4807daa3ac31bb764f06b4`, from the
[official scGPT repository](https://github.com/bowang-lab/scGPT). Equality with
the historical source copy is unverified; do not describe this pin as recovered
historical provenance.

The pretrained vocabulary/configuration/checkpoint are a separate manual step.
The official repository links its
[whole-human model folder](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y).
Place `vocab.json`, `args.json`, and `best_model.pt` together in `assets/scgpt`,
record hashes, and check compatibility with the adapter. The release has not
verified this folder's checkpoint identity against the historical local file.
No Drive credentials, task-trained scGPT checkpoint, or automatic fallback
weights are embedded in the release.

## Historical annotation selection

For fidelity to the ported implementation, public function lookup retains its
existing ordered-accession selection behavior: candidates are compared against
an already truncated 320-character result. For genes with multiple long
accession descriptions, this can select a later description instead of the
globally longest one. The asset importers preserve all accession rows; they do
not silently change this selection rule. Record annotation snapshots and this
behavior when comparing with a corrected lookup implementation.
