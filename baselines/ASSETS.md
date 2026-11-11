# Public baseline assets

Run these commands from the repository root after initializing the paper
workspace. All destination paths below are relative to that workspace, not the
source checkout. Acquisition fetches public code and generic knowledge inputs;
training is a separate command.

```bash
export S1PERT_ROOT=/path/to/workspace
python baselines/fetch_external.py --root "$S1PERT_ROOT" --list
python baselines/fetch_external.py --root "$S1PERT_ROOT" \
  --repo gears --repo perturbqa --repo state --repo scgpt --dry-run

# Explicit acquisition; omit sources/assets for methods you are not running.
python baselines/fetch_external.py --root "$S1PERT_ROOT" \
  --repo gears --repo perturbqa --repo state --repo scgpt
python baselines/fetch_external.py --root "$S1PERT_ROOT" \
  --asset kg --asset genept --asset gene2go
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

The initialized workspace stores the four raw matrices under
`$S1PERT_ROOT/raw/Nadig2024_GSE264667_raw/` and
`$S1PERT_ROOT/raw/scPerturb_prepro/`. If initialization used `--data-root`, the
configured external raw directory remains the input source. Classical
`raw_root` points to that same raw directory; no second copy is needed.
Prepared labels and target splits reside in `$S1PERT_ROOT/data`.

## SUMMER public edge database

```bash
python baselines/prepare_public_edges.py \
  --archive "$S1PERT_ROOT/assets/baselines/kg.zip" \
  --out "$S1PERT_ROOT/assets/public_edges.sqlite"
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

SUMMER also needs `$S1PERT_ROOT/assets/public_functions.sqlite`, built by the main
UniProt pipeline, and a generic Qwen3-8B snapshot in
`$S1PERT_ROOT/assets/qwen3_8b`. Acquire Qwen3-8B
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
python baselines/fetch_external.py --root "$S1PERT_ROOT" --asset esm2 --dry-run
python baselines/fetch_external.py --root "$S1PERT_ROOT" --asset esm2
python baselines/prepare_esm2.py \
  --source "$S1PERT_ROOT/assets/esm2_public" --out "$S1PERT_ROOT/assets/esm2" \
  --metadata "$S1PERT_ROOT/data/hepg2/dataset.json" \
  --metadata "$S1PERT_ROOT/data/jurkat/dataset.json" \
  --metadata "$S1PERT_ROOT/data/rpe1/dataset.json" \
  --metadata "$S1PERT_ROOT/data/k562_essential/dataset.json"
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
python baselines/fetch_external.py --root "$S1PERT_ROOT" --repo scgpt --dry-run
python baselines/fetch_external.py --root "$S1PERT_ROOT" --repo scgpt
```

The historical source copy had no Git metadata. This helper offers a verified
public rebuild pin, `cebd6fae655b9c585a4807daa3ac31bb764f06b4`, from the
[official scGPT repository](https://github.com/bowang-lab/scGPT). Equality with
the historical source copy is unverified; do not describe this pin as recovered
historical provenance.

The pretrained vocabulary/configuration/checkpoint are a separate manual step.
The official repository links its
[whole-human model folder](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y).
Place `vocab.json`, `args.json`, and `best_model.pt` together in
`$S1PERT_ROOT/assets/scgpt`,
record hashes, and check compatibility with the adapter. The release has not
verified this folder's checkpoint identity against the historical local file.
No Drive credentials, task-trained scGPT checkpoint, or automatic fallback
weights are embedded in the release.

## Qwen3-4B classifier graph

Place the generic Qwen3-4B snapshot at `$S1PERT_ROOT/assets/Qwen3-4B`, using the
revision documented in [the adapter instructions](language/README.md). Its
public graph is a separate Parquet asset; it is not SUMMER's SQLite edge file.
The packaged constructor maps the recorded MAP-KG and OmniPath inputs through
the recorded HGNC/NCBI identity bridges:

```bash
python scripts/download_assets.py --manifest baselines/language/graph_sources.json \
  --root "$S1PERT_ROOT" --group qwen_graph --dry-run
python scripts/download_assets.py --manifest baselines/language/graph_sources.json \
  --root "$S1PERT_ROOT" --group qwen_graph
python -m baselines.language.build_graph \
  --source-dir "$S1PERT_ROOT/assets/qwen_graph_sources" \
  --out "$S1PERT_ROOT/assets/public_edges.parquet"
```

Historical input hashes are checked before construction. Live public endpoints
can change; use the recorded snapshots for historical-input reproduction. The
builder's explicit `--allow-new-snapshots` mode records a new-public-input
protocol when intentionally rebuilding with updated inputs.

## ALIGNED signed knowledge and GO inputs

The bundled ALIGNED engine consumes two public-only databases built from the
packaged source manifest. Fetch into the same workspace and build them before
the ALIGNED preparation/training commands:

```bash
python scripts/download_assets.py --manifest data_prep/public_knowledge/sources.json \
  --root "$S1PERT_ROOT" --group public_knowledge --dry-run
python scripts/download_assets.py --manifest data_prep/public_knowledge/sources.json \
  --root "$S1PERT_ROOT" --group public_knowledge
python -m data_prep.build_public_knowledge --work-dir "$S1PERT_ROOT"
```

The outputs are `$S1PERT_ROOT/assets/public_knowledge.sqlite` and
`$S1PERT_ROOT/assets/public_go.sqlite`, with a provenance report identifying
every input snapshot and consumed graph row. These files are distinct from
the functional-description database and SUMMER's undirected retrieval graph.
See [ALIGNED instructions](aligned/README.md) and the builder's `--help` for
the source allowlist and the explicit updated-snapshot workflow.

## Historical annotation selection

For fidelity to the ported implementation, public function lookup retains its
existing ordered-accession selection behavior: candidates are compared against
an already truncated 320-character result. For genes with multiple long
accession descriptions, this can select a later description instead of the
globally longest one. The asset importers preserve all accession rows; they do
not silently change this selection rule. Record annotation snapshots and this
behavior when comparing with a corrected lookup implementation.
