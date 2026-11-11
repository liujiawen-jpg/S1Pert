"""Build Qwen3-4B-LoRA's historical public MAP-KG + OmniPath graph.

No download, model load or response-label read occurs here. Four public inputs
are required: MAP CSV, OmniPath TSV, NCBI gene2ensembl.gz, and HGNC complete TSV.
The graph definition is not SUMMER's GO/Reactome/STRING graph. Default execution
requires the four historical SHA256 identities. Live public endpoints may now
serve different data; --allow-new-snapshots explicitly creates a new-input run.

Inspect/download separately, from the repository root (download is opt-in):
  python scripts/download_assets.py --manifest baselines/language/graph_sources.json --root WORK --group qwen_graph --dry-run
  python scripts/download_assets.py --manifest baselines/language/graph_sources.json --root WORK --group qwen_graph
  python baselines/language/build_graph.py --source-dir WORK/assets/qwen_graph_sources --out WORK/assets/public_edges.parquet

If live endpoints fail the historical checksum, obtain the matching public
snapshot or explicitly use reviewed new inputs; never relabel them historical.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from baselines.language.graph_core import (_build_edges, _hgnc_maps,
    _human_gene_to_ensembl, sha256_file)

HERE = Path(__file__).resolve().parent
FILES = {'map': 'GENE-GENE_filtered_by_existing_genes.csv',
         'omnipath': 'omnipath_core_human.tsv', 'gene2ensembl': 'gene2ensembl.gz',
         'hgnc': 'hgnc_complete_set.tsv'}


def expected_sources():
    manifest = json.loads((HERE / 'graph_sources.json').read_text())
    return {row['id']: row['checksums']['sha256'] for row in manifest['assets']}


def build(source_dir, output, *, allow_new_snapshots=False, expected=None):
    source_dir, output = Path(source_dir), Path(output)
    report_path = output.with_suffix(output.suffix + '.json')
    if any(path.exists() or path.is_symlink() for path in (output, report_path)):
        raise FileExistsError('Use a fresh graph output; existing artifacts are preserved')
    paths = {key: source_dir / name for key, name in FILES.items()}
    identities = {key: sha256_file(path) for key, path in paths.items()}
    historical = expected_sources()
    required = historical if expected is None else expected
    mismatches = sorted(key for key in FILES if identities[key] != required.get(key))
    if mismatches and not allow_new_snapshots:
        raise ValueError('Public input SHA256 mismatch: ' + ', '.join(mismatches)
                         + '; obtain the recorded snapshot or explicitly allow new snapshots')
    geneid_map, ambiguous_geneids, bridge_stats = _human_gene_to_ensembl(paths['gene2ensembl'])
    maps, hgnc_exclusions, hgnc_stats = _hgnc_maps(paths['hgnc'], geneid_map)
    edges, exclusions, edge_stats = _build_edges(paths, maps)
    if edges.empty:
        raise ValueError('No canonical public gene-gene edges were produced')
    historical_inputs = identities == historical
    report = dict(schema='s1pert.qwen-public-graph.v1', status='PASS',
        graph_definition='historical MAP-KG + OmniPath; one-to-one HGNC/NCBI bridges',
        historical_input_hashes_match=historical_inputs,
        input_protocol='historical_inputs' if historical_inputs else 'new_public_input_snapshots',
        response_labels_read=False, dataset_target_selection_used=False,
        source_sha256=identities, canonical_edges=len(edges),
        bridge_statistics=bridge_stats, hgnc_statistics=hgnc_stats, edge_statistics=edge_stats,
        ambiguous_ncbi_ids=len(ambiguous_geneids),
        ambiguous_hgnc_ids={key: len(value) for key, value in hgnc_exclusions.items()},
        excluded_omnipath_rows=len(exclusions),
        constructor_sha256=sha256_file(HERE / 'graph_core.py'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.qwen-public-graph-', dir=output.parent) as tmp:
        staged = Path(tmp) / 'edges.parquet'
        edges.to_parquet(staged, index=False)
        report['output_sha256'] = sha256_file(staged)
        staged_report = Path(tmp) / 'report.json'
        staged_report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.link(staged, output)
        os.link(staged_report, report_path)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-dir', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--allow-new-snapshots', action='store_true',
                   help='Explicitly permit changed public inputs; report will not claim historical equality')
    a = p.parse_args(argv)
    work = os.environ.get('S1PERT_ROOT')
    if (a.source_dir is None or a.out is None) and not work:
        p.error('Supply --source-dir and --out, or set S1PERT_ROOT')
    source = a.source_dir or Path(work) / 'assets/qwen_graph_sources'
    out = a.out or Path(work) / 'assets/public_edges.parquet'
    print(json.dumps(build(source, out, allow_new_snapshots=a.allow_new_snapshots), indent=2))


if __name__ == '__main__':
    main()
