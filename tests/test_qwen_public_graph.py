"""Historical graph semantics on tiny public-like fixtures; no network/data runs."""
import gzip
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.language.build_graph import FILES, build
from baselines.language.graph_core import (_human_gene_to_ensembl, _hgnc_maps,
                                           normalize_ensembl, sha256_file)


@pytest.fixture
def public_inputs(tmp_path):
    source = tmp_path / 'inputs'
    source.mkdir()
    pd.DataFrame([
        ['ENSG000001.3', 'ENSG000002', 'interacts (with confidence: 0.8)'],
        ['ENSG000002', 'ENSG000001', 'interacts (with confidence: 0.9)'],
        ['ENSG000001', 'ENSG000001', 'self'],
        ['not_a_gene', 'ENSG000002', 'invalid'],
    ], columns=['Gene stable ID 1', 'Gene stable ID 2', 'relation']).to_csv(source / FILES['map'], index=False)
    pd.DataFrame([
        ['P1', 'P2', 'A', 'B', '1', '1', '0', 'PublicDB', 'PMID:1'],
        ['UNMAPPED', 'P2', 'AMBIGUOUS', 'B', '1', '0', '0', 'PublicDB', ''],
        ['P1', 'P1', 'A', 'A', '1', '0', '0', 'PublicDB', ''],
    ], columns=['source', 'target', 'source_genesymbol', 'target_genesymbol', 'is_directed',
                'is_stimulation', 'is_inhibition', 'sources', 'references']).to_csv(source / FILES['omnipath'], sep='\t', index=False)
    with gzip.open(source / FILES['gene2ensembl'], 'wt') as stream:
        stream.write('#tax_id\tGeneID\tEnsembl_gene_identifier\n'
                     '9606\t1\tENSG000001.3\n9606\t2\tENSG000002\n'
                     '9606\t9\tENSG000009\n9606\t9\tENSG000010\n'
                     '10090\t3\tENSMUSG000003\n')
    pd.DataFrame([
        ['Approved', 'A', 'ENSG000001', '1', 'P1'],
        ['Approved', 'B', 'ENSG000002', '2', 'P2'],
        ['Approved', 'AMBIGUOUS', 'ENSG000009', '9', 'P9'],
        ['Approved', 'AMBIGUOUS', 'ENSG000010', '9', 'P10'],
        ['Entry Withdrawn', 'OLD', 'ENSG000011', '11', 'P11'],
    ], columns=['status', 'symbol', 'ensembl_gene_id', 'entrez_id', 'uniprot_ids']).to_csv(source / FILES['hgnc'], sep='\t', index=False)
    return source


def test_gene_mapping_is_human_one_to_one_and_version_normalized(public_inputs):
    mapping, ambiguous, stats = _human_gene_to_ensembl(public_inputs / FILES['gene2ensembl'])
    assert mapping == {'1': 'ENSG000001', '2': 'ENSG000002'}
    assert set(ambiguous['9']) == {'ENSG000009', 'ENSG000010'}
    maps, excluded, _ = _hgnc_maps(public_inputs / FILES['hgnc'], mapping)
    assert 'AMBIGUOUS' not in maps['symbol']
    assert 'OLD' not in maps['symbol']
    assert maps['uniprot']['P1'] == 'ENSG000001'
    assert normalize_ensembl('ENSG000001.3') == 'ENSG000001'
    assert normalize_ensembl('random-symbol') is None


def test_public_graph_preserves_historical_edge_definition(public_inputs, tmp_path):
    expected = {key: sha256_file(public_inputs / name) for key, name in FILES.items()}
    out = tmp_path / 'graph.parquet'
    report = build(public_inputs, out, expected=expected)
    result = pd.read_parquet(out)
    assert len(result) == 2
    assert set(result['source_category']) == {'MAP', 'OmniPath'}
    map_edge = result[result['source_category'] == 'MAP'].iloc[0]
    assert map_edge['u'] == 'ENSG000001' and map_edge['v'] == 'ENSG000002'
    assert map_edge['relation'] == 'interacts'
    assert map_edge['evidence_rows'] == 2
    assert not map_edge['directed']
    omni = result[result['source_category'] == 'OmniPath'].iloc[0]
    assert omni['directed'] and json.loads(omni['signs_json']) == ['positive']
    assert report['excluded_omnipath_rows'] == 1
    assert report['input_protocol'] == 'new_public_input_snapshots'
    assert not report['response_labels_read'] and not report['dataset_target_selection_used']
    assert not report['historical_input_hashes_match']
    assert report == json.loads(out.with_suffix('.parquet.json').read_text())
    with pytest.raises(FileExistsError):
        build(public_inputs, out, expected=expected)


def test_new_public_inputs_require_explicit_opt_in(public_inputs, tmp_path):
    output = tmp_path / 'graph.parquet'
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        build(public_inputs, output)
    assert not output.exists()
    report = build(public_inputs, output, allow_new_snapshots=True)
    assert report['input_protocol'] == 'new_public_input_snapshots'


def test_graph_manifest_is_not_a_summer_graph_substitution():
    info = json.loads((ROOT / 'baselines/language/graph_sources.json').read_text())
    assert {x['id'] for x in info['assets']} == {'map', 'omnipath', 'gene2ensembl', 'hgnc'}
    assert all(len(x['checksums']['sha256']) == 64 for x in info['assets'])
    assert info['historical_canonical_graph']['rows'] == 228749
