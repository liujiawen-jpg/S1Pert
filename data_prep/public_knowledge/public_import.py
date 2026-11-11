"""Stream public biological facts into an independent, CPU-only SQLite KB.

No experimental outcomes are read. A molecular sign is never interpreted as
the direction of an mRNA response. Original public records remain in payloads.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS public_edges (id TEXT PRIMARY KEY, subject TEXT, object TEXT, relation TEXT, sign TEXT, directed INTEGER, layer TEXT, source TEXT, payload TEXT)",
    "CREATE TABLE IF NOT EXISTS terms (term TEXT PRIMARY KEY, name TEXT, definition TEXT, namespace TEXT, obsolete INTEGER, aliases TEXT)",
    "CREATE TABLE IF NOT EXISTS gene_annotations (gene TEXT, term TEXT, qualifiers TEXT, evidence TEXT, references_json TEXT, source TEXT, payload TEXT)",
    "CREATE TABLE IF NOT EXISTS genes (accession TEXT PRIMARY KEY, symbol TEXT, synonyms TEXT, protein_name TEXT, function_text TEXT, location_text TEXT, pathway_text TEXT)",
    "CREATE TABLE IF NOT EXISTS pathways (pathway TEXT PRIMARY KEY, name TEXT, species TEXT)",
    "CREATE TABLE IF NOT EXISTS pathway_membership (accession TEXT, pathway TEXT)",
    "CREATE TABLE IF NOT EXISTS pathway_hierarchy (parent TEXT, child TEXT)",
    "CREATE UNIQUE INDEX IF NOT EXISTS public_import_annotations_unique ON gene_annotations (gene, term, qualifiers, evidence, references_json, source, payload)",
    "CREATE UNIQUE INDEX IF NOT EXISTS public_import_membership_unique ON pathway_membership (accession, pathway)",
    "CREATE UNIQUE INDEX IF NOT EXISTS public_import_hierarchy_unique ON pathway_hierarchy (parent, child)",
)

_INSERT = {
    "public_edges": "INSERT OR IGNORE INTO public_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    "terms": "INSERT OR IGNORE INTO terms VALUES (?, ?, ?, ?, ?, ?)",
    "gene_annotations": "INSERT OR IGNORE INTO gene_annotations VALUES (?, ?, ?, ?, ?, ?, ?)",
    "genes": "INSERT OR IGNORE INTO genes VALUES (?, ?, ?, ?, ?, ?, ?)",
    "pathways": "INSERT OR IGNORE INTO pathways VALUES (?, ?, ?)",
    "pathway_membership": "INSERT OR IGNORE INTO pathway_membership VALUES (?, ?)",
    "pathway_hierarchy": "INSERT OR IGNORE INTO pathway_hierarchy VALUES (?, ?)",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@contextmanager
def _text(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield handle


class _Batches:
    """Only a bounded batch of any source is held in Python memory."""

    def __init__(self, connection, size=1000):
        self.connection = connection
        self.size = size
        self.pending = {}
        self.attempted = {}
        self.inserted = {}

    def add(self, table, row):
        self.pending.setdefault(table, []).append(row)
        self.attempted[table] = self.attempted.get(table, 0) + 1
        if len(self.pending[table]) >= self.size:
            self.flush(table)

    def flush(self, table=None):
        for name in ([table] if table else list(self.pending)):
            rows = self.pending.get(name, [])
            if rows:
                before = self.connection.total_changes
                self.connection.executemany(_INSERT[name], rows)
                self.inserted[name] = self.inserted.get(name, 0) + self.connection.total_changes - before
                rows.clear()

    def edge(self, subject, obj, relation, sign, directed, layer, source, payload):
        canonical = _json(payload)
        identifier = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.add("public_edges", (identifier, subject, obj, relation, sign, int(directed), layer, source, canonical))

    def audit(self):
        self.flush()
        return {
            "attempted": dict(self.attempted),
            "inserted": dict(self.inserted),
            "duplicates_ignored": {k: n - self.inserted.get(k, 0) for k, n in self.attempted.items()},
        }


def _bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "t"):
        return True
    if text in ("false", "0", "no", "f"):
        return False
    return None


def _parts(value, separator=";"):
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x)]
    return [x.strip() for x in str(value or "").split(separator) if x.strip()]


def _quoted(value):
    match = re.match(r'"(?:\\.|[^"\\])*"', value)
    if not match:
        return value
    try:
        return json.loads(match.group())
    except ValueError:
        return match.group()[1:-1].replace(r'\"', '"').replace(r"\\", "\\")


def _ontology(path, batches):
    header = {}
    aliases = {}
    obsolete = {}
    current = None
    stats = {"terms_read": 0, "ontology_edges": 0}

    def flush():
        if not current or not current.get("id", "").startswith("GO:"):
            return
        term = current["id"]
        is_obsolete = current.get("is_obsolete") == "true"
        obsolete[term] = is_obsolete
        for alias in current["alt_ids"]:
            aliases[alias] = term
        alias_payload = {k: current[k] for k in ("alt_ids", "synonyms", "replaced_by", "consider")}
        batches.add("terms", (term, current.get("name", ""), _quoted(current.get("def", "")), current.get("namespace", ""), int(is_obsolete), _json(alias_payload)))
        stats["terms_read"] += 1
        for relation, parent, raw in current["relationships"]:
            payload = {
                "source": "GO_OBO", "subject": term, "object": parent,
                "relation": relation, "raw_relationship": raw,
                "embedded_go_version": header.get("data-version"),
                "subject_obsolete": is_obsolete,
                "semantic_direction": "child_to_parent" if relation in ("go_is_a", "go_part_of") else "as_recorded_in_ontology",
                "implies_mrna_direction": False, "quantity_unknown": True,
            }
            batches.edge(term, parent, relation, "unknown", True, "ontology", "GO", payload)
            stats["ontology_edges"] += 1

    with _text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("["):
                flush()
                current = {"alt_ids": [], "synonyms": [], "replaced_by": [], "consider": [], "relationships": []} if line == "[Term]" else None
                continue
            if not line or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            if current is None:
                if key in ("data-version", "format-version", "ontology"):
                    header[key] = value
            elif key == "alt_id":
                current["alt_ids"].append(value)
            elif key == "synonym":
                # Preserve scope and cross-references, as well as quoted text.
                current["synonyms"].append({"text": _quoted(value), "raw": value})
            elif key in ("replaced_by", "consider"):
                current[key].append(value.split()[0])
            elif key == "is_a" and value.startswith("GO:"):
                current["relationships"].append(("go_is_a", value.split()[0], line))
            elif key == "relationship":
                parts = value.split()
                if len(parts) >= 2 and parts[1].startswith("GO:"):
                    current["relationships"].append(("go_" + parts[0], parts[1], line))
            elif key in ("id", "name", "def", "namespace", "is_obsolete"):
                current[key] = value
    flush()
    stats["ontology_header"] = header
    stats["quantity_unknown_edges"] = stats["ontology_edges"]
    return stats, aliases, obsolete


def _gaf(path, batches, aliases, obsolete):
    stats = {"rows_read": 0, "human_annotations": 0, "skipped_nonhuman": 0, "malformed_rows": 0, "negated_annotations": 0, "obsolete_annotations_retained": 0, "headers": []}
    with _text(path) as handle:
        for raw in handle:
            if raw.startswith("!"):
                stats["headers"].append(raw.rstrip())
                continue
            if not raw.strip():
                continue
            stats["rows_read"] += 1
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) < 15 or not fields[2] or not fields[4]:
                stats["malformed_rows"] += 1
                continue
            if fields[12].split("|")[0] != "taxon:9606":
                stats["skipped_nonhuman"] += 1
                continue
            qualifiers = _parts(fields[3], "|")
            negated = "NOT" in qualifiers
            term = aliases.get(fields[4], fields[4])
            is_obsolete = obsolete.get(term, False)
            payload = {
                "raw_columns": fields, "negated": negated,
                "original_term": fields[4], "term_obsolete": is_obsolete,
                "term_known": term in obsolete if obsolete else None,
                "implies_mrna_direction": False, "quantity_unknown": True,
            }
            batches.add("gene_annotations", (fields[2], term, _json(qualifiers), _json(_parts(fields[6], "|")), _json(_parts(fields[5], "|")), "GO_GAF", _json(payload)))
            stats["human_annotations"] += 1
            stats["negated_annotations"] += int(negated)
            stats["obsolete_annotations_retained"] += int(is_obsolete)
    return stats


def _legacy(path, batches):
    stats = {"rows_read": 0, "complex_records_retained": 0, "quantity_unknown_edges": 0, "layers": {}}
    with _text(path) as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if not isinstance(record, dict) or not record.get("subject") or not record.get("object"):
                raise ValueError(f"Invalid public graph record at {path}:{number}")
            stats["rows_read"] += 1
            subject, obj = str(record["subject"]), str(record["object"])
            relation = str(record.get("relation", "unknown"))
            sources = _parts(record.get("sources", []))
            ontology_sources = [s for s in sources if s in ("go_is_a", "go_part_of")]
            source = str(record.get("source", "unknown"))
            if ontology_sources or (subject.startswith("GO:") and obj.startswith("GO:")):
                layer = "ontology"
                if len(ontology_sources) == 1:
                    relation = ontology_sources[0]
            elif relation == "annotation" or any(s.startswith("go_annotation_") for s in sources):
                layer = "annotation"
            elif source.upper() == "STRING" or relation.startswith("string_") or relation == "association":
                layer = "association"
            elif relation in ("transcriptional", "transcriptional_regulation"):
                layer = "transcriptional"
            elif relation in ("post_translational", "protein_regulation", "post_translational_regulation"):
                layer = "protein_regulation"
            else:
                layer = "public_other"
            recorded_sign = str(record.get("sign", "unknown")).lower()
            sign = recorded_sign if recorded_sign in ("positive", "negative", "unknown", "conflict") else "unknown"
            # A signed annotation or association must never enter sign propagation.
            if layer in ("ontology", "annotation", "association"):
                sign = "unknown"
            complex_record = subject.startswith("COMPLEX:") or obj.startswith("COMPLEX:")
            metadata = {
                "layer": layer, "quantity_unknown": layer != "transcriptional",
                "target_quantity": "transcription" if layer == "transcriptional" else "unspecified",
                "source_quantity_unknown": True,
                "implies_mrna_direction": False, "complex_retained": complex_record,
                "recorded_sign": recorded_sign,
                "semantic_direction": "child_to_parent" if relation in ("go_is_a", "go_part_of") else "as_recorded",
                "independent_experiment": False,
                "evidence_group": hashlib.sha256(_json([layer, subject, obj]).encode()).hexdigest(),
            }
            batches.edge(subject, obj, relation, sign, _bool(record.get("directed")) is True, layer, source, {"record": record, "kb_metadata": metadata})
            stats["complex_records_retained"] += int(complex_record)
            stats["quantity_unknown_edges"] += int(layer != "transcriptional")
            stats["layers"][layer] = stats["layers"].get(layer, 0) + 1
    return stats


def _omnipath(path, batches, source_name="OmniPath-core", default_type="post_translational"):
    stats = {"rows_read": 0, "complex_records_retained": 0, "quantity_unknown_edges": 0, "sign_conflicts": 0, "unmapped_symbol_endpoints": 0, "consensus_direction_unknown": 0, "skipped_nonhuman": 0, "layers": {}}
    with _text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source", "target", "is_directed", "is_stimulation", "is_inhibition"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"OmniPath core header missing {sorted(required - set(reader.fieldnames or []))}")
        for record in reader:
            stats["rows_read"] += 1
            if any(record.get(key, "9606").strip() != "9606" for key in ("ncbi_tax_id_source", "ncbi_tax_id_target")):
                stats["skipped_nonhuman"] += 1
                continue
            if not record["source"] or not record["target"]:
                raise ValueError(f"OmniPath source/target missing at row {stats['rows_read']}")
            flags = {k: _bool(record.get(k)) for k in ("is_directed", "is_stimulation", "is_inhibition", "consensus_direction", "consensus_stimulation", "consensus_inhibition")}
            positive = flags["is_stimulation"] is True or flags["consensus_stimulation"] is True
            negative = flags["is_inhibition"] is True or flags["consensus_inhibition"] is True
            sign = "conflict" if positive and negative else "positive" if positive else "negative" if negative else "unknown"
            nodes = []
            complex_flags = []
            for endpoint in ("source", "target"):
                accession = record[endpoint].strip()
                complex_flag = accession.startswith("COMPLEX:")
                symbol = (record.get(endpoint + "_genesymbol") or "").strip()
                nodes.append(accession if complex_flag or not symbol else symbol)
                complex_flags.append(complex_flag)
                stats["unmapped_symbol_endpoints"] += int(not complex_flag and not symbol)
            references = _parts(record.get("references"))
            pmids = sorted({match.group(1) for reference in references for match in [re.search(r"(?:^|:)(\d+)$", reference)] if match})
            kind = record.get("type") or default_type
            layer = "transcriptional" if kind == "transcriptional" else "protein_regulation" if kind in ("post_translational", "protein_regulation") else "public_other"
            quantity_unknown = layer != "transcriptional"
            payload = {
                "record": record, "source": source_name, "flags": flags,
                "aggregated_sources": _parts(record.get("sources")),
                "references": references, "pmids": pmids,
                "source_accession": record["source"], "target_accession": record["target"],
                "complex_endpoints": {"source": complex_flags[0], "target": complex_flags[1]},
                "entity_quantity": "target_transcription" if layer == "transcriptional" else "protein_activity_or_abundance_unspecified",
                "quantity_unknown": quantity_unknown, "source_quantity_unknown": True,
                "target_quantity_unknown": quantity_unknown, "implies_mrna_direction": False,
                "independent_experiment": False,
                "evidence_group": hashlib.sha256(_json([layer, *nodes]).encode()).hexdigest(),
            }
            batches.edge(nodes[0], nodes[1], layer, sign, flags["is_directed"] is True, layer, source_name, payload)
            stats["complex_records_retained"] += int(any(complex_flags))
            stats["quantity_unknown_edges"] += int(quantity_unknown)
            stats["sign_conflicts"] += int(sign == "conflict")
            stats["consensus_direction_unknown"] += int(flags["consensus_direction"] is None)
            stats["layers"][layer] = stats["layers"].get(layer, 0) + 1
    return stats


_UNIPROT_HEADERS = {
    "accession": ("Entry", "accession"),
    "primary": ("Gene Names (primary)", "gene_primary", "gene_primary_names"),
    "synonyms": ("Gene Names (synonym)", "gene_synonym", "gene_synonyms"),
    "protein": ("Protein names", "protein_name"),
    "function": ("Function [CC]", "cc_function"),
    "location": ("Subcellular location [CC]", "cc_subcellular_location"),
    "pathway": ("Pathway", "Pathway [CC]", "cc_pathway"),
}


def _uniprot(path, batches):
    stats = {"rows_read": 0, "unambiguous_primary_gene_rows": 0, "ambiguous_primary_gene_rows": 0, "missing_primary_gene_rows": 0, "ambiguous_primary_gene_examples": []}
    with _text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        keys = {name: next((key for key in candidates if key in fields), None) for name, candidates in _UNIPROT_HEADERS.items()}
        if not keys["accession"]:
            raise ValueError("UniProt TSV has no Entry/accession column")
        for row in reader:
            stats["rows_read"] += 1
            value = lambda name: (row.get(keys[name], "") or "").strip() if keys[name] else ""
            accession = value("accession")
            if not accession:
                raise ValueError(f"UniProt accession missing at row {stats['rows_read']}")
            primary = list(dict.fromkeys(re.split(r"[\s;,]+", value("primary"))))
            primary = [s for s in primary if s]
            synonyms = [s for s in re.split(r"[\s;,]+", value("synonyms")) if s]
            symbol = primary[0] if len(primary) == 1 else ""
            if len(primary) > 1:
                stats["ambiguous_primary_gene_rows"] += 1
                if len(stats["ambiguous_primary_gene_examples"]) < 10:
                    stats["ambiguous_primary_gene_examples"].append({"accession": accession, "primary_names": primary})
            elif primary:
                stats["unambiguous_primary_gene_rows"] += 1
            else:
                stats["missing_primary_gene_rows"] += 1
            mapping = {"primary_names": primary, "gene_synonyms": sorted(set(synonyms)), "mapping_status": "unambiguous" if symbol else "ambiguous" if primary else "unmapped"}
            batches.add("genes", (accession, symbol, _json(mapping), value("protein"), value("function"), value("location"), value("pathway")))
    return stats


def _reactome_pathways(path, batches):
    stats = {"rows_read": 0, "human_pathways": 0, "skipped_nonhuman": 0, "malformed_rows": 0}
    with _text(path) as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if not fields or not any(fields):
                continue
            stats["rows_read"] += 1
            if len(fields) < 3:
                stats["malformed_rows"] += 1
            elif fields[2].strip() != "Homo sapiens":
                stats["skipped_nonhuman"] += 1
            else:
                batches.add("pathways", (fields[0], fields[1], "Homo sapiens"))
                stats["human_pathways"] += 1
    return stats


def _human_pathway(connection, pathway):
    return connection.execute("SELECT 1 FROM pathways WHERE pathway=? AND species='Homo sapiens'", (pathway,)).fetchone() is not None


def _reactome_hierarchy(path, batches):
    stats = {"rows_read": 0, "human_relationships": 0, "skipped_nonhuman_or_unknown": 0, "malformed_rows": 0}
    with _text(path) as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if not fields or not any(fields):
                continue
            stats["rows_read"] += 1
            if len(fields) < 2:
                stats["malformed_rows"] += 1
                continue
            parent, child = fields[:2]
            if not _human_pathway(batches.connection, parent) or not _human_pathway(batches.connection, child):
                stats["skipped_nonhuman_or_unknown"] += 1
                continue
            batches.add("pathway_hierarchy", (parent, child))
            payload = {"source": "Reactome", "raw_columns": fields, "relation": "pathway_parent_of", "sign": "unknown", "quantity_unknown": True, "implies_mrna_direction": False}
            batches.edge(parent, child, "pathway_parent_of", "unknown", True, "pathway_hierarchy", "Reactome", payload)
            stats["human_relationships"] += 1
    stats["quantity_unknown_edges"] = stats["human_relationships"]
    return stats


def _reactome_membership(path, batches):
    stats = {"rows_read": 0, "human_memberships": 0, "skipped_nonhuman_or_unknown": 0, "malformed_rows": 0}
    with _text(path) as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if not fields or not any(fields):
                continue
            stats["rows_read"] += 1
            if len(fields) < 2:
                stats["malformed_rows"] += 1
                continue
            accession, pathway = fields[:2]
            # Official UniProt2Reactome exports have species in column six.
            explicit_species = fields[5].strip() if len(fields) >= 6 else None
            if (explicit_species is not None and explicit_species != "Homo sapiens") or not _human_pathway(batches.connection, pathway):
                stats["skipped_nonhuman_or_unknown"] += 1
                continue
            batches.add("pathway_membership", (accession, pathway))
            payload = {"source": "Reactome", "raw_columns": fields, "relation": "member_of_pathway", "sign": "unknown", "quantity_unknown": True, "implies_mrna_direction": False}
            batches.edge(accession, pathway, "member_of_pathway", "unknown", False, "pathway_membership", "Reactome", payload)
            stats["human_memberships"] += 1
    stats["quantity_unknown_edges"] = stats["human_memberships"]
    return stats


def import_public(connection, sources: dict) -> dict:
    """Import provided public sources and return a JSON-serializable audit.

    Values are paths, or dictionaries containing ``path``. Keys are ``go_obo``,
    ``go_gaf``, ``legacy_graph``, ``omnipath_core``, ``collectri_raw``,
    ``signor_raw``, ``uniprot_tsv``,
    ``reactome_pathways``, ``reactome_hierarchy``, and ``reactome_membership``.
    Omitted sources are recorded as not provided. The caller owns the connection;
    this function uses a savepoint, and neither commits an outer transaction nor
    changes SQLite memory settings. Repeated identical records are ignored.

    JSON TEXT conventions: terms.aliases retains alt IDs, scoped synonyms and
    replacement suggestions; genes.synonyms retains primary name candidates and
    mapping status. Ambiguous primary gene names have an empty symbol field.
    """
    audit = {"schema": "jev.public_import.audit.v1", "sources": {}, "embedded_go_version": None, "complex_records_retained": 0, "quantity_unknown_edges": 0, "unknown_amount_flags_preserved": True, "implies_mrna_direction": False}
    aliases, obsolete = {}, {}
    parsers = {
        "legacy_graph": _legacy, "go_obo": _ontology, "go_gaf": _gaf,
        "omnipath_core": _omnipath,
        "collectri_raw": lambda path, batches: _omnipath(path, batches, "OmniPath/CollecTRI-raw", "transcriptional"),
        "signor_raw": lambda path, batches: _omnipath(path, batches, "OmniPath/SIGNOR-raw"),
        "uniprot_tsv": _uniprot,
        "reactome_pathways": _reactome_pathways,
        "reactome_hierarchy": _reactome_hierarchy,
        "reactome_membership": _reactome_membership,
    }
    connection.execute("SAVEPOINT public_import")
    try:
        for sql in _SCHEMA:
            connection.execute(sql)
        for key, parser in parsers.items():
            supplied = sources.get(key)
            if isinstance(supplied, dict):
                supplied = supplied.get("path")
            if not supplied:
                audit["sources"][key] = {"status": "not_provided"}
                continue
            path = Path(supplied).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            batches = _Batches(connection)
            if key == "go_obo":
                stats, aliases, obsolete = parser(path, batches)
                audit["embedded_go_version"] = stats["ontology_header"].get("data-version")
            elif key == "go_gaf":
                stats = parser(path, batches, aliases, obsolete)
                audit["embedded_gaf_header"] = stats["headers"]
            else:
                stats = parser(path, batches)
            sql_stats = batches.audit()
            audit["sources"][key] = {"status": "imported", "path": str(path), "size_bytes": path.stat().st_size, **stats, "sql": sql_stats}
            audit["complex_records_retained"] += stats.get("complex_records_retained", 0)
            audit["quantity_unknown_edges"] += stats.get("quantity_unknown_edges", 0)
        audit["counts"] = {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in _INSERT}
        connection.execute("RELEASE SAVEPOINT public_import")
    except BaseException:
        connection.execute("ROLLBACK TO SAVEPOINT public_import")
        connection.execute("RELEASE SAVEPOINT public_import")
        raise
    return audit

