"""Versioned, response-blind public graph and bounded query retrieval.

Activity and transcription are different quantities. Every propagated sign is
conditional, not a measured RNA response. No graph operation accepts labels.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import csv
import hashlib
import io
import json
import math
import re


RELATIONS = {"transcriptional": "rna_expression", "activity_regulation": "protein_activity",
             "post_translational": "protein_regulation_unspecified",
             "association": "unspecified", "annotation": "ontology_membership"}
SIGNS = {"positive", "negative", "unknown", "conflicting"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Edge:
    subject: str
    object: str
    relation: str
    sign: str
    directed: bool
    source: str
    source_record: str
    sources: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    applicable_context: str | None = None
    sign_scope: str = "source_record"

    def __post_init__(self):
        for name in ("subject", "object", "source", "source_record"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
                raise ValueError("invalid graph field: " + name)
        if self.relation not in RELATIONS or self.sign not in SIGNS or type(self.directed) is not bool:
            raise ValueError("invalid relation type, sign or direction")
        if self.source not in {"GO", "STRING", "OmniPath/CollecTRI", "OmniPath/SIGNOR", "fixture"}:
            raise ValueError("unsupported public provenance")
        if self.relation in ("association", "annotation") and self.sign != "unknown":
            raise ValueError("association/annotation cannot supply a causal sign")
        if self.source in ("GO", "STRING") and self.relation not in ("association", "annotation"):
            raise ValueError("GO/STRING must not be promoted to signed regulation")
        if self.sign != "unknown" and not self.directed:
            raise ValueError("signed record requires direction")
        if self.applicable_context is not None and (not isinstance(self.applicable_context, str) or not self.applicable_context):
            raise ValueError("invalid context")
        if self.sign_scope not in ("source_record", "aggregated_omnipath_record"):
            raise ValueError("invalid sign evidence scope")
        for values in (self.sources, self.references):
            if not isinstance(values, tuple) or any(not isinstance(x, str) or not x or len(x) > 256 for x in values):
                raise ValueError("invalid source/reference list")

    @property
    def uid(self):
        return identity(asdict(self))[:24]

    @property
    def quantity(self):
        return RELATIONS[self.relation]

    def record(self):
        return {**asdict(self), "id": self.uid, "object_quantity": self.quantity}


def edge_from_record(record):
    keys = set(Edge.__dataclass_fields__)
    if set(record) != keys:
        raise ValueError("graph record contains unknown or missing fields")
    row = dict(record)
    for key in ("sources", "references"):
        row[key] = tuple(row[key])
    return Edge(**row)


def parse_omnipath(text, *, dataset, source_sha256):
    """Parse only explicitly scoped human CollecTRI or SIGNOR API snapshots.

    UniProt complexes are excluded rather than split into unsupported single-gene
    claims. Conflicting signs remain conflicting, never arbitrarily resolved.
    """
    if dataset not in ("collectri", "signor") or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("invalid snapshot identity")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    required = {"source", "target", "source_genesymbol", "target_genesymbol", "is_directed",
                "is_stimulation", "is_inhibition", "sources", "references", "type",
                "ncbi_tax_id_source", "ncbi_tax_id_target"}
    if not required.issubset(reader.fieldnames or ()):
        raise ValueError("OmniPath response lacks required fields: " + str(sorted(required - set(reader.fieldnames or ()))))
    accepted, rejected = [], defaultdict(int)
    def flag(value):
        if value not in ("0", "1", "False", "True", "false", "true"):
            raise ValueError("invalid boolean in public API response")
        return value in ("1", "True", "true")
    for line, row in enumerate(reader, 2):
        if row["ncbi_tax_id_source"] != "9606" or row["ncbi_tax_id_target"] != "9606":
            rejected["nonhuman"] += 1
            continue
        if any("COMPLEX" in row[key].upper() for key in ("source", "target")):
            rejected["complex_entity"] += 1
            continue
        expected = "transcriptional" if dataset == "collectri" else "post_translational"
        if row["type"] != expected:
            rejected["other_quantity"] += 1
            continue
        symbols = [row[k] for k in ("source_genesymbol", "target_genesymbol")]
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", x or "") for x in symbols):
            rejected["missing_or_ambiguous_symbol"] += 1
            continue
        directed, positive, negative = (flag(row[k]) for k in ("is_directed", "is_stimulation", "is_inhibition"))
        if not directed:
            rejected["undirected"] += 1
            continue
        sources = tuple(sorted(set(filter(None, row["sources"].split(";")))))
        expected_source = "CollecTRI" if dataset == "collectri" else "SIGNOR"
        if not any(expected_source in source for source in sources):
            raise ValueError("API returned an out-of-scope resource")
        sign = "conflicting" if positive and negative else "positive" if positive else "negative" if negative else "unknown"
        accepted.append(Edge(*symbols, expected, sign, True, "OmniPath/" + expected_source,
                             f"{source_sha256}:{line}", sources,
                             tuple(sorted(set(filter(None, row["references"].split(";"))))),
                             sign_scope="aggregated_omnipath_record"))
    return accepted, {"accepted": len(accepted), "rejected": dict(rejected), "dataset": dataset}

