"""Build a public-annotation-only SQLite input. Never import response tables."""
from __future__ import annotations

import argparse
import csv
import gzip
import os
from pathlib import Path
import sqlite3
import tempfile


def import_public_tsv(source: Path, destination: Path) -> int:
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("knowledge output already exists; choose a fresh path")
    opener = gzip.open if source.suffix == ".gz" else open
    destination.parent.mkdir(parents=True, exist_ok=True)
    with opener(source, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = ["accession", "symbol", "protein_name", "function_text"]
        uniprot_columns = ["Entry", "Gene Names (primary)", "Protein names", "Function [CC]"]
        native = set(columns).issubset(reader.fieldnames or [])
        uniprot = set(uniprot_columns).issubset(reader.fieldnames or [])
        if not native and not uniprot:
            raise ValueError("public TSV requires accession, symbol, protein_name, function_text or the documented UniProt export headers")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".public-functions-", suffix=".sqlite", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        db = sqlite3.connect(temporary)
        count = 0
        try:
            with db:
                db.execute("CREATE TABLE genes(accession TEXT NOT NULL, symbol TEXT NOT NULL, protein_name TEXT, function_text TEXT)")
                for row in reader:
                    values = [row[c] for c in (columns if native else uniprot_columns)]
                    if any(value is None for value in values) or None in row:
                        raise ValueError("malformed public annotation TSV row")
                    accession, symbol, protein, function = [value.strip() for value in values]
                    # Some reviewed UniProt entries lack a primary gene symbol.
                    # These cannot be joined to measured genes and are omitted.
                    if uniprot and not symbol:
                        continue
                    if not accession or not symbol or symbol.isdigit():
                        raise ValueError("public annotation contains an empty identity or numeric gene symbol")
                    symbols = symbol.split() if uniprot else [symbol]
                    for name in symbols:
                        if name.isdigit():
                            raise ValueError("numeric gene symbol in public annotation")
                        db.execute("INSERT INTO genes VALUES(?,?,?,?)", (accession, name, protein, function))
                        count += 1
                if not count:
                    raise ValueError("public annotation export contains no usable gene rows")
            db.close()
            # Publish only a complete database and never replace an existing one.
            os.link(temporary, destination)
        finally:
            db.close()
            temporary.unlink(missing_ok=True)
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", type=Path, required=True, help="Public annotations, optionally gzip-compressed")
    p.add_argument("--out", type=Path, required=True, help="New public-only SQLite database")
    args = p.parse_args()
    rows = import_public_tsv(args.tsv, args.out)
    print(f"Imported {rows} public annotation rows; no response tables read.")


if __name__ == "__main__":
    main()
