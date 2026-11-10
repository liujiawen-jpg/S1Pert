#!/usr/bin/env python3
"""Fetch explicitly selected public assets; listing/dry runs never use the network."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/downloads.json"
BLOCK = 1024 * 1024


def checksum(path, algorithm="sha256"):
    digest = hashlib.new("sha1" if algorithm == "git_sha1" else algorithm)
    if algorithm == "git_sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_manifest(path, root=None):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "s1pert.public-assets.v1":
        raise ValueError("unsupported asset manifest schema")
    root = Path(root).resolve() if root else (path.parent / document["root"]).resolve()
    identifiers, destinations = set(), set()
    for asset in document["assets"]:
        name = asset["id"]
        relative = Path(asset["path"])
        if name in identifiers or relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("duplicate asset ID or unsafe destination")
        identifiers.add(name)
        destination = root / relative
        if not destination.resolve().is_relative_to(root) or destination in destinations:
            raise ValueError("asset destination escapes root or is duplicated")
        destinations.add(destination)
        parsed = urlsplit(asset["url"])
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("asset URLs must be public HTTPS URLs without credentials")
        if asset.get("size_bytes") is not None and (type(asset["size_bytes"]) is not int or asset["size_bytes"] < 0):
            raise ValueError("invalid asset size")
        for algorithm, value in asset.get("checksums", {}).items():
            lengths = {"sha256": 64, "md5": 32, "git_sha1": 40}
            if algorithm not in lengths or not re.fullmatch(f"[0-9a-f]{{{lengths[algorithm]}}}", value):
                raise ValueError("invalid asset checksum")
    return document, root


def verify(path, asset):
    size = path.stat().st_size
    if asset.get("size_bytes") is not None and size != asset["size_bytes"]:
        raise ValueError(f"size mismatch for {asset['id']}: {size} != {asset['size_bytes']}")
    for algorithm, expected in asset.get("checksums", {}).items():
        if checksum(path, algorithm) != expected:
            raise ValueError(f"{algorithm} mismatch for {asset['id']}; existing bytes retained")
    return size


def download(asset, root, timeout=60):
    """Resume a checked partial file and atomically publish without overwriting."""
    root = Path(root).resolve()
    destination = root / asset["path"]
    if not destination.resolve().is_relative_to(root):
        raise ValueError("asset destination escapes root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    metadata = destination.with_name(destination.name + ".part.json")
    receipt = destination.with_name(destination.name + ".download.json")
    lock = destination.with_name(destination.name + ".download.lock")
    for path in (destination, partial, metadata, receipt, lock):
        if path.is_symlink():
            raise ValueError(f"refusing symlink asset or sidecar: {path.name}")
    identity = {key: asset.get(key) for key in ("url", "size_bytes", "checksums")}
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        if destination.exists():
            size = verify(destination, asset)
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["identity"] != identity or saved["sha256"] != checksum(destination):
                    raise ValueError("existing asset disagrees with its receipt; refusing overwrite")
            elif not asset.get("checksums"):
                raise ValueError("existing file has no published checksum or download receipt; use it directly only after independent review")
            return {"id": asset["id"], "status": "verified_existing", "size_bytes": size}

        state = json.loads(metadata.read_text()) if metadata.exists() else None
        if state is not None and state["identity"] != identity:
            raise ValueError("partial asset belongs to a different source; bytes retained")
        if partial.exists() and state is None:
            raise ValueError("partial asset has no source metadata; bytes retained")
        offset = partial.stat().st_size if partial.exists() else 0
        expected = asset.get("size_bytes")
        if expected is not None and offset > expected:
            raise ValueError("partial file exceeds declared asset size; bytes retained")
        # A fully received file may survive a crash immediately before publication.
        complete = partial.exists() and expected is not None and offset == expected
        if not complete:
            headers = {"User-Agent": "S1Pert-public-assets/1", "Accept-Encoding": "identity"}
            if offset:
                if not state.get("validator") and not asset.get("checksums"):
                    raise ValueError("cannot safely resume without a server validator or published checksum")
                headers["Range"] = f"bytes={offset}-"
                if state.get("validator"):
                    headers["If-Range"] = state["validator"]
            with urlopen(Request(asset["url"], headers=headers), timeout=timeout) as response:
                status = response.status
                encoding = response.headers.get("Content-Encoding", "identity")
                if encoding != "identity":
                    raise ValueError("server ignored identity content encoding")
                validator = response.headers.get("ETag")
                if not validator or validator.startswith("W/"):
                    validator = response.headers.get("Last-Modified")
                if offset:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if status != 206 or not match or int(match[1]) != offset:
                        raise ValueError("server did not honor resume range; partial bytes retained")
                    if expected is not None and int(match[3]) != expected:
                        raise ValueError("resumed source size changed")
                    if state.get("validator") and validator != state["validator"]:
                        raise ValueError("resumed source validator changed")
                    if int(match[2]) != int(match[3]) - 1:
                        raise ValueError("server returned an incomplete range")
                    transfer_size = int(match[3]) - offset
                else:
                    if status != 200:
                        raise ValueError(f"unexpected download status {status}")
                    transfer_size = int(response.headers["Content-Length"]) if "Content-Length" in response.headers else None
                    if expected is not None and transfer_size is not None and transfer_size != expected:
                        raise ValueError("server size differs from pinned manifest")
                if state is None:
                    state = {"identity": identity, "validator": validator,
                             "uniprot_release": response.headers.get("X-UniProt-Release")}
                    write_json(metadata, state)
                received = 0
                with partial.open("ab" if offset else "wb") as stream:
                    while chunk := response.read(BLOCK):
                        received += len(chunk)
                        if ((transfer_size is not None and received > transfer_size)
                                or (expected is not None and offset + received > expected)):
                            raise ValueError("response exceeds declared size")
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                if transfer_size is not None and received != transfer_size:
                    raise ValueError("incomplete response; partial bytes retained for resume")
        size = verify(partial, asset)
        record = {"identity": identity, "sha256": checksum(partial), "size_bytes": size,
                  "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "validator": state.get("validator"), "uniprot_release": state.get("uniprot_release"),
                  "published_checksum_verified": bool(asset.get("checksums"))}
        # Hard-link creation is atomic and fails if another file already exists.
        os.link(partial, destination)
        write_json(receipt, record)
        partial.unlink()
        metadata.unlink()
        return {"id": asset["id"], "status": "downloaded", "size_bytes": size,
                "published_checksum_verified": record["published_checksum_verified"]}
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, help="override repository root for asset destinations")
    parser.add_argument("--group", action="append", help="data, annotations, or backbone; repeatable")
    parser.add_argument("--asset", action="append", help="select an individual manifest ID; repeatable")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-weights", action="store_true", help="explicitly allow large backbone weights")
    args = parser.parse_args(argv)
    document, root = load_manifest(args.manifest, args.root)
    groups, ids = set(args.group or []), set(args.asset or [])
    known_groups = {x["group"] for x in document["assets"]}
    known_ids = {x["id"] for x in document["assets"]}
    if groups - known_groups or ids - known_ids:
        parser.error("unknown asset or group")
    if not groups and not ids and not (args.list or args.dry_run):
        parser.error("select --group or --asset explicitly; inspect --list first")
    selected = [x for x in document["assets"] if not (groups or ids) or x["group"] in groups or x["id"] in ids]
    if args.list or args.dry_run:
        print(json.dumps([{**x, "destination": str(root / x["path"]),
                           "weights_require_opt_in": x.get("weights", False)} for x in selected], indent=2))
        return 0
    if any(x.get("weights") for x in selected) and not args.include_weights:
        parser.error("this selection includes weights; add --include-weights after reviewing --dry-run")
    for asset in selected:
        print(json.dumps(download(asset, root)), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"Asset download stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
