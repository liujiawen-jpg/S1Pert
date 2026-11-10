#!/usr/bin/env python3
"""Explicit, pinned source/asset acquisition; does not install or run baselines."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "gears": {"url": "https://github.com/snap-stanford/GEARS.git", "path": "external/GEARS",
              "commit": "f374e43e197b295016d80395d7a54ddb81cc6769"},
    "perturbqa": {"url": "https://github.com/Genentech/PerturbQA.git", "path": "external/PerturbQA",
                  "commit": "df2fe3a7802f2a46a43876b4f8674734aad3a30a"},
    "state": {"url": "https://github.com/ArcInstitute/state.git", "path": "external/state",
              "commit": "9bbfe78a434a55205e4de834e1ea99f85f7a3add"},
    "scgpt": {"url": "https://github.com/bowang-lab/scGPT.git", "path": "external/scGPT",
              "commit": "cebd6fae655b9c585a4807daa3ac31bb764f06b4",
              "reproduction_note": "Public rebuild pin; historical source lacked Git metadata, so equality is unverified."},
}
PIE_REVISION = "fb624a5158a7f37dfa13a015cdacadb985549d26"
PIE_BASE = f"https://huggingface.co/datasets/arcinstitute/PIE_sources/resolve/{PIE_REVISION}/esm2/"
ASSETS = {
    "gene2go": [{"id": "gears_gene2go", "path": "assets/baselines/gene2go_all.pkl",
                 "url": "https://dataverse.harvard.edu/api/access/datafile/6153417",
                 "size_bytes": 9462558, "checksums": {"md5": "77c9af0c61c30ea4d7a85680f4d122dc"},
                 "availability_note": "Official metadata verified; this environment's redirected HEAD request returned HTTP 403."}],
    "kg": [{"id": "perturbqa_kg", "path": "assets/baselines/kg.zip",
            "url": "https://zenodo.org/api/records/14915313/files/kg.zip/content",
            "size_bytes": 46080292, "checksums": {"md5": "90977691ebac65e02765e34c31b66bd1"}}],
    "genept": [{"id": "genept", "path": "assets/baselines/GenePT_emebdding_v2.zip",
                "url": "https://zenodo.org/api/records/10833191/files/GenePT_emebdding_v2.zip/content",
                "size_bytes": 574395233, "checksums": {"md5": "3f6ce4317e3a0091978ae5cb8fbf05a3"}}],
    "esm2": [
        {"id": "esm2_meta", "path": "assets/esm2_public/meta.json", "url": PIE_BASE + "meta.json",
         "size_bytes": 220332, "checksums": {"git_sha1": "a75b02fa311cd17094fe828f58e4d168550e504e"}},
        {"id": "esm2_embeddings", "path": "assets/esm2_public/embeddings.npy", "url": PIE_BASE + "embeddings.npy",
         "size_bytes": 98319488, "checksums": {"sha256": "427b867ccfddf2050686815b97063e2ceeb4044bc48c67b6baa8d3778f0bebcd"}},
        {"id": "esm2_aliases", "path": "assets/esm2_public/aliases.yaml", "url": PIE_BASE + "aliases.yaml",
         "size_bytes": 179, "checksums": {"git_sha1": "e4a5454c649ee47edc49c222824e21804d3641d2"}},
    ],
}
MANUAL = {
    "scgpt_pretrained": {"path": "assets/scgpt", "files": ["args.json", "best_model.pt", "vocab.json"],
        "source": "https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo",
        "official_whole_human_folder": "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
        "reason": "Official distribution is a Drive folder; exact historical checkpoint identity has not been verified."},
    "gene2go_if_host_unavailable": {"path": "assets/baselines/gene2go_all.pkl",
        "source": "https://dataverse.harvard.edu/api/access/datafile/6153417",
        "reason": "URL is pinned by GEARS source, but verification here reached HTTP 403 after redirect; acquire and verify explicitly."},
    "qwen3_8b": {"path": "assets/qwen3_8b", "source": "https://huggingface.co/Qwen/Qwen3-8B",
        "reason": "Supply a revision-pinned generic model snapshot and record hashes; baseline weights are not fetched by this helper."},
}


def downloader():
    spec = importlib.util.spec_from_file_location("s1pert_public_asset_download", ROOT / "scripts/download_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args, cwd=None):
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", *args],
                                   cwd=cwd, env=env, text=True, stderr=subprocess.PIPE).strip()


def verify_repo(path, spec):
    if path.is_symlink() or not (path / ".git").is_dir():
        raise ValueError("existing source is not a regular Git checkout; refusing replacement")
    if git("rev-parse", "HEAD", cwd=path) != spec["commit"]:
        raise ValueError("existing source commit differs; refusing replacement")
    if git("remote", "get-url", "origin", cwd=path) != spec["url"]:
        raise ValueError("existing source remote differs; refusing replacement")
    if git("status", "--porcelain", "--untracked-files=normal", cwd=path):
        raise ValueError("existing source has local changes; refusing replacement")


def fetch_repo(name, root):
    spec = REPOS[name]
    root = Path(root).resolve()
    destination = root / spec["path"]
    if not destination.resolve().is_relative_to(root):
        raise ValueError("external checkout escapes destination root")
    if destination.exists() or destination.is_symlink():
        verify_repo(destination, spec)
        return {"repo": name, "status": "verified_existing", "commit": spec["commit"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".fetch.lock")
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        # Only this new temporary checkout is cleaned if acquisition fails.
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "checkout"
            git("init", str(stage))
            git("remote", "add", "origin", spec["url"], cwd=stage)
            git("fetch", "--depth", "1", "origin", spec["commit"], cwd=stage)
            git("checkout", "--detach", "FETCH_HEAD", cwd=stage)
            verify_repo(stage, spec)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            stage.rename(destination)
        return {"repo": name, "status": "fetched", "commit": spec["commit"]}
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", choices=REPOS)
    parser.add_argument("--asset", action="append", choices=ASSETS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.list or args.dry_run:
        print(json.dumps({"root": str(args.root.resolve()),
            "repositories": {name: REPOS[name] for name in (args.repo or REPOS)},
            "assets": {name: ASSETS[name] for name in (args.asset or ASSETS)},
            "manual_requirements": MANUAL}, indent=2))
        return 0
    if not args.repo and not args.asset:
        parser.error("select --repo or --asset explicitly; use --list first")
    for name in dict.fromkeys(args.repo or []):
        print(json.dumps(fetch_repo(name, args.root)), flush=True)
    for name in dict.fromkeys(args.asset or []):
        for item in ASSETS[name]:
            print(json.dumps(downloader().download(item, args.root)), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"External asset acquisition stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
