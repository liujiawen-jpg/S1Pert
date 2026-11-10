"""Resolve explicit JSON configuration without depending on the working directory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sys

PATH_KEYS = {
    "project_root", "data_preprocessed_root", "label_resolution_root", "knowledge_db",
    "decider_snapshot", "tokenizer_snapshot", "decider_source", "data_root", "labels_root",
    "raw", "cache", "output", "source", "output_root", "raw_h5ad", "counts_path",
    "backbone_snapshot", "knowledge_graph",
}


def load_config(path: str | Path) -> dict:
    """Paths are relative to the config file; unresolved environment variables fail."""
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a JSON object")

    def resolve(value, follow_symlinks=True):
        expanded = os.path.expanduser(os.path.expandvars(str(value)))
        if re.search(r"\$(?:[A-Za-z_][A-Za-z_0-9]*|\{[^}]*\})", expanded):
            raise ValueError("unset environment variable in configured path")
        p = Path(expanded)
        absolute = path.parent / p if not p.is_absolute() else p
        return str(absolute.resolve()) if follow_symlinks else os.path.abspath(absolute)

    def visit(node):
        if isinstance(node, list):
            return [visit(v) for v in node]
        if not isinstance(node, dict):
            return node
        result = {}
        for key, value in node.items():
            if key == "raw_h5ad_paths" and isinstance(value, dict):
                result[key] = {k: resolve(v) for k, v in value.items()}
            elif key in PATH_KEYS and isinstance(value, str) and value:
                result[key] = resolve(value)
            else:
                result[key] = visit(value)
        return result

    cfg = visit(cfg)
    python = cfg.get("runtime_python")
    if python in (None, "", "python", "python3"):
        cfg["runtime_python"] = sys.executable
    elif os.sep in python:
        # Resolving a venv's executable symlink can silently switch interpreters
        # to the base installation and also change checkpoint configuration IDs.
        cfg["runtime_python"] = resolve(python, follow_symlinks=False)
    else:
        found = shutil.which(python)
        if not found:
            raise ValueError("configured runtime_python was not found")
        cfg["runtime_python"] = found
    if "decider_snapshot" in cfg:
        cfg.setdefault("tokenizer_snapshot", cfg["decider_snapshot"])
        cfg.setdefault("decider_source", cfg["decider_snapshot"])
    return cfg
