"""Explicit commands only: help/plan never load weights or start training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COMMANDS = {
    "preprocess": ("data/preprocess.py", []),
    "labels": ("labels/build_resolution_bundle.py", []),
    "validate-labels": ("labels/validate_resolution_bundle.py", []),
    "prepare": ("core/prepare.py", []),
    "encode": ("core/encode.py", []),
    "train": ("core/train_memory_safe.py", ["train"]),
    "evaluate": ("core/train_memory_safe.py", ["evaluate"]),
    "restore-check": ("core/train_memory_safe.py", ["restore-check"]),
    "head-smoke": ("core/train_memory_safe.py", ["smoke"]),
    "run": ("core/pipeline.py", []),
    "preflight": ("core/preflight.py", []),
    "launch": ("core/launch.py", []),
    "import-knowledge": ("knowledge.py", []),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "check-config", "compare"], nargs="?")
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    if args.command == "compare":
        from .evaluation import main as evaluate_registry
        return evaluate_registry(argv[1:])
    if args.command == "check-config":
        from .config import load_config
        p = argparse.ArgumentParser(description="Resolve config paths without accessing data/weights")
        p.add_argument("--config", type=Path, required=True)
        cfg = load_config(p.parse_args(argv[1:]).config)
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return 0
    script, prefix = COMMANDS[args.command]
    # Script compatibility keeps the audited dynamic module imports intact.
    # Passing the source parent also supports `PYTHONPATH=src python -m jev_a4`.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(HERE.parent), env.get("PYTHONPATH", "")])
    return subprocess.call([sys.executable, str(HERE / script), *prefix, *argv[1:]], env=env)
