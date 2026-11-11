"""Run selected methods, register aligned predictions, and regenerate Table II."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments.paper import METHODS, workspace, plan, execute, register
from plotting.make_table2_main import build


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=[m for m in METHODS if m != "P1"])
    p.add_argument("--python-map", type=Path, help="JSON mapping method name to an environment's Python executable")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--evaluate-only", action="store_true", help="Skip execution; read already completed raw predictions")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--registry-out", type=Path)
    p.add_argument("--table-out", type=Path)
    a = p.parse_args(argv); work = workspace(a.work_dir)
    interpreters = json.loads(a.python_map.read_text()) if a.python_map else {}
    if a.python_map:
        # Keep the executable symlink: resolving a venv's bin/python to the
        # system binary discards that environment's installed dependencies.
        interpreters = {k: os.path.abspath(os.path.expanduser(str(a.python_map.parent / v)))
                        for k, v in interpreters.items() if v}
    commands = []
    if not a.evaluate_only:
        # Construct the whole plan before expensive work; unavailable adapters
        # fail here, not after days of successful runs of the other methods.
        for method in a.methods:
            commands += plan(work, "main" if method == "S1Pert" else "baselines", [method],
                             python=interpreters.get(method))
    # Multiple classical methods share one staging workspace and public input
    # preparation; initialize it once before the independently fitted models.
    unique = []; preparations = set()
    for cmd in commands:
        reusable = ("init" in cmd and "--family" in cmd) or ("prepare.py" in cmd)
        key = tuple(cmd[1:])
        if reusable and key in preparations: continue
        if reusable: preparations.add(key)
        unique.append(cmd)
    commands = unique
    print(json.dumps(dict(commands=commands, execute=a.execute, methods=a.methods,
                         complete_table_requires_four_cells_times_five_seeds=True), indent=2))
    if a.execute:
        execute(commands, work, "table2")
        registry = register(work, a.methods, a.allow_partial, a.registry_out)
        build(registry, a.table_out or work / "paper/table2", methods=a.methods, allow_partial=a.allow_partial)


if __name__ == "__main__": main()
