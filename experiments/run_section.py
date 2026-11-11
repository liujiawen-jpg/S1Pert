"""Plan or explicitly execute one complete paper section and its output renderer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments.paper import workspace, execute, SEEDS


def section_plan(section, work, mode="fresh", python_map=None):
    def call(module, *args):
        return [sys.executable, "-m", module, *map(str, args), "--work-dir", str(work)]
    if section == "table3": return [call("plotting.make_table3_stats")]
    if section == "table4":
        return [call("experiments.ablation", "all", "--main-run-root", work / "runs/main"),
                call("plotting.make_table4_ablation", "--main-run-root", work / "runs/main")]
    if section == "fig1":
        return [call("experiments.prior_stress"), call("plotting.plot_fig1_prior_stress")]
    if section == "fig2":
        return ([call("data_prep.cell_pool", stage) for stage in ("build", "prepare", "fidelity")]
                + [call("experiments.cell_pool", "predict", "--seed", seed) for seed in SEEDS]
                + [call("experiments.cell_pool", "evaluate"), call("plotting.plot_fig2_label_boundary")])
    if section == "fig3":
        runtime_args = ["--plan", work / f"paper/runtime_{mode}_plan.json",
                        "--output", work / f"paper/runtime_{mode}"]
        if python_map is not None:
            runtime_args += ["--python-map", Path(python_map).absolute()]
        result = [call("experiments.runtime_recipes", "--mode", mode),
                  call("experiments.runtime", *runtime_args)]
        if mode == "fresh": result.append(call("plotting.plot_fig3_runtime", "--timing", work / "paper/runtime_fresh/timings.json"))
        return result
    raise ValueError(section)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("section", choices=("table3", "table4", "fig1", "fig2", "fig3"))
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--runtime-mode", choices=("online", "fresh"), default="fresh")
    p.add_argument("--python-map", type=Path, help="Fig. 3 method-to-interpreter JSON; preserves separate CUDA/library stacks")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv); work = workspace(a.work_dir)
    if a.python_map and a.section != "fig3":
        p.error("--python-map here applies only to fig3; Table II has its own launcher")
    commands = section_plan(a.section, work, a.runtime_mode, a.python_map)
    print(json.dumps(dict(section=a.section, commands=commands, execute=a.execute,
                         fresh_runtime_mode_trains_new_timing_models=a.section=="fig3" and a.runtime_mode=="fresh"), indent=2))
    if a.execute: execute(commands, work, a.section)


if __name__ == "__main__": main()
