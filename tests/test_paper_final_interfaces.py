"""Independent offline checks of the final paper entry points and Table II."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from experiments import main_table2, paper, run_section
from plotting.make_table2_main import build


def test_all_method_plan_includes_packaged_aligned_without_execution(tmp_path):
    paper.initialize(tmp_path/'work',gpu='3')
    work=tmp_path/'work'
    commands=paper.plan(work,'baselines',[m for m in paper.METHODS if m!='S1Pert'])
    aligned=[cmd for cmd in commands if 'baselines.aligned.run' in cmd]
    assert len(aligned)==60
    assert {cmd[3] for cmd in aligned}=={'prepare','train','evaluate'}
    assert all('--config' in cmd and '--work-dir' in cmd for cmd in aligned)
    assert not (work/'runs').exists() and not (work/'baselines').exists()


def test_python_map_preserves_venv_executable_symlink(tmp_path,capsys):
    executable=tmp_path/'venv/bin/python';executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    mapping=tmp_path/'interpreters.json'
    mapping.write_text(json.dumps({'ALIGNED':'venv/bin/python'}))
    main_table2.main(['--work-dir',str(tmp_path/'work'),'--methods','ALIGNED',
                     '--python-map',str(mapping)])
    result=json.loads(capsys.readouterr().out)
    assert result['execute'] is False
    assert all(cmd[0]==str(executable) for cmd in result['commands'])


def test_section_plans_are_complete_and_use_workspace(tmp_path):
    fig2=run_section.section_plan('fig2',tmp_path)
    assert len(fig2)==10
    assert [cmd[3] for cmd in fig2[:3]]==['build','prepare','fidelity']
    assert [int(cmd[cmd.index('--seed')+1]) for cmd in fig2[3:8]]==list(paper.SEEDS)
    assert fig2[-1][2]=='plotting.plot_fig2_label_boundary'
    fig3=run_section.section_plan('fig3',tmp_path,'fresh')
    assert [cmd[2] for cmd in fig3]==['experiments.runtime_recipes','experiments.runtime','plotting.plot_fig3_runtime']
    assert all('--work-dir' in cmd and str(tmp_path) in cmd for cmd in fig2+fig3)
    mapping=tmp_path/'method_python.json'
    mapped=run_section.section_plan('fig3',tmp_path,'fresh',mapping)
    assert mapped[1][mapped[1].index('--python-map')+1]==str(mapping)
    assert '--python-map' not in mapped[0] and '--python-map' not in mapped[2]


def test_execute_propagates_required_runtime_environment(tmp_path,monkeypatch):
    work=tmp_path/'work';paper.initialize(work,gpu='3')
    monkeypatch.delenv('PYTHONHASHSEED',raising=False)
    seen=[]
    def capture(command,**kwargs):
        seen.append(kwargs['env'])
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(paper.subprocess,'run',capture)
    paper.execute([[sys.executable,'-c','pass']],work,'synthetic_env')
    assert seen[0]['PYTHONHASHSEED']=='0'
    assert seen[0]['CUDA_VISIBLE_DEVICES']=='3'
    assert str(REPO/'src') in seen[0]['PYTHONPATH']


def test_table2_raw_to_latex_best_second_and_exact_five_split_sd(tmp_path):
    methods=['S1Pert','GEARS','SUMMER'];entries=[]
    truth=np.array([0,1,2,0,1,2])
    predictions={'S1Pert':truth,'GEARS':np.array([0,1,2,1,2,0]),
                 'SUMMER':(truth+1)%3}
    for cell in paper.CELLS:
        for seed in paper.SEEDS:
            folder=tmp_path/cell/f'seed_{seed}';folder.mkdir(parents=True)
            rows=np.array([(i,0,int(y)) for i,y in enumerate(truth)],
                dtype=[('target','i4'),('query','i4'),('label','i1')])
            np.save(folder/'records.npy',rows)
            for method,pred in predictions.items():
                path=folder/f'{method}.npz'
                np.savez(path,target=rows['target'],query=rows['query'],labels=truth,y_pred=pred)
                entries.append(dict(method=method,cell=cell,seed=seed,
                    records=str((folder/'records.npy').relative_to(tmp_path)),predictions=str(path.relative_to(tmp_path))))
    registry=tmp_path/'predictions.csv'
    with registry.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=entries[0]);writer.writeheader();writer.writerows(entries)
    rows,summary=build(registry,tmp_path/'table2',methods=methods)
    assert len(rows)==60
    result=next(r for r in summary if (r['method'],r['cell'],r['metric'])==('GEARS','hepg2','macro_f1'))
    assert result['valid_splits']==5 and result['mean']==.5 and result['sample_sd']==0.
    tex=(tmp_path/'table2/table_II.tex').read_text()
    assert r'\mathbf{1.000 \pm 0.000}' in tex
    assert r'\underline{0.500 \pm 0.000}' in tex
    assert r'\dagger' not in tex
    assert 'Macro-F1 & BA & Up-P & Up-R & Up-F1 & Same-P & Same-R & Same-F1 & Down-P & Down-R & Down-F1' in tex
    table_rows=[line for line in tex.splitlines() if any(line.startswith(c+' & ') for c in ('HepG2','Jurkat','RPE1','K562'))]
    assert len(table_rows)==12 and all(line.count('&')==12 for line in table_rows)
    coverage=json.loads((tmp_path/'table2/coverage.json').read_text())
    assert coverage['complete'] is True and coverage['complete_paper_matrix'] is False
