"""Small CPU-only execution tests; synthetic values are NOT paper results."""
import csv
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.paper_common import classification, auc_binary, summarize
from experiments.prior_stress import run as run_stress, train_prior, load_prediction
from experiments.runtime import run as run_runtime, measure_phase, read_python_map
from experiments.runtime_recipes import build as build_runtime_plan
from plotting.plot_fig1_prior_stress import render as plot_stress
from plotting.plot_fig3_runtime import render as plot_runtime

DTYPE=np.dtype([('target','i4'),('query','i4'),('label','i1')])

def rows(values):return np.array(values,dtype=DTYPE)

def make_workspace(tmp_path):
    registry=[]
    for cell in ['hepg2','k562_essential']:
        for seed in [17,29]:
            base=tmp_path/f'data/{cell}/seed_{seed}'
            (base/'train').mkdir(parents=True);(base/'test').mkdir()
            train=rows([(i,0,1) for i in range(60)]+[(i,1,0 if i==0 else 1) for i in range(12)]+
                       [(i,2,i%3) for i in range(3)])
            test=rows([(100+i,q,(i+q)%3) for i in range(6) for q in range(4)])
            np.save(base/'train/records.npy',train);np.save(base/'test/records.npy',test)
            for index,method in enumerate(['S1Pert','ALIGNED','GenePT-Prot','STATE','GEARS']):
                path=tmp_path/f'predictions/{method}/{cell}/seed_{seed}/test.npz';path.parent.mkdir(parents=True)
                rng=np.random.default_rng(seed+index)
                prob=rng.uniform(.1,.4,(len(test),3))
                prob[np.arange(len(test)),test['label']]+=1-index*.1
                prob/=prob.sum(1,keepdims=True)
                np.savez(path,target=test['target'],query=test['query'],labels=test['label'],probs=prob)
                registry.append(dict(method=method,cell=cell,seed=seed,
                    records=f'../data/{cell}/seed_{seed}/test/records.npy',
                    predictions=f'../predictions/{method}/{cell}/seed_{seed}/test.npz'))
    path=tmp_path/'paper/predictions.csv';path.parent.mkdir()
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=registry[0]);writer.writeheader();writer.writerows(registry)
    return path

def test_classification_invalid_is_not_dropped():
    result=classification([0,1,2],[0,-1,2])
    assert result['invalid_predictions']==1
    assert result['same_recall']==0
    assert result['macro_f1']==pytest.approx(2/3)
    assert result['within_gene_change_auc'] is None

def test_auc_ties_and_per_query():
    assert auc_binary([0,1,0,1],[.5,.5,.5,.5])==.5
    assert auc_binary([1,1],[.1,.2]) is None
    result=classification([0,1,2,1],[0,1,2,1],[0,0,1,1],[1,0,1,0],[1,.5,0,.5])
    assert result['within_gene_change_auc']==1
    assert result['within_gene_polarity_auc'] is None

def test_train_only_and_order_guard(tmp_path):
    train=rows([(0,0,1),(1,0,0)]);test=rows([(5,0,1),(5,1,2)])
    p,s,h=train_prior(train,test)
    np.testing.assert_allclose(p[1],p[0])
    assert s[1]==0
    with pytest.raises(ValueError,match='target overlap'):
        train_prior(train,rows([(0,0,1)]))
    path=tmp_path/'bad.npz'
    np.savez(path,target=test['target'],query=test['query'][::-1],labels=test['label'],probs=np.ones((2,3))/3)
    with pytest.raises(ValueError,match='alignment'):
        load_prediction(path,test)
    np.savez(path,target=test['target'],query=test['query'],labels=test['label'],probs=np.array([[0,1,0],[1,0,0]]))
    pred,change,polarity=load_prediction(path,test)
    np.testing.assert_array_equal(polarity,[0,1])

def test_raw_to_stress_summary_and_figure(tmp_path):
    make_workspace(tmp_path)
    report=run_stress(tmp_path)
    assert report['status']=='PASS' and report['split_count']==4
    assert report['entropy_mode']=='fixed-thirds'
    summary=list(csv.DictReader((tmp_path/'paper/prior_stress/summary.csv').open()))
    assert all(int(r['n_splits'])==4 for r in summary if r['cell']=='ALL_CELLS' and r['metric']=='macro_f1')
    figure=plot_stress(tmp_path,expected_splits=4,font='A deliberately absent font')
    assert figure['status']=='PASS' and figure['font']['selected']=='DejaVu Serif'
    assert (tmp_path/'paper/figures/fig_prior_stress.pdf').stat().st_size>1000
    assert len(figure['annotation_audit'])==6

def test_summary_preserves_all_comparators():
    records=[]
    for seed in [17,29]:
        for method,v in [('S1Pert',.8),('X',.4),('Y',.5)]:
            records.append(dict(model=method,cell='hepg2',seed=seed,regime='resolved',macro_f1=v))
    summary=summarize(records)
    row=next(r for r in summary if (r['model'],r['cell'],r['metric'])==('X','ALL_CELLS','macro_f1'))
    assert row['paired_n']==2 and row['wilcoxon_holm_p']>=row['wilcoxon_two_sided_p']

def test_actual_subprocess_runtime_and_figure(tmp_path):
    jobs=[]
    command=[sys.executable,'-c','import time; time.sleep(0.002)']
    for cell in ['hepg2','k562_essential']:
        for method in ['S1Pert','ALIGNED','SUMMER']:
            phases=[]
            if method=='S1Pert':phases.append(dict(name='offline_cache',command=command,amortize_over=5))
            if method!='SUMMER':phases.append(dict(name='training',command=command))
            phases.append(dict(name='inference',command=command,n_pairs=100,repeat=2))
            jobs.append(dict(method=method,cell=cell,seed=17,test_pairs=100,phases=phases))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(jobs=jobs)))
    report=run_runtime(tmp_path,plan)
    assert report['status']=='PASS'
    for row in report['jobs']:
        assert row['online_seconds_per_10000']>0 and row['end_to_end_seconds']>0
        assert row['online_scope']=='subprocess_wall_including_load'
        assert row['end_to_end_status']=='COMPLETE'
    ours=report['jobs'][0]
    offline=ours['phases'][0]
    assert offline['amortized_seconds']==pytest.approx(offline['seconds']/5)
    fig=plot_runtime(tmp_path,font='DejaVu Serif')
    assert fig['status']=='PASS'
    assert (tmp_path/'paper/figures/fig_runtime_efficiency.pdf').stat().st_size>1000
    with pytest.raises(FileExistsError):run_runtime(tmp_path,plan)

def test_runtime_rejects_incomplete_test_axis(tmp_path):
    command=[sys.executable,'-c','pass']
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({'jobs':[dict(method='SUMMER',cell='hepg2',seed=17,test_pairs=200,
        phases=[dict(name='inference',command=command,n_pairs=100)])]}))
    result=run_runtime(tmp_path,plan)
    assert result['jobs'][0]['end_to_end_seconds'] is None
    assert result['jobs'][0]['end_to_end_status']=='INCOMPLETE_TEST_AXIS'

def test_actual_recipe_builder_does_not_launch_models(tmp_path):
    make_workspace(tmp_path)
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/a4.json').write_text(json.dumps({'seeds':[17,29,43,61,89]}))
    result=build_runtime_plan(tmp_path,mode='fresh')
    assert result['models_started'] is False
    plan=json.loads(Path(result['plan']).read_text())
    assert len(plan['jobs'])==6
    assert plan['ALIGNED_status'].startswith('Packaged audited adapter')
    a4=plan['jobs'][0]
    assert a4['test_pairs']==24
    assert a4['phases'][0]['amortize_over']==5
    assert 'jev_a4.core.encode' in a4['phases'][0]['command']
    assert a4['phases'][2]['factory']=='experiments.runtime_factories:s1pert_online'
    assert not (tmp_path/'paper/runtime_cache').exists()
    assert plan['jobs'][1]['phases'][2]['factory']=='baselines.aligned.run:load_for_timing'
    assert plan['jobs'][1]['phases'][0]['amortize_over']==1
    assert '--prepared-root' in plan['jobs'][1]['phases'][0]['command']
    assert plan['jobs'][2]['phases'][0]['factory']=='experiments.runtime_factories:summer_offline'

def test_cold_runtime_refuses_existing_cache(tmp_path):
    cache=tmp_path/'existing_cache';cache.mkdir()
    phase=dict(name='offline_cache',fresh_outputs=[str(cache)],
        command=[sys.executable,'-c','raise AssertionError("must not execute")'])
    with pytest.raises(FileExistsError,match='already exists'):
        measure_phase(phase,tmp_path,tmp_path/'logs')
    with pytest.raises(ValueError,match='Cold offline'):
        measure_phase(dict(name='offline_cache',repeat=2,command=['unused']),tmp_path,tmp_path/'logs')

def test_factory_cleanup_on_failed_timing(tmp_path,monkeypatch):
    state={'cleaned':False}
    module=types.ModuleType('_s1pert_test_runtime_factory')
    def factory():
        def run():raise RuntimeError('intentional synthetic failure')
        def cleanup():state['cleaned']=True
        return dict(run=run,cleanup=cleanup,n_pairs=3,device='cpu')
    module.factory=factory
    monkeypatch.setitem(sys.modules,module.__name__,module)
    with pytest.raises(RuntimeError,match='intentional synthetic failure'):
        measure_phase(dict(name='inference',factory=module.__name__+':factory'),tmp_path,tmp_path/'logs')
    assert state['cleaned'] is True

def test_python_map_preserves_environment_symlink_and_validates(tmp_path):
    interpreter=tmp_path/'environment/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    path=tmp_path/'python_map.json'
    path.write_text(json.dumps({'S1Pert':'environment/bin/python'}))
    interpreters,provenance=read_python_map(path)
    assert interpreters['S1Pert']==str(interpreter)
    assert provenance['sha256']
    path.write_text(json.dumps({'SUMMER':'missing/python'}))
    with pytest.raises(ValueError,match='not executable'):
        read_python_map(path)

def isolated_plan(tmp_path,fail=False):
    phase=dict(name='inference',factory='runtime_worker_fixture:build',repeat=2,warmup=1,
        kwargs=dict(cleanup_path=str(tmp_path/'cleaned'),fail=fail),
        env=dict(PYTHONPATH=str(Path(__file__).parent),S1PERT_TEST_MARKER='separate-method'))
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(jobs=[dict(method='SUMMER',cell='hepg2',seed=17,
        test_pairs=12,phases=[phase])])))
    mapping=tmp_path/'python_map.json'
    mapping.write_text(json.dumps({'SUMMER':sys.executable}))
    return plan,mapping

def test_isolated_factory_measures_inside_worker_not_process_start(tmp_path):
    plan,mapping=isolated_plan(tmp_path)
    report=run_runtime(tmp_path,plan,python_map=mapping)
    assert report['status']=='PASS'
    job=report['jobs'][0];phase=job['phases'][0]
    assert phase['isolated_factory'] is True
    assert phase['scope']=='loaded_model_online'
    assert phase['worker_environment']['pid']!=os.getpid()
    assert phase['execution_python']==sys.executable
    assert phase['worker_environment']['python']==sys.executable
    assert len(phase['raw_timings'])==3
    assert sum(t['warmup'] for t in phase['raw_timings'])==1
    measured=[t['seconds'] for t in phase['raw_timings'] if not t['warmup']]
    assert phase['seconds']==pytest.approx(np.mean(measured))
    assert phase['seconds']<phase['worker_wall_seconds']
    assert phase['worker_overhead_seconds']>0
    assert phase['seconds_per_10000']==pytest.approx(phase['seconds']*10000/12)
    assert job['end_to_end_seconds']==pytest.approx(
        phase['seconds']+phase['setup_seconds']+phase['worker_overhead_seconds'])
    assert all(t['result']['pid']==phase['worker_environment']['pid'] for t in phase['raw_timings'])
    assert all(t['result']['marker']=='separate-method' for t in phase['raw_timings'])
    assert (tmp_path/'cleaned').read_text()=='cleaned'
    assert report['python_map']['interpreters']=={'SUMMER':sys.executable}

def test_isolated_factory_failure_remains_failure(tmp_path):
    plan,mapping=isolated_plan(tmp_path,fail=True)
    with pytest.raises(RuntimeError,match='Isolated method phase failed'):
        run_runtime(tmp_path,plan,python_map=mapping)
    output=tmp_path/'paper/runtime'
    assert json.loads((output/'timings.json').read_text())['status']=='FAILED'
    worker=json.loads((output/'logs/SUMMER/hepg2/seed_17/inference/worker_result.json').read_text())
    assert worker['status']=='FAILED' and 'intentional isolated' in worker['error']
    assert (tmp_path/'cleaned').read_text()=='cleaned'
    assert not (output/'per_split.csv').exists()

def test_python_map_selects_fresh_command_interpreter(tmp_path):
    interpreter=tmp_path/'selected-python'
    interpreter.symlink_to(sys.executable)
    mapping=tmp_path/'python_map.json';mapping.write_text(json.dumps({'SUMMER':str(interpreter)}))
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(jobs=[dict(method='SUMMER',cell='hepg2',seed=17,test_pairs=5,
        phases=[dict(name='inference',n_pairs=5,
            command=['{python}','-c','import sys; print(sys.executable)'])]) ])))
    report=run_runtime(tmp_path,plan,python_map=mapping)
    phase=report['jobs'][0]['phases'][0]
    assert phase['isolated_factory'] is False
    assert phase['executed_phase']['command'][0]==str(interpreter)
    assert Path(phase['raw_timings'][0]['log']).read_text().strip()==str(interpreter)
    assert phase['scope']=='subprocess_wall_including_load'
