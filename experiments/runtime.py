#!/usr/bin/env python3
"""Stage 7: execute real timing phases; never infer runtimes from metric CSVs.

JSON plan contains jobs with method,cell,seed and phases. Each phase specifies
name, command (argument array; no shell), repeat, optional warmup, and optional
n_pairs. Paths/arguments accept {work_dir} and {python}. A reusable offline phase
may specify cache_id and amortize_over: the first actual measurement is reused
inside this invocation, and its measured cost is divided explicitly.

For loaded-model online latency use factory='module:function', kwargs={...}.
The factory initializes/restores a model OUTSIDE the timed region and returns
{'run': callable, 'synchronize': callable, 'n_pairs': int}. CUDA factories must
provide synchronize (torch.cuda.synchronize); command wall times include process
startup/loading and are labeled accordingly, not called loaded-model latency.
"""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'src'))
try:
    from .paper_common import *
except ImportError:
    from paper_common import *

def expand(value, root, python=None):
    if isinstance(value,str):return value.replace('{work_dir}',str(root)).replace('{python}',python or sys.executable).replace('{repo_dir}',str(REPO))
    if isinstance(value,list):return [expand(v,root,python) for v in value]
    if isinstance(value,dict):return {k:expand(v,root,python) for k,v in value.items()}
    return value

def read_python_map(path):
    if path is None:return {},None
    path=Path(path).absolute()
    document=json.loads(path.read_text())
    if not isinstance(document,dict):raise ValueError('Python map must be an object keyed by method.')
    result={}
    for method,value in document.items():
        if value in (None,''):continue
        if not isinstance(method,str) or not isinstance(value,str):raise ValueError('Python map values must be executable paths.')
        # Resolving bin/python itself would discard a virtual environment.
        value=os.path.expanduser(os.path.expandvars(value))
        executable=os.path.abspath(os.path.join(path.parent,value))
        if not Path(executable).is_file() or not os.access(executable,os.X_OK):
            raise ValueError(f'Mapped Python is not executable: {method}: {executable}')
        result[method]=executable
    return result,dict(path=str(path),sha256=sha256(path),interpreters=result)

def measure_phase(phase, root, log_dir, python=None):
    repeat=int(phase.get('repeat',1));warmup=int(phase.get('warmup',0))
    if repeat<1 or warmup<0:raise ValueError('repeat >= 1 and warmup >= 0 required.')
    if phase['name']=='offline_cache' and (repeat!=1 or warmup!=0):
        raise ValueError('Cold offline builds run once per fresh destination; warm cache repetitions are not cold-build timing.')
    phase=expand(phase,root,python);log_dir.mkdir(parents=True,exist_ok=True)
    for destination in phase.get('fresh_outputs',[]):
        if Path(destination).exists():
            raise FileExistsError(f'Cold benchmark destination already exists: {destination}')
    factory=phase.get('factory'); command=phase.get('command')
    if bool(factory)==bool(command):raise ValueError('Exactly one of factory or command is required.')
    startup_start=time.perf_counter(); runner=sync=None;n_pairs=phase.get('n_pairs');factory_source=None
    if factory:
        module,name=factory.split(':',1)
        loaded=importlib.import_module(module)
        if getattr(loaded,'__file__',None):
            factory_source=dict(path=loaded.__file__,sha256=sha256(loaded.__file__))
        context=getattr(loaded,name)(**phase.get('kwargs',{}))
        runner=context['run'];sync=context.get('synchronize',lambda:None)
        if context.get('device','cpu').startswith('cuda') and 'synchronize' not in context:
            raise ValueError('CUDA factory must supply synchronize; asynchronous timing is invalid.')
        n_pairs=context.get('n_pairs',n_pairs)
        scope='loaded_model_online' if phase['name']=='inference' else 'in_process_phase'
    else:
        if not isinstance(command,list) or not command or not all(isinstance(s,str) for s in command):
            raise ValueError('command must be a nonempty argument array; shell scripts require explicit interpreter.')
        scope='subprocess_wall_including_load'
    setup_seconds=time.perf_counter()-startup_start
    records=[]
    try:
        for iteration in range(-warmup,repeat):
            log_path=log_dir/f'{"warmup" if iteration<0 else "repeat"}_{abs(iteration)}.log'
            if sync:sync()
            start=time.perf_counter()
            result=None
            if runner:
                result=runner()
                if sync:sync()
                if isinstance(result,dict) and result.get('status') in {'FAIL','FAILED','ERROR'}:
                    raise RuntimeError('Benchmark callable reported failure.')
            else:
                cwd=Path(phase.get('cwd',root))
                if not cwd.is_absolute():cwd=root/cwd
                env=os.environ.copy();env.update({str(k):str(v) for k,v in phase.get('env',{}).items()})
                with log_path.open('w') as stream:
                    completed=subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,
                        timeout=phase.get('timeout_seconds'),check=False)
                if completed.returncode:raise RuntimeError(f'Phase failed with exit {completed.returncode}: {log_path}')
            elapsed=time.perf_counter()-start
            records.append(dict(iteration=iteration,warmup=iteration<0,seconds=elapsed,
                scope=scope,log=str(log_path) if not runner else None,
                result=result if isinstance(result,dict) else None))
    finally:
        if factory and context.get('cleanup'):
            context['cleanup']()
    metadata=context.get('metadata',{}) if factory else {}
    measured=[r['seconds'] for r in records if not r['warmup']]
    amortize=int(phase.get('amortize_over',1))
    if amortize<1:raise ValueError('amortize_over must be positive.')
    if phase['name']!='offline_cache' and amortize!=1:raise ValueError('Only offline cache can be amortized.')
    mean=float(np.mean(measured))
    return dict(status='PASS',name=phase['name'],seconds=mean,
        sample_sd=float(np.std(measured,ddof=1)) if len(measured)>1 else None,
        repeat=repeat,warmup=warmup,setup_seconds=setup_seconds,
        amortize_over=amortize,amortized_seconds=mean/amortize,
        n_pairs=int(n_pairs) if n_pairs is not None else None,
        seconds_per_10000=mean*10000/int(n_pairs) if n_pairs else None,
        scope=scope,raw_timings=records,executed_phase=phase,metadata=metadata,factory_source=factory_source)

def measure_isolated_factory(phase,root,log_dir,python):
    """Measure inside the method environment; process startup is never online latency."""
    log_dir.mkdir(parents=True,exist_ok=True)
    request=log_dir/'worker_request.json';result=log_dir/'worker_result.json';log=log_dir/'worker.log'
    if any(p.exists() for p in (request,result,log)):
        raise FileExistsError('Isolated phase artifacts already exist; use a fresh timing output.')
    worker=REPO/'experiments/runtime_worker.py'
    executed=expand(phase,root,python)
    write_json(request,dict(phase=executed,work_dir=str(root),log_dir=str(log_dir/'phase')))
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(REPO/'src'),str(REPO),env.get('PYTHONPATH','')])
    env.update({str(k):str(v) for k,v in executed.get('env',{}).items()})
    command=[python,str(worker),'--request',str(request),'--result',str(result)]
    start=time.perf_counter()
    with log.open('x') as stream:
        completed=subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,
            timeout=executed.get('timeout_seconds'),check=False)
    wall=time.perf_counter()-start
    if completed.returncode or not result.is_file():
        raise RuntimeError(f'Isolated method phase failed (exit {completed.returncode}); see {log}')
    payload=json.loads(result.read_text())
    if payload.get('status')!='PASS' or payload.get('request_sha256')!=sha256(request):
        raise ValueError('Isolated result failed or does not bind the submitted request.')
    measured=payload['measurement']
    if measured.get('status')!='PASS' or measured.get('executed_phase')!=executed:
        raise ValueError('Isolated result phase identity differs.')
    accounted=measured['setup_seconds']+sum(row['seconds'] for row in measured['raw_timings'])
    overhead=max(0.,wall-accounted)
    measured.update(execution_python=python,isolated_factory=True,worker_environment=payload['environment'],
        worker_wall_seconds=wall,worker_overhead_seconds=overhead,
        worker_overhead_scope='Interpreter/import/IPC/cleanup outside factory setup and timed repetitions; excluded from online latency.',
        worker_source_sha256=sha256(worker),worker_request_sha256=sha256(request),
        worker_result_sha256=sha256(result),worker_log=str(log))
    return measured

def run(work_dir, plan, output=None, python_map=None):
    root=work_root(work_dir);plan=Path(plan).resolve()
    interpreters,map_provenance=read_python_map(python_map)
    cfg=json.loads(plan.read_text());jobs=cfg.get('jobs',[])
    if not jobs:raise ValueError('Timing plan has no jobs.')
    output=Path(output or root/'paper/runtime').resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'timings.json').exists():
        raise FileExistsError('Choose a new --output; completed benchmark logs are not overwritten.')
    phases=[];jobs_summary=[];cached={};seen=set();overall_start=time.perf_counter()
    packages={}
    for name in ['numpy','scipy','torch','transformers','vllm']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    manifest=dict(status='RUNNING',plan=str(plan),plan_sha256=sha256(plan),work_dir=str(root),jobs=[],
        execution_environment=dict(python=sys.version,platform=platform.platform(),packages=packages,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),pythonhashseed=os.environ.get('PYTHONHASHSEED')),
        python_map=map_provenance,
        runner_source_sha256=sha256(__file__),
        methodology='Actual measured wall times, explicit scopes, raw repetitions, no static timing constants.')
    write_json(output/'timings.json',manifest)
    try:
        for job in jobs:
            method,cell,seed=job['method'],job['cell'],int(job['seed'])
            key=(method,cell,seed)
            if key in seen:raise ValueError(f'Duplicate runtime job {key}')
            seen.add(key);current=[]
            execution_python=interpreters.get(method,sys.executable)
            for phase in job['phases']:
                if phase['name'] not in {'offline_cache','training','inference','preprocessing'}:
                    raise ValueError('Phase must be offline_cache, training, inference, or preprocessing.')
                cache_id=phase.get('cache_id')
                if cache_id and phase['name']!='offline_cache':raise ValueError('cache_id is only for offline_cache.')
                identity=json.dumps(dict(phase=expand({k:v for k,v in phase.items() if k!='cache_id'},root,execution_python),
                    execution_python=execution_python),sort_keys=True)
                if cache_id in cached:
                    old_identity,measurement=cached[cache_id]
                    if identity!=old_identity:raise ValueError('Reused cache_id has different command or amortization.')
                    measurement=dict(measurement,reused_measurement=True)
                else:
                    log_dir=output/'logs'/method/cell/f'seed_{seed}'/phase['name']
                    if method in interpreters and phase.get('factory'):
                        measurement=measure_isolated_factory(phase,root,log_dir,execution_python)
                    else:
                        measurement=measure_phase(phase,root,log_dir,execution_python)
                        measurement.update(execution_python=execution_python,isolated_factory=False)
                    measurement['reused_measurement']=False
                    if cache_id:cached[cache_id]=(identity,measurement)
                row=dict(method=method,cell=cell,seed=seed,**measurement)
                phases.append(row);current.append(row)
            names=[p['name'] for p in current]
            if len(names)!=len(set(names)):raise ValueError('Duplicate phase name in one job.')
            infer=next((p for p in current if p['name']=='inference'),None)
            if infer is None or not infer['n_pairs'] or infer['n_pairs']<=0:
                raise ValueError('Every runtime job must measure inference and supply positive n_pairs.')
            required={'training','inference'} if method!='SUMMER' else {'inference'}
            if method=='S1Pert':required.add('offline_cache')
            if job.get('require_offline_cache'):required.add('offline_cache')
            complete=required.issubset(names)
            # S1Pert figures include cache+training+test inference. Other phases
            # remain in the log, but preprocessing is excluded consistently.
            # Factory setup includes model restoration outside ONLINE timing,
            # but remains part of END-TO-END cost. Subprocess loading is already
            # inside its measured wall time and is not added a second time.
            end_to_end=sum(p['amortized_seconds'] +
                ((p['setup_seconds']+p.get('worker_overhead_seconds',0.))/p['amortize_over'] if p['executed_phase'].get('factory') else 0.)
                for p in current if p['name'] in {'offline_cache','training','inference'}) if complete else None
            row=dict(method=method,cell=cell,seed=seed,n_test_pairs=infer['n_pairs'],
                online_seconds_per_10000=infer['seconds_per_10000'],online_scope=infer['scope'],
                end_to_end_seconds=end_to_end,end_to_end_status='COMPLETE' if complete else 'INCOMPLETE_PHASES',
                end_to_end_scope=' + '.join(n for n in names if n!='preprocessing')+'; cache amortization explicitly recorded',
                missing_required_phases=','.join(sorted(required-set(names))))
            # End-to-end inference must be the COMPLETE test split, not a 10k
            # slice. Specify test_pairs to make this assertion machine-checkable.
            if not job.get('test_pairs') or int(job['test_pairs'])!=infer['n_pairs']:
                row.update(end_to_end_seconds=None,end_to_end_status='INCOMPLETE_TEST_AXIS')
            jobs_summary.append(row);manifest['jobs'].append(dict(**row,phases=current))
            write_json(output/'timings.json',manifest)
        manifest.update(status='PASS',runner_elapsed_seconds=time.perf_counter()-overall_start)
        write_json(output/'timings.json',manifest)
        write_csv(output/'per_split.csv',jobs_summary)
        write_csv(output/'phases.csv',[{k:v for k,v in row.items() if k not in {'raw_timings','executed_phase','metadata','factory_source','worker_environment'}} for row in phases])
        return manifest
    except Exception as error:
        manifest.update(status='FAILED',error=repr(error),runner_elapsed_seconds=time.perf_counter()-overall_start)
        write_json(output/'timings.json',manifest)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir');parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--python-map',type=Path,help='JSON method -> Python path; factories run in isolated method environments.')
    args=parser.parse_args();print(json.dumps(run(args.work_dir,args.plan,args.output,args.python_map),indent=2))
