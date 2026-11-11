#!/usr/bin/env python3
"""Write concrete model timing commands; writing a plan does not run models.

online: restore existing S1Pert/ALIGNED/SUMMER, measure complete test inference.
fresh: rebuild offline caches, train fresh S1Pert and the complete ALIGNED
       protocol, then measure native test inference. SUMMER is training-free.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
try:
    from .paper_common import work_root,write_json
except ImportError:
    from paper_common import work_root,write_json

def build(work_dir,mode='online',cells=('hepg2','k562_essential'),seed=17,output=None):
    root=work_root(work_dir);jobs=[]
    if mode=='fresh':
        config=json.loads((root/'configs/a4.json').read_text())
        if config.get('seeds')!=[17,29,43,61,89] or seed not in config['seeds']:
            raise ValueError('The five-way amortized paper recipe requires configured seeds [17,29,43,61,89].')
    for cell in cells:
        count=len(np.load(root/f'data/{cell}/seed_{seed}/test/records.npy',mmap_mode='r'))
        if count<=0:raise ValueError('No test pairs.')
        data='{work_dir}/data/'+cell
        if mode=='fresh':
            cache='{work_dir}/paper/runtime_cache/'+cell;model='{work_dir}/paper/runtime_models/'+cell+f'/seed_{seed}'
            phases=[dict(name='offline_cache',amortize_over=5,cache_id=f'a4_all_five_seeds_{cell}',fresh_outputs=[cache],
                command=['{python}','-m','jev_a4.core.encode','--config','{work_dir}/configs/a4.json','--dataset',cell,
                    '--data-dir',data,'--cache-dir',cache,'--all-seeds'],env={'PYTHONPATH':'{repo_dir}/src:{repo_dir}'}),
                dict(name='training',fresh_outputs=[model],command=['{python}','-m','jev_a4.core.train_memory_safe','train',
                    '--config','{work_dir}/configs/a4.json','--dataset',cell,'--seed',str(seed),
                    '--data-dir',data,'--cache-dir',cache,'--out',model],env={'PYTHONPATH':'{repo_dir}/src:{repo_dir}'})]
        elif mode=='online':
            cache='{work_dir}/cache/'+cell;model='{work_dir}/runs/main/'+cell+f'/seed_{seed}';phases=[]
        else:raise ValueError(mode)
        phases.append(dict(name='inference',factory='experiments.runtime_factories:s1pert_online',
            kwargs=dict(model_dir=model,data_dir=data,cache_dir=cache),warmup=1,repeat=3))
        jobs.append(dict(method='S1Pert',cell=cell,seed=seed,test_pairs=count,phases=phases))
        aligned='{work_dir}/baselines/aligned/runs/'+cell+f'/seed_{seed}'
        prepared='{work_dir}/baselines/aligned/prepared'
        phases=[]
        if mode=='fresh':
            aligned='{work_dir}/paper/runtime_aligned/'+cell+f'/seed_{seed}'
            prepared='{work_dir}/paper/runtime_aligned_prepared'
            phases=[dict(name='offline_cache',amortize_over=1,
                fresh_outputs=[prepared+'/'+cell+f'/seed_{seed}'],
                command=['{python}','-m','baselines.aligned.run','prepare','--work-dir','{work_dir}',
                    '--cell',cell,'--seed',str(seed),'--prepared-root',prepared],
                env={'PYTHONPATH':'{repo_dir}/src:{repo_dir}'}),
                dict(name='training',fresh_outputs=[aligned],
                command=['{python}','-m','baselines.aligned.run','train','--work-dir','{work_dir}',
                    '--cell',cell,'--seed',str(seed),'--prepared-root',prepared,'--output',aligned],
                env={'PYTHONPATH':'{repo_dir}/src:{repo_dir}'})]
        phases.append(dict(name='inference',factory='baselines.aligned.run:load_for_timing',
            kwargs=dict(work_dir='{work_dir}',cell=cell,seed=seed,run_dir=aligned,prepared_root=prepared),
            warmup=1,repeat=3))
        jobs.append(dict(method='ALIGNED',cell=cell,seed=seed,test_pairs=count,phases=phases,
            require_offline_cache=mode=='fresh'))
        summer='{work_dir}/baselines/summer' if mode=='online' else '{work_dir}/paper/runtime_summer/'+cell
        phases=[]
        if mode=='fresh':phases.append(dict(name='offline_cache',amortize_over=5,
            factory='experiments.runtime_factories:summer_offline',kwargs=dict(workspace=summer,
                config='{work_dir}/configs/summer.json',cell=cell,seed=seed)))
        phases.append(dict(name='inference',factory='experiments.runtime_factories:summer_online',
            kwargs=dict(workspace=summer,cell=cell,seed=seed),warmup=0,repeat=1))
        jobs.append(dict(method='SUMMER',cell=cell,seed=seed,test_pairs=count,phases=phases,
            require_offline_cache=mode=='fresh'))
    plan=dict(schema='s1pert.runtime-plan.v1',mode=mode,jobs=jobs,
        launch_environment={'PYTHONHASHSEED':'0','CUDA_VISIBLE_DEVICES':'match gpu_uuid in configs/a4.json and SUMMER configuration'},
        offline_amortization='Five split reuse: measured all-seed A4 cache or cell-wide public summaries divided by 5.',
        ALIGNED_status='Packaged audited adapter: cold public-graph preparation, native complete training, final aligned inference.',
        figure3_status='Fresh mode measures all three methods on both cell lines; plotting requires all jobs PASS.',
        fresh_scope='New runtime_cache/runtime_models/runtime_summer directories must be unused; scripts fail on conflicting sealed outputs.')
    destination=Path(output or root/f'paper/runtime_{mode}_plan.json')
    if destination.exists():raise FileExistsError(destination)
    write_json(destination,plan);return dict(plan=str(destination),jobs=len(jobs),models_started=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir');p.add_argument('--mode',choices=['online','fresh'],default='online')
    p.add_argument('--cells',nargs='+',default=['hepg2','k562_essential']);p.add_argument('--seed',type=int,default=17);p.add_argument('--output',type=Path)
    a=p.parse_args();print(json.dumps(build(a.work_dir,a.mode,a.cells,a.seed,a.output),indent=2))
