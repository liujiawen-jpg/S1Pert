"""Actual loaded-model benchmark factories for the packaged A4 and SUMMER code."""
from __future__ import annotations
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))

def _summer_modules(workspace):
    code=Path(workspace).resolve()/'code'
    for name in ['common','run','fast_summer']:
        old=sys.modules.get(name)
        if old is not None and Path(old.__file__).resolve().parent!=code:
            # Native modules hold configuration globals. Remove only our prior
            # SUMMER import, after its engine cleanup, before a new workspace.
            if getattr(old,'_S1PERT_RUNTIME_OWNED',False):del sys.modules[name]
            else:raise RuntimeError('Conflicting native module imports; run benchmark in a fresh process.')
    sys.path.insert(0,str(code))
    native=importlib.import_module('fast_summer');common=importlib.import_module('common');prompt=importlib.import_module('run')
    for module in [native,common,prompt]:module._S1PERT_RUNTIME_OWNED=True
    return native,common,prompt

def summer_offline(workspace,config,cell,seed):
    """Generate the real public-only summary cache in a NEW baseline workspace."""
    from baselines.run import init
    workspace=Path(workspace).resolve()
    if workspace.exists():raise FileExistsError('Cold SUMMER cache timing needs a NEW workspace.')
    init('summer',Path(config).resolve(),workspace)
    native,common,prompt=_summer_modules(workspace)
    data=common.Data(cell,int(seed));public=common.functions();engine=native.FastEngine()
    executed=False
    def build():
        nonlocal executed
        if executed:raise ValueError('Cold cache build runs once; repeated warm cache hits are not build timings.')
        executed=True
        summaries=prompt.summary(engine,data.t+data.q,public)
        return dict(status='PASS',genes=len(summaries))
    def cleanup():
        nonlocal engine
        engine=None;gc.collect()
        import torch
        torch.cuda.empty_cache()
    return dict(run=build,synchronize=lambda:None,cleanup=cleanup,device='vllm-worker',
        metadata=dict(scope='Cold public-only gene summary generation; no perturbation outcomes generated.'))

def s1pert_online(model_dir,data_dir,cache_dir,device=None,limit=None,batch_size=None):
    """Restore the selected sealed head and time its actual cached-feature forward."""
    import torch
    from experiments.ablation import load_checkpoint
    model,part,backend,seal=load_checkpoint(model_dir,data_dir,cache_dir,device)
    batch=int(batch_size or backend.settings(part.shared.cfg)['evaluation_batch'])
    n=len(part.records) if limit is None else min(int(limit),len(part.records))
    if n<=0:raise ValueError('Empty inference axis.')
    indices=np.arange(n);dev=str(next(model.parameters()).device)
    def synchronize():
        if dev.startswith('cuda'):torch.cuda.synchronize(dev)
    # Verify saved native outputs before timing when a completed test exists.
    reference=Path(model_dir)/'test_predictions.npz'
    if reference.exists():
        ids=np.arange(min(n,batch))
        observed=backend.predict(model,part,batch,indices=ids)
        with np.load(reference,allow_pickle=False) as saved:
            for key,field in [('target','target'),('query','query'),('labels','label')]:
                if not np.array_equal(saved[key],part.records[field]):raise ValueError('Saved A4 test order drift.')
            np.testing.assert_allclose(observed,saved['probabilities'][ids],atol=2e-5,rtol=0)
    def forward():
        value=backend.predict(model,part,batch,part.shared.guard,indices=indices)
        if value.shape!=(n,3) or not np.isfinite(value).all():raise ValueError('Invalid A4 timing predictions.')
        return {'status':'PASS','pairs':n,'probability_sum':float(value.sum())}
    def cleanup():
        nonlocal model,part
        model=part=None;gc.collect()
        if dev.startswith('cuda'):torch.cuda.empty_cache()
    return dict(run=forward,synchronize=synchronize,cleanup=cleanup,n_pairs=n,device=dev,
        metadata=dict(checkpoint_sha256=seal['best_checkpoint_sha256'],batch_size=batch,
                      includes_cache_gather=True,includes_model_load=False))

def summer_online(workspace,cell,seed,limit=None):
    """Fresh native rationale generation using prebuilt public-only summaries.

    No test-answer cache is read. Model initialization and public-summary loading
    are outside the online region; prompt retrieval/tokenization and native
    generation are inside it. Prefix KV state is reset between timed repetitions.
    """
    if os.environ.get('PYTHONHASHSEED')!='0':
        raise ValueError('Start runtime.py with PYTHONHASHSEED=0 for deterministic native retrieval.')
    workspace=Path(workspace).resolve();code=workspace/'code'
    database=workspace/'prepared/public_summaries.sqlite'
    if not database.is_file():
        raise FileNotFoundError('Build native SUMMER public summaries before loaded-model timing.')
    native,common,prompt_module=_summer_modules(workspace)
    data=common.Data(cell,int(seed));records=data.records('test',final=True)
    n=len(records) if limit is None else min(int(limit),len(records))
    if n<=0:raise ValueError('Empty SUMMER timing axis.')
    public=common.functions();engine=native.FastEngine();summaries={}
    conn=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
    try:
        for gene in sorted(set(data.t+data.q)):
            text=public.get(gene)
            if not text:
                summaries[gene]='No concise public function available; this is not evidence of no response.'
                continue
            key=common.objsha([engine.identity,gene,text,'summary-v1'])
            row=conn.execute('SELECT value FROM items WHERE key=?',(key,)).fetchone()
            if row is None:raise ValueError(f'Public summary identity missing for {gene}; run native preparation first.')
            summaries[gene]=json.loads(row[0])['text']
    finally:conn.close()
    batch=int(common.CFG['generation_batch'])
    def forward():
        if not hasattr(engine.llm,'reset_prefix_cache'):
            raise RuntimeError('vLLM must support reset_prefix_cache for unbiased repeated prompt timing.')
        engine.llm.reset_prefix_cache()
        retriever=common.Retrieval(data)
        digest=hashlib.sha256();valid=0
        for start in range(0,n,batch):
            part=records[start:min(start+batch,n)]
            prompts=[prompt_module.summer_prompt(data,row,summaries,retriever,'test') for row in part]
            answers=engine.generate(prompts,common.CFG['answer_tokens'])
            for answer in answers:
                digest.update(answer['text'].encode())
                valid+=bool(answer['ended'] and common.parse_answer(answer['text'])>=0)
        return dict(status='PASS',pairs=n,valid_labels=valid,text_digest=digest.hexdigest())
    def cleanup():
        nonlocal engine
        engine=None;gc.collect()
        import torch
        torch.cuda.empty_cache()
    # vLLM.generate returns only after result completion, including worker GPU
    # synchronization. A host torch.cuda synchronization cannot see worker streams.
    return dict(run=forward,synchronize=lambda:None,cleanup=cleanup,n_pairs=n,device='vllm-worker',
        metadata=dict(native_backend=engine.backend,public_summary_sha256=common.sha(database),
            includes_retrieval_tokenization=True,reads_old_test_answers=False,
            public_summary_build='outside online measurement; amortize its real build phase for cold end-to-end'))
