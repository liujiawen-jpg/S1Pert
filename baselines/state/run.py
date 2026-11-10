"""Finite, resumable native STATE training and separately restored TEST evaluation."""
from __future__ import annotations
import argparse
import gc
import json
import mmap
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from common import (ROOT,CFG,atomic,sha,identity,data,loss_queries,legal_cell_rows,
                    metrics,select_thresholds,hard_predictions,guard)
from native import build,forward,masked_native_loss

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed);torch.set_num_threads(CFG['threads'])
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(CFG['gpu_memory_budget_gib']*2**30/total)
    torch.backends.cuda.matmul.allow_tf32=True;torch.set_float32_matmul_precision('high')

def save_torch(path,obj):
    temp=Path(path).with_suffix('.tmp');torch.save(obj,temp);os.replace(temp,path)

def checked_code():
    sealed=json.loads((ROOT/'reports/launch_seal.json').read_text())
    if identity()!=sealed['identity']:raise RuntimeError('launch code or pinned vendor identity drift')
    return sealed['identity']

def embeddings(meta):
    source=Path(CFG['esm_source']);info=json.loads((source/'meta.json').read_text())
    index={g:i for i,g in enumerate(info['keys'])};matrix=np.load(source/'embeddings.npy',mmap_mode='r')
    a=np.zeros((len(meta['target_ids']),info['dim']),np.float32)
    good=np.asarray([g in index for g in meta['target_ids']])
    for i,g in enumerate(meta['target_ids']):
        if good[i]:a[i]=matrix[index[g]]
    if not np.isfinite(a).all():raise ValueError('nonfinite public ESM embedding')
    return a,good

class Pool:
    """Only legal TRAIN target cells plus real controls are ever copied to RAM.

    No raw VAL/TEST perturbed cell is needed for model input or validation.
    DONTNEED applies only to this process's read-only source mapping; not to
    global OS caches or another application's arrays.
    """
    def __init__(self,cell,meta,part,eligible=None):
        base=Path(CFG['expression_root'])/cell/'expression'
        report=json.loads((base/'report.json').read_text())
        if report['a4_metadata_sha256']!=sha(Path(CFG['a4_data'])/cell/'dataset.json'):
            raise ValueError('expression metadata differs from sealed A4 axis')
        if sha(base/'cell_target.npy')!=report['cell_target_sha256']:
            raise ValueError('raw cell-to-target identity drift')
        groups=np.load(base/'cell_target.npy',allow_pickle=False)
        source=np.load(base/'log1p_cp10k.npy',mmap_mode='r',allow_pickle=False)
        if source.shape!=tuple(report['shape']) or source.shape[1]!=len(meta['query_ids']):
            raise ValueError('expression shape drift')
        selected=legal_cell_rows(groups,part,[] if eligible is None else eligible)
        # At most the legal population is held, not a huge Cartesian pair cache.
        estimate=len(selected)*source.shape[1]*4
        if estimate>10*2**30:raise MemoryError('legal TRAIN/control bank exceeds declared 10 GiB')
        self.bank=np.empty((len(selected),source.shape[1]),np.float32)
        for lo in range(0,len(selected),256):
            hi=min(lo+256,len(selected));values=np.asarray(source[selected[lo:hi]])
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('invalid normalized cells')
            self.bank[lo:hi]=values
            del values
            if hasattr(source._mmap,'madvise'):source._mmap.madvise(mmap.MADV_DONTNEED)
            if lo%8192==0:guard()
        self.groups=groups[selected]
        self.control=np.flatnonzero(self.groups==-1)
        if not len(self.control):raise ValueError('missing controls')
        self.by_target={int(t):np.flatnonzero(self.groups==t) for t in ([] if eligible is None else eligible)}
        if any(not len(v) for v in self.by_target.values()):raise ValueError('missing legal TRAIN target cells')
        if eligible is not None and not (part[self.groups[self.groups>=0]]==0).all():
            raise ValueError('raw response leakage into training population')
        self.n_genes=source.shape[1]
        self.control_n=len(self.control);del source

    def train_batch(self,targets,rng):
        s=CFG['cell_set_len']
        c=np.stack([self.bank[rng.choice(self.control,s,replace=True)] for _ in targets])
        y=np.stack([self.bank[rng.choice(self.by_target[int(t)],s,replace=True)] for t in targets])
        return torch.from_numpy(c).to('cuda'),torch.from_numpy(y).to('cuda')

    def fixed_controls(self,seed):
        rng=np.random.default_rng(seed+9037)
        indices=rng.choice(self.control,CFG['eval_control_cells'],replace=len(self.control)<CFG['eval_control_cells'])
        return self.bank[indices].copy()

@torch.inference_mode()
def effects(model,rows,control_values,emb):
    model.eval();out=np.empty(len(rows),np.float32)
    targets=np.unique(rows['target']);by_target={int(t):np.flatnonzero(rows['target']==t) for t in targets}
    controls=torch.from_numpy(control_values).to('cuda')
    reference=controls.float().mean(0)
    for lo in range(0,len(targets),CFG['batch_conditions']):
        ts=targets[lo:lo+CFG['batch_conditions']]
        x=controls[None].expand(len(ts),-1,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=forward(model,x,emb[torch.as_tensor(ts,device='cuda',dtype=torch.long)])
        delta=(y.float().mean(1)-reference[None]).cpu().numpy()
        if not np.isfinite(delta).all():raise FloatingPointError('nonfinite native expression prediction')
        for i,t in enumerate(ts):
            at=by_target[int(t)];out[at]=delta[i,rows['query'][at]]
        guard()
    return out

def preflight():
    commit=subprocess.check_output(['git','-C',CFG['official_source'],'rev-parse','HEAD'],text=True).strip()
    if commit!=CFG['official_commit']:raise ValueError('official source revision mismatch')
    dirty=subprocess.check_output(['git','-C',CFG['official_source'],'status','--porcelain'],text=True)
    if dirty.strip():raise ValueError('pinned STATE checkout is modified')
    jobs=[]
    for cell in CFG['cells']:
        for seed in CFG['seeds']:
            meta,part,rr,binding=data(cell,seed)
            base=Path(CFG['expression_root'])/cell/'expression'
            er=json.loads((base/'report.json').read_text())
            if er['a4_metadata_sha256']!=binding['dataset']:raise ValueError('raw preparation binding mismatch')
            emb,good=embeddings(meta)
            q=loss_queries(rr['train'],len(part));eligible=np.unique(rr['train']['target'])
            if not all(len(q[t]) for t in eligible):raise ValueError('loss mask absent')
            if not all(part[t]==0 for t in eligible):raise ValueError('loss target leakage')
            jobs.append({'cell':cell,'seed':seed,'train_pairs':len(rr['train']),'val_pairs':len(rr['val']),
                'train_conditions':len(eligible),'data_binding':binding,
                'test_labels_opened':False,'esm_dim':emb.shape[1],
                'esm_targets_available':int(good.sum()),'esm_targets_missing':int((~good).sum()),
                'missing_target_symbols':[s for s,g in zip(meta['target_symbols'],good) if not g],
                'expression_preparation_sha256':sha(base/'report.json'),
                'normalized_source_stat':{'bytes':(base/'log1p_cp10k.npy').stat().st_size,
                                          'mtime_ns':(base/'log1p_cp10k.npy').stat().st_mtime_ns}})
    atomic(ROOT/'reports/preflight.json',{'status':'PASS','official_commit':commit,'jobs':jobs,
        'task_weights_from_scratch':True,'old_prediction_anchor':False,
        'test_labels_used_for_training':False,'response_loss_mask':'strict TRAIN pairs only',
        'native_energy_loss':True,'all_measured_query_genes_retained':True,
        'esm_source_meta_sha256':sha(Path(CFG['esm_source'])/'meta.json'),
        'esm_source_embeddings_sha256':sha(Path(CFG['esm_source'])/'embeddings.npy')})

def job_binding(cell,seed,current):
    pf=json.loads((ROOT/'reports/preflight.json').read_text())
    row=next(r for r in pf['jobs'] if r['cell']==cell and r['seed']==seed)
    if any(current[k]!=v for k,v in row['data_binding'].items()):raise ValueError('preflight data identity drift')
    base=Path(CFG['expression_root'])/cell/'expression';stat=(base/'log1p_cp10k.npy').stat()
    if {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=row['normalized_source_stat']:
        raise ValueError('normalized source changed since preflight')
    if sha(base/'report.json')!=row['expression_preparation_sha256']:raise ValueError('source report changed')
    if sha(Path(CFG['esm_source'])/'embeddings.npy')!=pf['esm_source_embeddings_sha256']:
        raise ValueError('ESM array drift')
    if sha(Path(CFG['esm_source'])/'meta.json')!=pf['esm_source_meta_sha256']:
        raise ValueError('ESM mapping drift')

def train(cell,seed):
    code=checked_code();seed_all(seed);start=time.time()
    dest=ROOT/'runs'/cell/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    meta,part,rr,binding=data(cell,seed);job_binding(cell,seed,binding)
    if (dest/'convergence.json').exists():
        c=json.loads((dest/'convergence.json').read_text())
        if c.get('status')=='PASS' and c['identity']==code and c['data_binding']==binding:return
        raise RuntimeError('existing unaccepted convergence seal; inspect before retry')
    q=loss_queries(rr['train'],len(part));eligible=np.unique(rr['train']['target'])
    atomic(dest/'progress.json',{'stage':'loading_legal_TRAIN_cells','status':'RUNNING','pid':os.getpid()})
    pool=Pool(cell,meta,part,eligible);cv=pool.fixed_controls(seed)
    emb_np,good=embeddings(meta);emb=torch.from_numpy(emb_np).to('cuda')
    model=build(len(meta['query_ids']),emb.shape[1]).to('cuda')
    optimizer=model.configure_optimizers()
    if not isinstance(optimizer,torch.optim.Optimizer):raise ValueError('unexpected upstream optimizer contract')
    rng=np.random.default_rng(seed);history=[];best=-1.;bad=0;best_epoch=0
    first_epoch=1;offset=0;order=None;loss_sum=0.;loss_n=0;updates=0
    last=dest/'last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['identity']!=code or ck['data_binding']!=binding:raise ValueError('own resume identity mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        for st in optimizer.state.values():
            for k,v in st.items():
                if torch.is_tensor(v) and k!='step':st[k]=v.to('cuda')
        history=ck['history'];best=ck['best'];bad=ck['bad'];best_epoch=ck['best_epoch']
        first_epoch=ck['epoch'];offset=ck['next_offset'];order=ck['order'];updates=ck['updates']
        loss_sum=ck['loss_sum'];loss_n=ck['loss_n']
        rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state(ck['cuda_rng']);del ck
    atomic(dest/'data_audit.json',{'data_binding':binding,'fresh_task_weights':not last.exists(),
        'resume_only_own_matching_checkpoint':last.exists(),'loss_pairs':len(rr['train']),
        'train_conditions':len(eligible),'raw_cell_rows_loaded':len(pool.groups),
        'raw_noncontrol_targets_all_TRAIN':bool((part[pool.groups[pool.groups>=0]]==0).all()),
        'input_control_cells':pool.control_n,'input_control_role':'real measured unperturbed biological state',
        'old_predictor_or_source_response_mean_anchor':False,
        'prediction_flow':'native STATE expression -> mean over real-control input set -> subtract same real-control mean -> VAL-only thresholds',
        'native_internal_residual':'upstream hidden representation skip only, no old prediction answer',
        'query_genes':len(meta['query_ids']),'missing_esm_targets':int((~good).sum())})
    def checkpoint(epoch,next_offset,order,loss_sum,loss_n):
        save_torch(last,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'identity':code,
            'data_binding':binding,'epoch':epoch,'next_offset':next_offset,'order':order,
            'history':history,'best':best,'bad':bad,'best_epoch':best_epoch,'updates':updates,
            'loss_sum':loss_sum,'loss_n':loss_n,'numpy_rng':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
    converged=False
    for epoch in range(first_epoch,CFG['max_epochs']+1):
        e_start=time.time();model.train()
        if order is None:order=rng.permutation(eligible);offset=0;loss_sum=0.;loss_n=0
        for lo in range(offset,len(order),CFG['batch_conditions']):
            ts=order[lo:lo+CFG['batch_conditions']]
            control,response=pool.train_batch(ts,rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=forward(model,control,emb[torch.as_tensor(ts,device='cuda',dtype=torch.long)])
            loss=masked_native_loss(model,pred,response,[q[int(t)] for t in ts])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();updates+=1;loss_sum+=float(loss.detach())*len(ts);loss_n+=len(ts)
            del pred,loss,response,control
            if updates==1 or updates%25==0:
                checkpoint(epoch,lo+len(ts),order,loss_sum,loss_n);guard()
                atomic(dest/'progress.json',{'stage':'train','status':'RUNNING','epoch':epoch,'updates':updates,
                    'conditions_completed':lo+len(ts),'conditions_per_epoch':len(order),
                    'best_epoch':best_epoch,'best_val_macro_f1':best,'resources':guard()})
        val_effect=effects(model,rr['val'],cv,emb);adapter=select_thresholds(val_effect,rr['val']['label'])
        vm=metrics(rr['val']['label'],hard_predictions(val_effect,adapter['down'],adapter['up']),rr['val']['target'])
        score=vm['macro_f1']
        if score>best+CFG['min_delta']:
            best=score;bad=0;best_epoch=epoch
            save_torch(dest/'best.pt',{'model':model.state_dict(),'identity':code,'data_binding':binding,
                'adapter':adapter,'epoch':epoch,'val':vm,'cell':cell,'seed':seed})
            np.save(dest/'best_val_effect.npy',val_effect)
        else:bad+=1
        rec={'epoch':epoch,'updates':updates,'train_energy_loss':loss_sum/max(1,loss_n),
             'val_macro_f1':score,'best_val_macro_f1':best,'best_epoch':best_epoch,
             'patience_used':bad,'epoch_seconds':time.time()-e_start,'adapter':adapter,'resources':guard()}
        history.append(rec);atomic(dest/'history.json',history);print(json.dumps({'cell':cell,'seed':seed,**rec}),flush=True)
        order=None;offset=0;loss_sum=0.;loss_n=0
        checkpoint(epoch+1,0,None,0.,0)
        if epoch>=CFG['min_epochs'] and bad>=CFG['patience']:converged=True;break
    atomic(dest/'convergence.json',{'status':'PASS' if converged else 'NOT_CONVERGED',
        'identity':code,'data_binding':binding,'epochs':len(history),'best_epoch':best_epoch,
        'best_val_macro_f1':best,'train_seconds':time.time()-start,'updates':updates,
        'stop_reason':'VAL patience' if converged else 'max_epochs; TEST remains locked',
        'checkpoint_sha256':sha(dest/'best.pt'),'selected_without_TEST':True})
    if not converged:raise RuntimeError('STATE not converged at epoch limit; no TEST evaluation')

def evaluate(cell,seed):
    code=checked_code();seed_all(seed);start=time.time()
    dest=ROOT/'runs'/cell/f'seed_{seed}'
    c=json.loads((dest/'convergence.json').read_text())
    if c['status']!='PASS' or c['identity']!=code:raise ValueError('convergence gate failed')
    meta,part,rr,binding=data(cell,seed);job_binding(cell,seed,binding)
    if binding!=c['data_binding'] or sha(dest/'best.pt')!=c['checkpoint_sha256']:raise ValueError('checkpoint binding drift')
    if (dest/'report.json').exists():
        report=json.loads((dest/'report.json').read_text())
        if report['status']=='PASS' and report['identity']==code and report['prediction_sha256']==sha(dest/'test_predictions.npz'):return
        raise RuntimeError('existing evaluation seal does not match')
    pool=Pool(cell,meta,part,None);controls=pool.fixed_controls(seed)
    en,available=embeddings(meta);emb=torch.from_numpy(en).to('cuda')
    model=build(len(meta['query_ids']),emb.shape[1]).to('cuda')
    ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']);del ck['model']
    current=effects(model,rr['val'],controls,emb);expected=np.load(dest/'best_val_effect.npy')
    if not np.allclose(current,expected,rtol=1e-4,atol=2e-4):raise ValueError('independent process restore validation mismatch')
    adapter=ck['adapter'];pred=hard_predictions(current,adapter['down'],adapter['up'])
    ref=hard_predictions(expected,adapter['down'],adapter['up'])
    if not np.array_equal(pred,ref):raise ValueError('restored VAL decisions changed')
    atomic(dest/'restore_report.json',{'status':'PASS','independent_pid':os.getpid(),
        'max_val_effect_absolute_difference':float(np.max(np.abs(current-expected))),
        'same_validation_classes':True,'checkpoint_sha256':c['checkpoint_sha256'],'test_opened':False})
    # Only this new restored process opens sealed TEST labels, after the gate.
    _,_,all_rows,full_binding=data(cell,seed,test=True);test=all_rows['test']
    values=effects(model,test,controls,emb);hard=hard_predictions(values,adapter['down'],adapter['up'])
    result=metrics(test['label'],hard,test['target'])
    dest_npz=dest/'test_predictions.npz'
    np.savez_compressed(dest_npz,target=test['target'],targets=test['target'],query=test['query'],labels=test['label'],
                        y_pred=hard,predicted=hard,effect=values)
    for cls in ['up','same','down']:
        # Label metrics only; continuous scores are not made into fake probabilities.
        result['per_class'][cls]['probability_metrics_available']=False
    good=available[test['target']]
    groups={name:metrics(test['label'][mask],hard[mask],test['target'][mask])
            for name,mask in [('esm_available',good),('esm_missing',~good)] if mask.any()}
    report={'status':'PASS','method':CFG['method'],'cell':cell,'seed':seed,'identity':code,
        'converged':True,'epochs':c['epochs'],'best_epoch':c['best_epoch'],'adapter':adapter,
        'test':result,'subgroup_metrics':groups,'train_seconds':c['train_seconds'],
        'evaluation_seconds':time.time()-start,'independent_restore':'PASS',
        'checkpoint_sha256':c['checkpoint_sha256'],'prediction_sha256':sha(dest_npz),
        'data_binding':full_binding,'no_test_subsampling':True,'no_TEST_threshold_tuning':True}
    atomic(dest/'report.json',report);print(json.dumps({'completed':cell,'seed':seed,'test':result}),flush=True)

def summarize():
    runs=[]
    for cell in CFG['cells']:
        for seed in CFG['seeds']:
            p=ROOT/'runs'/cell/f'seed_{seed}'/'report.json'
            if p.exists():
                r=json.loads(p.read_text())
                if r['status']=='PASS':runs.append(r)
    datasets={}
    for cell in CFG['cells']:
        rr=[r for r in runs if r['cell']==cell];mm={}
        for name in ['macro_f1','balanced_accuracy','up_precision','up_recall','up_f1',
                     'same_precision','same_recall','same_f1','down_precision','down_recall','down_f1']:
            values=[r['test'][name] if name in r['test'] else r['test']['per_class'][name.split('_')[0]][name.split('_',1)[1]] for r in rr]
            mm[name]={'mean':float(np.mean(values)) if values else None,
                      'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None,
                      'sample_variance':float(np.var(values,ddof=1)) if len(values)>1 else None,'values':values}
        datasets[cell]={'completed':len(rr),'seeds':[r['seed'] for r in rr],'metrics':mm}
    atomic(ROOT/'reports/summary.json',{'status':'PASS' if len(runs)==20 else 'PARTIAL',
        'completed':len(runs),'total':20,'datasets':datasets,
        'definition':'five repeated target holdouts, NOT disjoint five-fold CV; sample ddof=1',
        'scientific_gain_claimed':False})

def gpu_smoke():
    checked_code();seed_all(17);start=time.time()
    meta=json.loads((Path(CFG['a4_data'])/'hepg2/dataset.json').read_text())
    genes=len(meta['query_ids']);batch=CFG['batch_conditions']
    model=build(genes,1280).to('cuda');model.train()
    controls=torch.rand(batch,CFG['cell_set_len'],genes,device='cuda')
    pert=torch.rand(batch,1280,device='cuda');truth=torch.rand_like(controls)
    optimizer=model.configure_optimizers();before=model.project_out[0].weight.detach().clone()
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=forward(model,controls,pert)
    q=[np.arange(genes) for _ in range(batch)]
    loss=masked_native_loss(model,pred,truth,q);loss.backward();optimizer.step()
    changed=not torch.equal(before,model.project_out[0].weight.detach())
    if not changed:raise RuntimeError('native parameter update absent')
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):ref=forward(model,controls,pert).float().cpu().numpy()
    p=ROOT/'reports/gpu_smoke_weights.pt';save_torch(p,model.state_dict())
    new=build(genes,1280).to('cuda').eval();new.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):restored=forward(new,controls,pert).float().cpu().numpy()
    if not np.array_equal(ref,restored):raise RuntimeError('GPU smoke exact restoration failed')
    atomic(ROOT/'reports/gpu_smoke.json',{'status':'PASS','synthetic_fixture_only':True,
        'native_state_class':type(model).__name__,'full_backbone_layers':8,'hidden_width':768,
        'full_gene_axis':genes,'batch_conditions':batch,
        'cell_set_len':CFG['cell_set_len'],'energy_loss':float(loss.detach()),
        'parameter_updated':changed,'restore_identical':True,'seconds':time.time()-start,
        'gpu_name':torch.cuda.get_device_name(0),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
        'resources':guard(),'does_not_prove_prediction_quality':True})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','train','evaluate','smoke','summary'])
    ap.add_argument('--cell');ap.add_argument('--seed',type=int);args=ap.parse_args()
    try:
        if args.action=='preflight':preflight()
        elif args.action=='train':train(args.cell,args.seed)
        elif args.action=='evaluate':evaluate(args.cell,args.seed)
        elif args.action=='smoke':gpu_smoke()
        else:summarize()
    except BaseException as exc:
        atomic(ROOT/'reports'/f'failure_{args.action}_{args.cell}_{args.seed}.json',
            {'status':'PARTIAL','error':repr(exc),'traceback':traceback.format_exc(),'pid':os.getpid()})
        raise

if __name__=='__main__':main()
