"""Full-epoch baseline training, VAL selection, then independent-process TEST."""
from __future__ import annotations
import argparse, gc, json, math, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import (ROOT,CFG,data,write_json,digest_file,seed_all,probability_metrics,
                    hard_metrics,hard_predictions,select_thresholds,check_resources)
from models import (GraphClassifier,GenePTClassifier,genept_embeddings,gears_model,
                    gears_loss,scgpt_model,native_forward)

DEVICE='cuda'

def record_tensors(rows):
    return [torch.tensor(np.asarray(rows[k]).astype(np.int64),device=DEVICE) for k in ['target','query','label']]

def build(method,meta,rows,cell,seed):
    if method=='gat': return GraphClassifier(meta).to(DEVICE),{}
    if method.startswith('genept_'):
        target,query,audit=genept_embeddings(meta,rows,method.split('_')[1])
        return GenePTClassifier(target,query).to(DEVICE),audit
    if method=='gears': return gears_model(meta,cell,seed,DEVICE).to(DEVICE),{}
    model,tokens,positions,audit=scgpt_model(meta,DEVICE)
    return model,{'tokens':tokens,'positions':positions,'audit':audit}

def class_probabilities(model,rows):
    model.eval()
    with torch.no_grad():
        t,q,y=record_tensors(rows)
        result=model(t,q).float().softmax(-1).cpu().numpy()
    return result

def native_context(meta,cell,seed,training=False):
    base=ROOT/'prepared'/cell/'expression'
    x=np.load(base/'log1p_cp10k.npy',mmap_mode='r')
    groups=np.load(base/'cell_target.npy')
    _,part,_,_=data(cell,seed)
    ctrl_rows=np.flatnonzero(groups==-1)
    rng=np.random.default_rng(seed+7021)
    eval_ctrl_rows=rng.choice(ctrl_rows,CFG['native_evaluation_control_cells'],replace=False)
    context={'x':x,'groups':groups,'control':torch.tensor(np.asarray(x[ctrl_rows]),device=DEVICE),
             'eval_control':torch.tensor(np.asarray(x[eval_ctrl_rows]),device=DEVICE),
             'eval_control_rows':eval_ctrl_rows.tolist(),'control_mean':torch.tensor(np.load(base/'group_log_means.npy',mmap_mode='r')[0].copy(),device=DEVICE)}
    if training:
        train_rows=np.flatnonzero((groups>=0)&(part[np.maximum(groups,0)]==0))
        # Only TRAIN responses are moved onto GPU. Held-out cells are not model inputs.
        tensor=torch.empty((len(train_rows),x.shape[1]),device=DEVICE,dtype=torch.float32)
        for lo in range(0,len(train_rows),1024):
            tensor[lo:lo+1024]=torch.tensor(np.asarray(x[train_rows[lo:lo+1024]]),device=DEVICE)
        context.update(train=tensor,targets=torch.tensor(groups[train_rows],device=DEVICE,dtype=torch.long),
                       train_rows=len(train_rows))
        filters=json.loads((ROOT/'prepared'/cell/f'seed_{seed}'/'training_nonzero_genes.json').read_text())
        context['filters']={k:torch.tensor(v,device=DEVICE,dtype=torch.long) for k,v in filters.items()}
    return context

def chunks_with_targets(g,positions,limit):
    """Every query is covered; every chunk also contains the perturbed gene token."""
    required=np.unique(np.asarray(positions,dtype=np.int64))
    capacity=limit-len(required)
    if capacity<=0: raise ValueError('too many required perturbation tokens for sequence length')
    for lo in range(0,g,capacity):
        queries=np.arange(lo,min(lo+capacity,g),dtype=np.int64)
        yield np.unique(np.r_[required,queries]),queries

def native_effects(model,method,meta,rows,context,extra):
    model.eval();g=len(meta['query_ids'])
    requested=np.unique(rows['target']);preds={}
    ctrl=context['eval_control'];nctrl=len(ctrl)
    # Bounded target groups permit efficient inference without omitting intervention tokens.
    per_batch=4 if method=='scgpt' else max(1,CFG['gears']['batch']//nctrl)
    with torch.no_grad():
        for lo in range(0,len(requested),per_batch):
            targets=requested[lo:lo+per_batch]
            t=torch.tensor(np.repeat(targets,nctrl),device=DEVICE,dtype=torch.long)
            controls=ctrl.repeat(len(targets),1)
            if method=='gears':
                pred=native_forward(model,method,controls,t)
                effects=(pred-controls).reshape(len(targets),nctrl,g).mean(1).cpu().numpy()
            else:
                effects=np.empty((len(targets),g),dtype=np.float32)
                positions=extra['positions'][targets]
                for inds,queries in chunks_with_targets(g,positions,CFG['scgpt']['max_seq_len']):
                    idx=torch.tensor(inds,device=DEVICE,dtype=torch.long)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        prediction=native_forward(model,method,controls,t,idx,extra['tokens'],extra['positions'])
                    offsets=np.searchsorted(inds,queries)
                    values=prediction[:,offsets].float()-controls[:,queries]
                    effects[:,queries]=values.reshape(len(targets),nctrl,len(queries)).mean(1).cpu().numpy()
            if not np.isfinite(effects).all(): raise FloatingPointError('non-finite native population prediction')
            for i,target in enumerate(targets): preds[int(target)]=effects[i]
    return np.asarray([preds[int(t)][int(q)] for t,q in zip(rows['target'],rows['query'])],dtype=np.float32)

def checkpoint(out,model,meta_info):
    path=out/'best.pt';tmp=out/'best.pt.tmp'
    torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'info':meta_info},tmp)
    os.replace(tmp,path)

def validation(model,method,meta,rows,context,extra):
    if method in ['gears','scgpt']:
        effect=native_effects(model,method,meta,rows,context,extra)
        thresholds=select_thresholds(effect,rows['label'])
        predicted=hard_predictions(effect,thresholds['down'],thresholds['up'])
        metrics=hard_metrics(rows['label'],predicted,rows['target'])
        return metrics,{'thresholds':thresholds},effect
    probs=class_probabilities(model,rows)
    return probability_metrics(rows,probs),{},probs

def fit_genept(model,method,meta,records,out,cell,seed,extra):
    tensors=record_tensors(records['train']);history=[];best=-1.;selected=None
    for c in CFG['genept']['C']:
        for p in model.parameters(): p.data.zero_()
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=CFG['genept']['max_iter'],
            tolerance_grad=CFG['genept']['tolerance'],tolerance_change=1e-10,line_search_fn='strong_wolfe',history_size=30)
        calls=0
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            logits=model(tensors[0],tensors[1])
            loss=nn.functional.cross_entropy(logits,tensors[2])
            loss=loss+(model.target_weight.square().sum()+model.query_weight.square().sum())/(2*c*len(records['train']))
            if not torch.isfinite(loss):raise FloatingPointError('non-finite logistic loss')
            loss.backward();calls+=1
            return loss
        started=time.monotonic(); optimizer.step(closure)
        state=optimizer.state[model.target_weight]
        grad=max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None)
        iterations=int(state.get('n_iter',0));converged=iterations<CFG['genept']['max_iter'] or grad<=CFG['genept']['tolerance']
        metrics,conversion,pred=validation(model,method,meta,records['val'],None,extra)
        entry={'C':c,'optimizer_iterations':iterations,'closure_calls':calls,'gradient_max':grad,
               'converged':bool(converged),'val_macro_f1':metrics['macro_f1'],'seconds':time.monotonic()-started}
        history.append(entry);write_json(out/'history.json',history);print(json.dumps(entry),flush=True)
        if converged and metrics['macro_f1']>best:
            best=metrics['macro_f1'];selected={**entry,'method':method,'cell':cell,'seed':seed,
                'selection':'highest VAL Macro-F1 among converged C candidates; earlier candidate breaks ties',
                'extra_audit':extra,'converged':True}
            checkpoint(out,model,selected);np.save(out/'val_reference.npy',pred)
            write_json(out/'selection.json',selected)
        del optimizer;gc.collect();check_resources()
    if selected is None:raise RuntimeError('NO_CONVERGED_LOGISTIC_CANDIDATE')
    return selected

def fit_epochs(model,method,meta,records,out,cell,seed,extra):
    context=native_context(meta,cell,seed,True) if method in ['gears','scgpt'] else None
    if context:
        write_json(out/'native_input_audit.json',{'train_cells':context['train_rows'],'evaluation_control_rows':context['eval_control_rows'],
            'continuous_supervision':True,'training_response_scope':'TRAIN targets only',
            'graph_scope':'public GO plus TRAIN/control coexpression',
            'prediction_control_reference':'unperturbed observations, not an old predictor',
            'scgpt_audit':extra.get('audit')})
    p=CFG[method]
    optimizer=torch.optim.Adam(model.parameters(),lr=p['lr'],weight_decay=5e-4 if method=='gears' else p.get('weight_decay',0))
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.5 if method=='gears' else .9) if context else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3,min_lr=1e-6)
    tensors=record_tensors(records['train']) if context is None else None
    count=context['train_rows'] if context else len(records['train'])
    loss_fn=gears_loss() if method=='gears' else None
    best=-1.;best_epoch=0;history=[];selected=None;converged=False
    for epoch in range(1,CFG['max_epochs']+1):
        started=time.monotonic();model.train();order=torch.randperm(count,device=DEVICE)
        total=0.;updates=0
        for lo in range(0,count,p['batch']):
            inds=order[lo:lo+p['batch']]
            if method=='gears' and len(inds)<2:
                # BatchNorm requires >=2; merge final singleton into previous-sized draw.
                inds=torch.cat([inds,order[:1]])
            optimizer.zero_grad(set_to_none=True)
            if context is None:
                logits=model(tensors[0][inds],tensors[1][inds]);loss=nn.functional.cross_entropy(logits,tensors[2][inds])
            else:
                targets=context['targets'][inds]
                ctrl=context['control'][torch.randint(len(context['control']),(len(inds),),device=DEVICE)]
                truth=context['train'][inds]
                if method=='gears':
                    prediction=native_forward(model,method,ctrl,targets)
                    perts=[str(int(t)) for t in targets.cpu().numpy()]
                    loss=loss_fn(prediction.float(),truth,perts,ctrl=context['control_mean'],
                        direction_lambda=CFG['gears']['direction_lambda'],dict_filter=context['filters'])
                else:
                    positions=extra['positions'][targets.cpu().numpy()]
                    required=torch.tensor(np.unique(positions),device=DEVICE,dtype=torch.long)
                    candidates=torch.randperm(len(meta['query_ids']),device=DEVICE)
                    candidates=candidates[~torch.isin(candidates,required)]
                    q=torch.cat([required,candidates[:CFG['scgpt']['max_seq_len']-len(required)]])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        prediction=native_forward(model,method,ctrl,targets,q,extra['tokens'],extra['positions'])
                        measured=q<len(meta['query_ids'])
                        loss=nn.functional.mse_loss(prediction[:,measured].float(),truth[:,q[measured]])
            if not torch.isfinite(loss):raise FloatingPointError('non-finite full training loss')
            loss.backward()
            if method=='gears':nn.utils.clip_grad_value_(model.parameters(),1.)
            else:nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();total+=float(loss.detach())*len(inds);updates+=1
            if updates%100==0:
                print(json.dumps({'epoch':epoch,'updates':updates,'of':math.ceil(count/p['batch']),'loss':float(loss.detach())}),flush=True)
                check_resources()
        metrics,conversion,reference=validation(model,method,meta,records['val'],context,extra)
        score=metrics['macro_f1']
        improved=score>best+CFG['min_delta']
        if improved:
            best=score;best_epoch=epoch
            selected={'method':method,'cell':cell,'seed':seed,'epoch':epoch,'val_macro_f1':score,
                'selection':'VAL Macro-F1; earliest checkpoint retained for ties within min_delta',
                'training_units':'complete TRAIN epochs','conversion':conversion}
            checkpoint(out,model,selected);np.save(out/'val_reference.npy',reference)
            write_json(out/'selection.json',selected)
        entry={'epoch':epoch,'training_loss':total/count,'val_macro_f1':score,
            'val_metrics':metrics,'updates':updates,'train_units':count,'seconds':time.monotonic()-started,
            'lr':optimizer.param_groups[0]['lr'],'resources':check_resources()}
        history.append(entry);write_json(out/'history.json',history)
        print(json.dumps({k:v for k,v in entry.items() if k!='val_metrics'}),flush=True)
        if context:scheduler.step()
        else:scheduler.step(score)
        if epoch>=CFG['min_epochs'] and epoch-best_epoch>=CFG['patience']:
            converged=True;break
    selected['converged']=converged;selected['last_epoch']=epoch
    write_json(out/'selection.json',selected)
    if not converged:raise RuntimeError('MAX_EPOCHS_WITHOUT_CONVERGENCE: TEST remains locked')
    return selected

def run_train(method,cell,seed,out):
    out.mkdir(parents=True,exist_ok=True);seed_all(seed)
    meta,part,records,ident=data(cell,seed)
    write_json(out/'data_identity.json',ident);write_json(out/'config.json',CFG)
    model,extra=build(method,meta,records['train'],cell,seed)
    selected=fit_genept(model,method,meta,records,out,cell,seed,extra) if method.startswith('genept_') else fit_epochs(model,method,meta,records,out,cell,seed,extra)
    del model;gc.collect();torch.cuda.empty_cache()
    command=[sys.executable,str(ROOT/'train.py'),'restore','--method',method,'--cell',cell,'--seed',str(seed),'--out',str(out)]
    completed=subprocess.run(command,check=False)
    if completed.returncode:raise RuntimeError(f'independent restore/evaluation failed with exit {completed.returncode}')

def restore(method,cell,seed,out):
    seed_all(seed);selected=json.loads((out/'selection.json').read_text())
    if not selected['converged']:raise ValueError('TEST locked: not converged')
    meta,part,records,ident=data(cell,seed)
    if ident!=json.loads((out/'data_identity.json').read_text()):raise ValueError('sealed data identity drift')
    model,extra=build(method,meta,records['train'],cell,seed)
    saved=torch.load(out/'best.pt',map_location=DEVICE,weights_only=False)
    model.load_state_dict(saved['state'],strict=True)
    context=native_context(meta,cell,seed) if method in ['gears','scgpt'] else None
    _,_,reference=validation(model,method,meta,records['val'],context,extra)
    expected=np.load(out/'val_reference.npy')
    drift=float(np.max(np.abs(reference-expected)))
    if not np.allclose(reference,expected,rtol=1e-4,atol=2e-4):raise AssertionError(f'restore drift {drift}')
    write_json(out/'restore_report.json',{'status':'PASS','independent_process':True,'max_absolute_drift':drift})
    _,_,test_data,_=data(cell,seed,include_test=True);rows=test_data['test']
    if context:
        effect=native_effects(model,method,meta,rows,context,extra)
        threshold=selected['conversion']['thresholds']
        predicted=hard_predictions(effect,threshold['down'],threshold['up'])
        metrics=hard_metrics(rows['label'],predicted,rows['target'],effect)
        np.save(out/'test_effect.npy',effect);np.save(out/'test_predictions.npy',predicted)
        write_json(out/'probabilities_unavailable.json',{'reason':'native expression model plus deterministic threshold adapter; no fabricated probabilities'})
    else:
        probs=class_probabilities(model,rows);metrics=probability_metrics(rows,probs)
        np.save(out/'test_probabilities.npy',probs);np.save(out/'test_predictions.npy',probs.argmax(1))
    np.save(out/'test_keys.npy',np.column_stack([rows['target'],rows['query']]))
    write_json(out/'test_metrics.json',metrics)
    artifacts={p.name:digest_file(p) for p in out.iterdir() if p.is_file() and p.name not in ['report.json','artifact_sha256.json']}
    write_json(out/'artifact_sha256.json',artifacts)
    write_json(out/'report.json',{'status':'PASS','method':method,'cell':cell,'seed':seed,'selected':selected,
        'test_records':len(rows),'test_macro_f1':metrics['macro_f1'],'checkpoint_sha256':digest_file(out/'best.pt'),
        'reproduction_claim':'independent A4-protocol adaptation, not exact AROMA baseline reproduction'})

def smoke(method,cell,seed,out):
    """Real model forward/backward and recovery, on TRAIN-only examples."""
    seed_all(seed);meta,_,records,_=data(cell,seed);model,extra=build(method,meta,records['train'],cell,seed)
    model.train()
    if method in ['gat','genept_gene','genept_prot']:
        rows=records['train'][:64];t,q,y=record_tensors(rows)
        logits=model(t,q);loss=nn.functional.cross_entropy(logits,y)
    else:
        ctx=native_context(meta,cell,seed)
        rows=records['train'][:2];t=torch.tensor(rows['target'].astype(np.int64),device=DEVICE)
        control=ctx['eval_control'][:2]
        if method=='gears':prediction=native_forward(model,method,control,t)
        else:
            positions=extra['positions'][rows['target']]
            inds=np.unique(np.r_[positions,np.arange(64)])
            prediction=native_forward(model,method,control,t,torch.tensor(inds,device=DEVICE),extra['tokens'],extra['positions'])
        loss=prediction.square().mean()
    if not torch.isfinite(loss):raise FloatingPointError('smoke nonfinite loss')
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):raise AssertionError('invalid gradients')
    write_json(out,{'status':'PASS','method':method,'loss':float(loss.detach()),'gradient_tensors':len(grads),'test_read':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','restore','smoke']);p.add_argument('--method',required=True)
    p.add_argument('--cell',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        {'train':run_train,'restore':restore,'smoke':smoke}[a.action](a.method,a.cell,a.seed,a.out)
    except Exception as exc:
        traceback.print_exc()
        if a.action=='train':write_json(a.out/'report.json',{'status':'PARTIAL','method':a.method,'cell':a.cell,'seed':a.seed,'error':repr(exc),'test_success':False})
        raise
