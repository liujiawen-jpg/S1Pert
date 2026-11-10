"""Immutable A4 records, explicit classification adapter, bounded resources."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())

def atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(temp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20), b''): h.update(b)
    return h.hexdigest()

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def identity():
    paths = list(ROOT.glob('*.py'))+[ROOT/'config.json']
    vendor = Path(CFG['official_source'])
    paths += [vendor/p for p in ['src/state/tx/models/state_transition.py',
              'src/state/tx/models/utils.py','src/state/tx/models/base.py',
              'src/state/configs/model/state.yaml']]
    paths += [Path(CFG['a4_metrics'])]
    return {str(p):sha(p) for p in sorted(paths)}

def data(cell, seed, test=False):
    base = Path(CFG['a4_data'])/cell
    meta = json.loads((base/'dataset.json').read_text())
    part_path = base/f'seed_{seed}'/'target_partition.npy'
    part = np.load(part_path, allow_pickle=False)
    if part.shape != (len(meta['target_ids']),) or not set(np.unique(part)) <= {0,1,2}:
        raise ValueError('invalid sealed target partition')
    records={}; binding={'dataset':sha(base/'dataset.json'),'partition':sha(part_path)}
    for i,name in enumerate(['train','val','test'] if test else ['train','val']):
        p=base/f'seed_{seed}'/name/'records.npy';r=np.load(p,mmap_mode='r',allow_pickle=False)
        if r.dtype.names != ('target','query','label'):raise ValueError('record schema changed')
        if not np.isin(r['label'],[0,1,2]).all():raise ValueError('label changed')
        if (r['target']<0).any() or (r['target']>=len(part)).any():raise ValueError('target bound')
        if (r['query']<0).any() or (r['query']>=len(meta['query_ids'])).any():raise ValueError('query bound')
        if not (part[r['target']]==i).all():raise ValueError('target split leakage')
        ids=np.asarray(meta['target_ids']);qids=np.asarray(meta['query_ids'])
        if (ids[r['target']]==qids[r['query']]).any():raise ValueError('self pair')
        keys=r['target'].astype(np.int64)*len(qids)+r['query']
        if len(np.unique(keys))!=len(keys):raise ValueError('duplicate pair')
        records[name]=r;binding[name]=sha(p)
    return meta,part,records,binding

def loss_queries(records, n_targets):
    # Exactly the strict TRAIN pairs; no uncertain or self pair is added to loss.
    return [np.sort(records['query'][records['target']==i]).astype(np.int64) for i in range(n_targets)]

def legal_cell_rows(groups, part, eligible):
    valid=groups>=0
    keep=groups==-1
    train=np.zeros(len(part),bool);train[np.asarray(eligible,dtype=int)]=True
    train &= part==0
    keep[valid]=train[groups[valid]]
    return np.flatnonzero(keep)

def hard_predictions(effect, down, up):
    effect=np.asarray(effect)
    if not np.isfinite(effect).all() or not down<=0<=up:raise ValueError('invalid adapter')
    return np.where(effect>up,0,np.where(effect<down,2,1)).astype(np.int8)

def metrics(y, pred, targets=None):
    m=load_module(CFG['a4_metrics'],'sealed_a4_metrics')
    y=np.asarray(y);pred=np.asarray(pred)
    cm=np.bincount(y*3+pred,minlength=9).reshape(3,3)
    out=m.metrics_from_confusion(cm)
    changed=y!=1;called=pred!=1;tp=int((changed&called).sum())
    p=tp/max(1,int(called.sum()));r=tp/max(1,int(changed.sum()))
    out.update(records=len(y),class_order=['up','same','down'],
        changed={'precision':p,'recall':r,'f1':2*p*r/max(1e-30,p+r)},
        probability_status='Native expression scores are NOT three-class probabilities',
        nll=None,brier_multiclass_sum=None)
    if targets is not None:
        ts,inv,n=np.unique(targets,return_inverse=True,return_counts=True)
        weighted=np.bincount(y*3+pred,weights=1/n[inv],minlength=9).reshape(3,3)
        out['targets']=len(ts);out['equal_target_weighted_confusion_metrics']=m.metrics_from_confusion(weighted)
    return out

def select_thresholds(effect, labels):
    # Same predeclared asymmetric VAL-only grid as the GEARS/scGPT adapters.
    effect=np.asarray(effect,dtype=np.float64);labels=np.asarray(labels)
    if not np.isfinite(effect).all():raise FloatingPointError('nonfinite VAL effect')
    q=np.linspace(0,1,33);neg=np.maximum(-effect,0);pos=np.maximum(effect,0)
    downs=-np.unique(np.r_[0.,np.quantile(neg,q),float(neg.max())+1e-6])
    ups=np.unique(np.r_[0.,np.quantile(pos,q),float(pos.max())+1e-6])
    support=np.bincount(labels,minlength=3)
    sorted_class=[np.sort(effect[labels==i]) for i in range(3)];best=None
    for down in downs:
        lo=np.asarray([np.searchsorted(s,down,side='left') for s in sorted_class])
        for up in ups:
            hi=support-np.asarray([np.searchsorted(s,up,side='right') for s in sorted_class])
            cm=np.stack([hi,support-lo-hi,lo],axis=1).astype(float)
            den=cm.sum(0)+cm.sum(1)
            f1=np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den>0).mean()
            v=(float(f1),-float(abs(down)+up),float(down),float(up))
            if best is None or v>best:best=v
    return {'down':best[2],'up':best[3],'val_macro_f1':best[0],
            'selection':'VAL only, 33 score quantiles per side plus boundary, no TEST tuning'}

def guard():
    import psutil
    me=psutil.Process();rss=me.memory_info().rss
    if rss>CFG['max_rss_gib']*2**30:raise MemoryError('STATE_TASK_RSS_LIMIT')
    limit=Path('/sys/fs/cgroup/memory.max');current=limit.with_name('memory.current')
    if limit.exists() and limit.read_text().strip()!='max':
        stats=dict(line.split() for line in limit.with_name('memory.stat').read_text().splitlines())
        reclaim=sum(int(stats.get(k,0)) for k in ['inactive_file','slab_reclaimable'])
        headroom=int(limit.read_text())-int(current.read_text())+reclaim
        if headroom<2*2**30:raise MemoryError('STATE_CGROUP_HEADROOM_LIMIT')
    return {'rss_gib':rss/2**30}
