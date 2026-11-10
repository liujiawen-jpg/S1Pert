"""Common paper metrics with exact pair alignment and genuine ranking scores."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from .core.metrics import metrics_from_confusion

def binary_auc(y,score):
    y=np.asarray(y,bool);score=np.asarray(score,float)
    if y.shape!=score.shape or not np.isfinite(score).all():raise ValueError('Invalid AUC arrays')
    a=int(y.sum());b=len(y)-a
    return float((rankdata(score)[y].sum()-a*(a+1)/2)/(a*b)) if a and b else None

def within_gene(y,query,change,polarity):
    order=np.argsort(query,kind='stable');g=query[order];y=y[order]
    c=np.asarray(change)[order];p=np.asarray(polarity)[order]
    bounds=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];cs=[];ps=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        v=binary_auc(y[a:b]!=1,c[a:b])
        if v is not None:cs.append(v)
        changed=y[a:b]!=1;v=binary_auc(y[a:b][changed]==0,p[a:b][changed])
        if v is not None:ps.append(v)
    return {'within_gene_change_auc':float(np.mean(cs)) if cs else None,
            'within_gene_polarity_auc':float(np.mean(ps)) if ps else None,
            'valid_change_queries':len(cs),'valid_polarity_queries':len(ps)}

def load_prediction(records,path):
    """Prediction NPZs must carry the same target/query/label order as records."""
    if records.dtype.names!=('target','query','label') or not np.isin(records['label'],[0,1,2]).all():
        raise ValueError('Expected resolved S1Pert records')
    with np.load(path,allow_pickle=False) as f:
        for key,column in [('target','target'),('query','query'),('labels','label')]:
            if key not in f or not np.array_equal(f[key],records[column]):raise ValueError('Prediction pair order mismatch: '+key)
        prob_key=next((k for k in ('probabilities','probs') if k in f),None)
        pred_key=next((k for k in ('y_pred','predictions','prediction') if k in f),None)
        change=polarity=None
        if prob_key:
            prob=f[prob_key]
            if prob.shape!=(len(records),3) or not np.isfinite(prob).all() or np.any(prob<0) or not np.allclose(prob.sum(1),1,atol=2e-4):
                raise ValueError('Invalid [Up,Same,Down] probabilities')
            pred=f[pred_key] if pred_key else prob.argmax(1)
            change=prob[:,0]+prob[:,2];polarity=prob[:,0]/np.maximum(change,1e-30)
        elif pred_key:
            pred=f[pred_key]
            score_key=next((k for k in ('delta_expr','effect','scores') if k in f),None)
            if score_key:
                effect=f[score_key]
                if effect.shape!=(len(records),) or not np.isfinite(effect).all():raise ValueError('Invalid continuous effect')
                change=np.abs(effect);polarity=effect
        else:raise ValueError('Missing predictions or probabilities')
        # Invalid language-model outputs count as errors, never silently as Same.
        if pred.shape!=(len(records),) or not np.isin(pred,[-1,0,1,2]).all():raise ValueError('Invalid class codes')
        return pred.copy(),change,polarity

def evaluate(records,path):
    pred,change,polarity=load_prediction(records,path);y=records['label'].astype(int)
    if len(records)==0:raise ValueError('Empty evaluation partition')
    cm=np.zeros((3,4),dtype=np.int64);np.add.at(cm,(y,np.where(pred<0,3,pred)),1)
    actual=cm.sum(1);called=cm[:,:3].sum(0);tp=np.diag(cm[:,:3])
    pre=np.divide(tp,called,out=np.zeros(3),where=called>0)
    rec=np.divide(tp,actual,out=np.zeros(3),where=actual>0)
    f1=np.divide(2*tp,actual+called,out=np.zeros(3),where=actual+called>0)
    out={'n':len(y),'macro_f1':float(f1.mean()),'balanced_accuracy':float(rec[actual>0].mean()),
         'accuracy':float(np.mean(y==pred)),'invalid_rate':float(np.mean(pred<0)),
         'within_gene_change_auc':None,'within_gene_polarity_auc':None}
    for i,name in enumerate(('up','same','down')):
        out.update({name+'_precision':float(pre[i]),name+'_recall':float(rec[i]),
                    name+'_f1':float(f1[i]),name+'_support':int(actual[i])})
    if change is not None:out.update(within_gene(y,records['query'],change,polarity))
    return out

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)

def summarize(rows):
    groups={}
    for row in rows:
        for cell in (row['cell'],'ALL_CELLS'):groups.setdefault((row['method'],cell),[]).append(row)
    output=[]
    for (method,cell),group in groups.items():
        for metric in [k for k in group[0] if k not in {'method','cell','seed','predictions','records'}]:
            vals=np.asarray([r[metric] for r in group if r.get(metric) is not None],float)
            output.append(dict(method=method,cell=cell,metric=metric,valid_splits=len(vals),
                total_splits=len(group),mean=float(vals.mean()) if len(vals) else None,
                sample_sd=float(vals.std(ddof=1)) if len(vals)>1 else None,
                sample_variance=float(vals.var(ddof=1)) if len(vals)>1 else None))
    return output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True,help='CSV: method,cell,seed,records,predictions')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    with a.registry.open() as f:entries=list(csv.DictReader(f))
    keys=[(r['method'],r['cell'],int(r['seed'])) for r in entries]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate method/cell/seed')
    rows=[]
    for r in entries:
        paths={k:(a.registry.parent/Path(r[k])).resolve() for k in ('records','predictions')}
        records=np.load(paths['records'],mmap_mode='r',allow_pickle=False)
        rows.append({'method':r['method'],'cell':r['cell'],'seed':int(r['seed']),**evaluate(records,paths['predictions'])})
    a.out.mkdir(parents=True,exist_ok=False)
    write_csv(a.out/'per_split.csv',rows);write_csv(a.out/'summary.csv',summarize(rows))
    print(f'Evaluated {len(rows)} method/split entries. Missing ranking scores remain blank.')

if __name__=='__main__':main()
