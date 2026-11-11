#!/usr/bin/env python3
"""Stage 5: derive Fig. 1 metrics from aligned predictions and TRAIN-only priors.

Registry CSV: method,cell,seed,records,predictions; paths relative to CSV.
Optional train_records/prior_features columns override workspace defaults.
No model is trained here; Stage 3 produces the per-sample prediction files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .paper_common import *
except ImportError:
    from paper_common import *

ALIASES = {'GenePT-protein':'GenePT-Prot', 'GenePT':'GenePT-Gene',
           'PertReason':'Qwen3-4B-LoRA', 'A4':'S1Pert', 'ALIGNED-MLP':'ALIGNED',
           'STATE-ST-native-strict-A4-adapted':'STATE'}

def validate_records(records):
    if not records.dtype.names or not {'target','query','label'}.issubset(records.dtype.names):
        raise ValueError('records.npy must have target,query,label fields.')
    if not np.isin(records['label'], [0,1,2]).all() or np.any(records['query'] < 0):
        raise ValueError('Stage 5 expects resolved three-class records and nonnegative query IDs.')
    keys = np.column_stack([records['target'], records['query']])
    if len(np.unique(keys, axis=0)) != len(records):
        raise ValueError('Duplicate target/query pair would bias cross-target ranking.')

def load_prediction(path, records):
    with np.load(path, allow_pickle=False) as saved:
        for names, field in [(('target','targets'),'target'),(('query',),'query'),(('labels','label','y_true'),'label')]:
            key = next((k for k in names if k in saved), None)
            if key is None or not np.array_equal(saved[key], records[field]):
                raise ValueError(f'Prediction alignment failed: {path}/{field}.')
        probs_key = next((k for k in ('probabilities','probs') if k in saved), None)
        pred_key = next((k for k in ('y_pred','predictions','prediction','predicted') if k in saved), None)
        if probs_key:
            p = np.array(saved[probs_key], dtype=float)
            if p.shape != (len(records),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=2e-4):
                raise ValueError('Invalid [Up,Same,Down] probabilities.')
            pred = saved[pred_key].copy() if pred_key else p.argmax(1)
            change = p[:,0]+p[:,2]
            # Preserve the frozen evaluator's zero-denominator convention.
            # In particular, ALIGNED may legitimately return [0,1,0].
            polarity = p[:,0]/np.maximum(change,1e-30)
        else:
            if pred_key is None:
                raise ValueError('Need probabilities or native hard predictions.')
            pred = saved[pred_key].copy()
            key = next((k for k in ('delta_expr','effect','predicted_effect','scores') if k in saved), None)
            if key:
                polarity = np.asarray(saved[key], float).copy(); change = np.abs(polarity)
                if polarity.shape != (len(records),) or not np.isfinite(polarity).all():
                    raise ValueError('Invalid continuous effect score.')
            else:
                change = polarity = None
        if pred.shape != (len(records),) or not np.isin(pred,[-1,0,1,2]).all():
            raise ValueError('Invalid hard prediction values/axis.')
    return pred, change, polarity

def train_prior(train, test, prior_features=None):
    validate_records(train); validate_records(test)
    if np.intersect1d(train['target'],test['target']).size:
        raise ValueError('TRAIN/test target overlap: not target-held-out.')
    n = max(int(train['query'].max(initial=-1)), int(test['query'].max(initial=-1)))+1
    counts = np.bincount(train['query'].astype(np.int64)*3+train['label'],minlength=n*3).reshape(n,3)
    if counts.sum() == 0:
        raise ValueError('Empty TRAIN split cannot define historical propensity.')
    support = counts.sum(1); pi=(counts+1)/(support[:,None]+3)
    pi[support==0]=(counts.sum(0)+1)/(counts.sum()+3)
    entropy=-(pi*np.log(np.maximum(pi,1e-30))).sum(1)/np.log(3)
    if prior_features is not None:
        f=np.load(prior_features,mmap_mode='r')
        if f.shape[0]!=len(test) or f.shape[1]!=12 or not np.allclose(f[:,:3],pi[test['query']],atol=2e-6):
            raise ValueError('TRAIN-only reconstructed probabilities differ from frozen 12-D evidence.')
    return pi,support,entropy

def masks(records, pi, support, entropy, entropy_mode):
    if entropy_mode=='fixed-thirds':
        cuts=np.array([1/3,2/3])
    elif entropy_mode=='train-quantiles':
        # Quantiles are fitted on supported TRAIN query genes, never test labels.
        cuts=np.quantile(entropy[support>0],[1/3,2/3])
    else:
        raise ValueError(entropy_mode)
    q=records['query']; h=entropy[q]; s=support[q]
    return {
        'resolved':np.ones(len(records),bool),
        'entropy_low':h<cuts[0],
        'entropy_medium':(h>=cuts[0])&(h<cuts[1]),
        'entropy_high':h>=cuts[1],
        'counter_propensity':records['label']!=pi[q].argmax(1),
        'support_ge50':s>=50, 'support_10_49':(s>=10)&(s<50),
        'support_1_9':(s>=1)&(s<10), 'support_0':s==0,
    },cuts

def run(work_dir, registry=None, output=None, entropy_mode='fixed-thirds'):
    root=work_root(work_dir)
    registry=Path(registry or root/'paper/predictions.csv').resolve()
    output=Path(output or root/'paper/prior_stress').resolve(); output.mkdir(parents=True,exist_ok=True)
    rows=read_csv(registry)
    required={'method','cell','seed','records','predictions'}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError('Registry must contain method,cell,seed,records,predictions.')
    groups={}; seen=set()
    for row in rows:
        row=dict(row);row['method']=ALIASES.get(row['method'],row['method'])
        key=(row['method'],row['cell'],int(row['seed']))
        if key in seen: raise ValueError(f'Duplicate registry entry: {key}')
        seen.add(key)
        # P1 is constructed here, not trusted as an external prediction file.
        if row['method']!='P1':groups.setdefault((row['cell'],int(row['seed'])),[]).append(row)
    results=[]; provenance=[]; generated=[]
    for (cell,seed),entries in sorted(groups.items()):
        first=entries[0]
        test_path=relative_path(first['records'],registry.parent)
        test=np.load(test_path,allow_pickle=False);validate_records(test)
        train_path=relative_path(first['train_records'],registry.parent) if first.get('train_records') else root/f'data/{cell}/seed_{seed}/train/records.npy'
        feature_path=relative_path(first['prior_features'],registry.parent) if first.get('prior_features') else test_path.parent/'prior_features.npy'
        train=np.load(train_path,allow_pickle=False)
        pi,support,entropy=train_prior(train,test,feature_path if feature_path.exists() else None)
        subsets,cuts=masks(test,pi,support,entropy,entropy_mode)
        p=pi[test['query']]
        p1path=output/f'predictions/P1/{cell}/seed_{seed}/test_predictions.npz';p1path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p1path,target=test['target'],query=test['query'],labels=test['label'],probabilities=p,y_pred=p.argmax(1))
        generated.append(dict(method='P1',cell=cell,seed=seed,records=str(test_path),predictions=str(p1path)))
        selected=[('P1',p.argmax(1),p[:,0]+p[:,2],p[:,0]/(p[:,0]+p[:,2]),p1path)]
        for row in entries:
            rp=relative_path(row['records'],registry.parent)
            other=np.load(rp,allow_pickle=False)
            if any(not np.array_equal(other[f],test[f]) for f in ('target','query','label')):
                raise ValueError('Methods must use exactly the same test order and truth.')
            path=relative_path(row['predictions'],registry.parent)
            pred,change,polarity=load_prediction(path,test)
            selected.append((row['method'],pred,change,polarity,path))
        for method,pred,change,polarity,path in selected:
            for regime,mask in subsets.items():
                metrics=classification(test['label'][mask],pred[mask],test['query'][mask],
                    change[mask] if change is not None else None,polarity[mask] if polarity is not None else None)
                results.append(dict(model=method,cell=cell,seed=seed,regime=regime,**metrics))
            provenance.append(dict(method=method,cell=cell,seed=seed,predictions=str(path),prediction_sha256=sha256(path),
                records=str(test_path),records_sha256=sha256(test_path),train_records=str(train_path),train_sha256=sha256(train_path),
                entropy_mode=entropy_mode,entropy_cut_low=float(cuts[0]),entropy_cut_high=float(cuts[1]),
                prior_features_validated=str(feature_path) if feature_path.exists() else None,
                ranking_status='MEASURED' if change is not None else 'N/A_HARD_LABELS_ONLY'))
    summary=summarize(results)
    write_csv(output/'per_split.csv',results);write_csv(output/'summary.csv',summary)
    write_csv(output/'p1_registry.csv',generated);write_csv(output/'input_provenance.csv',provenance)
    report=dict(status='PASS',source_registry=str(registry),registry_sha256=sha256(registry),
        entropy_mode=entropy_mode,entropy_protocol='Historical fixed normalized thirds, NOT empirical tertiles.' if entropy_mode=='fixed-thirds' else 'Separate protocol: quantiles fitted on supported TRAIN query genes.',
        train_only_evidence=True,test_labels_used_for_stratum_truth_only=True,
        models=sorted({r['model'] for r in results}),split_count=len(groups),
        class_order=['Up','Same','Down'],p1='Per-query add-one TRAIN counts; TRAIN-global fallback for zero support.',
        auc='Equal-weight mean of defined per-query AUCs across held-out targets; polarity uses true changed pairs only.',
        holm_family='All registered comparators for each cell/regime/metric; plotting subsets do not change this family.',
        aggregation='Sample standard deviation, ddof=1; repeated held-out splits are not independent biological replicates.',
        output=str(output))
    write_json(output/'manifest.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir');parser.add_argument('--registry',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--entropy-mode',choices=['fixed-thirds','train-quantiles'],default='fixed-thirds')
    args=parser.parse_args();print(json.dumps(run(args.work_dir,args.registry,args.output,args.entropy_mode),indent=2))
