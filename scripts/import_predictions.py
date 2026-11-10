"""Convert baseline arrays to a pair-aligned NPZ without inventing probabilities."""
import argparse
from pathlib import Path
import numpy as np

def convert(records,keys,out,hard=None,probabilities=None,effect=None):
    rows=np.load(records,allow_pickle=False)
    k=np.load(keys,allow_pickle=False)
    expected=np.column_stack([rows['target'],rows['query']])
    if k.dtype.names:k=np.column_stack([k['target'],k['query']])
    if k.shape!=expected.shape or not np.array_equal(k,expected):raise ValueError('Pair keys/order mismatch')
    arrays={'target':rows['target'],'query':rows['query'],'labels':rows['label']}
    if probabilities is not None:
        p=np.load(probabilities,allow_pickle=False)
        if p.shape!=(len(rows),3) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=2e-4):
            raise ValueError('Invalid probabilities')
        arrays['probabilities']=p
    if hard is not None:
        h=np.load(hard,allow_pickle=False)
        if h.shape!=(len(rows),) or not np.isin(h,[-1,0,1,2]).all():raise ValueError('Invalid hard predictions')
        arrays['y_pred']=h
    elif probabilities is not None:arrays['y_pred']=arrays['probabilities'].argmax(1)
    else:raise ValueError('Hard predictions or true probabilities are required')
    if effect is not None:
        d=np.load(effect,allow_pickle=False)
        if d.shape!=(len(rows),) or not np.isfinite(d).all():raise ValueError('Invalid continuous effect')
        arrays['delta_expr']=d
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f:np.savez_compressed(f,**arrays)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('records','keys','out'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('hard','probabilities','effect'):p.add_argument('--'+k,type=Path)
    convert(**vars(p.parse_args()))
