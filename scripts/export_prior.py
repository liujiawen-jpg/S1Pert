"""Export independent P1 predictions from audited TRAIN-only prepared evidence."""
import argparse
from pathlib import Path
import numpy as np

def export(partition,out):
    rows=np.load(partition/'records.npy',allow_pickle=False)
    probabilities=np.load(partition/'prior_features.npy',allow_pickle=False)[:,:3]
    if rows.dtype.names!=('target','query','label') or probabilities.shape!=(len(rows),3):
        raise ValueError('Unexpected prepared schema')
    if not np.isfinite(probabilities).all() or np.any(probabilities<0) or not np.allclose(probabilities.sum(1),1):
        raise ValueError('Invalid evidence probabilities')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f:
        np.savez_compressed(f,target=rows['target'],query=rows['query'],labels=rows['label'],
                            y_pred=probabilities.argmax(1),probabilities=probabilities)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partition',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();export(a.partition,a.out)
