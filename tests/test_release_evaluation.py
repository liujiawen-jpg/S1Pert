import numpy as np
import pytest
from jev_a4.evaluation import evaluate,binary_auc,summarize

def fixture(tmp_path,pred,**kwargs):
    rows=np.array([(i,0,i%3) for i in range(6)],dtype=[('target','i4'),('query','i4'),('label','i1')])
    path=tmp_path/'pred.npz'
    np.savez(path,target=rows['target'],query=rows['query'],labels=rows['label'],y_pred=pred,**kwargs)
    return rows,path

def test_hard_labels_do_not_create_auc(tmp_path):
    rows,path=fixture(tmp_path,np.arange(6)%3)
    metrics=evaluate(rows,path)
    assert metrics['macro_f1']==1
    assert metrics['within_gene_polarity_auc'] is None

def test_order_rejected(tmp_path):
    rows,path=fixture(tmp_path,np.arange(6)%3)
    with pytest.raises(ValueError,match='order'):evaluate(rows[::-1],path)

def test_invalid_generation_counts_as_error(tmp_path):
    rows,path=fixture(tmp_path,np.full(6,-1))
    metrics=evaluate(rows,path)
    assert metrics['macro_f1']==0 and metrics['invalid_rate']==1

def test_ties_and_valid_ranking(tmp_path):
    assert binary_auc([1,0,1,0],[.5]*4)==.5
    rows,path=fixture(tmp_path,np.arange(6)%3,probabilities=np.eye(3)[np.arange(6)%3])
    metrics=evaluate(rows,path)
    assert metrics['within_gene_change_auc']==1 and metrics['within_gene_polarity_auc']==1

def test_sample_sd():
    result=summarize([dict(method='fixture',cell='A',seed=1,macro_f1=0.),dict(method='fixture',cell='A',seed=2,macro_f1=1.)])
    assert result[0]['sample_sd']==pytest.approx(2**-.5)
