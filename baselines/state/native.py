"""Pinned upstream STATE, fresh weights; no old response predictor is loaded."""
from __future__ import annotations
import contextlib
import io
import sys
from pathlib import Path
import torch
import yaml
from common import CFG

VENDOR=Path(CFG['official_source'])
sys.path.insert(0,str(VENDOR/'src'))
from state.tx.models.state_transition import StateTransitionPerturbationModel

def build(n_genes, pert_dim, tiny=False):
    config=yaml.safe_load((VENDOR/'src/state/configs/model/state.yaml').read_text())['kwargs']
    config['cell_set_len']=CFG['cell_set_len']
    b=config['transformer_backbone_kwargs']
    b['max_position_embeddings']=config['cell_set_len'];b['hidden_size']=config['hidden_dim']
    if tiny:
        config['hidden_dim']=32;config['cell_set_len']=8
        b.update(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                 num_attention_heads=4,num_key_value_heads=4,head_dim=8,max_position_embeddings=8)
    with contextlib.redirect_stdout(io.StringIO()):
        model=StateTransitionPerturbationModel(input_dim=n_genes,output_dim=n_genes,
            gene_dim=n_genes,hvg_dim=n_genes,pert_dim=pert_dim,output_space='gene',
            embed_key='X_hvg',lr=CFG['lr'],batch_size=CFG['batch_conditions'],
            decoder_cfg=None,basal_mapping_strategy='random',**config)
    # The unused official CLI auxiliary count decoder is omitted; gene-space
    # forward and native energy loss do not call it, and no raw count head is trained.
    if not model.predict_residual or model.output_space!='gene':raise ValueError('unexpected native data flow')
    return model

def forward(model, controls, pert):
    n,s,g=controls.shape
    out=model.forward({'ctrl_cell_emb':controls,'pert_emb':pert[:,None,:].expand(-1,s,-1)},padded=True)
    return out.reshape(n,s,g)

def masked_native_loss(model,pred,target,queries):
    # Keep the official energy distance, evaluated per condition on legal TRAIN
    # query dimensions only. Labels and unknown responses never become inputs.
    losses=[]
    with torch.autocast(device_type=pred.device.type,enabled=False):
        for i,q in enumerate(queries):
            if not len(q):raise ValueError('empty strict TRAIN query set')
            index=torch.as_tensor(q,device=pred.device,dtype=torch.long)
            a=pred[i:i+1].index_select(-1,index).float()
            b=target[i:i+1].index_select(-1,index).float()
            losses.append(model._compute_distribution_loss(a,b).mean())
    value=torch.stack(losses).mean()
    if not torch.isfinite(value):raise FloatingPointError('nonfinite native STATE energy loss')
    return value
