"""Original upstream cores plus explicitly named data/classification adapters."""
from __future__ import annotations
import importlib.util, json, logging, sys, types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from common import ROOT, CFG, load_module

class GraphClassifier(nn.Module):
    def __init__(self, meta):
        super().__init__()
        base=ROOT/'prepared'/'public_graph'
        nodes=json.loads((base/'nodes.json').read_text()); index={n:i for i,n in enumerate(nodes)}
        self.register_buffer('target_nodes',torch.tensor([index[g] for g in meta['target_ids']]))
        self.register_buffer('query_nodes',torch.tensor([index[g] for g in meta['query_ids']]))
        edges=np.load(base/'edge_index.npy'); edge_types=np.load(base/'edge_types.npy')
        graph=SimpleNamespace(x=torch.arange(len(nodes)),edge_index=torch.from_numpy(edges.T.copy()),
                              edge_attr=torch.from_numpy(edge_types))
        p=CFG['gat']
        args=SimpleNamespace(vocab_size=len(nodes),embed_dim=p['width'],num_kgs=int(edge_types.max())+1,
            num_layers=p['layers'],dropout=p['dropout'],ffn_embed_dim=p['ffn_width'])
        module=load_module(Path(CFG['perturbqa_source'])/'examples/sanity_check/model/gnn.py','upstream_perturbqa_gat')
        self.core=module.GAT(args,graph)
    def forward(self,target,query):
        return self.core({'pert':self.target_nodes[target],'gene':self.query_nodes[query]})

class GenePTClassifier(nn.Module):
    """Factorized multinomial logistic regression on concatenated frozen vectors.

    This computes the exact linear classifier without materializing [N,2D].
    It is NOT an old prediction plus a residual model.
    """
    def __init__(self,target_embeddings,query_embeddings):
        super().__init__()
        self.register_buffer('target_embeddings',torch.as_tensor(target_embeddings,dtype=torch.float64))
        self.register_buffer('query_embeddings',torch.as_tensor(query_embeddings,dtype=torch.float64))
        d=target_embeddings.shape[1]
        self.target_weight=nn.Parameter(torch.zeros(d,3,dtype=torch.float64))
        self.query_weight=nn.Parameter(torch.zeros(d,3,dtype=torch.float64))
        self.bias=nn.Parameter(torch.zeros(3,dtype=torch.float64))
    def forward(self,target,query):
        return (self.target_embeddings@self.target_weight)[target]+(self.query_embeddings@self.query_weight)[query]+self.bias

def genept_embeddings(meta,records,kind):
    base=ROOT/'prepared'/'genept'; names=json.loads((base/f'{kind}_names.json').read_text())
    matrix=np.load(base/f'{kind}.npy',mmap_mode='r'); index={str(n):i for i,n in enumerate(names)}
    out=[]; audit={}
    for axis,ids,symbols,col in [('target',meta['target_ids'],meta['target_symbols'],'target'),
                                ('query',meta['query_ids'],meta['query_symbols'],'query')]:
        mapped=np.asarray([index.get(g,index.get(s,-1)) for g,s in zip(ids,symbols)])
        train_known=np.unique(mapped[np.unique(records[col])]); train_known=train_known[train_known>=0]
        if not len(train_known): raise ValueError('no GenePT TRAIN embedding matches')
        fallback=np.asarray(matrix[train_known]).mean(0)
        emb=np.repeat(fallback[None],len(ids),axis=0)
        good=mapped>=0; emb[good]=matrix[mapped[good]]
        out.append(emb)
        audit[axis]={'matched':int(good.sum()),'total':len(good),'missing_policy':'mean embedding of TRAIN identities; not mean response'}
    return *out,audit

def gears_model(meta,cell,seed,device):
    base=ROOT/'prepared'/cell/f'seed_{seed}'
    co=np.load(base/'coexpression.npz'); go=np.load(base/'go.npz')
    args={'num_genes':len(meta['query_ids']),'num_perts':len(meta['target_ids']),
          'hidden_size':CFG['gears']['hidden_size'],'uncertainty':False,'num_go_gnn_layers':1,
          'decoder_hidden_size':16,'num_gene_gnn_layers':1,'no_perturb':False,'device':device,
          'G_coexpress':torch.from_numpy(co['edges']),'G_coexpress_weight':torch.from_numpy(co['weights']),
          'G_go':torch.from_numpy(go['edges']),'G_go_weight':torch.from_numpy(go['weights'])}
    module=load_module(Path(CFG['gears_source'])/'gears/model.py','upstream_gears_model')
    return module.GEARS_Model(args)

def gears_loss():
    # Execute the unchanged official loss function without importing unrelated
    # download/statistical helpers and their optional dependencies.
    import ast
    source=(Path(CFG['gears_source'])/'gears/utils.py').read_text()
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='loss_fct')
    module=ast.Module(body=[function],type_ignores=[])
    namespace={'torch':torch,'np':np}
    exec(compile(module,'official_GEARS_loss_fct','exec'),namespace)
    return namespace['loss_fct']

def scgpt_model(meta,device):
    # Load official model modules without importing unrelated trainer/tokenizer
    # packages (whose obsolete torchtext extension is not needed for this task).
    source=Path(CFG['scgpt_source'])/'scgpt'
    package=types.ModuleType('scgpt');package.__path__=[str(source)];package.logger=logging.getLogger('scGPT')
    sys.modules['scgpt']=package
    sub=types.ModuleType('scgpt.model');sub.__path__=[str(source/'model')];sys.modules['scgpt.model']=sub
    util=types.ModuleType('scgpt.utils')
    def map_ids(raw_ids,gene_ids):
        return torch.as_tensor(gene_ids,device=raw_ids.device)[raw_ids] if torch.is_tensor(raw_ids) else gene_ids[raw_ids]
    util.map_raw_id_to_vocab_id=map_ids;sys.modules['scgpt.utils']=util
    import importlib
    generator=importlib.import_module('scgpt.model.generation_model').TransformerGenerator
    pre=Path(CFG['scgpt_pretrained']); args=json.loads((pre/'args.json').read_text()); vocab=json.loads((pre/'vocab.json').read_text())
    for token in ['<pad>','<cls>','<eoc>']:
        if token not in vocab: vocab[token]=len(vocab)
    # Extend missing genes with fresh embeddings, instead of silently deleting test records.
    symbols=list(dict.fromkeys(meta['query_symbols']+meta['target_symbols']))
    missing=[]
    for symbol in symbols:
        if symbol not in vocab: vocab[symbol]=len(vocab);missing.append(symbol)
    model=generator(len(vocab),args['embsize'],args['nheads'],args['d_hid'],args['nlayers'],
        nlayers_cls=args['n_layers_cls'],n_cls=1,vocab=vocab,dropout=CFG['scgpt']['dropout'],
        pad_token='<pad>',pad_value=0,pert_pad_id=2,use_fast_transformer=False)
    state=torch.load(pre/'best_model.pt',map_location='cpu',weights_only=True)
    model_state=model.state_dict(); loaded=[]; skipped=[]
    for key,value in state.items():
        if not key.startswith(('encoder.','value_encoder.','transformer_encoder.')): continue
        mapped=key.replace('.self_attn.Wqkv.weight','.self_attn.in_proj_weight').replace('.self_attn.Wqkv.bias','.self_attn.in_proj_bias')
        if mapped=='encoder.embedding.weight' and value.shape[0]<=model_state[mapped].shape[0]:
            model_state[mapped][:value.shape[0]]=value;loaded.append(mapped)
        elif mapped in model_state and value.shape==model_state[mapped].shape:
            model_state[mapped]=value;loaded.append(mapped)
        else: skipped.append({'source':key,'mapped':mapped,'shape':list(value.shape)})
    expected=[k for k in model_state if k.startswith(('encoder.','value_encoder.','transformer_encoder.'))]
    absent=sorted(set(expected)-set(loaded))
    if absent or skipped: raise ValueError(f'incomplete generic scGPT backbone import: absent={absent}; skipped={skipped}')
    model.load_state_dict(model_state)
    model.to(device)
    genes=torch.tensor([vocab[s] for s in meta['query_symbols']],dtype=torch.long,device=device)
    target_query=np.asarray(meta['target_query_indices'])
    # Targets not measured in the query axis still receive a dedicated input token
    # with legal missing-control expression=0 and an explicit intervention flag.
    extra=[i for i,q in enumerate(target_query) if q<0]
    target_positions=target_query.copy()
    for offset,i in enumerate(extra):target_positions[i]=len(genes)+offset
    all_gene_tokens=torch.cat([genes,torch.tensor([vocab[meta['target_symbols'][i]] for i in extra],device=device,dtype=torch.long)])
    report={'generic_pretrained_source':str(pre),'loaded_backbone_tensors':len(loaded),
        'new_public_vocabulary_symbols':missing,'unmeasured_target_input_tokens':len(extra),
        'task_heads':'fresh random initialization','attention':'PyTorch SDPA with mathematically equivalent QKV key conversion'}
    return model,all_gene_tokens,target_positions,report

def native_forward(model,method,control,targets,query_indices=None,gene_tokens=None,target_positions=None):
    if method=='gears':
        b,g=control.shape
        batch=SimpleNamespace(x=control.reshape(-1,1),batch=torch.arange(b,device=control.device).repeat_interleave(g),
                              pert_idx=[[int(t)] for t in targets])
        return model(batch)
    assert query_indices is not None and gene_tokens is not None
    b=len(targets)
    total_genes=len(gene_tokens)
    if control.shape[1]<total_genes:
        control=torch.nn.functional.pad(control,(0,total_genes-control.shape[1]))
    positions=torch.as_tensor(target_positions,device=control.device)[targets]
    flags=(positions[:,None]==query_indices[None,:]).long()
    ids=gene_tokens[query_indices][None,:].expand(b,-1)
    return model(ids,control[:,query_indices],flags,src_key_padding_mask=torch.zeros_like(ids,dtype=torch.bool))['mlm_output']
