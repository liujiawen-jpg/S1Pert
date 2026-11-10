"""Bounded, audited inputs. Public knowledge is separate from TRAIN responses."""
from __future__ import annotations
import json, os, pickle, zipfile
from pathlib import Path
import h5py
import numpy as np
from scipy import sparse
from common import ROOT, CFG, write_json, digest_file, data, check_resources

RAW_FILES = {
    "hepg2": "Nadig2024_GSE264667_raw/GSE264667_hepg2_raw_singlecell_01.h5ad",
    "jurkat": "Nadig2024_GSE264667_raw/GSE264667_jurkat_raw_singlecell_01.h5ad",
    "rpe1": "scPerturb_prepro/ReplogleWeissman2022_rpe1.h5ad",
    "k562_essential": "scPerturb_prepro/ReplogleWeissman2022_K562_essential.h5ad",
}

def strings(node):
    return np.asarray(node.asstr()[:], dtype=str) if h5py.check_string_dtype(node.dtype) else np.asarray(node[:]).astype(str)

def categorical(group, name):
    col = group[name]
    if isinstance(col, h5py.Group):
        categories, codes = col["categories"], col["codes"][:]
    elif "__categories" in group and name in group["__categories"]:
        categories, codes = group["__categories"][name], col[:]
    elif "categories" in col.attrs:
        ref = col.attrs["categories"]
        if isinstance(ref, bytes): ref = ref.decode()
        categories = col.file[ref] if isinstance(ref, h5py.Reference) else group[ref]
        codes = col[:]
    else:
        return strings(col)
    values = strings(categories)
    if np.any(codes < -1) or np.any(codes >= len(values)):
        raise ValueError("categorical code out of bounds")
    out = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    out[valid] = values[codes[valid]]
    return out.astype(str)

def checked_archive(name, expected_md5):
    path = Path(CFG["assets_root"]) / name
    if digest_file(path, "md5") != expected_md5:
        raise ValueError(f"public archive checksum mismatch: {name}")
    return zipfile.ZipFile(path)

def prepare_graph():
    dest = ROOT / "prepared" / "public_graph"
    if (dest / "report.json").exists(): return dest
    dest.mkdir(parents=True, exist_ok=True)
    z = checked_archive("kg.zip", CFG["knowledge_resources"]["perturbqa_kg"]["md5"])
    # Exact public graph inputs, not AROMA trained GNN weights.
    members = [n for n in z.namelist() if n.endswith('.json') and '__MACOSX' not in n]
    selected = [n for n in members if Path(n).name in ['string.json','corum.json','go.json','reactome.json']]
    if not selected:
        raise ValueError(f"no recognized public graph resources: {members}")
    edge_pairs, types, node_set = [], [], set()
    provenance = []
    for i, name in enumerate(sorted(selected)):
        payload = json.loads(z.read(name))
        graph = payload[0] if isinstance(payload, list) else payload
        if not isinstance(graph, dict):
            raise ValueError(f"unrecognized graph structure: {name}")
        count = 0
        for a, neighbors in graph.items():
            if not isinstance(neighbors, list): raise ValueError(f"bad adjacency: {name}/{a}")
            a = str(a); node_set.add(a)
            for entry in neighbors:
                b = str(entry[0] if isinstance(entry, (list,tuple)) else entry)
                node_set.add(b)
                edge_pairs.extend([(a,b),(b,a)]); types.extend([i,i]); count += 2
        provenance.append({"member": name, "directed_edges_after_symmetrization": count})
    # Preserve canonical measured IDs; match public names only if symbol unambiguous.
    alias_sets = {}
    for cell in CFG["cells"]:
        meta = json.loads((Path(CFG["a4_data"])/cell/'dataset.json').read_text())
        for ids, symbols in [(meta['query_ids'],meta['query_symbols']), (meta['target_ids'],meta['target_symbols'])]:
            for ident, symbol in zip(ids,symbols):
                alias_sets.setdefault(str(symbol),set()).add(str(ident))
                node_set.add(str(ident))
    aliases = {k: next(iter(v)) for k,v in alias_sets.items() if len(v)==1}
    def canonical(n): return aliases.get(n,n)
    nodes = sorted({canonical(n) for n in node_set})
    index = {n:i for i,n in enumerate(nodes)}
    edges = np.asarray([(index[canonical(a)],index[canonical(b)]) for a,b in edge_pairs], dtype=np.int64)
    # Remove duplicate edges per relation, without dropping isolated measured genes.
    packed = np.column_stack([edges, np.asarray(types,dtype=np.int64)])
    packed = np.unique(packed,axis=0)
    np.save(dest/'edge_index.npy',packed[:,:2].T)
    np.save(dest/'edge_types.npy',packed[:,2])
    write_json(dest/'nodes.json',nodes)
    write_json(dest/'report.json',{"public_inputs":provenance,"nodes":len(nodes),"edges":len(packed),
        "ambiguous_symbol_aliases_not_merged":sum(len(v)>1 for v in alias_sets.values()),
        "source_responses_used":False,"archive_md5":digest_file(Path(CFG['assets_root'])/'kg.zip','md5')})
    return dest

def prepare_genept():
    dest=ROOT/'prepared'/'genept'
    if (dest/'report.json').exists(): return dest
    dest.mkdir(parents=True,exist_ok=True)
    z=checked_archive('GenePT_emebdding_v2.zip',CFG['knowledge_resources']['genept']['md5'])
    members=[n for n in z.namelist() if not '__MACOSX' in n and n.rstrip('.').endswith(('.pkl','.pickle'))]
    # Upstream spelling varies; prefer ada-002 vectors (1536 dimensions).
    candidates={}
    for name in members:
        low=name.lower()
        if 'emb' not in low: continue
        kind='prot' if any(k in low for k in ['prot','uniprot']) else 'gene'
        rank=(('ada' in low or 'v1' in low), not any(k in low for k in ['large','3_','3-']), name)
        candidates.setdefault(kind,[]).append((rank,name))
    if set(candidates) != {'gene','prot'}:
        raise ValueError(f'cannot identify official gene/protein vectors: {members}')
    report={"archive_md5":digest_file(Path(CFG['assets_root'])/'GenePT_emebdding_v2.zip','md5'),"members":members,"variants":{}}
    for kind, choices in candidates.items():
        name=max(choices)[1]
        # Trusted original-author data archive, checked against Zenodo MD5.
        obj=pickle.loads(z.read(name))
        if not isinstance(obj,dict): raise ValueError(f'expected gene->vector dictionary: {name}')
        names=[]; vectors=[]
        for key,value in obj.items():
            a=np.asarray(value,dtype=np.float32)
            if a.ndim!=1 or a.size<128 or not np.isfinite(a).all():
                raise ValueError(f'invalid embedding at {name}/{key}')
            names.append(str(key)); vectors.append(a)
        matrix=np.stack(vectors)
        np.save(dest/f'{kind}.npy',matrix)
        write_json(dest/f'{kind}_names.json',names)
        report['variants'][kind]={"member":name,"shape":list(matrix.shape)}
    write_json(dest/'report.json',report)
    return dest

def map_cell_targets(meta, identities, symbols=None):
    """Keep explicit IDs; use source symbols only when the source ID is absent."""
    alias={}
    for names in [meta['target_partition_group_ids'],meta['target_ids']]:
        for i,name in enumerate(names):
            if str(name) in alias and alias[str(name)]!=i:
                raise ValueError('ambiguous target identity alias')
            alias[str(name)]=i
    by_symbol={}
    for i,name in enumerate(meta['target_symbols']):by_symbol.setdefault(str(name),[]).append(i)
    controls=np.isin(identities,['non-targeting','CONTROL','control',''])
    groups=np.asarray([alias.get(str(x),-2) for x in identities],dtype=np.int32)
    rescued=0
    if symbols is not None:
        for row in np.flatnonzero((groups==-2)&np.isin(identities,['nan','NaN','None','NA'])):
            hits=by_symbol.get(str(symbols[row]),[])
            if len(hits)!=1:raise ValueError('missing source ID has ambiguous/unmapped source symbol')
            i=hits[0]
            if meta['target_ids'][i]!='SYMBOL:'+str(symbols[row]):
                raise ValueError('symbol fallback must match sealed source-namespaced target ID')
            groups[row]=i;rescued+=1
    groups[controls]=-1
    return groups,controls,rescued

def prepare_expression(cell):
    dest=ROOT/'prepared'/cell/'expression'
    if (dest/'report.json').exists(): return dest
    dest.mkdir(parents=True,exist_ok=True)
    meta=json.loads((Path(CFG['a4_data'])/cell/'dataset.json').read_text())
    raw=Path(CFG['raw_root'])/RAW_FILES[cell]
    stat=raw.stat()
    with h5py.File(raw,'r') as f:
        var=f['var']; obs=f['obs']
        genes=categorical(var,'gene_id' if cell in ['hepg2','jurkat'] else 'ensembl_id')
        if list(genes)!=meta['query_ids']: raise ValueError('raw and A4 gene axes differ')
        if cell in ['hepg2','jurkat']:
            identities=categorical(obs,'gene_id')
            groups,controls,rescued=map_cell_targets(meta,identities,categorical(obs,'gene'))
        else:
            identities=categorical(obs,'perturbation')
            alias={str(a):i for i,a in enumerate(meta['target_symbols'])}
            controls=np.isin(identities,['control','non-targeting',''])
            groups=np.asarray([alias.get(str(x),-2) for x in identities],dtype=np.int32)
            groups[controls]=-1;rescued=0
        if not controls.any(): raise ValueError('no controls')
        umi=np.asarray(obs['UMI_count'][:],dtype=np.float64)
        if not np.isfinite(umi).all() or (umi<=0).any(): raise ValueError('invalid library sizes')
        n,g=len(groups),len(genes)
        matrix_path=dest/'log1p_cp10k.npy'
        reused=matrix_path.exists()
        if reused:
            matrix=np.load(matrix_path,mmap_mode='r')
            if matrix.shape!=(n,g) or matrix.dtype!=np.float32:
                raise ValueError('existing normalized matrix identity/shape mismatch')
        else:
            matrix=np.lib.format.open_memmap(matrix_path,mode='w+',dtype=np.float32,shape=(n,g))
        sums=np.zeros((len(meta['target_ids'])+1,g),dtype=np.float64)
        ns=np.zeros(len(sums),dtype=np.int64)
        X=f['X']
        for lo in range(0,n,512):
            hi=min(lo+512,n)
            a=np.asarray(X[lo:hi],dtype=np.float32)
            if not np.isfinite(a).all() or (a<0).any() or not np.allclose(a,np.rint(a),atol=0,rtol=0):
                raise ValueError('source X is not raw integer counts')
            if (a.sum(1,dtype=np.float64)>umi[lo:hi]+1e-4).any(): raise ValueError('UMI denominator mismatch')
            a=np.log1p(a/umi[lo:hi,None]*10000).astype(np.float32)
            if reused:
                if not np.array_equal(matrix[lo:hi],a):
                    raise ValueError(f'existing normalized matrix differs from raw reconstruction at rows {lo}:{hi}')
            else:matrix[lo:hi]=a
            gr=groups[lo:hi]+1; valid=gr>=0
            np.add.at(sums,gr[valid],a[valid]); np.add.at(ns,gr[valid],1)
            if lo%16384==0:
                print(json.dumps({'stage':'expression','cell':cell,'rows':hi,'total':n}),flush=True)
                check_resources()
        if not reused:matrix.flush()
    if (ns[1:]==0).any(): raise ValueError('some sealed perturbation targets lack raw cells')
    moments=Path(CFG['moments'])/cell
    gids=np.load(moments/'group_id.npy').astype(str)
    oldn=np.load(moments/'group_n.npy')
    expected=dict(zip(gids,oldn))
    if any(int(ns[i+1])!=int(expected[t]) for i,t in enumerate(meta['target_ids'])):
        raise ValueError('raw target cell counts differ from sealed moments')
    np.save(dest/'cell_target.npy',groups)
    np.save(dest/'group_log_means.npy',(sums/np.maximum(1,ns[:,None])).astype(np.float32))
    np.save(dest/'group_n.npy',ns)
    # Validate against audited CONTROL means, allowing log Jensen inequality only.
    if int(ns[0])!=int(meta['control_n']): raise ValueError('control population changed')
    write_json(dest/'report.json',{'cell':cell,'source':str(raw),'source_size':stat.st_size,
        'source_mtime_ns':stat.st_mtime_ns,'shape':[n,g],'excluded_unmapped_cells':int((groups==-2).sum()),
        'control_cells':int(ns[0]),'normalization':'log1p(raw_X / original obs.UMI_count * 10000)',
        'missing_id_cells_resolved_by_explicit_source_symbol':rescued,
        'all_target_counts_match_sealed_moments':True,
        'normalized_matrix_reused_after_full_raw_reconstruction_check':reused,
        'all_measured_genes':True,'target_cells_used_for_training':'TRAIN target membership enforced later',
        'cell_target_sha256':digest_file(dest/'cell_target.npy'), 'a4_metadata_sha256':digest_file(Path(CFG['a4_data'])/cell/'dataset.json')})
    return dest

def prepare_native_graphs(cell,seed):
    import torch
    dest=ROOT/'prepared'/cell/f'seed_{seed}'
    if (dest/'graphs.json').exists(): return dest
    dest.mkdir(parents=True,exist_ok=True)
    meta,part,records,ident=data(cell,seed)
    expr=prepare_expression(cell)
    x=np.load(expr/'log1p_cp10k.npy',mmap_mode='r')
    groups=np.load(expr/'cell_target.npy')
    rows=np.flatnonzero((groups==-1)|((groups>=0)&(part[np.maximum(groups,0)]==0)))
    # Compute exact single-cell Pearson sufficient statistics on GPU using TRAIN+control only.
    g=x.shape[1]; device='cuda'; sums=torch.zeros(g,device=device,dtype=torch.float64)
    cross=torch.zeros((g,g),device=device,dtype=torch.float32)
    for lo in range(0,len(rows),1024):
        a=torch.as_tensor(np.asarray(x[rows[lo:lo+1024]]),device=device)
        sums+=a.double().sum(0); cross.addmm_(a.T,a)
    cov=cross-sums.float()[:,None]*sums.float()[None,:]/len(rows)
    sd=cov.diag().clamp(min=1e-12).sqrt()
    cor=(cov/sd[:,None]/sd[None,:]).abs().nan_to_num().clamp(0,1)
    vals,inds=cor.topk(CFG['gears']['top_k']+1,dim=1)
    targets=torch.arange(g,device=device)[:,None].expand_as(inds)
    keep=vals>CFG['gears']['correlation_threshold']
    np.savez(dest/'coexpression.npz',edges=torch.stack([inds[keep],targets[keep]]).cpu().numpy(),weights=vals[keep].cpu().numpy())
    del cor,cov,cross,sd,vals,inds,targets,sums
    # Public Gene Ontology Jaccard graph, same >0.1 and top-k convention as GEARS.
    with open(CFG['gene2go'],'rb') as f: gene2go=pickle.load(f)
    sets=[set(gene2go.get(s,[])) for s in meta['target_symbols']]
    terms=sorted(set().union(*sets)); term_index={t:i for i,t in enumerate(terms)}
    rr=[];cc=[]
    for i,terms_i in enumerate(sets):
        for t in terms_i: rr.append(i);cc.append(term_index[t])
    memberships=sparse.csr_matrix((np.ones(len(rr),dtype=np.float32),(rr,cc)),shape=(len(sets),len(terms)))
    intersection=(memberships@memberships.T).toarray()
    sizes=np.asarray([len(s) for s in sets],dtype=np.float32)
    union=sizes[:,None]+sizes[None,:]-intersection
    sim=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
    edges=[];weights=[]
    for i,row in enumerate(sim):
        candidates=np.flatnonzero(row>0.1)
        indices=candidates[np.argsort(row[candidates],kind='stable')[-(CFG['gears']['top_k']+1):]]
        edges.extend((int(j),i) for j in indices);weights.extend(float(row[j]) for j in indices)
    np.savez(dest/'go.npz',edges=np.asarray(edges,dtype=np.int64).T,weights=np.asarray(weights,dtype=np.float32))
    # Training-only all-zero filters; never read VAL/TEST means into a TRAIN loss.
    means=np.load(expr/'group_log_means.npy',mmap_mode='r')
    filters={str(i):np.flatnonzero((means[i+1]>0)|(means[0]>0)).tolist() for i in np.flatnonzero(part==0)}
    write_json(dest/'training_nonzero_genes.json',filters)
    write_json(dest/'graphs.json',{'cell':cell,'seed':seed,'train_cells_including_control':len(rows),
        'coexpression_response_scope':'TRAIN targets and unperturbed controls only','a4_identity':ident,
        'go_public_source':CFG['gene2go'],'go_source_sha256':digest_file(CFG['gene2go']),
        'go_unannotated_targets_retained':int(sum(not s for s in sets)),
        'coexpression_sha256':digest_file(dest/'coexpression.npz'),'go_sha256':digest_file(dest/'go.npz')})
    torch.cuda.empty_cache()
    return dest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['graph','genept','expression','native_graphs'])
    p.add_argument('--cell');p.add_argument('--seed',type=int,default=17);a=p.parse_args()
    if a.stage=='graph': prepare_graph()
    elif a.stage=='genept': prepare_genept()
    elif a.stage=='expression': prepare_expression(a.cell)
    else: prepare_native_graphs(a.cell,a.seed)
