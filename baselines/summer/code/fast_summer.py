"""Finite training-free SUMMER queue. Native generations, not teacher labels.

Only the execution engine changes. Old complete predictions are independent
benchmark results, never inputs. Every new answer is generated directly from
public descriptions, controls and legal retrieved strict TRAIN observations.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse
import fcntl
import gc
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
import traceback
import numpy as np

from common import CFG, ROOT, Data, Retrieval, functions, metrics, objsha, sha, read, write, parse_answer
from run import Store, summer_prompt, summary, code_identity

STOP = False


def stop_requested(sig, frame):
    global STOP
    STOP = True


def guard():
    """Count reclaimable cache honestly; never purge global/other task caches."""
    import psutil
    process = psutil.Process()
    rss = sum(p.memory_info().rss for p in [process] + process.children(recursive=True)
              if p.is_running())
    if rss > CFG['max_rss_gib'] * (1 << 30):
        raise MemoryError(f'Own process-tree RSS exceeded {CFG["max_rss_gib"]} GiB')
    headroom = None
    cg = Path('/sys/fs/cgroup')
    if all((cg / name).exists() for name in ('memory.max','memory.stat','memory.current')):
        limit = (cg / 'memory.max').read_text().strip()
        stats = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
        current = int((cg / 'memory.current').read_text())
        reclaimable = int(stats.get('inactive_file', 0)) + int(stats.get('slab_reclaimable', 0))
        headroom = None if limit == 'max' else (int(limit) - current + reclaimable) / (1 << 30)
    if headroom is not None and headroom < 2:
        raise MemoryError(f'Effective cgroup headroom {headroom:.2f} GiB below 2 GiB; predictions committed for resume')
    return {'own_tree_rss_gib': rss / (1 << 30), 'effective_cgroup_headroom_gib': headroom,
            'reclaimable_components': ['inactive_file', 'slab_reclaimable']}


class FastEngine:
    def __init__(self):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        import vllm
        import torch
        import transformers
        self.SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(CFG['model'], local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        # Same model identity as the public summary cache. Backend identity is
        # bound separately for prediction resumability; never mix engines there.
        self.identity = {'path': CFG['model'], 'config_sha256': sha(Path(CFG['model'])/'config.json'),
            'index_sha256': sha(Path(CFG['model'])/'model.safetensors.index.json'),
            'model_type': 'qwen3', 'generic_only': True}
        config = read(Path(CFG['model'])/'config.json')
        assert config['model_type'] == 'qwen3' and config['hidden_size'] == 4096
        assert config['eos_token_id']==self.tokenizer.eos_token_id
        self.backend = {'name': 'vllm', 'version': vllm.__version__, 'dtype': 'bfloat16',
            'torch_version':torch.__version__, 'transformers_version':transformers.__version__,
            'prefix_caching': True, 'chunked_prefill': True,
            'max_num_seqs': CFG['engine_max_num_seqs'],
            'max_num_batched_tokens': CFG['engine_max_num_batched_tokens'],
            'eos_token_id': self.tokenizer.eos_token_id,
            'weight_loading': 'sequential eager per-shard safetensors read from NFS; no weight copy',
            'input_limit': CFG['max_input_tokens'], 'answer_limit': CFG['answer_tokens'],
            'decoding': 'greedy; temperature=0; top_p=1; no constrained labels',
            'attention_backend': os.environ.get('VLLM_ATTENTION_BACKEND', 'auto')}
        self.batch = CFG['engine_max_num_seqs']
        self.llm = LLM(model=CFG['model'], tokenizer=CFG['model'], dtype='bfloat16',
            tensor_parallel_size=1, trust_remote_code=False, seed=17,
            max_model_len=CFG['max_input_tokens']+CFG['answer_tokens'],
            gpu_memory_utilization=CFG['max_gpu_gib']*2**30/torch.cuda.get_device_properties(0).total_memory,
            max_num_seqs=CFG['engine_max_num_seqs'],
            max_num_batched_tokens=CFG['engine_max_num_batched_tokens'],
            enable_prefix_caching=True, enable_chunked_prefill=True,
            disable_log_stats=True, swap_space=0, cpu_offload_gb=0,
            generation_config='vllm', load_format='safetensors',
            safetensors_load_strategy='eager')

    def ids(self, text, system=None):
        messages = ([{'role': 'system', 'content': system}] if system else [])
        messages += [{'role': 'user', 'content': text}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if len(ids) > CFG['max_input_tokens']:
            raise ValueError(f'Input has {len(ids)} tokens above {CFG["max_input_tokens"]}; no truncation')
        return ids

    def generate(self, texts, max_tokens, *, system=None):
        if STOP:
            raise InterruptedError('Stopped by user at committed batch boundary')
        prompts = [{'prompt_token_ids': self.ids(text, system)} for text in texts]
        params = self.SamplingParams(temperature=0, top_p=1, max_tokens=max_tokens,
            repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
            ignore_eos=False, skip_special_tokens=True)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(texts):
            raise RuntimeError('Generation count mismatch')
        values = []
        for result in outputs:
            if len(result.outputs) != 1:
                raise RuntimeError('Unexpected number of generation candidates')
            out = result.outputs[0]
            values.append({'text': out.text, 'ended': out.finish_reason == 'stop',
                           'new_tokens': len(out.token_ids), 'finish_reason': out.finish_reason})
        return values


def check_store_identity(store, identity):
    old = store.get('__identity__')
    if old is not None and old != identity:
        raise ValueError('Prediction identity mismatch: code, input, summary or backend changed')
    store.put('__identity__', identity)
    store.commit()


def evaluate_full(engine, data, summaries, retriever, out):
    rows = data.records('test', final=True)
    test_file = data.seed_dir/'test/records.npy'
    identity = objsha({'data': data.identity(), 'test_sha256': sha(test_file),
        'code': code_identity(), 'model': engine.identity, 'backend': engine.backend,
        'knowledge_sha256': sha(CFG['knowledge_db']), 'public_edges_sha256': sha(CFG['public_edges_db']),
        'public_summaries': objsha(summaries), 'phase': 'test'})
    store = Store(out/'test.sqlite')
    check_store_identity(store, identity)
    pred = np.full(len(rows), -1, dtype=np.int8)
    started = time.monotonic()
    generated = reused = tokens = 0
    try:
        for lo in range(0, len(rows), CFG['generation_batch']):
            indices = list(range(lo, min(lo+CFG['generation_batch'], len(rows))))
            cached = {i: store.get(i) for i in indices}
            missing = [i for i in indices if cached[i] is None]
            if missing:
                prompts = [summer_prompt(data, rows[i], summaries, retriever, 'test') for i in missing]
                outputs = engine.generate(prompts, CFG['answer_tokens'])
                for i, value, prompt_text in zip(missing, outputs, prompts):
                    value['prediction'] = parse_answer(value['text']) if value['ended'] else -1
                    value['target'] = int(rows[i]['target'])
                    value['query'] = int(rows[i]['query'])
                    value['prompt_sha256'] = hashlib.sha256(prompt_text.encode()).hexdigest()
                    cached[i] = value
                    store.put(i, value)
                    tokens += value['new_tokens']
                store.commit()
                generated += len(missing)
            reused += len(indices)-len(missing)
            for i in indices:
                value = cached[i]
                if (value['target'],value['query']) != (int(rows[i]['target']),int(rows[i]['query'])):
                    raise ValueError('Cached target/query identity mismatch')
                pred[i] = value['prediction']
            elapsed = time.monotonic()-started
            resource = guard()
            write(out/'progress.json', {'phase': 'full_test', 'done': lo+len(indices),
                'total': len(rows), 'new_generations_this_invocation': generated,
                'cache_hits_this_invocation': reused, 'elapsed_seconds': elapsed,
                'new_pairs_per_second': generated/max(elapsed,1e-9),
                'new_output_tokens_per_second': tokens/max(elapsed,1e-9),
                'backend': engine.backend, 'resources': resource, 'updated_unix': time.time()})
            print(json.dumps({'cell':data.cell,'seed':data.seed,'done':lo+len(indices),
                'total':len(rows),'pairs_per_second':generated/max(elapsed,1e-9)}), flush=True)
            if STOP:
                raise InterruptedError('Stopped after committing resumable predictions')
    finally:
        store.close()
    np.savez_compressed(out/'test_predictions.npz', target=rows['target'], query=rows['query'],
                        labels=rows['label'], y_pred=pred, prediction=pred)
    result = metrics(rows, pred)
    result.update(elapsed_seconds_this_invocation=time.monotonic()-started,
                  new_generations_this_invocation=generated, reused_predictions_this_invocation=reused)
    write(out/'test_metrics.json', result)
    return result




def aggregate():
    result={}
    for cell in CFG['cells']:
        reports=[]
        for seed in CFG['seeds']:
            path=ROOT/f'runs/{cell}/seed_{seed}/report.json'
            if path.exists() and read(path)['status']=='PASS':reports.append(read(path))
        values={}
        for report in reports:
            m=report['test']
            flat={key:m[key] for key in ['macro_f1','balanced_accuracy','accuracy','invalid_rate','up_down_f1']}
            flat.update({f'{c}.{key}':m['per_class'][c][key] for c in ['up','same','down'] for key in ['precision','recall','f1']})
            for k,v in flat.items():values.setdefault(k,[]).append(v)
        result[cell]={'completed_seeds':[r['seed'] for r in reports],'expected_seeds':CFG['seeds'],
            'complete':len(reports)==len(CFG['seeds']),
            'metrics':{k:{'mean':float(np.mean(v)),
                'sample_variance':float(np.var(v,ddof=1)) if len(v)>1 else None,
                'sample_std':float(np.std(v,ddof=1)) if len(v)>1 else None} for k,v in values.items()}}
    write(ROOT/'reports/results.json',{'method':CFG['summer']['name'],'datasets':result,
        'seeds_are_repeated_holdouts_not_folds':True,
        'backend_disclosure':'All configured groups generated with the configured vLLM backend'})


def smoke(engine, public):
    """Only TRAIN, with entire target fold excluded from retrieved responses."""
    data=Data(CFG['cells'][0],CFG['seeds'][0]); retriever=Retrieval(data)
    summaries=summary(engine,data.t+data.q,public)
    rows=data.records('train')
    indices=np.linspace(0,len(rows)-1,128,dtype=int)
    prompts=[summer_prompt(data,rows[i],summaries,retriever,'train') for i in indices]
    # Input must not depend on this pair's response label.
    changed=rows[indices[0]].copy();changed['label']=(int(changed['label'])+1)%3
    assert prompts[0]==summer_prompt(data,changed,summaries,retriever,'train')
    engine.generate(prompts[:4],CFG['answer_tokens'])
    started=time.monotonic()
    values=engine.generate(prompts,CFG['answer_tokens'])
    elapsed=time.monotonic()-started
    valid=sum(v['ended'] and parse_answer(v['text'])>=0 for v in values)
    report={'status':'PASS' if valid>=len(values)*.8 else 'FAIL',
        'scope':'128 TRAIN pairs; current target fold excluded from retrieved labels',
        'test_opened_for_smoke':False,'count':len(values),'valid':valid,
        'elapsed_seconds':elapsed,'pairs_per_second':len(values)/elapsed,
        'tokens_per_second':sum(v['new_tokens'] for v in values)/elapsed,
        'max_input_tokens':max(map(lambda p:len(engine.ids(p)),prompts)),
        'backend':engine.backend,'resources':guard(),
        'examples':[{'text':v['text'],'ended':v['ended']} for v in values[:4]]}
    write(ROOT/'reports/gpu_smoke.json',report)
    if report['status']!='PASS':raise RuntimeError('Native generation/parser smoke failed; no formal run')
    print(json.dumps({'gpu_smoke':report['status'],'pairs_per_second':report['pairs_per_second']}),flush=True)
    retriever.close.cache_clear()
    del retriever,data;gc.collect()


def preflight():
    jobs=[]
    for cell in CFG['cells']:
        for seed in CFG['seeds']:
            data=Data(cell,seed)
            for part in ['train','val']:
                a=data.records(part)
                assert len(a)==data.manifest['counts'][part]['strict_pairs']
            jobs.append({'cell':cell,'seed':seed,'test_expected':data.manifest['counts']['test']['strict_pairs'],
                         'identity':data.identity()})
    write(ROOT/'reports/preflight.json',{'status':'PASS','jobs':jobs,'new_test_labels_read':False})


def pipeline():
    started=time.time()
    jobs=[{'cell':c,'seed':s,'status':'PENDING'} for s in CFG['seeds'] for c in CFG['cells']]
    status={'status':'RUNNING','pid':os.getpid(),'gpu':CFG['gpu_uuid'],'jobs':jobs,
            'started_unix':started,'continuous_monitoring':False}
    write(ROOT/'status.json',status)
    try:
        seal=read(ROOT/'reports/launch_seal.json')
        if seal['source_audit_status']!='PASS' or seal['code']!=code_identity():
            raise RuntimeError('Code changed after workspace creation; create a new workspace')
        preflight()
        aggregate()
        write(ROOT/'status.json',status)
        from gpu_admission import admitted
        with admitted(CFG['gpu_uuid']):
            guard()
            engine=FastEngine()
            public=functions()
            smoke(engine,public)
            for job in jobs:
                if STOP:raise InterruptedError('Stopped by user')
                out=ROOT/f'runs/{job["cell"]}/seed_{job["seed"]}';out.mkdir(parents=True,exist_ok=True)
                if (out/'report.json').exists() and read(out/'report.json')['status']=='PASS':
                    previous=read(out/'report.json');current_data=Data(job['cell'],job['seed'])
                    expected={'code':code_identity(),'data':current_data.identity(),
                              'generic_model':engine.identity,'backend':engine.backend,
                              'knowledge_sha256':sha(CFG['knowledge_db']),
                              'public_edges_sha256':sha(CFG['public_edges_db']),
                              'test_records_sha256':sha(current_data.seed_dir/'test/records.npy')}
                    if any(previous.get(key)!=value for key,value in expected.items()):
                        raise ValueError('Completed SUMMER result identity changed; use a new workspace')
                    if previous.get('prediction_sha256')!=sha(out/'test_predictions.npz'):
                        raise ValueError('Completed SUMMER prediction integrity mismatch')
                    job['status']='PASS_REUSED_VERIFIED';continue
                job.update(status='RUNNING',started_unix=time.time())
                status['current_job']=job;write(ROOT/'status.json',status)
                data=Data(job['cell'],job['seed'])
                summaries=summary(engine,data.t+data.q,public)
                retriever=Retrieval(data)
                write(out/'selection.json',{'training':False,'method':'locked training-free configuration',
                    'no_test_tuning':True,'full_test':True,'backend':engine.backend})
                result=evaluate_full(engine,data,summaries,retriever,out)
                write(out/'report.json',{'status':'PASS','cell':job['cell'],'seed':job['seed'],
                    'method':CFG['summer']['name'],'test':result,'training':False,
                    'backend':engine.backend,'generic_model':engine.identity,'code':code_identity(),
                    'data':data.identity(),'adaptation':CFG['summer']['adaptation'],
                    'knowledge_sha256':sha(CFG['knowledge_db']),
                    'public_edges_sha256':sha(CFG['public_edges_db']),
                    'test_records_sha256':sha(data.seed_dir/'test/records.npy'),
                    'prediction_sha256':sha(out/'test_predictions.npz'),
                    'scientific_status':'RESULTS_AVAILABLE_NOT_CLAIM_OF_GAIN'})
                job.update(status='PASS',ended_unix=time.time())
                aggregate();write(ROOT/'status.json',status)
                retriever.close.cache_clear()
                del retriever,summaries,data;gc.collect()
        status['status']='PASS'
    except BaseException as exc:
        status['status']='STOPPED_BY_USER' if STOP else 'PARTIAL'
        status['error']=repr(exc);status['traceback']=traceback.format_exc()
        if status.get('current_job',{}).get('status')=='RUNNING':
            status['current_job']['status']='STOPPED_BY_USER' if STOP else 'FAILED'
        traceback.print_exc()
        raise
    finally:
        status['ended_unix']=time.time();write(ROOT/'status.json',status);aggregate()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['pipeline','preflight'])
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'pipeline':pipeline,'preflight':preflight}[args.action]()


if __name__=='__main__':main()
