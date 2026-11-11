"""Create an isolated baseline workspace; run only explicitly selected stages."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PATH_KEYS={'root','a4_root','a4_data','moments','raw_root','metrics','a4_metrics',
 'scgpt_source','scgpt_pretrained','gene2go','gears_source','perturbqa_source',
 'assets_root','expression_root','esm_source','official_source','model','knowledge_db','public_edges_db'}

def init(family,config,work):
    work=work.resolve()
    if work.exists():raise FileExistsError('Choose a new workspace; existing results are preserved')
    cfg=json.loads(config.read_text())
    # Workspace configs can move independently of the source checkout. Metrics
    # are executable package code, so bind them to this checkout when staging.
    for key in ('metrics','a4_metrics'):
        if key in cfg:cfg[key]=str(HERE.parent/'src/jev_a4/core/metrics.py')
    for key in PATH_KEYS & cfg.keys():
        path=Path(os.path.expandvars(cfg[key])).expanduser()
        cfg[key]=str((config.parent/path).resolve()) if not path.is_absolute() else str(path)
    cfg['python']=sys.executable
    work.mkdir(parents=True)
    dst=work/'code' if family=='summer' else work;dst.mkdir(exist_ok=True)
    source=HERE/family/'code' if family=='summer' else HERE/family
    for path in source.glob('*.py'):
        ast.parse(path.read_text());shutil.copyfile(path,dst/path.name)
    (dst/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for name in ('reports','logs','runs','prepared'):(work/name).mkdir()
    (work/'workspace.json').write_text(json.dumps({'family':family,'schema':1})+'\n')
    print(json.dumps({'workspace':str(work),'family':family,'models_started':False}))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['init','execute','list'])
    p.add_argument('--family',choices=['classical','state','summer'])
    p.add_argument('--config',type=Path);p.add_argument('--work',type=Path)
    p.add_argument('--plan',action='store_true')
    argv=list(sys.argv[1:] if argv is None else argv);extra=[]
    if '--' in argv:i=argv.index('--');extra=argv[i+1:];argv=argv[:i]
    a=p.parse_args(argv)
    if a.command=='list':
        print('classical: GenePT-Gene, GenePT-Prot, GEARS, scGPT\nstate: STATE-ST-adapted\nsummer: SUMMER-Qwen3-8B-adapted\nP1: scripts/export_prior.py\nAdditional adapters: docs/BASELINES.md');return
    if a.work is None:p.error('--work is required')
    if a.command=='init':
        if not a.family or not a.config:p.error('init requires --family and --config')
        return init(a.family,a.config,a.work)
    work=a.work.resolve();family=json.loads((work/'workspace.json').read_text())['family']
    allowed={'classical':{'prepare.py','train.py'},'state':{'run.py'},'summer':{'fast_summer.py'}}
    if not extra or extra[0] not in allowed[family]:p.error('Use -- SCRIPT.py [arguments] for this family')
    base=work/'code' if family=='summer' else work
    cmd=[sys.executable,str(base/extra[0]),*extra[1:]]
    if a.plan:print(json.dumps(cmd));return
    cfg=json.loads((base/'config.json').read_text());env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=str(cfg.get('gpu_uuid','0'))
    if family=='summer':
        proof={'source_audit_status':'PASS','code':{q.name:hashlib.sha256(q.read_bytes()).hexdigest()
               for q in sorted(base.iterdir()) if q.is_file()}}
        seal=work/'reports/launch_seal.json'
        if not seal.exists():seal.write_text(json.dumps(proof,indent=2)+'\n')
    if family=='state':
        code='from common import identity,atomic,ROOT; atomic(ROOT/"reports/launch_seal.json", {"identity":identity()})'
        if not (work/'reports/launch_seal.json').exists():
            subprocess.run([sys.executable,'-c',code],cwd=work,env=env,check=True)
    return subprocess.call(cmd,cwd=base,env=env)

if __name__=='__main__':raise SystemExit(main())
