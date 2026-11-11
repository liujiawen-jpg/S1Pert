"""One isolated real benchmark phase. Invoked by runtime.py, not a training stub."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from experiments.runtime import measure_phase
from experiments.paper_common import sha256,write_json


def execute(request,result):
    request,result=Path(request),Path(result)
    if result.exists():raise FileExistsError('Worker result already exists.')
    payload=json.loads(request.read_text())
    packages={}
    for name in ('numpy','scipy','torch','transformers','vllm'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    report=dict(status='RUNNING',request_sha256=sha256(request),
        environment=dict(python=sys.executable,version=sys.version,pid=os.getpid(),platform=platform.platform(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            pythonhashseed=os.environ.get('PYTHONHASHSEED'),packages=packages))
    try:
        report['measurement']=measure_phase(payload['phase'],Path(payload['work_dir']),Path(payload['log_dir']))
        report['status']='PASS'
        write_json(result,report)
    except Exception as error:
        report.update(status='FAILED',error=repr(error))
        write_json(result,report)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args();execute(args.request,args.result)
