"""Check the selected GPU without stopping or modifying other jobs."""
from contextlib import contextmanager
import subprocess
from common import CFG

@contextmanager
def admitted(gpu):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    parsed=[tuple(v.strip() for v in line.split(',')) for line in rows.splitlines()]
    match=[x for x in parsed if str(gpu) in x[:2]]
    if len(match)!=1:raise RuntimeError('Specify one GPU index or UUID')
    if int(match[0][2])<(CFG['max_gpu_gib']+3)*1024:
        raise RuntimeError('Insufficient free GPU memory; other jobs were not changed')
    yield
