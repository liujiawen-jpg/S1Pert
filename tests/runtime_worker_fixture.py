"""CPU-only live callable used by isolated-runtime tests, never paper results."""
import os
from pathlib import Path
import sys
import time


def build(cleanup_path, fail=False):
    time.sleep(0.005)

    def run():
        time.sleep(0.003)
        if fail:
            raise RuntimeError('intentional isolated synthetic failure')
        return dict(status='PASS',pid=os.getpid(),python=sys.executable,
                    marker=os.environ.get('S1PERT_TEST_MARKER'))

    def cleanup():
        Path(cleanup_path).write_text('cleaned')

    return dict(run=run,synchronize=lambda:None,n_pairs=12,device='cpu',cleanup=cleanup,
                metadata=dict(synthetic_test_only=True))
