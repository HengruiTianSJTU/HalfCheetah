"""Durable campaign driver: finish benchmarking, run the matrix and verify deliverables."""
import argparse
import json
import os
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait-benchmark-pid',type=int)
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    started=time.perf_counter();state=dict(started=datetime.now(timezone.utc).isoformat(),status='running',workers=a.workers)
    def save():
        Path('results/campaign_status.json').write_text(json.dumps(state,indent=2))
    def run(*args):
        state['phase']=args[0];save()
        subprocess.run([sys.executable,*args],check=True)
    try:
        if a.wait_benchmark_pid:
            state['phase']='waiting for reference benchmark';save()
            try:
                fd=os.pidfd_open(a.wait_benchmark_pid)
                select.select([fd],[],[]);os.close(fd)
            except ProcessLookupError:
                pass
        run('scripts/run_benchmarks.py')
        pilots=json.loads(Path('results/custom_benchmark.json').read_text())
        best=max(pilots,key=lambda r:r['steps_per_cpu_second'])
        if best['vec_backend']!='sync':
            raise RuntimeError('Measured CPU efficiency contradicts planned sync configuration; update saved config before formal runs')
        decision=dict(choice='CPU, 8 synchronous environments, 6 independent concurrent jobs',reason='Sync has higher transitions per total CPU-second despite async winning single-run wall time; parallel independent experiments exploit the 8-vCPU quota without 48 environment workers.',workers=a.workers,pilots=pilots,formal_timesteps=2000000,reference='results/benchmark/*/summary.json')
        Path('results/compute_decision.json').write_text(json.dumps(decision,indent=2))
        run('scripts/run_experiments.py','--workers',str(a.workers),'--timesteps','2000000')
        run('scripts/finalize.py')
        run('scripts/check_tutorial.py')
        run('scripts/qa_artifacts.py')
        state.update(status='artifacts_generated_and_checked',finished=datetime.now(timezone.utc).isoformat(),wall_clock=time.perf_counter()-started)
        save();print('CAMPAIGN_ARTIFACTS_READY',json.dumps(state),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),wall_clock=time.perf_counter()-started);save();raise


if __name__=='__main__':
    main()
