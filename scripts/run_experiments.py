"""A fixed, resumable experiment matrix; launch inside tmux on AutoDL."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--timesteps',type=int,default=2000000)
    p.add_argument('--config',default='configs/baseline.yaml')
    a=p.parse_args()
    matrix=[(lr,.2,s) for lr in [1e-4,3e-4,1e-3] for s in [42,43,44]]+[(3e-4,clip,s) for clip in [.1,.3] for s in [42,43,44]]+[(.03,.2,42)]
    plan=dict(timestamp=datetime.now(timezone.utc).isoformat(),workers=a.workers,total_timesteps=a.timesteps,matrix=[dict(learning_rate=lr,clip_range=clip,seed=seed) for lr,clip,seed in matrix],baseline_selection='Start with the median baseline final-curve model, then apply scripts/select_model.py stability validation on independent selection seeds before final testing.',final_test_seeds=list(range(30000,30020)))
    Path('results').mkdir(exist_ok=True)
    plan_path=Path('results/experiment_plan.json')
    if plan_path.exists():
        previous=json.loads(plan_path.read_text())
        if previous['total_timesteps']!=a.timesteps or previous['matrix']!=plan['matrix']:
            raise RuntimeError('Existing experiment plan differs; preserve it and use a separate checkout')
    else:
        plan_path.write_text(json.dumps(plan,indent=2))
    Path('raw_logs').mkdir(exist_ok=True)
    start=time.perf_counter()

    def run(case):
        lr,clip,seed=case;name=f'hc_lr{lr:g}_clip{clip:g}_seed{seed}'
        out=Path('results')/name
        if out.exists():
            metadata=json.loads((out/'metadata.json').read_text())
            if metadata.get('status')=='completed':
                print('SKIP COMPLETED',name,flush=True);return name
            raise RuntimeError(f'{name} is incomplete; preserve and investigate before rerunning')
        print('START',name,flush=True)
        command=[sys.executable,'-m','halfcheetah.train','--config',a.config,'--name',name,'--learning-rate',str(lr),'--clip-range',str(clip),'--seed',str(seed),'--total-timesteps',str(a.timesteps)]
        with (Path('raw_logs')/(name+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',name,flush=True)
        return name

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures=[executor.submit(run,case) for case in matrix]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    summary=dict(started=plan['timestamp'],finished=datetime.now(timezone.utc).isoformat(),matrix_wall_clock=time.perf_counter()-start,workers=a.workers,n_experiments=len(matrix))
    Path('results/suite_timing.json').write_text(json.dumps(summary,indent=2))
    print('EXPERIMENT_MATRIX_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
