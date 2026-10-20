"""Predeclared noise check: expand a noisy normal group from 3 to 5 training seeds."""
import concurrent.futures
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def main():
    groups=[(lr,.2) for lr in [1e-4,3e-4,1e-3]]+[(3e-4,clip) for clip in [.1,.3]]
    assessments=[];matrix=[]
    for lr,clip in groups:
        means=[]
        for seed in [42,43,44]:
            directory=Path('results')/f'hc_lr{lr:g}_clip{clip:g}_seed{seed}'
            metadata=json.loads((directory/'metadata.json').read_text())
            if metadata.get('status')!='completed':
                raise RuntimeError(f'Initial matrix is incomplete: {directory}')
            with (directory/'eval.csv').open() as f:
                rows=list(csv.DictReader(f))
            t=max(int(r['timesteps']) for r in rows)
            means.append(float(np.mean([float(r['episode_return']) for r in rows if int(r['timesteps'])==t])))
        cv=float(np.std(means,ddof=1)/max(abs(np.mean(means)),1.))
        trigger=cv>.30
        assessments.append(dict(learning_rate=lr,clip_range=clip,initial_seeds=[42,43,44],final_validation_means=means,coefficient_of_variation=cv,expand_to_five=trigger))
        if trigger:
            matrix.extend(dict(learning_rate=lr,clip_range=clip,seed=seed) for seed in [45,46])
    # Two async jobs fit the CPU quota and preserve ordered rollout math. With
    # more jobs sync is more CPU-efficient, as established by the 100k pilots.
    backend='async' if len(matrix)==2 else 'sync'
    protocol=json.loads(Path('results/experiment_plan.json').read_text())
    plan=dict(timestamp=datetime.now(timezone.utc).isoformat(),rule='Use only the initial 3 seeds: expand any normal group with sample SD / max(abs(mean),1) > 0.30 to seeds 42..46. Do not use final test outcomes.',assessments=assessments,matrix=matrix,total_timesteps=protocol['total_timesteps'],vec_backend=backend)
    path=Path('results/extension_plan.json')
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['matrix']!=matrix or previous['total_timesteps']!=plan['total_timesteps']:
            raise RuntimeError('Existing noise-extension plan differs; preserve and investigate')
    else:
        path.write_text(json.dumps(plan,indent=2))
    print('VARIABILITY_ASSESSMENT',json.dumps(assessments),flush=True)
    start=time.perf_counter()
    def run(case):
        lr,clip,seed=case['learning_rate'],case['clip_range'],case['seed']
        name=f'hc_lr{lr:g}_clip{clip:g}_seed{seed}';directory=Path('results')/name
        if directory.exists():
            m=json.loads((directory/'metadata.json').read_text())
            if m.get('status')=='completed':
                return
            raise RuntimeError(f'Preserve incomplete run: {directory}')
        print('START_EXTENSION',name,backend,flush=True)
        with (Path('raw_logs')/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-m','halfcheetah.train','--config','configs/baseline.yaml','--learning-rate',str(lr),'--clip-range',str(clip),'--seed',str(seed),'--total-timesteps',str(plan['total_timesteps']),'--vec-backend',backend],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE_EXTENSION',name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for future in concurrent.futures.as_completed([executor.submit(run,case) for case in matrix]):
            future.result()
    timing=Path('results/extension_timing.json')
    if not timing.exists():
        timing.write_text(json.dumps(dict(n_experiments=len(matrix),wall_clock=time.perf_counter()-start,vec_backend=backend),indent=2))


if __name__=='__main__':
    main()
