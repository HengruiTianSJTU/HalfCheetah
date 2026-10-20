"""Select a representative baseline, perform final held-out testing, render and report."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def main():
    baseline=[]
    for directory in sorted(Path('results').glob('hc_lr*')):
        c=yaml.safe_load((directory/'config.yaml').read_text())
        if c['learning_rate']!=3e-4 or c['clip_range']!=.2:
            continue
        metadata=json.loads((directory/'metadata.json').read_text())
        if metadata.get('status')!='completed':
            raise RuntimeError('Baseline is unfinished')
        with (directory/'eval.csv').open() as f:
            rows=list(csv.DictReader(f))
        final=max(int(r['timesteps']) for r in rows)
        mean=np.mean([float(r['episode_return']) for r in rows if int(r['timesteps'])==final])
        baseline.append((float(mean),directory))
    if len(baseline)!=3:
        raise RuntimeError('Expected all three baseline training seeds')
    baseline.sort(key=lambda x:x[0]);validation_mean,directory=baseline[len(baseline)//2]
    checkpoint=directory/'checkpoint.pt'
    subprocess.run([sys.executable,'scripts/evaluate.py',str(checkpoint),'--episodes','20','--seed','20000','--output','results/final_evaluation.csv'],check=True)
    subprocess.run([sys.executable,'scripts/evaluate.py',str(checkpoint),'--episodes','1','--seed','20000','--video','results/videos/halfcheetah_running.mp4','--output','results/video_evaluation.csv'],check=True)
    with Path('results/final_evaluation.csv').open() as f:
        rows=list(csv.DictReader(f))
    returns=[float(r['episode_return']) for r in rows]
    summary=dict(experiment_name=directory.name,selection_rule='Median final-validation baseline seed; fixed independent test seeds 20000..20019',validation_mean=validation_mean,episodes=len(rows),mean=float(np.mean(returns)),std=float(np.std(returns,ddof=1)),mean_episode_length=float(np.mean([int(r['episode_length']) for r in rows])),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),video_seed=20000,video_sha256=hashlib.sha256(Path('results/videos/halfcheetah_running.mp4').read_bytes()).hexdigest())
    Path('results/final_evaluation_summary.json').write_text(json.dumps(summary,indent=2))
    subprocess.run([sys.executable,'scripts/report.py'],check=True)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
