"""Audit the genuinely poor checkpoint captured while baseline seed44 kept training."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');a=p.parse_args()
    out=Path('results/failures/natural_baseline_seed44');checkpoint=out/'checkpoint.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    source=Path('results/hc_lr0.0003_clip0.2_seed44');step=state['timesteps']
    metrics=[r for r in read(source/'metrics.csv') if int(r['timesteps'])<=step]
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    evaluation=read(out/'eval.csv');values=[float(r['episode_return']) for r in evaluation]
    reference=[r for r in read(source/'eval.csv') if int(r['timesteps'])==step]
    assert len(reference)==len(evaluation)==5
    np.testing.assert_allclose(values,[float(r['episode_return']) for r in reference],rtol=0,atol=1e-7)
    assert np.mean(values)<0,'Snapshot must remain an actually poor policy, not just a failure label'
    source_metadata=json.loads((source/'metadata.json').read_text())
    final_mean=None
    if source_metadata.get('status')=='completed':
        rows=read(source/'eval.csv');last=max(int(r['timesteps']) for r in rows)
        final_mean=float(np.mean([float(r['episode_return']) for r in rows if int(r['timesteps'])==last]))
    result=dict(source_experiment=source.name,timesteps=step,captured_at=datetime.fromtimestamp(checkpoint.stat().st_mtime,timezone.utc).isoformat(),evaluation_mean=float(np.mean(values)),evaluation_std=float(np.std(values,ddof=1)),evaluation_episodes=len(values),training_return_at_capture=float(metrics[-1]['training_return_mean']),source_final_validation_mean=final_mean,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),purpose='Observed transient deterministic-policy collapse. The original run continues; this is a diagnostic checkpoint, not a separately trained full-budget experiment.')
    if a.render:
        subprocess.run([sys.executable,'scripts/evaluate.py',str(checkpoint),'--episodes','1','--seed','10000','--video','results/videos/natural_failure_seed44.mp4','--output',str(out/'video_eval.csv')],check=True)
        subprocess.run(['ffmpeg','-y','-loglevel','error','-ss','5','-i','results/videos/natural_failure_seed44.mp4','-frames:v','1','results/figures/failure_frame.png'],check=True)
        result.update(video='results/videos/natural_failure_seed44.mp4',poster='results/figures/failure_frame.png')
    (out/'snapshot.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
