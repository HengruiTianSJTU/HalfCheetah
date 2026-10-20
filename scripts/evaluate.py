"""Evaluate a trusted local custom PPO checkpoint on held-out seeds."""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','egl')
import torch
from halfcheetah.ppo import ActorCritic, RunningMeanStd
from halfcheetah.train import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed',type=int,default=20000)
    p.add_argument('--video')
    p.add_argument('--output',default='results/final_evaluation.csv')
    a=p.parse_args();torch.set_num_threads(1)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=ActorCritic();model.load_state_dict(checkpoint['model']);model.eval()
    rms=RunningMeanStd((17,));rms.load_state_dict(checkpoint['obs_rms'])
    rows=evaluate(model,rms,range(a.seed,a.seed+a.episodes),video=a.video)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['seed','episode_return','episode_length']);w.writeheader();w.writerows(rows)
    import numpy as np
    print(dict(mean=np.mean([r['episode_return'] for r in rows]),std=np.std([r['episode_return'] for r in rows],ddof=1) if len(rows)>1 else 0,episodes=len(rows)))


if __name__=='__main__':
    main()
