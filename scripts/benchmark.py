"""SB3 reference benchmark; one configuration per subprocess for honest resource isolation."""
import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


class EpisodeLog(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.file = path.open('w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['timesteps','episode_return','episode_length'])
        self.returns = []

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                e=info['episode']; self.returns.append(e['r'])
                self.writer.writerow([self.num_timesteps,e['r'],e['l']])
                self.file.flush()
        return True


def make_env():
    return Monitor(gym.make('HalfCheetah-v5'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--n-envs',type=int,required=True)
    p.add_argument('--vec',choices=['dummy','subproc'],default='dummy')
    p.add_argument('--steps',type=int,default=100000)
    a=p.parse_args()
    torch.set_num_threads(1)
    out=Path('results/benchmark')/f'{a.device}_n{a.n_envs}_{a.vec}'
    out.mkdir(parents=True,exist_ok=False)
    probe=gym.make('HalfCheetah-v5');probe.reset(seed=42);probe.action_space.seed(42)
    action=probe.action_space.sample()
    probe_start=time.perf_counter()
    for _ in range(10000):
        _,_,terminated,truncated,_=probe.step(action)
        if terminated or truncated:
            probe.reset()
    environment_fps=10000/(time.perf_counter()-probe_start);probe.close()
    env=(DummyVecEnv if a.vec=='dummy' else SubprocVecEnv)([make_env]*a.n_envs)
    env=VecNormalize(env,norm_obs=True,norm_reward=True,clip_obs=10)
    model=PPO('MlpPolicy',env,device=a.device,seed=42,n_steps=2048//a.n_envs,batch_size=64,n_epochs=10,learning_rate=3e-4,clip_range=.2,verbose=0)
    cb=EpisodeLog(out/'episodes.csv')
    resources=[]; stop=threading.Event(); process=psutil.Process(); children={}
    def monitor():
        while not stop.wait(1):
            row=dict(cpu_percent=process.cpu_percent(),system_cpu_percent=psutil.cpu_percent(),rss_mb=process.memory_info().rss/2**20)
            for child in process.children(recursive=True):
                children.setdefault(child.pid,child)
            row['cpu_tree_percent']=row['cpu_percent']+sum(child.cpu_percent() for child in children.values() if child.is_running())
            try:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
                row.update(gpu_percent=float(gpu[0]),gpu_memory_mb=float(gpu[1]))
            except Exception:
                pass
            resources.append(row)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    start=time.perf_counter()
    model.learn(total_timesteps=a.steps,callback=cb)
    elapsed=time.perf_counter()-start;stop.set();thread.join();cb.file.close()
    model.save(out/'reference.zip');env.save(out/'normalization.pkl')
    env.training=False;env.norm_reward=False
    env.seed(10000)
    from stable_baselines3.common.evaluation import evaluate_policy
    returns,lengths=evaluate_policy(model,env,n_eval_episodes=5,deterministic=True,return_episode_rewards=True)
    summary=dict(device=a.device,n_envs=a.n_envs,vec=a.vec,seed=42,requested_timesteps=a.steps,actual_timesteps=model.num_timesteps,wall_clock=elapsed,fps=model.num_timesteps/elapsed,environment_only_fps=environment_fps,training_return_last20=float(np.mean(cb.returns[-20:])),evaluation_returns=returns,evaluation_lengths=lengths,resource_mean={k:float(np.mean([r[k] for r in resources if k in r])) for k in resources[0]} if resources else {})
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'resources.json').write_text(json.dumps(resources))
    print(json.dumps(summary),flush=True);env.close()


if __name__=='__main__':
    main()
