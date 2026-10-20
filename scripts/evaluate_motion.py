"""Independent audit of actual forward displacement and raw reward decomposition."""
import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from halfcheetah.ppo import ActorCritic, RunningMeanStd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed',type=int,default=20000);p.add_argument('--output',default='results/final_motion.csv')
    a=p.parse_args();torch.set_num_threads(1)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=ActorCritic();model.load_state_dict(state['model']);model.eval()
    rms=RunningMeanStd((17,));rms.load_state_dict(state['obs_rms'])
    env=gym.make('HalfCheetah-v5');rows=[]
    try:
        for seed in range(a.seed,a.seed+a.episodes):
            obs,_=env.reset(seed=seed);start_x=float(env.unwrapped.data.qpos[0])
            reward_sum=forward_sum=control_sum=saturation=0.
            for length in range(1,1001):
                with torch.no_grad():
                    action=model.actor(torch.as_tensor(rms.normalize(obs))).numpy()
                saturation+=float(np.mean(np.abs(action)>1))
                obs,reward,terminated,truncated,info=env.step(np.clip(action,-1,1))
                # NumPy 2 scalar promotion can keep a running sum in float32.
                # Explicit Python floats give float64 accumulation for the audit.
                reward_sum+=float(reward);forward_sum+=float(info['reward_forward']);control_sum+=float(info['reward_ctrl'])
                if terminated or truncated:
                    break
            displacement=float(env.unwrapped.data.qpos[0])-start_x
            np.testing.assert_allclose(reward_sum,forward_sum+control_sum,rtol=1e-10,atol=1e-8)
            np.testing.assert_allclose(displacement/env.unwrapped.dt,forward_sum,rtol=1e-10,atol=1e-7)
            rows.append(dict(seed=seed,episode_return=reward_sum,episode_length=length,x_displacement_m=displacement,mean_forward_velocity_m_s=displacement/(length*env.unwrapped.dt),forward_reward_sum=forward_sum,control_reward_sum=control_sum,action_saturation_fraction=saturation/length))
    finally:
        env.close()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(dict(episodes=len(rows),mean_return=np.mean([r['episode_return'] for r in rows]),mean_forward_velocity_m_s=np.mean([r['mean_forward_velocity_m_s'] for r in rows])))


if __name__=='__main__':
    main()
