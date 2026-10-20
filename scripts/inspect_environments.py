"""Record HalfCheetah/Hopper dimensions and installed simulator defaults, without training Hopper."""
import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym

result={'gymnasium':importlib.metadata.version('gymnasium'),'mujoco':importlib.metadata.version('mujoco'),'environments':[]}
for name in ['HalfCheetah-v5','Hopper-v5']:
    env=gym.make(name);base=env.unwrapped
    obs,_=env.reset(seed=42)
    settings={key:str(getattr(base,key,None)) for key in ['_forward_reward_weight','_ctrl_cost_weight','_healthy_reward','_terminate_when_unhealthy','_healthy_state_range','_healthy_z_range','_healthy_angle_range','_exclude_current_positions_from_observation']}
    result['environments'].append(dict(environment=name,observation_dim=len(obs),action_dim=env.action_space.shape[0],action_low=env.action_space.low.tolist(),action_high=env.action_space.high.tolist(),nq=base.model.nq,nv=base.model.nv,nu=base.model.nu,dt=base.dt,frame_skip=base.frame_skip,max_episode_steps=env.spec.max_episode_steps,settings=settings))
    env.close()
Path('results/infrastructure/environments.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
