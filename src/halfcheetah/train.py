"""Train custom PPO with raw-return logging and independent frozen-normalizer evaluation."""
import argparse
import csv
import importlib.metadata
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from .ppo import ActorCritic, RunningMeanStd, clipped_objective, gae


def evaluate(model, rms, seeds, device='cpu', video=None):
    env = gym.make('HalfCheetah-v5', render_mode='rgb_array' if video else None)
    writer = None
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video, fps=env.metadata['render_fps'])
    results = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            total = 0.0
            for length in range(1, 1001):
                with torch.no_grad():
                    action = model.actor(torch.as_tensor(rms.normalize(obs), device=device)).cpu().numpy()
                obs, reward, terminated, truncated, _ = env.step(np.clip(action, -1, 1))
                total += reward
                if writer:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            results.append(dict(seed=int(seed), episode_return=float(total), episode_length=length))
    finally:
        if writer:
            writer.close()
        env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/baseline.yaml')
    p.add_argument('--seed', type=int)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--clip-range', type=float)
    p.add_argument('--total-timesteps', type=int)
    p.add_argument('--name')
    args = p.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    for key in ['seed', 'learning_rate', 'clip_range', 'total_timesteps']:
        if getattr(args, key) is not None:
            c[key] = getattr(args, key)
    name = args.name or f"hc_lr{c['learning_rate']:g}_clip{c['clip_range']:g}_seed{c['seed']}"
    out = Path('results') / name
    out.mkdir(parents=True, exist_ok=False)  # Preserve all previous experiments.
    (out / 'config.yaml').write_text(yaml.safe_dump(c))
    def git(*argv):
        return subprocess.run(['git', *argv], capture_output=True, text=True).stdout.strip()
    metadata = dict(experiment_name=name, git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')), timestamp=datetime.now(timezone.utc).isoformat(), python=__import__('platform').python_version(), versions={x: importlib.metadata.version(x) for x in ['torch', 'gymnasium', 'mujoco', 'numpy', 'stable-baselines3']}, **c)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.set_num_threads(c['torch_threads'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(c['device'])
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'], eps=1e-5)
    envs = [gym.make(c['environment']) for _ in range(c['n_envs'])]
    obs = np.array([env.reset(seed=c['seed'] + i)[0] for i, env in enumerate(envs)])
    obs_rms, ret_rms = RunningMeanStd((17,)), RunningMeanStd()
    obs_rms.update(obs)
    discounted = np.zeros(c['n_envs'])
    episode_returns = np.zeros(c['n_envs']); episode_lengths = np.zeros(c['n_envs'], dtype=int)
    shape = (c['n_steps'], c['n_envs'])
    batch_size = int(np.prod(shape))
    steps = 0; start = time.perf_counter()
    train_file = (out / 'metrics.csv').open('w', newline='')
    eval_file = (out / 'eval.csv').open('w', newline='')
    episode_file = (out / 'episodes.csv').open('w', newline='')
    train_writer = csv.DictWriter(train_file, fieldnames=['timesteps','wall_clock','fps','training_return_mean','policy_loss','value_loss','entropy','approx_kl','clip_fraction'])
    eval_writer = csv.DictWriter(eval_file, fieldnames=['timesteps','seed','episode_return','episode_length'])
    episode_writer = csv.writer(episode_file); episode_writer.writerow(['timesteps','episode_return','episode_length'])
    train_writer.writeheader(); eval_writer.writeheader()
    recent = []

    def run_eval():
        rows = evaluate(model, obs_rms, range(10000, 10000+c['eval_episodes']), device)
        for row in rows:
            eval_writer.writerow(dict(timesteps=steps, **row))
        eval_file.flush()
        print(json.dumps(dict(timesteps=steps, eval_mean=np.mean([r['episode_return'] for r in rows]), elapsed=time.perf_counter()-start)), flush=True)

    run_eval()
    next_eval = c['eval_interval']
    try:
        while steps < c['total_timesteps']:
            observations = np.empty((*shape,17), np.float32)
            actions = np.empty((*shape,6), np.float32)
            log_probs = np.empty(shape, np.float32)
            values = np.empty(shape, np.float32)
            rewards = np.empty(shape, np.float32)
            next_values = np.empty(shape, np.float32)
            ends = np.empty(shape, np.float32)
            for t in range(c['n_steps']):
                normalized = obs_rms.normalize(obs)
                observations[t] = normalized
                with torch.no_grad():
                    action, log_prob, _, value = model.policy(torch.as_tensor(normalized, device=device))
                raw_actions = action.cpu().numpy()
                actions[t], log_probs[t], values[t] = raw_actions, log_prob.cpu().numpy(), value.cpu().numpy()
                transitions = [env.step(np.clip(a, -1, 1)) for env,a in zip(envs,raw_actions)]
                next_obs = np.array([x[0] for x in transitions])
                raw_reward = np.array([x[1] for x in transitions])
                terminated = np.array([x[2] for x in transitions])
                done = np.array([x[2] or x[3] for x in transitions])
                if not np.isfinite(next_obs).all() or not np.isfinite(raw_reward).all():
                    raise FloatingPointError('Non-finite environment transition')
                # Reward scaling uses discounted-return variance; raw rewards remain in logs.
                discounted = c['gamma'] * discounted + raw_reward
                ret_rms.update(discounted)
                rewards[t] = np.clip(raw_reward / np.sqrt(ret_rms.var + 1e-8), -10, 10)
                with torch.no_grad():
                    next_values[t] = model.value(torch.as_tensor(obs_rms.normalize(next_obs), device=device)).cpu().numpy() * (~terminated)
                ends[t] = done
                steps += c['n_envs']
                episode_returns += raw_reward; episode_lengths += 1
                for i in np.flatnonzero(done):
                    episode_writer.writerow([steps, episode_returns[i], episode_lengths[i]])
                    recent.append(episode_returns[i]); recent = recent[-100:]
                    episode_returns[i] = 0; episode_lengths[i] = 0; discounted[i] = 0
                    next_obs[i] = envs[i].reset()[0]
                obs = next_obs
                obs_rms.update(obs)
            advantages, returns = gae(rewards, values, next_values, ends, c['gamma'], c['gae_lambda'])
            arrays = [observations.reshape(-1,17), actions.reshape(-1,6), log_probs.ravel(), advantages.ravel(), returns.ravel()]
            b_obs,b_actions,b_logp,b_adv,b_returns = [torch.as_tensor(x, device=device) for x in arrays]
            stats = []
            for _ in range(c['n_epochs']):
                indices = np.random.permutation(batch_size)
                for j in range(0,batch_size,c['minibatch_size']):
                    ix = indices[j:j+c['minibatch_size']]
                    _,new_logp,entropy,new_value = model.policy(b_obs[ix],b_actions[ix])
                    adv = b_adv[ix]
                    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
                    log_ratio = new_logp-b_logp[ix]
                    policy_loss = -clipped_objective(log_ratio,adv,c['clip_range']).mean()
                    value_loss = 0.5*((new_value-b_returns[ix])**2).mean()
                    loss = policy_loss+c['value_coef']*value_loss-c['entropy_coef']*entropy.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite PPO loss')
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm'])
                    optimizer.step()
                    with torch.no_grad():
                        stats.append([policy_loss.item(),value_loss.item(),entropy.mean().item(),((log_ratio.exp()-1)-log_ratio).mean().item(),((log_ratio.exp()-1).abs()>c['clip_range']).float().mean().item()])
            elapsed = time.perf_counter()-start
            averages = np.mean(stats,axis=0)
            train_writer.writerow(dict(timesteps=steps,wall_clock=elapsed,fps=steps/elapsed,training_return_mean=np.mean(recent) if recent else float('nan'),**dict(zip(['policy_loss','value_loss','entropy','approx_kl','clip_fraction'],averages))))
            train_file.flush(); episode_file.flush()
            if steps >= next_eval or steps >= c['total_timesteps']:
                run_eval(); next_eval += c['eval_interval']
                # Includes frozen statistics and optimizer for audit/reuse; excluded from Git.
                torch.save(dict(model=model.state_dict(),obs_rms=obs_rms.state_dict(),ret_rms=ret_rms.state_dict(),optimizer=optimizer.state_dict(),config=c,timesteps=steps),out/'checkpoint.pt')
        metadata.update(wall_clock_time=time.perf_counter()-start,actual_timesteps=steps,status='completed')
    except BaseException as exc:
        metadata.update(wall_clock_time=time.perf_counter()-start,actual_timesteps=steps,status='failed',error=repr(exc))
        raise
    finally:
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
        for env in envs:
            env.close()
        for f in [train_file,eval_file,episode_file]:
            f.close()


if __name__ == '__main__':
    main()
