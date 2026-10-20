"""Small PPO implementation: Gaussian actor, critic, GAE, clipped policy updates."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        mean, var, count = x.mean(axis=0), x.var(axis=0), len(x)
        delta = mean - self.mean
        total = self.count + count
        self.var = (self.var * self.count + var * count + delta**2 * self.count * count / total) / total
        self.mean += delta * count / total
        self.count = total

    def normalize(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load_state_dict(self, state):
        self.mean, self.var, self.count = state['mean'], state['var'], state['count']


def layer(in_features, out_features, gain=np.sqrt(2)):
    linear = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.zeros_(linear.bias)
    return linear


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=17, action_dim=6):
        super().__init__()
        self.actor = nn.Sequential(layer(obs_dim, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, action_dim, 0.01))
        self.critic = nn.Sequential(layer(obs_dim, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, 1, 1))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def policy(self, obs, action=None):
        dist = Normal(self.actor(obs), self.log_std.exp())
        action = dist.sample() if action is None else action
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(obs)


def gae(rewards, values, next_values, ends, gamma=0.99, lam=0.95):
    """next_values is zero ONLY for true termination, never time-limit truncation.

    ends stops recursion across either reset boundary. Thus a time-limit transition
    bootstraps its final observation but cannot leak the reset episode's advantage.
    """
    advantages = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        carry = delta + gamma * lam * (1 - ends[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def clipped_objective(log_ratio, advantages, clip_range):
    ratio = log_ratio.exp()
    return torch.minimum(ratio * advantages, ratio.clamp(1 - clip_range, 1 + clip_range) * advantages)
