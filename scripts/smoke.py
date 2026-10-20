"""Seeded random-policy validation, optionally with headless video."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.steps < 1000:
        parser.error("smoke requires at least 1000 steps")
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array" if args.video else None)
    writer = None
    if args.video:
        import imageio.v2 as imageio
        Path("results/videos").mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer("results/videos/random_policy_smoke.mp4", fps=env.metadata["render_fps"])
    terminated_count = truncated_count = 0
    reward_sum = 0.0
    try:
        obs, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        assert obs.shape == (17,)
        assert env.action_space.shape == (6,)
        assert np.isfinite(obs).all()
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(obs).all() and np.isfinite(reward)
            reward_sum += reward
            terminated_count += int(terminated)
            truncated_count += int(truncated)
            if writer:
                frame = env.render()
                assert frame.ndim == 3 and frame.shape[2] == 3
                writer.append_data(frame)
            if terminated or truncated:
                obs, _ = env.reset()
                assert np.isfinite(obs).all()
        result = dict(seed=args.seed, steps=args.steps, observation=str(env.observation_space), action=str(env.action_space), terminated=terminated_count, truncated=truncated_count, cumulative_reward=reward_sum)
        Path("results/infrastructure").mkdir(parents=True, exist_ok=True)
        Path("results/infrastructure/smoke.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
