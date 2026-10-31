"""Render a trusted checkpoint with an extended visual floor and verify unchanged dynamics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from halfcheetah.ppo import ActorCritic, RunningMeanStd
from halfcheetah.train import evaluate


def scene(path, width, height):
    env = gym.make('HalfCheetah-v5')
    original = Path(env.unwrapped.fullpath)
    env.close()
    tree = ET.parse(original)
    root = tree.getroot()
    root.find('.//geom[@name="floor"]').set('size', '2000 2000 40')
    material = root.find('.//material[@name="MatPlane"]')
    for key, value in dict(texrepeat='3000 3000', reflectance='0.08', shininess='0.15', specular='0.15').items():
        material.set(key, value)
    tex = root.find('.//texture[@name="texplane"]')
    tex.set('rgb1', '0.18 0.24 0.32'); tex.set('rgb2', '0.31 0.39 0.48')
    sky = root.find('.//texture[@type="skybox"]')
    sky.set('rgb1', '0.35 0.44 0.54'); sky.set('rgb2', '0.08 0.12 0.18')
    visual = root.find('visual')
    if visual is None: visual = ET.SubElement(root, 'visual')
    global_ = visual.find('global')
    if global_ is None: global_ = ET.SubElement(visual, 'global')
    global_.set('offwidth', str(width)); global_.set('offheight', str(height))
    stat = root.find('statistic')
    if stat is None: stat = ET.SubElement(root, 'statistic')
    stat.set('extent', '8')
    tree.write(path, encoding='unicode')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=540)
    a = p.parse_args()
    if a.output.exists(): p.error(f'Existing video preserved: {a.output}')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = ActorCritic(); model.load_state_dict(state['model']); model.eval()
    rms = RunningMeanStd((17,)); rms.load_state_dict(state['obs_rms'])
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='halfcheetah-scene-') as temp:
        xml = Path(temp)/'scene.xml'; scene(xml, a.width, a.height)
        env = gym.make('HalfCheetah-v5', xml_file=str(xml), width=a.width, height=a.height)
        obs, _ = env.reset(seed=a.seed)
        u = env.unwrapped
        renderer = mujoco.Renderer(u.model, height=a.height, width=a.width)
        writer = imageio.get_writer(a.output, fps=20, codec='libx264', quality=None, macro_block_size=1,
                                    ffmpeg_params=['-crf','19','-preset','fast','-pix_fmt','yuv420p',
                                                   '-movflags','+faststart','-threads','2'])
        total, start_x, maximum_x = 0., float(u.data.qpos[0]), 0.
        try:
            for step in range(1000):
                with torch.no_grad():
                    action = model.actor(torch.as_tensor(rms.normalize(obs))).numpy()
                obs, reward, terminated, truncated, _ = env.step(np.clip(action, -1, 1))
                total += float(reward)
                maximum_x = max(maximum_x, abs(float(u.data.qpos[0])))
                renderer.update_scene(u.data, camera='track')
                frame = renderer.render(); writer.append_data(frame)
                if step == 999:
                    imageio.imwrite(a.output.with_suffix('.jpg'), frame)
                if terminated or truncated:
                    assert step == 999
                    break
            displacement = float(u.data.qpos[0])-start_x
        finally:
            writer.close(); renderer.close(); env.close()
    expected = evaluate(model, rms, [a.seed], device='cpu')[0]['episode_return']
    assert abs(total-expected) < 1e-6 and maximum_x < 1900
    config = state['config']
    result = dict(run=a.checkpoint.parent.name, training_seed=config['seed'], demo_seed=a.seed,
                  learning_rate=config['learning_rate'], clip_range=config['clip_range'],
                  entropy_coef=config['entropy_coef'], training_device=config['device'], training_hardware='RTX 4090',
                  inference_device='cpu', frames=1000, fps=20, duration=50, width=a.width, height=a.height,
                  episode_return=total, original_return=expected, return_difference=total-expected,
                  displacement=displacement, mean_velocity=displacement/50, floor_half_extent_m=2000,
                  ground_clearance_margin=2000-maximum_x,
                  checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                  video_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                  elapsed=time.perf_counter()-started)
    a.output.with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
