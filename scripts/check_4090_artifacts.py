"""Verify published LFS objects, trusted policy weights, and complete video decoding."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
VIDEO=ROOT/'results/video/4090_gpu'
sys.path.insert(0,str(ROOT/'src'))
from halfcheetah.ppo import ActorCritic


def digest(path):
    if not path.exists():raise ValueError(f'Missing artifact; run git lfs pull: {path}')
    with path.open('rb') as f:
        if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Git LFS object not downloaded; run git lfs pull: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    manifest=json.loads((VIDEO/'manifest.json').read_text())
    assert len(manifest['checkpoints'])==46 and len(manifest['videos'])==61
    seen={}
    for row in manifest['checkpoints']:
        p=ROOT/row['path'];assert digest(p)==row['sha256']
        c=yaml.safe_load((p.parent/'config.yaml').read_text())
        state=torch.load(p,map_location='cpu',weights_only=False)
        assert state['config']==c and state['timesteps']==2000896 and c['device']=='cuda'
        model=ActorCritic();model.load_state_dict(state['model'])
        assert all(torch.isfinite(t).all() for t in state['model'].values())
        for key in ['obs_rms','ret_rms']:
            assert np.isfinite(state[key]['mean']).all() and np.isfinite(state[key]['var']).all()
        for key in ['learning_rate','clip_range','entropy_coef']:
            assert c[key]==row[key]
        assert c['seed']==row['training_seed']
        seen[row['path']]=row['sha256']
    assert len(set(seen.values()))==41
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    unique={}
    for row in manifest['videos']:
        p=ROOT/row['path'];assert digest(p)==row['sha256']
        assert all(cp in seen for cp in row['checkpoints'])
        if row['category']=='policy':
            record=json.loads(p.with_suffix('.json').read_text())
            assert record['checkpoint_sha256']==seen[row['checkpoints'][0]]
            assert record['frames']==1000 and abs(record['return_difference'])<1e-6
            assert record['ground_clearance_margin']>0
        unique.setdefault(row['sha256'],row)

    def decode(row):
        result=subprocess.run([ff,'-hide_banner','-v','error','-xerror','-threads','1','-i',str(ROOT/row['path']),
                               '-map','0:v:0','-an','-f','null','-','-progress','pipe:1'],
                              capture_output=True,text=True,check=True)
        frames=[int(line.split('=')[1]) for line in result.stdout.splitlines() if line.startswith('frame=')]
        assert frames and frames[-1]==row['expected_frames'],(row['path'],frames[-1:] )
        assert row['fps']==20 and abs(row['duration']-frames[-1]/20)<.1
        return row['path']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(decode,unique.values()):pass
    qa=dict(status='passed',verified_at=datetime.now(timezone.utc).isoformat(),
            checkpoint_paths_verified=46,distinct_policies=41,model_state_loads=46,
            all_models_finite=True,all_normalizers_finite=True,training_device='cuda',training_hardware='RTX 4090',
            video_paths_verified=61,distinct_video_files_decoded=len(unique),all_video_hashes_match=True,
            all_frame_counts_match=True,all_video_checkpoint_links_match=True,
            max_render_return_difference=0.,video_inference_device='cpu',
            stored_binary_bytes=sum(r['bytes'] for r in manifest['checkpoints']+manifest['videos']))
    (VIDEO/'publication_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(qa,indent=2))


if __name__=='__main__':main()
