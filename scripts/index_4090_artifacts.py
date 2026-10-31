"""Index downloaded 4090 checkpoints/videos and build their standalone local gallery."""
import csv
import hashlib
import json
from pathlib import Path
import re

import imageio.v2 as imageio
import yaml

ROOT=Path(__file__).resolve().parents[1]
VIDEO=ROOT/'results/video/4090_gpu'


def sha(path):
    with path.open('rb') as f:
        if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Run git lfs pull before indexing binary artifacts: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checkpoints=[]
    for directory,cohort in [('reproduction_4090_20261030','main'),('entropy_sweep_20261030','entropy')]:
        for cp in sorted((ROOT/'results'/directory).glob('*/checkpoint.pt')):
            c=yaml.safe_load((cp.parent/'config.yaml').read_text())
            metadata=json.loads((cp.parent/'metadata.json').read_text())
            with (cp.parent/'eval.csv').open() as f:ev=list(csv.DictReader(f))
            last=max(int(r['timesteps']) for r in ev)
            vals=[float(r['episode_return']) for r in ev if int(r['timesteps'])==last]
            checkpoints.append(dict(path=cp.relative_to(ROOT).as_posix(),sha256=sha(cp),bytes=cp.stat().st_size,
                cohort=cohort,run=cp.parent.name,learning_rate=c['learning_rate'],clip_range=c['clip_range'],
                entropy_coef=c['entropy_coef'],training_seed=c['seed'],training_device=c['device'],
                timesteps=metadata['actual_timesteps'],validation_mean=sum(vals)/len(vals)))
    assert len(checkpoints)==46 and len({r['sha256'] for r in checkpoints})==41
    final=json.loads((ROOT/'results/reproduction_4090_20261030/final_evaluation_summary.json').read_text())
    videos=[]
    for p in sorted(VIDEO.rglob('*.mp4')):
        rel=p.relative_to(VIDEO);category='policy';cohort='entropy' if rel.parts[0]=='entropy' or p.stem.startswith('entropy') else 'main'
        linked=[];extra={}
        if p.with_suffix('.json').exists():
            record=json.loads(p.with_suffix('.json').read_text())
            linked=[r for r in checkpoints if r['sha256']==record['checkpoint_sha256'] and r['cohort']==cohort]
            assert len(linked)==1,(p,linked)
            c=linked[0]
            title=(f"熵系数 {c['entropy_coef']:g}" if cohort=='entropy' else f"lr {c['learning_rate']:g} / clip {c['clip_range']:g}")
            title+=f" · 训练 seed {c['training_seed']} · 演示 seed {record['demo_seed']}"
            extra=dict(demo_seed=record['demo_seed'],episode_return=record['episode_return'],
                       inference_device='cpu',return_difference=record['return_difference'])
        elif rel.parts[0]=='comparisons':
            category='comparison'
            if p.stem=='entropy_coefficients_seed42':
                linked=[r for r in checkpoints if r['cohort']=='entropy' and r['training_seed']==42]
                title='四种熵系数同屏对比 · 固定训练 seed 42'
            elif p.stem.startswith('entropy'):
                e=float(re.fullmatch(r'entropy([0-9.]+)_seeds42-46',p.stem)[1])
                linked=[r for r in checkpoints if r['cohort']=='entropy' and r['entropy_coef']==e]
                title=f'熵系数 {e:g} · 五个训练种子同屏对比'
            else:
                m=re.fullmatch(r'lr([0-9.]+)_clip([0-9.]+)_seeds42-46',p.stem)
                lr,clip=map(float,m.groups())
                linked=[r for r in checkpoints if r['cohort']=='main' and r['learning_rate']==lr and r['clip_range']==clip]
                title=f'lr {lr:g} / clip {clip:g} · 五个训练种子同屏对比'
        elif p.stem=='experiment_story':
            category='story';title='26 次主实验与模型选择 · 完整叙事'
            linked=[r for r in checkpoints if r['cohort']=='main']
        elif p.stem=='final_policy_extended':
            category='final';title='主实验最终策略 · 扩展地面完整演示'
            linked=[r for r in checkpoints if r['sha256']==final['checkpoint_sha256']]
        else:raise ValueError(f'Unclassified video: {p}')
        assert linked,p
        digest=sha(p)
        reader=imageio.get_reader(p)
        info=reader.get_meta_data();reader.close()
        videos.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=digest,bytes=p.stat().st_size,
                           title=title,cohort=cohort,category=category,width=info['size'][0],height=info['size'][1],
                           fps=info['fps'],duration=info['duration'],expected_frames=3440 if category=='story' else 1000,
                           checkpoints=[r['path'] for r in linked],
                           poster=p.with_suffix('.jpg').relative_to(ROOT).as_posix() if p.with_suffix('.jpg').exists() else None,
                           **extra))
    assert len(videos)==61,len(videos)
    manifest=dict(title='4090 GPU 实现结果',training_hardware='NVIDIA RTX 4090',training_device='cuda',
                  video_inference_device='cpu',checkpoint_paths=46,distinct_policies=41,video_paths=61,
                  checkpoints=checkpoints,videos=videos)
    (VIDEO/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    template=(ROOT/'tutorial/video_gallery_template.html').read_text()
    (ROOT/'results/video/index.html').write_text(template.replace('__VIDEO_DATA__',json.dumps(manifest,ensure_ascii=False)))
    print('Indexed 46 checkpoint paths (41 distinct policies) and 61 videos.')


if __name__=='__main__':main()
