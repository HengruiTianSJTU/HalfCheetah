"""Compose real entropy-policy clips with fixed-seed comparisons and cohort statistics."""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
W,H=1920,1080
BG,FG,MUTED,BLUE='#101b2b','#f1f4f7','#b2c0d1','#79b5ee'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clips',type=Path,default=ROOT/'results/video/4090_gpu/entropy')
    p.add_argument('--results',type=Path,default=ROOT/'results/entropy_sweep_20261030')
    p.add_argument('--output',type=Path,default=ROOT/'results/video/4090_gpu/comparisons')
    p.add_argument('--font',type=Path,default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    a=p.parse_args()
    if not a.font.exists():p.error('Supply a Chinese-capable TrueType/OpenType font using --font')
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.results/'group_summary.csv').open() as f:
        groups={float(r['entropy_coef']):r for r in csv.DictReader(f)}
    font=lambda n:ImageFont.truetype(str(a.font),n)
    def text(im,xy,s,n=28,color=FG):
        draw=ImageDraw.Draw(im)
        box=draw.textbbox(xy,s,font=font(n))
        assert box[2]<W-20 and box[3]<H-8,(s,box)
        draw.text(xy,s,font=font(n),fill=color)
    def base(title,subtitle,footer):
        im=Image.new('RGB',(W,H),BG)
        text(im,(64,28),'4090 GPU 实现结果  /  HALFCHEETAH PPO',26,MUTED)
        text(im,(64,78),title,48)
        text(im,(64,151),subtitle,26,MUTED)
        ImageDraw.Draw(im).line((64,201,1856,201),fill='#36465b',width=2)
        text(im,(64,1018),footer,24,MUTED)
        return im
    def panel(im,e,seed,rect,label):
        name=f'hc_entropy{e:g}_seed{seed}_demo10000'
        path=a.clips/f'{name}.mp4'
        with path.open('rb') as f:
            if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError('Download videos with git lfs pull before composition')
        record=json.loads((a.clips/f'{name}.json').read_text())
        x,y,w,h=rect
        text(im,(x,y-39),label,28)
        text(im,(x,y+h+8),f"演示 R {record['episode_return']:,.0f} · {record['mean_velocity']:.2f} m/s",22,MUTED)
        return dict(path=path,rect=rect)
    scenes=[]
    for e in [0.,.001,.005,.01]:
        g=groups[e]
        im=base(f'熵系数 {e:g}：同一设置下的五个训练种子',
                '学习率 0.0003 / clip 0.2 / 演示初始种子 10000 / 正常速度 1×',
                '训练：RTX 4090 CUDA；视频策略推理：CPU。单回合演示与多种子验证统计分别标注。')
        rects=[(64,260,544,306),(680,260,544,306),(1296,260,544,306),(64,650,544,306),(680,650,544,306)]
        inputs=[panel(im,e,s,rect,f'训练种子 {s}') for s,rect in zip(range(42,47),rects)]
        text(im,(1296,650),'五个训练种子的验证结果',28)
        text(im,(1296,710),f"{float(g['mean']):,.1f} ± {float(g['std']):,.1f}",42,BLUE)
        text(im,(1296,787),'均值 ± 跨训练种子样本标准差',23,MUTED)
        text(im,(1296,834),'每个模型使用 5 个验证回合',23,MUTED)
        text(im,(1296,891),'保留全部种子，不只展示最高分。',25)
        scenes.append((f'entropy{e:g}_seeds42-46',im,inputs))
    im=base('固定训练种子 42：四种熵系数的配对演示',
            '学习率 0.0003 / clip 0.2 / 相同训练预算 / 同一演示初始种子 10000',
            '单个种子的跑动不能代表组平均；β=0.001 的均值提升有限，尚不能证明显著优于基线。')
    inputs=[]
    for e,x in zip([0.,.001,.005,.01],[64,528,992,1456]):
        inputs.append(panel(im,e,42,(x,320,416,234),f'熵系数 β = {e:g}'))
        g=groups[e]
        text(im,(x,665),'五种子验证均值 ± 标准差',23,MUTED)
        text(im,(x,716),f"{float(g['mean']):,.0f} ± {float(g['std']):,.0f}",38,BLUE)
        text(im,(x,800),f"全程 AUC {float(g['mean_normalized_auc']):,.0f}",27)
    text(im,(64,930),'训练：RTX 4090 CUDA；视频推理：CPU，固定均值动作；五种子统计来自训练时的 CUDA 验证。',26,MUTED)
    scenes.append(('entropy_coefficients_seed42',im,inputs))
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix='entropy-layouts-') as temp:
        def encode(item):
            name,im,inputs=item
            out=a.output/f'{name}.mp4'
            if out.exists():raise ValueError(f'Existing video preserved: {out}')
            bg=Path(temp)/f'{name}.png';im.save(bg)
            cmd=[ff,'-y','-hide_banner','-loglevel','error','-loop','1','-framerate','20','-i',str(bg)]
            for source in inputs:cmd+=['-threads','1','-i',str(source['path'])]
            filters=[];prev='0:v'
            for i,source in enumerate(inputs,1):
                x,y,w,h=source['rect']
                filters.append(f'[{i}:v]setpts=PTS-STARTPTS,scale={w}:{h},setsar=1[v{i}]')
                filters.append(f'[{prev}][v{i}]overlay={x}:{y}:shortest=1[o{i}]');prev=f'o{i}'
            filters.append(f'[{prev}]format=yuv420p[out]')
            cmd+=['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]',
                  '-t','50','-r','20','-an','-c:v','libx264','-preset','fast','-crf','20','-threads','2',
                  '-movflags','+faststart',str(out)]
            subprocess.run(cmd,check=True)
            subprocess.run([ff,'-y','-loglevel','error','-ss','4','-i',str(out),'-frames:v','1',str(out.with_suffix('.jpg'))],check=True)
            print('COMPOSED',name,flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for future in concurrent.futures.as_completed([pool.submit(encode,item) for item in scenes]):future.result()


if __name__=='__main__':main()
