"""Summarize measured SB3 reference speed, quality and resource use."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows=[]
for path in sorted(Path('results/benchmark').glob('*/summary.json')):
    s=json.loads(path.read_text());r=s['resource_mean']
    rows.append(dict(device=s['device'],n_envs=s['n_envs'],vec=s['vec'],timesteps=s['actual_timesteps'],fps=s['fps'],wall_clock=s['wall_clock'],environment_only_fps=s['environment_only_fps'],evaluation_mean=float(np.mean(s['evaluation_returns'])),evaluation_episode_sd=float(np.std(s['evaluation_returns'],ddof=1)),cpu_percent=r['cpu_percent'],cpu_tree_percent=r.get('cpu_tree_percent',r['cpu_percent']),gpu_percent=r['gpu_percent'],gpu_memory_mb=r['gpu_memory_mb']))
if len(rows)!=8:
    raise RuntimeError(f'Expected all 8 completed benchmarks, found {len(rows)}')
with Path('results/benchmark_summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
Path('results/benchmark_summary.json').write_text(json.dumps(rows,indent=2))
labels=[f"{r['device']} / {r['n_envs']} envs / {r['vec']}" for r in rows]
fig,ax=plt.subplots(figsize=(9,5.5));bars=ax.barh(labels,[r['fps'] for r in rows],color=['#538d82' if r['device']=='cpu' else '#af7954' for r in rows])
ax.bar_label(bars,fmt='%.0f',padding=4);ax.set_xlim(0,max(r['fps'] for r in rows)*1.12);ax.set_xlabel('Environment transitions / second');ax.set_title('Tesla T4 / 8 vCPU: SB3 PPO 100k-step reference benchmark');ax.grid(axis='x',alpha=.2)
fig.tight_layout();Path('results/figures').mkdir(exist_ok=True);fig.savefig('results/figures/benchmark.png',dpi=160)
print(json.dumps(rows,indent=2))
