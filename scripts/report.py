"""Regenerate figures, tables and offline tutorial data only from recorded CSV files."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def read_csv(path):
    with path.open() as f:
        return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]


def main():
    runs=[]
    for directory in sorted(Path('results').glob('hc_lr*')):
        if not (directory/'eval.csv').exists():
            continue
        c=yaml.safe_load((directory/'config.yaml').read_text())
        train=read_csv(directory/'metrics.csv');evaluations=read_csv(directory/'eval.csv')
        steps=sorted(set(r['timesteps'] for r in evaluations))
        curve=[dict(timesteps=t,mean=float(np.mean([r['episode_return'] for r in evaluations if r['timesteps']==t])),std=float(np.std([r['episode_return'] for r in evaluations if r['timesteps']==t],ddof=1))) for t in steps]
        if not curve or not train:
            continue
        metadata=json.loads((directory/'metadata.json').read_text())
        runs.append(dict(name=directory.name,config=c,metadata=metadata,train=train,evaluation=curve,final=curve[-1]))
    if not runs:
        raise RuntimeError('No real experiment metrics found')
    baseline=[r for r in runs if r['config']['learning_rate']==3e-4 and r['config']['clip_range']==.2]
    normal=[r for r in runs if r['config']['learning_rate']<=.001]
    failure=[r for r in runs if r['config']['learning_rate']>.001]
    natural_path=Path('results/failures/natural_baseline_seed44/snapshot.json')
    natural=json.loads(natural_path.read_text()) if natural_path.exists() else None
    figures=Path('results/figures');figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'figure.figsize':(8.5,5),'font.size':11,'axes.grid':True,'grid.alpha':.25,'savefig.dpi':160})

    def draw_group(group,label,kind='evaluation'):
        if not group:
            return
        key='mean' if kind=='evaluation' else 'training_return_mean'
        # Restrict to common support: no extrapolation of shorter/unfinished runs.
        max_step=min(r[kind][-1]['timesteps'] for r in group)
        xs=np.array([v['timesteps'] for v in group[0][kind] if v['timesteps']<=max_step])
        ys=np.array([np.interp(xs,[v['timesteps'] for v in r[kind]],[v[key] for v in r[kind]]) for r in group])
        mean=np.mean(ys,axis=0);line=plt.plot(xs/1e6,mean,label=f'{label} (n={len(group)})')[0]
        if len(group)>1:
            std=np.std(ys,axis=0,ddof=1)
            plt.fill_between(xs/1e6,mean-std,mean+std,color=line.get_color(),alpha=.18)

    def finish(filename,title,ylabel='Deterministic evaluation return'):
        plt.title(title);plt.xlabel('Environment transitions (millions)');plt.ylabel(ylabel);plt.legend(fontsize=9);plt.tight_layout();plt.savefig(figures/filename);plt.close()

    draw_group(baseline,'Baseline','train')
    finish('baseline_training.png','PPO baseline: training return (mean ± seed SD)','Raw return (last 100 completed episodes)')
    draw_group(baseline,'Baseline')
    finish('evaluation.png','PPO baseline: held-out evaluation (mean ± seed SD)')
    for lr in [1e-4,3e-4,1e-3]:
        draw_group([r for r in normal if r['config']['learning_rate']==lr and r['config']['clip_range']==.2],f'lr={lr:g}')
    finish('learning_rate.png','Learning-rate ablation; clip=0.2')
    for clip in [.1,.2,.3]:
        draw_group([r for r in normal if r['config']['learning_rate']==3e-4 and r['config']['clip_range']==clip],f'clip={clip:g}')
    finish('clip_range.png','Clip-range ablation; lr=0.0003')
    for run in baseline:
        draw_group([run],f"Seed {run['config']['seed']}")
    finish('seed_variability.png','Baseline training-seed variability')
    draw_group(baseline,'Baseline lr=0.0003')
    for run in failure:
        draw_group([run],f"Aggressive lr={run['config']['learning_rate']:g}")
    if natural:
        draw_group([r for r in runs if r['name']==natural['source_experiment']],'Natural collapse: baseline seed44')
        plt.scatter([natural['timesteps']/1e6],[natural['evaluation_mean']],color='red',zorder=5,label='Saved poor checkpoint')
    finish('failure_vs_success.png','Aggressive learning rate versus baseline')
    rows=[]
    for run in runs:
        ev=run['evaluation'];xs=np.array([e['timesteps'] for e in ev]);ys=np.array([e['mean'] for e in ev])
        rows.append(dict(experiment_name=run['name'],seed=run['config']['seed'],learning_rate=run['config']['learning_rate'],clip_range=run['config']['clip_range'],timesteps=run['final']['timesteps'],final_eval_mean=run['final']['mean'],final_episode_sd=run['final']['std'],evaluation_auc=float(np.trapezoid(ys,xs)/(xs[-1]-xs[0])) if xs[-1]>xs[0] else 0,wall_clock=run['metadata'].get('wall_clock_time',run['train'][-1]['wall_clock']),status=run['metadata'].get('status','running')))
    with Path('results/summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=[]
    for lr,clip in sorted(set((r['learning_rate'],r['clip_range']) for r in rows)):
        group=[r for r in rows if r['learning_rate']==lr and r['clip_range']==clip]
        groups.append(dict(learning_rate=lr,clip_range=clip,n_seeds=len(group),final_mean=float(np.mean([r['final_eval_mean'] for r in group])),seed_sd=float(np.std([r['final_eval_mean'] for r in group],ddof=1)) if len(group)>1 else None,auc_mean=float(np.mean([r['evaluation_auc'] for r in group])),completed=all(r['status']=='completed' for r in group)))
    fig,ax=plt.subplots(figsize=(10,1.4+.5*len(groups)));ax.axis('off')
    cells=[[f"{g['learning_rate']:g}",str(g['clip_range']),str(g['n_seeds']),f"{g['final_mean']:.1f}",f"{g['seed_sd']:.1f}" if g['seed_sd'] is not None else 'N/A',f"{g['auc_mean']:.1f}"] for g in groups]
    table=ax.table(cellText=cells,colLabels=['Learning rate','Clip','Seeds','Final mean','Seed SD','Eval AUC'],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.6)
    ax.set_title('Final validation return by configuration (raw reward)',pad=15);fig.tight_layout();fig.savefig(figures/'summary_table.png');plt.close(fig)
    payload=dict(runs=runs,summary=rows,groups=groups,total_training_wall_clock=sum(r['wall_clock'] or 0 for r in rows))
    if natural:
        payload['natural_failure']=natural
    benchmark_path=Path('results/benchmark_summary.json')
    if benchmark_path.exists():
        payload['benchmarks']=json.loads(benchmark_path.read_text())
    final_path=Path('results/final_evaluation_summary.json')
    if final_path.exists():
        payload['final_evaluation']=json.loads(final_path.read_text())
        final_rows=read_csv(Path('results/final_evaluation.csv'))
        plt.bar([int(r['seed']) for r in final_rows],[r['episode_return'] for r in final_rows],color='#398f83')
        plt.axhline(payload['final_evaluation']['mean'],color='#b45728',linestyle='--',label='20-episode mean')
        plt.xlabel('Held-out evaluation seed');plt.ylabel('Raw episodic return');plt.title('Final policy: independent evaluation episodes');plt.ticklabel_format(useOffset=False,style='plain',axis='x');plt.legend();plt.tight_layout();plt.savefig(figures/'final_evaluation.png');plt.close()
    # Training downsampling only affects tutorial size, never figures or summary calculation.
    for run in runs:
        stride=max(1,len(run['train'])//250)
        run['train']=run['train'][::stride]+([run['train'][-1]] if (len(run['train'])-1)%stride else [])
    def clean(value):
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):
            return [clean(v) for v in value]
        return None if isinstance(value,float) and not np.isfinite(value) else value
    payload=clean(payload)
    Path('results/summary.json').write_text(json.dumps(payload,separators=(",",":"),allow_nan=False))
    template=Path('tutorial/template.html').read_text()
    html=template.replace('__EXPERIMENT_DATA__',json.dumps(payload,allow_nan=False))
    Path('tutorial/index.html').write_text(html)
    print(f'Generated figures, summary CSV/JSON and offline tutorial from {len(runs)} real runs')


if __name__=='__main__':
    main()
