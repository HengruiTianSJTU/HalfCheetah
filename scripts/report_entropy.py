"""Regenerate the separate entropy study using CSV/config metadata; weights are optional."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def csv_write(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)


def load_runs(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    coefficients, seeds = protocol['entropy_coefficients'], protocol['training_seeds']
    expected = {(float(e), int(s)) for e in coefficients for s in seeds}
    hashes = json.loads((directory/'checkpoint_hashes.json').read_text()) if (directory/'checkpoint_hashes.json').exists() else {}
    reuse = protocol.get('baseline_reuse', {})
    reused = set(reuse) if isinstance(reuse, dict) else {Path(r['destination']).name for r in reuse}
    runs, seen = [], set()
    versions = None
    for path in sorted(directory.glob('hc_entropy*')):
        if not path.is_dir():
            continue
        c = yaml.safe_load((path/'config.yaml').read_text())
        meta = json.loads((path/'metadata.json').read_text())
        key = (float(c['entropy_coef']), int(c['seed']))
        if key in seen or key not in expected:
            raise ValueError(f'Duplicate or unexpected entropy/seed pair: {path}')
        seen.add(key)
        fixed = dict(protocol['fixed_config'], entropy_coef=key[0], seed=key[1])
        if c != fixed or c['learning_rate'] != .0003 or c['clip_range'] != .2:
            raise ValueError(f'Fixed-baseline configuration mismatch: {path}')
        if versions is None:
            versions = meta['versions']
        if meta['versions'] != versions:
            raise ValueError(f'Mixed dependency versions: {path}')
        batch = c['n_envs']*c['n_steps']
        target = math.ceil(c['total_timesteps']/batch)*batch
        if meta.get('status') != 'completed' or meta.get('actual_timesteps') != target:
            raise ValueError(f'Incomplete training/budget mismatch: {path}')
        train = [{k:float(v) for k,v in r.items()} for r in rows(path/'metrics.csv')]
        ev = rows(path/'eval.csv')
        if len(train) != target//batch or [r['timesteps'] for r in train] != list(range(batch,target+1,batch)):
            raise ValueError(f'Missing or reordered training rows: {path}')
        if any(not math.isfinite(v) for r in train for k,v in r.items() if k != 'training_return_mean'):
            raise ValueError(f'Nonfinite training statistic: {path}')
        if len({(r['timesteps'],r['seed']) for r in ev}) != len(ev):
            raise ValueError(f'Duplicate evaluation rows: {path}')
        steps = sorted({int(r['timesteps']) for r in ev})
        expected_steps = sorted({0, target, *(math.ceil(t/batch)*batch for t in range(c['eval_interval'],target,c['eval_interval']))})
        if steps != expected_steps:
            raise ValueError(f'Missing evaluation checkpoints: {path}')
        curve = []
        for step in steps:
            block = [r for r in ev if int(r['timesteps']) == step]
            if {int(r['seed']) for r in block} != set(protocol['validation_seeds']):
                raise ValueError(f'Evaluation seed mismatch: {path}')
            vals = [float(r['episode_return']) for r in block]
            if not np.isfinite(vals).all() or any(int(r['episode_length']) != 1000 for r in block):
                raise ValueError(f'Invalid evaluation: {path}')
            curve.append(dict(timesteps=step, mean=float(np.mean(vals))))
        cp = path/'checkpoint.pt'
        digest = hashlib.sha256(cp.read_bytes()).hexdigest() if cp.exists() else hashes.get(path.name)
        runs.append(dict(name=path.name, config=c, metadata=meta, curve=curve, train=train,
                         checkpoint_sha256=digest, reused_baseline=path.name in reused))
    if seen != expected:
        raise ValueError(f'Incomplete entropy matrix; missing {expected-seen}')
    runs.sort(key=lambda r:(r['config']['entropy_coef'],r['config']['seed']))
    return protocol, runs


def summarize(runs):
    per_seed, groups = [], []
    for run in runs:
        c, curve = run['config'], run['curve']
        xs, ys = [r['timesteps'] for r in curve], [r['mean'] for r in curve]
        per_seed.append(dict(entropy_coef=c['entropy_coef'], seed=c['seed'], final_validation_mean=ys[-1],
                             normalized_auc=float(np.trapezoid(ys,xs)/(xs[-1]-xs[0])),
                             training_seconds=run['metadata']['wall_clock_time'], reused_baseline=run['reused_baseline'],
                             checkpoint_sha256=run['checkpoint_sha256']))
    baseline = {r['seed']:r['final_validation_mean'] for r in per_seed if r['entropy_coef']==0}
    for row in per_seed:
        row['paired_difference_vs_baseline'] = row['final_validation_mean']-baseline[row['seed']]
    for e in sorted({r['entropy_coef'] for r in per_seed}):
        block = [r for r in per_seed if r['entropy_coef']==e]
        vals = np.array([r['final_validation_mean'] for r in block]); mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if len(vals)>1 else None
        groups.append(dict(entropy_coef=e, n_training_seeds=len(vals), mean=mean, std=sd,
                           median=float(np.median(vals)), min=float(vals.min()), max=float(vals.max()),
                           cv=sd/abs(mean) if sd is not None and mean else None,
                           mean_normalized_auc=float(np.mean([r['normalized_auc'] for r in block]))))
    return per_seed, groups


def plots(runs, directory):
    directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    coefficients = sorted({r['config']['entropy_coef'] for r in runs})
    for metric, filename, ylabel in [('curve','validation_curves','Validation episode return'),
                                     ('train','policy_entropy','Gaussian policy differential entropy')]:
        fig, ax = plt.subplots(figsize=(10,6),layout='constrained')
        for i,(e,color) in enumerate(zip(coefficients,COLORS)):
            block=[r for r in runs if r['config']['entropy_coef']==e]
            xs=np.array([v['timesteps'] for v in block[0][metric]])/1e6
            key='mean' if metric=='curve' else 'entropy'
            vals=np.array([[v[key] for v in r[metric]] for r in block]); mean=vals.mean(axis=0)
            ax.plot(xs,mean,color=color,linestyle=['-','--','-.',':'][i],label=f'entropy = {e:g}')
            if len(block)>1:
                sd=vals.std(axis=0,ddof=1); ax.fill_between(xs,mean-sd,mean+sd,color=color,alpha=.12)
        ax.set(xlabel='Environment transitions (millions)',ylabel=ylabel,
               title='Entropy sweep | mean ± sample SD across training seeds')
        ax.legend();ax.grid(alpha=.2);fig.savefig(directory/f'{filename}.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    for j,(e,color) in enumerate(zip(coefficients,COLORS)):
        block=[r for r in runs if r['config']['entropy_coef']==e]
        vals=np.array([r['curve'][-1]['mean'] for r in block]); xs=j+np.linspace(-.12,.12,len(block))
        ax.scatter(xs,vals,color=color,s=45)
        for k,(x,y,r) in enumerate(zip(xs,vals,block)):
            offset=[(3,7),(-22,8),(7,8),(6,-14),(3,7)][k%5]
            if e==.01 and k==0:offset=(-18,-15)
            if e==.01 and k==1:offset=(-8,16)
            ax.annotate(str(r['config']['seed']),(x,y),xytext=offset,textcoords='offset points',fontsize=9)
        ax.errorbar(j,vals.mean(),yerr=vals.std(ddof=1) if len(vals)>1 else None,fmt='k_',capsize=8,markersize=20)
    ax.set(xticks=range(len(coefficients)),xticklabels=[f'{e:g}' for e in coefficients],xlabel='Entropy coefficient',
           ylabel='Final validation mean return',title='Training seeds | black marker: mean ± sample SD')
    ax.grid(axis='y',alpha=.2);fig.savefig(directory/'final_seed_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
    seed_colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8C6D31']
    for ax,e in zip(axes.flat,coefficients):
        for i,r in enumerate(r for r in runs if r['config']['entropy_coef']==e):
            ax.plot([v['timesteps']/1e6 for v in r['curve']],[v['mean'] for v in r['curve']],
                    color=seed_colors[i%5],linestyle=['-','--','-.',':',(0,(5,1))][i%5],label=f"seed {r['config']['seed']}")
        ax.set(title=f'Entropy = {e:g}',xlabel='Transitions (millions)',ylabel='Validation return')
        ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.savefig(directory/'individual_seed_curves.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/entropy_sweep_20261030')
    parser.add_argument('--figures',type=Path,default=ROOT/'results/figures/entropy')
    parser.add_argument('--html',type=Path,default=ROOT/'tutorial/entropy.html')
    args=parser.parse_args()
    protocol,runs=load_runs(args.results)
    per_seed,groups=summarize(runs)
    csv_write(args.results/'per_seed_summary.csv',per_seed);csv_write(args.results/'group_summary.csv',groups)
    payload=dict(groups=groups,per_seed=per_seed,scope='GPU deterministic validation, seeds 10000–10004; no held-out test used for tuning'
                 if protocol['fixed_config']['device']=='cuda' else 'CPU deterministic validation; no held-out test used for tuning',
                 std_definition='sample standard deviation across training-seed means')
    (args.results/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False))
    plots(runs,args.figures)
    # Only the display is downsampled; statistics and exported figures use full CSVs.
    view_runs=[]
    for run in runs:
        train=run['train']; stride=max(1,len(train)//250)
        points=train[::stride]
        if points[-1]['timesteps']!=train[-1]['timesteps']:points=points+[train[-1]]
        view_runs.append(dict(name=run['name'],config=run['config'],curve=run['curve'],
                             train=[dict(timesteps=p['timesteps'],entropy=p['entropy']) for p in points]))
    display=dict(groups=groups,per_seed=per_seed,runs=view_runs,protocol=protocol,
                 home_path=Path(os.path.relpath(ROOT/'tutorial/index.html',args.html.resolve().parent)).as_posix(),
                 results_path=Path(os.path.relpath(args.results.resolve(),args.html.resolve().parent)).as_posix())
    template=(ROOT/'tutorial/entropy_template.html').read_text()
    args.html.parent.mkdir(parents=True,exist_ok=True)
    args.html.write_text(template.replace('__ENTROPY_DATA__',json.dumps(display,ensure_ascii=False,allow_nan=False)))
    print(f'Generated entropy report from {len(runs)} runs; no checkpoints required.')


if __name__=='__main__':
    main()
