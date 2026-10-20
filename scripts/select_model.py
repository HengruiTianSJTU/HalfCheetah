"""Choose a representative policy and validate stability before touching final test seeds."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from halfcheetah.ppo import ActorCritic, RunningMeanStd
from halfcheetah.train import evaluate


def curve_mean(path):
    with (path/'eval.csv').open() as f:
        rows=list(csv.DictReader(f))
    last=max(int(r['timesteps']) for r in rows)
    return float(np.mean([float(r['episode_return']) for r in rows if int(r['timesteps'])==last]))


def main():
    torch.set_num_threads(1)
    candidates=[];baseline=[]
    for path in sorted(Path('results').glob('hc_lr*')):
        config=yaml.safe_load((path/'config.yaml').read_text())
        if config['learning_rate']>.001:
            continue
        assert json.loads((path/'metadata.json').read_text())['status']=='completed'
        candidates.append(path)
        if config['learning_rate']==3e-4 and config['clip_range']==.2:
            baseline.append(path)
    assert len(baseline) in [3,5]
    baseline.sort(key=curve_mean);representative=baseline[len(baseline)//2]
    scores=[]

    def assess(path):
        state=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
        model=ActorCritic();model.load_state_dict(state['model']);model.eval()
        rms=RunningMeanStd((17,));rms.load_state_dict(state['obs_rms'])
        rows=evaluate(model,rms,range(11000,11020))
        with (path/'selection_eval.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['seed','episode_return','episode_length']);writer.writeheader();writer.writerows(rows)
        values=np.array([r['episode_return'] for r in rows]);mean=float(values.mean());std=float(values.std(ddof=1))
        row=dict(experiment_name=path.name,mean=mean,std=std,minimum=float(values.min()),p10=float(np.quantile(values,.1)),cv=std/max(abs(mean),1.),curve_validation_mean=curve_mean(path))
        scores.append(row);return row

    score=assess(representative)
    stable=score['minimum']>1000 and score['cv']<=.30
    chosen=score;mode='median_baseline_after_stability_validation'
    if not stable:
        for path in candidates:
            if path!=representative:
                assess(path)
        eligible=[r for r in scores if r['minimum']>1000 and r['cv']<=.30]
        if not eligible:
            raise RuntimeError('No normal candidate passed the predeclared stability validation; investigate before final testing')
        chosen=max(eligible,key=lambda r:(r['p10'],r['mean']))
        mode='robust_validation_fallback'
    result=dict(chosen_experiment=chosen['experiment_name'],mode=mode,selection_seeds=list(range(11000,11020)),rule='Start with median final-curve baseline seed; keep it if all 20 selection returns >1000 and episode CV <=0.30. Otherwise assess every normal candidate on the same selection seeds, filter by that same stability rule, and choose highest 10th percentile, breaking ties by mean. Final test seeds 30000..30019 are untouched.',scores=scores,chosen_score=chosen)
    Path('results/model_selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
