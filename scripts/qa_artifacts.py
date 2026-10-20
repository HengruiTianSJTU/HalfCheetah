"""Check final delivery against real files and independently recomputed metrics."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    expected={(lr,.2,s) for lr in [1e-4,3e-4,1e-3] for s in [42,43,44]}|{(3e-4,c,s) for c in [.1,.3] for s in [42,43,44]}
    extension=json.loads(Path('results/extension_plan.json').read_text())
    expected|={(r['learning_rate'],r['clip_range'],r['seed']) for r in extension['matrix']}
    seen=set();durations=[];baseline=[];failures=[]
    for directory in Path('results').glob('hc_lr*'):
        c=yaml.safe_load((directory/'config.yaml').read_text())
        m=json.loads((directory/'metadata.json').read_text())
        if c['learning_rate']<=.001:
            assert m['status']=='completed',directory
            key=(c['learning_rate'],c['clip_range'],c['seed'])
            assert key not in seen,key
            seen.add(key);durations.append(m['actual_timesteps'])
        assert m['git_commit'] and m['source_sha256'] and m['versions']
        assert m['seed']==c['seed'] and m['total_timesteps']==c['total_timesteps']
        for file,sha in m['source_sha256'].items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha,(directory,file)
            committed=subprocess.check_output(['git','show',f"{m['git_commit']}:{file}"])
            assert hashlib.sha256(committed).hexdigest()==sha,(directory,file,'recorded Git revision does not reproduce source')
        metrics=rows(directory/'metrics.csv')
        assert metrics and all(float(r['fps'])>0 for r in metrics)
        for row in metrics:
            for key,value in row.items():
                if key!='training_return_mean':
                    assert np.isfinite(float(value)),(directory,key)
        evaluation=rows(directory/'eval.csv')
        last=max(int(r['timesteps']) for r in evaluation)
        selected=[r for r in evaluation if int(r['timesteps'])==last]
        assert len(selected)==c['eval_episodes']
        assert all(np.isfinite(float(r['episode_return'])) and int(r['episode_length'])==1000 for r in evaluation)
        mean=np.mean([float(r['episode_return']) for r in selected])
        if c['learning_rate']==3e-4 and c['clip_range']==.2:
            baseline.append(mean)
        if c['learning_rate']>.001:
            failures.append(mean)
    assert seen==expected,(seen,expected)
    assert len(set(durations))==1 and durations[0]>=2000000
    assert failures and min(failures)<np.mean(baseline)*.5,'Need a real clearly poor experiment'
    pilot=rows(Path('results/pilot_sync_100k/metrics.csv'))
    formal=rows(Path('results/hc_lr0.0003_clip0.2_seed42/metrics.csv'))
    keys=['timesteps','training_return_mean','policy_loss','value_loss','entropy','approx_kl','clip_fraction']
    assert all(all(a[k]==b[k] for k in keys) for a,b in zip(pilot,formal))
    assert len(formal)>=len(pilot)
    final=rows(Path('results/final_evaluation.csv'))
    assert len(final)==20 and {int(r['seed']) for r in final}==set(range(30000,30020))
    selection=json.loads(Path('results/model_selection.json').read_text())
    assert set(selection['selection_seeds']).isdisjoint({int(r['seed']) for r in final})
    chosen=selection['chosen_score']
    assert chosen['minimum']>1000 and chosen['cv']<=.30
    values=[float(r['episode_return']) for r in final]
    motion=rows(Path('results/final_motion.csv'))
    assert len(motion)==20
    np.testing.assert_allclose([float(r['episode_return']) for r in motion],values)
    for row in motion:
        np.testing.assert_allclose(float(row['forward_reward_sum'])+float(row['control_reward_sum']),float(row['episode_return']))
    summary=json.loads(Path('results/final_evaluation_summary.json').read_text())
    np.testing.assert_allclose(summary['mean'],np.mean(values))
    np.testing.assert_allclose(summary['std'],np.std(values,ddof=1))
    assert summary['mean']>1000,'Selected policy has not reached the declared forward-motion validation target'
    checkpoint=Path('results')/summary['experiment_name']/'checkpoint.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==summary['checkpoint_sha256']
    assert hashlib.sha256(Path('results/videos/halfcheetah_running.mp4').read_bytes()).hexdigest()==summary['video_sha256']
    video_rows=rows(Path('results/video_evaluation.csv'))
    assert len(video_rows)==1 and int(video_rows[0]['seed'])==30000
    np.testing.assert_allclose(float(video_rows[0]['episode_return']),float(final[0]['episode_return']))
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,nb_frames:format=duration','-of','json','results/videos/halfcheetah_running.mp4'],text=True))
    assert int(probe['streams'][0]['nb_frames'])==1000
    assert probe['streams'][0]['codec_name']=='h264'
    assert abs(float(probe['format']['duration'])-50)<.1
    for name in ['baseline_training','evaluation','learning_rate','clip_range','seed_variability','failure_vs_success','summary_table','final_evaluation']:
        image=Path('results/figures')/(name+'.png')
        assert image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') and image.stat().st_size>1000
    html=Path('tutorial/index.html').read_text()
    embedded=json.loads(re.search(r'const DATA=(.*?);\nconst COLORS=',html,re.S).group(1))
    assert len(embedded['runs'])>=16
    np.testing.assert_allclose(embedded['final_evaluation']['mean'],summary['mean'])
    browser=json.loads(Path('results/qa/tutorial.json').read_text())
    assert not browser['javascript_errors'] and not browser['partial_data'] and browser['offline_file_open'] and browser['zoom_and_reset']
    natural_path=Path('results/failures/natural_baseline_seed44/snapshot.json')
    if natural_path.exists():
        natural=json.loads(natural_path.read_text())
        case=natural_path.parent
        assert natural['evaluation_mean']<0
        np.testing.assert_allclose(natural['evaluation_mean'],np.mean([float(r['episode_return']) for r in rows(case/'eval.csv')]))
        assert hashlib.sha256((case/'checkpoint.pt').read_bytes()).hexdigest()==natural['checkpoint_sha256']
    subprocess.run(['git','diff','--check'],check=True)
    tracked=subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
    for name in filter(None,tracked):
        file=Path(name)
        assert file.stat().st_size<3*1024*1024,(name,'oversized tracked artifact')
        assert file.suffix not in ['.pt','.pth','.mp4','.pkl','.key','.pem'],name
        if file.suffix not in ['.png','.jpg']:
            text=file.read_text(errors='replace')
            assert not re.search(r'-----BEGIN [A-Z ]*PRIVATE KEY-----',text),name
            assert not re.search(r'(?i)(password|api[_-]?key|secret[_-]?key)\s*[:=]\s*[\"\']?[A-Za-z0-9+/]{16,}',text),name
    result=dict(formal_normal_experiments=len(seen),failure_experiments=len(failures),matched_training_budgets=True,fixed_seed_100k_prefix_reproduced=True,raw_metric_traceability=True,source_hashes_match=True,independent_evaluation=True,valid_final_video=True,figures=True,tutorial_embedded_data=True,tracked_secret_and_size_audit=True)
    Path('results/qa/artifacts.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    main()
