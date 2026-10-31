"""Guard against mixing experiment cohorts or silently skipping missing weights."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


driver=script('run_entropy_experiments')
report=script('report_entropy')
REFERENCE=ROOT/'results/entropy_sweep_20261030'


def test_matrix_changes_only_entropy_seed_device_and_budget():
    base=yaml.safe_load((ROOT/'configs/entropy/entropy0.yaml').read_text())
    before=copy.deepcopy(base)
    cases=driver.make_cases(base,[42,43],'cpu',2048)
    assert len(cases)==8 and len({driver.name_of(c) for c in cases})==8
    for c in cases:
        assert c==dict(base,entropy_coef=c['entropy_coef'],seed=c['seed'],device='cpu',total_timesteps=2048)
    assert base==before


def test_reference_csvs_cannot_be_reused_without_weights():
    path=REFERENCE/'hc_entropy0_seed42'
    c=yaml.safe_load((path/'config.yaml').read_text())
    m=json.loads((path/'metadata.json').read_text())
    with pytest.raises(ValueError,match='Required baseline artifact missing'):
        driver.validate_reuse(path,c,m['source_sha256'],m['versions'])


@pytest.fixture
def copied_results(tmp_path):
    target=tmp_path/'results'
    shutil.copytree(REFERENCE,target)
    return target


def test_report_rejects_baseline_mixing(copied_results):
    p=copied_results/'hc_entropy0_seed42/config.yaml'
    c=yaml.safe_load(p.read_text());c['device']='cpu';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='Fixed-baseline configuration mismatch'):
        report.load_runs(copied_results)


def test_reuse_rejects_checkpoint_with_wrong_budget(copied_results):
    path=copied_results/'hc_entropy0_seed42'
    c=yaml.safe_load((path/'config.yaml').read_text())
    m=json.loads((path/'metadata.json').read_text())
    torch.save(dict(config=c,timesteps=2048),path/'checkpoint.pt')
    with pytest.raises(ValueError,match='checkpoint configuration/budget mismatch'):
        driver.validate_reuse(path,c,m['source_sha256'],m['versions'])


def test_report_rejects_incomplete_seed_matrix(copied_results):
    shutil.rmtree(copied_results/'hc_entropy0.001_seed46')
    with pytest.raises(ValueError,match='Incomplete entropy matrix'):
        report.load_runs(copied_results)


def test_report_rejects_duplicate_evaluation(copied_results):
    p=copied_results/'hc_entropy0.005_seed42/eval.csv'
    with p.open('a') as f:f.write(p.read_text().splitlines()[1]+'\n')
    with pytest.raises(ValueError,match='Duplicate evaluation rows'):
        report.load_runs(copied_results)


def test_group_spread_is_across_training_seeds_and_pairs_match():
    _,runs=report.load_runs(REFERENCE)
    per_seed,groups=report.summarize(runs)
    g=next(g for g in groups if g['entropy_coef']==.001)
    np.testing.assert_allclose([g['mean'],g['std']],[3422.1041297927586,1118.867963935939])
    diff=next(r['paired_difference_vs_baseline'] for r in per_seed if r['entropy_coef']==.001 and r['seed']==42)
    np.testing.assert_allclose(diff,-4917.136958212406)
    assert sum(r['reused_baseline'] for r in per_seed)==5


def test_single_seed_has_no_invented_uncertainty():
    _,runs=report.load_runs(REFERENCE)
    _,groups=report.summarize([r for r in runs if r['config']['seed']==42])
    assert all(g['n_training_seeds']==1 and g['std'] is None and g['cv'] is None for g in groups)
