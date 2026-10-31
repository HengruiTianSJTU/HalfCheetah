"""Audit committed entropy data and optionally exercise the offline browser UI."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from report_entropy import ROOT, load_runs, rows, summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser',action='store_true',help='Requires Playwright and its Chromium browser')
    parser.add_argument('--output',type=Path,default=ROOT/'.local/entropy_qa')
    args=parser.parse_args()
    directory=ROOT/'results/entropy_sweep_20261030'
    protocol,runs=load_runs(directory)
    per_seed,groups=summarize(runs)
    for name,digest in json.loads((directory/'raw_data_sha256.json').read_text()).items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name
    for run in runs:
        for source,digest in run['metadata']['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
            recorded=subprocess.check_output(['git','-C',str(ROOT),'show',f"{protocol['source_commit']}:{source}"])
            assert hashlib.sha256(recorded).hexdigest()==digest
    saved=rows(directory/'group_summary.csv')
    for actual,expected in zip(groups,saved):
        assert actual.keys()==expected.keys()
        for key,value in actual.items():np.testing.assert_allclose(value,float(expected[key]),rtol=1e-12)
    saved=rows(directory/'per_seed_summary.csv')
    assert len(saved)==len(per_seed)==20
    for actual,expected in zip(per_seed,saved):
        assert actual['checkpoint_sha256']==expected['checkpoint_sha256']
        assert str(actual['reused_baseline'])==expected['reused_baseline']
        for key in ['entropy_coef','seed','final_validation_mean','normalized_auc','paired_difference_vs_baseline']:
            np.testing.assert_allclose(actual[key],float(expected[key]),rtol=1e-12)
    replay=rows(directory/'acceptance_replay.csv')
    assert len(replay)==20 and {r['run'] for r in replay}=={r['name'] for r in runs}
    for record in replay:
        source=rows(directory/record['run']/'eval.csv')
        expected=next(r for r in source if int(r['timesteps'])==2000896 and int(r['seed'])==10000)
        np.testing.assert_allclose(float(record['recorded_return']),float(expected['episode_return']))
        assert float(record['absolute_difference'])==0
    result=dict(raw_files_verified=100,source_revision_verified=protocol['source_commit'],runs=20,
                group_and_paired_metrics_recomputed=True,historical_replay_records_reconciled=True,
                weights_required=False)
    if args.browser:
        from playwright.sync_api import sync_playwright
        args.output.mkdir(parents=True,exist_ok=True)
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1280,'height':900})
            errors=[];network=[]
            page.on('pageerror',lambda err:errors.append(str(err)))
            page.route('http://**/*',lambda route:(network.append(route.request.url),route.abort()))
            page.route('https://**/*',lambda route:(network.append(route.request.url),route.abort()))
            page.goto((ROOT/'tutorial/entropy.html').as_uri())
            page.wait_for_selector('#chart svg',timeout=7000)
            assert page.locator('#chart polyline').count()==4
            assert page.locator('#groups tr').count()==4 and page.locator('#seeds tr').count()==5
            assert page.evaluate('ENTROPY_DATA.runs.length')==20
            for e in ['0','0.001','0.005','0.01']:
                page.select_option('#view','seed');page.select_option('#coefficient',e)
                assert page.locator('#chart polyline').count()==5
                for metric in ['train','curve']:
                    page.select_option('#metric',metric)
                    assert page.locator('#chart polyline').count()==5
            page.select_option('#view','group')
            before=page.locator('#chart').inner_html()
            page.locator('#until').evaluate("el=>{el.value='1001472';el.dispatchEvent(new Event('input'))}")
            assert page.locator('#chart').inner_html()!=before
            page.click('#reset')
            assert page.locator('#chart').inner_html()==before
            old=page.locator('#tip').inner_text()
            page.locator('#inspect').focus();page.keyboard.press('End')
            assert page.locator('#tip').inner_text()!=old
            page.locator('#chart').hover(position={'x':250,'y':160})
            assert 'β=' in page.locator('#tip').inner_text()
            config=page.locator('#config').inner_text();page.select_option('#run','19')
            assert page.locator('#config').inner_text()!=config
            for href in page.locator('#sources a, #runSources a, #home').evaluate_all('links=>links.map(a=>a.getAttribute("href"))'):
                assert (ROOT/'tutorial'/href).resolve().is_file(),href
            page.click('#reset');page.screenshot(path=str(args.output/'desktop.png'),full_page=True)
            page.locator('#curves').screenshot(path=str(args.output/'chart.png'))
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.screenshot(path=str(args.output/'mobile.png'),full_page=True)
            assert not errors,errors
            assert not network,network
            result.update(offline_browser=True,coefficient_and_seed_controls=True,entropy_metric=True,
                          horizon_and_reset=True,keyboard_and_hover=True,local_links=True,mobile_layout=True,
                          javascript_errors=errors,remote_requests=network)
            browser.close()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
