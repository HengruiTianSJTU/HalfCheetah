"""Exercise the local file:// tutorial in Chromium, including actual controls."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument('--allow-partial',action='store_true',help='Developer check while real experiments are still training; never final QA')
args=parser.parse_args()

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1280,'height':900})
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(Path('tutorial/index.html').resolve().as_uri())
    page.wait_for_selector('#experimentChart svg')
    assert page.locator('#summaryTable tr').count()>=(1 if args.allow_partial else 5)
    if not args.allow_partial:
        assert page.evaluate('TUTORIAL_DATA.runs.length')>=16
    before=page.locator('#objective').inner_html()
    page.locator('#epsilon').evaluate("el=>{el.value='0.35';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    assert page.locator('#epsilonValue').text_content()=='0.35'
    assert before!=page.locator('#objective').inner_html()
    page.locator('#epsilon').focus();page.keyboard.press('ArrowRight')
    assert page.locator('#epsilonValue').text_content()=='0.36'
    page.locator('#advantage').evaluate("el=>{el.value='-1';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    assert page.locator('#advantageValue').text_content()=='-1.0'
    assert page.locator('#benchmarkTable tr').count()==8
    config_before=page.locator('#runConfig').inner_text()
    page.locator('#runSelect').select_option('1')
    assert page.locator('#runConfig').inner_text()!=config_before
    page.locator('#runSelect').select_option('0')
    for view in ['clip','seed','baseline','all','lr']:
        page.locator('#view').select_option(view)
        assert page.locator('#experimentChart polyline').count()>0
    page.locator('#metric').select_option('train')
    assert page.locator('#experimentChart polyline').count()==3
    page.locator('#metric').select_option('evaluation')
    chart=page.locator('#experimentChart')
    chart.scroll_into_view_if_needed()
    box=chart.bounding_box()
    page.mouse.move(box['x']+box['width']*.4,box['y']+box['height']*.5)
    assert page.locator('#experimentTip').inner_text()
    original=chart.inner_html()
    page.mouse.move(box['x']+box['width']*.25,box['y']+box['height']*.5);page.mouse.down()
    page.mouse.move(box['x']+box['width']*.75,box['y']+box['height']*.5);page.mouse.up()
    assert original!=chart.inner_html()
    page.locator('#reset').click()
    assert original==chart.inner_html()
    chart.scroll_into_view_if_needed();box=chart.bounding_box()
    page.mouse.move(box['x']+2,box['y']+box['height']*.5);page.mouse.down()
    page.mouse.move(box['x']+box['width']*.07,box['y']+box['height']*.5);page.mouse.up()
    assert original==chart.inner_html(),'Dragging in the axis-label margin must not invert the zoom domain'
    assert page.evaluate('document.querySelector("video").canPlayType("video/mp4")')
    page.locator('#epsilon').evaluate("el=>{el.value='0.2';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    page.locator('#advantage').evaluate("el=>{el.value='1';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    Path('results/qa').mkdir(parents=True,exist_ok=True)
    page.screenshot(path='results/qa/tutorial_desktop.png',full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1')
    page.screenshot(path='results/qa/tutorial_mobile.png',full_page=True)
    assert not errors,errors
    report=dict(partial_data=args.allow_partial,offline_file_open=True,objective_controls=True,comparison_selectors=True,run_config_selector=True,hover=True,zoom_and_reset=True,mobile_layout=True,javascript_errors=errors)
    Path('results/qa/tutorial.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report));browser.close()
