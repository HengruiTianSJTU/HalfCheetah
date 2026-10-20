"""Exercise the local file:// tutorial in Chromium, including actual controls."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1280,'height':900})
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(Path('tutorial/index.html').resolve().as_uri())
    page.wait_for_selector('#experimentChart svg')
    assert page.locator('#summaryTable tr').count()>=5
    before=page.locator('#objective').inner_html()
    page.locator('#epsilon').evaluate("el=>{el.value='0.35';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    assert page.locator('#epsilonValue').text_content()=='0.35'
    assert before!=page.locator('#objective').inner_html()
    page.locator('#advantage').evaluate("el=>{el.value='-1';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    assert page.locator('#advantageValue').text_content()=='-1.0'
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
    assert page.evaluate('document.querySelector("video").canPlayType("video/mp4")')
    Path('results/qa').mkdir(parents=True,exist_ok=True)
    page.screenshot(path='results/qa/tutorial_desktop.png',full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1')
    page.screenshot(path='results/qa/tutorial_mobile.png',full_page=True)
    assert not errors,errors
    report=dict(offline_file_open=True,objective_controls=True,comparison_selectors=True,hover=True,zoom_and_reset=True,mobile_layout=True,javascript_errors=errors)
    Path('results/qa/tutorial.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report));browser.close()
