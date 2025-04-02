# %%
from bs4 import BeautifulSoup

with open('/data03/starmage/projects/UICoder/outputs/eval_gpt4v/eval_long_20240518094311/a14b0de610ccdc6a7747c1f3f56efbc4/prediction.html') as f:
    soup = BeautifulSoup(f.read(), "html.parser")

# %%
soup.get_text()

# %%
import sys, os
sys.path.append(os.path.abspath('.'))
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = './.playwright'
from playwright.sync_api import sync_playwright,Page,BrowserContext,Browser
import argparse
from PIL import Image
import time
from pathlib import Path
from tqdm import tqdm
import json
def all_images_loaded(page:Page, timeout=3):
    t_end = time.time()+timeout
    images = page.query_selector_all('img') 
    for image in images:        
        while time.time() < t_end:
            loaded = page.evaluate('(img) => img.complete', image)
            if loaded:
                break
            else:
                time.sleep(0.01)

def take_screenshot_single(browser:Browser, url, output_file, output_bbox=False):
    # Convert local path to file:// URL if it's a file
    try:          
        context:BrowserContext = browser.new_context()
        page = context.new_page()
    
        # Navigate to the URL
        with open(url, 'r', encoding='utf-8') as f:
            html_content = f.read()
        page.set_content(html_content, timeout=10000, wait_until='domcontentloaded')
        all_images_loaded(page)

        # Take the screenshot
        page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=10000)
        if output_bbox:
            bboxTree = page.evaluate('''
                () => {
                    let depth = 0
                    function generateBbox(element,depth) {
                        if(depth>20) {
                            return
                        }
                        let rect = element.getBoundingClientRect();
                        let style = window.getComputedStyle(element);
                        let content = '';
                        if (element.childNodes.length === 1 && element.childNodes[0].nodeType === Node.TEXT_NODE) {
                            content = element.childNodes[0].textContent.trim();
                            if(content[0] === '<') content = ''
                        }
                        if (((rect.width === 0 || rect.height === 0) && content === '') || style.display === 'none' || style.visibility === 'hidden') {
                            return null;
                        }
                        return {
                            type: element.tagName.toLowerCase(),
                            content: content,
                            style: element.getAttribute('style'),
                            bbox: [parseInt(rect.left + window.scrollX),parseInt(rect.top + window.scrollY),parseInt(rect.width),parseInt(rect.height)],
                            children: Array.from(element.children).map(item => generateBbox(item,depth+1)).filter(item => item)
                        }
                    }
                    return generateBbox(document.body,0)
                }
            ''')
            bbox_outfile=Path(str(output_file)).parent / (Path(output_file).name.split('.')[0]+'_bbox.json')
            with open(bbox_outfile,'w') as f:
                f.write(json.dumps(bboxTree,indent=2))            

        context.close()            
    except Exception as e: 
        print(f"Failed to take screenshot due to: {e}. Generating a blank image. path:\n {url}")
        # Generate a blank image 
        img = Image.new('RGB', (1280, 960), color = 'white')
        img.save(output_file)

# %%
with sync_playwright() as p:    
    browser = p.chromium.launch()
    take_screenshot_single(browser, '/data03/starmage/projects/UICoder/outputs/eval_gpt4v/eval_long_20240518094311/1e22b049add7640965796cf4b58cd623/prediction.html', 'tmp.jpg')


