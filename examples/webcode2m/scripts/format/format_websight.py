import sys
sys.setrecursionlimit(6000)
from datasets import Dataset,load_dataset
from io import BytesIO
import os
import glob
import time
from pyppeteer import launch
from bs4 import BeautifulSoup,Comment
from tqdm import tqdm
import cssutils
import logging
import asyncio
import re
import multiprocessing
import json
cssutils.log.setLevel(logging.CRITICAL)

data_path = "/data02/users/lz/code/UICoder/datasets/WebSight/"
output_path = "/data02/users/lz/code/UICoder/datasets/WebSight-format/"
batch_size = 10000

def move_styles(html_content):
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取style标签中的样式
    style_tag = soup.find('style')
    if style_tag:
        css_content = str(style_tag.string)
        styles = cssutils.parseString(css_content)
        
        for tag in soup.find_all():
            if 'style' in tag.attrs:
                tag['style'] = tag.attrs['style']
            else:
                tag['style'] = '' 

        for rule in styles:
            if rule.type == rule.STYLE_RULE:
                selectors = rule.selectorText.split(',')
                for selector in selectors:
                    selector = selector.strip()
                    elements = soup.select(selector)
                    for element in elements:
                        element['style'] +=  '; '+rule.style.cssText if element['style'] else rule.style.cssText
                        
        for tag in soup.find_all():
            if tag['style']:
                modified_css = re.sub(r'/\*.*?\*/', '', tag['style'], flags=re.DOTALL)
                modified_css = re.sub(r'\s+', ' ', modified_css, flags=re.MULTILINE)
                tag['style'] = modified_css
            else:
                del tag['style']
            del tag['class']
            del tag['id']

        style_tag.decompose()

    html = re.sub(r'\n\s*\n', '\n', str(soup), flags=re.MULTILINE)
    return html

async def gen_bbox(html,width,height):
    browser = await launch()
    page = await browser.newPage()
    await page.setViewport({'width': width, 'height': height})
    await page.setContent(html)
    
    bboxes = await page.evaluate('''
        () => {
            let depth = 0
            function generateBbox(element,depth) {
                if(depth>100) {
                    return
                }
                let rect = element.getBoundingClientRect();
                return {
                    type: element.tagName.toLowerCase(),
                    bbox: [parseInt(rect.left + window.scrollX),parseInt(rect.top + window.scrollY),parseInt(rect.width),parseInt(rect.height)],
                    style: element.getAttribute('style'),
                    depth: depth,
                    children: Array.from(element.children).map(item => generateBbox(item,depth+1))
                }
            }
            return generateBbox(document.body,0)
        }
    ''')
    
    await page.close()
    await browser.close()
    return bboxes


err_count = 0

async def worker(batch_index):
    global err_count
    ds = load_dataset(path=data_path)['train'].select(range(batch_size*batch_index,batch_size*(batch_index+1)))
    data = []
        
    with tqdm(total=len(ds)) as bar:
        bar.set_description_str(f"{'00' if batch_index<10 else ('0' if batch_index<100 else'')}{batch_index}")
        for item in ds:
            bar.update(1)
            try:
                html = move_styles(item['text'])
                bbox = await gen_bbox(html,item['image'].width,item['image'].height)
            except:
                err_count += 1
                bar.set_postfix_str(f'Parse Error Count: {err_count}')
                continue
            item = {
                'image': item['image'],
                'html': html,
                'bbox': json.dumps(bbox)
            }
            data.append(item)
        
    ds2 = Dataset.from_list(data)
    # ds2.save_to_disk(f'{output_path}/{'00' if batch_index>100 else ('0' if batch_index>10 else'')}{batch_index}')
    ds2.to_parquet(f"{output_path}/{'00' if batch_index<10 else ('0' if batch_index<100 else'')}{batch_index}.parquet")
    
    del ds
    del ds2
    
async def main():
    os.makedirs(output_path,exist_ok=True)
    tasks = []
    for i in range(30,40):
        tasks.append(asyncio.create_task(worker(i)))
    await asyncio.gather(*tasks)    
    
    
if __name__ == '__main__':
     asyncio.run(main())









