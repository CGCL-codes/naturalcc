
import time,os,sys
sys.path.append(os.path.abspath('.'))
from io import BytesIO
from PIL import Image
import cssutils
import playwright
from playwright.async_api import async_playwright, Playwright, Browser, BrowserContext, Page
import logging
cssutils.log.setLevel(logging.CRITICAL)
import asyncio
from tools.log import logger
import argparse
import socket
from pathlib import Path
import json
import traceback
from format_utils import move_style_inline


async def all_images_loaded(page:Page, timeout=3):
    t_end = time.time()+timeout
    images = await page.query_selector_all('img') 
    for image in images:        
        while time.time() < t_end:
            loaded = await page.evaluate('(img) => img.complete', image)
            if loaded:
                break
            else:
                await asyncio.sleep(0.01)

async def gen_shortcut_and_bbox(html, context: BrowserContext, ratio_range=[0.5,2]):    
    page = await context.new_page()   
    await page.set_content(html)

    bodyHandle = await page.query_selector('body')
    bodyWidth = await page.evaluate('(body) => body.scrollWidth', bodyHandle)
    bodyHeight = await page.evaluate('(body) => body.scrollHeight', bodyHandle)

    if bodyHeight<bodyWidth*ratio_range[0] or bodyHeight>bodyWidth*ratio_range[1] or bodyWidth>2000:
        raise ValueError('TIPS: image size too big')

    await all_images_loaded(page)
    screenshot_bytes = await page.screenshot(full_page=True)
    image_file = BytesIO(screenshot_bytes)
    image = Image.open(image_file)

    bboxTree = await page.evaluate('''
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

    return image,bboxTree

def save(item_dir:Path,html,image,bboxes):
    item_dir.mkdir(parents=True,exist_ok=True)
    with open(item_dir /'index.html','w') as f:
        f.write(html.replace('>\n<','><'))
    with open(item_dir /'bbox.json','w') as f:
        f.write(json.dumps(bboxes,indent=2))
    image.save(os.path.join(item_dir,'image.png'))

async def get_screenshot(pw: Playwright, browser:Browser, formated_dir, html, ratio_range, proxy, timeout, chunk=0, volume=0, idx=0):
    try:   
        context = None
        if not browser.is_connected():  
            browser:Browser = await pw.chromium.launch(headless = True, proxy={"server":proxy})
            logger.debug(f"{traceback.format_exc()}")
            logger.error("Restart the browser.") 
        
        # 保存目录已经存在则跳过
        item_dir = formated_dir / f"{chunk:03}/{volume:03}_{idx:05}"
        if item_dir.exists():
            return
        html_inline = move_style_inline(html) 
        context = await browser.new_context()                 
        image, bboxes = await asyncio.wait_for(gen_shortcut_and_bbox(html_inline, context, ratio_range), timeout) 
        await context.close()
        save(item_dir,html,image,bboxes)        
        return "done"        
    except Exception as e:
        exception_info = traceback.format_exc()
        if 'TIPS' not in exception_info:
            logger.debug(exception_info)
            
        if context: # todo: 验证多次调用close是否会引起崩溃，多情况测试
            await context.close()            
        return "failed"


def recv_cmd(socket:socket.socket):
    def send_ok():
        socket.send("ack".encode('utf-8'))
    cmd = socket.recv(1024).decode('utf-8')    
    send_ok() 
    content_len =  int(socket.recv(1024).decode('utf-8'))
    send_ok()
    content = ''
    while len(content) < content_len:
        content += socket.recv(1024).decode('utf-8')
    send_ok()
    return cmd, content
    
            
async def main(port, save_dir, ratio_range, proxy, timeout):
    formated_dir = Path(save_dir) / "formated"
    formated_dir.mkdir(exist_ok=True, parents=True)
    async with async_playwright() as pw:        
        browser = await pw.chromium.launch(headless = True, proxy={"server":proxy})
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('127.0.0.1', port))
        while(True):
            cmd, content = recv_cmd(client)        
            if cmd == "exit":
                break
            elif cmd == "screenshot":
                obj = json.loads(content)
                res = await get_screenshot(pw, browser, formated_dir, obj["html"], ratio_range, proxy, timeout, obj["chunk"], obj["volume"], obj["idx"])
                client.send(f"{res}".encode('utf-8'))
            else:
                logger.info(f"Unknown command {cmd}.")
        client.close()
        await browser.close()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-pt", type=int, required=True)
    parser.add_argument("--save_dir", "-o", type=str, default="./data")
    parser.add_argument("--ratio_range", "-r", type=str, default="0.5-2")
    parser.add_argument("--proxy", "-p", type=str, default="http://127.0.0.1:7890")
    parser.add_argument("--timeout", "-t", type=int, default=5)
    args = parser.parse_args()
    ratio_range = [round(float(x),1) for x in args.ratio_range.split('-')]
    asyncio.run(main(args.port, args.save_dir, ratio_range, args.proxy, args.timeout))