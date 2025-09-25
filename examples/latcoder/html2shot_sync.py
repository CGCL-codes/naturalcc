import sys, os
sys.path.append(os.path.abspath('.'))
from playwright.sync_api import sync_playwright,Page,BrowserContext,Browser
import argparse
from PIL import Image
import time
from pathlib import Path
from tqdm import tqdm
import json
from utils.utils import *
from utils.log import logger
import traceback

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


def take_screenshot_single(browser:Browser, html_content, output_file=None):
    try:          
        context:BrowserContext = browser.new_context()
        page = context.new_page()
        page.set_content(html_content, timeout=50000, wait_until='networkidle')
        all_images_loaded(page, 3)

        # Take the screenshot and save to memory
        screenshot_bytes = page.screenshot(full_page=True, animations="disabled", timeout=50000)
        image = Image.open(io.BytesIO(screenshot_bytes))  # Load image into memory

        # Optionally, save the screenshot to a file
        if output_file:
            image.save(output_file)
        context.close()            
    except Exception as e: 
        exception_info = traceback.format_exc()
        logger.error(f"Failed to take screenshot due to: {e}. Generating a blank image.\n \
                     Tace:\n{exception_info}\
                     HTML\n:{html_content}")
        # Generate a blank image
        image = Image.new('RGB', (1280, 960), color='white')
        if output_file:
            image.save(output_file)
    return image

        
def html2shot(html_content,save_path=None):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        result_image = take_screenshot_single(browser, html_content, save_path)
        browser.close()


    return result_image