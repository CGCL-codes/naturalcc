import asyncio
import os
import time
import io
from PIL import Image
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

async def all_images_loaded(page: Page, timeout: float = 3.0):
    """
  
    """
    t_end = time.time() + timeout
    images = await page.query_selector_all('img')
    for image in images:
        while time.time() < t_end:
            loaded = await page.evaluate('(img) => img.complete', image)
            if loaded:
                break
            else:
                await asyncio.sleep(0.01) 

async def take_screenshot_single_async(browser: Browser, html_content: str, output_file: str = None) -> Image.Image:
    """
  

    """
    try:
        context: BrowserContext = await browser.new_context()
        page: Page = await context.new_page()
        await page.set_content(html_content, timeout=50000, wait_until='networkidle')
        await all_images_loaded(page, timeout=3.0)

        screenshot_bytes = await page.screenshot(full_page=True, animations="disabled", timeout=50000)
        image = Image.open(io.BytesIO(screenshot_bytes))  
        
 
        if output_file:
            image.save(output_file)
        await context.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
    
        image = Image.new('RGB', (1280, 960), color='white')
        if output_file:
            image.save(output_file)
    return image

async def html2shot(html_content: str,save_path=None) -> Image.Image:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        image = await take_screenshot_single_async(browser, html_content, save_path)
        await browser.close()

    return image

async def main():
    base_path = "./screenshots"
    html_content = """
    <html>
        <body>
            <h1>Hello, World!</h1>
            <img src='https://via.placeholder.com/150' />
        </body>
    </html>
    """
    md5 = "d41d8cd98f00b204e9800998ecf8427e"
    index = 1

    image = await html2shot(base_path, html_content, md5, index)
    print(f"Screenshot saved and returned: {image}")

if __name__ == "__main__":
    asyncio.run(main())
