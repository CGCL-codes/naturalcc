import time
from io import BytesIO
from PIL import Image
from pyppeteer import launch

async def gen_shortcut(html):
    browser = await launch(
        handleSIGINT=False,
        handleSIGTERM=False,
        handleSIGHUP=False
    )
    page = await browser.newPage()
    await page.setViewport({'width': 640, 'height': 360})
    await page.setContent(html)
    # 
    bodyHandle = await page.querySelector('body')
    bodyWidth = await page.evaluate('(body) => body.scrollWidth', bodyHandle)
    bodyHeight = await page.evaluate('(body) => body.scrollHeight', bodyHandle)
    
    await page.setViewport({'width': bodyWidth, 'height': bodyHeight})
    
    time.sleep(1)

    screenshot_bytes = await page.screenshot()
    image_file = BytesIO(screenshot_bytes)
    image = Image.open(image_file)
    
    await bodyHandle.dispose()
    await page.close()
    return image