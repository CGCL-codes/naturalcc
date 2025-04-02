import gpt4v
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import time
import threading
import pandas as pd


GPT4V_SEMAPHORE = threading.Semaphore(8)   # GPT4v有访问限制，使用信号量
def main(images_path, out_dir, error_file, max_worders):
    prompt_system = """
    You are an expert Tailwind developer
    You take screenshots of a reference web page from the user, and then build single page apps 
    using Tailwind, HTML and JS.

    - Make sure the app looks exactly like the screenshot.
    - Make sure the app has the same page layout like the screenshot, i.e., the gereated html elements should be at the same place with the corresponding part in the screenshot and the generated  html containers should have the same hierachy structure as the screenshot.
    - Pay close attention to background color, text color, font size, font family, 
    padding, margin, border, etc. Match the colors and sizes exactly.
    - Use the exact text from the screenshot.
    - Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.
    - Repeat elements as needed to match the screenshot. For example, if there are 15 items, the code should have 15 items. DO NOT LEAVE comments like "<!-- Repeat for each news item -->" or bad things will happen.
    - For images, use placeholder images from https://placehold.co and include a detailed description of the image in the alt text so that an image generation AI can generate the image later.

    In terms of libraries,

    - Use this script to include Tailwind: <script src="https://cdn.tailwindcss.com"></script>
    - You can use Google Fonts
    - Font Awesome for icons: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

    Return only the full code in <html></html> tags.
    Do not include markdown "```" or "```html" at the start or end.
    """
    prompt_user = "Turn this into a single html file using tailwind."
    
    pool = ThreadPoolExecutor(max_worders)  
    def reuqest_func(img_path, out_path, error_file): 
        GPT4V_SEMAPHORE.acquire()   
        def release_func():
            GPT4V_SEMAPHORE.release()
        release_timer = threading.Timer(60, release_func)
        release_timer.start()
        res = gpt4v.request(img_path, prompt_system, prompt_user, 1080, 4096, 3, compress_rate=50)
        if res.find(gpt4v.ERROR_KEY) == -1:
            res = res.strip("```html")
            res = res.strip("```")
            with open(out_path, "w") as file:
                file.write(res)      
        else:
            with open(error_file, "a+") as file:
                # write errors to csv, (image_path, output_path, error_discription)
                file.write(f"{str(img_path)},{str(out_path)},{res}\n")       
    images_path = Path(images_path)
    if images_path.is_dir():
        files = [images_path / f for f in os.listdir(images_path)]
    else:
        df_files = pd.read_csv(images_path, usecols=[0], names=["path"])
        files = [ Path(f) for f in df_files["path"].to_list()]
    
    futures = []
    for file in files:        
        out_path = Path(out_dir) / f"{file.name.split('.')[0]}.html"
        futures.append(pool.submit(reuqest_func, file, out_path, error_file))   
             
        if len(futures) > 32:
            for futrue in as_completed(futures):
                res = futrue.result()
            futures = []
    pool.shutdown()
        
if __name__ == "__main__":
    import argparse
    argpaser = argparse.ArgumentParser()
    argpaser.add_argument("-i", "--images", type= str, required=True)
    argpaser.add_argument("-o", "--out_dir", type = str, required=True)
    argpaser.add_argument("-e", "--error_file", type = str, required=True)
    argpaser.add_argument("-w", "--max_workers", default=8, type= int)
    args = argpaser.parse_args()
    main(args.images, args.out_dir, args.error_file, args.max_workers)
