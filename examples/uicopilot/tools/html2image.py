from selenium import webdriver
from pathlib import Path
import os
import io

import cv2
import numpy as np
from PIL import Image,ImageChops

def trim_img(image_path, border):
    # 读取图片
    image = Image.open(image_path)
    # 取分布最多的颜色为背景色
    colors = image.getcolors(image.size[0]*image.size[1])
    colors.sort(key= lambda x : x[0], reverse=True)
    bg_color = colors[0][1]
  
    bg = Image.new(image.mode, image.size, bg_color)
    diff = ImageChops.difference(image, bg).convert("RGB") # 必须转换，否则会有问题
    diff = ImageChops.add(diff, diff)
    bbox = diff.getbbox()  
    if bbox:
        image = image.crop((max(0, bbox[0]-border), max(0, bbox[1]-border),min(image.size[0],bbox[2]+border),min(image.size[1],bbox[3]+border)))
    # 保存裁剪后的图片
    image.convert("RGBA").save(image_path, "PNG")

def main(html_dir, out_dir):
    html_dir = Path(html_dir)
    out_dir = Path(out_dir)
    # 设置Chrome驱动程序
    driver = webdriver.Chrome()
    # Resize the browser window
    driver.set_window_size(4976, 2566)
    
    for file in os.listdir(html_dir):
        html_path = html_dir/ file
        out_path = out_dir / f"{file.split('.')[0]}.png"
        # 打开你的html文件
        driver.get(f'file://{str(html_path)}')  # 将path_to_your_file替换为你的html文件的实际路径           
        # 截图并保存
        driver.save_screenshot(out_path)
        trim_img(str(out_path),10)
    # 关闭浏览器
    driver.quit()
    
if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--html_dir", type= str, required=True)
    parser.add_argument("-o", "--out_dir", type = str, required=True)
    args = parser.parse_args()
    main(args.html_dir,args.out_dir)
