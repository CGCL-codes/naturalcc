import os
os.environ['TOKENIZERS_PARALLELISM'] = 'True'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image,ImageDraw
import easyocr
import numpy as np
import json
import re
import torch
from tqdm import tqdm
from scripts.train.utils import smart_tokenizer_and_embedding_resize,Html2BboxTree,move_to_device,BboxTree2Html,add_special_tokens, Html2BboxTree, BboxTree2StyleList, BboxTree2Html
from scripts.train.vars import *
from scripts.train.my_dataset import UICoderDataset,UICoderCollater
from transformers import AutoProcessor, Pix2StructForConditionalGeneration,AddedToken
from datasets import Dataset, load_dataset
from agents import *
from agents.utils.tools import *
from utils import image2md5, save_result
import time
import sys

torch.manual_seed(SEED)

device = 'cuda:5'
bbox_model_path = "/data02/users/lz/code/UICoder/checkpoints/stage1/l2048_p1024_vu2048_3m*1/checkpoint-50000"
# bbox_model_path = "/data02/users/lz/code/UICoder/checkpoints_bak/stage1/l2048_p1024_vu2048_3m*1/checkpoint-55000" # 无剪枝


model_name1 = 'ablation-vu-agent-dep4'
model_name2 = 'ablation-vu-agent-opt-dep4'
test_data_name = 'vision2ui'

result_path1 = f'/data02/projects/vision2ui/results/{model_name1}-{test_data_name}'
result_path2 = f'/data02/projects/vision2ui/results/{model_name2}-{test_data_name}'
data_path = f'/data02/projects/vision2ui/datasets/{test_data_name}_benchmark'




processor = AutoProcessor.from_pretrained(processor_name_or_path)
model_bbox = Pix2StructForConditionalGeneration.from_pretrained(bbox_model_path,is_encoder_decoder=True,device_map=device,torch_dtype=torch.float16)
add_special_tokens(model_bbox,processor.tokenizer)


agent_i2c = AgentI2C()
agent_optimize = AgentOptimize()
# agent_optimize =AgentOptimizeStable()






def drawBboxOnImage(draw: ImageDraw,bbox_node):
    bbox = bbox_node['bbox']
    if bbox[2] > 0 and bbox[3] > 0:
        draw.rectangle((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]),outline="red",width=2)
    for node in bbox_node['children']:
        drawBboxOnImage(draw, node)
        
def stickImages(images):
    sizes = list(map(lambda x: x.size,images))

    max_width = max(list(map(lambda x: x[0],sizes)))
    max_height = max(list(map(lambda x: x[1],sizes)))

    new_image = Image.new('RGB', ((max_width+10)*len(images), max_height))
    
    for idx,image in enumerate(images):
        new_image.paste(image, ((max_width+10)*idx, 0))

    return new_image

def remove_bbox(html_content):
    bbox_pattern = r' bbox=\[[^\]]*\]'
    cleaned_html = re.sub(bbox_pattern, '', html_content)
    return cleaned_html

def infer_bbox(image):
    model_bbox.eval()
    with torch.no_grad():
        input = f'<body bbox=['
        decoder_input_ids = processor.tokenizer.encode(input,return_tensors='pt',add_special_tokens=True)[...,:-1]
        encoding = processor(images=[image],text=[""],max_patches=1024,return_tensors='pt')
        item = {
            'decoder_input_ids': decoder_input_ids,
            'flattened_patches': encoding['flattened_patches'].half(),
            'attention_mask': encoding['attention_mask']
        }
        item = move_to_device(item,device)
    
        outputs = model_bbox.generate(**item,max_new_tokens=2560,eos_token_id=processor.tokenizer.eos_token_id,do_sample=True)
                
        prediction_html = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return prediction_html

def locateByIndex(bboxTree,index):
    target = bboxTree
    for i in list(filter(lambda x: x,index.split('-'))):
        target = target['children'][int(i)]
    return target

def extract_html(html):
    if '```' in html:
        html = html.split('```')[1]
    if html[:4] == 'html':
        html = html[4:]
    html = html.strip()
    return html

def pruning(node, now_depth, max_depth, min_area):
    bbox = node['bbox']
    area = abs((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
    if area < min_area:
        return None
    if now_depth >= max_depth:
        node['children'] = []
    else:
        for idx, cnode in enumerate(node['children']):
            node['children'][idx] = pruning(cnode, now_depth+1, max_depth, min_area)
        node['children'] = list(filter(lambda x:x, node['children']))
    return node

def gen(image, ind=None,max_depth=100, min_area=100,retries=5, retry_delay=2):
    for attempt in range(retries):
        try:
            imgs = []
            # infer
            print('Predicting bboxTree...')
            prediction_html = infer_bbox(image)

            # draw bbox on image
            pBbox = Html2BboxTree(prediction_html, size=image.size)

            # 剪枝
            pruning(pBbox, 1, max_depth, min_area)

            # aImage = image.copy()
            pImage = image.copy()

            # drawBboxOnImage(ImageDraw.Draw(aImage),json.loads(ds[idx]['bbox']))
            drawBboxOnImage(ImageDraw.Draw(pImage),pBbox)
            # pair = stickImages([aImage,pImage])

            # iter leaf node to gen ctree by agent
            bboxTree = Html2BboxTree(prediction_html, size=image.size)
            indexList = BboxTree2StyleList(bboxTree, skip_leaf=False)
            # only leaf filter
            indexList = list(filter(lambda x:not len(x['children']), indexList))
            img_count = 0
            for item in indexList:
                bbox = item['bbox']
                index = item['index']

                image_crop = image.crop((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]))
                
                if item['type'] == 'img':
                    imgs.append(image_crop)
                    new_src = f'{img_count}.png'
                    part_html = new_src
                    img_count += 1
                    # part_html = re.sub(r'(<img [^>]*?src=")(.*?)("[^>]*>)', r'\g<1>%s\3' % new_src, part_html)
                else:
                    part_html = agent_i2c.infer(image_crop)
                    part_html = extract_html(part_html)

                target = locateByIndex(bboxTree, index)
                target['children'] = [part_html]

            html = BboxTree2Html(bboxTree, style=True)

            # optmize by agent
            html2 = agent_optimize.infer(image, html)
            html2 = extract_html(html2)
            return html,html2,imgs

        except Exception as e:
            tqdm.write(f"Attempt {attempt + 1} failed for index {ind}: {e}")
            sys.stdout.flush()
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                tqdm.write(f"Failed after multiple attempts for index {ind}")
                sys.stdout.flush()
                # 返回一个默认的 HTML 作为异常处理
                return (f"<html><body><h1>Error Occurred</h1><p>Failed after multiple attempts for index {ind}</p></body></html>",
                        f"<html><body><h1>Error Occurred</h1><p>Failed after multiple attempts for index {ind}</p></body></html>")






if test_data_name == 'vision2ui':
    ds = load_dataset('parquet', data_dir=data_path)['train']
else:
    ds = Dataset.load_from_disk(data_path)



# 特定的索引集合
target_indices = []

for index in target_indices:
    item = ds[index]  # 获取对应索引的数据项
    image = item['image']
    md5 = image2md5(image)

    
    save_path1 = result_path1
    save_path2 = result_path2
    if test_data_name == 'vision2ui':
        tokens = sum(item['tokens'])
        size = 'short' if tokens < 2048 else ('mid' if tokens < 4096 else 'long')
        save_path1 = os.path.join(result_path1, size)
        save_path2 = os.path.join(result_path2, size)

    # Ensure the save path exists
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
        print(f"Created directory: {save_path1}")
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)
        print(f"Created directory: {save_path2}")


    t_start = time.time()
    html,html2,imgs= gen(image,ind=index) # 传递 index 以便记录失败的索引
    
    
    # 检查生成的 HTML 是否有效
    max_attempts = 5  # 设置最大尝试次数
    attempt = 0
    while attempt < max_attempts:
        if 'Failed' not in html and 'Failed' not in html2 and isinstance(html, str) and isinstance(html2, str):
            duration = time.time() - t_start
            save_result(save_path1, image, item['text'] if 'text' in item else item['html'], html, duration,imgs=imgs)
            save_result(save_path2, image, item['text'] if 'text' in item else item['html'], html2, duration,imgs=imgs)
            tqdm.write(f"{index}")
            sys.stdout.flush()
            break  # 成功后退出循环
        else:
            attempt += 1
            tqdm.write(f"Attempt {attempt} failed to generate HTML for index {index}: {md5}")
            sys.stdout.flush()
            if attempt < max_attempts:  # 只有在未达到最大尝试次数时才重新请求
                html,html2,imgs = gen(image,ind=index)  # 重新请求
            else:
                tqdm.write(f"Failed after {max_attempts} attempts for index {index}: {md5}")
                sys.stdout.flush()
    sys.stdout.flush()