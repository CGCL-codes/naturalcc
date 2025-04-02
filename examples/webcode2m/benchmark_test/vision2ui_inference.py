import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train.utils import *
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from tqdm import tqdm
from utils import *
from PIL import Image,ImageDraw,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import easyocr
import time
import traceback

bbox_padding = 8

model_name = 'vision2ui_ws'
test_data_name = 'websight'

processor_name_or_path = '/data02/models/pix2struct-base/'
bbox_model_path = "/data02/users/lz/code/UICoder/checkpoints/stage1/l2048_p1024_ws_3m*1/checkpoint-35000"
# bbox_model_path = "/data02/users/lz/code/UICoder/checkpoints/stage1/l2048_p1024_vu2048_3m*1/checkpoint-55000"
style_model_path = "/data02/users/lz/code/UICoder/checkpoints/stage2/l256_p512_ws_30k*3/checkpoint-40000"
result_path = f'/data02/projects/vision2ui/results/{model_name}-{test_data_name}'
data_path = f'/data02/projects/vision2ui/datasets/{test_data_name}_benchmark'

DEVICE = 'cuda:7'

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model_bbox = Pix2StructForConditionalGeneration.from_pretrained(bbox_model_path,is_encoder_decoder=True,device_map=DEVICE,torch_dtype=torch.float16)
model_style = Pix2StructForConditionalGeneration.from_pretrained(style_model_path,is_encoder_decoder=True,device_map=DEVICE,torch_dtype=torch.float16)
add_special_tokens(model_bbox,processor.tokenizer)
add_special_tokens(model_style,processor.tokenizer)


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

def locateByIndex(bboxTree,index):
    target = bboxTree
    for i in list(filter(lambda x: x,index.split('-'))):
        target = target['children'][int(i)]
    return target

def predictStyle(image,styleItem):
    cnode_type_bbox_list = list(map(lambda x: f'{x["type"]}({x["bbox"][0]-styleItem["bbox"][0]},{x["bbox"][1]-styleItem["bbox"][1]},{x["bbox"][2]},{x["bbox"]})', styleItem['children']))
    input = f"{styleItem['type']}<{','.join(cnode_type_bbox_list)}> %{styleItem['style']}%<%"
    decoder_input_ids = processor.tokenizer.encode(input,return_tensors='pt')[...,:-1]
    encoding = processor(images=[image],text=[""],max_patches=512,return_tensors='pt')
    item = {
        'decoder_input_ids': decoder_input_ids,
        'flattened_patches': encoding['flattened_patches'].half(),
        'attention_mask': encoding['attention_mask']
    }
    item = move_to_device(item, DEVICE)
    with torch.no_grad():
        outputs = model_style.generate(**item,max_new_tokens=256,eos_token_id=processor.tokenizer.eos_token_id)
        predictions = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    css = '<'.join(predictions[0].split('<')[2:]).strip()
    if css and css[0] == '%':
        css = css[1:]
    if css and css[-1] == '>':
        css = css[:-1]
    if css and css[-1] == '%':
        css = css[:-1]
    css = css.split('%,%')
    return css

def predictBbox(image):
    # stage1
    model_bbox.eval()
    with torch.no_grad():
        input = f'<body bbox=['
        decoder_input_ids = processor.tokenizer.encode(input,   return_tensors='pt',add_special_tokens=True)[...,:-1]
        encoding = processor(images=[image],text=[""],  max_patches=1024,return_tensors='pt')
        item = {
            'decoder_input_ids': decoder_input_ids,
            'flattened_patches': encoding['flattened_patches']. half(),
            'attention_mask': encoding['attention_mask']
        }
        item = move_to_device(item,DEVICE)

        outputs = model_bbox.generate(**item,max_new_tokens=2048,eos_token_id=processor.tokenizer.eos_token_id,do_sample=True)
            
        prediction_html = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return prediction_html

def predict(image):
    # stage1
    predictied_html = predictBbox(image)

    # process
    bboxTree = Html2BboxTree(predictied_html, size=image.size)
    indexList = BboxTree2StyleList(bboxTree, skip_leaf=False)

    # stage2
    for item in tqdm(indexList):
        bbox = item['bbox']
        index = item['index']

        if not item['children'] or bbox[2] <= bbox_padding*2 or bbox[3] <= bbox_padding*2:
            continue
        image_crop = image.crop((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]))
        predicted_css = predictStyle(image_crop,item)

        for idx, css_item in enumerate(predicted_css):
            index_tmp = f"{index}{'-' if index else ''}{idx}"
            target = locateByIndex(bboxTree, index_tmp)
            target['style'] = css_item

    # text and image apply
    reader = easyocr.Reader(['ch_sim','en'])
    imgs = []
    for item in tqdm(indexList):
        if not len(item['children']):
            bbox = item['bbox']
            index = item['index']
            if bbox[2] <= bbox_padding*2 or bbox[3] <= bbox_padding*2:
                continue
            image_crop = image.crop((bbox[0]-bbox_padding,bbox[1]-bbox_padding,bbox[0]+bbox[2]+bbox_padding,bbox[1]+bbox[3]+bbox_padding))
            target = locateByIndex(bboxTree, index)
            if item['type'] == 'img':
                target['children'] = [f'{len(imgs)}.png']
                imgs.append(image_crop)
                target['style'] += f'width: {bbox[2]}px; height: {bbox[3]}px;'
            else:
                result = reader.readtext(np.array(image_crop))
                text = '\n'.join(list(map(lambda x: x[1],   result)))
                target['children'] = [text]

    predicted_html_with_style = BboxTree2Html(bboxTree,style=True).replace("style=''","").replace("src=''","")  

    return predicted_html_with_style, imgs


if test_data_name == 'vision2ui':
    ds = load_dataset('parquet',data_dir=data_path)['train']
else:
    ds = Dataset.load_from_disk(data_path)

for index,item in enumerate(tqdm(ds)):
    while True:
        try:
            image = item['image']
            md5 = image2md5(image)
            save_path = result_path
            if test_data_name == 'vision2ui':
                tokens = sum(item['tokens'])
                size = 'short' if tokens<2048 else ('mid' if tokens<4096 else 'long')
                save_path = os.path.join(result_path,size)

            if os.path.exists(os.path.join(result_path,md5)):
                break

            t_start = time.time()
            html, imgs = predict(image)
            html = f'<html>{html}</html>'

            duration = time.time()-t_start
            save_result(save_path, image, item['text'] if 'text' in item else item['html'], html, duration, imgs=imgs)
            break
        except Exception as e:
            print(f'Error {e}')
            pass

