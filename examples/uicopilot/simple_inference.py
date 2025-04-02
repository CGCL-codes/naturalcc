import os
import torch
from PIL import Image
from scripts.train.utils import Html2BboxTree, move_to_device, BboxTree2Html, add_special_tokens, Html2BboxTree, BboxTree2StyleList, BboxTree2Html
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from agents.utils.tools import *
from agents import *

device = 'cuda'

processor = AutoProcessor.from_pretrained("anonymouscodee/webcoder")
model_bbox = Pix2StructForConditionalGeneration.from_pretrained("anonymouscodee/webcoder", is_encoder_decoder=True, device_map=device, torch_dtype=torch.float16)
add_special_tokens(model_bbox,processor.tokenizer)

agent_i2c = AgentI2C()
agent_optimize = AgentOptimize()

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

def gen(image, max_depth=100, min_area=100):
    imgs = []
    # infer
    prediction_html = infer_bbox(image)

    # draw bbox on image
    pBbox = Html2BboxTree(prediction_html, size=image.size)

    # pruning
    pruning(pBbox, 1, max_depth, min_area)

    # iter leaf node to gen ctree by agent
    bboxTree = Html2BboxTree(prediction_html, size=image.size)
    indexList = BboxTree2StyleList(bboxTree, skip_leaf=False)
    # only leaf filter
    indexList = list(filter(lambda x:not len(x['children']), indexList))
    img_count = 0
    for item in indexList:
        bbox = item['bbox']
        index_ = item['index']

        image_crop = image.crop((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]))
                
        if item['type'] == 'img':
            imgs.append(image_crop)
            new_src = f'{img_count}.png'
            part_html = new_src
            img_count += 1
        else:
            part_html = agent_i2c.infer(image_crop)
            part_html = extract_html(part_html)

        target = locateByIndex(bboxTree, index_)
        target['children'] = [part_html]

    html = BboxTree2Html(bboxTree, style=True)

    # optmize by agent
    html2 = agent_optimize.infer(image, html)
    html2 = extract_html(html2)
    return html, html2, imgs

def __main__():
    output_dir = './output'
    image = Image.open('test.png')

    _, html, imgs = gen(image)
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/index.html', 'w') as f:
        f.write(html)
    for idx, img in enumerate(imgs):
        img.save(f'{idx}.png')
