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
from PIL import Image,ImageDraw
import numpy as np
import easyocr
import time

model_name = 'end2end_vu'
test_data_name = 'vision2ui'

processor_name_or_path = '/data02/models/pix2struct-base/'
model_path = "/data02/users/lz/code/UICoder/checkpoints/stage0/l1024_p1024_vu_3m*3/checkpoint-4000"
# model_path = "/data03/starmage/projects/UICoder/checkpoints/final/stage0/ws_30w_l3072_p1024_ws_1m_3"
result_path = f'/data02/projects/vision2ui/results/{model_name}-{test_data_name}'
data_path = f'/data02/projects/vision2ui/datasets/{test_data_name}_benchmark'

DEVICE = 'cuda:6'

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = Pix2StructForConditionalGeneration.from_pretrained(model_path,is_encoder_decoder=True,device_map=DEVICE,torch_dtype=torch.float16)
add_special_tokens(model, processor.tokenizer)

def predict(image):
    model.eval()
    with torch.no_grad():
        input = f'<!DOCTYPE html>\n<html>'
        decoder_input_ids = processor.tokenizer.encode(input,return_tensors='pt',add_special_tokens=True)[...,:-1]
        encoding = processor(images=[image],text=[""],max_patches=1024,return_tensors='pt')
        item = {
            'decoder_input_ids': decoder_input_ids,
            'flattened_patches': encoding['flattened_patches'].half(),
            'attention_mask': encoding['attention_mask']
        }
        item = move_to_device(item,DEVICE)

        outputs = model.generate(**item,max_new_tokens=2048,eos_token_id=processor.tokenizer.eos_token_id,do_sample=True)

        prediction_html = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return prediction_html

if test_data_name == 'vision2ui':
    ds = load_dataset('parquet',data_dir=data_path)['train']
else:
    ds = Dataset.load_from_disk(data_path)

for item in tqdm(ds):
    t_start = time.time()

    image = item['image']
    md5 = image2md5(image)
    save_path = result_path
    if test_data_name == 'vision2ui':
        tokens = sum(item['tokens'])
        size = 'short' if tokens<2048 else ('mid' if tokens<4096 else 'long')
        save_path = os.path.join(result_path,size)

    if os.path.exists(os.path.join(save_path,md5)):
        continue
    html = predict(image)
    
    duration = time.time()-t_start
    save_result(save_path, image, item['text'] if 'text' in item else item['html'], html, duration)
