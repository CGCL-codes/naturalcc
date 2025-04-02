
import os
import io
import hashlib
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm
from utils import *
import time

model_name = 'cogagent-vqa-hf'
test_data_name = 'vision2ui'


tokenizer_path = '/data02/models/vicuna-7b-v1.5'
model_path = f'/data02/projects/vision2ui/models/{model_name}'
result_path = f'/data02/projects/vision2ui/results/{model_name}-{test_data_name}'
data_path = f'/data02/projects/vision2ui/datasets/{test_data_name}_benchmark'

DEVICE = 'cuda:1'

prompt = "Write the HTML code."

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True
).to(DEVICE).eval()

def predict(image):
    image = image.convert('RGB')
    query = f'{prompt}'
    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE)]],
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE)]]

    gen_kwargs = {
        "max_length": 2048,
        "temperature": 0.5,
        "repetition_penalty": 1.1,
        "top_p": 1.0,
        "top_k": 1,  
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("</s>")[0]
        
    return response

if test_data_name == 'vision2ui':
    ds = load_dataset('parquet',data_dir=data_path)['train']
else:
    ds = Dataset.load_from_disk(data_path)

for item in tqdm(ds):
    image = item['image']
    md5 = image2md5(image)
    save_path = result_path
    if test_data_name == 'vision2ui':
        tokens = sum(item['tokens'])
        size = 'short' if tokens<2048 else ('mid' if tokens<4096 else 'long')
        save_path = os.path.join(result_path,size)

    if os.path.exists(os.path.join(save_path,md5)):
        continue

    t_start = time.time()
    html = predict(image)

    duration = time.time()-t_start
    save_result(save_path, image, item['text'] if 'text' in item else item['html'], html, duration)
