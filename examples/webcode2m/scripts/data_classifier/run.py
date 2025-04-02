from datasets import load_dataset,concatenate_datasets,Dataset
from torchvision import datasets, transforms, models
from glob import glob
from tqdm import tqdm
import torch.nn as nn
import os
import json
import torch
import hashlib
import io
import asyncio
import threading

device = 'cuda:7'
batch_size = 128
chunk_idx = '15'
path = f'/data02/users/lz/code/UICoder/datasets/c4-wash/chunks/chunk{chunk_idx}-format-parquet'
ckpt = '/data02/users/lz/code/UICoder/checkpoints/classifier/c4_500.pth'
output_path = f'/data02/users/lz/code/UICoder/datasets/c4-wash/scored-chunks/chunk{chunk_idx}-format-scored-parquet'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(ckpt):
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    return model.to(device)

def image_to_md5(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_data = image_bytes.getvalue()
    md5_hash = hashlib.md5(image_data)
    md5_hex = md5_hash.hexdigest()
    return str(md5_hex)


def predict(model,images):
    inputs = torch.stack([transform(image) for image in images]).to(device)
    outputs = model(inputs).detach().cpu()
    return [min(max(round(output.item()),0),5) for output in outputs]

def save(ds,scores):
    print('Saving splited data... ')
    for score in scores:
        ds2 = ds.select(scores[score])
        ds2.to_parquet(os.path.join(output_path,f'{score}.parquet'))

def work(parquet_paths):
    global path,ckpt,output_path
    os.makedirs(output_path,exist_ok=True)
    model = load_model(ckpt)
    scores = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': []
    }
    
    ds = None
    for path in tqdm(parquet_paths,desc='Loading parquet files'):
        if ds:
            try:
                temp = load_dataset('parquet', data_files=path)['train']
                ds = concatenate_datasets([ds, temp])
            except:
                pass
        else:   
            print(path)
            ds = load_dataset('parquet', data_files=path)['train']

    with tqdm(total=len(ds)) as tbar:
        for i in range(0,len(ds),batch_size):
            tbar.update(batch_size)
            try:
                images = [item['image'].convert("RGB") for item in ds.select(range(i,min(i+batch_size,len(ds))))]
                scores_ = predict(model,images)
                for idx, score in enumerate(scores_):
                    scores[str(score)].append(i+idx)
                tbar.set_postfix_str(', '.join([f'{key}: {len(scores[key])}' for key in scores]))
            except:
                pass
        # for idx,item in enumerate(ds):
        #     tbar.update(1)
        #     image = item['image'].convert("RGB")
        #     score = predict(model,image)
        #     scores[str(score)].append(idx)
        #     tbar.set_postfix_str(', '.join([f'{key}: {len(scores[key])}' for key in scores]))
        save(ds,scores)
    
def main():
    parquet_paths = sorted(glob(os.path.join(path,'*.parquet')),key=lambda x: int(x.split('.')[0].split('/')[-1]))
    work(parquet_paths)
    
if __name__ == '__main__':
    main()

