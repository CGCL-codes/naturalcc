import os,sys
sys.path.append(os.path.abspath('.'))
import torch
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from transformers import AutoProcessor,Pix2StructForConditionalGeneration,TrainingArguments,Trainer,HfArgumentParser,AddedToken, pipeline
from glob import glob
from datasets import load_dataset,concatenate_datasets,Dataset
from scripts.train.vars import *
from scripts.train.utils import smart_tokenizer_and_embedding_resize,move_to_device
from pathlib import Path
from metrics import *
import time


processor_name_or_path = '/data02/models/pix2struct-base/'
ckpt_path='/data03/starmage/projects/UICoder/checkpoints/final/stage0/ws_30w_l3072_p1024_ws_1m_3'
dataset_path= '/data02/starmage/datasets/cc/bench/short.parquet'
out_dir = '/data03/starmage/projects/UICoder/outputs/stage0_m-p2c_d-ws_l3072_p1024_v2i_short'
device = 'cuda:7'


def load(path):
    ds = None
    if path.endswith('.parquet'):
        parquet_paths = [path]
    else:
        parquet_paths = glob(os.path.join(path,'*.parquet'))
    if len(parquet_paths):
        for path in tqdm(parquet_paths,desc='Loading parquet data'):
            if ds:
                try:
                    temp = load_dataset('parquet', data_files=path)['train']
                except:
                    continue
                ds = concatenate_datasets([ds, temp])
            else:   
                ds = load_dataset('parquet', data_files=path)['train']
    else:
        if os.path.exists(path):
            ds = Dataset.load_from_disk(path)      

    if not len(ds):
        raise ValueError(f'No invalid parquet file found in {path}')
    
    return ds

def predict(model, item, processor, device):
    input='<html>'
    decoder_input_ids = processor.tokenizer.encode(input,   return_tensors='pt',add_special_tokens=True)[...,:-1]
    encoding = processor(images=[item['image']],text=[""],  max_patches=1024,return_tensors='pt')
    item = {
        'decoder_input_ids': decoder_input_ids,
        'flattened_patches': encoding['flattened_patches'].half(),
        'attention_mask': encoding['attention_mask']
    }
    item = move_to_device(item,device)

    outputs = model.generate(**item,max_new_tokens=2048,eos_token_id=processor.tokenizer.eos_token_id,do_sample=True)
    prediction_html = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return prediction_html

def eval():
    
    torch.manual_seed(SEED)
    
    # 模型   
    print('Loading checkpoint ...')
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = Pix2StructForConditionalGeneration.from_pretrained(
        ckpt_path,
        is_encoder_decoder=True,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    smart_tokenizer_and_embedding_resize(model, processor.tokenizer, {
        'bos_token': AddedToken('<s>', rstrip=False, lstrip=False, single_word=False, normalized=True),
    })   
    
    global out_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)   
    dataset=load(dataset_path)
    # generate preds
    print("Inferencing ...")
    model.eval()    

    
    with torch.no_grad():
        for idx,item in enumerate(tqdm(dataset)):
            s_start=time.time()
            subdir = out_dir / f'{idx}'
            subdir.mkdir(exist_ok=True, parents=True)      
            pred_html = predict(model, item, processor, device)         
            duration = time.time() - s_start
            with open(subdir / f'answer.html','w',encoding='utf-8') as fi:
                fi.write(item['text'])
            with open(subdir / f'prediction.html','w',encoding='utf-8') as fi:
                fi.write(pred_html)    
            with open(subdir / f'time.csv','a+') as fi:
                fi.write(f'{duration}\n')    
               
if __name__ == '__main__':
    torch.manual_seed(SEED)
    eval()