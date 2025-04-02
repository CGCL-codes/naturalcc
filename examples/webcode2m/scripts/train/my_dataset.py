from torch.utils.data import Dataset as BaseDataset
import datasets
from datasets import load_dataset, concatenate_datasets, Dataset
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import torch
import json
import math
import os
from glob import glob
import traceback
from vars import *
from utils import BboxTree2Html, BboxTree2StyleList
from pathlib import Path
from processor import MultiProcessor
import math
import multiprocessing

def setup_proxy(unset=False):
    os.environ['https_proxy']='127.0.0.1:17890' if not unset else ''
    os.environ['http_proxy']='127.0.0.1:17890' if not unset else ''

class UICoderDataset(BaseDataset):
    def __init__(self, path, preprocess=True, make_patches_while_training=True , stage=1, 
                 processor=None, max_length=None, max_patches=None, max_num=None, transform=None, 
                 drop_longer=False, workers=max(1, int(0.8*multiprocessing.cpu_count()))):
        self.data_ = [] 
        self.load(path)      
        if preprocess:
            self.transform = transform
            self.max_length = max_length
            self.max_patches = max_patches
            self.max_num = max_num
            self.drop_longer = drop_longer
            self.processor = processor
            self.stage = stage
            self.workers=workers   
            self.make_patches_while_training = make_patches_while_training            
            self.make_all()
            if not self.make_patches_while_training:
                self.process_all()
        else:
            self.data_ = self.volumes
            
    def save(self, path):
        Path(path).mkdir(exist_ok=True, parents=True)
        self.data_.save_to_disk(path)
            
    def load(self, path):
        volumes = []
        if path.endswith('.parquet'):
            parquet_paths = [path]
        else:
            parquet_paths = sorted(glob(os.path.join(path,'*.parquet')),key=lambda x: int(x.split('.')[0].split('/')[-1]))
        if len(parquet_paths):
            ds = None
            for path in tqdm(parquet_paths,desc='Loading parquet data'):
                if ds:
                    try:
                        temp = load_dataset('parquet', data_files=path)['train']
                    except:
                        continue
                    ds = concatenate_datasets([ds, temp])
                else:   
                    ds = load_dataset('parquet', data_files=path)['train']
            volumes = ds   
        else:
            if os.path.exists(path):
                volumes = Dataset.load_from_disk(path)
            else: # try to load hf dataset
                setup_proxy()
                volumes = load_dataset(path)  
                if path == "SALT-NLP/Design2Code-hf":
                    volumes = volumes["train"]
                setup_proxy(True)      

        if not len(volumes):
            raise ValueError(f'No invalid parquet file found in {path}')
        
        self.volumes = volumes     

    def make_data(self, begin=0, end=-1, chunk_idx=0):
        def generator():
            volumes = self.volumes.select(range(begin,end))
            for idx,volume in enumerate(tqdm(volumes,desc=f'Making data: chucnk[{chunk_idx}]')):   
                try:
                    html = volume['text'] if 'text' in volume else volume["html"]
                    if self.stage != 0:
                        bbox = volume['bbox']
                        while isinstance(bbox,str):
                            bbox = json.loads(bbox)
                        root_node = bbox

                    if self.stage == 0:
                        input_ids = self.processor.tokenizer(html,max_length=self.max_length,padding='max_length',truncation=True,     return_tensors='pt',add_special_tokens=True)['input_ids'].squeeze()

                        if self.drop_longer and input_ids[-1] != self.processor.tokenizer.pad_token_id:
                            continue
                        
                        labels = input_ids.clone()
                        input_ids[...,1:] = labels[...,:-1]
                        input_ids[0] = self.processor.tokenizer.bos_token_id
                        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_ID
                        yield {
                            'image_idx': begin+idx,
                            'decoder_input_ids': input_ids,
                            'labels': labels,
                        }
                    elif self.stage == 1:
                        html = BboxTree2Html(root_node, style=False, size=self.volumes[begin+idx]['image'].size)
                        input_ids = self.processor.tokenizer(html,max_length=self.max_length,padding='max_length',truncation=True,      return_tensors='pt',add_special_tokens=True)['input_ids'].squeeze()
                        if self.drop_longer and input_ids[-1] != self.processor.tokenizer.pad_token_id:
                            continue
                        
                        labels = input_ids.clone()
                        input_ids[...,1:] = labels[...,:-1]
                        input_ids[0] = self.processor.tokenizer.bos_token_id
                        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_ID
                        yield {
                            'image_idx': begin+idx,
                            'decoder_input_ids': input_ids,
                            'labels': labels,
                        }
                    elif self.stage == 2:
                        styleList = BboxTree2StyleList(root_node, skip_leaf=False)
                        for styleItem in styleList:
                            ibbox = styleItem['bbox']
                            if ibbox[2] <= bbox_padding*2 or ibbox[3] <= bbox_padding*2:
                                continue

                            cnode_type_bbox_list = list(map(lambda x: f'{x["type"]}({round((x["bbox"][0]-ibbox[0])/ibbox[2],precision)},{round((x["bbox"][1]-ibbox[1])/ibbox[3],precision)},{round(x["bbox"][2]/ibbox[2],precision)},{round(x["bbox"][3]/ibbox[3],precision)})', styleItem['children']))
                            input = f"{styleItem['type']}<{','.join(cnode_type_bbox_list)}> %{styleItem['style']}%<%{'%,%'.join(list(map(lambda x:x['style'], styleItem['children'])))}%>"

                            input_ids = self.processor.tokenizer(input,max_length=self.max_length,padding='max_length',  truncation=True, return_tensors='pt',add_special_tokens=True)['input_ids'].squeeze()

                            labels = input_ids.clone()
                            input_ids[...,1:] = labels[...,:-1]
                            input_ids[0] = self.processor.tokenizer.bos_token_id
                            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_ID

                            # second_bracket_index = (input_ids == self.processor.tokenizer.convert_tokens_to_ids('%')).nonzero (as_tuple=True)[0][1]
                            # labels[:second_bracket_index+1] = IGNORE_ID

                            yield {
                                'image_idx': begin+idx,
                                'bbox': ibbox,
                                'decoder_input_ids': input_ids,
                                'labels': labels
                            }
                    else:
                        raise ValueError(f'Invalid training stage: {self.stage}')
                except Exception as e:
                    # traceback.print_exc()
                    continue
        ds = Dataset.from_generator(generator).with_format("torch")
        return ds

    def make_all(self):
        volumes = self.volumes.select(range(0, min(self.max_num, len(self.volumes)))) if self.max_num!=-1 else self.volumes
        step = math.ceil(len(volumes)/self.workers)
        data_maker = MultiProcessor("DataMaker", num_processes=self.workers)
        ds_list = []
        def cb(result):
            ds_list.append(result)

        for i in range(0, len(volumes), step):
            begin = i 
            end = min(i+step, len(volumes))
            if end > begin:
                data_maker.add_task(self.make_data, (begin, end, i//step), cb)

        data_maker.shutdown()
        print('concatenate maked data results...')
        self.data_ = concatenate_datasets(ds_list)

    def process_data(self, begin=0, end=-1, chunk_idx=0):
        def generator():
            for item in tqdm(self.data_.select(range(begin,end)),desc=f'Processing data: chucnk[{chunk_idx}]'):
                try:
                    image_idx = int(item['image_idx'])
                    image = self.volumes[image_idx]['image']
                    if 'bbox' in item:
                        bbox = item['bbox'].tolist()
                        image = image.crop((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]))
                    encoding = self.processor(images=[image],text=[""],max_patches=self.max_patches,return_tensors='pt')
                    yield {
                        **item,
                        'flattened_patches': encoding['flattened_patches'][0],
                        'attention_mask': encoding['attention_mask'][0]
                    }
                except:
                    continue
        ds = Dataset.from_generator(generator).with_format('torch')
        return ds

    def process_all(self):        
        step = math.ceil(len(self.data_)/self.workers)
        data_processor = MultiProcessor("DataProcessor", num_processes=self.workers)
        ds_list = []
        def cb(result):
            ds_list.append(result)

        for i in range(0, len(self.data_), step):
            begin = i 
            end = min(i+step, len(self.data_))
            if end > begin:
                data_processor.add_task(self.process_data, (begin, end, i//step), cb)

        data_processor.shutdown()
        print('concatenate processed data results...')
        self.data_ = concatenate_datasets(ds_list)

    def process_one(self, item):
        image_idx = int(item['image_idx'])
        image = self.volumes[image_idx]['image']
        if 'bbox' in item:
            bbox = item['bbox'].tolist()
            image = image.crop((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]))
        # if image.width * image.height > 1980 * 1980 * 2:
        #     ratio = (1980 * 1980 * 2) / (image.width * image.height)
        #     image.resize((int(image.width*ratio),int(image.height*ratio)))
        encoding = self.processor(images=[image],text=[""],max_patches=self.max_patches,return_tensors='pt')
        return {
            **item,
            'image': image,
            'flattened_patches': encoding['flattened_patches'][0],
            'attention_mask': encoding['attention_mask'][0]
        }
    
    def __len__(self):
        return len(self.data_)
    
    def __getitem__(self,idx):
        return self.process_one(self.data_[idx]) if self.make_patches_while_training else self.data_[idx]
    
class UICoderCollater(object):
    def __call__(self, instances):
        decoder_input_ids, labels, flattened_patches, attention_mask = tuple(
                [instance[key] for instance in instances]
                for key in ("decoder_input_ids", "labels", "flattened_patches", "attention_mask")
        )
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids,
            batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True)
        flattened_patches = torch.nn.utils.rnn.pad_sequence(
            flattened_patches,
            batch_first=True)
        flattened_patches = flattened_patches.half() if (hasattr(self,'eval') and self.eval) else flattened_patches
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True)
        

        # Batch dict
        batch = dict(
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            image=[instance['image'] for instance in instances]
        ) if (hasattr(self,'eval') and self.eval) else dict(
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            flattened_patches=flattened_patches,
            attention_mask=attention_mask
        )

        return batch