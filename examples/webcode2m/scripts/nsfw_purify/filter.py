from datasets import load_dataset
from PIL import Image
from transformers import pipeline
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import os
import re

try:
    mp.set_start_method('spawn')
except:
    pass


def load_bad_words(directory):
    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    words = []
    for file_path in tqdm(txt_files):
        with open(file_path, 'r', errors='ignore') as f:
            words += list(filter(lambda x:x and len(x)>1, map(lambda x: x.replace(',','').strip(), f.read().split('\n'))))

    return words

def find_bad_ids(dataset, begin, end, classifier=None, bad_words=None, drop_ru=False, threshold=20, max_score=0.04):
    end = end if end < len(dataset) else len(dataset)-1
    bad_idxs = []
    pattern = re.compile(r'\b(?:'+'|'.join(map(re.escape, bad_words))+ r')\b', re.IGNORECASE) if bad_words else None
    for idx, item in enumerate(tqdm(dataset.select(range(begin,end)), desc=f"{begin}-{end}")):
        if drop_ru and item['lang'] == 'ru':
            bad_idxs.append(begin+idx)
            continue
        if bad_words:
            count = 0
            for match in pattern.finditer(item['text']):
                count += 1
                if count >= threshold:
                    bad_idxs.append(begin+idx)
                    break
            if count >= threshold:
                continue
        if classifier:
            scores = classifier(item['image'])
            if scores[1]['score'] > max_score:
                bad_idxs.append(begin+idx)
    return np.array(bad_idxs).astype(np.uint32)

def clean(data_path, only_idx=False):
    dataset = load_dataset('parquet', data_files=data_path)['train']
    classifier = pipeline("image-classification", model="./nsfw_image_detection", device_map="cuda")
    bad_words = load_bad_words('./bad_words/')
    max_worker = 20
    batch_size = int(len(dataset)/max_worker)
    input_data_list = [(dataset, begin, min(begin+batch_size, len(dataset)), classifier, bad_words, True, 20, 0.04) for begin in range(0, len(dataset), batch_size)]
    with Pool(processes=100) as pool:
        results = pool.starmap(find_bad_ids, input_data_list)

    bad_idxs = np.concatenate(results)

    good_idxs = list(filter(lambda x: x not in bad_idxs, range(0, len(dataset))))

    if only_idx:
        return good_idxs, bad_idxs
    else:
        good_dataset = dataset.select(good_idxs)
        bad_dataset = dataset.select(bad_idxs)
        return good_dataset, bad_dataset

if __name__ == '__main__':
    origin_dir = '/data02/users/lz/code/vision2ui/data/data_origin'
    good_dir = '/data02/users/lz/code/vision2ui/data/data_good'
    bad_dir = '/data02/users/lz/code/vision2ui/data/data_bad'
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)
    for name in tqdm(sorted(os.listdir(origin_dir))):
        good_dataset, bad_dataset = clean(f'{origin_dir}/{name}')
        print(len(good_dataset), len(bad_dataset))
        good_dataset.to_parquet(f'{good_dir}/{name}')
        bad_dataset.to_parquet(f'{bad_dir}/{name}')
