import os
os.environ['http_proxy']='127.0.0.1:17890'
os.environ['https_proxy']='127.0.0.1:17890'
from tqdm import tqdm
from datasets import load_dataset,concatenate_datasets,Dataset,DatasetDict
import time

chunk_idx = '10'
path = f'/data02/users/lz/code/UICoder/datasets/c4-wash/scored-chunks/chunk{chunk_idx}-format-scored-parquet'
score_limit = 2

# train_test_split = ds.train_test_split(test_size=0.2)
# train_validation_split = train_test_split['test'].train_test_split(test_size=0.5)

# 重新构建数据集，包含train/test/eval三个部分
# new_dataset = DatasetDict({
#     'train': train_test_split['train'],
#     'test': train_validation_split ['train'],
#     'val': train_validation_split['test']
# })

# ds_list = []

score_list = [5,4,3,2]
i = 0
while i<len(score_list):
    try:
        volume_path = os.path.join(path, f'{score_list[i]}.parquet')
        ds = load_dataset('parquet',data_files=volume_path)['train']
        ds.push_to_hub('xcodemind/cc_scored',token="xx", data_dir=f"chunk{chunk_idx}/Score[{score_list[i]}]")
        i+=1
    except:
        continue

# for i in sorted(range(3, 6), reverse=True):
#     print(f'Upload score[{i}]')
#     volume_path = os.path.join(path, f'{i}')
    
    # filenames = sorted(os.listdir(volume_path),key=lambda x:int(x.split('.')[0]))
    # j = 0
    # while(j<len(filenames)):
    #     filename = filenames[j]
    #     try:
    #         print(f'File {filename}')
    #         ds = load_dataset('parquet',data_files=os.path.join(volume_path,filename))['train']
    #         ds.push_to_hub('xcodemind/cc_scored',token="xx", data_dir=f"chunk{chunk_idx}/Score[{i}]/{os.path.splitext(filename)[0]}")
    #         j+=1
    #     except:
    #         time.sleep(10)
    #         continue


# ds_all = concatenate_datasets(ds_list)

# ds_all.push_to_hub('xcodemind/cc_scored',token="xx", data_dir="chunk05")