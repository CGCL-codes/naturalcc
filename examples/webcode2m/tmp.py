from git import Repo
import time
from tqdm import tqdm
import traceback
import datasets
import os
import pandas as pd


ds_train_0 = datasets.load_from_disk("/data02/starmage/datasets/cc/arrows_8-14_processed")
ds_train_1 = datasets.load_from_disk("/data02/starmage/datasets/cc/train_arrows_5-8_15")
hashes_train_0 = pd.read_csv("/data02/starmage/datasets/cc/8-15_hash.csv")
no_dup_ids = hashes_train_0[~hashes_train_0.duplicated('hash')].index
ds_train_0_nodup = ds_train_0.select(no_dup_ids)
ds_total = datasets.are_progress_bars_disabledconcatenate_datasets([ds_train_1, ds_train_0_nodup])

os.environ['https_proxy']='127.0.0.1:17890'
os.environ['http_proxy']='127.0.0.1:17890'
repo_path="/data03/starmage/datasets/cc/final"
repo = Repo(repo_path)
def git_push(repo:Repo, file_path, msg):
    try:
        repo.git.add(file_path)
        repo.git.commit('-m', msg)
        repo.git.push()
        return True
    except Exception as e:
        print('Push failed:', e)
        print(traceback.format_exc())
        return False
chunk_size = 1536
start_chunk = 2
for id,start in tqdm(enumerate(range(chunk_size*start_chunk, len(ds_total), chunk_size))):
    fpath=f"{repo_path}/data/{id+start_chunk:045}.parquet"
    ds_total.select(range(start,min(start+chunk_size, len(ds_total)))).to_parquet(fpath)
    while not git_push(repo, fpath, f"commit chunk {id+start_chunk}."):
        time.sleep(5)