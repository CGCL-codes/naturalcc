from git import Repo
import time
from tqdm import tqdm
import traceback
import datasets
import os
import pandas as pd

os.environ['https_proxy']='127.0.0.1:17890'
os.environ['http_proxy']='127.0.0.1:17890'
repo_path="/data03/starmage/datasets/cc/final"
repo = Repo(repo_path)

def git_push(repo:Repo):
    try:
        repo.git.push()
        return True
    except Exception as e:
        print('Push failed:', e)
        print(traceback.format_exc())
        return False
    
num_chunks = 2068
start_chunk = 1728
for id in range(start_chunk, num_chunks):
    fpath=f"{repo_path}/data/{id:05}.parquet"
    if not os.path.exists(fpath):
        print(f"{fpath} not exists.")
        continue
    repo.git.add(fpath)
    repo.git.commit('-m', f"commit chunk {id}.")    
    try_count = 0
    while not git_push(repo):
        time.sleep(60*10)
        os.system('export https_proxy="http://127.0.0.1:17890"')
        try_count += 1
        if try_count > 3:
            print("max try conuts. exit")
            exit(1)
