import os
import requests
import gzip
import shutil
from tqdm import tqdm
from pathlib import Path
import traceback
from tools.log import logger
from tools.download import download_file
import time

CC_BASE = 'https://data.commoncrawl.org/'
CC_WARC = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-50/warc.paths.gz"
# 原始文件中共90000行，分为100组，每组900行（即为900个warc文件）
CC_CHUNK_NUM = 100
CC_CHUNK_SIZE = 900     

def wait_for_space(path, required_space, timeout=30):
    duraiton = 0
    while duraiton < timeout :
        _, _, free_space = shutil.disk_usage(path)
        if free_space >= required_space:
            return True
        else:
            print(f"等待可用空间足够... 目前可用空间: {free_space / 1024**3} GB")
            time.sleep(10)  # 每10秒检查一次
            duraiton += 10
    return False     

def download_warc(warc_table, chunk, volume, warc_dir:Path, rnew=False):
    line = warc_table[chunk*CC_CHUNK_SIZE + volume].strip()
    url = CC_BASE+line
    gzip_file_path = warc_dir / f"{chunk:03}_{volume:03}.warc.gz"
    gzip_file_path.parent.mkdir(parents=True, exist_ok=True)
    try:       
        res = download_file(url, gzip_file_path,rnew)   
        logger.debug(f"Gzip文件成功下载到: {gzip_file_path}")        
        return gzip_file_path if res else None
    except requests.exceptions.RequestException as e:
        logger.debug(f"下载文件时发生错误: {e}")
        return None
    except Exception as e:
        exception_info = traceback.format_exc()     
        logger.debug(exception_info)
        return None

def unzip_warc(chunk, volume, gzip_file_path, warc_dir:Path):
    warc_path = warc_dir / f"{chunk:03}_{volume:03}.warc"
    warc_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Checking disk space.")
    if not wait_for_space(warc_path.parent, 6*(1024**3), 5*60):
        raise ValueError("No enough space for unzip.")
        # 解压gzip文件
    with gzip.open(gzip_file_path, 'rb') as gz_file:
        with open(warc_path, 'wb') as dest_file:
            shutil.copyfileobj(gz_file, dest_file)

    logger.debug(f"文件成功解压到: {warc_path}")  
    return  warc_path

def extract_html(warc_path, volume_dir:Path, tbar=None):
    html_temp = ''
    index = 0             
 
    logger.debug("Checking disk space.")
    if not wait_for_space(volume_dir, 10*(1024**3), 5*60):
        raise ValueError("No enough space for extraction.")
    with open(warc_path, "r", errors='ignore') as f: 
        line = 'start'
        while line:
            line = f.readline()
            try:
                if 'WARC-Target-URI' in line:
                    uri = 'http'+line.split('http')[1]
                if '<html' in line:
                    html_temp = line
                elif '</html>' in line:
                    if tbar is not None:
                        tbar.update(1)
                    html_temp += line
                    html_path = volume_dir / f"{index:05}.html"
                    with open(html_path,'w') as fhtml:
                        fhtml.write('<!-- '+uri.strip()+' -->\n'+html_temp)
                    index += 1
                    html_temp = ''
                    uri = ''
                elif html_temp:
                    html_temp += line
            except:
                html_temp = ''
                uri = ''
                continue
                
def get_warc_table(out_dir):
    warc_table_path = out_dir / "warc.paths"    
    if not warc_table_path.exists():
        gz_path = str(warc_table_path) + '.gz'
        download_file(CC_WARC, gz_path)
        # 解压gzip文件
        with gzip.open(gz_path, 'rb') as gz_file:
            with open(warc_table_path, 'wb') as dest_file:
                shutil.copyfileobj(gz_file, dest_file)
        logger.debug(f"warc table文件成功解压到: {warc_table_path}")
        os.remove(gz_path)   
        
    with open(warc_table_path, "r") as f:
        warc_table = f.readlines()
        
    return warc_table
    
def download_and_extract(chunk:int, volume:int, out_dir:Path, tbar : tqdm):    
    warc_table = get_warc_table(out_dir)

    warc_dir = out_dir / "warc"
    warc_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Downloading chunk {chunk:03}, volume {volume:03} ...")
    warc_path = download_warc(warc_table, chunk, volume, warc_dir)    
    if warc_path is not None:
        logger.info(f"Download done chunk {chunk:03}, volume {volume:03}.")
        volume_dir = out_dir / f"src/{chunk:03}/{volume:03}"
        volume_dir.mkdir(parents=True, exist_ok= True)        
        extract_html(warc_path, volume_dir, tbar)        
        os.remove(warc_path)
        return volume_dir
    else:
        logger.info(f"Fail to download chunk {chunk:03}, volume {volume:03}.")
        return None

if __name__ == "__main__":
    tq = tqdm()
    download_and_extract(0,0, Path("/data03/starmage/projects/UICoder/data"), tq)
