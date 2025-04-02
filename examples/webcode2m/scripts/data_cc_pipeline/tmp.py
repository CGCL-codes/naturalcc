import sys, os
sys.path.append(os.path.abspath('.'))
from pathlib import Path
from tools.log import logger

from scripts.data_cc_pipeline.pipeline import download_task,MulThreading

pool = MulThreading(10)
shares = 1
first_bin = list(range(550,900))
volumes = []
for i in first_bin:
    volumes += [i+j*(900//shares) for j in range(shares)]
chunk=12
out_dir = Path("/data02/starmage/datasets/cc")
print(len(volumes))

def download_done( res):
    logger.info(f"Download done {res}")

for v in volumes:
    pool.add_task(download_task, (chunk, v, out_dir), download_done)

pool.shutdown()