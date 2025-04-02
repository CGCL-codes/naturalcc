from PIL import Image
from tqdm import tqdm
import re
import os
from agents import *
from agents.utils.tools import *
from datasets import Dataset, load_dataset
from utils import image2md5, save_result
import time
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


model_name = '4v_agent'
test_data_name = 'vision2ui'
result_path = f'/data02/projects/vision2ui/results/{model_name}-{test_data_name}'
data_path = f'/data02/projects/vision2ui/datasets/{test_data_name}_benchmark'


agent_split = AgentSplit()
agent_i2c = AgentI2C()
agent_assemble =AgentAssemble()

def gen(image, retries=5, retry_delay=2, index=None):
    for attempt in range(retries):
        try:
            node_list = agent_split.infer(image)
            for node in tqdm(node_list):
                html = agent_i2c.infer(crop_image(image, node['bbox']))
                node['html'] = html

            prompt = assemble_node_list(node_list)
            code = agent_assemble.infer(image, prompt)
            html = code.split('```html')[-1].replace('```', '').strip()
            return html
        except Exception as e:
            tqdm.write(f"Attempt {attempt + 1} failed for index {index}: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                tqdm.write(f"Failed after multiple attempts for index {index}")
                # 返回一个默认的 HTML 作为异常处理
                return "<html><body><h1>Error Occurred</h1><p>Failed after multiple attempts</p></body></html>"



if test_data_name == 'vision2ui':
    ds = load_dataset('parquet', data_dir=data_path)['train']
else:
    ds = Dataset.load_from_disk(data_path)

start_index=172

# 从指定的索引开始处理数据集
for index, item in tqdm(enumerate(ds), total=len(ds)):
    if index < start_index:
        continue  # 跳过到 start_index 的前面
    image = item['image']
    md5 = image2md5(image)
    
    save_path = result_path
    if test_data_name == 'vision2ui':
        tokens = sum(item['tokens'])
        size = 'short' if tokens < 2048 else ('mid' if tokens < 4096 else 'long')
        save_path = os.path.join(result_path, size)

    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")


    t_start = time.time()
    html = gen(image, index=index)  # 传递 index 以便记录失败的索引
    
    
    # 检查生成的 HTML 是否有效
    max_attempts = 5  # 设置最大尝试次数
    attempt = 0
    while attempt < max_attempts:
        if 'Failed' not in html and isinstance(html, str):
            duration = time.time() - t_start
            save_result(save_path, image, item['text'] if 'text' in item else item['html'], html, duration)
            tqdm.write(f"{index}")
            sys.stdout.flush()
            break  # 成功后退出循环
        else:
            attempt += 1
            tqdm.write(f"Attempt {attempt} failed to generate HTML for index {index}: {md5}")
            sys.stdout.flush()
            if attempt < max_attempts:  # 只有在未达到最大尝试次数时才重新请求
                html = gen(image, index=index)  # 重新请求
            else:
                tqdm.write(f"Failed after {max_attempts} attempts for index {index}: {md5}")
                sys.stdout.flush()
    sys.stdout.flush()