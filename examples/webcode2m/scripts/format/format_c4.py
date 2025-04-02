from tqdm.asyncio import tqdm
from PIL import Image
from datasets import Dataset
import json
import os
import asyncio

path = '/data02/users/lz/code/UICoder/datasets/c4-wash/H128-2560_C128-4096_R2/c4-format'
output_path = '/data02/users/lz/code/UICoder/datasets/c4-wash/H128-2560_C128-4096_R2/c4-format-parquet'
parquet_size = 10000

# def mergexywh(node):
#     node['bbox'] = [node['x'],node['y'],node['width'],node['height']]
#     del node['x']
#     del node['y']
#     del node['width']
#     del node['height']
#     del node['depth']
#     for cnode in node['children']:
#         mergexywh(cnode)

async def worker(volume_paths,output_path):
    ds = []
    for volume_path in tqdm(volume_paths,desc=f"{os.path.basename(output_path).split('.')[0]}"):
        try:
            image = Image.open(os.path.join(volume_path,'image.png'))
            with open(os.path.join(volume_path,'bbox.json')) as f:
                bbox = json.load(f)
            with open(os.path.join(volume_path,'index.html')) as f:
                html = f.read()
            # mergexywh(bbox)
            ds.append({
                'image': image,
                'html': html,
                'bbox': json.dumps(bbox)
            })
        except:
            continue
    ds2 = Dataset.from_list(ds)
    print(f'Saving {len(output_path)} {output_path}')
    ds2.to_parquet(f"{output_path}")

async def main():
    os.makedirs(output_path,exist_ok=True)
    volume_paths = [os.path.join(path,volume_name) for volume_name in os.listdir(path)]
    tasks = []
    for i in range(0,len(volume_paths),parquet_size):
        
        tasks.append(asyncio.create_task(worker(volume_paths[i:i+parquet_size],f"{output_path}/{'00' if i//parquet_size<10 else ('0' if i//parquet_size<100 else'')}{i//parquet_size}.parquet")))
    await asyncio.gather(*tasks)    

if __name__ == '__main__':
    asyncio.run(main())
    